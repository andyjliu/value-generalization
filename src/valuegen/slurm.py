"""SBATCH generation + submit/poll/resubmit orchestration.

One generic loop over declarative :class:`Stage` specs, in place of
hand-rolled YAML→SBATCH renderers and copy-edited lockstep array scripts.

Design:

- A :class:`Stage` is a named list of :class:`Task`\\ s plus resources. Each
  task carries its own shell command and a filesystem ``done`` predicate.
  **State is the filesystem** — there is no job database, so every verb is
  idempotent and resumable: ``run`` after an interruption re-arrays only the
  tasks whose outputs are missing (this is also what ``resubmit-missing``
  means; they are the same code path).
- Task maps are generated from config (model × value grids); the old
  hand-synced ``PIDS``/``MTAGS`` arrays across build/train/eval scripts are
  gone — drivers build both sides from :mod:`valuegen.values`.
- ``--dry-run`` writes the SBATCH scripts without submitting; scripts are
  inspectable artifacts under ``slurm_jobs/{experiment}/``, like today.
- Node excludes from ``cluster.yaml`` are **always** emitted on GPU stages.
- Exclusive-node clusters (``slurm.gpus_per_node`` set; e.g. the AMD AUP
  cluster, where jobs get whole 8-GPU nodes and the scheduler tracks no GRES
  or memory at all) pack single-GPU arrays ``gpus_per_node`` tasks per
  element with GNU parallel (:meth:`Orchestrator._render_packed`), count
  nodes' worth of GPUs in the throttle, and clamp walltimes to
  ``slurm.partition_max_time`` — scaling the retry budget by the same ratio so
  resumable stages keep their configured wall budget. ``request_gres`` /
  ``request_mem: false`` drop the corresponding directives.
- Server stages (``server=True``, e.g. a persistent vLLM judge) are submitted
  before their dependents, publish a ``hostname:port`` hostfile, and are
  cancelled when the pipeline finishes — or earlier, when a stage names them in
  ``cancel_servers`` (the labeling oracle is released before training starts,
  instead of sitting on two GPUs for the rest of the run). Under ``--no-wait``
  there is no polling loop to cancel from, so a server's walltime bounds it.
  Dependent tasks poll the hostfile themselves (see
  :func:`valuegen.ground_truth.evaluation.wait_ready_sh`), so the orchestrator
  never needs to know when a server is "ready" — only whether it is still
  *alive*, which it re-checks for each stage naming it in ``needs_servers``
  before every retry attempt, and resubmits if not.

Pipelines supply stage lists (override *data*, not the loop). The one
behavioral subclass is :class:`RunPipelineOrchestrator`, which delegates to
persona_vectors' ``run_pipeline.py`` in v1 while presenting the same
``run/status`` surface; v2 retires it by translating run_pipeline's four
stages into native :class:`Stage` specs.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from valuegen.config import ClusterConfig
from valuegen.config import config_id_matches

_ACTIVE_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED",
}


def _scale_mem(mem: str, factor: int) -> str:
    """Multiply an ``#SBATCH --mem`` value by ``factor``, keeping its unit.

    SLURM mem is an integer with an optional unit suffix (``K``/``M``/``G``/
    ``T``, default M); ``0`` means "all node memory" and is passed through. A
    packed chunk runs ``factor`` tasks at once, so its reservation is the
    per-task value times the number of tasks sharing the node.
    """
    m = re.fullmatch(r"(\d+)([KMGT]?)", mem.strip())
    if m is None:
        raise ValueError(f"cannot scale unrecognized --mem value {mem!r}")
    value, unit = int(m.group(1)), m.group(2)
    if value == 0:
        return mem
    return f"{value * factor}{unit}"


def slurm_time_seconds(spec: str) -> int:
    """Seconds in a SLURM time spec: ``D-HH:MM:SS``, ``D-HH:MM``, ``D-HH``,
    ``HH:MM:SS``, ``MM:SS`` or bare minutes (sbatch(1) ``--time``)."""
    spec = spec.strip()
    days = 0
    if "-" in spec:
        d, spec = spec.split("-", 1)
        days = int(d)
        parts = [int(x) for x in spec.split(":")]
        while len(parts) < 3:
            parts.append(0)  # D-HH and D-HH:MM pad on the right
        h, m, sec = parts
    else:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 1:
            h, m, sec = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]
        elif len(parts) == 3:
            h, m, sec = parts
        else:
            raise ValueError(f"unrecognized SLURM time {spec!r}")
    return ((days * 24 + h) * 60 + m) * 60 + sec


@dataclass
class Task:
    """One array element: a shell command plus a done predicate.

    ``done`` is either a Path (exists and non-empty ⇒ done) or a nullary
    callable. Multi-line ``command`` strings are fine — they land verbatim in
    a ``case`` arm of the generated script.
    """

    key: str
    command: str
    done: Path | Callable[[], bool]
    config_id: str | None = None
    config_record: Path | None = None

    def is_done(self) -> bool:
        if self.config_id is not None and (
            self.config_record is None
            or not config_id_matches(self.config_record, self.config_id)
        ):
            return False
        if isinstance(self.done, Path):
            return self.done.is_file() and self.done.stat().st_size > 0
        return bool(self.done())


@dataclass
class Stage:
    """Declarative spec for one SBATCH submission."""

    name: str
    tasks: list[Task]
    time: str
    mem: str
    gpus: int = 0
    env: str = "default"  # key into cluster.yaml conda envs
    throttle: int = 0  # max concurrent array tasks; 0 = derive from GPU cap
    # >1: pack this many single-GPU tasks into one allocation, fanned across
    # that many local GPUs with GNU parallel, instead of one SLURM array
    # element per task. Only meaningful when gpus == 1 on a partition that
    # allocates whole nodes (OverSubscribe=EXCLUSIVE), where a one-GPU array
    # element otherwise leaves the node's other GPUs idle. 0 = take the
    # cluster's ``gpus_per_node`` (off wherever that is unset); 1 = never
    # pack. See Orchestrator._pack / _chunks.
    pack: int = 0
    gpu_headroom: int = 0  # GPUs to leave free (e.g. for a judge server)
    # Force the cpu partition even with gpus == 0 (the default for non-server
    # stages anyway). On clusters with ``cpu_stage_gpus > 0`` such stages still
    # request that many untyped GPUs, because no partition accepts 0.
    cpu_partition: bool = False
    # A long-lived poll-only controller (an experiment driver that submits
    # and watches other jobs). Placed on ``cluster.controller_partition`` /
    # ``controller_qos`` when set, else the cpu partition; GPU-less like a
    # cpu stage (``cpu_stage_gpus`` still applies where every partition
    # demands one).
    controller: bool = False
    # Route this stage's GPU tasks to the cluster's array partition/QOS (a
    # second GPU budget) instead of the default one. That QOS is typically
    # preemptible with requeue, so only set it on stages whose tasks are short
    # and idempotent — a requeue must cost minutes, not a training run. Ignored
    # when the cluster config declares no array partition.
    array_partition: bool = False
    # Narrow this stage's GPU request to one configured type (or a subset),
    # e.g. pin a throughput-bound vLLM server to L40S. None = the cluster's
    # gpu_type. Scheduling only.
    gpu_type: str | list[str] | None = None
    server: bool = False  # persistent single job; submit, don't wait, cancel at end
    # Server stages only: do not submit when ``run`` walks the stage list;
    # leave it to the first dependent stage's ``needs_servers`` check
    # (``_ensure_servers``, "never started — resubmitting"). Set by
    # ``interleave_waves`` so a judge does not sit on a node through the
    # first training wave.
    on_demand: bool = False
    hostfile: Path | None = None  # server stages: removed before submit
    # Export NCCL_P2P_DISABLE/NCCL_IB_DISABLE on this stage's multi-GPU jobs.
    # Off by default: on NVLink nodes the flags force every collective through
    # host memory, roughly halving FSDP training and TP serving throughput.
    # Set per stage — or cluster-wide via slurm.nccl_conservative — only where
    # P2P is actually broken (the original A6000 workaround).
    nccl_conservative: bool = False
    # Server stages to cancel once this stage completes: a server that only
    # feeds an early stage (an oracle judged against by the label array) must
    # not hold its GPUs through the training that follows.
    cancel_servers: tuple[str, ...] = ()
    # Server stages this stage's tasks talk to. Checked alive before *every*
    # attempt, and resubmitted if not: a server that dies mid-array otherwise
    # leaves a stale hostfile that every retry polls until it times out, so one
    # server death costs the whole retry budget without running any work.
    needs_servers: tuple[str, ...] = ()
    # A reaper is a tiny post-work job whose only purpose is to release a
    # server the dependency-chained (``--no-wait``) path would otherwise leak:
    # ``submit`` chains it with ``afterany`` (it must run whether or not its
    # predecessor succeeded — a failed eval must still free the judge), and the
    # polling ``run`` path skips it entirely (it reaps servers itself, on the
    # way out or via ``cancel_servers``). The stage's own task issues the
    # ``scancel``; it holds no server and needs no GPU.
    reaper: bool = False
    extra_exports: dict[str, str] = field(default_factory=dict)
    # Shell run ONCE per array element, after the preamble and before the
    # element's task(s). On a packed stage that is once per node, so it is
    # where a sidecar the node's tasks share is started -- e.g. the prefill
    # judge served on the GPU left free by a reduced ``pack`` (the tasks
    # inherit its exported PREFILL_JUDGE_URL through GNU parallel). It must
    # clean up after itself with its own ``trap ... EXIT``.
    node_setup: str = ""
    max_retries: int = 3
    # Controller-side callback, run by the polling ``run`` once the stage
    # completes (after ``cancel_servers``; never on a dry run, never by the
    # dependency-chained ``submit``). Must be idempotent: a restarted
    # controller completes finished stages again and re-fires it. Used by
    # ``interleave_waves`` to free a wave's checkpoints after their evals.
    after: Callable[[], None] | None = None

    def pending(self) -> list[Task]:
        return [t for t in self.tasks if not t.is_done()]

    def done_count(self) -> int:
        return sum(t.is_done() for t in self.tasks)


def sbatch_available() -> bool:
    return (
        subprocess.run(
            ["which", "sbatch"], capture_output=True, text=True
        ).returncode
        == 0
    )


class Orchestrator:
    """The generic submit/poll/resubmit loop over :class:`Stage` specs.

    Stages run strictly in list order (server stages excepted: they are
    submitted when reached and left running). This matches how every old
    pipeline actually sequenced work; SLURM ``--dependency`` chains are
    deliberately not used — the polling loop is the sequencer, and a killed
    orchestrator just gets re-run (idempotent, filesystem-checked).
    """

    def __init__(
        self,
        name: str,
        stages: Sequence[Stage],
        cluster: ClusterConfig,
        dry_run: bool = False,
        poll_interval: int = 60,
        config_id: str | None = None,
        config_record: Path | None = None,
    ):
        self.name = name
        self.stages = list(stages)
        self.cluster = cluster
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        if config_id is not None:
            for stage in self.stages:
                for task in stage.tasks:
                    task.config_id = config_id
                    task.config_record = config_record
        self.script_dir = cluster.repo / "slurm_jobs" / name
        self.log_dir = cluster.slurm_logs / name
        self._server_jobs: dict[str, int] = {}
        self._server_stages = {s.name: s for s in self.stages if s.server}
        self._clamp_noted: set[str] = set()
        self._server_revivals: dict[str, int] = {}

    # ── preamble ─────────────────────────────────────────────────────────────

    def _preamble(self, stage: Stage) -> str:
        exports = "".join(
            f'export {k}="{v}"\n' for k, v in stage.extra_exports.items()
        )
        nccl = (
            "export NCCL_P2P_DISABLE=1\nexport NCCL_IB_DISABLE=1\n"
            if stage.gpus > 1
            and (stage.nccl_conservative or self.cluster.nccl_conservative)
            else ""
        )
        # After activate(), which sources .env -- this is deliberately the
        # last word on HF_HOME, overriding whatever .env set for the login
        # node. See ClusterConfig.node_local_hf_home.
        hf_home = (
            f'export HF_HOME="{self.cluster.node_local_hf_home}"\n'
            f'mkdir -p "{self.cluster.node_local_hf_home}"\n'
            if self.cluster.node_local_hf_home
            else ""
        )
        # The HF hub client reads HF_TOKEN; .env carries the same secret as
        # HF_API_KEY (the name the rest of the repo uses). Without this, a
        # train/export job that resolves a *private* base repo by id (e.g.
        # value-generalization/neutral-sft-v3-qwen3-8b) 401s on the Hub even
        # though the snapshot is cached. Mirror multivalue eval_stage's
        # HF_TOKEN-or-HF_API_KEY fallback so every rendered job authenticates.
        hf_token = 'export HF_TOKEN="${HF_TOKEN:-${HF_API_KEY:-}}"\n'
        # `-e` matters for task commands that do work *after* the payload:
        # the shared-base eval runs its eval inside an `if`, then links the
        # result into this run's dir. Without it a failed eval fell through to
        # the `ln`, which succeeded and became the task's exit status -- Slurm
        # recorded COMPLETED and left a dangling symlink (job 131419_0). The
        # done-check follows the link, so the stage still retried rather than
        # passing, but the job state lied and all three attempts burned.
        return (
            "set -euo pipefail\n"
            f"{self.cluster.activate(stage.env)}"
            f"{hf_token}"
            f"{hf_home}"
            "export PYTHONUNBUFFERED=1\n"
            f"{nccl}{exports}"
            f"cd {self.cluster.repo}\n"
        )

    # ── placement ────────────────────────────────────────────────────────────

    def _placement(self, stage: Stage) -> tuple[str | None, str | None]:
        """``(partition, qos)`` a stage submits to; ``(None, None)`` leaves
        both to the cluster default (a GPU stage on a cluster that names no
        gpu_partition, or a GPU-less server)."""
        c = self.cluster
        # A stage may opt into the cluster's array partition/QOS (a higher
        # submit cap for big sharded arrays). This applies whether the stage
        # owns its GPUs (stage.gpus > 0) or is a CPU stage borrowing the
        # mandatory cpu_stage_gpus — the label array is the latter, and on a
        # 50-job default QOS a >50-shard array is rejected outright.
        use_array = bool(
            stage.array_partition and c.array_partition and c.array_qos
        )
        if stage.gpus > 0:
            if use_array:
                return c.array_partition, c.array_qos
            if c.gpu_partition and c.gpu_qos:
                return c.gpu_partition, c.gpu_qos
            return None, None
        if stage.controller:
            return (c.controller_partition or c.cpu_partition,
                    c.controller_qos or c.cpu_qos)
        if stage.cpu_partition or not stage.server:
            if use_array:
                return c.array_partition, c.array_qos
            return c.cpu_partition, c.cpu_qos
        return None, None

    def _walltime(self, stage: Stage) -> tuple[str, int]:
        """``(time to request, retry multiplier)`` after the partition cap.

        A stage asking for more than ``cluster.partition_max_time`` allows on
        its partition is clamped to the cap, and its retry budget scaled by
        ``ceil(requested / cap)`` so the stage keeps its configured wall
        budget across resubmits. This is only useful for work that resumes
        (checkpointed label shards, a server that is revived on death, a
        trainer that restarts from its last checkpoint); a task that has to
        finish inside one job simply needs a time within the cap.
        """
        partition, _ = self._placement(stage)
        cap = self.cluster.partition_max_time.get(partition) if partition else None
        if cap is None:
            return stage.time, 1
        want, have = slurm_time_seconds(stage.time), slurm_time_seconds(cap)
        if want <= have:
            return stage.time, 1
        if stage.name not in self._clamp_noted:
            self._clamp_noted.add(stage.name)
            print(
                f"  {stage.name}: time {stage.time} exceeds {partition}'s cap "
                f"{cap}; clamped, retries x{math.ceil(want / have)}",
                flush=True,
            )
        return cap, math.ceil(want / have)

    def _pack(self, stage: Stage) -> int:
        """Tasks per array element for ``stage``, or 0 when packing is off.

        Packing applies to a single-GPU, non-server stage only: explicit
        ``stage.pack`` first, else the cluster's ``gpus_per_node`` (so it is
        off on every cluster that does not declare one).
        """
        if stage.server or stage.gpus != 1 or stage.cpu_partition:
            return 0
        p = stage.pack or (self.cluster.gpus_per_node or 0)
        return p if p > 1 else 0

    def _chunks(self, stage: Stage, tasks: Sequence[Task]) -> list[list[Task]] | None:
        """Group tasks into per-node chunks, or None when packing is off.

        Chunks are contiguous slices of ``pack`` tasks, so array element i
        owns tasks ``[i*pack : (i+1)*pack]``.
        """
        p = self._pack(stage)
        if p and len(tasks) > 1:
            return [list(tasks[i : i + p]) for i in range(0, len(tasks), p)]
        return None

    def _node_setup(self, stage: Stage) -> str:
        """``stage.node_setup`` as a block, or empty. Once per array element."""
        if not stage.node_setup.strip():
            return ""
        return "\n# ── node setup ──\n" + stage.node_setup.rstrip() + "\n"

    def _task_script(self, stage: Stage, task: Task) -> Path:
        """Path of the per-task shell script a packed chunk runs via GNU parallel."""
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", task.key)
        return self.script_dir / stage.name / f"{safe}.sh"

    def _node_gpus(self, stage: Stage, gpus: int) -> int:
        """GPUs one array element really occupies, for throttle math.

        On an exclusive-node cluster (``gpus_per_node`` set) any GPU stage's
        element holds whole nodes, so a nominal 2-GPU request costs a node's
        worth; the same rounding applies to ``gpu_headroom`` (a tp=2 judge
        on such a cluster holds 8 GPUs). Elsewhere GPUs are what they say.
        """
        gpn = self.cluster.gpus_per_node
        if gpn and stage.gpus > 0 and gpus > 0:
            return math.ceil(gpus / gpn) * gpn
        return gpus

    # ── script rendering ─────────────────────────────────────────────────────

    def _directives(
        self, stage: Stage, n_units: int, gpus_per_unit: int, packed: bool = False
    ) -> str:
        """SBATCH directives. A *unit* is one array element: a task normally,
        or a packed chunk of ``pack`` tasks, which is why gres and the
        array/throttle math key on ``gpus_per_unit`` rather than ``stage.gpus``.

        ``packed`` scales the memory request by the chunk size: the chunk runs
        that many tasks at once, so the per-task ``stage.mem`` would cap the
        cgroup for all of them together and OOM the node.
        """
        c = self.cluster
        time, _ = self._walltime(stage)
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --time={time}",
            f"#SBATCH --job-name={self.name}_{stage.name}",
            # Every stage is single-node (multi-GPU work is tp / torchrun on
            # one node). Saying so is a no-op for SLURM proper, but the AMD
            # AUP submit filter rejects jobs that leave the node count implied
            # ("requested nodecount exceeds maximum of 4 nodes").
            "#SBATCH --nodes=1",
        ]
        if c.request_mem:
            mem = _scale_mem(stage.mem, gpus_per_unit) if packed else stage.mem
            lines.append(f"#SBATCH --mem={mem}")
        lines += [
            f"#SBATCH --mail-type={c.mail_type}",
            f"#SBATCH --mail-user={c.mail_user}",
        ]
        if c.account:
            lines.append(f"#SBATCH --account={c.account}")
        if stage.server or n_units == 1:
            lines.append(f"#SBATCH --output={self.log_dir}/{stage.name}_%j.out")
        else:
            lines.append(
                f"#SBATCH --output={self.log_dir}/{stage.name}_%A_%a.out"
            )
        # GPUs this job actually requests: the stage's own (per unit -- a whole
        # packed node when packing), or the cluster's mandatory minimum for
        # stages that need none (GPU-only partitions).
        gpus = gpus_per_unit if gpus_per_unit > 0 else c.cpu_stage_gpus
        partition, qos = self._placement(stage)
        if stage.gpus > 0:
            if c.request_gres:
                lines += c.gres_lines(gpus_per_unit, gpu_type=stage.gpu_type)
            if partition:
                lines.append(f"#SBATCH --partition={partition}")
                lines.append(f"#SBATCH --qos={qos}")
        elif partition:
            lines.append(f"#SBATCH --partition={partition}")
            lines.append(f"#SBATCH --qos={qos}")
            if gpus > 0 and c.request_gres:
                # Untyped: a poller/label shard runs on any GPU.
                lines.append(f"#SBATCH --gres=gpu:{gpus}")
        if gpus > 0 and c.exclude:
            lines.append(f"#SBATCH --exclude={c.exclude_arg}")
        if not stage.server and n_units > 1:
            throttle = stage.throttle
            if throttle == 0 and gpus > 0:
                # The cap counts GPUs, not units: a unit is a whole node when
                # packing (or on any exclusive-node cluster), one GPU otherwise.
                unit = self._node_gpus(stage, gpus)
                headroom = self._node_gpus(stage, stage.gpu_headroom)
                throttle = max(1, (c.max_concurrent_gpus - headroom) // unit)
            spec = f"0-{n_units - 1}" + (f"%{throttle}" if throttle else "")
            lines.append(f"#SBATCH --array={spec}")
        return "\n".join(lines) + "\n"

    def render(self, stage: Stage, tasks: Sequence[Task]) -> str:
        """SBATCH script text arraying over ``tasks`` (or packed chunks)."""
        chunks = self._chunks(stage, tasks)
        if chunks is not None:
            return self._render_packed(stage, chunks)
        body = (
            self._directives(stage, len(tasks), stage.gpus)
            + "\n"
            + self._preamble(stage)
            + self._node_setup(stage)
        )
        if stage.server or len(tasks) == 1:
            task = tasks[0]
            body += f"\n# task: {task.key}\n{task.command}\n"
            return body
        body += "\ncase ${SLURM_ARRAY_TASK_ID:-0} in\n"
        for i, task in enumerate(tasks):
            body += f"{i})  # {task.key}\n{task.command}\n;;\n"
        body += "esac\n"
        return body

    def _render_packed(self, stage: Stage, chunks: list[list[Task]]) -> str:
        """One array element per chunk; each fans its ≤pack single-GPU tasks
        across the node's GPUs with GNU parallel.

        Each task's command lives in its own ``_task_script`` file (written by
        ``_write_script``), so arbitrary multi-line commands need no heredoc
        escaping and stay inspectable. ``{#}`` is parallel's 1-based *sequence
        number* (fixed by input order), so input line n is pinned to GPU n-1
        and each task sees exactly one GPU as device 0. Both the CUDA and the
        HIP spelling are exported: ROCm honours either, and a CUDA node ignores
        the HIP one. Deliberately not ``{%}`` (parallel's *concurrency slot*):
        that is reused whenever a slot frees up, which is the wrong primitive
        for a stable per-task GPU index and collided in practice (2026-08-20:
        two tasks of one chunk both got slot 2 -> the same physical GPU -> one
        OOM'd the other's KV cache init). No ``--halt``: like a plain array,
        every task in the chunk runs regardless of its neighbours' failures,
        and parallel's default exit status is the number of failed jobs — so a
        chunk with any failure exits non-zero, is marked FAILED, and the retry
        loop re-runs it. Because render only ever sees ``stage.pending()``, the
        next attempt re-packs just the still-failed tasks.
        """
        pack = self._pack(stage)
        body = (
            self._directives(stage, len(chunks), pack, packed=True)
            + "\n"
            + self._preamble(stage)
            + self._node_setup(stage)
        )
        body += "\ncase ${SLURM_ARRAY_TASK_ID:-0} in\n"
        for i, chunk in enumerate(chunks):
            files = " ".join(
                shlex.quote(str(self._task_script(stage, t))) for t in chunk
            )
            keys = ", ".join(t.key for t in chunk)
            body += (
                f"{i})  # {keys}\n"
                f"printf '%s\\n' {files} | "
                f"parallel --jobs {pack} --line-buffer "
                f"--tagstring '{{/.}}' "
                f"'CUDA_VISIBLE_DEVICES=$(({{#}} - 1)) "
                f"HIP_VISIBLE_DEVICES=$(({{#}} - 1)) bash {{}}'\n"
                f";;\n"
            )
        body += "esac\n"
        return body

    def _write_script(self, stage: Stage, tasks: Sequence[Task]) -> Path:
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.script_dir / f"{stage.name}.sbatch"
        path.write_text(self.render(stage, tasks))
        path.chmod(0o755)
        chunks = self._chunks(stage, tasks)
        if chunks is not None:
            # Each packed task runs as `bash task.sh` in a parallel subshell, so
            # it inherits the preamble's exported env (venv, HF_HOME, cd) but not
            # its `set -euo pipefail` — re-establish it per task to keep the same
            # fail-fast the non-packed case gets from the shared parent shell.
            (self.script_dir / stage.name).mkdir(parents=True, exist_ok=True)
            for t in tasks:
                tp = self._task_script(stage, t)
                tp.write_text("set -euo pipefail\n" + t.command + "\n")
                tp.chmod(0o755)
        # Sidecar task map: which array index is which task (for humans/sacct).
        # In pack mode an index is a chunk, so it maps to the chunk's task keys.
        task_map = self.script_dir / f"{stage.name}_task_map.json"
        if chunks is not None:
            mapping = {str(i): [t.key for t in c] for i, c in enumerate(chunks)}
        else:
            mapping = {str(i): t.key for i, t in enumerate(tasks)}
        task_map.write_text(json.dumps(mapping, indent=2))
        return path

    # ── SLURM plumbing ───────────────────────────────────────────────────────

    def _sbatch(self, script: Path, dependency: str | None = None) -> int | None:
        if self.dry_run:
            suffix = f" (dependency: {dependency})" if dependency else ""
            print(f"  [dry-run] would submit: {script}{suffix}")
            return None
        command = ["sbatch", "--parsable"]
        if dependency:
            command.append(f"--dependency={dependency}")
        command.append(str(script))
        result = subprocess.run(
            command, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  sbatch failed: {result.stderr.strip()}", file=sys.stderr)
            return None
        # --parsable prints "jobid" or "jobid;cluster"
        match = re.match(r"(\d+)", result.stdout.strip())
        if not match:
            print(f"  cannot parse job id: {result.stdout!r}", file=sys.stderr)
            return None
        job_id = int(match.group(1))
        print(f"  submitted job {job_id} ({script.name})")
        return job_id

    def _find_active_job(self, stage: Stage) -> int | None:
        """The id of a live SLURM job already carrying ``stage``'s job name.

        Job names are ``{pipeline}_{stage}`` (see ``_directives``), so a
        queued/running job with this name is a previous controller's
        submission of the very same work -- a train array still running
        after the controller that launched it was cancelled or preempted, or
        a served judge that outlived it. ``_server_jobs`` is in-memory and
        does not survive a controller restart, so without this lookup a
        resubmitted controller stacks a second array (two writers on the same
        checkpoints) or a second tp=8 judge.

        Matching uses ``%F`` (array master id), not ``%A``: SLURM gives every
        *running* array element its own job id and reports it as ``%A``, so
        ``%A`` would adopt a single element. Waiting on that id returns as soon
        as that one task ends -- the controller then calls the whole stage
        finished, burns a retry on the still-pending rest, and can exhaust its
        attempts while the array is healthy. ``%F`` is the array itself (and
        the job id for a non-array job), which stays active until every element
        is done.
        """
        if self.dry_run:
            return None
        result = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-n",
             f"{self.name}_{stage.name}", "-o", "%F %T"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] in _ACTIVE_STATES:
                return int(parts[0])
        return None

    @staticmethod
    def _job_active(job_id: int) -> bool:
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            capture_output=True,
            text=True,
        )
        return any(
            s.strip() in _ACTIVE_STATES for s in result.stdout.splitlines()
        )

    def _wait(self, job_id: int, label: str, servers: Sequence[str] = ()) -> None:
        """Poll until ``job_id`` leaves the queue.

        ``servers`` names server stages this job depends on; a server that
        dies mid-array (the oracle crashing, or timing out at a partition's
        walltime cap, during label task 0) is revived here so the array's
        *remaining* tasks find a live one, instead of each polling the corpse
        and failing until the whole array drains. ``wait_ready`` re-reads the
        hostfile every poll, so tasks already blocked on the dead host:port
        pick up the revived server too.
        """
        while self._job_active(job_id):
            if servers:
                self._ensure_servers(servers)
            now = time.strftime("%H:%M:%S")
            print(f"  [{now}] {label}: job {job_id} still running ...", flush=True)
            time.sleep(self.poll_interval)
        print(f"  {label}: job {job_id} finished")

    def _settle(self, stage: Stage, attempts: int = 6, delay: float = 5.0) -> None:
        """Let the shared filesystem catch up with a just-finished job.

        ``squeue`` drops a job before NFS necessarily shows what it wrote, so a
        done-check fired the instant a task ends can miss its own output and
        resubmit the whole array for nothing. Short-lived tasks lose that race
        routinely; long ones can still lose it on their last writer.
        """
        for _ in range(attempts):
            if not stage.pending():
                return
            time.sleep(delay)

    def _ensure_servers(self, names: Sequence[str]) -> None:
        """Resubmit any of ``names`` whose job is no longer active.

        A vLLM server can die mid-array (node kill, OOM, preemption) long after
        it came up healthy. Nothing downstream notices: the hostfile it wrote is
        still on disk with a dead ``host:port``, so every subsequent attempt's
        ``wait_ready`` polls a corpse for its full timeout and then exits 1
        without running a single eval. Reviving here restores the invariant the
        retry loop assumes — that a resubmitted task can actually make progress.
        ``run_stage`` unlinks the hostfile before it resubmits a server, so the
        dependents block on the *new* server rather than racing the stale file.
        """
        if self.dry_run:
            return
        for name in names:
            stage = self._server_stages.get(name)
            if stage is None:
                continue
            job_id = self._server_jobs.get(name)
            if job_id is not None and self._job_active(job_id):
                continue
            adopted = self._find_active_job(stage)
            if adopted is not None:
                print(f"  server {name}: adopting live job {adopted}", flush=True)
                self._server_jobs[name] = adopted
                continue
            was = f"job {job_id} is gone" if job_id is not None else "never started"
            # A server that dies on arrival (bad serve flags, a model the
            # image can't load) would otherwise be resubmitted on every poll
            # for as long as its dependents keep waiting. Cap revivals at the
            # stage's retry budget, scaled like any clamped stage's; past it
            # the dependents time out on their own and the stage fails.
            _, multiplier = self._walltime(stage)
            budget = stage.max_retries * multiplier
            revivals = self._server_revivals.get(name, 0)
            if revivals >= budget:
                if revivals == budget:
                    self._server_revivals[name] = revivals + 1
                    print(
                        f"  server {name}: {was}; not resubmitting again after "
                        f"{budget} revivals — check its logs",
                        file=sys.stderr, flush=True,
                    )
                continue
            self._server_revivals[name] = revivals + 1
            print(f"  server {name}: {was} — resubmitting ({revivals + 1}/{budget})", flush=True)
            self.run_stage(stage)

    # ── verbs ────────────────────────────────────────────────────────────────

    def run_stage(
        self, stage: Stage, wait: bool = True, dependency: str | None = None
    ) -> tuple[bool, int | None]:
        """Submit a stage's pending tasks; poll + retry until done or spent.

        Returns ``(complete, job_id)``. ``job_id`` is non-None only when this
        invocation submitted work, and is used by :meth:`submit` to wire a
        non-blocking pipeline with SLURM dependencies. ``run`` ignores it and
        retains its filesystem-polling semantics.
        """
        if stage.server:
            adopted = self._find_active_job(stage)
            if adopted is not None:
                # Keep its hostfile: the live server wrote it, and unlinking
                # would strand every dependent on a file nobody rewrites.
                print(f"  {stage.name}: adopting live server job {adopted}", flush=True)
                self._server_jobs[stage.name] = adopted
                return True, adopted
            if stage.hostfile is not None and not self.dry_run:
                stage.hostfile.unlink(missing_ok=True)
            script = self._write_script(stage, stage.tasks)
            job_id = self._sbatch(script, dependency)
            if job_id is not None:
                self._server_jobs[stage.name] = job_id
            return True, job_id

        # A stage clamped to its partition's walltime cap gets proportionally
        # more attempts, so resumable work keeps its configured wall budget.
        _, multiplier = self._walltime(stage)
        max_retries = stage.max_retries * multiplier
        for attempt in range(1, max_retries + 1):
            pending = stage.pending()
            if not pending:
                print(f"\n{stage.name}: all {len(stage.tasks)} tasks done")
                return True, None
            print(
                f"\n{stage.name} (attempt {attempt}/{max_retries}): "
                f"{len(pending)}/{len(stage.tasks)} tasks pending"
            )
            for t in pending:
                print(f"  - {t.key}")
            self._ensure_servers(stage.needs_servers)
            adopted = self._find_active_job(stage)
            if adopted is not None:
                # A previous controller's array is still on this work: wait on
                # it as this attempt rather than submitting a duplicate.
                print(f"  {stage.name}: adopting live job {adopted}", flush=True)
                if not wait:
                    return False, adopted
                self._wait(adopted, stage.name, stage.needs_servers)
                self._settle(stage)
                continue
            script = self._write_script(stage, pending)
            job_id = self._sbatch(script, dependency)
            if self.dry_run:
                return True, None
            if not wait:
                return False, job_id
            if job_id is None:
                continue
            self._wait(job_id, stage.name, stage.needs_servers)
            self._settle(stage)

        remaining = stage.pending()
        if remaining:
            print(
                f"\n{stage.name}: {len(remaining)} tasks FAILED after "
                f"{max_retries} attempts: {[t.key for t in remaining]}",
                file=sys.stderr,
            )
            return False, None
        return True, None

    def run(self) -> bool:
        """Run all stages in order; cancel any server jobs on the way out.

        A stage that names ``cancel_servers`` releases those servers as soon as
        it completes, rather than at the end of the pipeline; its ``after``
        hook, if any, runs right behind that.
        """
        ok = True
        try:
            for stage in self.stages:
                if stage.reaper:
                    # The poller reaps its own servers (below / on the way out);
                    # the reaper job only exists for the non-blocking submit path.
                    print(f"\n{stage.name}: skipped — the poller releases servers itself")
                    continue
                if stage.server and stage.on_demand:
                    print(f"\n{stage.name}: on demand — served by its first dependent stage")
                    continue
                complete, _ = self.run_stage(stage)
                if not complete:
                    ok = False
                    break
                if stage.cancel_servers:
                    self._cancel_servers(stage.cancel_servers)
                if stage.after is not None and not self.dry_run:
                    stage.after()
        finally:
            self._cancel_servers()
        return ok

    @staticmethod
    def _value_key(task: Task) -> str:
        """The part of a train_{mtag} task's key that identifies *what* it
        trains, independent of *which model*. ``train_stages()`` always keys
        tasks ``"{mtag}/{value}"``; this is what has to line up positionally
        between sibling ``train_{mtag}`` stages for ``aftercorr`` to be safe.
        """
        return task.key.split("/", 1)[-1]

    def submit(self, seed_dependency: str | None = None) -> None:
        """Submit a non-blocking pipeline with ordered SLURM dependencies.

        ``seed_dependency`` (e.g. ``"afterany:12345"``) is applied to the first
        submitted stage, to hang this pipeline off a job submitted elsewhere —
        such as wiring arm eval onto an already-running train array. Later
        stages chain from it as usual.

        Unlike :meth:`run`, this does not poll or retry.  Every pending stage
        is submitted immediately, but each stage is held until its predecessor
        succeeds (``afterok``).  A persistent server is special: its successor
        uses ``after`` because the server is supposed to remain running; the
        task-level hostfile/HTTP readiness check handles actual availability.
        Completed stages introduce no dependency, so a later invocation can
        safely submit only missing work.

        Sibling ``train_{mtag}`` stages are a special case: ``train_stages()``
        builds every model's array by iterating the same ``values`` list in
        the same order, so array index i means the same value in every model's
        stage. Between two such stages we use ``aftercorr`` instead — value i
        of the next model can start the moment value i of this model finishes,
        instead of waiting for the whole array's stragglers. This is only
        submitted when the two stages' *pending* task lists still line up
        index-for-index (checked explicitly, not just same length): a resumed
        run can leave different models with different subsets pending, which
        would silently miswire ``aftercorr``'s index correspondence. Any
        stage pair outside train_*/train_* (evals, servers, mismatched
        resumes) keeps the existing after/afterok behavior.
        """
        prev: tuple[Stage, list[Task], int] | None = None
        seeded = False
        for stage in self.stages:
            pending = stage.pending()
            if not pending:
                # A completed predecessor is already a satisfied dependency.
                prev = None
                continue

            dependency: str | None = None
            if prev is None and not seeded and seed_dependency is not None:
                # Hang the first submitted stage off an external job.
                dependency = seed_dependency
            seeded = True
            if prev is not None:
                prev_stage, prev_pending, prev_job_id = prev
                if stage.reaper:
                    # Fire once the predecessor reaches ANY terminal state: a
                    # reaper that only ran on success would leak the server
                    # exactly when the eval it fed failed.
                    dependency = f"afterany:{prev_job_id}"
                elif prev_stage.server:
                    dependency = f"after:{prev_job_id}"
                elif (
                    prev_stage.name.startswith("train_")
                    and stage.name.startswith("train_")
                    and [self._value_key(t) for t in prev_pending]
                    == [self._value_key(t) for t in pending]
                ):
                    dependency = f"aftercorr:{prev_job_id}"
                else:
                    dependency = f"afterok:{prev_job_id}"

            _, job_id = self.run_stage(stage, wait=False, dependency=dependency)
            if job_id is None:
                if self.dry_run:
                    # There is no real job id to reference, but dry-run must
                    # still render every stage for inspection.
                    prev = None
                    continue
                # Do not enqueue children if their immediate prerequisite
                # could not be submitted; surface the failure on this run.
                break
            prev = (stage, list(pending), job_id)

    def _cancel_servers(self, names: Sequence[str] | None = None) -> None:
        targets = list(self._server_jobs) if names is None else list(names)
        for name in targets:
            job_id = self._server_jobs.pop(name, None)
            if job_id is None:
                continue
            subprocess.run(["scancel", str(job_id)], capture_output=True)
            print(f"  cancelled server job {job_id} ({name})")

    def status(self) -> None:
        print(f"\nPipeline status: {self.name}")
        width = max((len(s.name) for s in self.stages), default=10) + 2
        print("─" * (width + 30))
        for stage in self.stages:
            if stage.server:
                state = "(server stage — submitted on demand)"
                print(f"  {stage.name:<{width}} {state}")
                continue
            done, total = stage.done_count(), len(stage.tasks)
            mark = "OK " if done == total else "   "
            print(f"  {stage.name:<{width}} {mark}{done}/{total}")
            for t in stage.tasks:
                if not t.is_done():
                    print(f"      - {t.key}")
        print("─" * (width + 30))


class RunPipelineOrchestrator:
    """v1 delegate to persona_vectors' ``run_pipeline.py``.

    Presents the same ``run``/``status`` surface as :class:`Orchestrator` but
    shells out to the submodule's own orchestrator in its own conda env (it is
    already idempotent, array-based, and filesystem-checked). v2 replaces this
    with native :class:`Stage` specs for its four stages
    (generate/extract/sweep/aggregate).
    """

    def __init__(
        self,
        cluster: ClusterConfig,
        value_set_path: Path,
        model: str,
        experiment: str,
        extra_args: Sequence[str] = (),
        gpus: int = 1,
        dry_run: bool = False,
    ):
        from valuegen._external import persona_vectors_root

        self.root = persona_vectors_root()
        self.cluster = cluster
        self.value_set_path = Path(value_set_path).resolve()
        self.model = model
        self.experiment = experiment
        self.extra_args = list(extra_args)
        self.gpus = max(1, int(gpus))
        self.dry_run = dry_run

    def _command(self, verb: str) -> list[str]:
        cmd = [
            "python",
            "run_pipeline.py",
            verb,
            "--value-set",
            str(self.value_set_path),
            "--model",
            self.model,
            "--experiment",
            self.experiment,
        ]
        if verb == "run":
            cmd += ["--exclude", self.cluster.exclude_arg]
            cmd += self._cluster_flags()
            # Same concurrency cap the native Orchestrator derives for its
            # GPU arrays; run_pipeline defaults to unthrottled otherwise.
            throttle = max(1, self.cluster.max_concurrent_gpus // self.gpus)
            cmd += ["--array-throttle", str(throttle)]
            if self.dry_run:
                cmd.append("--dry-run")
            cmd += self.extra_args
        return cmd

    def _cluster_flags(self) -> list[str]:
        """cluster.yaml's partition/QOS/GPU facts as run_pipeline flags.

        Always explicit, so the fork's own defaults never decide where jobs
        land. Its GPU arrays (extraction, layer sweep) are short idempotent
        tasks, so they take the array pool when one is configured, like the
        native persona sweep; an empty value omits the directive."""
        c = self.cluster
        gpu_partition = c.array_partition or c.gpu_partition or ""
        gpu_qos = (c.array_qos if c.array_partition else c.gpu_qos) or ""
        types = c.gpu_types
        gpu_type = types[0] if len(types) == 1 else ""
        constraint = "|".join(types) if len(types) > 1 else ""
        return [
            "--cpu-partition", c.cpu_partition or "",
            "--cpu-qos", c.cpu_qos or "",
            "--gpu-partition", gpu_partition,
            "--gpu-qos", gpu_qos,
            "--cpu-stage-gpus", str(c.cpu_stage_gpus),
            "--gpu-type", gpu_type,
            "--gpu-constraint", constraint,
        ]

    def _shell(self, verb: str) -> bool:
        inner = self.cluster.activate("persona") + shlex.join(self._command(verb))
        result = subprocess.run(["bash", "-c", inner], cwd=self.root)
        return result.returncode == 0

    def run(self) -> bool:
        return self._shell("run")

    def status(self) -> None:
        self._shell("status")
