"""Cluster and experiment configuration loading.

Nothing under ``src/valuegen`` hardcodes cluster facts (conda paths, tir roots,
node excludes, ...). They all come from ``configs/cluster.yaml``, loaded here.
Point valuegen at a different cluster file with ``--cluster`` on the CLI or the
``VALUEGEN_CLUSTER`` environment variable.

Experiment configs (``configs/experiments/ground_truth/*.yaml``) are loaded as plain dicts —
the schema belongs to the method drivers (each validates the keys it uses), and
the ``train:`` block in particular is forwarded verbatim to TRL's ``TrlParser``
without valuegen redeclaring any field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from valuegen._external import REPO_ROOT

DEFAULT_CLUSTER_YAML = REPO_ROOT / "configs" / "cluster.yaml"

# Ground truth decomposes into two config blocks with separate identities:
# `intervention:` (how models are steered — trained checkpoints) and `evaluation:` (how steered
# models are scored into a GT artifact). Eval knobs must never move an
# intervention's hash, and vice versa — that separation is what lets one set
# of checkpoints be evaluated under many eval configs without retraining.
INTERVENTION_SCHEMA_VERSIONS = {
    "label_subset": 2,
    "conflictscope_pairs": 2,
    "model_spec_aft": 1,
    "none": 1,
}
# Methods whose interventions are trained checkpoints (the rest are eval-time).
TRAINING_METHODS = ("label_subset", "conflictscope_pairs", "model_spec_aft")

EVAL_SCHEMA_VERSIONS = {
    "conflictscope": 1,
}

# Scenario pools (generate → filter → dedup → split) are their own shared
# artifact: the conflictscope_pairs intervention trains on a pool's train
# half, the conflictscope eval scores on its test half (or on any plain
# scenario dir). Keyed by pool identity, not by experiment.
POOL_SCHEMA_VERSION = 1


@dataclass
class GcsConfig:
    """Opt-in remote checkpoint storage (``gcs:`` block in cluster.yaml).

    Present only when the block exists with ``enabled: true``; everything
    downstream keys off ``cluster.gcs is not None``, so clusters without the
    block behave exactly as before. ``bucket`` is the ``gs://`` bucket plus
    prefix that merged checkpoints are pushed under; ``stage_dir`` is
    node-local scratch on compute nodes (e.g. ``/mnt/localssd/...``) where
    eval jobs stage remote checkpoints back in.
    """

    bucket: str
    stage_dir: Path


@dataclass
class ClusterConfig:
    """Typed view of ``configs/cluster.yaml``."""

    envs: dict[str, str]
    repo: Path
    finetune_root: Path
    finetune_root_legacy: Path
    data: Path
    slurm_logs: Path
    mail_type: str
    mail_user: str
    # None on clusters whose GRES is untyped -- `scontrol show config` reports
    # `GresTypes = gpu` and nodes advertise a bare `gpu:N`. A typed request
    # matches no node there, so the type segment has to be omitted entirely.
    # One name (``A6000``) renders the typed ``--gres=gpu:A6000:N``; a list
    # (``[L40S, A6000]``) renders an untyped ``--gres=gpu:N`` plus
    # ``--constraint=L40S|A6000`` so a job takes whichever pool frees first.
    gpu_type: str | list[str] | None
    max_concurrent_gpus: int
    default_time: str
    default_mem: str
    cpu_partition: str
    cpu_qos: str
    # GPU partition/QOS for ordinary (non-array-opt-in) GPU stages. None falls
    # back to whatever partition the cluster defaults to, which is how this
    # repo behaved before these existed. Set them when the default is
    # preemptible and you have something better: a requeued job can restart
    # on top of its own checkpoints, so long-lived stages (a served judge,
    # a multi-hour train) want a PreemptMode=OFF partition even when it
    # queues longer.
    gpu_partition: str | None = None
    gpu_qos: str | None = None
    # Emitted as --account on every job. Usually redundant (SLURM resolves
    # the association's default account), but a grant-specific QOS is often
    # tied to one account, and being explicit survives a changed default.
    account: str | None = None
    # Optional second GPU pool for short, idempotent array stages. On clusters
    # where the default QOS caps GPUs (and running jobs) per user, a separate
    # array QOS lets predictor extractions run alongside training/eval instead
    # of queueing behind them. Usually preemptible — only stages that can be
    # requeued for free should opt in (Stage.array_partition).
    array_partition: str | None = None
    array_qos: str | None = None
    # Per-job HF cache, exported into every generated SBATCH preamble. Points
    # at node-local scratch on clusters where the shared filesystem can't hold
    # model weights: a 51.8GB judge is transient serving state and has no
    # business on a quota'd or full shared mount. None leaves HF_HOME alone
    # (whatever .env or the environment says).
    #
    # Only safe because generated jobs always run on compute nodes. Don't
    # reuse this for anything the login node evaluates -- and don't expect it
    # to survive the job: under `job_container/tmpfs` (check `scontrol show
    # config | grep NamespaceType`) each job gets a private /tmp that is
    # destroyed at exit, so every job re-fetches. That is the tradeoff being
    # made -- re-download per job, in exchange for not needing shared space.
    node_local_hf_home: str | None = None
    # Export NCCL_P2P_DISABLE/NCCL_IB_DISABLE on every multi-GPU stage. The
    # escape hatch for clusters where GPU peer-to-peer is actually broken (the
    # original A6000 workaround) — on NVLink nodes the flags roughly halve
    # multi-GPU throughput, so the default is off and stages can opt in
    # individually via Stage.nccl_conservative.
    nccl_conservative: bool = False
    # GPUs attached to stages that ask for none (``Stage.gpus == 0``). 0 keeps
    # them pure-CPU. Clusters whose every partition demands >=1 GPU
    # set 1, which also makes those arrays count against
    # ``max_concurrent_gpus`` for throttling.
    cpu_stage_gpus: int = 0
    # Partition/QOS for long-lived poll-only controllers (``gt run`` drivers,
    # experiment controllers). Unset → ``cpu_partition``/``cpu_qos``. A
    # preemptible partition with a long walltime cap suits them: they resume
    # by resubmit and the work they watch lives in their own jobs.
    controller_partition: str | None = None
    controller_qos: str | None = None
    # ── Exclusive-node clusters (no GRES; whole nodes per job) ──────────────
    # Whether to emit ``--gres`` / ``--mem`` at all. The AMD AUP cluster has
    # ``GresTypes=(null)`` (any --gres is rejected) and advertises RealMemory=1
    # (any --mem is "node configuration not available"): GPUs are implied by
    # the partition and memory by the node. Both default on, which is every
    # cluster this repo ran on before.
    request_gres: bool = True
    request_mem: bool = True
    # GPUs per node on a partition that allocates whole nodes
    # (OverSubscribe=EXCLUSIVE). When set, a multi-task single-GPU stage
    # (Stage.pack) grabs a full node per array element and fans that many
    # tasks across its GPUs with GNU parallel, instead of leaving N-1 GPUs
    # idle per element; and the throttle counts nodes' worth of GPUs, not the
    # stage's nominal request. None = one array element per task, as before.
    gpus_per_node: int | None = None
    # Walltime cap per partition (SLURM time syntax), for clusters whose
    # submit filter enforces caps sinfo does not show. A stage whose time
    # exceeds its partition's cap is clamped to it and its retry budget is
    # scaled by the ratio, so resumable stages (checkpointed label shards, a
    # revived server) keep their configured wall budget across resubmits.
    partition_max_time: dict[str, str] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)
    model_paths: dict[str, Path] = field(default_factory=dict)
    # Remote checkpoint storage; None (no `gcs:` block, or enabled: false)
    # keeps every checkpoint on finetune_root as before.
    gcs: GcsConfig | None = None
    source_path: Path | None = None

    def env(self, key: str) -> str:
        """uv venv path for a role key (``default``, ``persona``, ...)."""
        try:
            return self.envs[key]
        except KeyError:
            raise KeyError(
                f"No env {key!r} in {self.source_path}; "
                f"known: {sorted(self.envs)}"
            ) from None

    def venv(self, key: str) -> Path:
        """Absolute path to role ``key``'s venv."""
        p = Path(self.env(key))
        return p if p.is_absolute() else self.repo / p

    def activate(self, key: str) -> str:
        """Shell that puts role ``key``'s interpreter on PATH, newline-terminated.

        The single place an env becomes shell — SBATCH preambles and the fork
        subprocesses all route through here, so nothing re-derives it.

        Also sources the repo's gitignored ``.env``. Under conda the API keys rode
        along in ``conda env config vars``: invisible, unshareable, and secrets kept
        in env metadata. A venv has no such channel, so the keys come from a file
        a collaborator can actually be told to create.
        """
        return (
            f'source "{self.venv(key)}/bin/activate"\n'
            f'set -a; [ -f "{self.repo}/.env" ] && . "{self.repo}/.env"; set +a\n'
        )

    @property
    def gpu_types(self) -> list[str]:
        if self.gpu_type is None:
            return []
        return [self.gpu_type] if isinstance(self.gpu_type, str) else list(self.gpu_type)

    def gres_lines(self, n: int, gpu_type: str | list[str] | None = None) -> list[str]:
        """``#SBATCH`` lines requesting ``n`` GPUs of any configured type.

        ``gpu_type`` narrows a single stage to one type or subset (e.g. a
        throughput-sensitive vLLM server pinned to L40S) without touching the
        cluster-wide default; it must be one of the configured types."""
        types = self.gpu_types
        if gpu_type:
            wanted = [gpu_type] if isinstance(gpu_type, str) else list(gpu_type)
            unknown = [t for t in wanted if types and t not in types]
            if unknown:
                raise ValueError(
                    f"gpu_type {unknown} not among the cluster's gpu_type {types}"
                )
            types = wanted
        if len(types) == 1:
            return [f"#SBATCH --gres=gpu:{types[0]}:{n}"]
        lines = [f"#SBATCH --gres=gpu:{n}"]
        if types:
            lines.append(f"#SBATCH --constraint={'|'.join(types)}")
        return lines

    @property
    def exclude_arg(self) -> str:
        """Comma-joined node list for ``#SBATCH --exclude`` (always emitted)."""
        return ",".join(self.exclude)

    def model_path(self, model: str) -> Path | None:
        """Cluster-local load path for a served model, if one is configured."""
        return self.model_paths.get(model)


def load_cluster(path: str | Path | None = None) -> ClusterConfig:
    """Load the cluster config (default: repo ``configs/cluster.yaml``).

    Resolution order: explicit ``path`` arg, ``VALUEGEN_CLUSTER`` env var,
    the repo default.
    """
    if path is None:
        path = os.environ.get("VALUEGEN_CLUSTER") or DEFAULT_CLUSTER_YAML
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    paths, slurm = raw["paths"], raw["slurm"]
    repo = Path(paths["repo"])
    gcs = _load_gcs(raw.get("gcs"), path)

    def _repo_rel(key: str) -> Path:
        p = Path(paths[key])
        return p if p.is_absolute() else repo / p

    return ClusterConfig(
        envs=dict(raw["envs"]),
        repo=repo,
        finetune_root=Path(paths["finetune_root"]),
        finetune_root_legacy=Path(paths["finetune_root_legacy"]),
        data=_repo_rel("data"),
        slurm_logs=_repo_rel("slurm_logs"),
        mail_type=slurm["mail_type"],
        mail_user=slurm["mail_user"],
        gpu_type=slurm.get("gpu_type") or None,
        max_concurrent_gpus=int(slurm["max_concurrent_gpus"]),
        default_time=slurm["default_time"],
        default_mem=slurm["default_mem"],
        cpu_partition=slurm["cpu_partition"],
        cpu_qos=slurm["cpu_qos"],
        gpu_partition=slurm.get("gpu_partition"),
        gpu_qos=slurm.get("gpu_qos"),
        account=slurm.get("account"),
        array_partition=slurm.get("array_partition"),
        array_qos=slurm.get("array_qos"),
        node_local_hf_home=paths.get("node_local_hf_home"),
        nccl_conservative=bool(slurm.get("nccl_conservative", False)),
        cpu_stage_gpus=int(slurm.get("cpu_stage_gpus") or 0),
        controller_partition=slurm.get("controller_partition"),
        controller_qos=slurm.get("controller_qos"),
        request_gres=bool(slurm.get("request_gres", True)),
        request_mem=bool(slurm.get("request_mem", True)),
        gpus_per_node=_optional_int(slurm, "gpus_per_node", path),
        partition_max_time={
            str(k): str(v) for k, v in (slurm.get("partition_max_time") or {}).items()
        },
        exclude=list(slurm.get("exclude") or []),
        model_paths={
            model: Path(local_path)
            for model, local_path in (paths.get("model_paths") or {}).items()
        },
        gcs=gcs,
        source_path=path,
    )


def _optional_int(block: dict, key: str, source: Path) -> int | None:
    value = block.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source}: slurm.{key} must be a positive integer, got {value!r}")
    return value


def _load_gcs(block, source: Path) -> GcsConfig | None:
    if not block or not block.get("enabled"):
        return None
    bucket = str(block.get("bucket") or "").rstrip("/")
    if not bucket.startswith("gs://"):
        raise ValueError(
            f"{source}: gcs.bucket must be a gs:// URI (got {bucket!r})"
        )
    stage_dir = block.get("stage_dir")
    if not stage_dir:
        raise ValueError(f"{source}: gcs.stage_dir is required when gcs is enabled")
    stage_dir = Path(stage_dir)
    if not stage_dir.is_absolute():
        raise ValueError(
            f"{source}: gcs.stage_dir must be absolute (it names node-local "
            f"scratch on compute nodes, not a repo path); got {stage_dir}"
        )
    return GcsConfig(bucket=bucket, stage_dir=stage_dir)


def _merge_defaults(target: dict, defaults: dict) -> None:
    for key, value in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            _merge_defaults(target[key], value)


def resolve_pool(pool: dict, default_value_set=None) -> dict:
    """Materialize a scenario-pool block's defaults (in place; also returned)."""
    if default_value_set is not None:
        pool.setdefault("value_set", default_value_set)
    if "value_set" not in pool:
        raise ValueError("scenario pool block needs a 'value_set'")
    _merge_defaults(pool, {
        "schema_version": POOL_SCHEMA_VERSION,
        "scenario_gen": {
            "model": "gpt-4.1", "num_scenarios": 10, "temperature": None,
            "time": "2-00:00:00", "mem": "16G", "gpus": 0, "env": "eval_api",
        },
        "dedup": {
            "embedding_model": "all-MiniLM-L6-v2",
            "threshold": 0.9,
            "existing_train": [],
        },
        "split": {"test_size": 200, "seed": 42},
    })
    pool["scenario_gen"].setdefault("filter_model", pool["scenario_gen"]["model"])
    return pool


def _all_values(value_set) -> list[str]:
    from valuegen.values import load_value_set

    path = Path(str(value_set))
    if path.suffix != ".json":
        path = REPO_ROOT / "value_sets" / f"{path}.json"
    elif not path.is_absolute():
        path = REPO_ROOT / path
    return list(load_value_set(path).keys())


def _resolve_intervention(iv: dict) -> None:
    method = iv.get("method")
    if method not in INTERVENTION_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unknown intervention.method {method!r}; known: "
            f"{sorted(INTERVENTION_SCHEMA_VERSIONS)}"
        )
    if "value_set" not in iv:
        raise ValueError("intervention block needs a 'value_set'")
    if "models" not in iv:
        raise ValueError("intervention block needs 'models'")
    iv.setdefault("schema_version", INTERVENTION_SCHEMA_VERSIONS[method])

    if method in TRAINING_METHODS:
        if not iv.get("values"):
            raise ValueError(f"intervention.method {method} needs 'values' to train on")
        if "max_pairs" not in iv:
            nested = iv.get("labels", {}) if method == "label_subset" else iv.get("pairs", {})
            iv["max_pairs"] = nested.get("max_pairs")
        if method == "model_spec_aft" and iv.setdefault("algo", "sft") != "sft":
            raise ValueError(
                "model_spec_aft produces positive-only SFT chat data; "
                "algo must be 'sft'"
            )
        _merge_defaults(iv, {
            "max_pairs": None,
            "algo": "dpo",
            "wandb_project": "valuegen",
            "train": {},
            "resources": {
                "train": {"time": "05:00:00", "mem": "48G", "gpus": 1, "nproc": 1},
            },
        })

    if method == "label_subset":
        # Two modes, and only the configured one's defaults are materialized:
        # `labels.oracle` (generate the judgments) or `labels.dir` (reuse
        # existing ones). Merging the oracle block into a prelabeled config
        # would move its hash — and so its artifact id — without changing a
        # byte of its behavior.
        if "oracle" in iv.get("labels", {}):
            from valuegen.ground_truth.label_subset import STATEMENT_TEMPLATE

            _merge_defaults(iv, {"labels": {
                "tau": 0.5, "seed": 42, "sources": ["hh", "pku"],
                "oracle": {
                    # Served like the eval judge: an HF name with `serve: local`,
                    # or any OpenAI-API endpoint via `api_base`.
                    "model": None, "serve": None, "api_base": None,
                    "gpus": 2, "mem": "96G", "time": "1-00:00:00", "env": "eval_api",
                    # Judgment-affecting knobs (hashed into the label id):
                    "orderings": 2, "max_chars": 8000, "max_rows": None,
                    "template": STATEMENT_TEMPLATE, "principles": None,
                    # Scheduling knobs (deliberately not hashed):
                    "shards": {"hh": 1, "pku": 4}, "concurrency": 64,
                    "label": {"time": "18:00:00", "mem": "24G", "env": "default"},
                },
            }})
        else:
            _merge_defaults(iv, {"labels": {
                "tau": 0.5, "seed": 42,
                "hh_csv": "hh_test_seed_4ob.csv", "pku_csv": "pku_test_4ob.csv",
            }})
    elif method == "conflictscope_pairs":
        has_pool = isinstance(iv.get("pool"), dict)
        has_train = "train_scenarios" in iv
        if has_pool == has_train:
            raise ValueError(
                "conflictscope_pairs needs exactly one of 'pool' (a scenario-"
                "pool block to build/reuse) or 'train_scenarios' (an existing "
                "scenario dir to train on)"
            )
        if has_pool:
            resolve_pool(iv["pool"], iv["value_set"])
    elif method == "model_spec_aft":
        # Optional value-neutral control row (see interventions.control_value).
        # Validated, never defaulted: an absent block must leave every
        # existing intervention id untouched.
        control = iv.get("control")
        if control is not None:
            if not isinstance(control, dict) or set(control) != {"value", "spec"}:
                raise ValueError(
                    "intervention.control must be {value: <row tag>, spec: <text>}"
                )
            tag = str(control["value"])
            if not re.fullmatch(r"[A-Za-z0-9_]+", tag):
                raise ValueError(
                    f"intervention.control.value {tag!r} must be [A-Za-z0-9_]+ "
                    "(it names dataset dirs, checkpoints and eval CSVs)"
                )
            if tag in _all_values(iv["value_set"]) or tag in iv["values"]:
                raise ValueError(
                    f"intervention.control.value {tag!r} collides with a value "
                    "of the value set; the control is a pseudo-value"
                )
            if not str(control["spec"]).strip():
                raise ValueError("intervention.control.spec must be non-empty")
        _merge_defaults(iv, {"generation": {
            # Generator endpoint: an API model_id, or a served model's name
            # plus its OpenAI-API endpoint (an already-running vLLM server,
            # http://node:8000/v1) — then no Anthropic credits are spent.
            # `api_base` is hashed into the identity, so it is for stable
            # endpoints only; for an ephemeral SLURM-served vLLM leave it
            # null and export VALUEGEN_MSM_API_BASE, which stays out of the
            # hash (the served *model* is identified by model_id).
            "model_id": "claude-sonnet-5", "api_base": None,
            # serve: "local" serves model_id on its own vLLM stage instead
            # (like the label oracle), released the moment generation
            # finishes; the generate array itself is CPU-only.
            "serve": None, "gpus": 2, "mem": "96G", "time": "1-00:00:00",
            "serve_env": "eval_api",
            "generate": {"time": "12:00:00", "mem": "16G"},
            # Dataset-shaping knobs, forwarded to the fork's AFT pipeline.
            "n_samples": 1000, "questions_per_domain": 50,
            "response_style": "value", "prompt_version": "v1",
            "temperature": 1.0, "max_tokens": 2048,
            "disable_thinking": False,
            "use_llm_filter": True, "dedup_threshold": 0.91,
            # Train on the <think>-stripped dataset (the fork writes both).
            "strip_cot": True,
            # Substituted into the fork's prompts; datasets are model-agnostic
            # across the mtag grid, so keep these generic.
            "model_name": "Assistant", "provider_name": "its developer",
            # Scheduling. NOTE: the whole resolved block is hashed into the
            # intervention id, so changing these regenerates the data.
            "max_concurrent": 20, "env": "msm",
        }})
    elif method == "none":
        # Base models only — a pure eval run with no steered rows.
        iv.setdefault("values", [])


def _resolve_evaluation(ev: dict) -> None:
    method = ev.get("method")
    if method not in EVAL_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unknown evaluation.method {method!r}; known: {sorted(EVAL_SCHEMA_VERSIONS)}"
        )
    ev.setdefault("schema_version", EVAL_SCHEMA_VERSIONS[method])
    if method == "conflictscope":
        scenarios = ev.get("scenarios")
        if isinstance(scenarios, dict):
            if set(scenarios) != {"pool"}:
                raise ValueError(
                    "evaluation.scenarios is either a scenario-dir path or "
                    "{pool: <scenario-pool block>}"
                )
            resolve_pool(scenarios["pool"], ev["value_set"])
        elif not scenarios:
            raise ValueError(
                "evaluation.scenarios needs a scenario dir or a {pool: ...} block"
            )
        _merge_defaults(ev, {
            "mode": "interactive", "cache": True, "filter": True,
            "temperature": None, "max_tokens": None, "max_scenarios": None,
            "judge": {
                "model": None, "serve": None, "api_base": None,
                "gpus": 2, "mem": "96G", "time": "2-00:00:00",
                "env": "default",
            },
            # A served assistant only fits eval-time interventions (prompt
            # steering): checkpoint interventions are each their own assistant.
            "assistant": {
                "serve": None, "gpus": 1, "mem": "48G",
                "time": "2-00:00:00", "env": "default",
            },
            "matrices": {"variants": ["likert_normalized"], "cols": "all"},
            "resources": {"time": "04:00:00", "mem": "32G", "gpus": 1},
        })
        # Derived defaults are also made explicit before hashing.
        ev.setdefault("user_model", ev["judge"].get("model"))


def resolve_experiment(cfg: dict) -> dict:
    """Materialize every behavioral default used in config identity.

    An experiment is an ``intervention:`` block (inline) or an
    ``evaluation.intervention:`` reference (an existing intervention artifact
    ID — cross-config sharing by explicit reference), never both,
    plus an ``evaluation:`` block. A referenced intervention must spell out
    ``evaluation.value_set``, since there is no inline block to inherit from.
    """
    cfg = copy.deepcopy(cfg)
    ev = cfg.get("evaluation")
    if not isinstance(ev, dict):
        raise ValueError("experiment config needs an 'evaluation' block")
    iv = cfg.get("intervention")
    ref = ev.get("intervention")
    if iv is not None and ref is not None:
        raise ValueError(
            "config has both an inline 'intervention' block and an "
            "'evaluation.intervention' reference; pick one"
        )
    if iv is None and ref is None:
        raise ValueError(
            "config needs an 'intervention' block (use method: none for a "
            "base-model-only eval) or an 'evaluation.intervention' reference"
        )
    if iv is not None:
        _resolve_intervention(iv)
        ev.setdefault("value_set", iv["value_set"])
    elif "value_set" not in ev:
        raise ValueError(
            "evaluation.value_set is required when the intervention is an "
            "artifact-ID reference"
        )
    _resolve_evaluation(ev)
    _resolve_schedule(cfg)
    return cfg


def _resolve_schedule(cfg: dict) -> None:
    """Validate the top-level ``schedule:`` block — controller pacing only.

    ``schedule.wave: W`` makes ``gt run`` alternate training and evaluation
    in waves of W values per model (train W checkpoints, eval them, repeat)
    instead of training every value before the first eval. Pure scheduling:
    it changes when results land, never what they are, so it is excluded from
    every identity and record (see ``canonical_experiment``).

    ``schedule.delete_exports: true`` frees each wave's checkpoints (merged
    dir + trainer weight shards) once this GT run's evals of them are done
    (``interventions.delete_checkpoint``). It rides on the wave hook, so it
    requires ``wave``.

    ``schedule.prune_trainer_save`` drops a full-FT trainer output's weight
    shards as soon as its export verifies, inside the training task, instead
    of holding them until the wave's deletion (the fp32 save is 2x the
    export). Defaults to ``delete_exports``: a run that frees the checkpoint
    after its evals has no later reader of the trainer save either.

    ``schedule.stop_after_wave: N`` schedules only the first N waves per model
    (``interventions.interleave_waves``): a pilot pass of the real run. The
    base eval rides in wave N instead of the final one. Dropping the key (or
    raising N) and rerunning resumes from wave N+1 — the done-checks and
    tombstones skip what already ran. Requires ``wave``.
    """
    schedule = cfg.get("schedule")
    if schedule is None:
        return
    if not isinstance(schedule, dict):
        raise ValueError("schedule: must be a mapping")
    unknown = set(schedule) - {
        "wave", "delete_exports", "prune_trainer_save", "stop_after_wave",
    }
    if unknown:
        raise ValueError(f"schedule: unknown keys {sorted(unknown)}")
    wave = schedule.get("wave")
    if wave is not None and (not isinstance(wave, int) or wave < 1):
        raise ValueError(f"schedule.wave must be a positive integer, got {wave!r}")
    delete = schedule.get("delete_exports")
    if delete is not None and not isinstance(delete, bool):
        raise ValueError(
            f"schedule.delete_exports must be a boolean, got {delete!r}"
        )
    prune = schedule.get("prune_trainer_save")
    if prune is not None and not isinstance(prune, bool):
        raise ValueError(
            f"schedule.prune_trainer_save must be a boolean, got {prune!r}"
        )
    if delete and wave is None:
        raise ValueError(
            "schedule.delete_exports requires schedule.wave: checkpoints are "
            "freed by the per-wave hook, after that wave's evals"
        )
    stop = schedule.get("stop_after_wave")
    if stop is not None:
        if isinstance(stop, bool) or not isinstance(stop, int) or stop < 1:
            raise ValueError(
                f"schedule.stop_after_wave must be a positive integer, got {stop!r}"
            )
        if wave is None:
            raise ValueError("schedule.stop_after_wave requires schedule.wave")


# Top-level keys that never enter a config identity or record: private
# (``_``-prefixed, e.g. ``_path``) and controller scheduling (``schedule``).
_UNHASHED_TOP_LEVEL = ("schedule",)


def canonical_experiment(cfg: dict) -> dict:
    return {
        key: value
        for key, value in cfg.items()
        if not key.startswith("_") and key not in _UNHASHED_TOP_LEVEL
    }


def config_hash(cfg: dict, length: int = 12) -> str:
    payload = json.dumps(
        canonical_experiment(cfg), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


# labels.oracle keys that tune *how the server runs*, never what it judges.
# Stripped before hashing so flipping them (e.g. a vLLM backend workaround)
# doesn't fork the intervention id; absent keys hash identically, so ids of
# existing configs are unaffected.
_ORACLE_SCHEDULING_KEYS = ("serve_args",)
# Likewise for model_spec_aft's `generation` block: which GPU model hosts the
# generator changes wall time, never the sampled bytes.
_GENERATION_SCHEDULING_KEYS = ("gpu_type",)


def _strip_oracle_scheduling(block: dict) -> dict:
    oracle = block.get("labels", {}).get("oracle")
    generation = block.get("generation")
    strip_oracle = isinstance(oracle, dict) and any(
        k in oracle for k in _ORACLE_SCHEDULING_KEYS
    )
    strip_generation = isinstance(generation, dict) and any(
        k in generation for k in _GENERATION_SCHEDULING_KEYS
    )
    if not (strip_oracle or strip_generation):
        return block
    block = json.loads(json.dumps(block))
    if strip_oracle:
        for k in _ORACLE_SCHEDULING_KEYS:
            block["labels"]["oracle"].pop(k, None)
    if strip_generation:
        for k in _GENERATION_SCHEDULING_KEYS:
            block["generation"].pop(k, None)
    return block


def intervention_id(cfg: dict) -> str:
    """The intervention artifact's identity.

    Hashes the resolved ``intervention:`` block alone — training-affecting
    knobs only, never the evaluation side — so re-evaluating under a new eval
    config reuses the checkpoints. A referenced intervention
    (``evaluation.intervention``) *is* the identity, verbatim.
    """
    ref = cfg["evaluation"].get("intervention")
    if ref:
        return str(ref)
    return f"{cfg['name']}-{config_hash(_strip_oracle_scheduling(cfg['intervention']))}"


def gt_id(cfg: dict) -> str:
    """The GT run's identity: the resolved evaluation block plus the
    intervention artifact ID it scores (never the intervention's internals —
    two configs sharing an intervention by reference or by identical inline
    blocks land on the same checkpoints, but their GT runs stay separate)."""
    evaluation = {
        key: value
        for key, value in cfg["evaluation"].items()
        if key != "intervention"
    }
    payload = {"intervention": intervention_id(cfg), "evaluation": evaluation}
    return f"{cfg['name']}-{config_hash(payload)}"


def pool_identity(pool: dict) -> dict:
    """A scenario pool's identity: the knobs that change its scenarios —
    and nothing else (scheduling knobs deliberately excluded, like labels)."""
    gen = pool["scenario_gen"]
    return {
        "schema_version": pool["schema_version"],
        "value_set": str(pool["value_set"]),
        "scenario_gen": {
            key: gen.get(key)
            for key in ("model", "filter_model", "num_scenarios", "temperature")
        },
        "dedup": pool["dedup"],
        "split": pool["split"],
    }


def pool_id(pool: dict) -> str:
    import re

    short = re.sub(
        r"[^A-Za-z0-9._-]", "_", str(pool["scenario_gen"]["model"]).split("/")[-1]
    )
    return f"{short}-{config_hash(pool_identity(pool))}"


def config_record_matches(path: str | Path, config_id: str, resolved: dict) -> bool:
    """Whether a persisted resolved-config record carries exactly this identity."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        record = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return False
    return bool(
        isinstance(record, dict)
        and record.get("config_id") == config_id
        and record.get("resolved_config") == resolved
    )


def config_id_matches(path: str | Path, expected: str) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        record = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return False
    return bool(isinstance(record, dict) and record.get("config_id") == expected)


def ensure_config_record(path: str | Path, config_id: str, resolved: dict) -> Path:
    """Persist a resolved-config record, refusing to adopt mismatched outputs."""
    path = Path(path)
    if path.exists():
        if not config_record_matches(path, config_id, resolved):
            raise RuntimeError(f"Config-ID mismatch at {path}; refusing output reuse")
        return path
    if path.parent.exists() and any(path.parent.iterdir()):
        raise RuntimeError(
            f"Missing config identity in non-empty output directory {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "config_id": config_id,
        "resolved_config": resolved,
    }, sort_keys=False))
    return path


def load_experiment(path: str | Path) -> dict:
    """Load and fully resolve an experiment YAML."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "name" not in cfg:
        raise ValueError(f"{path}: experiment config needs a top-level 'name'")
    cfg = resolve_experiment(cfg)
    cfg["_path"] = str(path.resolve())
    return cfg
