"""``mv eval``: one packed single-GPU SLURM task per served checkpoint.

    valuegen mv eval -c CFG --dry-run                  # render slurm_jobs/multivalue/{exp_id}/eval.sbatch
    valuegen mv eval -c CFG [--no-wait] [--suite S …] [--arm ID …]
    valuegen mv eval -c CFG --no-wait --after afterany:JOBID   # hang off a running train array
    valuegen mv eval -c CFG --smoke [--arm ID]         # base model, 1 repeat, a few units -> evals_smoke/
    valuegen mv eval -c CFG --build SUITE              # (re)freeze a suite's inputs

Candidates (:class:`Candidate`) are the untrained base, every run of the
train stage whose export is verified (``run_record.json`` with this
experiment's identity), and every imported arm with an export. A task
serves one candidate with vLLM in its own allocation
(``evaluation.serve_local_sh``) and runs one worker per enabled suite
(``python -m valuegen.multivalue.eval_worker run``); the task is done when
every suite's ``COMPLETE.json`` satisfies the current protocol
(:mod:`valuegen.multivalue.evals.runner`). On exclusive-node clusters the
orchestrator packs ``gpus_per_node`` tasks per node.

The base is served from a staged export under
``{finetune_root}/multivalue/{exp_id}/base/export`` (weights symlinked into
the HF cache, pinned chat template + sidecar applied, verified); the task
stages it in-job when missing. When an imported experiment registered a
verified base copy (``imports.json: base_model_dirs``) at the same repo and
revision, that copy is served instead. Imported arms are served from their
own export; a completed suite run of the source experiment is reused in
place (symlinked) when its recorded protocol matches this one exactly.

Trained runs without a verified export are listed, not scheduled, unless
``--all`` (or ``--after``, where the export appears while the job waits).
A run whose export was deleted after its evals completed (``mv run
--delete-exports``, tombstone in ``run_record.json``) keeps its finished
suites and is never scheduled: a new suite needs a retrain.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from valuegen.config import ClusterConfig
from valuegen.ground_truth import gcs
from valuegen.ground_truth.evaluation import serve_local_sh
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import train as mvtrain
from valuegen.multivalue.config import BASE_ARM, MultivalueConfig
from valuegen.multivalue.evals import get_suite
from valuegen.multivalue.evals.base import Candidate, EvalError
from valuegen.multivalue.evals import runner as R
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.sets import Arm
from valuegen.slurm import Orchestrator, Stage, Task, sbatch_available

q = mvtrain.q
WORKER = "python -m valuegen.multivalue.eval_worker"
CANDIDATE_DEFAULTS = {"max_model_len": 16384, "max_num_seqs": 64, "gpu_memory_utilization": 0.9, "server_seed": 10000}


# ── the base model ───────────────────────────────────────────────────────────


def _base_key(cfg: MultivalueConfig) -> str:
    return f"{cfg.train.base_model}@{cfg.train.revision}"


def resolve_base_model_dir(cfg: MultivalueConfig, layout: Layout, imports_record: Mapping) -> tuple[Path, str]:
    """``(dir, source)``: a verified base copy registered by an import at the
    pinned repo+revision, else the export this package stages."""
    if cfg.train.revision:
        d = (imports_record.get("base_model_dirs") or {}).get(_base_key(cfg))
        if d and (Path(d) / "export_verification.json").is_file() and (Path(d) / "config.json").is_file():
            return Path(d), "imported"
    return layout.base_export_dir, "staged"


def stage_base_export(cfg: MultivalueConfig, layout: Layout, log=print) -> dict:
    """Stage the untrained base as a servable export (idempotent).

    Weights are symlinked (HF cache blobs or the local dir), everything else
    copied; the pinned chat format is applied (template, eos/pad, sidecar)
    and the result verified like a trained export. A failed verification
    leaves ``export.partial`` behind and raises.
    """
    from valuegen.ground_truth.chat_formats import get_chat_format
    from valuegen.ground_truth.training import align_chat_format, verify_export

    dst = layout.base_export_dir
    if (dst / "export_verification.json").is_file():
        return mvdata.read_json(dst / "export_verification.json")
    fmt = get_chat_format(cfg.train.chat_format)
    src = Path(cfg.train.base_model)
    if src.is_dir():
        src, origin = src.resolve(), "local"
    else:
        from huggingface_hub import snapshot_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        log(f"fetching {cfg.train.base_model}@{cfg.train.revision or 'main'} into the HF cache ...")
        src = Path(snapshot_download(cfg.train.base_model, revision=cfg.train.revision, token=token,
                                     allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"]))
        origin = "hf"
    staging = dst.with_name(dst.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        if p.suffix == ".safetensors":
            (staging / p.name).symlink_to(p.resolve())
        elif p.is_file():
            shutil.copy2(p.resolve(), staging / p.name)
    align_chat_format(staging, fmt)
    mvdata.write_json(staging / "run_metadata.json", {
        "kind": "base", "base_repo": cfg.train.base_model, "base_revision": cfg.train.revision,
        "chat_format": fmt.name, "origin": origin, "source_dir": str(src),
        "note": "staged serving copy of the untrained base; weights are symlinks", "staged_at": mvdata.now(),
    })
    result = verify_export(staging, chat_format=fmt)
    mvdata.write_json(staging / "export_verification.json", result)
    if result.get("problems"):
        raise EvalError(f"staged base at {staging} failed verification: {result['problems']}")
    os.replace(staging, dst)
    log(f"base staged at {dst}")
    return result


# ── candidates ───────────────────────────────────────────────────────────────


def _run_export(layout: Layout, cluster: ClusterConfig, arm: Arm, seed: int) -> tuple[bool, bool]:
    """``(servable, deleted)`` for a trained run: servable when its record
    is verified under this identity and the export is on disk or in GCS;
    deleted when the record carries the ``export_deleted`` tombstone."""
    root = layout.checkpoint_dir(arm.id, seed)
    rec = mvdata.read_json(root / "run_record.json") if (root / "run_record.json").is_file() else None
    if not rec or rec.get("export_verified") is not True:
        return False, False
    ident = rec.get("identity") or {}
    if (ident.get("exp_id"), ident.get("arm_id"), ident.get("seed")) != (layout.exp_id, arm.id, seed):
        return False, False
    export = root / "export"
    servable = (export / "config.json").is_file() or (cluster.gcs is not None and gcs.ready(export))
    deleted = isinstance(rec.get("export_deleted"), dict)
    return servable and not deleted, deleted


def _run_verified(layout: Layout, cluster: ClusterConfig, arm: Arm, seed: int) -> bool:
    return _run_export(layout, cluster, arm, seed)[0]


def plan_candidates(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm], *,
                    only: Sequence[str] | None = None) -> list[Candidate]:
    """Every checkpoint the eval stage can serve, base first, then trained
    runs (arm x seed) and imported arms."""
    if only:
        unknown = sorted(set(only) - {a.id for a in arms})
        if unknown:
            raise EvalError(f"unknown arm ids: {unknown}")
    imports_rec = mvimports.load_imports_record(layout)
    out: list[Candidate] = []
    for a in sorted(arms, key=lambda a: (a.kind != "base", a.kind == "imported", a.family, a.id)):
        if only and a.id not in set(only):
            continue
        if a.kind == "base":
            model_dir, source = resolve_base_model_dir(cfg, layout, imports_rec)
            external = {}
            if source == "imported":  # the import's own base evals travel with its base copy
                external = dict((imports_rec.get("base_evals") or {}).get(_base_key(cfg), {}))
            out.append(Candidate(ckpt_id=BASE_ARM, arm=a, seed=None, kind="base", model_dir=model_dir,
                                 export_verified=(model_dir / "export_verification.json").is_file(),
                                 model_source=source, external_suites=external))
        elif a.kind == "imported":
            e = a.extra
            seed = e.get("seed")
            model_dir = Path(e["model_dir"]) if e.get("model_dir") else None
            out.append(Candidate(
                ckpt_id=layout.ckpt_id(a.id, int(seed)) if seed is not None else a.id, arm=a,
                seed=None if seed is None else int(seed), kind="imported", model_dir=model_dir,
                export_verified=bool(e.get("export_verified")) and model_dir is not None
                and (model_dir / "config.json").is_file(),
                model_source="import", external_suites=dict(e.get("evals") or {})))
        else:
            for seed in cfg.train.seeds:
                export = layout.checkpoint_dir(a.id, int(seed)) / "export"
                servable, deleted = _run_export(layout, cluster, a, int(seed))
                out.append(Candidate(
                    ckpt_id=layout.ckpt_id(a.id, int(seed)), arm=a, seed=int(seed), kind="trained",
                    model_dir=export, export_verified=servable, model_source="export",
                    gcs_uri=gcs.remote_uri(cluster.gcs, export) if cluster.gcs is not None else None,
                    export_deleted=deleted))
    return out


def select_suites(cfg: MultivalueConfig, requested: Sequence[str] | None) -> list[str]:
    """The suites to run: ``--suite`` names (must be enabled) or every
    enabled suite."""
    enabled = cfg.evals.enabled()
    names = list(dict.fromkeys(requested)) if requested else enabled
    for n in names:
        get_suite(n)
    for n in names:
        if n not in enabled:
            raise EvalError(f"suite {n!r} is disabled in evals.suites")
    if not names:
        raise EvalError("no enabled suites in evals.suites")
    return names


# ── commands / stage ─────────────────────────────────────────────────────────


def _worker_common(cfg: MultivalueConfig, cluster: ClusterConfig) -> str:
    s = f"-c {q(cfg.path)}"
    if cluster.source_path is not None:
        s += f" --cluster {q(cluster.source_path)}"
    return s


def eval_command(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, cand: Candidate,
                 suites: Sequence[str], *, smoke: bool = False) -> str:
    """Shell for one task: (stage the base) -> (stage in from GCS) -> serve
    -> one worker per suite."""
    if cand.model_dir is None:
        raise EvalError(f"{cand.ckpt_id}: no model dir to serve")
    work = layout.candidate_dir(cand.ckpt_id, smoke=smoke) / "serve"
    common = _worker_common(cfg, cluster)
    lines = [f"mkdir -p {q(work)}"]
    if cand.kind == "base" and cand.model_source == "staged":
        lines.append(f"if [ ! -f {q(cand.model_dir / 'export_verification.json')} ]; then\n"
                     f"    {WORKER} stage-base {common}\nfi")
    model_dir = cand.model_dir
    if cluster.gcs is not None and cand.kind == "trained":
        # the export may already be pruned to GCS when the task runs
        link = work / "model"
        lines.append(gcs.stage_in_sh(cluster.gcs, cand.model_dir))
        lines.append(f'ln -sfn "$MODEL_DIR" {q(link)}')
        model_dir = link
    c = {**CANDIDATE_DEFAULTS, **cfg.evals.candidate}
    lines.append(serve_local_sh(
        model_dir=model_dir, served_name=cand.served_name, hostfile=work / "cand_host.txt",
        log_file=work / "vllm_serve.log", chat_format=cfg.train.chat_format,
        max_model_len=int(c["max_model_len"]), max_num_seqs=int(c["max_num_seqs"]),
        gpu_memory_utilization=float(c["gpu_memory_utilization"]), seed=int(c["server_seed"]),
    ))
    for s in suites:
        lines.append(f'{WORKER} run {common} --ckpt {q(cand.ckpt_id)} --suite {s} --base-url "$CAND_BASE_URL"'
                     + (" --smoke" if smoke else ""))
    return "\n".join(lines)


def needs_local_judge(cfg: MultivalueConfig, suites: Sequence[str]) -> str | None:
    """The grader that has to be served in-job, or None.

    A suite whose grader names no known API provider (e.g. the prefill judge
    ``Qwen/Qwen3.6-27B``) cannot be reached over the network: Inspect has no
    provider for it. Such a grader is served locally by the eval element and
    routed through ``PREFILL_JUDGE_URL`` (``evals.runner.grader_model``).
    """
    for s in suites:
        name = cfg.evals.grader(s)
        if R.grader_env_var(name) is None:
            return name
    return None


def judge_node_setup(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, grader: str,
                     *, pack: int, smoke: bool = False) -> str:
    """Shell serving ``grader`` once per eval element, on the GPUs that the
    element's packed candidates (devices ``0..pack-1``) leave free.

    Exports ``PREFILL_JUDGE_URL`` / ``PREFILL_JUDGE_MODEL`` /
    ``PREFILL_JUDGE_NO_THINK``, which the per-task scripts inherit through GNU
    parallel, and reaps the server on exit. Mirrors the parameters the frozen
    prefill runs were graded under, so new arms are judged exactly like the
    imported ones.
    """
    tp = int(cfg.evals.suites["prefill"].extra.get("judge_gpus", 1)) if "prefill" in cfg.evals.suites else 1
    devs = ",".join(str(pack + i) for i in range(tp))
    short = grader.rsplit("/", 1)[-1]
    log = (layout.smoke_evals_dir if smoke else layout.evals_dir) / "judge_vllm_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.log"
    serve = (f"vllm serve {grader} --served-model-name {grader} --dtype bfloat16 --api-key api "
             f"--host 127.0.0.1 --port ${{JUDGE_PORT}} --tensor-parallel-size {tp} "
             f"--gpu-memory-utilization 0.9 --max-model-len 8192 --max-num-seqs 64")
    return "\n".join([
        f"mkdir -p {q((layout.smoke_evals_dir if smoke else layout.evals_dir))}",
        "JUDGE_PORT=$(shuf -i 9700-9899 -n 1)",
        f'JUDGE_LOG="{log}"',
        f'echo "[$(date)] serving {grader} judge on gpu(s) {devs} port ${{JUDGE_PORT}} -> $JUDGE_LOG"',
        f'HIP_VISIBLE_DEVICES={devs} CUDA_VISIBLE_DEVICES={devs} setsid bash -c "{serve}" > "$JUDGE_LOG" 2>&1 &',
        "JUDGE_PID=$!",
        "trap 'echo \"[$(date)] stopping judge\"; kill -- -\"$JUDGE_PID\" 2>/dev/null "
        "|| kill \"$JUDGE_PID\" 2>/dev/null || true' EXIT",
        'export PREFILL_JUDGE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"',
        f"export PREFILL_JUDGE_MODEL={grader}",
        "export PREFILL_JUDGE_NO_THINK=1",
        'echo "[$(date)] waiting for judge at ${PREFILL_JUDGE_URL} ..."',
        "judge_deadline=$((SECONDS + 3600))",
        f'until curl -sf -m 10 -H "Authorization: Bearer api" "${{PREFILL_JUDGE_URL}}/models" 2>/dev/null '
        f'| grep -q "{short}"; do',
        '    kill -0 "$JUDGE_PID" 2>/dev/null || { echo "ERROR: judge vllm died; see $JUDGE_LOG"; exit 1; }',
        '    (( SECONDS > judge_deadline )) && { echo "ERROR: judge never became ready"; exit 1; }',
        "    sleep 15",
        "done",
        'echo "[$(date)] judge READY"',
    ])


def candidate_states(cfg: MultivalueConfig, layout: Layout, cand: Candidate, suites: Sequence[str],
                     inputs: Mapping[str, dict], *, smoke: bool = False) -> dict[str, str]:
    return {s: R.suite_state(cfg, layout, cand, s, inputs[s], smoke=smoke) for s in suites}


def candidate_done(cfg: MultivalueConfig, layout: Layout, cand: Candidate, suites: Sequence[str],
                   inputs: Mapping[str, dict], *, smoke: bool = False) -> bool:
    return all(st == "done" for st in candidate_states(cfg, layout, cand, suites, inputs, smoke=smoke).values())


def build_stage(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, cands: Sequence[Candidate],
                suites: Sequence[str], inputs: Mapping[str, dict], *, smoke: bool = False,
                name: str | None = None) -> Stage:
    res = cfg.evals.resources
    tasks = [Task(key=f"eval/{c.ckpt_id}", command=eval_command(cfg, cluster, layout, c, suites, smoke=smoke),
                  done=(lambda c=c: candidate_done(cfg, layout, c, suites, inputs, smoke=smoke))) for c in cands]
    # A grader with no API provider is served in-job: give up one GPU per node
    # for it and pack that many fewer candidates. `pack` (not `gpu_headroom`)
    # is the right knob -- the judge sits *inside* the element's allocation, so
    # the throttle still counts one node per element.
    pack, node_setup = 0, ""
    grader = needs_local_judge(cfg, suites)
    if grader is not None:
        gpn = int(cluster.gpus_per_node or 0)
        tp = int(cfg.evals.suites["prefill"].extra.get("judge_gpus", 1)) if "prefill" in cfg.evals.suites else 1
        if gpn - tp < 1:
            raise EvalError(
                f"grader {grader!r} has no API provider and must be served in-job, but the cluster's "
                f"gpus_per_node={gpn or 'unset'} leaves no room for a {tp}-GPU judge beside a candidate. "
                "Use an API grader for this suite, or serve the judge out of band."
            )
        pack = gpn - tp
        node_setup = judge_node_setup(cfg, cluster, layout, grader, pack=pack, smoke=smoke)
    return Stage(name=name or ("eval_smoke" if smoke else "eval"), tasks=tasks, time=str(res["time"]),
                 mem=str(res.get("mem") or cluster.default_mem), gpus=int(res["gpus"]), env=cfg.evals.env,
                 gpu_headroom=0, pack=pack, node_setup=node_setup,
                 extra_exports={"OPENAI_API_KEY": "${OPENAI_API_KEY:-dummy}"})


def write_candidates_record(layout: Layout, cands: Sequence[Candidate], scheduled: Sequence[Candidate],
                            suites: Sequence[str], *, smoke: bool = False) -> Path:
    path = (layout.smoke_evals_dir / "candidates.json") if smoke else layout.candidates_record
    sched = {c.ckpt_id for c in scheduled}
    rec = {"schema_version": 1, "exp_id": layout.exp_id, "suites": list(suites), "smoke": smoke,
           "candidates": {c.ckpt_id: {**c.to_json(), "scheduled": c.ckpt_id in sched} for c in cands},
           "written_at": mvdata.now()}
    mvdata.write_json(path, rec)
    return path


# ── entry points ─────────────────────────────────────────────────────────────


def evaluate(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm], *,
         suites: Sequence[str] | None = None, dry_run: bool = False, no_wait: bool = False,
         only: Sequence[str] | None = None, after: str | None = None, include_unverified: bool = False,
         smoke: bool = False, log=print) -> int:
    """The ``mv eval`` entry point. Returns an exit code."""
    names = select_suites(cfg, suites)
    if after and not no_wait:
        raise EvalError("--after requires --no-wait (it seeds the dependency-chained submit)")
    inputs = {s: R.ensure_inputs(cfg, layout, s, log=log) for s in names}
    if smoke and not only:
        only = [BASE_ARM]
    cands = plan_candidates(cfg, cluster, layout, arms, only=only)
    if not cands:
        raise EvalError("no candidates: the design has no base arm and nothing was trained or imported")
    include_unverified = include_unverified or bool(after)
    deleted = [c for c in cands if c.export_deleted]
    ready = [c for c in cands if c not in deleted and (c.kind == "base" or c.export_verified
                                                        or (include_unverified and c.kind == "trained"))]
    skipped = [c for c in cands if c not in ready and c not in deleted]
    for c in ready:
        for s in names:
            if c.kind == "imported" and R.suite_state(cfg, layout, c, s, inputs[s], smoke=smoke) == "reusable":
                R.link_external_suite(layout, c, s)
                log(f"  {c.ckpt_id}/{s}: reusing the imported run at {c.external_suites[s]}")
    if skipped:
        log(f"eval: {len(skipped)} candidates without a verified export are not scheduled: "
            f"{[c.ckpt_id for c in skipped]}" + ("" if include_unverified else " (--all schedules them anyway)"))
    if deleted:
        incomplete = [c.ckpt_id for c in deleted if not candidate_done(cfg, layout, c, names, inputs, smoke=smoke)]
        log(f"eval: {len(deleted)} candidates have their exports deleted (mv run --delete-exports) and keep their "
            f"finished suites; {len(incomplete)} of them are incomplete for {names} -- a new suite needs a retrain "
            f"(rm the run dir, then `mv train --arm ID`): {incomplete}")
    for s in names:
        var, present = R.grader_key_present(cfg.evals.grader(s), cluster)
        if not present:
            msg = f"grader {cfg.evals.grader(s)} for {s}: {var} is neither in the environment nor in {cluster.repo / '.env'}"
            if not dry_run:
                raise EvalError(msg)
            log(f"WARNING: {msg}")
    write_candidates_record(layout, cands, ready, names, smoke=smoke)
    stage = build_stage(cfg, cluster, layout, ready, names, inputs, smoke=smoke)
    pending = stage.pending()
    res = cfg.evals.resources
    log(f"{'smoke ' if smoke else ''}eval: {len(ready)} candidates x {names}, {len(pending)} pending, "
        f"{int(res['gpus'])} GPU x {res['time']} each")
    for c in ready:
        states = candidate_states(cfg, layout, c, names, inputs, smoke=smoke)
        if any(st != "done" for st in states.values()):
            log(f"  - {c.ckpt_id} [{c.kind}]: " + ", ".join(f"{s}={st}" for s, st in states.items()))
    if not pending:
        log("all evals done")
        return 0
    orch = Orchestrator(name=mvtrain.orchestrator_name(layout), stages=[stage], cluster=cluster, dry_run=dry_run)
    if dry_run:
        orch.run()
        log(f"dry run: rendered {layout.slurm_jobs_dir / (stage.name + '.sbatch')}")
        return 0
    if not sbatch_available():
        log("sbatch not found; run from the SLURM login node or use --dry-run")
        return 1
    if no_wait:
        orch.submit(seed_dependency=after)
        return 0
    return 0 if orch.run() else 1


def build_inputs_main(cfg: MultivalueConfig, layout: Layout, suite: str, log=print) -> int:
    get_suite(suite)
    if suite not in cfg.evals.suites:
        raise EvalError(f"suite {suite!r} is not in evals.suites")
    rec = R.ensure_inputs(cfg, layout, suite, rebuild=True, log=log)
    keys = {k: v for k, v in rec.items() if k in ("n_panel", "n_conditions", "n_eligible", "n_samples", "n_pool",
                                                   "n_scenarios", "n_dropped", "inputs_sha256", "params")}
    log(f"{suite}: {keys}")
    return 0


def find_candidate(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, ckpt_id: str) -> Candidate:
    arms = mvimports.all_arms(layout)
    for c in plan_candidates(cfg, cluster, layout, arms):
        if c.ckpt_id == ckpt_id:
            return c
    raise EvalError(f"{ckpt_id}: not a candidate of this experiment")
