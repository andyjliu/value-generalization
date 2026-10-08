"""``mv run``: train -> eval in waves under one polling controller.

    valuegen mv run -c CFG [--wave W] [--delete-exports] [--dry-run]
                           [--suite S …] [--arm ID …] [--seed S …]
    valuegen mv run -c CFG --submit [...]     # sbatch a self-resubmitting controller for the same

The trainable runs (``train.plan_runs`` order: family, arm, seed) are cut
into waves of ``W`` checkpoints (``schedule.wave`` or ``--wave``; unset =
one wave; ``schedule.order: stride`` interleaves them instead of slicing, see
:func:`cut_waves`) and walked strictly in order::

    train_w0 -> eval_w0 -> train_w1 -> eval_w1 -> …

Each wave is two SLURM submissions with their own resource shapes, exactly
as ``mv train`` / ``mv eval`` make them: an ``N``-GPU train array throttled
by ``train.concurrent``, then a packed single-GPU eval array over the
checkpoints that wave verified. The base and every imported arm join the
first eval wave, so delta-vs-base exists as soon as the first checkpoints
are scored. Stages run through :meth:`Orchestrator.run_stage` one at a time
-- the eval stage of a wave is built *after* its training finished, so a run
that exhausted its retries is reported and skipped (retried by the next
``mv run``) rather than fed to an eval task that fails three more times.

Everything is filesystem-checked and idempotent: re-running derives the
same waves and skips every done task; a changed ``W`` re-cuts the waves but
changes no task. Job names are ``multivalue/{exp_id}_train_w{k}`` /
``_eval_w{k}``, so a restarted controller adopts a live array instead of
stacking a second one on the same checkpoints.

``--delete-exports`` (or ``schedule.delete_exports``) removes a checkpoint's
``export/`` after its eval wave, but only once *every* enabled suite scored
it (``train.delete_export``; a tombstone in ``run_record.json`` keeps the
run done). Peak export storage is then about ``W`` checkpoints instead of
all of them.

``--submit`` renders ``run_driver.sbatch`` -- a poll-only controller job on
``cluster.controller_partition`` (else ``cpu_partition``), walltime =
that partition's cap -- and submits it. The driver submits its own successor
with ``afterany`` *before* doing any work and cancels it on a clean exit, so
a controller killed at the walltime cap is simply replaced: the successor
adopts live arrays and skips done tasks. A successor that finds nothing
pending exits 0 and cancels *its* successor, so the chain always terminates.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from valuegen.config import ClusterConfig
from valuegen.multivalue import eval_stage as ES
from valuegen.multivalue import train as T
from valuegen.multivalue.config import WAVE_ORDERS, MultivalueConfig
from valuegen.multivalue.evals import runner as R
from valuegen.multivalue.evals.base import Candidate, EvalError
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.sets import Arm
from valuegen.slurm import Orchestrator, Stage, sbatch_available

DRIVER_STAGE = "run"
DRIVER_SCRIPT = "run_driver.sbatch"


class WaveError(RuntimeError):
    pass


# ── planning ─────────────────────────────────────────────────────────────────


def cut_waves(runs: Sequence[T.Run], wave: int | None, order: str = "contiguous") -> list[list[T.Run]]:
    """Waves of at most ``wave`` runs; one wave when ``wave`` is unset. No
    runs -> one empty wave (base + imports still get their eval).
    ``contiguous``: slices in ``plan_runs`` order. ``stride``: wave ``i`` takes
    every ``n_waves``-th run starting at ``i`` -- arm ids of a stratified family
    ascend in the sampling metric, so each wave then spans its whole range
    (interim results cover every bin; a run that dies late costs no one bin)."""
    runs = list(runs)
    if wave is not None and wave < 1:
        raise WaveError(f"wave must be a positive integer, got {wave!r}")
    if order not in WAVE_ORDERS:
        raise WaveError(f"unknown wave order {order!r}; known: {WAVE_ORDERS}")
    if not runs:
        return [[]]
    w = wave or len(runs)
    if order == "stride":
        n_waves = -(-len(runs) // w)
        return [runs[i::n_waves] for i in range(n_waves)]
    return [runs[i:i + w] for i in range(0, len(runs), w)]


def resolve_schedule(cfg: MultivalueConfig, wave: int | None, delete_exports: bool | None) -> tuple[int | None, bool]:
    """CLI overrides on top of ``schedule:``."""
    w = wave if wave is not None else cfg.schedule.wave
    if w is not None and w < 1:
        raise WaveError(f"--wave must be a positive integer, got {w!r}")
    d = cfg.schedule.delete_exports if delete_exports is None else bool(delete_exports)
    return w, d


def wave_candidates(cands: Sequence[Candidate], trained: Sequence[T.Run], *, first: bool) -> list[Candidate]:
    """The candidates one eval wave serves: this wave's verified checkpoints,
    plus the base and every servable imported arm in the first wave."""
    ids = {r.ckpt_id for r in trained}
    out = []
    for c in cands:
        if c.export_deleted:
            continue
        if c.kind == "trained" and c.ckpt_id in ids:
            out.append(c)
        elif first and (c.kind == "base" or (c.kind == "imported" and c.export_verified)):
            out.append(c)
    return out


# ── the loop ─────────────────────────────────────────────────────────────────


def run_waves(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm], *,
              wave: int | None = None, delete_exports: bool | None = None, suites: Sequence[str] | None = None,
              only: Sequence[str] | None = None, seeds: Sequence[int] | None = None, dry_run: bool = False,
              log=print, orchestrator=Orchestrator) -> int:
    """The ``mv run`` entry point. Returns 0 when every wave completed (all
    runs trained, every scheduled eval done), 1 otherwise."""
    wave, delete = resolve_schedule(cfg, wave, delete_exports)
    names = ES.select_suites(cfg, suites)
    inputs = {s: R.ensure_inputs(cfg, layout, s, log=log) for s in names}
    for s in names:
        var, present = R.grader_key_present(cfg.evals.grader(s), cluster)
        if not present:
            msg = f"grader {cfg.evals.grader(s)} for {s}: {var} is neither in the environment nor in {cluster.repo / '.env'}"
            if not dry_run:
                raise EvalError(msg)
            log(f"WARNING: {msg}")
    manifest = T.load_manifest(layout)
    runs = T.plan_runs(cfg, layout, arms, manifest, only=only, seeds=seeds)
    if runs:
        T.write_runs_record(layout, runs, cluster)
    waves = cut_waves(runs, wave, cfg.schedule.order)
    res_t, res_e = cfg.train.resources, cfg.evals.resources
    log(f"run: {len(runs)} runs in {len(waves)} wave(s) of {wave or len(runs) or 0} ({cfg.schedule.order}), suites {names}, "
        f"train {int(res_t['gpus'])} GPUs x {res_t['time']} ({cfg.train.concurrent} concurrent), "
        f"eval {int(res_e['gpus'])} GPU x {res_e['time']}, delete_exports={delete}")
    orch = orchestrator(name=T.orchestrator_name(layout), stages=[], cluster=cluster, dry_run=dry_run)
    ok = True
    failed_runs: list[str] = []
    failed_evals: list[str] = []
    deleted: list[str] = []
    scheduled: list[Candidate] = []
    all_cands = ES.plan_candidates(cfg, cluster, layout, arms)
    for k, wave_runs in enumerate(waves):
        tag = f"w{k}"
        # ── train ──
        if wave_runs:
            stage = T.build_stage(cfg, cluster, layout, wave_runs, name=f"train_{tag}")
            pending = stage.pending()
            log(f"\n[{tag}] train: {len(wave_runs)} runs, {len(pending)} pending")
            for r in wave_runs:
                state = T.run_state(layout, cluster, r)
                if state not in ("done", "pruned"):
                    log(f"  - {r.ckpt_id}: {state}")
            if pending:
                if not dry_run and not sbatch_available():
                    log("sbatch not found; run from the SLURM login node or use --dry-run")
                    return 1
                complete, _ = orch.run_stage(stage)
                if not complete:
                    ok = False
        if dry_run:
            trained = list(wave_runs)  # render every eval as if training succeeded
        else:
            trained = [r for r in wave_runs if T.run_done(layout, cluster, r)]
        failed = [r.ckpt_id for r in wave_runs if r not in trained]
        if failed:
            failed_runs += failed
            log(f"[{tag}] {len(failed)} runs not trained; their evals are skipped this pass "
                f"(the next `mv run` retries them): {failed}")
        # ── eval ──
        if not dry_run:
            all_cands = ES.plan_candidates(cfg, cluster, layout, arms)  # exports verified by this wave
        cands = wave_candidates(all_cands, trained, first=(k == 0))
        for c in cands:
            for s in names:
                if c.kind == "imported" and R.suite_state(cfg, layout, c, s, inputs[s]) == "reusable":
                    R.link_external_suite(layout, c, s)
                    log(f"  {c.ckpt_id}/{s}: reusing the imported run at {c.external_suites[s]}")
        scheduled += [c for c in cands if c not in scheduled]
        ES.write_candidates_record(layout, all_cands, scheduled, names)
        if cands:
            stage = ES.build_stage(cfg, cluster, layout, cands, names, inputs, name=f"eval_{tag}")
            pending = stage.pending()
            log(f"[{tag}] eval: {len(cands)} candidates x {names}, {len(pending)} pending")
            for c in cands:
                states = ES.candidate_states(cfg, layout, c, names, inputs)
                if any(st != "done" for st in states.values()):
                    log(f"  - {c.ckpt_id} [{c.kind}]: " + ", ".join(f"{s}={st}" for s, st in states.items()))
            if pending:
                if not dry_run and not sbatch_available():
                    log("sbatch not found; run from the SLURM login node or use --dry-run")
                    return 1
                complete, _ = orch.run_stage(stage)
                if not complete:
                    ok = False
                    failed_evals += [t.key.split("/", 1)[1] for t in stage.pending()]
        else:
            log(f"[{tag}] eval: nothing to serve")
        # ── delete ──
        if delete and not dry_run:
            for r in trained:
                done, why = T.delete_export(cfg, cluster, layout, r)
                log(f"  {'deleted' if done else 'kept'}: {why}")
                if done:
                    deleted.append(r.ckpt_id)

    if dry_run:
        render_driver(cfg, cluster, layout, wave=wave, delete_exports=delete, suites=suites, only=only, seeds=seeds,
                      orchestrator=orchestrator)
        log(f"\ndry run: rendered {len(waves)} wave(s) under {layout.slurm_jobs_dir} + {DRIVER_SCRIPT}")
        return 0
    remaining = [r.ckpt_id for r in runs if not T.run_done(layout, cluster, r)]
    if failed_runs or failed_evals:
        ok = False
    log(f"\nrun: {len(waves)} wave(s); {len(runs) - len(remaining)}/{len(runs)} runs trained, "
        f"{len(deleted)} exports deleted, {len(failed_runs)} runs failed {failed_runs}, "
        f"{len(failed_evals)} evals failed {failed_evals}")
    if ok:
        log("all waves complete")
    return 0 if ok else 1


# ── the self-resubmitting driver ─────────────────────────────────────────────


def _driver_stage(cfg: MultivalueConfig, cluster: ClusterConfig) -> Stage:
    partition = cluster.controller_partition or cluster.cpu_partition
    cap = cluster.partition_max_time.get(partition) if partition else None
    return Stage(name=DRIVER_STAGE, tasks=[], time=str(cap or cluster.default_time), mem=str(cluster.default_mem),
                 gpus=0, env=cfg.train.env, controller=True)


def run_command(cfg: MultivalueConfig, cluster: ClusterConfig, *, wave: int | None, delete_exports: bool,
                suites: Sequence[str] | None, only: Sequence[str] | None, seeds: Sequence[int] | None) -> str:
    cmd = f"python -m valuegen.multivalue.cli run -c {T.q(cfg.path)}"
    if cluster.source_path is not None:
        cmd += f" --cluster {T.q(cluster.source_path)}"
    if wave is not None:
        cmd += f" --wave {int(wave)}"
    if delete_exports:
        cmd += " --delete-exports"
    for s in suites or ():
        cmd += f" --suite {T.q(s)}"
    for a in only or ():
        cmd += f" --arm {T.q(a)}"
    for s in seeds or ():
        cmd += f" --seed {int(s)}"
    return cmd


def render_driver(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, *, wave: int | None,
                  delete_exports: bool, suites: Sequence[str] | None = None, only: Sequence[str] | None = None,
                  seeds: Sequence[int] | None = None, orchestrator=Orchestrator) -> Path:
    """Write ``slurm_jobs/multivalue/{exp_id}/run_driver.sbatch``."""
    orch = orchestrator(name=T.orchestrator_name(layout), stages=[], cluster=cluster, dry_run=True)
    stage = _driver_stage(cfg, cluster)
    script = orch.script_dir / DRIVER_SCRIPT
    body = (
        orch._directives(stage, 1, 0) + "\n" + orch._preamble(stage)
        + "\n# Successor first: this controller only polls (the work lives in the arrays it\n"
        "# submits), so a kill at the walltime cap loses nothing -- the successor adopts\n"
        "# live arrays by job name and skips done tasks. A clean exit cancels it.\n"
        f"NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID {T.q(script)}) || NEXT=\"\"\n"
        'echo "successor: ${NEXT:-none}"\n'
        "set +e\n"
        f"{run_command(cfg, cluster, wave=wave, delete_exports=delete_exports, suites=suites, only=only, seeds=seeds)}\n"
        "rc=$?\n"
        "set -e\n"
        'if [ "$rc" -eq 0 ] && [ -n "$NEXT" ]; then scancel "$NEXT"; fi\n'
        "exit $rc\n"
    )
    orch.script_dir.mkdir(parents=True, exist_ok=True)
    orch.log_dir.mkdir(parents=True, exist_ok=True)
    script.write_text(body)
    script.chmod(0o755)
    return script


def submit_driver(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, *, wave: int | None = None,
                  delete_exports: bool | None = None, suites: Sequence[str] | None = None,
                  only: Sequence[str] | None = None, seeds: Sequence[int] | None = None, dry_run: bool = False,
                  log=print, orchestrator=Orchestrator) -> int:
    """``mv run --submit``: render the driver and sbatch it, unless a live
    chain for this experiment already exists."""
    wave, delete = resolve_schedule(cfg, wave, delete_exports)
    ES.select_suites(cfg, suites)
    script = render_driver(cfg, cluster, layout, wave=wave, delete_exports=delete, suites=suites, only=only,
                           seeds=seeds, orchestrator=orchestrator)
    log(f"driver: {script}")
    if dry_run:
        log("dry run: not submitted")
        return 0
    if not sbatch_available():
        log("sbatch not found; run from the SLURM login node or use --dry-run")
        return 1
    orch = orchestrator(name=T.orchestrator_name(layout), stages=[], cluster=cluster)
    live = orch._find_active_job(_driver_stage(cfg, cluster))
    if live is not None:
        log(f"a controller chain for {layout.exp_id} is already live (job {live}); not submitting a second one")
        return 1
    result = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        log(f"sbatch failed: {result.stderr.strip()}")
        return 1
    log(f"submitted controller job {result.stdout.strip().split(';')[0]} ({script.name}); "
        f"logs under {orch.log_dir}/{DRIVER_STAGE}_<jobid>.out")
    return 0
