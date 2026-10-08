"""``valuegen`` console entry point.

Ground-truth surface::

    valuegen gt run       -c configs/experiments/ground_truth/X.yaml [--dry-run] [--no-wait] [--submit]
                          [--delete-exports | --no-delete-exports]
    valuegen gt intervene -c configs/experiments/ground_truth/X.yaml [--dry-run] [--no-wait]
    valuegen gt eval      -c configs/experiments/ground_truth/X.yaml [--dry-run] [--no-wait]
    valuegen gt status    -c configs/experiments/ground_truth/X.yaml
    valuegen gt matrices  -c configs/experiments/ground_truth/X.yaml

An experiment is an ``intervention:`` block (or an ``evaluation.intervention:``
artifact-ID reference) plus an ``evaluation:`` block, with separate identities.
``intervene`` builds just the intervention artifact (datasets, DPO training,
LoRA merge + manifest); ``eval`` scores an existing artifact
(N+1 eval arrays, base evals shared across GT runs); ``run`` does both.
``eval`` never trains — a referenced or incomplete intervention is an error,
not an implicit build. Datasets are built inline and idempotently; the stage
lists go to the SLURM orchestrator (submit pending, poll, resubmit).
``--dry-run`` writes SBATCH scripts under ``slurm_jobs/{id}/`` without
submitting; ``--no-wait`` submits with SLURM dependencies and returns.
``status`` is read-only. ``matrices`` builds the eval method's GT artifact
(steerability matrices + values.json + provenance) from finished eval CSVs.
A top-level ``schedule: {wave: W}`` makes ``run`` alternate training and eval
in waves of W values per model (train W, eval W, repeat; judge released
between waves) instead of training everything first — polling mode only, and
outside every identity, so it can be added to or removed from a config freely.
``schedule.delete_exports: true`` (or ``gt run --delete-exports``; needs
``schedule.wave``) frees each wave's checkpoints — merged dir + trainer weight
shards — once this GT run's evals of them are done, leaving a
``<merged>.deleted`` tombstone so nothing retrains. Scoring the same
intervention under another eval config afterwards needs a retrain (remove the
tombstone, rerun ``gt intervene``); ``gt eval`` refuses up front otherwise.

Data / predictor surface::

    valuegen data build --method conflictscope_action_prompt --value-set ... \\
        --model allenai/OLMo-2-1124-7B-SFT [--values ...] [--param k=v ...]
    valuegen data list
    valuegen predict --method persona --value-set ... --model ... \\
        [--data default_llm | <artifact-id>] [--layer N] [--build] [--dry-run]

``--data`` omitted -> the predictor's default data method (the policy model
defaults to the extraction base model — the on-policy default). Auto-build is
cost-gated: cheap builds run silently; anything that submits SLURM prints the
job plan and requires confirmation or ``--build``.

Analysis surface::

    valuegen analyze correlate --target <matrix> --pred <matrix> [--pred ...] \\
        [--ceiling F | --ceiling-halves A B] [--all-cells] [--out CSV]
    valuegen analyze mds --matrix <matrix> [--variants metric_raw,nonmetric] \\
        [--out DIR]
    valuegen analyze cluster --matrix <matrix> [--methods upgma,kmedoids,mds,hdbscan] \\
        [--max-k N] [--k N] [--subset VALUES.json] [--maps] [--stability] [--out DIR]

``mds`` draws the map; ``cluster`` does the clustering and scores it
(silhouette, z against a matched null, Hubert's Γ), writes cluster membership
and medoids, and optionally maps and subsampling stability.

A ``<matrix>`` is any ``.npy`` sharing the values.json contract (GT matrices,
predictor grids, half-matrices — all the same call), or predictor sugar
``predictor[:data_method[:model[:name]]]`` resolved through the store.
"""

from __future__ import annotations

import argparse
import functools
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from valuegen.config import (
    canonical_experiment,
    ensure_config_record,
    gt_id,
    intervention_id,
    load_cluster,
    load_experiment,
)
from valuegen.ground_truth import common, interventions, scenario_pool
from valuegen.ground_truth.evals import get_eval
from valuegen.multivalue import cli as mv_cli
from valuegen.slurm import Orchestrator, Stage, Task, sbatch_available


def _gt_parser(sub: argparse._SubParsersAction) -> None:
    gt = sub.add_parser("gt", help="ground-truth pipelines")
    gt_sub = gt.add_subparsers(dest="verb", required=True)
    for verb, help_text in (
        ("run", "intervene + eval: the whole experiment"),
        ("intervene", "training side only: build the intervention artifact"),
        ("eval", "evaluation side only: score an existing intervention"),
        ("status", "show per-stage done counts (read-only)"),
        ("matrices", "build the GT artifact from finished eval outputs"),
    ):
        p = gt_sub.add_parser(verb, help=help_text)
        p.add_argument("--config", "-c", required=True, help="experiment YAML")
        p.add_argument("--cluster", default=None, help="cluster YAML override")
        if verb in ("run", "intervene", "eval"):
            p.add_argument("--dry-run", action="store_true",
                           help="write SBATCH scripts, submit nothing")
            p.add_argument("--no-wait", action="store_true",
                           help="submit pending stages with SLURM dependencies "
                                "and return (no polling/retries)")
        if verb == "run":
            p.add_argument("--delete-exports", default=None,
                           action=argparse.BooleanOptionalAction,
                           help="delete each wave's checkpoints once this GT "
                                "run's evals of them are done (overrides "
                                "schedule.delete_exports; needs schedule.wave)")
            p.add_argument("--stop-after-wave", type=int, default=None, metavar="N",
                           help="schedule only the first N waves per model, "
                                "base eval included (overrides "
                                "schedule.stop_after_wave; needs schedule.wave); "
                                "rerun without it to resume from wave N+1")
            p.add_argument("--submit", action="store_true",
                           help="sbatch a self-resubmitting controller job that "
                                "runs this command (poll-only; placed on "
                                "slurm.controller_partition) instead of polling "
                                "from this shell")
        if verb in ("run", "intervene"):
            p.add_argument("--until", default=None, metavar="STAGE",
                           help="stop after the named stage (e.g. 'generate' "
                                "to build model_spec_aft datasets without "
                                "training); later stages are not submitted")


def _gt_eval_stages(cfg: dict, cluster, manifest: dict) -> list:
    """Eval-side stages stamped with the GT run identity (pools excluded —
    the caller schedules pools once for both sides)."""
    driver = get_eval(cfg["evaluation"]["method"])
    return interventions.stamp(
        driver.stages(cfg, cluster, manifest),
        gt_id(cfg),
        common.gt_record_path(cfg, cluster),
    )


def _delete_exports(cfg: dict, args, cluster) -> bool:
    """``gt run``'s effective ``schedule.delete_exports`` (CLI over YAML)."""
    schedule = cfg.get("schedule") or {}
    override = getattr(args, "delete_exports", None)
    delete = bool(schedule.get("delete_exports")) if override is None else override
    if not delete:
        return False
    if not schedule.get("wave"):
        raise SystemExit(
            "--delete-exports needs schedule.wave: checkpoints are freed by "
            "the per-wave hook, after that wave's evals."
        )
    if cluster.gcs is not None:
        raise SystemExit(
            f"delete_exports: cluster stages checkpoints to {cluster.gcs.bucket} "
            "(gcs: block); local copies are already push-and-pruned, so "
            "deletion is a no-op — drop the flag."
        )
    return True


def _sequence(
    cfg: dict,
    train_stages: list,
    eval_stages: list,
    manifest: dict,
    no_wait: bool,
    delete=None,
    stop_after: int | None = None,
) -> list:
    """Order the training and eval stages for the orchestrator.

    Default: every training stage, then every eval stage. With a top-level
    ``schedule.wave: W`` and both sides present (``gt run``), the two are
    re-cut into alternating train/eval waves of W values per model (see
    ``interventions.interleave_waves``); the first W checkpoints are scored
    while the rest still train. Waves rely on the polling controller's
    server cancel/resubmit cycle, which the dependency-chained ``--no-wait``
    submission does not have (it would leave the judge resident through
    every training wave), so the two are refused together. ``delete``
    (``schedule.delete_exports``, which requires a wave) is the per-entry
    deleter each eval wave runs once it completes. ``stop_after``
    (``--stop-after-wave``) overrides ``schedule.stop_after_wave``.
    """
    schedule = cfg.get("schedule") or {}
    wave = schedule.get("wave")
    if stop_after is None:
        stop_after = schedule.get("stop_after_wave")
    if stop_after is not None and (stop_after < 1 or not wave):
        raise SystemExit("--stop-after-wave needs a positive N and schedule.wave.")
    if not (wave and train_stages and eval_stages):
        return train_stages + eval_stages
    if no_wait:
        raise SystemExit(
            f"schedule.wave: {wave} needs the polling controller — drop "
            "--no-wait, or remove schedule.wave to submit a dependency chain."
        )
    if stop_after is not None:
        print(f"schedule: stopping after wave {stop_after} per model "
              f"(waves of {wave}); rerun without the stop to resume.")
    return interventions.interleave_waves(
        train_stages, eval_stages, manifest, wave, delete=delete,
        stop_after=stop_after,
    )


def _tombstoned_pending(manifest: dict, eval_stages: list) -> list[tuple[dict, dict]]:
    """Manifest entries whose checkpoint was deleted on purpose but whose eval
    task in *this* GT run is still pending — unscoreable without a retrain."""
    tasks = {t.key: t for s in eval_stages if not s.server for t in s.tasks}
    blocked = []
    for entry in manifest["interventions"]:
        record = interventions.tombstone(entry)
        task = tasks.get(entry["tag"])
        if record is not None and task is not None and not task.is_done():
            blocked.append((entry, record))
    return blocked


def _run_orchestrator(args, name: str, stages: list, cluster) -> int:
    orch = Orchestrator(
        name=name,
        stages=stages,
        cluster=cluster,
        dry_run=getattr(args, "dry_run", False),
    )
    if not orch.dry_run and not sbatch_available():
        print("sbatch not found — run this from a SLURM login node, or pass "
              "--dry-run to inspect the scripts.", file=sys.stderr)
        return 1
    if getattr(args, "no_wait", False):
        orch.submit()
        return 0
    return 0 if orch.run() else 1


DRIVER_STAGE = "run"
DRIVER_SCRIPT = "run_driver.sbatch"


def _driver_stage(cluster) -> Stage:
    """The poll-only controller job (same placement as ``mv run --submit``)."""
    partition = cluster.controller_partition or cluster.cpu_partition
    cap = cluster.partition_max_time.get(partition) if partition else None
    return Stage(name=DRIVER_STAGE, tasks=[], time=str(cap or cluster.default_time),
                 mem=str(cluster.default_mem), gpus=0, controller=True)


def _driver_command(args, cluster) -> str:
    cmd = f"python -m valuegen.cli gt run -c {shlex.quote(str(Path(args.config).resolve()))}"
    if cluster.source_path is not None:
        cmd += f" --cluster {shlex.quote(str(cluster.source_path))}"
    if args.delete_exports is not None:
        cmd += " --delete-exports" if args.delete_exports else " --no-delete-exports"
    if args.stop_after_wave is not None:
        cmd += f" --stop-after-wave {int(args.stop_after_wave)}"
    if getattr(args, "until", None):
        cmd += f" --until {shlex.quote(args.until)}"
    return cmd


def _render_driver(args, name: str, cluster) -> Path:
    """Write ``slurm_jobs/{gt_id}/run_driver.sbatch``."""
    orch = Orchestrator(name=name, stages=[], cluster=cluster, dry_run=True)
    stage = _driver_stage(cluster)
    script = orch.script_dir / DRIVER_SCRIPT
    body = (
        orch._directives(stage, 1, 0) + "\n" + orch._preamble(stage)
        + "\n# Successor first: this controller only polls (the work lives in the arrays it\n"
        "# submits), so a kill at the walltime cap loses nothing -- the successor adopts\n"
        "# live arrays by job name and skips done tasks. The controller returning on its\n"
        "# own cancels it -- also on failure (unlike `mv run`): a stage that exhausted its\n"
        "# retries would otherwise be retried by every successor, 8 GPUs at a time.\n"
        f"NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID {shlex.quote(str(script))}) || NEXT=\"\"\n"
        'echo "successor: ${NEXT:-none}"\n'
        "set +e\n"
        f"{_driver_command(args, cluster)}\n"
        "rc=$?\n"
        "set -e\n"
        'if [ -n "$NEXT" ]; then scancel "$NEXT"; fi\n'
        "exit $rc\n"
    )
    orch.script_dir.mkdir(parents=True, exist_ok=True)
    orch.log_dir.mkdir(parents=True, exist_ok=True)
    script.write_text(body)
    script.chmod(0o755)
    return script


def _submit_driver(args, name: str, cluster) -> int:
    """``gt run --submit``: render the driver and sbatch it, unless a live
    chain for this GT run already exists."""
    script = _render_driver(args, name, cluster)
    print(f"driver: {script}")
    if args.dry_run:
        print("dry run: not submitted")
        return 0
    if not sbatch_available():
        print("sbatch not found — run this from a SLURM login node, or pass "
              "--dry-run to inspect the scripts.", file=sys.stderr)
        return 1
    orch = Orchestrator(name=name, stages=[], cluster=cluster)
    live = orch._find_active_job(_driver_stage(cluster))
    if live is not None:
        print(f"a controller chain for {name} is already live (job {live}); "
              "not submitting a second one")
        return 1
    result = subprocess.run(["sbatch", "--parsable", str(script)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"submitted controller job {result.stdout.strip().split(';')[0]} "
          f"({script.name}); logs under {orch.log_dir}/{DRIVER_STAGE}_<jobid>.out")
    return 0


def cmd_gt(args: argparse.Namespace) -> int:
    cfg = load_experiment(args.config)
    cluster = load_cluster(args.cluster)
    verb = args.verb
    if getattr(args, "submit", False):
        if args.no_wait:
            raise SystemExit("--submit runs the polling controller as a job; drop --no-wait.")
        # Same refusals the controller would hit, before a job is queued.
        _delete_exports(cfg, args, cluster)
        if args.stop_after_wave is not None and (
            args.stop_after_wave < 1 or not (cfg.get("schedule") or {}).get("wave")
        ):
            raise SystemExit("--stop-after-wave needs a positive N and schedule.wave.")
        return _submit_driver(args, gt_id(cfg), cluster)
    trains = verb in ("run", "intervene") and cfg.get("intervention") is not None
    evaluates = verb in ("run", "eval", "matrices")
    dry_run = getattr(args, "dry_run", False)

    # The manifest is the seam: built in memory for an inline block, loaded
    # from disk for a reference. A reference that does not exist fails here —
    # `gt eval` never trains, and never silently builds someone else's work.
    manifest = interventions.load_manifest(cfg, cluster)

    if verb == "matrices":
        driver = get_eval(cfg["evaluation"]["method"])
        ensure_config_record(
            common.gt_record_path(cfg, cluster),
            gt_id(cfg),
            canonical_experiment(cfg),
        )
        written = driver.build_ground_truth(cfg, cluster, manifest)
        if not written:
            print("No ground truth written (eval outputs incomplete?)")
            return 1
        return 0

    if evaluates and verb != "matrices" and not dry_run:
        # Tombstoned checkpoints (deleted after their evals) are not missing
        # work; whether they block this run is decided against the eval
        # done-checks below (_tombstoned_pending).
        missing = [
            e for e in interventions.missing_entries(manifest)
            if interventions.tombstone(e) is None
        ]
        if missing and not trains:
            print(
                f"ERROR: {len(missing)}/{len(manifest['interventions'])} "
                f"intervention payloads missing (e.g. {missing[0]['path']}). "
                "Run `valuegen gt intervene` (or `run`) on the config that "
                "owns the intervention first.", file=sys.stderr,
            )
            return 1
        scenarios = cfg["evaluation"].get("scenarios")
        if not isinstance(scenarios, dict):
            scenario_dir = common.configured_path(scenarios, cfg, cluster)
            if not scenario_dir.is_dir():
                print(f"WARNING: evaluation.scenarios {scenario_dir} does not "
                      "exist — the eval stages will fail. Build or import the "
                      "scenario set for this value set first.", file=sys.stderr)

    # Pools this verb touches, deduplicated across the two sides (the same
    # pool block on both sides is trained-on and evaluated-on once).
    pools = []
    if trains or verb == "status":
        pools += scenario_pool.intervention_pools(cfg)
    if evaluates or verb == "status":
        pools += scenario_pool.evaluation_pools(cfg)

    stages = scenario_pool.pending_stages(cfg, cluster, pools)
    train_stages: list = []
    eval_stages: list = []
    if trains:
        if getattr(args, "delete_exports", None) is not None:
            # --delete-exports also decides the trainer-save prune default,
            # which is rendered into the training tasks below.
            cfg["schedule"] = {
                **(cfg.get("schedule") or {}), "delete_exports": args.delete_exports,
            }
        if not dry_run:
            # Claims the intervention artifact + inline, idempotent dataset
            # build (label_subset loads HF datasets on a cold cache).
            scenario_pool.claim_pools(cfg, cluster, pools)
            interventions.build_data(cfg, cluster)
        train_stages = interventions.stages(cfg, cluster)
    if verb in ("run", "eval"):
        if not dry_run:
            scenario_pool.claim_pools(cfg, cluster, pools)
            ensure_config_record(
                common.gt_record_path(cfg, cluster),
                gt_id(cfg),
                canonical_experiment(cfg),
            )
            get_eval(cfg["evaluation"]["method"]).claim_base_evals(cfg, cluster)
        eval_stages = _gt_eval_stages(cfg, cluster, manifest)
        blocked = _tombstoned_pending(manifest, eval_stages)
        if blocked:
            entry, record = blocked[0]
            marker = interventions.gcs.deleted_marker_path(entry["path"])
            print(
                f"ERROR: {len(blocked)} checkpoint(s) with pending evals were "
                f"deleted after gt_id {record.get('gt_id', '?')} scored them "
                f"(e.g. {entry['tag']}). To rescore, remove {marker} and rerun "
                "`valuegen gt intervene`.", file=sys.stderr,
            )
            return 1
    if verb == "status":
        train_stages = interventions.stages(cfg, cluster)
        eval_stages = _gt_eval_stages(cfg, cluster, manifest)
    delete = None
    if verb == "run" and _delete_exports(cfg, args, cluster):
        delete = functools.partial(interventions.delete_checkpoint, cfg, cluster)
    stages += _sequence(
        cfg, train_stages, eval_stages, manifest,
        getattr(args, "no_wait", False), delete,
        getattr(args, "stop_after_wave", None) if verb == "run" else None,
    )

    if verb == "status":
        Orchestrator(gt_id(cfg), stages, cluster).status()
        pruned = [
            e["tag"] for e in manifest["interventions"]
            if interventions.tombstone(e) is not None
        ]
        if pruned:
            print(f"  pruned (checkpoint deleted after its evals): "
                  f"{len(pruned)}/{len(manifest['interventions'])}")
            for tag in pruned:
                print(f"      - {tag}")
        return 0

    until = getattr(args, "until", None)
    if until:
        names = [st.name for st in stages]
        if until not in names:
            print(f"ERROR: --until {until!r}: no such pending stage; pending "
                  f"stages are {names or '[] (everything is already done)'}",
                  file=sys.stderr)
            return 1
        stages = stages[: names.index(until) + 1]

    name = intervention_id(cfg) if verb == "intervene" else gt_id(cfg)
    return _run_orchestrator(args, name, stages, cluster)


# ── data / predict ───────────────────────────────────────────────────────────


def _parse_params(pairs: list[str] | None) -> dict:
    """``k=v`` strings -> dict, YAML-coercing values (ints, bools, null)."""
    params = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects k=v, got {pair!r}")
        key, _, value = pair.partition("=")
        params[key] = yaml.safe_load(value)
    return params


def _confirm(
    plan_lines: list[str],
    approved: bool,
    what: str,
    cost: str = "will submit SLURM jobs",
) -> None:
    """The cost gate: work that spends real resources needs --build or a yes."""
    print(f"\n{what} {cost}:")
    for line in plan_lines:
        print(f"  {line}")
    if approved:
        return
    if sys.stdin.isatty():
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply in ("y", "yes"):
            return
    raise SystemExit(
        "Aborted — a cold cache must never silently launch a training fleet. "
        "Re-run with --build to approve, or --dry-run to inspect the scripts."
    )


def _confirm_gate(args):
    """The gate as a callable, for predictors that build auxiliary data."""
    def gate(plan_lines: list[str], what: str) -> None:
        _confirm(plan_lines, args.build, what, cost="will spend API credits")

    return gate


def _data_parser(sub: argparse._SubParsersAction) -> None:
    data = sub.add_parser("data", help="elicitation datasets (the DATA layer)")
    data_sub = data.add_subparsers(dest="verb", required=True)

    build = data_sub.add_parser("build", help="build (or resume) a pair/description artifact")
    build.add_argument("--method", "-m", required=True, help="data method (see elicitation.registry)")
    build.add_argument("--value-set", required=True, help="registry name (or a path into value_sets/)")
    build.add_argument("--model", default=None, help="policy model (HF name or API model)")
    build.add_argument("--values", nargs="+", default=None, help="subset (default: whole value set)")
    build.add_argument("--param", action="append", default=None, metavar="K=V",
                       help="method param (e.g. scenarios_dir=..., n_pairs=50); repeatable")
    build.add_argument("--dry-run", action="store_true", help="render scripts, submit nothing")
    build.add_argument("--cluster", default=None, help="cluster YAML override")

    lst = data_sub.add_parser("list", help="show existing artifacts + manifests")
    lst.add_argument("--cluster", default=None, help="cluster YAML override")


def cmd_data(args: argparse.Namespace) -> int:
    from valuegen.elicitation import datasets as D
    from valuegen.elicitation import registry

    cluster = load_cluster(args.cluster)
    if args.verb == "list":
        artifacts = D.list_artifacts(cluster)
        if not artifacts:
            print("no artifacts under data/elicitation/")
            return 0
        for a in artifacts:
            manifest = D.read_manifest(a.root) or {}
            counts = manifest.get("n_pairs_per_value", {})
            n = sum(counts.values()) if counts else "-"
            print(f"{a.artifact_id:<28} {a.method:<34} {a.cfg['value_set']:<20} "
                  f"model={a.cfg.get('model')} values={len(a.values)} pairs={n} "
                  f"date={manifest.get('date')}")
            print(f"  {a.root}")
        return 0

    cfg = registry.resolve_config(
        args.method, args.value_set, model=args.model,
        values=args.values, params=_parse_params(args.param),
    )
    if registry.needs_slurm(cfg):
        # An explicit `data build` is already an intentional launch; show the
        # plan for the record but don't demand a second confirmation.
        print("job plan:")
        for line in registry.build_plan(cfg, cluster):
            print(f"  {line}")
        if not args.dry_run and not sbatch_available():
            print("sbatch not found — run from a SLURM login node or pass "
                  "--dry-run.", file=sys.stderr)
            return 1
    artifact = registry.build(cfg, cluster, dry_run=args.dry_run)
    if not args.dry_run:
        state = "complete" if artifact.is_complete() else "INCOMPLETE (rerun to resume)"
        print(f"artifact {artifact.artifact_id} at {artifact.root}: {state}")
        return 0 if artifact.is_complete() else 1
    return 0


def _predict_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("predict", help="build predictor vectors + similarity matrices")
    p.add_argument("--method", "-m", required=True, help="persona | grad_proj | weight_steer | sentence_emb")
    p.add_argument("--value-set", required=True, help="registry name (or a path into value_sets/)")
    p.add_argument("--model", required=True,
                   help="base model whose representations are probed "
                        "(sentence_emb: ignored in favor of --encoder)")
    p.add_argument("--data", default=None,
                   help="data method name or an existing artifact ID/path "
                        "(default: the predictor's default data method, "
                        "auto-built if missing)")
    p.add_argument("--values", nargs="+", default=None, help="subset (default: whole value set)")
    p.add_argument("--param", action="append", default=None, metavar="K=V",
                   help="data-method param for auto-build (repeatable)")
    p.add_argument("--predictor-param", action="append", default=None,
                   metavar="K=V",
                   help="predictor-level knob (see each predictor's "
                        "PREDICTOR_PARAMS, e.g. persona/grad_proj template=urial0, "
                        "weight_steer scaffold=urial0); repeatable")
    p.add_argument("--layer", type=int, default=None, help="persona: layer for the similarity grid")
    p.add_argument("--pooling", default=None, help="persona: vector kind (default response_avg_diff)")
    p.add_argument("--sweep-on-general", action="store_true",
                   help="persona: run the layer sweep on the main GPU "
                        "partition instead of the optional array pool "
                        "(cluster.yaml array_partition), e.g. when that pool "
                        "is preemptible")
    p.add_argument("--encoder", default=None, help="sentence_emb: encoder model")
    p.add_argument("--gpus", type=int, default=1, help="GPUs per extraction/train task")
    p.add_argument("--build", action="store_true",
                   help="approve SLURM submission without prompting")
    p.add_argument("--dry-run", action="store_true", help="render scripts, submit nothing")
    p.add_argument("--cluster", default=None, help="cluster YAML override")


def cmd_predict(args: argparse.Namespace) -> int:
    from valuegen.elicitation import datasets as D
    from valuegen.elicitation import registry
    from valuegen.predictors import PREDICTOR_SCHEMA_VERSIONS, get_module, get_predictor

    cluster = load_cluster(args.cluster)
    predictor = get_predictor(args.method)

    # ── resolve the data artifact ────────────────────────────────────────────
    artifact = None
    data_ref = args.data or predictor.default_data
    if args.data:
        artifact = D.find_artifact_by_id(cluster, args.data)
    if artifact is None:
        if registry.ALIASES.get(data_ref, data_ref) not in registry.DATA_METHODS:
            raise SystemExit(
                f"--data {data_ref!r} is neither an existing artifact ID/path "
                f"nor a known data method ({sorted(registry.DATA_METHODS)})"
            )
        data_cfg = registry.resolve_config(
            data_ref, args.value_set, model=args.model,
            values=args.values, params=_parse_params(args.param),
        )
        artifact = D.resolve_artifact(cluster, data_cfg)
        if not artifact.is_complete():
            if registry.needs_slurm(data_cfg) and not args.dry_run:
                _confirm(registry.build_plan(data_cfg, cluster), args.build,
                         f"auto-building data artifact {artifact.artifact_id}")
            artifact = registry.build(data_cfg, cluster, dry_run=args.dry_run)
            if args.dry_run and not artifact.is_complete():
                print("[dry-run] data artifact not built; predictor scripts "
                      "below assume it exists")
    if artifact.artifact_type != predictor.requires:
        raise SystemExit(
            f"predictor {predictor.name} requires a {predictor.requires!r} "
            f"artifact; {artifact.artifact_id} is {artifact.artifact_type!r}"
        )

    # ── resolve the predictor config ─────────────────────────────────────────
    values = args.values or artifact.values
    outside = [v for v in values if v not in artifact.values]
    if outside:
        raise SystemExit(f"values {outside} not in artifact {artifact.artifact_id}")
    cfg = {
        "predictor": predictor.name,
        "schema_version": PREDICTOR_SCHEMA_VERSIONS[predictor.name],
        "model": args.model,
        "values": list(values),
        "gpus": args.gpus,
    }
    if args.layer is not None:
        cfg["layer"] = args.layer
    if args.pooling:
        cfg["pooling"] = args.pooling
    if getattr(args, "sweep_on_general", False):
        cfg["sweep_array_partition"] = False
    if args.encoder:
        cfg["encoder"] = args.encoder

    module = get_module(predictor.name)
    extra = _parse_params(getattr(args, "predictor_param", None))
    if extra:
        allowed = set(getattr(module, "PREDICTOR_PARAMS", ()))
        unknown = set(extra) - allowed
        if unknown:
            raise SystemExit(
                f"--predictor-param: {predictor.name} accepts "
                f"{sorted(allowed) or 'no keys'}; unknown {sorted(unknown)}"
            )
        cfg.update(extra)
    # Some predictors need auxiliary data of their own (persona's layer sweep
    # steers on a held-out eval pool). Same auto-build-behind-the-gate rule.
    ensure_aux = getattr(module, "ensure_aux_data", None)
    if ensure_aux is not None:
        ensure_aux(cfg, cluster, artifact, _confirm_gate(args), dry_run=args.dry_run)
    if not args.dry_run and module.needs_slurm(cfg, cluster, artifact):
        _confirm(module.plan(cfg, cluster, artifact), args.build,
                 f"predictor {predictor.name}")
        if not sbatch_available():
            print("sbatch not found — run from a SLURM login node or pass "
                  "--dry-run.", file=sys.stderr)
            return 1
    out = module.run(cfg, cluster, artifact, dry_run=args.dry_run)
    return 0 if (args.dry_run or out is not None) else 1


# ── analyze ──────────────────────────────────────────────────────────────────


def _analyze_parser(sub: argparse._SubParsersAction) -> None:
    analyze = sub.add_parser("analyze", help="correlate matrices / MDS maps")
    an_sub = analyze.add_subparsers(dest="verb", required=True)

    corr = an_sub.add_parser(
        "correlate",
        help="correlate a target matrix against predictor/reference matrices",
    )
    corr.add_argument("--target", "-t", required=True,
                      help="matrix path or predictor spec (the y-axis)")
    corr.add_argument("--pred", "-p", action="append", required=True,
                      metavar="SPEC",
                      help="matrix path or predictor[:data_method[:model[:name]]]; "
                           "repeatable")
    corr.add_argument("--model", default=None,
                      help="default model for predictor specs that omit it")
    corr.add_argument("--data-method", default=None,
                      help="default data method for predictor specs that omit it")
    corr.add_argument("--all-cells", action="store_true",
                      help="include the diagonal/self column (default: off-diag only)")
    corr.add_argument("--n-boot", type=int, default=5000, help="bootstrap resamples")
    corr.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    corr.add_argument("--ceiling", type=float, default=None,
                      help="reliability ceiling for ceiling-relative reporting")
    corr.add_argument("--ceiling-halves", nargs=2, metavar=("A", "B"), default=None,
                      help="two half-matrices; their off-diag Pearson is "
                           "Spearman–Brown-corrected into the ceiling")
    corr.add_argument("--out", default=None, help="also write the leaderboard CSV here")
    corr.add_argument("--cluster", default=None, help="cluster YAML override")

    mds = an_sub.add_parser(
        "mds", help="2D MDS map of a similarity matrix (clustering lives in "
                    "`analyze cluster`)")
    mds.add_argument("--matrix", "-m", required=True,
                     help="matrix path or predictor spec (square similarity)")
    mds.add_argument("--variants", default="metric_raw,nonmetric",
                     help="comma list of metric_raw|nonmetric (raw 1−cos distances)")
    mds.add_argument("--model", default=None,
                     help="default model for a predictor spec that omits it")
    mds.add_argument("--data-method", default=None,
                     help="default data method for a predictor spec that omits it")
    mds.add_argument("--out", default=None,
                     help="output dir (default data/analysis/mds/{label})")
    mds.add_argument("--cluster", default=None, help="cluster YAML override")

    cl = an_sub.add_parser(
        "cluster", help="cluster a similarity matrix: curves, members, maps, stability")
    cl.add_argument("--matrix", "-m", required=True,
                    help="matrix path or predictor spec (square similarity)")
    cl.add_argument("--methods", default="upgma,kmedoids,mds,hdbscan",
                    help="comma list of upgma|kmedoids|mds|hdbscan")
    cl.add_argument("--max-k", type=int, default=8, help="largest k to sweep")
    cl.add_argument("--k", type=int, default=None,
                    help="k for membership/medoids/maps (default: best by z)")
    cl.add_argument("--subset", default=None,
                    help="JSON list or dict of value names to restrict to")
    cl.add_argument("--mcs", type=int, default=4, help="HDBSCAN min_cluster_size")
    cl.add_argument("--nrand", type=int, default=2000,
                    help="random partitions for the z null on the full matrix. "
                         ">=2000 for a single z quoted as a result; 500 is enough "
                         "for curve shape. See --nrand-boot for the stability pass")
    cl.add_argument("--dendrogram", action="store_true",
                    help="draw the UPGMA tree (every k at once)")
    cl.add_argument("--maps", action="store_true",
                    help="also draw cluster maps with medoids starred")
    cl.add_argument("--stability", action="store_true",
                    help="also run subsampling stability (best k, rank, leave-one-out)")
    cl.add_argument("--boot", type=int, default=100, help="subsamples for --stability")
    cl.add_argument("--nrand-boot", type=int, default=400,
                    help="null draws inside each --stability subsample. Lower "
                         "than --nrand on purpose: this runs --boot times over, "
                         "and best-k is an argmax over the z curve, where null "
                         "noise partly cancels across k. Do not go below ~300 — "
                         "at 120 the null sd is underestimated, which inflates z "
                         "and manufactures spurious rank instability")
    cl.add_argument("--frac", type=float, default=0.8, help="fraction kept per subsample")
    cl.add_argument("--model", default=None,
                    help="default model for a predictor spec that omits it")
    cl.add_argument("--data-method", default=None,
                    help="default data method for a predictor spec that omits it")
    cl.add_argument("--out", default=None,
                    help="output dir (default data/analysis/cluster/{label})")
    cl.add_argument("--cluster-config", dest="cluster", default=None,
                    help="cluster YAML override")


def cmd_analyze(args: argparse.Namespace) -> int:
    from valuegen.analysis import correlate as C

    cluster = load_cluster(args.cluster)

    if args.verb == "correlate":
        target, rows, cols, label = C.resolve_matrix(
            args.target, cluster, args.model, args.data_method
        )
        ceiling = args.ceiling
        if args.ceiling_halves:
            spec_a, spec_b = args.ceiling_halves
            m_a, r_a, c_a, _ = C.resolve_matrix(spec_a, cluster, args.model, args.data_method)
            m_b, r_b, c_b, _ = C.resolve_matrix(spec_b, cluster, args.model, args.data_method)
            aligned_b = C.reindex(m_b, r_b, c_b, r_a, c_a)
            half = C.correlate(m_a, aligned_b,
                               C.offdiag_mask(r_a, c_a) if not args.all_cells else None,
                               n_boot=0)
            ceiling = C.spearman_brown(half.pearson_r)
            print(f"split-half r = {half.pearson_r:+.3f} (n={half.n})  →  "
                  f"Spearman–Brown ceiling = {ceiling:.3f}")
        predictors = {}
        for spec in args.pred:
            matrix, p_rows, p_cols, p_label = C.resolve_matrix(
                spec, cluster, args.model, args.data_method
            )
            predictors[p_label] = (matrix, p_rows, p_cols)
        entries = C.leaderboard(
            target, rows, cols, predictors,
            off_diag_only=not args.all_cells,
            n_boot=args.n_boot, seed=args.seed, ceiling=ceiling,
        )
        print(f"\ntarget: {label}  ({len(rows)}×{len(cols)}, "
              f"{'all cells' if args.all_cells else 'off-diagonal'})")
        print(C.format_leaderboard(entries, ceiling=ceiling))
        if args.out:
            import csv

            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
                writer.writeheader()
                writer.writerows(entries)
            print(f"wrote {out}")
        return 0

    if args.verb == "mds":
        import csv

        import numpy as np

        from valuegen.analysis import cluster as CL
        from valuegen.analysis import mds as M_
        from valuegen.analysis import plots

        matrix, rows, cols, label = C.resolve_matrix(
            args.matrix, cluster, args.model, args.data_method
        )
        if rows != cols:
            raise SystemExit(f"{label} is not a square similarity matrix")
        sim, values = M_.drop_nan_values(matrix, rows)
        if len(values) < len(rows):
            dropped = sorted(set(rows) - set(values))
            print(f"dropped NaN-padded values: {dropped}")

        out_dir = Path(args.out) if args.out else (
            cluster.data / "analysis" / "mds" / label.replace("/", "_")
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        results = {}
        for variant in variants:
            coords, stress = M_.mds_variant(sim, variant)
            results[variant] = (coords, f"stress={stress:.4f}")
            print(f"{variant:11s} stress={stress:.4f}")
            with open(out_dir / f"coords_{variant}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["value", "dim1", "dim2"])
                writer.writerows(
                    [v, f"{x:.6f}", f"{y:.6f}"]
                    for v, (x, y) in zip(values, results[variant][0])
                )

        E = sim.copy()
        np.fill_diagonal(E, np.nan)
        mean_offdiag = np.nanmean(E, axis=1)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(results), figsize=(8.0 * len(results), 7.0),
                                 squeeze=False)
        for ax, (variant, (coords, note)) in zip(axes[0], results.items()):
            plots.embedding_map(
                coords, values, ax=ax, title=f"{variant}\n{note}",
                color_values=mean_offdiag,
                color_label="mean off-diag cosine (gradient proxy)",
            )
        fig.suptitle(label, fontsize=12)
        plots.save(fig, out_dir / "map.png")

        print(f"outputs in {out_dir}")
        return 0

    if args.verb == "cluster":
        import csv
        import json as _json

        import numpy as np

        from valuegen.analysis import cluster as CL
        from valuegen.analysis import mds as M_

        matrix, rows, cols, label = C.resolve_matrix(
            args.matrix, cluster, args.model, args.data_method
        )
        if rows != cols:
            raise SystemExit(f"{label} is not a square similarity matrix")
        sim, values = M_.drop_nan_values(matrix, rows)
        if len(values) < len(rows):
            print(f"dropped NaN-padded values: {sorted(set(rows) - set(values))}")

        if args.subset:
            raw = _json.loads(Path(args.subset).read_text())
            want = set(raw)
            missing = want - set(values)
            if missing:
                raise SystemExit(
                    f"--subset names {len(missing)} value(s) absent from {label}: "
                    f"{sorted(missing)[:5]}"
                )
            keep = [i for i, v in enumerate(values) if v in want]
            sim = sim[np.ix_(keep, keep)]
            values = [values[i] for i in keep]
            print(f"subset: {len(values)} of {len(rows)} values")

        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = set(methods) - set(CL.CLUSTER_METHODS)
        if unknown:
            raise SystemExit(f"unknown method(s) {sorted(unknown)}; "
                             f"known: {list(CL.CLUSTER_METHODS)}")

        out_dir = Path(args.out) if args.out else (
            cluster.data / "analysis" / "cluster" / label.replace("/", "_")
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Which model produced this matrix is not always in its filename --
        # the GT matrices carry it, the predictor grids do not -- so read the
        # provenance sidecar, which sits beside the .npy in some packages and
        # in a sibling provenance/ dir in others.
        def _model_of(spec: str) -> str:
            mp = Path(spec)
            if mp.suffix != ".npy":
                return ""
            for cand in (mp.with_name(f"{mp.stem}_provenance.yaml"),
                         mp.parent.parent / "provenance"
                         / f"{mp.stem}_provenance.yaml"):
                if cand.is_file():
                    try:
                        rec = yaml.safe_load(cand.read_text()) or {}
                    except Exception:
                        return ""
                    name = rec.get("model") or (
                        rec.get("resolved_config") or {}).get("model") or ""
                    return str(name).rstrip("/").split("/")[-1]
            return ""

        model_name = _model_of(args.matrix)
        title = f"{label}  —  n={len(values)}"
        if model_name:
            title += f"  —  {model_name}"

        dist = M_.cos_to_dist(sim)
        coords, stress = M_.mds_variant(sim, "nonmetric")
        ks = list(range(2, args.max_k + 1))
        # The null depends on the distance matrix and k, never on the method, so
        # it is drawn once here and shared across every method below.
        n_mean, n_sd, _ = CL.random_partition_stats(dist, ks, n_rand=args.nrand)

        print(f"{label}: n={len(values)}, nonmetric MDS stress={stress:.3f}\n")
        curves, best_k = {}, {}
        for m in methods:
            if m == "hdbscan":
                lab = CL.cluster_labels(dist, 0, m, min_cluster_size=args.mcs)
                sc = CL.score_partition(dist, lab, None, None)
                curves[m] = None
                print(f"{m:10s} {sc['n_clusters']} clusters, "
                      f"{sc['n_noise']}/{len(values)} noise, "
                      f"sil={sc['silhouette']:.3f} Γ={sc['gamma']:.3f} (no k, no z)")
                continue
            rows_out = []
            for i, k in enumerate(ks):
                lab = CL.cluster_labels(dist, k, m, coords=coords)
                rows_out.append(CL.score_partition(dist, lab, n_mean[i], n_sd[i]))
            curves[m] = rows_out
            zs = [r["z"] for r in rows_out]
            b = int(np.nanargmax(zs))
            best_k[m] = ks[b]
            print(f"{m:10s} best k={ks[b]}  z={zs[b]:5.1f}  "
                  f"sil={rows_out[b]['silhouette']:.3f}  Γ={rows_out[b]['gamma']:.3f}")
            print(f"{'':10s} z by k: "
                  + "  ".join(f"k{k}:{z:.1f}" for k, z in zip(ks, zs)))

        with open(out_dir / "curves.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "k", "silhouette", "z", "gamma"])
            for m, rr in curves.items():
                if rr is None:
                    continue
                for k, r in zip(ks, rr):
                    w.writerow([m, k, f"{r['silhouette']:.4f}", f"{r['z']:.3f}",
                                f"{r['gamma']:.4f}"])

        kpick = args.k or (min(best_k.values()) if best_k else 3)
        print(f"\nmembership at k={kpick}"
              + ("" if args.k else "  (lowest best-k across methods)"))
        with open(out_dir / f"clusters_k{kpick}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "value", "cluster", "is_medoid"])
            for m in methods:
                lab = CL.cluster_labels(dist, kpick, m, coords=coords,
                                        min_cluster_size=args.mcs)
                med = CL.medoids_of(dist, lab)
                print(f"  {m}")
                for c, mi in med.items():
                    n = int((lab == c).sum())
                    print(f"    ({n:2d})  {values[mi]}")
                if (lab == -1).any():
                    print(f"    noise: {int((lab == -1).sum())}")
                for i, v in enumerate(values):
                    w.writerow([m, v, int(lab[i]), int(i in med.values())])

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        drawn = [m for m in methods if curves[m] is not None]
        if drawn:
            fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
            for st, m in zip(["^-", "s-", "o-", "d-"], drawn):
                for ax, key in zip(axes, ["silhouette", "z", "gamma"]):
                    ax.plot(ks, [r[key] for r in curves[m]], st, label=m)
            for ax, key, ttl in zip(axes, ["silhouette", "z", "gamma"],
                                    ["Silhouette", "z-score", "Hubert Γ"]):
                ax.set_xlabel("k (clusters)")
                ax.set_ylabel(key)
                ax.set_xticks(ks)
                ax.set_title(ttl, fontsize=12)
                ax.legend(fontsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            fig.suptitle(title, fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(out_dir / "curves.png", dpi=150)
            print(f"\nwrote {out_dir / 'curves.png'}")

        if args.maps:
            cmap = matplotlib.colormaps["tab10"]
            fig, axes = plt.subplots(1, len(methods),
                                     figsize=(8.2 * len(methods), 8.0),
                                     squeeze=False)
            for ax, m in zip(axes[0], methods):
                lab = CL.cluster_labels(dist, kpick, m, coords=coords,
                                        min_cluster_size=args.mcs)
                med = CL.medoids_of(dist, lab)
                noise = lab == -1
                if noise.any():
                    ax.scatter(coords[noise, 0], coords[noise, 1], s=30,
                               color="0.78", label=f"noise ({int(noise.sum())})")
                for c in sorted(set(lab[~noise].tolist())):
                    sel = lab == c
                    ax.scatter(coords[sel, 0], coords[sel, 1], s=50,
                               color=cmap(c % 10), edgecolor="white",
                               linewidth=0.5,
                               label=f"n={int(sel.sum())} — {values[med[c]]}")
                for i, v in enumerate(values):
                    if i in med.values():
                        continue
                    ax.annotate(v, coords[i], fontsize=5.6, alpha=0.75,
                                xytext=(3.5, 2), textcoords="offset points")
                for c, mi in med.items():
                    ax.scatter(*coords[mi], s=380, marker="*",
                               color=cmap(c % 10), edgecolor="black",
                               linewidth=1.3, zorder=4)
                    ax.annotate(values[mi], coords[mi], fontsize=9,
                                fontweight="bold", xytext=(9, 6),
                                textcoords="offset points", zorder=5)
                # Every panel shares one nonmetric-MDS layout so the
                # partitions can be compared, but only the `mds` method
                # actually clusters in that space -- the others cluster the raw
                # distances and are merely drawn here, so the axes are named
                # for what they are (a layout) rather than for a method.
                where = ("clustered in this space" if m == "mds"
                         else "clustered on the raw distances")
                ktxt = "" if m == "hdbscan" else f"  k={kpick}"
                ax.set_title(f"{m}{ktxt}   ({where})", fontsize=13)
                ax.set_xlabel("layout dim 1  (nonmetric MDS)")
                ax.set_ylabel("layout dim 2  (nonmetric MDS)")
                ax.legend(fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            fig.suptitle(title, fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(out_dir / f"maps_k{kpick}.png", dpi=130)
            print(f"wrote {out_dir / f'maps_k{kpick}.png'}")

        if args.dendrogram:
            from scipy.cluster.hierarchy import dendrogram

            Z = CL.upgma_linkage(dist)
            # Cutting Z at k reproduces silhouette_curve_upgma's labels exactly,
            # so this is a view of the same clustering, not a second one.
            thresh = Z[-(kpick - 1), 2] if kpick > 1 else 0.0
            fig, ax = plt.subplots(figsize=(max(12.0, 0.34 * len(values)), 8.0))
            dendrogram(Z, ax=ax, labels=values, color_threshold=thresh,
                       leaf_font_size=7.0)
            ax.axhline(thresh, color="gray", linestyle="--", linewidth=1)
            ax.set_ylabel("merge height (1 − cos)")
            ax.set_title(f"{title} — UPGMA tree\n"
                         f"dashed line cuts at k = {kpick}", fontsize=13)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(out_dir / "dendrogram.png", dpi=150)
            print(f"\nwrote {out_dir / 'dendrogram.png'}")

        if args.stability:
            from collections import Counter

            from sklearn.metrics import adjusted_rand_score

            rng = np.random.default_rng(0)
            n = len(values)
            msz = int(round(args.frac * n))
            bk = {m: Counter() for m in drawn}
            # Subsampling is *without* replacement: a classical bootstrap draws
            # duplicates, and duplicate rows sit at distance 0, which distorts
            # both the clustering and the silhouette.
            for _ in range(args.boot):
                idx = np.sort(rng.choice(n, size=msz, replace=False))
                sub = dist[np.ix_(idx, idx)]
                sc_ = M_.mds_coords(sub, metric=False)[0]
                sm, ss, _ = CL.random_partition_stats(
                    sub, ks, n_rand=args.nrand_boot)
                for m in drawn:
                    zz = [CL.score_partition(
                        sub, CL.cluster_labels(sub, k, m, coords=sc_),
                        sm[i], ss[i])["z"] for i, k in enumerate(ks)]
                    bk[m][ks[int(np.nanargmax(zz))]] += 1
            print(f"\nbest k over {args.boot} subsamples ({msz}/{n} values):")
            for m in drawn:
                mode = bk[m].most_common(1)[0]
                dd = "  ".join(f"k{k}:{bk[m][k]}" for k in ks if bk[m][k])
                print(f"  {m:10s} full={best_k[m]}  mode={mode[0]} "
                      f"({100 * mode[1] / args.boot:.0f}%)   {dd}")

            print("\nleave-one-out ARI (lower = the structure leans on that value):")
            with open(out_dir / "leave_one_out.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "dropped_value", "ari"])
                for m in drawn:
                    full = CL.cluster_labels(dist, kpick, m, coords=coords)
                    aris = []
                    for i in range(n):
                        kp = [j for j in range(n) if j != i]
                        sub = dist[np.ix_(kp, kp)]
                        sc_ = M_.mds_coords(sub, metric=False)[0]
                        a = adjusted_rand_score(
                            full[kp], CL.cluster_labels(sub, kpick, m, coords=sc_))
                        aris.append(a)
                        w.writerow([m, values[i], f"{a:.4f}"])
                    worst = int(np.argmin(aris))
                    print(f"  {m:10s} mean {np.mean(aris):.3f}  "
                          f"min {min(aris):.3f} ({values[worst]})")

        print(f"\noutputs in {out_dir}")
        return 0

    raise SystemExit(f"unknown analyze verb {args.verb!r}")


def _controller_parser(sub) -> None:
    p = sub.add_parser(
        "controller",
        help="run a shell script of valuegen commands as a long-lived SLURM controller job",
        description=(
            "Submit SCRIPT (e.g. paper/regenerate/rq1.sh) as one GPU-less job on "
            "the cluster's controller partition (cluster.yaml controller_partition/"
            "controller_qos, else cpu_partition/cpu_qos), with the repo's venv and "
            ".env loaded. The script's own valuegen commands submit and wait on "
            "the real work; every step resumes from its artifacts, so a killed "
            "controller is just resubmitted."
        ),
    )
    p.add_argument("script", type=Path, help="bash script to run")
    p.add_argument("--time", default="3-00:00:00", help="walltime (default: 3 days)")
    p.add_argument("--mem", default="8G")
    p.add_argument("--dry-run", action="store_true", help="write the .sbatch, submit nothing")
    p.add_argument("--cluster", default=None, help="cluster YAML override")


def cmd_controller(args: argparse.Namespace) -> int:
    cluster = load_cluster(args.cluster)
    script = args.script.resolve()
    if not script.is_file():
        raise SystemExit(f"{args.script}: no such file")
    name = script.stem
    stage = Stage(
        name=name,
        tasks=[Task(key=name, command=f"bash {shlex.quote(str(script))}",
                    done=lambda: False)],
        time=args.time,
        mem=args.mem,
        controller=True,
    )
    orch = Orchestrator("controller", [stage], cluster, dry_run=args.dry_run)
    path = orch._write_script(stage, stage.tasks)
    job = orch._sbatch(path)
    if job is not None:
        print(f"  log: {orch.log_dir}/{name}_{job}.out")
    return 0 if (job is not None or args.dry_run) else 1


def _smoke_parser(sub) -> None:
    p = sub.add_parser(
        "smoke",
        help="submit the GPU-node environment check (and optionally the test suite) as a job",
        description=(
            "Submit tests/fixtures/cuda_smoke.sh as one 1-GPU job placed from "
            "cluster.yaml (gpu_partition/gpu_qos, account, gpu_type, exclude), "
            "with the repo's venv and .env loaded: torch CUDA init plus a real "
            "vLLM engine bringup. --tests then runs the whole pytest suite in "
            "the same job, which is how the GPU end-to-end tests get a GPU."
        ),
    )
    p.add_argument("--tests", action="store_true",
                   help="also run the full test suite (incl. GPU e2e) after the check")
    p.add_argument("--time", default=None,
                   help="walltime (default: 00:30:00, or 01:30:00 with --tests)")
    p.add_argument("--mem", default="48G")
    p.add_argument("--dry-run", action="store_true", help="write the .sbatch, submit nothing")
    p.add_argument("--cluster", default=None, help="cluster YAML override")


def cmd_smoke(args: argparse.Namespace) -> int:
    cluster = load_cluster(args.cluster)
    name = "smoke_tests" if args.tests else "cuda_smoke"
    command = f"bash {shlex.quote(str(cluster.repo / 'tests/fixtures/cuda_smoke.sh'))}"
    if args.tests:
        command += "\npython -B -m pytest -p no:cacheprovider -rs tests/"
    stage = Stage(
        name=name,
        tasks=[Task(key=name, command=command, done=lambda: False)],
        time=args.time or ("01:30:00" if args.tests else "00:30:00"),
        mem=args.mem,
        gpus=1,
    )
    orch = Orchestrator("smoke", [stage], cluster, dry_run=args.dry_run)
    path = orch._write_script(stage, stage.tasks)
    job = orch._sbatch(path)
    if job is not None:
        print(f"  log: {orch.log_dir}/{name}_{job}.out")
    return 0 if (job is not None or args.dry_run) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="valuegen",
        description="Measure and predict value generalization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _gt_parser(sub)
    _data_parser(sub)
    _predict_parser(sub)
    _analyze_parser(sub)
    _controller_parser(sub)
    _smoke_parser(sub)
    mv_cli.add_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "gt":
        return cmd_gt(args)
    if args.command == "data":
        return cmd_data(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "controller":
        return cmd_controller(args)
    if args.command == "smoke":
        return cmd_smoke(args)
    if args.command == "mv":
        return mv_cli.run(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
