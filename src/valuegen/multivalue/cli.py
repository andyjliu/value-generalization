"""``valuegen mv …``: the multivalue command surface.

    valuegen mv sets    -c CFG --scheme {random,stratified} [--metric coverage]
                        [--store sentence] [--candidates N] [--bins B] [--seed S]
                        [--binning {quantile,width}] [--trim PCT]
                        [--freeze] [--force]
    valuegen mv metrics -c CFG
    valuegen mv data    -c CFG [--arm ID …] [--skip-token-audit]
    valuegen mv train   -c CFG [--dry-run|--no-wait] [--arm ID …] [--seed S …]
    valuegen mv train   -c CFG --record CKPT_ID     (called by the train job)
    valuegen mv eval    -c CFG [--dry-run|--no-wait] [--suite S …] [--arm ID …] [--all] [--smoke]
    valuegen mv eval    -c CFG --no-wait --after afterany:JOBID
    valuegen mv eval    -c CFG --build SUITE        (re)freeze a suite's inputs
    valuegen mv run     -c CFG [--wave W] [--delete-exports] [--dry-run] [--suite S …] [--arm ID …] [--seed S …]
    valuegen mv run     -c CFG --submit [...]       sbatch a self-resubmitting controller for the same
    valuegen mv analyze -c CFG
    valuegen mv import  -c CFG --from DIR [--prefix P] [--exclude ID …] [--allow-base-mismatch] [--force]
    valuegen mv status  -c CFG

``sets`` previews by default and prints the readouts every time; only
``--freeze`` writes ``sets.json`` and a frozen design is never replaced
without ``--force``. ``run`` alternates train and eval in waves of W
checkpoints under one polling controller (see :mod:`valuegen.multivalue.waves`);
``train`` and ``eval`` remain the one-off / dependency-chained verbs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from valuegen.config import load_cluster
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import sets as mvsets
from valuegen.multivalue.config import load_config
from valuegen.multivalue.embeddings import load_stores
from valuegen.multivalue.layout import Layout, NotFrozenError
from valuegen.multivalue.metrics import METRICS

VERBS = ("sets", "metrics", "data", "train", "eval", "run", "analyze", "import", "status")


def add_parser(sub: argparse._SubParsersAction) -> None:
    mv = sub.add_parser("mv", help="multivalue (k-value-set) experiments")
    mv_sub = mv.add_subparsers(dest="verb", required=True)
    ps = {}
    for verb, help_text in (
        ("sets", "sample / preview / freeze the k-value sets"),
        ("metrics", "coverage + tightness per arm per store -> metrics/metrics.csv"),
        ("data", "build per-arm training mixes"),
        ("train", "train + export one checkpoint per arm x seed (SLURM)"),
        ("eval", "run Inspect suites on every checkpoint (SLURM)"),
        ("run", "train -> eval in waves under one polling controller (SLURM)"),
        ("analyze", "outcomes, correlations, figures, REPORT.md"),
        ("import", "register externally trained arms"),
        ("status", "arm x stage completeness"),
    ):
        p = mv_sub.add_parser(verb, help=help_text)
        p.add_argument("--config", "-c", required=True, help="multivalue YAML")
        p.add_argument("--cluster", default=None, help="cluster YAML override")
        ps[verb] = p
    p = ps["sets"]
    p.add_argument("--scheme", choices=("random", "stratified"), default="stratified")
    p.add_argument("--metric", choices=METRICS, default="coverage")
    p.add_argument("--store", default=None, help="embedding store to stratify on (default: sentence)")
    p.add_argument("--candidates", type=int, default=mvsets.DEFAULT_CANDIDATES)
    p.add_argument("--bins", type=int, default=mvsets.DEFAULT_BINS)
    p.add_argument("--binning", choices=mvsets.BINNINGS, default="quantile",
                   help="quantile: equal pool mass per bin; width: equal-width bins over the metric's range")
    p.add_argument("--trim", type=float, default=0.0,
                   help="--binning width: percentile dropped from each end of the pool before cutting bins")
    p.add_argument("--seed", type=int, default=None, help="override config seed for the draw")
    p.add_argument("--freeze", action="store_true", help="write sets.json (else preview only)")
    p.add_argument("--force", action="store_true", help="replace an existing frozen sets.json")
    for verb in ("train", "eval"):
        ps[verb].add_argument("--dry-run", action="store_true")
        ps[verb].add_argument("--no-wait", action="store_true")
    ps["train"].add_argument("--arm", action="append", default=None, help="only these arm ids")
    ps["train"].add_argument("--seed", action="append", type=int, default=None, help="only these seeds")
    ps["train"].add_argument("--record", metavar="CKPT_ID", default=None,
                             help="verify a finished run and write its run_record.json (used by the job itself)")
    ps["eval"].add_argument("--suite", action="append", default=None, help="only these enabled suites")
    ps["eval"].add_argument("--arm", action="append", default=None, help="only these arm ids")
    ps["eval"].add_argument("--all", action="store_true", help="schedule trained runs without a verified export")
    ps["eval"].add_argument("--after", default=None, metavar="DEP",
                            help="with --no-wait: SLURM dependency for the eval array (e.g. afterany:JOBID); "
                                 "implies --all")
    ps["eval"].add_argument("--smoke", action="store_true",
                            help="base model (or --arm), 1 repeat, a few units per suite -> evals_smoke/")
    ps["eval"].add_argument("--build", default=None, metavar="SUITE", help="(re)freeze a suite's inputs and exit")
    p = ps["run"]
    p.add_argument("--dry-run", action="store_true", help="render every wave's sbatch + the driver; submit nothing")
    p.add_argument("--wave", type=int, default=None, metavar="W",
                   help="checkpoints (arm x seed) per wave (overrides schedule.wave; default: one wave)")
    p.add_argument("--delete-exports", action="store_true", default=None,
                   help="delete a checkpoint's export/ once every enabled suite scored it (overrides "
                        "schedule.delete_exports)")
    p.add_argument("--submit", action="store_true",
                   help="sbatch a self-resubmitting controller job that runs this command (poll-only; "
                        "cluster.yaml's controller_partition/cpu_partition)")
    p.add_argument("--suite", action="append", default=None, help="only these enabled suites")
    p.add_argument("--arm", action="append", default=None, help="only these arm ids")
    p.add_argument("--seed", action="append", type=int, default=None, help="only these seeds")
    ps["data"].add_argument("--arm", action="append", default=None, help="only these arm ids")
    ps["data"].add_argument("--skip-token-audit", action="store_true",
                            help="do not tokenize (no TRL-drop / geometry record; tests only)")
    ps["import"].add_argument("--from", dest="from_dir", required=True,
                              help="an RQ3 experiment dir (config.yaml / out/runs.jsonl) or its out root")
    ps["import"].add_argument("--prefix", default=None, help="prefix imported arm ids (collision avoidance)")
    ps["import"].add_argument("--exclude", action="append", default=None, metavar="ID",
                              help="skip these source arm ids (pre-prefix), e.g. a driver's k=49 controls")
    ps["import"].add_argument("--allow-base-mismatch", action="store_true",
                              help="import runs trained on another base/revision/chat format (analysis only)")
    ps["import"].add_argument("--force", action="store_true", help="replace an existing import of the same id")


def _load(args):
    cfg = load_config(args.config)
    cluster = load_cluster(args.cluster)
    return cfg, cluster, Layout(cfg, cluster)


def cmd_sets(args) -> int:
    cfg, cluster, layout = _load(args)
    universe, _ = mvdata.resolve_universe(cfg, cluster)
    external, _ = mvdata.load_external(cfg)
    stores = load_stores(cfg, universe, external)
    record, readouts = mvsets.build_sets(
        cfg, universe, stores, scheme=args.scheme, metric=args.metric, store=args.store,
        n_candidates=args.candidates, bins=args.bins, seed=args.seed, binning=args.binning, trim=args.trim,
    )
    print(mvsets.format_readouts(record, readouts))
    layout.preview_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{record['scheme']}_{record['metric']}_{record['store']}_s{record['seed']}"
    if record.get("binning"):
        tag += f"_{record['binning']}{record['bins']}_trim{record['trim']:g}"
    (layout.preview_dir / f"{tag}.json").write_text(
        json.dumps({"record": record, "readouts": readouts}, indent=2, ensure_ascii=False) + "\n")
    print(f"\npreview written to {layout.preview_dir / (tag + '.json')}")
    if layout.frozen:
        prev = mvsets.load_sets(layout.sets_path)
        same = prev["arms_sha256"] == record["arms_sha256"]
        print(f"frozen sets.json exists ({'identical design' if same else 'DIFFERENT design'}): {layout.sets_path}")
    if not args.freeze:
        print("not frozen (pass --freeze to write sets.json)")
        return 0
    mvsets.write_sets(layout.sets_path, record, force=args.force)
    print(f"frozen: {layout.sets_path}\nexp_id: {layout.exp_id}")
    return 0


def cmd_metrics(args) -> int:
    from valuegen.multivalue.metrics_stage import write_metrics

    cfg, cluster, layout = _load(args)
    df = write_metrics(cfg, cluster, layout)
    print(df.to_string(index=False, max_rows=60))
    print(f"\nwrote {layout.metrics_csv}")
    return 0


def cmd_data(args) -> int:
    cfg, cluster, layout = _load(args)
    arms = mvimports.all_arms(layout)
    manifest = mvdata.build_mixes(cfg, cluster, layout, arms, token_audit_enabled=not args.skip_token_audit,
                                  only=args.arm)
    n = len(manifest["arms"])
    print(f"\n{n} mixes under {layout.mixes_dir} (exp_id {layout.exp_id})")
    for aid, a in manifest["arms"].items():
        if a["warnings"]:
            print(f"  WARN {aid}: {'; '.join(a['warnings'])}")
    print(f"manifest: {layout.mixes_dir / 'manifest.json'}")
    return 0


def cmd_import(args) -> int:
    cfg, cluster, layout = _load(args)
    if layout.frozen:
        universe = list(mvsets.load_sets(layout.sets_path)["universe"])
    else:
        universe, _ = mvdata.resolve_universe(cfg, cluster)
    res = mvimports.import_experiment(cfg, cluster, layout, args.from_dir, universe=universe,
                                      allow_base_mismatch=args.allow_base_mismatch, force=args.force,
                                      prefix=args.prefix, exclude=args.exclude)
    skipped = f", {len(res['skipped'])} excluded ({', '.join(res['skipped'])})" if res["skipped"] else ""
    print(f"{res['experiment']}: {len(res['added'])} arms imported, {len(res['unchanged'])} already registered"
          f"{skipped} -> {mvimports.imports_path(layout)} ({len(res['record']['arms'])} imported arms total)")
    return 0


def cmd_train(args) -> int:
    from valuegen.multivalue import train as mvtrain

    cfg, cluster, layout = _load(args)
    arms = mvimports.all_arms(layout)
    try:
        if args.record:
            return mvtrain.record_main(cfg, cluster, layout, arms, args.record)
        return mvtrain.train(cfg, cluster, layout, arms, dry_run=args.dry_run, no_wait=args.no_wait,
                             only=args.arm, seeds=args.seed)
    except (mvtrain.TrainError, NotFrozenError) as e:
        print(f"valuegen mv train: {e}", file=sys.stderr)
        return 1


def cmd_eval(args) -> int:
    from valuegen.multivalue import eval_stage
    from valuegen.multivalue.evals.base import EvalError

    cfg, cluster, layout = _load(args)
    try:
        if args.build:
            return eval_stage.build_inputs_main(cfg, layout, args.build)
        arms = mvimports.all_arms(layout)
        return eval_stage.evaluate(cfg, cluster, layout, arms, suites=args.suite, dry_run=args.dry_run,
                               no_wait=args.no_wait, only=args.arm, after=args.after, include_unverified=args.all,
                               smoke=args.smoke)
    except (EvalError, NotFrozenError) as e:
        print(f"valuegen mv eval: {e}", file=sys.stderr)
        return 1


def cmd_run(args) -> int:
    from valuegen.multivalue import train as mvtrain
    from valuegen.multivalue import waves
    from valuegen.multivalue.evals.base import EvalError

    cfg, cluster, layout = _load(args)
    try:
        arms = mvimports.all_arms(layout)
        if args.submit:
            return waves.submit_driver(cfg, cluster, layout, wave=args.wave, delete_exports=args.delete_exports,
                                       suites=args.suite, only=args.arm, seeds=args.seed, dry_run=args.dry_run)
        return waves.run_waves(cfg, cluster, layout, arms, wave=args.wave, delete_exports=args.delete_exports,
                               suites=args.suite, only=args.arm, seeds=args.seed, dry_run=args.dry_run)
    except (mvtrain.TrainError, EvalError, waves.WaveError, NotFrozenError) as e:
        print(f"valuegen mv run: {e}", file=sys.stderr)
        return 1


def cmd_analyze(args) -> int:
    from valuegen.multivalue import analyze as mvanalyze

    cfg, cluster, layout = _load(args)
    try:
        arms = mvimports.all_arms(layout)
        mvanalyze.analyze(cfg, cluster, layout, arms)
        return 0
    except (mvanalyze.AnalyzeError, NotFrozenError) as e:
        print(f"valuegen mv analyze: {e}", file=sys.stderr)
        return 1


def cmd_status(args) -> int:
    """Status intentionally skips embedding-file checks and writes nothing."""
    from valuegen.multivalue import status as mvstatus

    cfg = load_config(args.config, require_files=False)
    cluster = load_cluster(args.cluster)
    layout = Layout(cfg, cluster)
    if not layout.frozen:
        print(f"multivalue {cfg.name}: sets=pending ({layout.sets_path})")
        print(f"run `valuegen mv sets -c {cfg.path} --freeze` first")
        return 0
    arms = mvimports.all_arms(layout)
    mvstatus.status(cfg, cluster, layout, arms)
    return 0


def run(args) -> int:
    if args.verb == "sets":
        return cmd_sets(args)
    if args.verb == "metrics":
        return cmd_metrics(args)
    if args.verb == "data":
        return cmd_data(args)
    if args.verb == "import":
        return cmd_import(args)
    if args.verb == "train":
        return cmd_train(args)
    if args.verb == "eval":
        return cmd_eval(args)
    if args.verb == "run":
        return cmd_run(args)
    if args.verb == "analyze":
        return cmd_analyze(args)
    if args.verb == "status":
        return cmd_status(args)
    raise AssertionError(f"unhandled mv verb {args.verb!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="valuegen mv")
    sub = parser.add_subparsers(dest="command", required=True)
    add_parser(sub)
    args = parser.parse_args(["mv", *(argv if argv is not None else sys.argv[1:])])
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
