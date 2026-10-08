"""Per-checkpoint worker, executed inside the ``mv eval`` SLURM tasks.

    python -m valuegen.multivalue.eval_worker run        -c CFG [--cluster C] --ckpt ID --suite S --base-url URL [--smoke]
    python -m valuegen.multivalue.eval_worker stage-base -c CFG [--cluster C]
    python -m valuegen.multivalue.eval_worker probe      -c CFG [--cluster C] --ckpt ID --base-url URL

``run`` evaluates one suite against the candidate the enclosing shell
already serves at ``--base-url`` (this process never starts a model) through
the shared runner: generate, grade from the persisted log, write rows and
the completion marker. ``stage-base`` stages the untrained base as a
servable export. ``probe`` checks the endpoint and makes one real call to
each enabled suite's grader.
"""

from __future__ import annotations

import argparse
import json
import sys

from valuegen.config import load_cluster
from valuegen.ground_truth import inspect_runner as IR
from valuegen.multivalue import eval_stage
from valuegen.multivalue.config import load_config
from valuegen.multivalue.evals import runner as R
from valuegen.multivalue.evals.base import EvalError
from valuegen.multivalue.layout import Layout, NotFrozenError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m valuegen.multivalue.eval_worker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {}
    for name in ("run", "stage-base", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--config", "-c", required=True)
        p.add_argument("--cluster", default=None)
        ps[name] = p
    for name in ("run", "probe"):
        ps[name].add_argument("--ckpt", required=True, help="candidate checkpoint id (served name)")
        ps[name].add_argument("--base-url", required=True, help="served candidate endpoint (http://host:port/v1)")
    ps["run"].add_argument("--suite", required=True)
    ps["run"].add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config, require_files=False)
        cluster = load_cluster(args.cluster)
        layout = Layout(cfg, cluster)
        if args.cmd == "stage-base":
            rec = eval_stage.stage_base_export(cfg, layout)
            print(json.dumps({"base_export_dir": str(layout.base_export_dir), "problems": rec.get("problems", [])}))
            return 0
        cand = eval_stage.find_candidate(cfg, cluster, layout, args.ckpt)
        if args.cmd == "probe":
            out = {"endpoint": IR.probe_endpoint(args.base_url, cand.served_name),
                   "graders": {s: R.grader_smoke(cfg.evals.grader(s)) for s in cfg.evals.enabled()}}
            print(json.dumps(out, indent=2, default=str))
            return 0
        return R.run_suite(cfg, layout, cand, args.suite, args.base_url, smoke=args.smoke)
    except (EvalError, NotFrozenError) as e:
        print(f"eval_worker {args.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
