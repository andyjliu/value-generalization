"""Array-task body for SLURM pair generation.

Each task rebuilds its value from the *persisted* resolved config
(``data_config.yaml``) rather than re-deriving it from CLI flags, so the
artifact identity the controller hashed is exactly what the worker builds.

Usage (emitted by ``registry.generation_stage``)::

    python -m valuegen.elicitation.worker --config .../data_config.yaml --value no_white_lies
"""

from __future__ import annotations

import argparse

import yaml

from valuegen.config import load_cluster
from valuegen.elicitation import registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one value of a pair artifact")
    parser.add_argument("--config", required=True, help="artifact data_config.yaml")
    parser.add_argument("--value", required=True, action="append",
                        help="value(s) to build (repeatable)")
    parser.add_argument("--cluster", default=None, help="cluster YAML override")
    args = parser.parse_args(argv)

    record = yaml.safe_load(open(args.config))
    cfg = record["resolved_config"]
    cluster = load_cluster(args.cluster)
    # Workers never finalize: the controller writes the manifest once every
    # value is present, so a straggler task cannot race a half-built manifest.
    registry.build_inline(cfg, cluster, only_values=args.value, finalize=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
