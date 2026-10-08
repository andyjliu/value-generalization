"""Write the released DPO pairs where a DPO config's intervention expects them.

The constitution-v3 DPO pairs (49 tenets x 4,000 pairs) are built by
label_subset: a Qwen3.6-27B oracle labels eight preference pools, then a
min-cost flow assigns the pairs to tenets. The result is released on HF as
value-generalization/constitution-v3-dpo, one table with a `tenet` column.
This script writes it back out as the per-tenet `dataset.jsonl` files under
the config's intervention datasets/ dir, in label_subset's own row format
({scenario_id, prompt, chosen, rejected, score, source}), so nothing is
relabeled. It:

  1. claims the intervention artifact (its record + manifest);
  2. downloads the HF set at REVISION and writes one `dataset.jsonl` per
     tenet in the config, 4,000 rows each, in the released row order;
  3. records the import in `hf_import.json` next to datasets/. The multivalue
     code reads the revision from there: it enters each row's identity, and
     so the order in which `valuegen mv data` draws an arm's rows.

`valuegen mv` (the paper's RQ2, paper/regenerate/rq2.sh) reads these files as
its `source.artifact`. Idempotent: a tenet whose file already has 4,000 rows
is left untouched.

    .venvs/core/bin/python scripts/prepare_dpo_data.py \
        -c configs/experiments/ground_truth/const_v3_dpo_full49.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from valuegen.config import intervention_id, load_cluster, load_experiment
from valuegen.ground_truth import interventions

HF_DATASET = "value-generalization/constitution-v3-dpo"
REVISION = "f286661b4af2c394cd67d97ccc47ca2c7d32ba39"
EXPECT_ROWS = 4000
# label_subset's dataset.jsonl columns, in its order.
COLUMNS = ("scenario_id", "prompt", "chosen", "rejected", "score", "source")


def materialize_datasets(cfg: dict, cluster) -> None:
    root = interventions.datasets_dir(cfg, cluster)
    values = list(cfg["intervention"]["values"])
    want = set(values)

    # Which tenets still need writing? (idempotent -- a full 4,000-row file is
    # left untouched.)
    todo = []
    for v in values:
        dst = root / v / "dataset.jsonl"
        if dst.is_file() and sum(1 for _ in open(dst)) == EXPECT_ROWS:
            continue
        todo.append(v)
    if not todo:
        print(f"  all {len(values)} datasets already materialized under {root}")
        return

    print(f"  loading {HF_DATASET}@{REVISION[:12]} (need {len(todo)}/{len(values)} tenets) ...")
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train", revision=REVISION)
    missing = want - set(ds.unique("tenet"))
    if missing:
        print(f"  ERROR: {len(missing)} config tenets absent from HF: "
              f"{sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # One pass, bucketed by tenet -> jsonl. Only rows for a needed tenet are
    # written; a tenet already on disk is skipped even if in `want`.
    todo_set = set(todo)
    writers: dict[str, object] = {}
    counts: Counter[str] = Counter()
    try:
        for row in ds:
            t = row["tenet"]
            if t not in todo_set:
                continue
            f = writers.get(t)
            if f is None:
                (root / t).mkdir(parents=True, exist_ok=True)
                f = writers[t] = open(root / t / "dataset.jsonl", "w", encoding="utf-8")
            f.write(json.dumps({c: row[c] for c in COLUMNS}, ensure_ascii=False) + "\n")
            counts[t] += 1
    finally:
        for f in writers.values():
            f.close()

    bad = {t: n for t, n in counts.items() if n != EXPECT_ROWS}
    if bad or set(counts) != todo_set:
        print(f"  ERROR: row-count / coverage mismatch. bad={bad} "
              f"wrote={len(counts)} expected={len(todo_set)}", file=sys.stderr)
        sys.exit(1)
    print(f"  wrote {len(counts)} tenets x {EXPECT_ROWS} rows under {root}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    cluster = load_cluster(None)
    if cfg["intervention"]["method"] != "label_subset":
        print(f"ERROR: {args.config} is not a label_subset (DPO) config", file=sys.stderr)
        return 1
    print(f"config       {args.config}\nintervention {intervention_id(cfg)}")

    # 0. claim the intervention artifact dir FIRST: the datasets written in
    #    step 1 live inside it, and a claim refuses a non-empty unclaimed dir.
    print("\n[0] claim intervention artifact")
    interventions.claim(cfg, cluster)

    print("\n[1] per-tenet datasets (imported from HF)")
    materialize_datasets(cfg, cluster)

    print("\n[2] import record")
    record = interventions.datasets_dir(cfg, cluster).parent / "hf_import.json"
    record.write_text(json.dumps({"dataset": HF_DATASET, "revision": REVISION}, indent=1) + "\n")
    print(f"  record {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
