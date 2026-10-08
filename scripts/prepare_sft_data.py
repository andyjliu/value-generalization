"""Prepare an SFT-arm config (const_v3_sft_*) for `valuegen gt run`.

The constitution-v3 SFT arm trains one full-FT SFT per tenet off the *raw*
base, on data IMPORTED from HF (value-generalization/constitution-v3-sft), not
generated. `valuegen gt run` uses the model_spec_aft driver, whose
build_data() only *skips* generation when every per-tenet dataset already
exists on disk -- so this script must run first. It:

  1. downloads the HF set and writes one `dataset.jsonl` per tenet in the
     config (TRL pair format {prompt:[user], chosen:[assistant]}, 5,000 rows
     each) under this config's intervention datasets/ dir. The HF rows are
     already in that exact shape (prompt/chosen message lists + a `tenet`
     column), so they are written through verbatim.
  2. claims the intervention artifact via interventions.build_data(), which
     -- with all datasets present -- runs model_spec_aft.build_data() with
     generation fully suppressed (it only writes the inert specs/*.txt).
  3. claims the GT run dir and the shared base_evals store. Their first-turn
     cache.json is seeded by the eval tasks themselves, from the scenarios
     dir (fill it once with scripts/fetch_eval_inputs.py).

Idempotent: every step is skipped when its output already exists. Run once
per config (they share nothing but the HF download cache):

    .venvs/core/bin/python scripts/prepare_sft_data.py \
        -c configs/experiments/ground_truth/const_v3_sft_full66.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from valuegen.config import (
    canonical_experiment,
    ensure_config_record,
    gt_id,
    intervention_id,
    load_cluster,
    load_experiment,
)
from valuegen.ground_truth import common, interventions
from valuegen.ground_truth.evals import conflictscope as cs_eval

HF_DATASET = "value-generalization/constitution-v3-sft"
EXPECT_ROWS = 5000


def materialize_datasets(cfg: dict, cluster) -> None:
    root = interventions.datasets_dir(cfg, cluster)
    values = list(cfg["intervention"]["values"])
    want = set(values)

    # Which tenets still need writing? (idempotent -- a full 5,000-row file is
    # left untouched.)
    todo = []
    for v in values:
        dst = interventions.dataset_file(root / v)
        if dst.is_file() and sum(1 for _ in open(dst)) == EXPECT_ROWS:
            continue
        todo.append(v)
    if not todo:
        print(f"  all {len(values)} datasets already materialized under {root}")
        return

    print(f"  loading {HF_DATASET} (need {len(todo)}/{len(values)} tenets) ...")
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    ds_tenets = set(ds.unique("tenet"))
    missing = want - ds_tenets
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
                f = writers[t] = open(interventions.dataset_file(root / t), "w")
            f.write(json.dumps(
                {"prompt": row["prompt"], "chosen": row["chosen"]},
                ensure_ascii=False,
            ) + "\n")
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
    store_dir = cs_eval.base_evals_dir(cfg, cluster)
    print(f"config       {args.config}\nintervention {intervention_id(cfg)}\n"
          f"gt           {gt_id(cfg)}\n"
          f"base store   {cs_eval.base_eval_id(cfg, cluster)}")

    # 0. claim the intervention artifact dir FIRST: ensure_config_record
    #    refuses a non-empty dir without its identity record, and the datasets
    #    written in step 1 live inside that dir (hit 2026-09-05 on first run).
    print("\n[0] claim intervention artifact")
    interventions.claim(cfg, cluster)

    # 1. per-tenet datasets from HF (must precede build_data so no generation).
    print("\n[1] per-tenet datasets (imported from HF)")
    materialize_datasets(cfg, cluster)

    # 2. intervention artifact: claim + model_spec_aft.build_data (generation
    #    suppressed because every dataset exists; only inert specs are written).
    print("\n[2] intervention artifact")
    interventions.build_data(cfg, cluster)
    root = interventions.datasets_dir(cfg, cluster)
    print(f"  claimed {root.parent}")

    # 3. GT run dir + eval cache.
    print("\n[3] GT run dir")
    ensure_config_record(common.gt_record_path(cfg, cluster), gt_id(cfg),
                         canonical_experiment(cfg))
    print(f"  record {common.gt_record_path(cfg, cluster)}")

    # 4. shared base-eval store (claimed idempotently: the DPO arm shares it).
    print("\n[4] base-eval store")
    cs_eval.claim_base_evals(cfg, cluster)
    print(f"  record {store_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
