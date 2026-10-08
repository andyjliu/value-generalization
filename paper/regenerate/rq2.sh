#!/usr/bin/env bash
# Rebuild RQ2's raw data (data/paper/rq2/): 64 DPO models of the neutral-SFT
# Qwen3-8B, each trained on a 6-value set drawn evenly over the range of
# persona tightness, and the prefill-robustness eval of each
# (configs/experiments/multivalue/rq3_ew64_qwen8b.yaml; `rq3_` is the
# project's earlier numbering for this experiment).
#
#   valuegen controller paper/regenerate/rq2.sh --time 7-00:00:00
#
# Launch from a shell with HF_TOKEN set and HF_HOME on a large disk. Every
# step resumes from what is on disk, so a killed controller is just
# resubmitted.
#
# What is rebuilt and what is reused. Rebuilt: the 64 training mixes, the 64
# checkpoints and their judged prefill rows. Reused from data/paper/rq2
# (scripts/fetch_paper_data.py), because they define the experiment rather
# than measure anything:
#   * the frozen design (sets.json), drawn before launch from the persona
#     vectors under the preregistered rule;
#   * the embedding stores the tightness of each set is computed from. The
#     persona vectors of the 49 constitution values are the Qwen DPO arm's of
#     RQ1 (layer 18; rq1.sh rebuilds them). The Model-Spec stores only feed
#     the `coverage` columns, which the paper does not report;
#   * the prefill suite's frozen inputs (the injected turns the neutral SFT
#     wrote for each scenario). Without them the suite rebuilds its own, which
#     is a different set of prefills.
#
# Cost, as the config is written: 64 full fine-tunes on 8 GPUs each (FSDP,
# 750 steps, 1.5 h walltime cap), two at a time, in waves of 8; after each
# wave a prefill eval of its checkpoints against a Qwen3.6-27B judge served in
# the eval job on 4 GPUs (~30 min per job of 4 checkpoints in the original
# run). The judge shares a node with the candidates it grades, so cluster.yaml
# must set slurm.gpus_per_node (8 in the original run: 4 for the judge, 4
# candidates); `mv run` stops at the first eval wave without it. With fewer
# than 16 GPUs, set train.concurrent to 1 in the config: train.concurrent,
# train.resources and schedule are not in the experiment ID.
# Checkpoints are deleted once scored (schedule.delete_exports), so peak disk
# on finetune_root is one wave of 8.
#
# Not bit-exact: each arm's 12,000 pairs are drawn in an order that hashes the
# source revision, so they are not the paper's rows; training and the
# candidates' sampling (temperature 0.7) are stochastic. rq2.py checks the
# published numbers and will report how far a rebuild moved.
set -euo pipefail

CFG=configs/experiments/multivalue/rq3_ew64_qwen8b.yaml
PAPER=data/paper/rq2
EXP=multivalue/rq3_ew64_qwen8b
EXP_ID=rq3_ew64_qwen8b-9b66a6dd75a9

# Evaluation inputs: scenarios + the paper's first-turn cache (hash-checked).
python scripts/fetch_eval_inputs.py

# The 49 tenets' DPO pairs, imported from HF (no relabeling).
python scripts/prepare_dpo_data.py -c configs/experiments/ground_truth/const_v3_dpo_full49.yaml

# Embedding stores and the frozen design, where the config looks for them.
mkdir -p data/embeddings "data/$EXP"
cp -rn "$PAPER/embeddings/." data/embeddings/
cp -n "$PAPER/$EXP/sets.json" "data/$EXP/sets.json"

# The prefill suite's frozen inputs. The shipped record names the first-turn
# cache at its path on the original cluster; point it at the same file here
# (fetch_eval_inputs.py's, same sha256) so the suite takes the inputs as built.
INPUTS="data/$EXP/$EXP_ID/eval_inputs/prefill/inputs.json"
if [[ ! -f "$INPUTS" ]]; then
    mkdir -p "$(dirname "$INPUTS")"
    python - "$PAPER/$EXP/$EXP_ID/eval_inputs/prefill/inputs.json" "$INPUTS" <<'EOF'
import hashlib, json, sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:])
cache = "data/scenarios/const_v3_cs/cache.json"
rec = json.loads(src.read_text(encoding="utf-8"))
if hashlib.sha256(Path(cache).read_bytes()).hexdigest() != rec["opening_cache_sha256"]:
    sys.exit(f"{cache} is not the first-turn cache {src} was built from")
rec["params"]["opening_cache"] = rec["opening_cache"] = cache
dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
EOF
fi

# Tightness of every set, then each arm's 12,000-pair training mix.
valuegen mv metrics -c "$CFG"
valuegen mv data -c "$CFG"

# Train + evaluate in waves (polls until every checkpoint is scored), then the
# per-arm prefill metrics and their correlations with tightness.
valuegen mv run -c "$CFG"
valuegen mv analyze -c "$CFG"

# Into data/paper/rq2: the judged rows as one compact table, plus the small
# records `mv analyze` reads beside them.
python scripts/export_paper_data.py rq2 "data/$EXP/$EXP_ID"
cp "data/$EXP/metrics/metrics.csv" "$PAPER/$EXP/metrics/metrics.csv"
cp "data/$EXP/$EXP_ID/config.yaml" "$PAPER/$EXP/$EXP_ID/config.yaml"
(cd "data/$EXP/$EXP_ID" && find evals -name candidates.json -o -name COMPLETE.json) |
    while read -r f; do
        mkdir -p "$PAPER/$EXP/$EXP_ID/$(dirname "$f")"
        cp "data/$EXP/$EXP_ID/$f" "$PAPER/$EXP/$EXP_ID/$f"
    done
