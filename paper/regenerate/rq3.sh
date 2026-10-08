#!/usr/bin/env bash
# Rebuild RQ3's raw data (data/paper/rq3/): the persona geometry of the 266
# Values-in-the-Wild level-1 values on OLMo-3.1-32B-Instruct-SFT, the k=4
# taxonomy clustered from it, the layer-32 persona vectors that carry the
# taxonomy onto the constitution tenets, and the two external taxonomies'
# tenet labels (LitmusValues, VITW level 3).
#
#   valuegen controller paper/regenerate/rq3.sh --time 3-00:00:00
#
# Pairs are written by the neutral-SFT OLMo-3-7B (all-Gemini recipe), not by
# the 32B: the 32B only supplies activations, read at layer 32 (pinned, the
# depth-scaled counterpart of the 7B's layer 16; no sweep). The tenet pairs
# are the ones rq1.sh builds for the OLMo DPO arm (run rq1.sh first, or this
# builds them).
#
# Cost, from the original runs: the VITW pairs ~$60-90 of Gemini; 32B
# extraction ~5-8 min per value on 2 GPUs, so ~60 GPU-hours for the 266 VITW
# values and ~15 for the 66 tenets. LitmusValues labels: none with the shipped cache, else 66 short API calls.
# VITW-L3 labels and the clustering are CPU-only and deterministic.
#
# Not bit-exact: the Gemini-written pairs are sampled, so the grid, the
# clusters and the vectors match the paper's within that noise (rq3.py checks
# the published clustering and will report how far a rebuild moved). The
# LitmusValues labels used gpt-5.6-luna because the paper's model
# (claude-3-5-sonnet-20241022) is retired; scripts/taxonomy_labels.py falls
# back the same way. After this, run `paper/reproduce/rq3.py --bootstrap`:
# the stored scenario bootstrap only matches the published data.
set -euo pipefail

VIEWS="$(python -c 'from valuegen.config import load_cluster; print(load_cluster().finetune_root)')/model_views"

python scripts/build_model_view.py value-generalization/neutral-sft-v3-olmo3-7b --revision e93fa4d880af487009cb828b93665e88a428cc16
python scripts/build_model_view.py allenai/Olmo-3.1-32B-Instruct-SFT --revision 152782ecc41a86c5cbe3fb6afa68bd90934de48a --symlink

GEMINI=(--param trait_method=default --param generate_model=gemini-3.7-flash
        --param generate_provider=gemini --param generation_max_tokens=20000
        --param judge_model=gemini-3.5-flash-lite
        --param judge_scoring_protocol=gemini_fulltext_0_100_v1)

# Pairs, written by the 7B (artifact -ba00a75cf796 / -9e781b0f5a83 at the paper's paths).
valuegen data build --method default_llm --value-set vitw_l1_266 --model "$VIEWS/neutral-sft-v3-olmo3-7b" \
    "${GEMINI[@]}" --param experiment=vitw266_gemini_20260915 --param layer_start=16 --param layer_end=16
valuegen data build --method default_llm --value-set constitution_tenets_v3 --model "$VIEWS/neutral-sft-v3-olmo3-7b" \
    "${GEMINI[@]}" --param experiment=tenets69_gemini_20260827 --param layer_start=15 --param layer_end=15

# 32B activations on those 7B-written pairs (passed by path: the glob matches the
# one artifact just built) at the pinned layer: vectors (all layers) + the L32 grid.
PAIRS=data/elicitation/pairs/default_llm
valuegen predict -m persona --value-set vitw_l1_266 --model "$VIEWS/Olmo-3.1-32B-Instruct-SFT" \
    --data $PAIRS/vitw_l1_266/neutral-sft-v3-olmo3-7b-* --layer 32 --gpus 2 --build
valuegen predict -m persona --value-set constitution_tenets_v3 --model "$VIEWS/Olmo-3.1-32B-Instruct-SFT" \
    --data $PAIRS/constitution_tenets_v3/neutral-sft-v3-olmo3-7b-* --layer 32 --gpus 2 --build

# The comparison taxonomies' tenet labels.
python scripts/taxonomy_labels.py vitw
# LitmusValues: seed the completion cache with the paper's (data/paper/rq3/litmus_cache,
# from scripts/fetch_paper_data.py) so the labels replay exactly; delete the seeded
# files to re-query the model instead.
mkdir -p data/taxonomy_labels/cache
cp -n data/paper/rq3/litmus_cache/*.json data/taxonomy_labels/cache/
python scripts/taxonomy_labels.py litmus

# Grid, layer-32 vectors, k=4 clusters and labels into data/paper/rq3.
python scripts/export_paper_data.py rq3
