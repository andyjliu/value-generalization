#!/usr/bin/env bash
# Rebuild the paper's ground truth: the four 7-8B steerability matrices
# (data/gt/<gt_id>/matrices/) and their per-scenario evals
# (data/paper/evals/<gt_id>.parquet), the inputs RQ1 and RQ3 score against,
# then the same four matrices at 30B scale (Qwen3-30B-A3B, Olmo-3 32B).
#
#   valuegen controller paper/regenerate/gt.sh --time 7-00:00:00
#
# The 30B-scale section is last and has its own hardware needs (see there).
# SKIP_BIGMODELS=1 stops before it; with it, ask for --time 21-00:00:00.
#
# Arms: DPO on the neutral-SFT Qwen3-8B / OLMo-3-7B (49 tenets x 4,000 pairs),
# SFT on the pretrained bases (66 tenets x 5,000 rows, imported from HF).
# Every step resumes from what is on disk, so a killed controller is just
# resubmitted.
#
# Cost (not measured end to end; the paper's runs were spread over weeks):
# per arm, one full fine-tune per tenet (49 or 66) plus a ConflictScope eval
# of each (11,188 scenarios) against a Qwen3.6-27B judge served on 8 GPUs.
# The DPO arms also label all eight preference pools with that judge as an
# oracle before training (label_subset); that labeling was the slowest stage
# of the original runs. Budget on the order of thousands of GPU-hours.
#
# Not bit-exact: sampling (temperature 1.0) and the judge are stochastic, so
# rebuilt matrices match the shipped ones within eval-sampling noise (the RQ3
# scenario bootstrap gives its size). `gt run` writes the matrices in place
# over the shipped ones under data/gt; `git diff --stat data/gt` shows what
# moved and `git checkout data/gt` restores the published matrices.
set -euo pipefail

# Evaluation inputs: scenarios + the paper's first-turn cache (hash-checked).
python scripts/fetch_eval_inputs.py

# DPO preference pools that are built locally (the HF sets are read directly).
python scripts/build_ca_wf_pools.py
python scripts/build_ca_wf_mt_pools.py

# SFT arms train on data imported from HF; write it where gt run expects it.
python scripts/prepare_sft_data.py -c configs/experiments/ground_truth/const_v3_sft_full66.yaml
python scripts/prepare_sft_data.py -c configs/experiments/ground_truth/const_v3_sft_full66_olmo3.yaml

# Train + evaluate every arm (each polls until its matrices are written).
valuegen gt run -c configs/experiments/ground_truth/const_v3_dpo_full49_olmo3.yaml
valuegen gt run -c configs/experiments/ground_truth/const_v3_dpo_full49.yaml
valuegen gt run -c configs/experiments/ground_truth/const_v3_sft_full66_olmo3.yaml
valuegen gt run -c configs/experiments/ground_truth/const_v3_sft_full66.yaml

# Compact per-scenario evals. The OLMo DPO run comes first: its base CSV fixes
# the scenario order of evals/scenarios.parquet, which the RQ3 bootstrap's
# resamples depend on.
python scripts/export_paper_data.py evals \
    data/gt/const_v3_dpo_full49_olmo3-caef4772a4ab \
    data/gt/const_v3_dpo_full49-6b356df7db0a \
    data/gt/const_v3_sft_full66_olmo3-bb17778645f9 \
    data/gt/const_v3_sft_full66-d9607bca37b2

# ── 30B scale: Qwen3-30B-A3B and Olmo-3 32B ───────────────────────────────────
# Same tenets, data, hyperparameters and ConflictScope eval as above; each
# config's header lists what model size forces. The DPO arms train on the 49
# datasets built above (the label store is keyed by the oracle block, not the
# trained model, so nothing is relabeled).
# Hardware, as the configs are written:
#   * training is 8-GPU FSDP2 per tenet (configs/fsdp/), ~20 min each;
#   * a candidate is evaluated in-process on ONE GPU, which must hold a ~65 GB
#     bf16 model. CONFLICTSCOPE_GPU_MEMORY_GB tells the conflictscope fork the
#     card size (it assumes 48 GB otherwise and wants more GPUs than the stage
#     requests); sbatch passes the submitting environment to its jobs, so the
#     export below reaches every eval task. CHANGE IT to your GPUs' memory. On
#     48 GB cards, raise evaluation.resources.gpus in the configs instead.
#   * disk on finetune_root: runs go in waves of 8 tenets (schedule.wave) and
#     delete each wave's checkpoints once scored. A wave needs ~65 GB per
#     tenet plus 130-260 GB of in-flight trainer saves; lower schedule.wave
#     (not hashed) if you have less.
# Each run is resumable; to inspect a pilot first, launch one by hand with
# `--stop-after-wave 1` (see its header).
[[ -n "${SKIP_BIGMODELS:-}" ]] && exit 0
export CONFLICTSCOPE_GPU_MEMORY_GB="${CONFLICTSCOPE_GPU_MEMORY_GB:-256}"

python scripts/prepare_sft_data.py -c configs/experiments/ground_truth/const_v3_sft_full66_olmo3_32b.yaml
python scripts/prepare_sft_data.py -c configs/experiments/ground_truth/const_v3_sft_full66_qwen3_30b_a3b.yaml

valuegen gt run -c configs/experiments/ground_truth/const_v3_dpo_full49_olmo3_32b.yaml
valuegen gt run -c configs/experiments/ground_truth/const_v3_dpo_full49_qwen3_30b_a3b.yaml
valuegen gt run -c configs/experiments/ground_truth/const_v3_sft_full66_olmo3_32b.yaml
valuegen gt run -c configs/experiments/ground_truth/const_v3_sft_full66_qwen3_30b_a3b.yaml

python scripts/export_paper_data.py evals \
    data/gt/const_v3_dpo_full49_olmo3_32b-66f7857035ca \
    data/gt/const_v3_dpo_full49_qwen3_30b_a3b-54fbfe3f9393 \
    data/gt/const_v3_sft_full66_olmo3_32b-7ec026a23d2a \
    data/gt/const_v3_sft_full66_qwen3_30b_a3b-3baa45ab00cb
