#!/usr/bin/env bash
# Rebuild RQ1's predictor grids (data/paper/rq1/grids/<arm>/<predictor>.npy):
# five predictors x eight arms (four at 7-8B, four at 30-32B), each built on the model that arm's training
# starts from: the neutral-SFT checkpoint for the DPO arms, the pretrained
# base for the SFT arms.
#
#   valuegen controller paper/regenerate/rq1.sh --time 5-00:00:00
#
# The last section repeats the predictors at 30B scale (Qwen3-30B-A3B, Olmo-3
# 32B); it needs far more disk and time (see there). SKIP_BIGMODELS=1 stops
# before it; with it, ask for --time 14-00:00:00.
#
# Predictors:
#   Persona       persona vectors (response-average activation difference)
#   Gradient      grad_proj, DPO-loss gradient at init, projected at a layer
#   Weight        weight_steer, LoRA task vectors (trained in the ws-train env)
#   Behavior-Embd the persona responses, embedded with all-mpnet-base-v2
#   Description-Embd  the value descriptions, embedded with all-mpnet-base-v2
#
# The persona pairs are generated and judged by Gemini (the "all-Gemini"
# recipe): gemini-3.7-flash writes the trait data, gemini-3.5-flash-lite
# judges. The neutral-SFT persona layer comes from a Gemini-judged steering
# sweep on held-out questions (it never sees the GT): the paper's sweep picked
# 18 (Qwen) and 16 (OLMo). Gradient reuses that layer; the base-model arms
# pin the same layers and read their own self-generated pairs under the
# urial0 scaffold.
#
# Cost, from the original runs: per neutral-SFT model ~55-60 GPU-hours and
# ~$15-18 of Gemini for all predictors (weight_steer dominates: 132 LoRA
# trains, ~1.8 TB of task-vector cache); the base-model arms are similar.
# Description-Embd and Behavior-Embd take minutes. Needs ws-train
# (scripts/setup_env.sh --ws-train) for weight_steer.
#
# Not bit-exact: the Gemini trait data, pairs and sweep judgments are
# sampled, so grids match the paper's within that noise (the sweep may also
# land on a neighboring layer). Data artifact IDs hash the --model path, so
# they equal the paper's (-5a2708f91afb, -9e781b0f5a83, -d16065ac5dae,
# -f22fe71e8417) only when finetune_root is the paper's path; Description-
# Embd is now built on the 66-tenet v3 descriptions (the paper's grid came
# from the 69-tenet v2 set; on the shared 66 they agree to 1e-6).
set -euo pipefail

FT="$(python -c 'from valuegen.config import load_cluster; print(load_cluster().finetune_root)')"
VIEWS="$FT/model_views"

# Revision-pinned model views (the --model paths every artifact records).
python scripts/build_model_view.py value-generalization/neutral-sft-v3-qwen3-8b --revision f1550fe714e01e2116cc4fdf3a1d510f8fd379c7
python scripts/build_model_view.py value-generalization/neutral-sft-v3-olmo3-7b --revision e93fa4d880af487009cb828b93665e88a428cc16
python scripts/build_model_view.py Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --symlink
python scripts/build_model_view.py allenai/Olmo-3-1025-7B --revision a81bae42db3975be1671e27b9c9a56da1a9f980f --symlink

# The all-Gemini pairs recipe (data-method params; they enter the artifact ID).
PAIRS=(--param experiment=tenets69_gemini_20260827 --param trait_method=default
       --param generate_model=gemini-3.7-flash --param generate_provider=gemini
       --param generation_max_tokens=20000 --param judge_model=gemini-3.5-flash-lite
       --param judge_scoring_protocol=gemini_fulltext_0_100_v1
       --param layer_start=15 --param layer_end=15)
BASE_PAIRS=("${PAIRS[@]}" --param chat_template_family=urial0)
# The persona layer sweep: Gemini-generated held-out questions, Gemini judge.
SWEEP=(--predictor-param judge_model=gemini-3.5-flash-lite --predictor-param judge_eval_type=0_100_text
       --predictor-param layer_step=2 --predictor-param generate_model=gemini-3.7-flash
       --predictor-param generate_provider=gemini --predictor-param generation_max_tokens=20000
       --predictor-param trait_method=default)
V3=(--value-set constitution_tenets_v3 --data default_llm)

# ── DPO arms: neutral-SFT models; persona sweeps the layer, grad_proj reuses it ──
valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/neutral-sft-v3-qwen3-8b" "${PAIRS[@]}" "${SWEEP[@]}" --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/neutral-sft-v3-qwen3-8b" "${PAIRS[@]}" --build
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/neutral-sft-v3-qwen3-8b" "${PAIRS[@]}" --build

valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/neutral-sft-v3-olmo3-7b" "${PAIRS[@]}" "${SWEEP[@]}" --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/neutral-sft-v3-olmo3-7b" "${PAIRS[@]}" --predictor-param template=native --build
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/neutral-sft-v3-olmo3-7b" "${PAIRS[@]}" --build

# ── SFT arms: pretrained bases, self-generated urial0 pairs, layers pinned ─────
valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/Qwen3-8B-Base" "${BASE_PAIRS[@]}" --predictor-param template=urial0 --layer 18 --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/Qwen3-8B-Base" "${BASE_PAIRS[@]}" --predictor-param template=urial0 --layer 18 --build
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/Qwen3-8B-Base" "${BASE_PAIRS[@]}" --predictor-param scaffold=urial0 --build

valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/Olmo-3-1025-7B" "${BASE_PAIRS[@]}" --predictor-param template=urial0 --layer 16 --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/Olmo-3-1025-7B" "${BASE_PAIRS[@]}" --predictor-param template=urial0 --layer 16 --build
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/Olmo-3-1025-7B" "${BASE_PAIRS[@]}" --predictor-param scaffold=urial0 --build

# ── Model-free text baselines ──────────────────────────────────────────────────
valuegen predict -m sentence_emb --value-set constitution_tenets_v3 --model all-mpnet-base-v2 --build
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model neutral-sft-v3-qwen3-8b
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model neutral-sft-v3-olmo3-7b
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model Qwen3-8B-Base
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model Olmo-3-1025-7B

# ── Appendix F: the pretrained bases reading the neutral models' pairs ─────────
# (the DPO-pairs bars of appendix F have no producer in this release: the
# data_proximity_t1b data method was not ported)
PAIRS_DIR=data/elicitation/pairs/default_llm/constitution_tenets_v3
valuegen predict -m persona   --value-set constitution_tenets_v3 --model "$VIEWS/Qwen3-8B-Base" \
    --data $PAIRS_DIR/neutral-sft-v3-qwen3-8b-* --predictor-param template=urial0 --layer 18 --build
valuegen predict -m grad_proj --value-set constitution_tenets_v3 --model "$VIEWS/Qwen3-8B-Base" \
    --data $PAIRS_DIR/neutral-sft-v3-qwen3-8b-* --predictor-param template=urial0 --layer 18 --build
valuegen predict -m persona   --value-set constitution_tenets_v3 --model "$VIEWS/Olmo-3-1025-7B" \
    --data $PAIRS_DIR/neutral-sft-v3-olmo3-7b-* --predictor-param template=urial0 --layer 16 --build
valuegen predict -m grad_proj --value-set constitution_tenets_v3 --model "$VIEWS/Olmo-3-1025-7B" \
    --data $PAIRS_DIR/neutral-sft-v3-olmo3-7b-* --predictor-param template=urial0 --layer 16 --build

# Into the data/paper layout paper/reproduce reads: RQ1's grids, appendix D's
# per-layer grids and layer sweeps (from the neutral persona runs above), and
# appendix F's base-reads-neutral grids.
python scripts/export_paper_data.py rq1
python scripts/export_paper_data.py appd
python scripts/export_paper_data.py appf

# ── 30B scale: Qwen3-30B-A3B and Olmo-3 32B ───────────────────────────────────
# The same five predictors on the neutral-SFT checkpoints and the pretrained
# bases of the two larger models, scored against the 30B/32B matrices gt.sh
# builds. Each model writes its own pairs from the trait data generated above.
# What model size changes:
#   * Layers are pinned at the sweeps' relative depth (18/36 and 16/32, both
#     0.5 -> 24/48 and 32/64). No sweep runs here.
#   * GPUs per task, for 48 GB cards (a bf16 copy is 61-65 GB): 2 for pairs
#     and persona, 3 (Qwen) or 4 (Olmo) for grad_proj, 2 per weight_steer
#     LoRA. `gpus` is a data-method param, so it enters the pairs artifact ID.
#   * weight_steer uses the model-parallel recipe copies (`*-mp.yml`: the same
#     8-bit LoRA with `device_map: balanced`). On the MoE the recipe's LoRA
#     targets include all 18,432 expert linears (1.69B LoRA parameters; 87M
#     at 8B).
# Disk and memory, measured on the paper's runs (weight_steer is the heavy
# step, so it runs last per model):
#   * task-vector cache on finetune_root: 106-129 GB per value, 7-8 TB per
#     model. Each cache is deleted below once its grid is written, so the peak
#     is one model's; KEEP_TASKVECS=1 keeps them.
#   * LoRA adapters: 132 per model, kept; ~17-19 GB each on the MoE (~2.2 TB
#     per model), ~8 GB on Olmo-3 32B. Over half of each is axolotl's
#     checkpoint-N/ copy, which is safe to delete once training finishes.
#   * the build stage loads the base in fp32 on CPU: ~430 GiB RSS.
#   * Put array stages on a non-preemptible partition: a preempted task
#     reloads the model (10-40 min).
# Cost, per model: pairs ~12 h wall (+ ~$12 of Gemini judging), grad_proj
# ~4 h; weight_steer LoRA training ~20 GPU-hours on Olmo-3 32B and ~130 on
# the MoE (20-120 min per LoRA), then ~6 h build and 8-26 h gather. About 1.5
# days per Olmo model, 3-4 per Qwen model.
[[ -n "${SKIP_BIGMODELS:-}" ]] && exit 0

python scripts/build_model_view.py value-generalization/neutral-sft-v3-olmo3-32b --revision b05f9980d3886842c5562363fbed83eec5e93455
python scripts/build_model_view.py value-generalization/neutral-sft-v3-qwen3-30b-a3b --revision 95b8b6a5cf3833b053f8f2abbf2d7b13421a9b5a
python scripts/build_model_view.py allenai/Olmo-3-1125-32B --revision c2b61dae89a1ad10e4ad5653d0e46b590902607b --symlink
python scripts/build_model_view.py Qwen/Qwen3-30B-A3B-Base --revision 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 --symlink

# The pairs recipe above with 2 GPUs and the fork's build-time sweep pinned to
# the layer that is read (the last layer_start/layer_end given wins).
PAIRS_Q=("${PAIRS[@]}" --param gpus=2 --param layer_start=24 --param layer_end=24)
PAIRS_O=("${PAIRS[@]}" --param gpus=2 --param layer_start=32 --param layer_end=32)
WS=(--predictor-param recipe=multimodel/lora-sft-mp.yml --gpus 2)
WS_BASE=(--predictor-param recipe=multimodel/lora-sft-inputoutput-mp.yml --predictor-param scaffold=urial0 --gpus 2)
URIAL=(--param chat_template_family=urial0)

# Neutral-SFT models. grad_proj: the Olmo checkpoint needs its own tokenizer
# template (`native`), as at 7B; Qwen's default is the ChatML it trained on.
valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/neutral-sft-v3-olmo3-32b" "${PAIRS_O[@]}" --gpus 2 --layer 32 --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/neutral-sft-v3-olmo3-32b" "${PAIRS_O[@]}" --predictor-param template=native --gpus 4 --layer 32 --build
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model neutral-sft-v3-olmo3-32b
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/neutral-sft-v3-olmo3-32b" "${PAIRS_O[@]}" "${WS[@]}" --build
[[ -n "${KEEP_TASKVECS:-}" ]] || rm -rf "$FT"/weight_steering_taskvecs/neutral-sft-v3-olmo3-32b-*

valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/neutral-sft-v3-qwen3-30b-a3b" "${PAIRS_Q[@]}" --gpus 2 --layer 24 --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/neutral-sft-v3-qwen3-30b-a3b" "${PAIRS_Q[@]}" --gpus 3 --layer 24 --build
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model neutral-sft-v3-qwen3-30b-a3b
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/neutral-sft-v3-qwen3-30b-a3b" "${PAIRS_Q[@]}" "${WS[@]}" --build
[[ -n "${KEEP_TASKVECS:-}" ]] || rm -rf "$FT"/weight_steering_taskvecs/neutral-sft-v3-qwen3-30b-a3b-*

# Pretrained bases, self-generated urial0 pairs.
valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/Olmo-3-1125-32B" "${PAIRS_O[@]}" "${URIAL[@]}" --predictor-param template=urial0 --gpus 2 --layer 32 --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/Olmo-3-1125-32B" "${PAIRS_O[@]}" "${URIAL[@]}" --predictor-param template=urial0 --gpus 4 --layer 32 --build
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model Olmo-3-1125-32B
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/Olmo-3-1125-32B" "${PAIRS_O[@]}" "${URIAL[@]}" "${WS_BASE[@]}" --build
[[ -n "${KEEP_TASKVECS:-}" ]] || rm -rf "$FT"/weight_steering_taskvecs/Olmo-3-1125-32B-*

valuegen predict -m persona      "${V3[@]}" --model "$VIEWS/Qwen3-30B-A3B-Base" "${PAIRS_Q[@]}" "${URIAL[@]}" --predictor-param template=urial0 --gpus 2 --layer 24 --build
valuegen predict -m grad_proj    "${V3[@]}" --model "$VIEWS/Qwen3-30B-A3B-Base" "${PAIRS_Q[@]}" "${URIAL[@]}" --predictor-param template=urial0 --gpus 3 --layer 24 --build
python scripts/sentemb_behavior.py --value-set constitution_tenets_v3 --pairs-model Qwen3-30B-A3B-Base
valuegen predict -m weight_steer "${V3[@]}" --model "$VIEWS/Qwen3-30B-A3B-Base" "${PAIRS_Q[@]}" "${URIAL[@]}" "${WS_BASE[@]}" --build
[[ -n "${KEEP_TASKVECS:-}" ]] || rm -rf "$FT"/weight_steering_taskvecs/Qwen3-30B-A3B-Base-*

# The 30B arms' grids and the neutral models' per-layer grids, as above.
python scripts/export_paper_data.py rq1 --30b
python scripts/export_paper_data.py appd --30b
