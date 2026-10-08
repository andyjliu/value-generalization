#!/usr/bin/env bash
# Build the valuegen environments with uv. Idempotent; safe to re-run.
#
#   ./scripts/setup_env.sh              # core (what almost everyone needs)
#   ./scripts/setup_env.sh --ws-train   # + the pinned axolotl env for retraining adapters
#   ./scripts/setup_env.sh --msm        # + the env for generating spec-aligned SFT data
#   VALUEGEN_CORE_ENV=/tmp/valuegen-core UV_LINK_MODE=copy ./scripts/setup_env.sh
#                                    # core packages on node-local storage
#
# Also creates .env and configs/cluster.yaml from their templates if missing,
# then runs the test suite.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || {
  echo "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}

if [ -d .git ] || git rev-parse --git-dir >/dev/null 2>&1; then
  echo "==> submodules"
  git submodule update --init --recursive
fi

core_env_path="${VALUEGEN_CORE_ENV:-.venvs/core}"
echo "==> core env ($core_env_path)"
uv venv --allow-existing --python 3.12 "$core_env_path"
# CUDA-12.x stack: index + strategy come from [tool.uv.pip] in pyproject.toml;
# the overrides file pins torchcodec to a cu12 build (vllm asks for a version
# that only exists CUDA-13-linked). Both are needed on ANY install into this env.
uv pip install --python "$core_env_path" --overrides requirements/overrides.txt -e ".[dev]"

if [ "${1:-}" = "--msm" ]; then
  # Only needed for model_spec_aft data generation (the MSM fork's AFT
  # pipeline). Cannot be merged into core: its safetytooling dependency
  # `==`-pins anthropic/transformers/datasets versions that would downgrade
  # the vllm/trl stack. CPU-only, so torch comes from the cpu wheel index.
  echo "==> msm env (.venvs/msm) — model_spec_midtraining AFT data generation"
  uv venv --allow-existing --python 3.11 .venvs/msm
  uv pip install --python .venvs/msm torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venvs/msm -r requirements/msm.txt
fi

if [ "${1:-}" = "--ws-train" ]; then
  # Only needed to TRAIN weight-steering LoRA adapters from scratch. Consuming
  # existing adapters is peft-free and happens in core. Cannot be merged into
  # core: axolotl `==`-pins torch 2.5.1 / transformers 4.50.3 / peft 0.15.0, and
  # that peft pin decides how `modules_to_save` keys are serialized — get it wrong
  # and every task vector silently loses embed_tokens/lm_head.
  echo "==> ws-train env (.venvs/ws-train) — pinned axolotl"
  uv venv --allow-existing --python 3.11 .venvs/ws-train
  uv pip install --python .venvs/ws-train -r requirements/ws-train.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "==> created .env from .env.example — PUT YOUR API KEYS IN IT."
  echo "    valuegen needs OPENAI_API_KEY (judge/eval), ANTHROPIC_API_KEY"
  echo "    (scenario generation), GOOGLE_API_KEY (the Gemini persona recipe) and"
  echo "    HF_API_KEY (exported as HF_TOKEN in every job). It is gitignored."
fi

NEW_CLUSTER_YAML=0
if [ ! -f configs/cluster.yaml ]; then
  # repo/data are known here; everything else is cluster-specific.
  sed -e "s#^  repo: /path/to/value-generalization\$#  repo: $(pwd)#" \
      -e "s#^  data: data\$#  data: $(pwd)/data#" \
      configs/cluster.yaml.example > configs/cluster.yaml
  NEW_CLUSTER_YAML=1
  echo
  echo "==> created configs/cluster.yaml from configs/cluster.yaml.example —"
  echo "    EDIT IT before running anything that submits SLURM jobs."
  echo "    Every 'CHANGE THIS' line needs a real value for your machine/cluster."
  echo "    It is gitignored (machine-specific, not something to commit)."
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo
  echo "==> no SLURM (sbatch) found on this machine."
  echo "    'valuegen gt run' / 'predict --build' submit SLURM jobs and need it."
  echo "    Everything else — the test suite, 'valuegen analyze correlate/mds'"
  echo "    against artifacts you already have — works fine without it."
fi

echo
echo "==> sanity check: running the test suite (first run downloads the ~1.5 GB"
echo "    Qwen3-0.6B test checkpoint; GPU end-to-end tests skip without a GPU)"
.venvs/core/bin/python -B -m pytest -q -p no:cacheprovider -rs tests/ || TEST_STATUS=$?
echo
if [ -n "${TEST_STATUS:-}" ]; then
  echo "done, but the test suite FAILED (exit $TEST_STATUS) -- see above."
else
  echo "done."
fi
if [ "$NEW_CLUSTER_YAML" = 1 ]; then
  echo "Next: fill in configs/cluster.yaml, then edit .env for your API keys."
fi
echo "To verify the CUDA stack on a GPU node (torch CUDA + vllm engine bringup, ~5 min),"
echo "once configs/cluster.yaml is filled in:"
echo "  .venvs/core/bin/valuegen smoke            # add --tests to also run the GPU e2e tests"
exit "${TEST_STATUS:-0}"
