#!/usr/bin/env bash
# Can the uv core env actually initialize CUDA on a GPU node?
#
# Import-clean is not CUDA-clean: the original core build took vllm's PyPI wheel,
# which is CUDA-13-linked and imported fine everywhere while being unable to init
# CUDA under the cluster's 12.x driver. This is the fast (<10 min) GPU-node
# check for the cu128/cu129 stack: torch CUDA init + matmul, a torchvision CUDA
# op, then a real vllm engine bringup + generate on Qwen3-0.6B.
#
#   valuegen smoke                # submit it as a 1-GPU job, placed from cluster.yaml
#   bash tests/fixtures/cuda_smoke.sh  # or run it directly on a GPU you already hold
#
# `valuegen smoke` renders the usual job preamble (partition/QOS/account from
# cluster.yaml, venv activated, .env sourced). Run directly, this script
# activates the env itself.
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV="${VALUEGEN_VENV:-${VIRTUAL_ENV:-.venvs/core}}"
PY="$VENV/bin/python"
test -x "$PY" || { echo "no interpreter at $PY -- run scripts/setup_env.sh"; exit 1; }

# Activate, don't just call $PY: flashinfer JIT shells out to `ninja` (and
# friends) via PATH at vllm engine bringup, and only activation puts the venv's
# bin/ there. Calling the interpreter by path loads the right packages but
# leaves PATH bare — the engine core then dies with FileNotFoundError: 'ninja'
# AFTER the model has loaded. ClusterConfig.activate() does this for all
# generated jobs; hand-written harnesses have to do it themselves.
source "$VENV/bin/activate"

unset PYTHONPATH
export PYTHONPATH="$PWD/src"

echo "== node: $(hostname), driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"

echo "== torch CUDA"
"$PY" - <<'EOF'
import torch, torchvision
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
dev = torch.cuda.get_device_name(0)
a = torch.randn(1024, 1024, device="cuda")
b = (a @ a).sum().item()
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device="cuda")
keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
print(f"torch {torch.__version__} on {dev}: matmul ok ({b:.1f}), cuda nms ok ({len(keep)} kept)")
EOF

echo "== vllm engine bringup + generate (Qwen3-0.6B)"
"$PY" - <<'EOF'
from vllm import LLM, SamplingParams
# Same pinned checkpoint the test suite stages (tests/conftest.py).
llm = LLM(model="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
          max_model_len=1024, gpu_memory_utilization=0.5)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
text = out[0].outputs[0].text
print(f"vllm generate ok: {text!r}")
assert text.strip(), "vllm returned empty generation"
EOF

echo "CUDA SMOKE PASSED"
