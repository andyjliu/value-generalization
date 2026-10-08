"""End-to-end test for model_spec_aft: real fork subprocess, mock generator.

Runs the whole :func:`model_spec_aft.build_data` path — spec rendering, the
``src.aft.generate_chat`` subprocess in ``.venvs/msm`` (domains → questions →
dedup → responses → LLM filter), and the chat→pair dataset conversion —
against a stdlib HTTP server that speaks just enough of the OpenAI chat
completions API. No credits, no GPUs; it exercises exactly the plumbing the
unit tests mock out (PYTHONPATH isolation, absolute spec paths, cwd-relative
fork outputs, the served-generator ``api_base`` wiring).

Skipped when the optional msm venv isn't built (``setup_env.sh --msm``); fails
with the fix-it command when the submodule is missing. The dedup step loads sentence-transformers' all-MiniLM-L6-v2 from
the HF cache (HF_HOME rides in from ``.env`` via ``ClusterConfig.activate``).
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from valuegen._external import REPO_ROOT
from valuegen.config import ClusterConfig, resolve_experiment
from valuegen.ground_truth import interventions, model_spec_aft

MSM_VENV = REPO_ROOT / ".venvs" / "msm"
MSM_FORK = REPO_ROOT / "external" / "model_spec_midtraining" / "src"

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def msm_environment():
    if not MSM_FORK.is_dir():
        pytest.fail("needs the model_spec_midtraining submodule: run "
                    "git submodule update --init --recursive", pytrace=False)
    if not (MSM_VENV / "bin" / "python").exists():
        pytest.skip("needs the optional msm environment: run "
                    "./scripts/setup_env.sh --msm")

# Distinct topics so the cosine-similarity dedup doesn't collapse the mock
# questions to one survivor.
TOPICS = [
    "gardening", "astronomy", "cooking", "carpentry", "chess", "sailing",
    "photography", "pottery", "cycling", "birdwatching", "calligraphy",
    "beekeeping", "origami", "juggling", "archery", "knitting",
]


class _MockGenerator(BaseHTTPRequestHandler):
    """Just enough OpenAI chat-completions API for the AFT pipeline."""

    topics = itertools.cycle(TOPICS)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = payload.get("messages", [])
        user_text = " ".join(
            m["content"] for m in messages if m["role"] == "user"
        )
        if any(m["role"] == "system" for m in messages):
            # Response generation (the only stage that sets a system prompt).
            content = (
                "<think>The spec says to prioritize the value.</think>"
                "I would approach this while prioritizing the spec's value."
            )
        elif "<verdict>" in user_text:
            content = "All criteria PASS.\n<verdict>INCLUDE</verdict>"
        else:
            # Domain or question generation. Deliberately include numbered
            # reasoning outside <output> and a planning header inside it: the
            # fork must keep only validated items from the final tagged list.
            count_match = re.search(r"\b(\d+)\b", user_text)
            count = min(int(count_match.group(1)) if count_match else 5, 10)
            items = "\n".join(
                f"{i + 1}. How should one approach {next(self.topics)}?"
                for i in range(max(count, 1))
            )
            content = (
                "1. Analyze the request before answering.\n"
                "2. Decide how to format the final list.\n"
                "<output>\n"
                "1. **Analyze User Input:**\n"
                f"{items}\n"
                "</output>"
            )
        body = json.dumps({
            "id": "mock", "object": "chat.completion",
            "model": payload.get("model", "mock"),
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockGenerator)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.fixture
def real_msm_cluster(tmp_path: Path) -> ClusterConfig:
    """Real repo (venvs, value_sets, .env) but throwaway data roots."""
    return ClusterConfig(
        envs={"default": ".venvs/core", "msm": ".venvs/msm"},
        repo=REPO_ROOT,
        finetune_root=tmp_path / "finetune",
        finetune_root_legacy=tmp_path / "legacy-finetune",
        data=tmp_path / "data",
        slurm_logs=tmp_path / "logs",
        mail_type="END",
        mail_user="test@example.com",
        gpu_type="A6000",
        max_concurrent_gpus=8,
        default_time="1:00:00",
        default_mem="4G",
        cpu_partition="cpu",
        cpu_qos="cpu_qos",
        exclude=[],
        source_path=tmp_path / "cluster.yaml",
    )


def test_build_data_end_to_end_against_mock_generator(
    mock_server, real_msm_cluster, monkeypatch
):
    monkeypatch.delenv("VALUEGEN_MSM_API_BASE", raising=False)
    cfg = resolve_experiment({
        "name": "aft_e2e",
        "intervention": {
            "method": "model_spec_aft",
            "value_set": "constitution_tenets_v3",
            "values": ["calibrated_uncertainty"],
            "models": {"olmo7b_base": "allenai/OLMo-2-1124-7B"},
            "generation": {
                "n_samples": 4,
                "questions_per_domain": 2,
                "model_id": "mock/generator",
                "api_base": mock_server,
                "max_concurrent": 4,
            },
        },
        "evaluation": {
            "method": "conflictscope",
            "scenarios": "data/scenarios/const_v3_cs",
            "judge": {"model": "mock/judge"},
        },
    })

    interventions.claim(cfg, real_msm_cluster)
    model_spec_aft.build_data(cfg, real_msm_cluster)

    dataset = (
        interventions.datasets_dir(cfg, real_msm_cluster) / "calibrated_uncertainty" / "dataset.jsonl"
    )
    assert dataset.is_file()
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert 1 <= len(rows) <= 4
    for row in rows:
        assert row["prompt"][0]["role"] == "user"
        assert row["chosen"][0]["role"] == "assistant"
        assert "Analyze User Input" not in row["prompt"][0]["content"]
        assert "format the final list" not in row["prompt"][0]["content"]
        # strip_cot default: think tags must not reach the SFT targets.
        assert "<think>" not in row["chosen"][0]["content"]

    # Idempotent: a second call must not re-run the fork (the mock server is
    # still up, but a dead one would also work — nothing should be contacted).
    before = dataset.stat().st_mtime_ns
    model_spec_aft.build_data(cfg, real_msm_cluster)
    assert dataset.stat().st_mtime_ns == before
