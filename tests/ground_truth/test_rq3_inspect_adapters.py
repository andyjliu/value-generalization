"""Inspect adapters: candidate routing, grader provider, grade cache and
completeness; plus the DPO trainer's explicit chat-format path and the serve
helpers.

Inspect-dependent tests skip when ``inspect_ai`` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from valuegen.ground_truth import inspect_runner as IR

REPO = Path(__file__).resolve().parents[2]


def test_candidate_routing_never_touches_global_openai_env(monkeypatch):
    assert IR.candidate_model_name("rq3-abc") == "openai-api/rq3cand/rq3-abc"
    pytest.importorskip("inspect_ai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    m = IR.candidate_model("rq3-abc", "http://127.0.0.1:9123/v1", IR.CandidateConfig())
    assert m.name.endswith("rq3cand/rq3-abc")
    assert m.api.base_url == "http://127.0.0.1:9123/v1"
    cfg = m.config
    assert cfg.temperature == 0.7 and cfg.top_p == 1.0 and cfg.max_tokens == 4096 and cfg.seed is None
    assert cfg.frequency_penalty == 0.0 and cfg.presence_penalty == 0.0
    import os

    assert "OPENAI_BASE_URL" not in os.environ


def test_grader_must_use_google_provider():
    pytest.importorskip("inspect_ai")
    with pytest.raises(ValueError):
        IR.grader_model("openai/gpt-5")


def test_grade_cache_round_trip_and_failure_semantics(tmp_path):
    cache = IR.GradeCache(tmp_path / "grades.jsonl")
    k = IR.GradeCache.key("prefill", "ckpt", "s", 3, "response text", "google/x")
    assert cache.get(k) is None
    cache.put(k, {"grader_failed": True, "error": "boom"})
    assert cache.get(k) is None  # failures never satisfy a lookup
    cache.put(k, {"value": {"harmful": 1.0}, "grader_failed": False})
    assert cache.get(k)["value"] == {"harmful": 1.0}
    reloaded = IR.GradeCache(tmp_path / "grades.jsonl")
    assert reloaded.get(k)["value"] == {"harmful": 1.0}
    # identity is exact: another epoch or a changed completion is a miss
    assert reloaded.get(IR.GradeCache.key("prefill", "ckpt", "s", 4, "response text", "google/x")) is None
    assert reloaded.get(IR.GradeCache.key("prefill", "ckpt", "s", 3, "response text!", "google/x")) is None


def test_unit_status_from_synthetic_log():
    Score = SimpleNamespace
    samples = [
        SimpleNamespace(scores={"s": Score(metadata={"grader_failed": False}, value=1.0)}, error=None),
        SimpleNamespace(scores={"s": Score(metadata={"grader_failed": True}, value=0.0)}, error=None),
        SimpleNamespace(scores={}, error=None),
    ]
    st = IR.unit_grading_status_from_log(SimpleNamespace(samples=samples))
    assert st == {"n_samples": 3, "n_scored": 1, "n_failed": 1, "n_unscored": 1}


# ── DPO trainer explicit chat format (regression) ────────────────────────────


def test_dpo_script_arguments_carry_chat_format_and_gates():
    pytest.importorskip("trl")
    from valuegen.ground_truth.training import _pair_script_arguments

    cls = _pair_script_arguments()
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    assert {"chat_format", "expect_train_rows", "expect_max_steps", "expect_warmup_steps", "expect_world_size",
            "check_reference_frozen", "base_revision", "val_fraction"} <= fields


def test_train_dpo_pins_explicit_format_on_policy_and_reference(monkeypatch):
    """The DPO path must resolve --chat_format and apply it to tokenizer,
    policy and reference (the old path ignored the field entirely)."""
    pytest.importorskip("trl")
    from valuegen.ground_truth import training

    seen = {}

    class FakeGen:
        eos_token_id = [100257]
        pad_token_id = None

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True, architectures=["Olmo3ForCausalLM"])
            self.generation_config = FakeGen()

        def named_buffers(self):
            return []

    def fake_pin(tokenizer, model_name_or_path, dataset_name, chat_format=None):
        seen["pin"] = chat_format.name if chat_format else None

    calls = []

    def fake_adopt(tokenizer, gen, family, chat_format=None):
        calls.append((family, chat_format.name if chat_format else None))
        return 100257

    class FakeTok:
        pad_token = "<|pad|>"
        eos_token = "<|endoftext|>"
        chat_template = None

    monkeypatch.setattr(training, "_pin_chat_template", fake_pin)
    monkeypatch.setattr(training, "adopt_turn_ender_eos", fake_adopt)
    monkeypatch.setattr(training, "_set_seeds", lambda s: None)
    import transformers

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: FakeModel()))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: FakeTok()))

    class Stop(Exception):
        pass

    monkeypatch.setattr(training, "_load_pair_dataset", lambda *a, **k: (_ for _ in ()).throw(Stop()))
    argv = ["--model_name_or_path", "/nonexistent/olmo3", "--dataset_name", "x.jsonl", "--output_dir", "/tmp/x",
            "--chat_format", "olmo3_chatml", "--use_peft", "false", "--report_to", "none",
            "--bf16", "false"]  # DPOConfig defaults to bf16, which refuses a CPU-only host
    with pytest.raises(Stop):
        training.train_dpo(argv)
    assert seen["pin"] == "olmo3_chatml"
    # policy + reference both adopted under the explicit format, path family ignored
    assert calls == [(None, "olmo3_chatml"), (None, "olmo3_chatml")]


def test_export_command_carries_format_and_verify():
    from valuegen.ground_truth.training import export_command

    cmd = export_command("base", "/t", "/o", chat_format="olmo3_chatml", base_revision="abc", verify=True)
    assert "--chat-format olmo3_chatml" in cmd and "--base-revision abc" in cmd and "--verify" in cmd


def test_olmo_fsdp_config_wraps_the_verified_decoder():
    cfg = json.loads((REPO / "configs/fsdp/olmo3_full_shard.json").read_text())
    assert cfg["transformer_layer_cls_to_wrap"] == ["Olmo3DecoderLayer"]
    assert cfg["activation_checkpointing"] is True and cfg["state_dict_type"] == "FULL_STATE_DICT"
    assert cfg["use_orig_params"] is True and cfg["version"] == 1
    pytest.importorskip("transformers")
    from valuegen.ground_truth.training import fsdp_wrap_class

    assert fsdp_wrap_class("Olmo3ForCausalLM") == "Olmo3DecoderLayer"


def test_train_block_pins_recipe_and_gates():
    import yaml

    block = yaml.safe_load((REPO / "configs/experiments/rq3_subsets6_dpo.yaml").read_text())
    assert block["chat_format"] == "qwen_chatml" and block["max_length"] == 2048 and block["truncation_mode"] == "keep_start"
    assert block["learning_rate"] == 5e-6 and block["warmup_steps"] == 0.1 and block["num_train_epochs"] == 1
    assert block["per_device_train_batch_size"] == 2 and block["gradient_accumulation_steps"] == 1
    assert block["expect_train_rows"] == 12000 and block["expect_max_steps"] == 750 and block["expect_warmup_steps"] == 75
    assert block["expect_world_size"] == 8 and block["check_reference_frozen"] is True
    assert block["precompute_ref_log_probs"] is False and block["save_strategy"] == "no" and block["val_fraction"] == 0.0
    assert block["fsdp_config"] == "configs/fsdp/qwen3_full_shard.json"
    import math

    assert math.ceil(0.1 * 750) == 75 and 12000 // (2 * 1 * 8) == 750


def test_serve_local_sh_checks_format_and_uses_served_name():
    from valuegen.ground_truth.evaluation import serve_local_sh

    sh = serve_local_sh("/m/export", "rq3-x", "/w/host.txt", "/w/log", chat_format="olmo3_chatml",
                        max_model_len=16384, seed=10000)
    assert "chat_formats check --model-dir /m/export --expect olmo3_chatml" in sh
    assert "--served-model-name rq3-x" in sh and "--seed 10000" in sh and "--max-model-len 16384" in sh
    assert "trap" in sh and "CAND_BASE_URL" in sh and "wait_ready" in sh


def test_eval_command_gates_local_dirs_on_explicit_format(tmp_path):
    from valuegen.ground_truth.evaluation import EvalSpec, eval_command

    spec = EvalSpec(scenarios_dir=tmp_path, output_dir=tmp_path, chat_format="olmo3_chatml")
    cmd = eval_command(spec, "/ckpt/dir", "out.csv")
    assert "chat_formats check --model-dir /ckpt/dir --expect olmo3_chatml" in cmd
    served = EvalSpec(scenarios_dir=tmp_path, output_dir=tmp_path, chat_format="olmo3_chatml",
                      assistant_api_base="http://127.0.0.1:1/v1")
    assert "chat_formats check" not in eval_command(served, "rq3-name", "out.csv")


def test_chat_format_check_cli_rejects_a_bare_dir(tmp_path):
    pytest.importorskip("transformers")
    from valuegen.ground_truth.chat_formats import check_model_dir

    with pytest.raises(Exception):
        check_model_dir(tmp_path, "olmo3_chatml")
