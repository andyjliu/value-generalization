"""Launch gates for the value-neutral SFT runs (OLMo-3-32B, Qwen3-30B-A3B).

Explicit chat-format identities, terminators, completion masks, EOS/template
alignment of exports, resume forwarding, FSDP wrap resolution, and the
seed-42 split. Tests that need the pinned tokenizers or the canonical dataset
skip when those are not on this machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from huggingface_hub.constants import HF_HUB_CACHE

from valuegen.ground_truth import training
from valuegen.ground_truth.chat_formats import CHAT_FORMATS, get_chat_format, read_sidecar

HF_HUB = Path(HF_HUB_CACHE)
OLMO_SNAP = HF_HUB / "models--allenai--Olmo-3-1125-32B/snapshots/c2b61dae89a1ad10e4ad5653d0e46b590902607b"
QWEN_SNAP = HF_HUB / "models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
DATASET = Path(__file__).resolve().parents[2] / "data/sft_neutral/tulu3_v3/dataset.jsonl"
# The neutral-SFT runs' recorded split ids (split_seed*_{train,holdout}_row_ids.txt).
_PROV_ENV = os.environ.get("VALUEGEN_NEUTRAL_SFT_PROV")
PROV = Path(_PROV_ENV) if _PROV_ENV else None

USER = [{"role": "user", "content": "What is 2+2?"}]
FULL = [*USER, {"role": "assistant", "content": "4."}]
SYS_FULL = [{"role": "system", "content": "Be terse."}, *FULL]
MULTI = [*FULL, {"role": "user", "content": "And 3+3?"}, {"role": "assistant", "content": "6."}]


def _tok(snap: Path, fmt_name: str):
    if not snap.is_dir():
        pytest.skip(f"pinned snapshot not cached: {snap}")
    from transformers import AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(str(snap))
    gen = GenerationConfig.from_pretrained(str(snap))
    fmt = get_chat_format(fmt_name)
    training._pin_chat_template(tok, "anything", "x.jsonl", chat_format=fmt)
    training.adopt_turn_ender_eos(tok, gen, None, chat_format=fmt)
    return tok, gen, fmt


# ── explicit identities and pinned bytes ─────────────────────────────────────


def test_registry_pins_published_templates():
    olmo, qwen = CHAT_FORMATS["olmo3_chatml"], CHAT_FORMATS["qwen_chatml"]
    assert olmo.template_sha256 == "51c7f8c700135ac224994f0d4a8b33562e9fe8edf9d33ad50909bed0a84ca9b6"
    assert qwen.template_sha256 == "333c256df3738c5351d58126b41fb323efae0514c1b7df837b11107d95441cdf"
    assert qwen.template == training.CHAT_TEMPLATES["qwen"]
    assert "{%- generation -%}" in olmo.template and "{%- endgeneration -%}" in olmo.template
    assert olmo.stop_tokens() == ("<|endoftext|>",) and olmo.pad_token == "<|pad|>"
    assert qwen.stop_tokens() == ("<|im_end|>", "<|endoftext|>") and qwen.pad_token == "<|endoftext|>"
    with pytest.raises(KeyError):
        get_chat_format("olmo")


def test_olmo3_terminators_intermediate_im_end_final_endoftext():
    fmt = get_chat_format("olmo3_chatml")
    assert fmt.render(USER, True) == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert fmt.render(FULL, False) == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4.<|endoftext|>"
    assert fmt.render(SYS_FULL, False).startswith("<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\n")
    multi = fmt.render(MULTI, False)
    assert "<|im_start|>assistant\n4.<|im_end|>\n<|im_start|>user\nAnd 3+3?<|im_end|>\n<|im_start|>assistant\n6.<|endoftext|>" in multi
    # no tools -> no auto-injected function-calling system turn
    assert "function-calling" not in fmt.render(USER, True)
    assert "<functions>" in fmt.render(USER, True, tools=[{"name": "f"}])
    # prompt render is a strict prefix of the full render (trl boundary property)
    assert fmt.render(FULL, False).startswith(fmt.render(USER, True))


def test_qwen_plain_chatml_no_think():
    fmt = get_chat_format("qwen_chatml")
    assert fmt.render(USER, True) == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert fmt.render(FULL, False) == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4.<|im_end|>"
    assert "<think>" not in fmt.render(USER, True) and "<think>" not in fmt.render(MULTI, False)


def test_explicit_format_beats_path_heuristics():
    class Tok:
        chat_template = "base"

    tok = Tok()
    training._pin_chat_template(tok, "allenai/Olmo-3-1125-32B", "x.jsonl", chat_format=get_chat_format("olmo3_chatml"))
    assert tok.chat_template == get_chat_format("olmo3_chatml").template
    # the legacy heuristic would have chosen the OLMo-2/Tulu template for this path
    assert training.infer_chat_family("neutral_sft_olmo3_32b_base") == "olmo"
    tok = Tok()
    training._pin_chat_template(tok, "neutral_sft_olmo3_32b_base", "x.jsonl")
    assert tok.chat_template == training.CHAT_TEMPLATES["olmo"]


# ── eos adoption / export alignment ──────────────────────────────────────────


def test_olmo3_eos_adoption_matches_published_artifact():
    tok, gen, fmt = _tok(OLMO_SNAP, "olmo3_chatml")
    assert tok.eos_token == "<|endoftext|>" and tok.eos_token_id == 100257
    assert tok.pad_token == "<|pad|>" and tok.pad_token_id == 100277
    assert gen.eos_token_id == [100257] and gen.pad_token_id == 100277
    assert tok.convert_tokens_to_ids("<|im_end|>") == 100265 and 100265 not in gen.eos_token_id


def test_qwen_eos_adoption_matches_published_artifact():
    tok, gen, fmt = _tok(QWEN_SNAP, "qwen_chatml")
    assert tok.eos_token == "<|im_end|>" and tok.eos_token_id == 151645
    assert tok.pad_token == "<|endoftext|>" and tok.pad_token_id == 151643
    assert gen.eos_token_id == [151645, 151643] and gen.pad_token_id == 151643


@pytest.mark.parametrize("snap,fmt_name", [(OLMO_SNAP, "olmo3_chatml"), (QWEN_SNAP, "qwen_chatml")])
def test_export_alignment_round_trip(tmp_path, snap, fmt_name):
    tok, gen, fmt = _tok(snap, fmt_name)
    out = tmp_path / "export"
    out.mkdir()
    # simulate a base-tokenizer export with a stray template file that would take precedence
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(str(snap)).save_pretrained(str(out))
    (out / "chat_template.jinja").write_text("{{ 'WRONG' }}")
    (out / "config.json").write_text(json.dumps({"model_type": "llama", "eos_token_id": 0}))
    summary = training.align_chat_format(out, fmt)
    check = AutoTokenizer.from_pretrained(str(out))
    assert check.chat_template == fmt.template
    assert check.eos_token == fmt.turn_ender
    assert summary["stop_token_ids"] == [check.convert_tokens_to_ids(t) for t in fmt.stop_tokens()]
    side = read_sidecar(out)
    assert side["chat_format"] == fmt.name and side["template_sha256"] == fmt.template_sha256
    # idempotent
    assert training.align_chat_format(out, fmt) == summary


# ── completion masks (exact TRL semantics) ──────────────────────────────────


@pytest.mark.parametrize("snap,fmt_name", [(OLMO_SNAP, "olmo3_chatml"), (QWEN_SNAP, "qwen_chatml")])
def test_completion_mask_supervises_answer_and_terminator_only(snap, fmt_name):
    tok, gen, fmt = _tok(snap, fmt_name)
    r = training.tokenize_prompt_completion(tok, USER, [{"role": "assistant", "content": "4."}])
    ids, labels = r["input_ids"], r["labels"]
    assert not r["prompt_mismatch"]
    n_p = r["prompt_len"]
    assert all(l == -100 for l in labels[:n_p])
    sup = [t for t in labels[n_p:]]
    assert sup == ids[n_p:] and len(sup) >= 2
    assert tok.decode(sup) == "4." + fmt.turn_ender
    assert ids[-1] == tok.convert_tokens_to_ids(fmt.turn_ender)
    assert tok.decode(ids[:n_p]) == fmt.render(USER, True)
    if fmt_name == "olmo3_chatml":
        f = tok.apply_chat_template(FULL, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        assert list(f["assistant_masks"]) == [0] * n_p + [1] * (len(ids) - n_p)
    if fmt_name == "qwen_chatml":
        think = tok.convert_tokens_to_ids("<think>")
        assert think not in ids


def test_padding_labels_ignored():
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    col = DataCollatorForLanguageModeling(pad_token_id=7)
    b = col([{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}, {"input_ids": [1, 2], "labels": [-100, 2]}])
    assert b["labels"][1].tolist() == [-100, 2, -100, -100]
    assert b["attention_mask"][1].tolist() == [1, 1, 0, 0] and b["input_ids"][1].tolist() == [1, 2, 7, 7]


# ── resume forwarding / checkpoints ─────────────────────────────────────────


def test_resolve_resume(tmp_path):
    assert training._resolve_resume(None, str(tmp_path)) is None
    assert training._resolve_resume("false", str(tmp_path)) is None
    assert training._resolve_resume("true", str(tmp_path)) is True
    assert training._resolve_resume("auto", str(tmp_path)) is None
    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-3" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-6").mkdir()  # incomplete: no trainer_state.json
    assert training._resolve_resume("auto", str(tmp_path)) == str(tmp_path / "checkpoint-3")
    (tmp_path / "checkpoint-6" / "trainer_state.json").write_text("{}")
    assert training._resolve_resume("auto", str(tmp_path)) == str(tmp_path / "checkpoint-6")
    assert training._resolve_resume(str(tmp_path / "checkpoint-3"), str(tmp_path)) == str(tmp_path / "checkpoint-3")
    with pytest.raises(FileNotFoundError):
        training._resolve_resume(str(tmp_path / "nope"), str(tmp_path))


def test_use_cache_false_under_wrapper_level_ac():
    class Args:
        gradient_checkpointing = False
        fsdp_config = {"activation_checkpointing": True}

    assert training.resolve_use_cache(Args()) is False
    Args.fsdp_config = {"activation_checkpointing": "false"}
    assert training.resolve_use_cache(Args()) is True
    Args.gradient_checkpointing = True
    assert training.resolve_use_cache(Args()) is False


# ── FSDP wrap resolution ────────────────────────────────────────────────────


def test_fsdp_wrap_class_resolution():
    pytest.importorskip("transformers")
    assert training.fsdp_wrap_class("Olmo3ForCausalLM") == "Olmo3DecoderLayer"
    assert training.fsdp_wrap_class("Qwen3MoeForCausalLM") == "Qwen3MoeDecoderLayer"
    with pytest.raises(KeyError):
        training.fsdp_wrap_class("NotAModel")
    for name in ("fsdp2_olmo3", "fsdp2_qwen3_moe"):
        cfg = json.loads((Path(__file__).resolve().parents[2] / "configs/fsdp" / f"{name}.json").read_text())
        assert cfg["version"] == 2 and cfg["activation_checkpointing"] is True and cfg["cpu_offload"] is False
        assert cfg["state_dict_type"] == "SHARDED_STATE_DICT"
        assert cfg["transformer_layer_cls_to_wrap"] in (["Olmo3DecoderLayer"], ["Qwen3MoeDecoderLayer"])


# ── seed-42 split ───────────────────────────────────────────────────────────


def test_seed42_split_membership_and_order():
    if not DATASET.is_file() or PROV is None or not PROV.is_dir():
        pytest.skip("canonical dataset / provenance lists not on this machine")
    split = training._load_pair_dataset(str(DATASET), 42, 0.1)
    train_ids, hold_ids = list(split["train"]["row_id"]), list(split["test"]["row_id"])
    assert (len(train_ids), len(hold_ids)) == (17677, 1965)
    assert train_ids == (PROV / "split_seed42_train_row_ids.txt").read_text().split()
    assert hold_ids == (PROV / "split_seed42_holdout_row_ids.txt").read_text().split()
    assert not set(train_ids) & set(hold_ids)
    # seed 0 is reference material only and must NOT be what val_fraction/seed 42 produces
    assert train_ids != (PROV / "split_seed0_train_row_ids.txt").read_text().split()
