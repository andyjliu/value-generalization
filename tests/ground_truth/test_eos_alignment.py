"""Instruct-style eos metadata for chat-family exports.

A model trained under the pinned templates ends every assistant turn in the
family's turn-ender (Qwen ``<|im_end|>``, Llama ``<|eot_id|>``) and nothing
else, exactly like the instruct releases. The base tokenizer/generation config
we export declare only the pretraining terminator, so vLLM never stopped
(2026-08-29 neutral_sft_v3_qwen3_8b: 35% runaway turns). These tests pin the
in-memory adoption both trainers apply and the on-disk export alignment,
including the transformers-5 ``chat_template.jinja`` precedence that let a
hand-patched ``tokenizer_config.json`` keep serving the base template.
"""

import json

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerFast

from valuegen.ground_truth import training


def _tiny_tokenizer(specials: list[str], eos: str, pad: str | None):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {tok: i for i, tok in enumerate([*specials, "hi", "user", "assistant"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token=specials[0]))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token=eos,
        pad_token=pad,
        additional_special_tokens=[t for t in specials if t not in (eos, pad)],
    )


QWEN_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
LLAMA_SPECIALS = ["<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def test_adopt_qwen_matches_instruct_convention():
    tok = _tiny_tokenizer(QWEN_SPECIALS, eos="<|endoftext|>", pad="<|endoftext|>")
    gen = GenerationConfig(eos_token_id=tok.convert_tokens_to_ids("<|endoftext|>"))
    tid = training.adopt_turn_ender_eos(tok, gen, "qwen")
    assert tid == tok.convert_tokens_to_ids("<|im_end|>")
    assert tok.eos_token == "<|im_end|>"
    assert tok.pad_token == "<|endoftext|>"  # pad stays the base terminator
    assert gen.eos_token_id == [tid, tok.convert_tokens_to_ids("<|endoftext|>")]
    # Idempotent: a second pass (export after training) changes nothing.
    training.adopt_turn_ender_eos(tok, gen, "qwen")
    assert gen.eos_token_id == [tid, tok.convert_tokens_to_ids("<|endoftext|>")]


def test_adopt_llama_pad_falls_back_to_base_eos_not_turn_ender():
    tok = _tiny_tokenizer(LLAMA_SPECIALS, eos="<|end_of_text|>", pad=None)
    gen = GenerationConfig(eos_token_id=tok.convert_tokens_to_ids("<|end_of_text|>"))
    training.adopt_turn_ender_eos(tok, gen, "llama")
    assert tok.eos_token == "<|eot_id|>"
    assert tok.pad_token == "<|end_of_text|>"
    assert gen.eos_token_id[0] == tok.convert_tokens_to_ids("<|eot_id|>")
    assert gen.pad_token_id == tok.pad_token_id


def test_adopt_is_noop_where_template_already_ends_in_eos():
    tok = _tiny_tokenizer(["<|endoftext|>", "<|user|>", "<|assistant|>"], eos="<|endoftext|>", pad="<|endoftext|>")
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
    gen = GenerationConfig(eos_token_id=eos_id)
    training.adopt_turn_ender_eos(tok, gen, "olmo")
    assert tok.eos_token == "<|endoftext|>"
    assert gen.eos_token_id == [eos_id]


OLMO3_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


def test_adopt_olmo3_keeps_eos_and_adds_im_end_stop():
    # Olmo-3 ChatML ends the final assistant turn in eos itself, so the
    # turn-ender IS <|endoftext|>; <|im_end|> (every non-final turn's closer)
    # joins the stop list as insurance, like allenai/Olmo-3-7B-Instruct-SFT's
    # own [100265, 100257].
    tok = _tiny_tokenizer(OLMO3_SPECIALS, eos="<|endoftext|>", pad="<|pad|>")
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    gen = GenerationConfig(eos_token_id=[eos_id], pad_token_id=tok.pad_token_id)
    tid = training.adopt_turn_ender_eos(tok, gen, "olmo-3")
    assert tid == eos_id
    assert tok.eos_token == "<|endoftext|>" and tok.pad_token == "<|pad|>"
    assert gen.eos_token_id == [eos_id, im_end]
    training.adopt_turn_ender_eos(tok, gen, "olmo-3")  # idempotent
    assert gen.eos_token_id == [eos_id, im_end]


def test_adopt_without_family_is_noop():
    tok = _tiny_tokenizer(QWEN_SPECIALS, eos="<|endoftext|>", pad="<|endoftext|>")
    gen = GenerationConfig(eos_token_id=0)
    assert training.adopt_turn_ender_eos(tok, gen, None) is None
    assert tok.eos_token == "<|endoftext|>" and gen.eos_token_id == 0


def test_adopt_refuses_tokenizer_without_turn_ender():
    tok = _tiny_tokenizer(["<|endoftext|>", "<|user|>"], eos="<|endoftext|>", pad="<|endoftext|>")
    with pytest.raises(ValueError, match="turn-ender"):
        training.adopt_turn_ender_eos(tok, GenerationConfig(eos_token_id=0), "qwen")


def _base_export(tmp_path, name: str, specials, eos: str):
    """A dir laid out like our exports before alignment: base tokenizer, base
    generation config, and the base's own chat_template.jinja (which
    transformers 5 prefers over any tokenizer_config.json entry)."""
    out = tmp_path / name
    tok = _tiny_tokenizer(specials, eos=eos, pad=eos)
    tok.chat_template = "{{ 'BASE TEMPLATE' }}"
    tok.save_pretrained(out)
    assert (out / "chat_template.jinja").is_file()
    GenerationConfig(eos_token_id=tok.convert_tokens_to_ids(eos)).save_pretrained(out)
    (out / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return out


def test_align_export_serves_pinned_template_and_instruct_eos(tmp_path):
    out = _base_export(tmp_path, "run_qwen3_8b_merged", QWEN_SPECIALS, "<|endoftext|>")
    training.align_eos_with_template(out)
    tok = AutoTokenizer.from_pretrained(out)
    gen = GenerationConfig.from_pretrained(out)
    assert tok.chat_template == training.CHAT_TEMPLATES["qwen"]
    assert tok.eos_token == "<|im_end|>"
    assert tok.pad_token == "<|endoftext|>"
    assert gen.eos_token_id == [
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"),
    ]


def test_align_hand_patched_config_alone_would_not_have_served(tmp_path):
    # The 2026-08-29 GCS hand patch: tokenizer_config.json rewritten, jinja
    # left behind. transformers 5 still loads the jinja. The aligner's reload
    # check is what makes that impossible to ship silently.
    out = _base_export(tmp_path, "run_qwen3_8b_merged", QWEN_SPECIALS, "<|endoftext|>")
    cfg = json.loads((out / "tokenizer_config.json").read_text())
    cfg["chat_template"] = training.CHAT_TEMPLATES["qwen"]
    (out / "tokenizer_config.json").write_text(json.dumps(cfg))
    assert AutoTokenizer.from_pretrained(out).chat_template == "{{ 'BASE TEMPLATE' }}"
    training.align_eos_with_template(out)
    assert AutoTokenizer.from_pretrained(out).chat_template == training.CHAT_TEMPLATES["qwen"]


def test_align_reload_check_fails_loudly_when_nothing_took(tmp_path, monkeypatch):
    # Simulate a save that leaves the base template in place (e.g. a stray
    # template file winning on reload): the aligner must refuse, not print
    # success over a checkpoint that still serves the wrong bytes.
    out = _base_export(tmp_path, "run_qwen3_8b_merged", QWEN_SPECIALS, "<|endoftext|>")
    monkeypatch.setattr(PreTrainedTokenizerFast, "save_pretrained", lambda self, *a, **k: None)
    with pytest.raises(RuntimeError, match="did not take"):
        training.align_eos_with_template(out)


def test_align_chat_family_override_and_noop_without_family(tmp_path):
    out = _base_export(tmp_path, "run_nofamily_merged", QWEN_SPECIALS, "<|endoftext|>")
    before = (out / "generation_config.json").read_text()
    training.align_eos_with_template(out)
    assert (out / "generation_config.json").read_text() == before
    training.align_eos_with_template(out, chat_family="qwen")
    assert AutoTokenizer.from_pretrained(out).eos_token == "<|im_end|>"
