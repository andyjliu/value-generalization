"""Training-time chat templates must byte-match eval-time formatting.

JSONL pair datasets are rendered through ``training.CHAT_TEMPLATES[family]``
(pinned onto the tokenizer by ``_pin_chat_template``), while the eval side
renders through conflictscope's hand-rolled
``model_wrappers.format_messages_for_{family}``. These tests hold the two
byte-identical: exact-string tests for the families the configs actually
train (no heavy imports), plus a parity sweep against the fork itself when
vllm is importable.

Known, deliberate deviation: the fork appends the next-turn tag
unconditionally; the templates gate it on ``add_generation_prompt`` so DPO
completions don't end with a user-turn opener. The parity tests encode that
as ``TRAILING_TAG_WHEN_ASSISTANT_LAST``.
"""

import copy

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from valuegen.ground_truth import training

USER = [{"role": "user", "content": "What is 2+2?"}]
SYS_USER = [{"role": "system", "content": "Be terse."}, *USER]
FULL = [*USER, {"role": "assistant", "content": "4."}]
SYS_FULL = [*SYS_USER, {"role": "assistant", "content": "4."}]
MULTI = [*FULL, {"role": "user", "content": "And 3+3?"}]

# What the fork appends after an assistant-final conversation (it always
# opens the next turn); the training templates deliberately don't.
TRAILING_TAG_WHEN_ASSISTANT_LAST = {
    "tulu": "",
    "olmo": "",
    "olmo-3": "\n<|im_start|>user\n",
    "llama": "<|start_header_id|>user<|end_header_id|>\n\n",
    "qwen": "\n<|im_start|>user\n",
    "gemma": "<start_of_turn>user\n",
    "mistral": "",
}


# The fork has no formatter of its own for olmo-3: its 'olmo-3' branch
# delegates to format_messages_for_qwen (ChatML).
FORK_FORMATTER = {"olmo-3": "qwen"}

# Where a family's final assistant turn deliberately ends differently from the
# fork's formatter: (training bytes, fork bytes). Olmo-3 ChatML closes the
# final assistant turn with eos (<|endoftext|>), the fork's qwen formatter
# with <|im_end|>; only the prompt side has to match at eval.
FINAL_ENDER_VS_FORK = {"olmo-3": ("<|endoftext|>", "<|im_end|>")}


def render(family: str, messages, add_generation_prompt: bool) -> str:
    # Same Jinja environment settings transformers uses for chat templates.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(training.CHAT_TEMPLATES[family]).render(
        messages=copy.deepcopy(messages),
        add_generation_prompt=add_generation_prompt,
    )


def test_every_family_has_a_template():
    assert set(training.CHAT_TEMPLATES) == set(training.CHAT_FAMILIES)


# ── Exact renders for the families the experiment configs train ─────────────


def test_olmo_render_exact():
    assert render("olmo", USER, True) == (
        "<|endoftext|><|user|>\nWhat is 2+2?\n<|assistant|>\n"
    )
    assert render("olmo", FULL, False) == (
        "<|endoftext|><|user|>\nWhat is 2+2?\n<|assistant|>\n4.<|endoftext|>\n"
    )
    assert render("olmo", SYS_USER, True) == (
        "<|endoftext|><|user|>\nBe terse.\n\nWhat is 2+2?\n<|assistant|>\n"
    )


def test_olmo3_render_exact():
    # Olmo-3 ChatML == the checkpoint's own chat_template.jinja
    # (value-generalization/neutral-sft-v3-olmo3-7b, tool turn gated off):
    # system as its own turn, <|im_end|> after every turn but the final
    # assistant one, which ends in eos.
    assert render("olmo-3", USER, True) == (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert render("olmo-3", SYS_USER, True) == (
        "<|im_start|>system\nBe terse.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert render("olmo-3", FULL, False) == (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n4.<|endoftext|>"
    )
    assert render("olmo-3", MULTI, True) == (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n4.<|im_end|>\n"
        "<|im_start|>user\nAnd 3+3?<|im_end|>\n<|im_start|>assistant\n"
    )


def test_olmo3_family_dispatch():
    # Hyphenated spelling (Olmo-3 post-trains and checkpoints derived from
    # them) -> ChatML; our own un-hyphenated olmo3_* dirs keep the OLMo-2
    # template they were trained under; tulu still wins first.
    assert training.infer_chat_family("allenai/Olmo-3-7B-Instruct-SFT") == "olmo-3"
    assert training.infer_chat_family(
        "/tmp/valuegen-stage/merged/neutral_sft_v3_olmo-3_7b_merged") == "olmo-3"
    assert training.infer_chat_family("neutral_sft_v3_olmo3_7b_merged") == "olmo"
    # The raw base's HF id is hyphenated too, so a base-init run now trains
    # under ChatML -- which is what the fork already served for that name.
    assert training.infer_chat_family("allenai/Olmo-3-1025-7B") == "olmo-3"
    assert training.infer_chat_family("allenai/OLMo-2-1124-7B-SFT") == "olmo"
    assert training.infer_chat_family("allenai/Llama-3.1-Tulu-3-8B-SFT") == "tulu"
    assert training._TEMPLATE_TURN_ENDER["olmo-3"] == "<|endoftext|>"


def test_tulu_render_exact():
    assert render("tulu", USER, True) == "<|user|>\nWhat is 2+2?\n<|assistant|>\n"
    assert render("tulu", FULL, False) == (
        "<|user|>\nWhat is 2+2?\n<|assistant|>\n4.<|endoftext|>\n"
    )


def test_qwen_render_exact():
    assert render("qwen", USER, True) == (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert render("qwen", FULL, False) == (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n4.<|im_end|>"
    )


# ── trl boundary property ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "family", [f for f in training.CHAT_FAMILIES if f != "llama"]
)
def test_prompt_render_is_prefix_of_full_render(family):
    """trl slices chosen/rejected off at the common prefix of render(prompt)
    and render(prompt+completion); when the former is a strict prefix, the
    whole generation context lands on the prompt side, exactly as at eval.
    (llama is exempt: the fork joins turns with a separator but appends the
    generation prompt without one, so the boundary slides to the last
    <|eot_id|> — the concatenated bytes still match eval.)"""
    prompt = render(family, USER, True)
    full = render(family, FULL, False)
    assert full.startswith(prompt)
    assert full != prompt


# ── Template pinning in the trainers ─────────────────────────────────────────


class _Tok:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


def test_pin_chat_template_overrides_with_family_template():
    tok = _Tok(chat_template="the tokenizer's own template")
    training._pin_chat_template(tok, "allenai/OLMo-2-1124-7B-SFT", "x.jsonl")
    assert tok.chat_template == training.CHAT_TEMPLATES["olmo"]
    # Tulu HF names contain "Llama"; tulu must win (dispatch order).
    tok = _Tok()
    training._pin_chat_template(tok, "allenai/Llama-3.1-Tulu-3-8B-SFT", "x.jsonl")
    assert tok.chat_template == training.CHAT_TEMPLATES["tulu"]


def test_pin_chat_template_rejects_conversational_data_without_family():
    with pytest.raises(ValueError, match="chat family"):
        training._pin_chat_template(_Tok(), "some/unknown-model", "x.jsonl")


def test_pin_chat_template_keeps_legacy_csv_behavior():
    # CSV data never touches the template: unknown family is fine, and the
    # SIMPLE_CHAT_TEMPLATE fallback still covers template-less tokenizers.
    tok = _Tok(chat_template="original")
    training._pin_chat_template(tok, "some/unknown-model", "x.csv")
    assert tok.chat_template == "original"
    tok = _Tok(chat_template=None)
    training._pin_chat_template(tok, "some/unknown-model", "x.csv")
    assert tok.chat_template == training.SIMPLE_CHAT_TEMPLATE


# ── Byte parity against the fork (needs the [vllm] extra) ───────────────────


def _fork_formatter(family: str):
    pytest.importorskip("vllm")
    from valuegen._external import load_conflictscope

    mw = load_conflictscope().model_wrappers
    return getattr(mw.VLLMClient, f"format_messages_for_{FORK_FORMATTER.get(family, family)}")


@pytest.mark.parametrize("family", training.CHAT_FAMILIES)
@pytest.mark.parametrize(
    "messages", [USER, SYS_USER, MULTI], ids=["user", "sys_user", "multi_turn"]
)
def test_user_final_matches_fork(family, messages):
    # User-final == the prompt side: the fork appends its generation prompt,
    # the template's add_generation_prompt branch must produce the same bytes.
    fork = _fork_formatter(family)
    expected = fork(None, copy.deepcopy(messages))
    assert render(family, messages, True) == expected


@pytest.mark.parametrize("family", training.CHAT_FAMILIES)
@pytest.mark.parametrize(
    "messages", [FULL, SYS_FULL], ids=["full", "sys_full"]
)
def test_assistant_final_matches_fork(family, messages):
    # Assistant-final == prompt+completion: identical to the fork up to its
    # unconditional next-turn opener, which training must not emit.
    fork = _fork_formatter(family)
    expected = fork(None, copy.deepcopy(messages))
    rendered = render(family, messages, False)
    if family in FINAL_ENDER_VS_FORK:
        ours, forks = FINAL_ENDER_VS_FORK[family]
        assert rendered.endswith(ours)
        rendered = rendered[: -len(ours)] + forks
    assert rendered + TRAILING_TAG_WHEN_ASSISTANT_LAST[family] == expected


# ── Zero-shot scaffolds (raw base models) ────────────────────────────────────


def test_urial0_instruction_bytes():
    # The trailing-space-before-newline quirks are part of the published
    # URIAL prompt; a "cleanup" here would silently change the eval.
    assert training.URIAL_INSTRUCTION.startswith("# Instruction\n\nBelow is a list")
    assert "(you). \n" in training.URIAL_INSTRUCTION
    assert training.URIAL_INSTRUCTION.endswith("problem at hand.\n")


def test_scaffold_spec_carries_stops_not_fences():
    spec = training.scaffold_spec("urial0")
    assert spec["chat_template"] == training.SCAFFOLD_TEMPLATES["urial0"]
    assert spec["stop"] == ["\n# Query:", "\n# Instruction"]
    assert "```" not in spec["stop"]  # stopping on a fence truncates code answers
    assert spec["strip_suffix"] == "```"
    with pytest.raises(KeyError, match="Unknown scaffold"):
        training.scaffold_spec("urial9")


# ── Explicit chat formats as base-slot scaffolds ─────────────────────────────


@pytest.mark.parametrize("fmt", ["qwen_chatml", "olmo3_chatml"])
def test_chat_format_scaffold_is_template_only(fmt):
    from valuegen.ground_truth.chat_formats import CHAT_FORMATS

    # No extra stops or suffix stripping: the model stops on its own eos, as
    # it does without a scaffold; only the formatter changes.
    assert training.scaffold_spec(fmt) == {"chat_template": CHAT_FORMATS[fmt].template}


@pytest.mark.parametrize("fmt", ["qwen_chatml", "olmo3_chatml"])
@pytest.mark.parametrize(
    "messages", [USER, SYS_USER, MULTI], ids=["user", "sys_user", "multi_turn"]
)
def test_chat_format_scaffold_prompt_matches_fork(fmt, messages):
    # A chat base referenced by its HF name is evaluated under this scaffold;
    # its prompt bytes must equal the fork's ChatML formatter, which is what
    # base evals under ChatML-dispatching names have always used.
    from valuegen.ground_truth.chat_formats import CHAT_FORMATS

    fork = _fork_formatter("qwen")
    assert CHAT_FORMATS[fmt].render(messages, True) == fork(None, copy.deepcopy(messages))
