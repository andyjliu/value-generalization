"""Parity for the weight_steer base-model + scaffold arm.

The base+urial weight_steer arm trains LoRAs on ``input_output`` segments that
``ws_scaffold`` renders from cs_prep_data's message pairs. For the arm to be
comparable to persona/grad_proj (which read activations under the same
scaffold), the segment text must byte-match ``training.SCAFFOLD_TEMPLATES`` and
split the response span the way grad_worker does. These tests lock that; the
tokenizer-level check against a real base view runs only when the view exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from valuegen.ground_truth.training import SCAFFOLD_TEMPLATES, render_chat_template
from valuegen.predictors.ws_scaffold import to_segments

MESSAGES = [
    {"role": "user", "content": "Should I disclose the flaw to the customer?"},
    {"role": "assistant", "content": "Yes — be transparent about the defect."},
]


@pytest.mark.parametrize("scaffold", sorted(SCAFFOLD_TEMPLATES))
def test_segments_reconstruct_full_render(scaffold):
    """prompt + answer segments concatenate to the full scaffold render, and the
    prompt segment is exactly the generation-prompt render — so axolotl trains on
    the same bytes the scaffold produces, with the answer masked in."""
    rec = to_segments(MESSAGES, scaffold)
    (prompt_seg, prompt_label), (answer_seg, answer_label) = (
        (s["text"], s["label"]) for s in rec["segments"]
    )
    assert prompt_label is False and answer_label is True

    full = render_chat_template(
        SCAFFOLD_TEMPLATES[scaffold], MESSAGES, add_generation_prompt=False
    )
    prompt_only = render_chat_template(
        SCAFFOLD_TEMPLATES[scaffold], MESSAGES[:-1], add_generation_prompt=True
    )
    assert prompt_seg == prompt_only
    assert prompt_seg + answer_seg == full
    assert answer_seg  # response span is non-empty (something is trained on)
    assert MESSAGES[-1]["content"] in answer_seg


def test_urial0_answer_closes_the_fence():
    """urial0 opens the answer fence in the prompt; the assistant span must close
    it — the terminating signal the base-template probe found urial0 gets right."""
    rec = to_segments(MESSAGES, "urial0")
    prompt_seg, answer_seg = (s["text"] for s in rec["segments"])
    assert prompt_seg.endswith("# Answer:\n```\n")
    assert answer_seg.endswith("\n```")


def test_missing_assistant_turn_raises():
    with pytest.raises(ValueError):
        to_segments(MESSAGES[:1], "urial0")


@pytest.mark.parametrize("scaffold", sorted(SCAFFOLD_TEMPLATES))
def test_render_matches_tokenizer_apply_chat_template(scaffold):
    """render_chat_template must equal a real base tokenizer's apply_chat_template
    once the scaffold is pinned — otherwise ws_scaffold's bytes drift from what
    grad_worker/persona actually feed the model. Skipped when no base view is
    present (CI); set WS_URIAL_TEST_VIEWS to a model_views dir to opt in."""
    views = os.environ.get("WS_URIAL_TEST_VIEWS")
    if not views:
        pytest.skip("WS_URIAL_TEST_VIEWS not set")
    candidates = [Path(views) / "Qwen3-8B-Base", Path(views) / "Olmo-3-1025-7B"]
    view = next((c for c in candidates if (c / "config.json").is_file()), None)
    if view is None:
        pytest.skip("no base model view available for tokenizer parity")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(view))
    tok.chat_template = SCAFFOLD_TEMPLATES[scaffold]
    expected_full = tok.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=False
    )
    got_full = render_chat_template(
        SCAFFOLD_TEMPLATES[scaffold], MESSAGES, add_generation_prompt=False
    )
    assert got_full == expected_full
