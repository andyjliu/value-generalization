"""Scaffold templates for pretrained (non-chat) models: ``urial0``.

Not chat families — nothing dispatches to them by model name; persona and
grad_proj opt in with ``template=urial0`` and the rendered bytes key the
vectors, the matrix name and the identity.
"""

import pandas as pd
import pytest

from valuegen.elicitation import datasets as D
from valuegen.ground_truth import training
from valuegen.predictors import grad_proj, grad_worker, persona

SYS_USER = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "What is 2+2?"},
]
SYS_FULL = [*SYS_USER, {"role": "assistant", "content": "4."}]


def render(name, messages, add_generation_prompt):
    return training.render_chat_template(
        training.SCAFFOLD_TEMPLATES[name], messages, add_generation_prompt
    )


def test_urial0_render_exact():
    prompt = render("urial0", SYS_USER, True)
    assert prompt.startswith(training.URIAL_INSTRUCTION)
    assert prompt.endswith(
        "Be terse.\n\n# Query:\n```\nWhat is 2+2?\n```\n\n# Answer:\n```\n"
    )
    assert render("urial0", SYS_FULL, False) == prompt + "4.\n```"
    # No system: one blank line between the preamble and the query.
    assert render("urial0", SYS_USER[1:], True) == (
        training.URIAL_INSTRUCTION + "\n# Query:\n```\nWhat is 2+2?\n```\n\n# Answer:\n```\n"
    )


def test_urial0_is_not_a_chat_family():
    assert "urial0" not in training.CHAT_FAMILIES
    assert training.infer_chat_family("org/urial0-model") is None


def test_grad_worker_pins_scaffold():
    class Tok:
        chat_template = "own"

    tok = Tok()
    assert grad_worker.pin_template(tok, "org/Qwen3-8B-Base", "urial0") == "scaffold:urial0"
    assert tok.chat_template == training.SCAFFOLD_TEMPLATES["urial0"]
    assert grad_worker.template_kwargs("scaffold:urial0") == {}
    assert grad_proj.template_of({"template": "urial0"}) == "urial0"
    assert grad_proj._vector_kind({"template": "urial0"}) == f"{grad_proj.KIND}_urial0"


def test_split_and_recover_system_prompts():
    chatml = "<|im_start|>system\nS1<|im_end|>\n<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n"
    olmo = "<|endoftext|><|system|>\nS2\n<|user|>\nQ2\n<|assistant|>\n"
    assert D.split_rendered_prompt(chatml) == ("chatml", "S1", "Q1")
    assert D.split_rendered_prompt(olmo) == ("olmo_tulu", "S2", "Q2")
    assert D.split_rendered_prompt("plain text") is None

    df = pd.DataFrame({
        "question": ["Q1", "Q2"], "system_prompt": ["", float("nan")],
        "answer": ["a", "b"], "prompt": [chatml, olmo],
    })
    out = D.recover_system_prompts(df)
    assert out.system_prompt.tolist() == ["S1", "S2"]
    # Existing system prompts win; frames without `prompt` pass through.
    df2 = df.assign(system_prompt=["keep", ""])
    assert D.recover_system_prompts(df2).system_prompt.tolist() == ["keep", "S2"]
    passthrough = D.recover_system_prompts(df.drop(columns="prompt"))
    assert passthrough.system_prompt.iloc[0] == ""
    with pytest.raises(ValueError, match="could not recover"):
        D.recover_system_prompts(df.assign(prompt=["plain", olmo]))


def test_persona_template_keys_paths_and_name(tmp_path):
    cfg = {"model": "org/base", "values": ["v"], "template": "urial0"}
    with pytest.raises(ValueError, match="expected one of"):
        persona.template_of({"template": "bogus"})
    assert persona.template_of({}) is None
    assert persona.template_of(cfg) == "urial0"
    assert "template" in persona.PREDICTOR_PARAMS
    # A scaffold never borrows fork-native vectors and never sweeps.
    art = D.Artifact(root=tmp_path, cfg={"method": "default_llm", "experiment": "x",
                                         "values": ["v"], "artifact": "pairs"})
    assert persona._fork_vec_dir(cfg, art) is None
    assert D.fork_compat_dir(art, "org/base", "urial0") == tmp_path / "fork_compat" / "base" / "urial0"
    assert D.fork_compat_dir(art, "org/base") == tmp_path / "fork_compat" / "base"


def test_split_rendered_prompt_urial0():
    prompt = render("urial0", SYS_USER, True)
    assert D.split_rendered_prompt(prompt) == ("urial0", "Be terse.", "What is 2+2?")
    # No system paragraph: nothing to recover, same as the chat families.
    assert D.split_rendered_prompt(render("urial0", SYS_USER[1:], True)) is None
    df = pd.DataFrame({"prompt": [prompt], "system_prompt": [""]})
    assert D.recover_system_prompts(df)["system_prompt"].tolist() == ["Be terse."]
