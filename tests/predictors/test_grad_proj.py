"""grad_proj predictor: layer inheritance, stage plumbing, and worker math."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from valuegen.elicitation import datasets as D
from valuegen.predictors import get_predictor, grad_proj, grad_worker, persona, store

VALUES = ["alpha", "beta", "missing"]


def _artifact(tmp_path: Path) -> D.Artifact:
    cfg = {
        "method": "test_pairs",
        "artifact": "pairs",
        "schema_version": 1,
        "value_set": "test",
        "values": list(VALUES),
        "legacy": False,
        "model": "policy/model",
    }
    return D.Artifact(tmp_path / "artifact", cfg)


def test_registry_contract():
    predictor = get_predictor("grad_proj")
    assert predictor.requires == "pairs"
    assert predictor.default_data == "default_llm"


def test_layer_is_pinned_or_inherited_from_persona_sweep_never_fit(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES}
    assert grad_proj.resolve_layer({**cfg, "layer": 7}, cluster, artifact) == 7
    # No persona sweep and no --layer: refuse rather than fall back to
    # anything that has seen the target matrix.
    with pytest.raises(ValueError, match="No layer available"):
        grad_proj.resolve_layer(cfg, cluster, artifact)

    sweep_dir = store.sweeps_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    sweep_dir.mkdir(parents=True)
    pd.DataFrame({"layer": [0, 1, 2], "mean_trait_score": [0.1, 0.9, 0.4]}).to_csv(
        sweep_dir / persona.AGGREGATED_SWEEP, index=False
    )
    assert grad_proj.resolve_layer(cfg, cluster, artifact) == 1
    # A different model must not find this sweep (model_short keys the path).
    with pytest.raises(ValueError, match="No layer available"):
        grad_proj.resolve_layer({**cfg, "model": "org/other"}, cluster, artifact)


def test_extraction_stage_commands_and_resume(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:2], "gpus": 1}
    assert grad_proj.needs_slurm(cfg, cluster, artifact)
    stage = grad_proj._extraction_stage(cfg, cluster, artifact)
    assert stage.env == "default" and stage.gpus == 1 and len(stage.tasks) == 2
    vec_dir = grad_proj._vectors_dir(cfg, cluster, artifact)
    for value, task in zip(VALUES, stage.tasks):
        assert "-m valuegen.predictors.grad_worker" in task.command
        assert f"--pos {artifact.pos_path(value)}" in task.command
        assert "--template family" in task.command
        assert task.done == vec_dir / f"{value}_{grad_proj.KIND}.pt"

    vec_dir.mkdir(parents=True)
    for value in VALUES[:2]:
        torch.save(torch.ones(3, 4), vec_dir / f"{value}_{grad_proj.KIND}.pt")
    assert not grad_proj.needs_slurm(cfg, cluster, artifact)
    assert grad_proj.plan(cfg, cluster, artifact) == []


def test_run_builds_similarity_from_existing_vectors(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {
        "model": "org/base", "values": VALUES, "layer": 1,
        "predictor": "grad_proj", "schema_version": 1,
    }
    vec_dir = grad_proj._vectors_dir(cfg, cluster, artifact)
    store.claim_dir(vec_dir, artifact.artifact_id)
    torch.save(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), vec_dir / f"alpha_{grad_proj.KIND}.pt")
    torch.save(torch.tensor([[1.0, 0.0], [1.0, 1.0]]), vec_dir / f"beta_{grad_proj.KIND}.pt")
    # A present but unusable zero vector exercises cosine_grid's NaN padding
    # without making the predictor correctly schedule a missing extraction.
    torch.save(torch.zeros(2, 2), vec_dir / f"missing_{grad_proj.KIND}.pt")
    out = grad_proj.run(cfg, cluster, artifact)
    assert out.name == "gradproj_dpoinit_respavg_L1.npy"
    matrix, values = store.load_similarity(out)
    assert values == VALUES
    assert np.allclose(matrix[:2, :2], [[1, 2 ** -0.5], [2 ** -0.5, 1]])
    assert np.isnan(matrix[2]).all()
    provenance = yaml.safe_load(
        out.with_name(out.stem + "_provenance.yaml").read_text()
    )
    assert provenance["layer_source"] == "pinned"
    assert provenance["identity"] == {"loss": "dpoinit", "pooling": "respavg", "layer": 1}


# ── worker math on a tiny model ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def _batch(seqs: list[tuple[list[int], int]], signs: list[float]):
    max_t = max(len(ids) for ids, _ in seqs)
    input_ids = torch.zeros(len(seqs), max_t, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_t, dtype=torch.long)
    resp_mask = torch.zeros(len(seqs), max_t, dtype=torch.bool)
    for b, (ids, start) in enumerate(seqs):
        input_ids[b, :len(ids)] = torch.tensor(ids)
        attention_mask[b, :len(ids)] = 1
        resp_mask[b, start:len(ids)] = True
    return input_ids, attention_mask, resp_mask, torch.tensor(signs)


def test_identical_pos_neg_gives_zero_update(tiny_model):
    # A pair whose chosen and rejected responses are the same tokens has zero
    # DPO-at-init gradient — the signed losses cancel exactly.
    seq = ([1, 2, 3, 4, 5, 6], 3)
    pooled = grad_worker.batch_update_grads(
        tiny_model, *_batch([seq, seq], [1.0, -1.0])
    )
    assert pooled.shape[0] == tiny_model.config.num_hidden_layers + 1
    update = 0.5 * (pooled[:, 0] + pooled[:, 1])
    assert torch.allclose(update, torch.zeros_like(update), atol=1e-6)


def test_batched_grads_match_single_pair_runs(tiny_model):
    # Summed batch loss ⇒ each sequence's activation rows carry exactly its
    # own loss gradient, so one backward per microbatch is exact per-example.
    pos_a, neg_a = ([1, 2, 3, 4, 5], 2), ([1, 2, 3, 7, 8], 2)
    pos_b, neg_b = ([9, 10, 11, 12], 1), ([9, 10, 13, 14, 15], 1)

    together = grad_worker.batch_update_grads(
        tiny_model, *_batch([pos_a, neg_a, pos_b, neg_b], [1.0, -1.0, 1.0, -1.0])
    )
    alone_a = grad_worker.batch_update_grads(
        tiny_model, *_batch([pos_a, neg_a], [1.0, -1.0])
    )
    alone_b = grad_worker.batch_update_grads(
        tiny_model, *_batch([pos_b, neg_b], [1.0, -1.0])
    )
    assert torch.allclose(together[:, :2], alone_a, atol=1e-5)
    assert torch.allclose(together[:, 2:], alone_b, atol=1e-5)


def test_freezing_params_does_not_change_grads(tiny_model):
    # Freezing is a memory optimisation only: the activation gradients must be
    # bitwise identical, or vectors already on disk stop being comparable.
    args = _batch([([1, 2, 3, 4, 5], 2), ([1, 2, 3, 7, 8], 2)], [1.0, -1.0])
    trainable = grad_worker.batch_update_grads(tiny_model, *args)

    grad_worker.freeze_params(tiny_model)
    assert sum(p.requires_grad for p in tiny_model.parameters()) == 1
    frozen = grad_worker.batch_update_grads(tiny_model, *args)
    assert torch.equal(trainable, frozen)

    # Restore: the fixture is module-scoped and later tests read grads too.
    tiny_model.requires_grad_(True)


def test_update_direction_lowers_the_loss(tiny_model):
    # Directional-derivative sanity: nudging the layer-0 activations along the
    # pooled update direction (−∂L/∂a) must decrease the DPO-at-init loss.
    seqs = [([1, 2, 3, 4, 5], 2), ([1, 2, 6, 7], 2)]
    input_ids, attention_mask, resp_mask, sign = _batch(seqs, [1.0, -1.0])

    def loss_with_offset(delta):
        embed = tiny_model.get_input_embeddings()
        handle = None
        if delta is not None:
            handle = embed.register_forward_hook(lambda m, i, o: o + delta)
        try:
            out = tiny_model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            if handle:
                handle.remove()
        logprobs = torch.log_softmax(out.logits[:, :-1], dim=-1)
        token_lp = logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        target = (resp_mask[:, 1:] & attention_mask[:, 1:].bool()).float()
        return -(sign * (token_lp * target).sum(-1)).sum()

    pooled = grad_worker.batch_update_grads(
        tiny_model, input_ids, attention_mask, resp_mask, sign
    )
    # Broadcast the per-sequence layer-0 update over all that sequence's
    # positions — a crude steer, but the first-order term must still win.
    delta = 1e-3 * pooled[0].unsqueeze(1)  # [B, 1, H]
    with torch.no_grad():
        base, steered = loss_with_offset(None), loss_with_offset(delta)
    assert steered < base


def test_encode_pair_side_boundary_and_truncation():
    class StubTokenizer:
        chat_template = "unused"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
            return text + ("<assistant>" if add_generation_prompt else "")

        def __call__(self, text, add_special_tokens):
            return {"input_ids": [ord(c) for c in text]}

    tok = StubTokenizer()
    ids, resp_start = grad_worker.encode_pair_side(tok, "hi", "yo", "", max_len=100)
    # response starts right after the generation prompt (an exact prefix here)
    assert "".join(map(chr, ids[resp_start:])) == "yo"
    assert "".join(map(chr, ids[:resp_start])).endswith("<assistant>")
    # a truncation that eats the whole response is a skip, not a zero-token row
    assert grad_worker.encode_pair_side(tok, "hi", "yo", "", max_len=5) is None
    # system prompt is included when present, dropped when blank
    with_sys, _ = grad_worker.build_messages("q", "a", "be nice")
    without, _ = grad_worker.build_messages("q", "a", "")
    assert with_sys[0]["role"] == "system" and without[0]["role"] == "user"


def test_native_template_is_its_own_vector_kind_and_identity(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/neutral-sft-qwen3", "values": VALUES[:1], "gpus": 1,
           "template": "native"}
    stage = grad_proj._extraction_stage(cfg, cluster, artifact)
    task = stage.tasks[0]
    assert "--template native" in task.command
    kind = f"{grad_proj.KIND}_native"
    assert task.done.name == f"alpha_{kind}.pt"
    # family vectors on disk do not satisfy a native request
    vec_dir = grad_proj._vectors_dir(cfg, cluster, artifact)
    vec_dir.mkdir(parents=True)
    torch.save(torch.ones(3, 4), vec_dir / f"alpha_{grad_proj.KIND}.pt")
    assert grad_proj.needs_slurm(cfg, cluster, artifact)
    with pytest.raises(ValueError, match="template"):
        grad_proj.template_of({"template": "bogus"})


def test_pin_template_native_keeps_tokenizer_template_and_disables_thinking():
    class Tok:
        chat_template = "{{ messages }}"

    tok = Tok()
    assert grad_worker.pin_template(tok, "org/neutral-sft-qwen3", "native") == "native"
    assert tok.chat_template == "{{ messages }}"  # not overwritten by the qwen family
    assert grad_worker.template_kwargs("native") == {"enable_thinking": False}
    assert grad_worker.template_kwargs("family:qwen") == {}
    tok.chat_template = None
    with pytest.raises(ValueError, match="no chat template"):
        grad_worker.pin_template(tok, "org/neutral-sft-qwen3", "native")
    with pytest.raises(ValueError, match="template mode"):
        grad_worker.pin_template(tok, "org/x", "bogus")


def test_encode_pair_side_forwards_render_kwargs():
    class StubTokenizer:
        chat_template = "unused"
        seen = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            self.seen.append(kw)
            think = "<think></think>" if kw.get("enable_thinking") is False else ""
            if add_generation_prompt:
                ctx, tail = messages, ""
            else:
                ctx, tail = messages[:-1], messages[-1]["content"]
            text = "".join(f"<{m['role']}>{m['content']}" for m in ctx)
            return text + "<assistant>" + think + tail

        def __call__(self, text, add_special_tokens):
            return {"input_ids": [ord(c) for c in text]}

    tok = StubTokenizer()
    ids, resp_start = grad_worker.encode_pair_side(
        tok, "hi", "yo", "", max_len=100, render_kwargs={"enable_thinking": False}
    )
    assert tok.seen == [{"enable_thinking": False}] * 2
    # the empty think block lands in the prompt, not the response
    assert "".join(map(chr, ids[resp_start:])) == "yo"
    assert "".join(map(chr, ids[:resp_start])).endswith("<think></think>")


def test_hook_recorder_matches_output_hidden_states(tiny_model):
    """The worker's own hooks reproduce transformers' hidden_states layout
    exactly (single device, where both paths are valid)."""
    import torch
    ids = torch.tensor([[1, 5, 9, 2, 7, 3]])
    mask = torch.ones_like(ids)
    with grad_worker.record_hidden_states(tiny_model) as recorded:
        out = tiny_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    assert len(recorded) == len(out.hidden_states) == tiny_model.config.num_hidden_layers + 1
    for ours, theirs in zip(recorded, out.hidden_states):
        assert torch.equal(ours, theirs)


def test_non_finite_gradients_fail_the_task(tiny_model):
    # A NaN in the residual stream (what the broken-P2P L40S nodes produce
    # under a multi-GPU device_map) must raise, never be pooled into a vector.
    import pytest
    layer = tiny_model.model.layers[-1]
    handle = layer.register_forward_hook(
        lambda m, a, o: (o[0].mul(float("nan")),) + tuple(o[1:])
        if isinstance(o, tuple) else o.mul(float("nan"))
    )
    try:
        with pytest.raises(RuntimeError, match="non-finite hidden-state gradients"):
            grad_worker.batch_update_grads(
                tiny_model, *_batch([([1, 2, 3, 4], 2), ([1, 2, 5, 6], 2)], [1.0, -1.0])
            )
    finally:
        handle.remove()
