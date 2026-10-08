"""GPU worker for the grad_proj predictor: one value's update-direction stack.

Run as a module inside a SLURM array task (core env)::

    python -m valuegen.predictors.grad_worker \\
        --model allenai/Llama-3.1-Tulu-3-8B \\
        --pos data/.../{value}_pos.csv --neg data/.../{value}_neg.csv \\
        --out data/vectors/grad_proj/.../{value}_dpoinit_respavg_grad.pt

Per pair *i* (pos row *i* with neg row *i*, the pairs-artifact contract) the
DPO-at-init loss is ``L_i = −[log p(ans_pos | ctx) − log p(ans_neg | ctx)]``,
summed over response tokens (matching DPO's summed logprobs; the reference
model cancels at θ₀ — see :mod:`valuegen.predictors.grad_proj`). The saved
vector is the mean over pairs of the response-position-averaged **negative**
gradient of ``L_i`` w.r.t. the residual stream, i.e. the update direction the
first DPO step pushes activations toward, at every ``hidden_states`` index
(``[num_hidden_layers + 1, hidden]``, embeddings at 0 — the persona fork's
layer convention, load-bearing for sweep-layer inheritance).

Rendering goes through the pinned family chat template
(``training.CHAT_TEMPLATES``), the exact bytes the DPO trainer feeds — the
repo's training-text = eval-text convention. A model matching no family falls
back to the tokenizer's own template (or trl's SIMPLE_CHAT_TEMPLATE), with the
choice recorded in the sidecar ``*_meta.json``.

Batching note: with the batch loss summed over sequences, each sequence's
activation rows receive exactly their own loss term's gradient, so one
``autograd.grad`` per microbatch yields exact per-example gradients; the model
runs bf16 on CUDA, accumulation is fp32 on CPU.

Memory: a 7B backward at 8 sequences OOMs a 48 GB A6000, so params are frozen
and ``pairs_per_batch`` defaults to 1. Measured peak on OLMo-2-1124-7B-SFT
over the longest pairs: 34.9 GB trainable vs 28.0 GB frozen at 1 pair/batch;
4 pairs/batch OOMs either way. Neither change moves the numbers — freezing is
bitwise identical and batching is exact per-example, both asserted in
``tests/predictors/test_grad_proj.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path


def build_messages(question: str, answer: str, system_prompt: str = "") -> tuple[list, list]:
    """(prompt_messages, full_messages) for one pair row."""
    prompt = []
    if system_prompt and str(system_prompt).strip() and str(system_prompt) != "nan":
        prompt.append({"role": "system", "content": str(system_prompt)})
    prompt.append({"role": "user", "content": str(question)})
    return prompt, prompt + [{"role": "assistant", "content": str(answer)}]


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ``urial0`` (and any other ``training.SCAFFOLD_TEMPLATES`` name) renders a
# plain-text scaffold for pretrained models whose chat tags are untrained.
TEMPLATE_MODES = ("family", "native", "urial0")
# ``native``: hybrid-thinking templates (Qwen3) only put the empty
# ``<think>\n\n</think>`` block in the generation prompt when thinking is
# disabled; checkpoints SFT'd under that template saw it on every assistant
# turn, so the prompt render must carry it too or the response boundary lands
# in the wrong place. Non-thinking templates ignore the kwarg.
NATIVE_TEMPLATE_KWARGS = {"enable_thinking": False}


def pin_template(tokenizer, model_name: str, mode: str = "family") -> str:
    """Pin the chat template (training bytes); returns the source used.

    ``family`` pins ``training.CHAT_TEMPLATES`` by model-name substring (the
    fork-trained checkpoints); ``native`` keeps the tokenizer's own template
    (checkpoints trained under it, e.g. the neutral SFT models) and fails if
    the tokenizer ships none.
    """
    from valuegen.ground_truth.training import (
        CHAT_TEMPLATES, SCAFFOLD_TEMPLATES, SIMPLE_CHAT_TEMPLATE, infer_chat_family,
    )

    if mode not in TEMPLATE_MODES:
        raise ValueError(f"template mode {mode!r}; expected one of {TEMPLATE_MODES}")
    if mode in SCAFFOLD_TEMPLATES:
        tokenizer.chat_template = SCAFFOLD_TEMPLATES[mode]
        return f"scaffold:{mode}"
    if mode == "native":
        if tokenizer.chat_template is None:
            raise ValueError(
                f"{model_name} ships no chat template; --template native needs one"
            )
        return "native"
    family = infer_chat_family(model_name)
    if family is not None:
        tokenizer.chat_template = CHAT_TEMPLATES[family]
        return f"family:{family}"
    if tokenizer.chat_template is not None:
        print(f"warning: {model_name} matches no chat family; using the "
              "tokenizer's own template (not byte-matched to training)")
        return "tokenizer"
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    print(f"warning: {model_name} matches no chat family and has no template; "
          "using SIMPLE_CHAT_TEMPLATE")
    return "simple"


def template_kwargs(template_source: str) -> dict:
    """``apply_chat_template`` kwargs for a ``pin_template`` source."""
    return dict(NATIVE_TEMPLATE_KWARGS) if template_source == "native" else {}


def encode_pair_side(tokenizer, question, answer, system_prompt, max_len,
                     render_kwargs: dict | None = None):
    """Token ids + response start for one sequence, or None if the response
    would be entirely truncated away.

    The response boundary is the common token prefix of the generation-prompt
    render and the full render: for most families the former is an exact
    prefix; where a separator differs (llama's ``\\n\\n`` before the assistant
    header) the boundary slides a few tokens earlier, pulling the header into
    the "response" — consistent across pos/neg and across values, so the diffs
    stay comparable.
    """
    prompt_msgs, full_msgs = build_messages(question, answer, system_prompt)
    render_kwargs = render_kwargs or {}
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True, **render_kwargs
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False, **render_kwargs
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]
    resp_start = common_prefix_len(prompt_ids, full_ids)
    if resp_start >= len(full_ids):
        return None
    return full_ids, resp_start


def freeze_params(model) -> None:
    """Drop parameter grads so backward stops saving layer inputs.

    The input embedding stays trainable: its output *is* ``hidden_states[0]``,
    and if nothing upstream requires grad no graph is built at all.
    """
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)


def describe_dead_hidden_states(model, hidden_states, dead) -> str:
    """Explain an autograd failure: which recorded activations the loss does
    not depend on, with per-entry device / grad_fn and the shard map."""
    rows = []
    for i, h in enumerate(hidden_states):
        fn = type(h.grad_fn).__name__ if h.grad_fn is not None else "leaf"
        rows.append(f"  [{i:3d}] {'DEAD' if i in dead else 'ok  '} dev={h.device} "
                    f"req_grad={h.requires_grad} grad_fn={fn}")
    dmap = getattr(model, "hf_device_map", None)
    return (
        f"{len(dead)}/{len(hidden_states)} hidden_states entries are not on the "
        f"loss graph: {dead}\n" + "\n".join(rows) + f"\nhf_device_map={dmap}"
    )


@contextlib.contextmanager
def record_hidden_states(model):
    """Collect the residual stream with the worker's own forward hooks.

    Yields a list filled during the next forward with the same layout as
    transformers' ``output_hidden_states``: ``[embeddings, layer_0 out, ...,
    layer_{L-2} out, final-norm out]`` (the fork's ``hidden_states`` indexing,
    which the inherited persona-sweep layer refers to). The tensors are the
    ones actually on the loss graph, on whichever shard produced them.

    ``output_hidden_states=True`` cannot be used under a multi-GPU
    ``device_map``: accelerate's top-level hook copies every returned tensor
    back to the first device, and those copies are side branches the loss
    never touched, so ``autograd.grad`` rejects them as unused (Qwen3-30B-A3B
    on 3 GPUs, 2026-09-05; a GPU+CPU split hides it because offloaded layers
    still execute on the first device and nothing is copied).
    """
    inner = getattr(model, model.base_model_prefix, model)
    layers, norm = inner.layers, inner.norm
    embed_out, layer_out, norm_out = [], [None] * len(layers), []

    def _tensor(output):
        return output[0] if isinstance(output, tuple) else output

    handles = [model.get_input_embeddings().register_forward_hook(
        lambda m, a, o: embed_out.append(o))]
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(
            lambda m, a, o, i=i: layer_out.__setitem__(i, _tensor(o))))
    handles.append(norm.register_forward_hook(lambda m, a, o: norm_out.append(o)))
    recorded: list = []
    try:
        yield recorded
        if not (len(embed_out) == 1 and len(norm_out) == 1 and all(t is not None for t in layer_out)):
            raise RuntimeError("hidden-state hooks did not all fire exactly once")
        recorded.extend([embed_out[0], *layer_out[:-1], norm_out[0]])
    finally:
        for h in handles:
            h.remove()


def batch_update_grads(model, input_ids, attention_mask, resp_mask, sign):
    """Per-sequence response-pooled negative loss gradients at every layer.

    ``L = −Σ_b sign_b · Σ_{t∈resp_b} log p(token_t)`` — sign +1 for chosen,
    −1 for rejected rows. Returns ``[n_hidden_states, B, hidden]`` fp32 CPU:
    the mean over each sequence's response positions of ``−∂L/∂a``.
    """
    import torch

    with record_hidden_states(model) as hidden_states:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    # log_softmax in the model dtype (it max-subtracts internally), fp32 only
    # after the vocab dim is gathered away — halves peak activation memory.
    logprobs = torch.log_softmax(out.logits[:, :-1], dim=-1)
    token_lp = logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).float()
    # token j is predicted at position j-1, so shift the response mask left.
    target_mask = (resp_mask[:, 1:] & attention_mask[:, 1:].bool()).to(token_lp.dtype)
    seq_lp = (token_lp * target_mask).sum(dim=-1)
    loss = -(sign.to(seq_lp.dtype) * seq_lp).sum()
    grads = torch.autograd.grad(loss, hidden_states, allow_unused=True)
    dead = [i for i, g in enumerate(grads) if g is None]
    if dead:
        raise RuntimeError(describe_dead_hidden_states(model, hidden_states, dead))

    pool = (resp_mask & attention_mask.bool()).unsqueeze(-1)  # [B, T, 1]
    counts = pool.sum(dim=(1, 2)).clamp(min=1)  # [B]
    pooled = [
        (-(g.float()) * pool.to(g.device)).sum(dim=1).cpu() / counts.unsqueeze(-1).cpu()
        for g in grads
    ]
    stacked = torch.stack(pooled, dim=0)  # [n_hidden_states, B, hidden]
    bad = [i for i in range(stacked.shape[0]) if not torch.isfinite(stacked[i]).all()]
    if bad:
        # Non-finite gradients arrive in whole device-shard blocks on certain
        # L40S x8 nodes under a multi-GPU device_map (2026-09-07: n5-32, o5-16,
        # o5-20, o5-28, p5-20, p5-28, q5-16, q5-20, q5-24 — same hardware,
        # healthy on other nodes). A NaN stack poisons the value's mean vector
        # at every affected layer, so fail the task instead of writing it.
        import socket
        raise RuntimeError(
            f"non-finite hidden-state gradients at layers {bad} on "
            f"{socket.gethostname()}; hf_device_map="
            f"{getattr(model, 'hf_device_map', None)}"
        )
    return stacked


def value_vector(model, tokenizer, pairs, max_len=2048, pairs_per_batch=1, device="cpu",
                 render_kwargs: dict | None = None):
    """Mean update-direction stack over ``pairs`` (list of (question,
    system_prompt, pos_answer, neg_answer)). Returns (tensor, n_used, n_skipped).
    """
    import torch

    total = None
    used = skipped = 0
    for start in range(0, len(pairs), pairs_per_batch):
        chunk, encoded = pairs[start:start + pairs_per_batch], []
        for question, system_prompt, pos_ans, neg_ans in chunk:
            sides = [
                encode_pair_side(tokenizer, question, ans, system_prompt, max_len,
                                 render_kwargs)
                for ans in (pos_ans, neg_ans)
            ]
            if any(s is None for s in sides):
                skipped += 1
                continue
            encoded.append(sides)
        if not encoded:
            continue

        seqs = [side for pair in encoded for side in pair]  # pos, neg, pos, ...
        max_t = max(len(ids) for ids, _ in seqs)
        input_ids = torch.zeros(len(seqs), max_t, dtype=torch.long)
        attention_mask = torch.zeros(len(seqs), max_t, dtype=torch.long)
        resp_mask = torch.zeros(len(seqs), max_t, dtype=torch.bool)
        for b, (ids, resp_start) in enumerate(seqs):
            input_ids[b, :len(ids)] = torch.tensor(ids)
            attention_mask[b, :len(ids)] = 1
            resp_mask[b, resp_start:len(ids)] = True
        sign = torch.tensor([1.0, -1.0] * len(encoded))

        pooled = batch_update_grads(
            model,
            input_ids.to(device),
            attention_mask.to(device),
            resp_mask.to(device),
            sign.to(device),
        )  # [L, B, H]
        # Per-pair update = mean of the pos and neg rows' pooled −grads (both
        # carry the pair's loss gradient; averaging matches respavg pooling).
        pair_vecs = 0.5 * (pooled[:, 0::2] + pooled[:, 1::2])  # [L, n_pairs, H]
        chunk_sum = pair_vecs.sum(dim=1)
        total = chunk_sum if total is None else total + chunk_sum
        used += len(encoded)

    if total is None:
        raise RuntimeError("no usable pairs: every row was skipped")
    return total / used, used, skipped


def main(argv: list[str] | None = None) -> None:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pos", required=True, type=Path)
    parser.add_argument("--neg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pairs-per-batch", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--template", choices=TEMPLATE_MODES, default="family",
                        help="family: pin training.CHAT_TEMPLATES by model name; "
                             "native: the tokenizer's own template, thinking off; "
                             "urial0: plain-text URIAL scaffold, no exemplars "
                             "(pretrained base models)")
    args = parser.parse_args(argv)

    from valuegen.elicitation.datasets import recover_system_prompts

    # default_llm artifacts keep the steering system prompt only inside the
    # rendered `prompt` column (system_prompt blank); recover it so the pair
    # context carries the instruction the answers were elicited under.
    pos = recover_system_prompts(pd.read_csv(args.pos))
    neg = recover_system_prompts(pd.read_csv(args.neg))
    if len(pos) != len(neg):
        raise ValueError(
            f"pos/neg row counts differ ({len(pos)} vs {len(neg)}); pairs "
            "artifacts pair rows positionally"
        )
    pairs = [
        (p.question, getattr(p, "system_prompt", ""), p.answer, n.answer)
        for p, n in zip(pos.itertuples(), neg.itertuples())
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template_source = pin_template(tokenizer, args.model, args.template)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device_map="auto" shards models that do not fit one GPU (32B-class
    # activation runs get --gpus 2+); inputs go to the first shard and each
    # layer's gradient is pooled on its own device (batch_update_grads).
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()  # no dropout; grads flow regardless
    model.config.use_cache = False  # nothing generates here; don't retain a KV cache
    freeze_params(model)

    stack, used, skipped = value_vector(
        model, tokenizer, pairs,
        max_len=args.max_len,
        pairs_per_batch=args.pairs_per_batch,
        device=device,
        render_kwargs=template_kwargs(template_source),
    )
    print(f"{args.pos.stem}: {used} pairs used, {skipped} skipped, "
          f"stack {tuple(stack.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".pt.tmp")
    torch.save(stack, tmp)
    os.replace(tmp, args.out)  # atomic: the done predicate is file existence
    args.out.with_name(args.out.stem + "_meta.json").write_text(json.dumps({
        "model": args.model,
        "n_pairs_used": used,
        "n_pairs_skipped": skipped,
        "template_source": template_source,
        "template_mode": args.template,
        "max_len": args.max_len,
        "n_hidden_states": int(stack.shape[0]),
    }, indent=2))


if __name__ == "__main__":
    main()
