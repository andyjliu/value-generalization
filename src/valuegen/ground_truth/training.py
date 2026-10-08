"""DPO/SFT training + LoRA merge.

Run as a module inside a training job::

    python -m valuegen.ground_truth.training dpo --config train.yaml \\
        --dataset_name data/.../dataset.jsonl --model_name_or_path allenai/... \\
        --output_dir ... --run_name ...

    python -m valuegen.ground_truth.training merge \\
        --base-model allenai/... --checkpoint-root <train output_dir> \\
        --merged-path .../merged/<exp>_<mtag>_<value>_merged

    python -m valuegen.ground_truth.training export \\
        --base-model allenai/... --trained <train output_dir> \\
        --out .../merged/<exp>_<mtag>_<value>_merged

``merge`` is the LoRA finishing step; ``export`` is its full-FT counterpart
(``use_peft: false``). There is no adapter to merge then, but the merge step
did three things besides merging that the eval still depends on — bf16
weights, the base tokenizer, a family-keyword destination — and ``export``
reproduces exactly those (see :func:`export_full_ft`). Both land at the same
``merged_path`` identity, so the manifest, done-checks, and GCS plumbing
never care which one produced the artifact.

**Hparams pass through, no parallel schema**: the ``dpo``/``sft`` subcommands
hand everything after the subcommand to ``TrlParser.parse_args_and_config``
(ScriptArguments/DPOConfig|SFTConfig/ModelConfig dataclasses), which accepts a
YAML ``--config`` natively. The experiment YAML's ``train:`` block is dumped
verbatim to that file by the method drivers, so every TRL field (beta, lr,
rpo_alpha, lora_*, deepspeed, ...) is settable per experiment without valuegen
redeclaring it. For the multi-GPU DeepSpeed ZeRO-3 path (32B) the drivers wrap
the same invocation in ``torchrun --nproc_per_node=N`` and set ``deepspeed:``
in the train block (see ``configs/deepspeed/zero3_offload.json``).

Deliberately dropped from the port: the XSTest evaluation callback (old
``CustomArguments.xstest_*``) — every canonical run disabled it, and it pulled
a second model into the training job.

**Merged-dir naming constraint (load-bearing)**: conflictscope's vLLM clients
select the chat template by *substring of the model path*
(``model_wrappers.VLLMClient.format_messages`` checks tulu/olmo-3/olmo/llama/
qwen/gemma/mistral, in that order — tulu before llama because Tulu HF names
contain "Llama", the hyphenated ``olmo-3`` before ``olmo`` because Olmo-3
post-trains speak ChatML while our own ``olmo3_*`` checkpoints trained under
the OLMo-2 template must keep the plain branch). A merged checkpoint whose
directory name lacks its family keyword
gets the wrong template silently at eval time. :func:`infer_chat_family`
implements the same match; the ``merge`` subcommand refuses a ``--merged-path``
with no recognizable keyword unless ``--chat-family`` confirms one explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

from valuegen.ground_truth.chat_formats import (
    CHAT_FORMATS,
    ChatFormat,
    get_chat_format,
    write_sidecar,
)

# Substring→template dispatch order copied from conflictscope
# model_wrappers.format_messages; first hit wins. "olmo-3" (hyphen) is the
# ChatML family of the Olmo-3 post-trains and of checkpoints derived from
# them (e.g. the collaborator-trained neutral SFT mirrored as
# neutral_sft_v3_olmo-3_7b_merged). Every hyphenated Olmo-3 HF id lands here,
# the raw base allenai/Olmo-3-1025-7B included -- the fork already serves
# those names as ChatML, so training agrees with eval instead of silently
# pinning the OLMo-2 template. The un-hyphenated "olmo3" spelling of our own
# tulu-template checkpoints deliberately falls through to "olmo".
#
# Explicit alternative (no heuristics): ``--chat_format <name>`` selects a
# :class:`chat_formats.ChatFormat` (``olmo3_chatml``, ``qwen_chatml``) for
# training, export, serving metadata and evaluation; see that module.
CHAT_FAMILIES = ("tulu", "olmo-3", "olmo", "llama", "qwen", "gemma", "mistral")

# ── Training-time chat templates, byte-matched to eval-time formatting ──────
#
# JSONL pair datasets go through the tokenizer's chat template (see
# ``_load_pair_dataset``), and eval goes through conflictscope's hand-rolled
# ``model_wrappers.format_messages_for_{family}`` — which is *near* but not
# byte-identical to the HF tokenizers' own templates. These Jinja ports render
# the fork's exact bytes so the preference gradient lands on the token
# contexts the eval actually serves; ``tests/ground_truth/test_chat_templates.py`` asserts
# parity against the fork. One deliberate deviation: the fork appends the
# next-turn tag unconditionally (its call sites always want a generation
# prompt), the ports gate it on ``add_generation_prompt`` so completions don't
# train with a trailing user-turn opener.

# tulu/olmo/gemma/mistral fold a leading system message into the first user
# turn (the fork never emits a system turn for these families).
_FOLD_SYSTEM = (
    "{%- set ns = namespace(sys='') -%}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{%- set ns.sys = messages[0]['content'] -%}"
    "{%- endif -%}"
)

_TULU_BODY = (
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'user' -%}"
    "{{- '<|user|>\n' -}}"
    "{%- if ns.sys -%}{{- ns.sys + '\n\n' -}}{%- set ns.sys = '' -%}{%- endif -%}"
    "{{- m['content'] + '\n' -}}"
    "{%- elif m['role'] == 'assistant' -%}"
    "{{- '<|assistant|>\n' + m['content'] + '<|endoftext|>\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|assistant|>\n' -}}{%- endif -%}"
)

CHAT_TEMPLATES = {
    "tulu": _FOLD_SYSTEM + _TULU_BODY,
    "olmo": _FOLD_SYSTEM + "{{- '<|endoftext|>' -}}" + _TULU_BODY,
    # Olmo-3 ChatML: for tool-free conversations, byte-identical to
    # chat_formats "olmo3_chatml" (allenai/Olmo-3-7B-Instruct
    # chat_template.jinja minus the tool-calling system turn); prompt bytes are
    # exactly the fork's qwen formatter (which the fork's 'olmo-3' branch
    # delegates to). The ONE difference from "qwen" is the terminator:
    # mid-conversation assistant turns end with <|im_end|> but the FINAL
    # assistant turn with eos_token = <|endoftext|>, so a completion here ends
    # in <|endoftext|> too.
    "olmo-3": (
        "{%- for m in messages -%}"
        "{%- if not loop.first -%}{{- '\n' -}}{%- endif -%}"
        "{{- '<|im_start|>' + m['role'] + '\n' + m['content'] -}}"
        "{%- if loop.last and m['role'] == 'assistant' -%}"
        "{{- '<|endoftext|>' -}}"
        "{%- else -%}"
        "{{- '<|im_end|>' -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '\n<|im_start|>assistant\n' -}}{%- endif -%}"
    ),
    # llama/qwen keep system as its own turn, like the fork. The fork joins
    # turns with a separator but appends the generation prompt without one;
    # ported as-is, so for llama the trl prompt/completion boundary slides to
    # the last <|eot_id|> (the concatenation still matches eval bytes).
    "llama": (
        "{{- '<|begin_of_text|>' -}}"
        "{%- for m in messages -%}"
        "{%- if not loop.first -%}{{- '\n\n' -}}{%- endif -%}"
        "{{- '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n'"
        " + m['content'] + '<|eot_id|>' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}"
        "{%- endif -%}"
    ),
    "qwen": (
        "{%- for m in messages -%}"
        "{%- if not loop.first -%}{{- '\n' -}}{%- endif -%}"
        "{{- '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '\n<|im_start|>assistant\n' -}}{%- endif -%}"
    ),
    "gemma": _FOLD_SYSTEM + (
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'user' -%}"
        "{{- '<start_of_turn>user\n' -}}"
        "{%- if ns.sys -%}{{- ns.sys + '\n\n' -}}{%- set ns.sys = '' -%}{%- endif -%}"
        "{{- m['content'] + '<end_of_turn>\n' -}}"
        "{%- elif m['role'] == 'assistant' -%}"
        "{{- '<start_of_turn>model\n' + m['content'] + '<end_of_turn>\n' -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '<start_of_turn>model\n' -}}{%- endif -%}"
    ),
    # mistral has no generation prompt: the model continues after [/INST].
    # The fork puts a space between turns (not after the last).
    "mistral": _FOLD_SYSTEM + (
        "{{- '<s>' -}}"
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'user' -%}"
        "{{- ' [INST] ' -}}"
        "{%- if ns.sys -%}{{- ns.sys + '\n\n' -}}{%- set ns.sys = '' -%}{%- endif -%}"
        "{{- m['content'] + ' [/INST]' -}}"
        "{%- if not loop.last -%}{{- ' ' -}}{%- endif -%}"
        "{%- elif m['role'] == 'assistant' -%}"
        "{{- m['content'] + '</s>' -}}"
        "{%- if not loop.last -%}{{- ' ' -}}{%- endif -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
    ),
}

# ── Scaffold templates for *pretrained* (non-chat) models ───────────────────
#
# A base model has no trained chat tags: Qwen3-8B-Base's <|im_start|>/<|im_end|>
# rows are untrained and OLMo-3 base's are reserved-but-unused, so a ChatML
# render drifts (2026-09-08 probe, data/_scratch/template_probe: 54-83% of
# greedy answers hit the cap, role echo, loops). These are plain-text
# scaffolds that pretraining did see. They are NOT chat families — nothing in
# ``infer_chat_family`` dispatches to them; a predictor opts in explicitly via
# ``--predictor-param template=<name>``. Same Jinja contract as CHAT_TEMPLATES
# (``messages`` + ``add_generation_prompt``), so the grad worker pins them on
# the tokenizer exactly like a family.
#
# ``urial0``: URIAL (Lin et al., ICLR 2024) with *zero* in-context exemplars —
# the verbatim ``# Instruction`` preamble of Re-Align/URIAL
# ``urial_prompts/inst_1k_v4.help.txt``, the system prompt as a plain
# paragraph, and fenced ``# Query:`` / ``# Answer:`` blocks. The generation
# prompt ends inside the open answer fence; a completed answer closes it.
URIAL_INSTRUCTION = (
    "# Instruction\n\n"
    "Below is a list of conversations between a human and an AI assistant (you). \n"
    "As an AI assistant, you will engage in conversations with users, responding "
    "to their queries which are presented under the heading \"# Query:\". \n"
    "Your responses should be entered under the heading \"# Answer:\". \n"
    "You excel in a wide range of tasks including, but not limited to, providing "
    "general information, conducting reasoning, engaging in role-play, creative "
    "writing, planning, and solving mathematical and coding problems. \n"
    "Your responses should be well-structured, comprehensive, and aim to "
    "thoroughly address the user's query or problem at hand.\n"
)

SCAFFOLD_TEMPLATES = {
    "urial0": (
        "{{- urial_instruction -}}"
        "{%- if messages[0]['role'] != 'system' -%}{{- '\n' -}}{%- endif -%}"
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'system' -%}"
        "{{- m['content'] + '\n\n' -}}"
        "{%- elif m['role'] == 'user' -%}"
        "{{- '# Query:\n```\n' + m['content'] + '\n```\n\n# Answer:\n```\n' -}}"
        "{%- elif m['role'] == 'assistant' -%}"
        "{{- m['content'] + '\n```' -}}"
        "{%- if not loop.last -%}{{- '\n\n' -}}{%- endif -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
    ).replace("urial_instruction", repr(URIAL_INSTRUCTION)),
}

# Generation-side half of a scaffold: where an answer ends. The template
# leaves the model inside an open ``# Answer:`` fence, so it closes the fence
# and, left alone, keeps writing the next turn -- stop on the next header
# (never on the fence itself: that truncates any answer with a code block),
# then strip the closing fence from what came back.
SCAFFOLD_STOPS = {
    "urial0": {"stop": ["\n# Query:", "\n# Instruction"], "strip_suffix": "```"},
}


def scaffold_spec(name: str) -> dict:
    """The JSON-serialisable spec conflictscope's ``--assistant-scaffold``
    consumes: chat template plus stops. Raises on an unknown scaffold.

    Besides the prompt scaffolds above, ``name`` may be an explicit chat
    format (``chat_formats.CHAT_FORMATS``): the model is then formatted by
    that pinned template instead of the fork's model-name heuristic, and
    stops on its own generation-config eos as usual (no extra stops). That
    is how a base model whose HF name would hit the wrong heuristic branch
    (e.g. ``…-qwen3-8b`` -> the ``<think>`` scaffold) is evaluated."""
    if name in SCAFFOLD_TEMPLATES:
        return {"chat_template": SCAFFOLD_TEMPLATES[name], **SCAFFOLD_STOPS.get(name, {})}
    if name in CHAT_FORMATS:
        return {"chat_template": CHAT_FORMATS[name].template}
    raise KeyError(
        f"Unknown scaffold {name!r}; known: {sorted(SCAFFOLD_TEMPLATES)} "
        f"or a chat format {sorted(CHAT_FORMATS)}"
    )


def render_chat_template(template: str, messages: list[dict],
                         add_generation_prompt: bool) -> str:
    """Render a CHAT_TEMPLATES / SCAFFOLD_TEMPLATES string the way transformers
    does (same sandboxed Jinja settings), without a tokenizer."""
    import copy

    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template).render(
        messages=copy.deepcopy(messages), add_generation_prompt=add_generation_prompt
    )


# Inlined from trl.trainer.utils.SIMPLE_CHAT_TEMPLATE (trl==0.15.2): the
# symbol was dropped from trl's public API in a later release, and this repo
# only ever needs the constant, not the rest of trl.trainer.utils.
SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


def infer_chat_family(name_or_path: str | Path) -> str | None:
    """Chat-template family a model path selects at eval time, or None."""
    lowered = str(name_or_path).lower()
    for family in CHAT_FAMILIES:
        if family in lowered:
            return family
    return None


def latest_checkpoint(output_dir: str | Path) -> Path:
    """Highest-step ``checkpoint-*`` dir under a trainer output dir."""
    output_dir = Path(output_dir)
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        # save_strategy "no" (the disk-discipline recipe) writes no
        # intermediate checkpoints; the trainer's final save_model still lands
        # a complete adapter at the output root.
        if (output_dir / "adapter_config.json").is_file():
            return output_dir
        raise FileNotFoundError(
            f"No checkpoint-* under {output_dir}; training failed or never ran."
        )
    return checkpoints[-1]


_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def prune_trainer_dir(trainer_dir: str | Path, *, checkpoints: bool = False) -> bool:
    """Drop a trainer output dir's weight shards once a servable copy (or its
    scores) exists elsewhere; returns whether anything was removed.

    Removes the root ``*.safetensors`` / ``pytorch_model*.bin`` shards and
    renames ``model.safetensors.index.json`` -> ``.pruned`` so nothing tries
    to load the now-shardless save. ``checkpoints=True`` also removes the
    ``checkpoint-*/`` and ``global_step*/`` dirs (weights + optimizer state).
    The small files stay: ``trainer_state.json``, configs, logs, and the LoRA
    adapter — lifted to the root first when only a ``checkpoint-*`` holds it.
    """
    root = Path(trainer_dir)
    if not root.is_dir():
        return False
    removed = False
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        for shard in root.glob(pattern):
            if shard.name not in _ADAPTER_FILES:
                shard.unlink()
                removed = True
    index = root / "model.safetensors.index.json"
    if index.is_file():
        index.rename(root / "model.safetensors.index.json.pruned")
    if checkpoints:
        if not (root / "adapter_model.safetensors").is_file():
            try:
                latest = latest_checkpoint(root)
            except FileNotFoundError:
                latest = root
            if latest != root and (latest / "adapter_model.safetensors").is_file():
                for name in _ADAPTER_FILES:
                    if (latest / name).is_file():
                        shutil.copy2(latest / name, root / name)
        for pattern in ("checkpoint-*", "global_step*"):
            for sub in root.glob(pattern):
                if sub.is_dir():
                    shutil.rmtree(sub)
                    removed = True
    return removed


# Turn-ender each training template closes an assistant message with. Loss is
# on the completion, which the pinned template renders ending in this token and
# nothing else (trl's conversational prompt/completion path appends no eos), so
# the trained model stops by emitting *this* -- not the base tokenizer's eos.
_TEMPLATE_TURN_ENDER = {
    "tulu": "<|endoftext|>",
    "olmo": "<|endoftext|>",
    "olmo-3": "<|endoftext|>",
    "llama": "<|eot_id|>",
    "qwen": "<|im_end|>",
    "gemma": "<end_of_turn>",
    "mistral": "</s>",
}


# Extra stop tokens a family's artifact should also declare in
# generation_config.eos_token_id, after the turn-ender. Olmo-3 ChatML models
# see <|im_end|> close every non-final assistant turn in training, so a
# sampled <|im_end|> is a plausible (if off-convention) turn end; without it
# in the stop set vLLM would run past the turn (allenai/Olmo-3-7B-Instruct-SFT
# itself declares [100265, 100257]).
_TEMPLATE_EXTRA_STOPS = {
    "olmo-3": ("<|im_end|>",),
}


def _extra_stop_ids(tokenizer, family: str) -> list[int]:
    vocab = tokenizer.get_vocab()
    return [
        tokenizer.convert_tokens_to_ids(t)
        for t in _TEMPLATE_EXTRA_STOPS.get(family, ())
        if t in vocab
    ]


def turn_ender_id(tokenizer, family: str) -> int:
    """Vocab id of ``family``'s turn-ender; raises if the tokenizer lacks it."""
    ender = _TEMPLATE_TURN_ENDER[family]
    # Membership test, not an unk-id comparison: OLMo-3 aliases unk to eos
    # (<|endoftext|> = 100257 is both), which made its own turn-ender look
    # unknown and hard-failed every olmo export (2026-08-30, chat_sft_train_a).
    if ender not in tokenizer.get_vocab():
        raise ValueError(
            f"{family} turn-ender {ender!r} not in vocab of "
            f"{getattr(tokenizer, 'name_or_path', tokenizer)}"
        )
    return tokenizer.convert_tokens_to_ids(ender)


def _eos_list(eos) -> list[int]:
    return [] if eos is None else [eos] if isinstance(eos, int) else list(eos)


def adopt_chat_format_eos(tokenizer, generation_config, fmt: ChatFormat) -> int:
    """Explicit-format counterpart of :func:`adopt_turn_ender_eos`.

    ``tokenizer.eos_token`` becomes the format's turn-ender, ``pad_token`` the
    format's pad (when it names one), and ``generation_config.eos_token_id``
    becomes ``[turn_ender, *extra_stops, *previous]`` (deduplicated, order
    kept). Idempotent. For ``olmo3_chatml`` on an OLMo-3 base this is a
    no-op by construction (eos <|endoftext|>, pad <|pad|>, eos ids [100257]);
    for ``qwen_chatml`` on a Qwen3 base it yields the published instruct
    metadata (eos <|im_end|>, pad <|endoftext|>, eos ids [151645, 151643]).
    """
    vocab = tokenizer.get_vocab()
    for tok in fmt.stop_tokens():
        if tok not in vocab:
            raise ValueError(
                f"{fmt.name} stop token {tok!r} not in vocab of "
                f"{getattr(tokenizer, 'name_or_path', tokenizer)}"
            )
    if fmt.pad_token is not None:
        if fmt.pad_token not in vocab:
            raise ValueError(f"{fmt.name} pad token {fmt.pad_token!r} not in vocab")
        tokenizer.pad_token = fmt.pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = fmt.turn_ender
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in fmt.stop_tokens()]
    if generation_config is not None:
        eos = _eos_list(generation_config.eos_token_id)
        generation_config.eos_token_id = stop_ids + [e for e in eos if e not in stop_ids]
        generation_config.pad_token_id = tokenizer.pad_token_id
    return stop_ids[0]


def adopt_turn_ender_eos(
    tokenizer, generation_config, family: str | None, chat_format: ChatFormat | None = None
) -> int | None:
    """Give a base checkpoint its family's *instruct* eos convention, in memory.

    With ``chat_format`` given, the explicit :func:`adopt_chat_format_eos`
    applies instead and ``family`` is ignored.

    Pretraining checkpoints declare the document terminator as eos (Qwen
    ``<|endoftext|>``, Llama ``<|end_of_text|>``); the instruct releases
    declare the chat turn-ender (Qwen3-8B: tokenizer eos ``<|im_end|>``,
    generation eos ``[151645, 151643]``; Llama-3.1-8B-Instruct:
    ``<|eot_id|>``, ``[128001, 128008, 128009]``). Training under the pinned
    templates teaches exactly the instruct behaviour -- the completion ends in
    the turn-ender and nothing else (trl's conversational path appends no
    eos) -- so the artifact must carry the instruct metadata too, or vLLM's
    stop set (tokenizer eos + ``generation_config.eos_token_id``) never fires
    and the model samples past the turn into repetition/gibberish
    (2026-08-29: 35% of neutral_sft_v3_qwen3_8b turns on ConflictScope).

    Sets ``tokenizer.eos_token`` to the turn-ender (pad falls back to the
    *base* eos first, keeping pad != eos like the instruct releases) and
    prepends its id to ``generation_config.eos_token_id``. Both trainers call
    it before building the trainer, so every ``save_model`` output is right
    by construction; :func:`align_eos_with_template` applies the same thing
    to exports and verifies it. No-op (returns None) without a chat family;
    tulu/olmo are unchanged by construction (their template ends in eos).
    """
    if chat_format is not None:
        return adopt_chat_format_eos(tokenizer, generation_config, chat_format)
    if family is None:
        return None
    tid = turn_ender_id(tokenizer, family)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = _TEMPLATE_TURN_ENDER[family]
    if generation_config is not None:
        eos = _eos_list(generation_config.eos_token_id)
        head = [tid] + [e for e in _extra_stop_ids(tokenizer, family) if e != tid]
        generation_config.eos_token_id = head + [e for e in eos if e not in head]
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = tokenizer.pad_token_id
    return tid


def align_chat_format(out_dir: str | Path, fmt: ChatFormat) -> dict:
    """Explicit-format counterpart of :func:`align_eos_with_template`.

    Pins ``fmt``'s template as the served ``chat_template`` (transformers 5
    writes it to ``chat_template.jinja``; any stray ``chat_template*.jinja``
    that would take precedence is removed first), applies
    :func:`adopt_chat_format_eos` to tokenizer + generation config, writes the
    ``valuegen_chat_format.json`` sidecar, then reloads the dir from disk and
    verifies every piece took. Returns the verified summary. Idempotent.
    """
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    out_dir = Path(out_dir)
    for stray in out_dir.glob("chat_template*.jinja"):
        stray.unlink()
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
    if (out_dir / "generation_config.json").is_file():
        gen = GenerationConfig.from_pretrained(str(out_dir))
    else:
        gen = GenerationConfig.from_model_config(AutoConfig.from_pretrained(str(out_dir)))
    tid = adopt_chat_format_eos(tokenizer, gen, fmt)
    tokenizer.chat_template = fmt.template
    gen.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    write_sidecar(out_dir, fmt, tokenizer)

    check = AutoTokenizer.from_pretrained(str(out_dir))
    gen_check = GenerationConfig.from_pretrained(str(out_dir))
    stop_ids = [check.convert_tokens_to_ids(t) for t in fmt.stop_tokens()]
    problems = []
    if check.chat_template != fmt.template:
        problems.append("served chat_template is not the pinned training template")
    if check.eos_token != fmt.turn_ender:
        problems.append(f"tokenizer eos_token is {check.eos_token!r}, not {fmt.turn_ender!r}")
    if fmt.pad_token is not None and check.pad_token != fmt.pad_token:
        problems.append(f"tokenizer pad_token is {check.pad_token!r}, not {fmt.pad_token!r}")
    gen_eos = _eos_list(gen_check.eos_token_id)
    if gen_eos[: len(stop_ids)] != stop_ids:
        problems.append(f"generation_config.eos_token_id {gen_eos} does not start with {stop_ids}")
    if gen_check.pad_token_id != check.pad_token_id:
        problems.append(f"generation pad {gen_check.pad_token_id} != tokenizer pad {check.pad_token_id}")
    if problems:
        raise RuntimeError(
            f"{out_dir}: chat-format alignment did not take after reload -- " + "; ".join(problems)
        )
    summary = {
        "chat_format": fmt.name,
        "template_sha256": fmt.template_sha256,
        "turn_ender": fmt.turn_ender,
        "turn_ender_id": tid,
        "stop_token_ids": stop_ids,
        "generation_eos_token_id": gen_eos,
        "tokenizer_eos": check.eos_token,
        "tokenizer_pad": check.pad_token,
        "pad_token_id": check.pad_token_id,
    }
    print(f"chat format aligned: {json.dumps(summary)}")
    return summary


def align_eos_with_template(
    out_dir: str | Path,
    name_or_path: str | None = None,
    chat_family: str | None = None,
    chat_format: ChatFormat | None = None,
) -> None:
    """Make a servable dir stop -- and prompt -- where its training template does.

    With ``chat_format`` given, :func:`align_chat_format` applies instead
    (explicit identity; no path inference).

    Both export paths save the *base* tokenizer; this rewrites the dir's
    metadata to the instruct convention (:func:`adopt_turn_ender_eos`) and
    pins the training template as the served chat template, so a server that
    formats via the tokenizer (vLLM chat completions) renders the bytes the
    model was trained on. Then reloads the dir and checks all of it took:
    transformers 5 gives a ``chat_template.jinja`` file priority over the
    ``tokenizer_config.json`` entry, which is how a hand-patched config can
    still serve the base template (neutral_sft_v3_qwen3_8b, 2026-09-02).
    No-op when neither ``chat_family`` nor the dir name carries a family.
    """
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    if chat_format is not None:
        align_chat_format(out_dir, chat_format)
        return
    out_dir = Path(out_dir)
    family = chat_family or infer_chat_family(name_or_path or out_dir.name)
    if family is None:
        return
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
    if (out_dir / "generation_config.json").is_file():
        gen = GenerationConfig.from_pretrained(str(out_dir))
    else:
        gen = GenerationConfig.from_model_config(AutoConfig.from_pretrained(str(out_dir)))
    tid = adopt_turn_ender_eos(tokenizer, gen, family)
    tokenizer.chat_template = CHAT_TEMPLATES[family]
    gen.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    ender = _TEMPLATE_TURN_ENDER[family]
    check = AutoTokenizer.from_pretrained(str(out_dir))
    gen_check = GenerationConfig.from_pretrained(str(out_dir))
    problems = []
    if check.chat_template != CHAT_TEMPLATES[family]:
        problems.append("served chat_template is not the pinned training template")
    if check.eos_token != ender:
        problems.append(f"tokenizer eos_token is {check.eos_token!r}, not {ender!r}")
    if tid not in _eos_list(gen_check.eos_token_id):
        problems.append(f"generation_config.eos_token_id {gen_check.eos_token_id} lacks {tid}")
    for extra in _extra_stop_ids(check, family):
        if extra not in _eos_list(gen_check.eos_token_id):
            problems.append(f"generation_config.eos_token_id {gen_check.eos_token_id} lacks extra stop {extra}")
    if problems:
        raise RuntimeError(
            f"{out_dir}: eos/template alignment did not take after reload -- "
            + "; ".join(problems)
            + ". Look for stray chat_template*.jinja files in the dir."
        )
    print(
        f"eos aligned to {family} template: {ender}={tid}; "
        f"tokenizer eos={check.eos_token!r} pad={check.pad_token!r}; "
        f"generation eos={gen_check.eos_token_id}"
    )


def merge_lora(
    base_model: str,
    checkpoint_dir: str | Path,
    merged_path: str | Path,
    chat_family: str | None = None,
) -> None:
    """Merge a LoRA adapter into its base model and save a vLLM-servable dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_path))
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(merged_path))
    align_eos_with_template(merged_path, chat_family=chat_family)
    del base, model, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Merge complete: {merged_path}")


def export_full_ft(
    base_model: str,
    trained_dir: str | Path,
    out_dir: str | Path,
    chat_family: str | None = None,
    chat_format: ChatFormat | None = None,
    base_revision: str | None = None,
) -> None:
    """Re-save a full-FT trainer output as a vLLM-servable dir.

    Mirrors :func:`merge_lora`'s tail — the three things the merge step does
    besides merging, which a full-FT run still needs: bf16 weights (a fp32
    ``torch_dtype`` in config.json would have vLLM load the checkpoint at
    twice the size and change the arithmetic under the eval), the *base*
    tokenizer re-aligned to the training template and instruct eos
    (:func:`align_eos_with_template`), and a destination whose name carries
    the chat-family keyword (checked by the ``export`` CLI wrapper before
    anything is loaded).

    CPU-safe by construction: the trainer output is loaded on the CPU (no
    ``device_map``), cast to bf16 and re-saved as sharded safetensors; no GPU
    ever holds a full copy. With ``chat_format`` the explicit alignment
    (:func:`align_chat_format`) runs and the artifact carries the
    ``valuegen_chat_format.json`` sidecar. The export is written to a
    ``<out_dir>.partial`` staging dir and renamed on success, so a killed
    export never leaves a half-written dir that ``export`` would later skip.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    staging = out_dir.with_name(out_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    model = AutoModelForCausalLM.from_pretrained(
        str(trained_dir), dtype=torch.bfloat16, device_map=None
    )
    model.config.use_cache = True  # servable artifact; training-time False is not a model property
    model.save_pretrained(str(staging), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, revision=base_revision).save_pretrained(str(staging))
    # The trainer's generation_config (already adopted to the format) travels
    # with the model; the base tokenizer overwrote nothing generation-related.
    align_eos_with_template(staging, chat_family=chat_family, chat_format=chat_format)
    for extra in ("run_metadata.json", "step_geometry.json", "holdout_diagnostic.json"):
        src = Path(trained_dir) / extra
        if src.is_file():
            shutil.copy2(src, staging / extra)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    staging.rename(out_dir)
    print(f"Export complete: {out_dir}")


def verify_export(out_dir: str | Path, chat_format: ChatFormat | None = None) -> dict:
    """Independent reload check of an exported dir (fresh process, CPU, meta).

    Asserts: sharded safetensors index + config load, weights are bf16, the
    architecture matches config, the saved chat template is byte-identical to
    the pinned one and the stop ids follow the format's convention. Cheap: the
    model is instantiated on the meta device, only the safetensors headers
    are read for dtype/shape checks.
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    out_dir = Path(out_dir)
    config = AutoConfig.from_pretrained(str(out_dir))
    index = out_dir / "model.safetensors.index.json"
    shards = (
        sorted({v for v in json.loads(index.read_text())["weight_map"].values()})
        if index.is_file()
        else ["model.safetensors"]
    )
    dtypes, n_params = set(), 0
    for shard in shards:
        with safe_open(str(out_dir / shard), framework="pt") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                dtypes.add(sl.get_dtype())
                n = 1
                for d in sl.get_shape():
                    n *= d
                n_params += n
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    meta_params = sum(p.numel() for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
    gen = GenerationConfig.from_pretrained(str(out_dir))
    result = {
        "path": str(out_dir),
        "architectures": config.architectures,
        "config_dtype": str(getattr(config, "dtype", None)),
        "weight_dtypes": sorted(dtypes),
        "n_shards": len(shards),
        "n_params_saved": n_params,
        "n_params_config": meta_params,
        "tokenizer_eos": tokenizer.eos_token,
        "tokenizer_pad": tokenizer.pad_token,
        "generation_eos_token_id": _eos_list(gen.eos_token_id),
        "generation_pad_token_id": gen.pad_token_id,
        "chat_template_sha256": hashlib_sha256(tokenizer.chat_template or ""),
        "sidecar": json.loads((out_dir / "valuegen_chat_format.json").read_text())
        if (out_dir / "valuegen_chat_format.json").is_file()
        else None,
    }
    problems = []
    if dtypes != {"BF16"}:
        problems.append(f"weights are {sorted(dtypes)}, expected only BF16")
    if n_params != meta_params:
        problems.append(f"saved params {n_params} != config params {meta_params}")
    if chat_format is not None:
        if tokenizer.chat_template != chat_format.template:
            problems.append("chat_template differs from the pinned training template")
        stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in chat_format.stop_tokens()]
        if result["generation_eos_token_id"][: len(stop_ids)] != stop_ids:
            problems.append(f"generation eos {result['generation_eos_token_id']} != {stop_ids}...")
        if tokenizer.eos_token != chat_format.turn_ender:
            problems.append(f"tokenizer eos {tokenizer.eos_token!r} != {chat_format.turn_ender!r}")
        if result["sidecar"] is None or result["sidecar"]["chat_format"] != chat_format.name:
            problems.append("missing/mismatched valuegen_chat_format.json sidecar")
    result["problems"] = problems
    if problems:
        raise RuntimeError(f"{out_dir}: export verification failed -- " + "; ".join(problems))
    return result


def hashlib_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Command factories (used by the method drivers' Stage builders) ──────────


def train_command(
    algo: str,
    train_config: str | Path,
    dataset: str | Path,
    model: str,
    output_dir: str | Path,
    run_name: str,
    nproc: int = 1,
) -> str:
    """Render the training invocation for one SLURM task.

    ``train_config`` is the experiment YAML's ``train:`` block dumped verbatim
    (the drivers write it once per experiment); per-task fields ride as CLI
    overrides, which win over the config file in ``TrlParser``. ``nproc > 1``
    selects the ``torchrun`` multi-GPU path (pair it with a sharding backend
    in the train block: ``fsdp:``/``fsdp_config:``, e.g.
    ``configs/fsdp/full_shard.json``, or ``deepspeed:``). The rendezvous
    endpoint binds port 0 so two trainings landing on one node don't collide
    on torchrun's default port.
    """
    assert algo in ("dpo", "sft"), algo
    launcher = (
        f"torchrun --nproc_per_node={nproc} "
        "--rdzv-backend=c10d --rdzv-endpoint=localhost:0 -m"
        if nproc > 1
        else "python -m"
    )
    return (
        f"{launcher} valuegen.ground_truth.training {algo} \\\n"
        f"    --config {train_config} \\\n"
        f"    --dataset_name {dataset} \\\n"
        f"    --model_name_or_path {model} \\\n"
        f"    --output_dir {output_dir} \\\n"
        f"    --run_name {run_name}"
    )


def merge_command(
    base_model: str, checkpoint_root: str | Path, merged_path: str | Path
) -> str:
    return (
        f"python -m valuegen.ground_truth.training merge \\\n"
        f"    --base-model {base_model} \\\n"
        f"    --checkpoint-root {checkpoint_root} \\\n"
        f"    --merged-path {merged_path}"
    )


def export_command(
    base_model: str,
    trained_root: str | Path,
    out_path: str | Path,
    chat_format: str | None = None,
    base_revision: str | None = None,
    verify: bool = False,
    prune_trained: bool = False,
) -> str:
    cmd = (
        f"python -m valuegen.ground_truth.training export \\\n"
        f"    --base-model {base_model} \\\n"
        f"    --trained {trained_root} \\\n"
        f"    --out {out_path}"
    )
    if chat_format:
        cmd += f" \\\n    --chat-format {chat_format}"
    if base_revision:
        cmd += f" \\\n    --base-revision {base_revision}"
    if verify:
        cmd += " \\\n    --verify"
    if prune_trained:
        cmd += " \\\n    --prune-trained"
    return cmd


def _resolve_device_map(training_args) -> str | None:
    """Naive model-parallel placement, only when nothing else owns placement.

    ``"auto"`` spreads one model across the visible GPUs (accelerate's
    balanced device map) — right for a single big-model process, wrong under
    deepspeed, FSDP, or any multi-process launch (torchrun sets WORLD_SIZE),
    where the distributed wrapper shards the model itself and a pre-sharded
    device map fights it.
    """
    if training_args.deepspeed or getattr(training_args, "fsdp", None):
        return None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return None
    return "auto"


def _set_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)


def _pin_chat_template(
    tokenizer,
    model_name_or_path: str,
    dataset_name: str,
    chat_format: ChatFormat | None = None,
) -> None:
    """Make training-time formatting match eval-time formatting.

    JSONL (conversational) datasets are rendered through the tokenizer's chat
    template, so pin it to the family template conflictscope will select for
    the merged checkpoint. Legacy CSV datasets never touch the template; for
    those the SIMPLE_CHAT_TEMPLATE fallback only covers template-less
    tokenizers, as before. An explicit ``chat_format`` wins over the path
    heuristic.
    """
    if chat_format is not None:
        tokenizer.chat_template = chat_format.template
        return
    family = infer_chat_family(model_name_or_path)
    if family is not None:
        tokenizer.chat_template = CHAT_TEMPLATES[family]
    elif dataset_name.endswith((".json", ".jsonl")):
        raise ValueError(
            f"{model_name_or_path!r} matches no chat family "
            f"({'/'.join(CHAT_FAMILIES)}). A conversational dataset would be "
            "trained under the tokenizer's own template, which conflictscope "
            "cannot reproduce at eval time — use a model path carrying a "
            "family keyword."
        )
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE


def _pair_script_arguments():
    from dataclasses import dataclass
    from trl import ScriptArguments

    @dataclass
    class PairScriptArguments(ScriptArguments):
        val_fraction: float = 0.1
        # Explicit chat-format identity (chat_formats.CHAT_FORMATS); None keeps
        # the legacy path-keyword inference.
        chat_format: str | None = None
        # Rows of the held-out split scored by the final completion-token
        # diagnostic (None: all). Pilots cap it; production scores everything.
        holdout_max_rows: int | None = None
        # Resumable checkpoints kept on disk. Older ones are deleted right
        # BEFORE a new save (storage: a 32B fp32+AdamW checkpoint is ~384 GB),
        # so at most this many exist at any time; None keeps Trainer's own
        # after-save rotation (save_total_limit).
        checkpoint_keep: int | None = None
        # MoE gradient/expert-selection audit at these optimizer steps
        # (comma-separated, "first,last" allowed); empty disables.
        moe_audit_steps: str = ""
        # Dump the first training microbatch (ids, labels, decoded tokens) on
        # rank 0 to output_dir/first_batch_audit.json.
        dump_first_batch: bool = True
        # Pre-tokenize with the boundary-preserving length cap
        # (prepare_prompt_completion_dataset) instead of TRL's blind
        # [:max_length] slice. Requires a ChatML-shaped chat format.
        pretokenize: bool = False
        # Completion tokens (content + terminator) an over-long row keeps at
        # minimum before its prompt content is cut (pretokenize only).
        min_completion_tokens: int = 256
        # Pilot resume test: stop training cleanly after this optimizer step
        # (after its checkpoint save), so a second invocation with the SAME
        # max_steps resumes with an identical schedule. None: never stop early.
        stop_after_step: int | None = None
        # ── Launch gates (DPO): the run aborts BEFORE the first optimizer
        # step when the trainer's resolved geometry differs from what the
        # experiment declares, so a silently dropped row or a changed world
        # size can never produce a checkpoint that looks like the others.
        # Rows in the dataset file before any trainer preprocessing.
        expect_raw_rows: int | None = None
        # Rows in the trainer's train set after its own preprocessing (TRL
        # drops rows whose prompt alone fills max_length). When the recipe
        # accepts those drops, launchers pass the per-run post-drop count.
        expect_train_rows: int | None = None
        # Mix-build gate flag consumed by `valuegen mv data` (multivalue
        # token audit): accept TRL's prompt-fills-max_length drops instead of
        # failing the gate. Declared here only so a recipe carrying it passes
        # TrlParser's unknown-key check; the trainer itself ignores it.
        accept_trl_prompt_drops: bool = False
        # Resolved optimizer steps / warmup steps / torchrun world size.
        expect_max_steps: int | None = None
        expect_warmup_steps: int | None = None
        expect_world_size: int | None = None
        # Fingerprint policy + reference parameters at train begin/end and
        # assert the policy moved while the reference did not
        # (dpo_integrity.json). Also records non-finite loss/grad-norm logs.
        check_reference_frozen: bool = False
        # Hub revision of the base checkpoint, recorded in run_metadata.json
        # (a local snapshot dir carries no revision of its own).
        base_revision: str | None = None

    return PairScriptArguments


class _StopAfterStepCallback:
    def __new__(cls, step: int):
        from transformers import TrainerCallback

        class StopAfterStepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= step:
                    control.should_training_stop = True
                return control

        return StopAfterStepCallback()


def tokenize_prompt_completion(tokenizer, prompt, completion, chat_template=None) -> dict:
    """TRL 1.12 ``SFTTrainer`` prompt/completion tokenization, verbatim.

    Prompt rendered with ``add_generation_prompt=True``, prompt+completion
    rendered jointly, completion mask = every token after the separately
    tokenized prompt (so a BPE merge across the boundary -- a completion that
    starts with whitespace -- leaves the merged token unsupervised, exactly as
    TRL does), labels -100 elsewhere. Returns ids, labels, prompt length and
    whether the joint ids start with the prompt ids (TRL's mismatch warning).
    """
    kw = {"chat_template": chat_template} if chat_template is not None else {}
    prompt_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True, return_dict=True, **kw
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        list(prompt) + list(completion), tokenize=True, return_dict=True, **kw
    )["input_ids"]
    n_p = len(prompt_ids)
    labels = [-100] * min(n_p, len(full_ids)) + list(full_ids[n_p:])
    return {
        "input_ids": list(full_ids),
        "labels": labels,
        "prompt_len": n_p,
        "prompt_mismatch": full_ids[:n_p] != prompt_ids,
    }


def truncate_keep_boundaries(
    input_ids: list[int],
    labels: list[int],
    max_length: int,
    header_len: int,
    boundary_len: int,
    min_completion: int = 256,
) -> dict:
    """Length-cap a prompt/completion row without destroying its structure.

    TRL's ``keep_start`` slice ``[:max_length]`` drops the assistant
    terminator of every over-long row and drops rows whose prompt alone fills
    the window (changing membership). This instead keeps the user header,
    the ``<|im_end|>\\n<|im_start|>assistant\\n`` boundary and the supervised
    terminator intact and trims content: the completion keeps at least
    ``min(min_completion, completion_len)`` tokens (content + terminator), the
    prompt gets the remaining budget (its user content is cut from the END),
    and whatever completion content still does not fit is cut before the
    terminator. Rows already within ``max_length`` are returned unchanged.
    ``header_len``/``boundary_len`` are the prompt's leading header tokens
    (``<|im_start|>user\\n``) and trailing boundary tokens.
    """
    n = len(input_ids)
    n_p = sum(1 for l in labels if l == -100) if -100 in labels else 0
    # prompt tokens are the leading -100 run
    n_p = 0
    while n_p < n and labels[n_p] == -100:
        n_p += 1
    n_c = n - n_p
    if n <= max_length:
        return {"input_ids": input_ids, "labels": labels, "prompt_dropped": 0, "completion_dropped": 0}
    if n_c < 1:
        raise ValueError("row has no completion tokens")
    c_keep = min(n_c, max(min_completion, max_length - n_p))
    p_budget = max_length - c_keep
    if n_p > p_budget:
        if p_budget < header_len + boundary_len + 1:
            raise ValueError(f"max_length {max_length} too small for header+boundary")
        content_keep = p_budget - header_len - boundary_len
        prompt_ids = input_ids[:header_len] + input_ids[header_len:header_len + content_keep] + input_ids[n_p - boundary_len:n_p]
        prompt_dropped = n_p - len(prompt_ids)
    else:
        prompt_ids = input_ids[:n_p]
        prompt_dropped = 0
    comp_ids = input_ids[n_p:]
    comp_keep = max_length - len(prompt_ids)
    if comp_keep < n_c:
        terminator = comp_ids[-1]
        comp_ids = comp_ids[: comp_keep - 1] + [terminator]
    completion_dropped = n_c - len(comp_ids)
    new_ids = prompt_ids + comp_ids
    new_labels = [-100] * len(prompt_ids) + comp_ids
    assert len(new_ids) == max_length, (len(new_ids), max_length)
    return {
        "input_ids": new_ids, "labels": new_labels,
        "prompt_dropped": prompt_dropped, "completion_dropped": completion_dropped,
    }


def chatml_boundary_lengths(tokenizer, chat_template=None) -> tuple[int, int]:
    """(header_len, boundary_len) of a single-user-turn prompt under the
    tokenizer's chat template: the tokens rendering ``<|im_start|>user\\n`` and
    the trailing ``<|im_end|>\\n<|im_start|>assistant\\n``. Verified by
    decoding rather than assumed."""
    kw = {"chat_template": chat_template} if chat_template is not None else {}
    probe = [{"role": "user", "content": "x y z"}]
    ids = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=True, return_dict=True, **kw)["input_ids"]
    text = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False, **kw)
    header = None
    for h in range(1, 8):
        if tokenizer.decode(ids[:h]) == "<|im_start|>user\n":
            header = h
            break
    boundary = None
    for b in range(1, 10):
        if tokenizer.decode(ids[len(ids) - b:]) == "<|im_end|>\n<|im_start|>assistant\n":
            boundary = b
            break
    if header is None or boundary is None or not text.startswith("<|im_start|>user\n"):
        raise ValueError(
            f"template is not single-turn ChatML shaped: header={header} boundary={boundary} text={text!r}"
        )
    return header, boundary


def prepare_prompt_completion_dataset(
    dataset, tokenizer, max_length: int, chat_template=None, min_completion: int = 256,
    num_proc: int | None = None, desc: str = "tokenize",
):
    """Pre-tokenize a prompt/completion dataset for SFTTrainer.

    Same ids/labels as TRL's own preparation (:func:`tokenize_prompt_completion`),
    plus :func:`truncate_keep_boundaries` instead of TRL's blind slice. The
    returned dataset carries ``input_ids``/``labels`` (SFTTrainer treats it as
    already processed: no re-tokenization, its ``[:max_length]`` slice and
    fully-masked-row filter are no-ops by construction) and a stats dict.
    """
    header_len, boundary_len = chatml_boundary_lengths(tokenizer, chat_template)
    ender_id = tokenizer.eos_token_id

    def fn(ex):
        tok = tokenize_prompt_completion(tokenizer, ex["prompt"], ex["completion"], chat_template)
        n_full = len(tok["input_ids"])
        tr = truncate_keep_boundaries(
            tok["input_ids"], tok["labels"], max_length, header_len, boundary_len, min_completion
        )
        return {
            "input_ids": tr["input_ids"], "labels": tr["labels"],
            "full_len": n_full, "prompt_len": tok["prompt_len"],
            "prompt_mismatch": tok["prompt_mismatch"],
            "prompt_dropped": tr["prompt_dropped"], "completion_dropped": tr["completion_dropped"],
            "ends_with_supervised_terminator": tr["labels"][-1] == ender_id,
            "n_supervised": sum(1 for l in tr["labels"] if l != -100),
        }

    processed = dataset.map(fn, num_proc=num_proc, desc=desc)
    stats = {
        "n_rows": len(processed),
        "max_length": max_length,
        "min_completion": min_completion,
        "header_len": header_len,
        "boundary_len": boundary_len,
        "terminator_id": ender_id,
        "n_over_max_before": int(sum(1 for x in processed["full_len"] if x > max_length)),
        "n_prompt_truncated": int(sum(1 for x in processed["prompt_dropped"] if x > 0)),
        "n_completion_truncated": int(sum(1 for x in processed["completion_dropped"] if x > 0)),
        "prompt_tokens_dropped": int(sum(processed["prompt_dropped"])),
        "completion_tokens_dropped": int(sum(processed["completion_dropped"])),
        "n_prompt_mismatch": int(sum(1 for x in processed["prompt_mismatch"] if x)),
        "n_rows_without_supervised_terminator": int(sum(1 for x in processed["ends_with_supervised_terminator"] if not x)),
        "n_rows_without_supervision": int(sum(1 for x in processed["n_supervised"] if x == 0)),
        "total_supervised_tokens": int(sum(processed["n_supervised"])),
        "max_len_after": int(max((len(x) for x in processed["input_ids"]), default=0)),
    }
    full_lens = processed["full_len"]
    over_idx = [i for i, x in enumerate(full_lens) if x > max_length]
    row_ids = processed["row_id"] if "row_id" in processed.column_names else None
    pdrop, cdrop = processed["prompt_dropped"], processed["completion_dropped"]
    stats["truncated_rows"] = [
        {"index": i, "row_id": row_ids[i] if row_ids else None, "full_len": full_lens[i],
         "prompt_dropped": pdrop[i], "completion_dropped": cdrop[i]}
        for i in over_idx
    ]
    processed = processed.select_columns(["input_ids", "labels"])
    return processed, stats


def _resolve_resume(resume, output_dir: str):
    """Normalize ``--resume_from_checkpoint`` for ``Trainer.train``.

    ``"auto"`` resumes from the newest *complete* ``checkpoint-*`` under
    ``output_dir`` (one whose ``trainer_state.json`` exists -- Trainer writes
    it last) and starts fresh when there is none; ``"true"`` defers to
    Trainer's own last-checkpoint lookup (which errors when none exists); a
    path is used as-is; ``None``/``"false"`` start fresh.
    """
    if resume is None:
        return None
    if isinstance(resume, bool):
        return resume or None
    value = str(resume)
    if value.lower() in ("", "false", "no", "0", "none"):
        return None
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() == "auto":
        return latest_complete_checkpoint(output_dir)
    if not Path(value).is_dir():
        raise FileNotFoundError(f"--resume_from_checkpoint {value!r} is not a directory")
    return value


def latest_complete_checkpoint(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir() and (p / "trainer_state.json").is_file():
            try:
                candidates.append((int(p.name.rsplit("-", 1)[-1]), p))
            except ValueError:
                continue
    if not candidates:
        return None
    return str(max(candidates)[1])


def _fsdp_activation_checkpointing(training_args) -> bool:
    cfg = getattr(training_args, "fsdp_config", None) or {}
    if isinstance(cfg, str):
        return False
    value = cfg.get("activation_checkpointing", False)
    return str(value).lower() in ("true", "1", "yes")


def _uses_fsdp(training_args) -> bool:
    fsdp = getattr(training_args, "fsdp", None)
    return bool(fsdp) and fsdp not in ("", [], "false")


def resolve_use_cache(training_args) -> bool:
    """``model.config.use_cache`` for training: False under Trainer
    gradient checkpointing OR wrapper-level FSDP activation checkpointing."""
    return not (
        bool(getattr(training_args, "gradient_checkpointing", False))
        or _fsdp_activation_checkpointing(training_args)
    )


# Decoder-block classes to shard/checkpoint per architecture. Verified by
# introspection against the installed transformers (``fsdp_wrap_class``) so a
# stale name fails at launch, not silently.
FSDP_WRAP_CLASSES = {
    "Olmo3ForCausalLM": "Olmo3DecoderLayer",
    "Qwen3MoeForCausalLM": "Qwen3MoeDecoderLayer",
    "Qwen3ForCausalLM": "Qwen3DecoderLayer",
    "Olmo2ForCausalLM": "Olmo2DecoderLayer",
    "LlamaForCausalLM": "LlamaDecoderLayer",
}


def fsdp_wrap_class(model_or_config) -> str:
    """Decoder-layer class name for FSDP wrapping, checked against the code.

    Accepts a model, a config, or a model_type/architecture string. Prefers
    the architecture's ``_no_split_modules`` (the class that actually exists
    in the pinned modeling file) and cross-checks it with
    :data:`FSDP_WRAP_CLASSES`.
    """
    import transformers

    if isinstance(model_or_config, str):
        arch = model_or_config
    elif hasattr(model_or_config, "architectures"):
        arch = (model_or_config.architectures or [None])[0]
    else:
        arch = type(model_or_config).__name__
    expected = FSDP_WRAP_CLASSES.get(arch)
    if expected is None:
        raise KeyError(f"No FSDP wrap class registered for architecture {arch!r}")
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise ImportError(f"transformers {transformers.__version__} has no {arch}")
    no_split = list(getattr(cls, "_no_split_modules", None) or [])
    if expected not in no_split:
        raise ValueError(
            f"{arch}._no_split_modules={no_split} does not list {expected!r}; "
            "update FSDP_WRAP_CLASSES against the installed transformers"
        )
    module = sys.modules[cls.__module__]
    if not hasattr(module, expected):
        raise ImportError(f"{cls.__module__} defines no class {expected}")
    return expected


def _load_pair_dataset(dataset_name: str, seed: int, val_fraction: float = 0.1):
    """Load a DPO-pair dataset, retaining the legacy split by default.

    ``.json``/``.jsonl`` keep prompt/chosen/rejected as real message lists, so
    DPOTrainer sees them as conversational and applies the chat template;
    ``.csv`` keeps the legacy stringified-list-as-raw-text behavior.
    The default is the legacy 90/10 seeded split in both trainers.
    ``val_fraction: 0`` trains on all pairs, as specified by the v3 recipe.
    """
    from datasets import load_dataset

    fmt = "json" if dataset_name.endswith((".json", ".jsonl")) else "csv"
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    dataset = load_dataset(fmt, data_files=dataset_name)
    if val_fraction == 0:
        from datasets import DatasetDict
        return DatasetDict(train=dataset["train"], test=dataset["train"].select([]))
    return dataset["train"].train_test_split(test_size=val_fraction, seed=seed)


def train_dpo(argv: list[str]) -> None:
    """Full-parameter (or LoRA) DPO with an explicit chat-format identity.

    ``--chat_format`` (PairScriptArguments) pins the tokenizer's template and
    instruct eos for BOTH the policy and the frozen reference, exactly like
    :func:`train_sft`; without it the legacy path-keyword inference applies.
    The ``expect_*`` gates abort before the first optimizer step when the
    trainer's resolved geometry (rows after TRL preprocessing, optimizer
    steps, warmup steps, world size) differs from what the caller declared.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import (
        DPOConfig,
        DPOTrainer,
        ModelConfig,
        TrlParser,
        get_peft_config,
        get_quantization_config,
    )
    parser = TrlParser((_pair_script_arguments(), DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(argv)
    _set_seeds(training_args.seed)
    fmt = get_chat_format(script_args.chat_format) if script_args.chat_format else None
    rank0 = int(os.environ.get("RANK", "0")) == 0
    t_start = time.time()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if script_args.expect_world_size is not None and world_size != script_args.expect_world_size:
        raise RuntimeError(
            f"expect_world_size={script_args.expect_world_size} but WORLD_SIZE={world_size}"
        )

    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=_resolve_device_map(training_args),
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    # Not a from_pretrained kwarg: newer model classes (Gemma3ForConditional-
    # Generation, Qwen3_5ForCausalLM on transformers 5) reject unknown kwargs
    # in __init__; OLMo/Llama merely tolerated it. False under Trainer
    # gradient checkpointing OR wrapper-level FSDP activation checkpointing.
    model.config.use_cache = resolve_use_cache(training_args)
    if _uses_fsdp(training_args) and isinstance(training_args.fsdp_config, dict):
        wanted = training_args.fsdp_config.get("transformer_layer_cls_to_wrap")
        if wanted:
            verified = fsdp_wrap_class(model.config)
            if verified not in list(wanted):
                raise ValueError(
                    f"fsdp_config.transformer_layer_cls_to_wrap={wanted} does not "
                    f"include the verified decoder class {verified!r} for "
                    f"{model.config.architectures}"
                )
    peft_config = get_peft_config(model_args)
    ref_model = (
        AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        if peft_config is None
        else None
    )
    if ref_model is not None:
        ref_model.config.use_cache = model.config.use_cache
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _pin_chat_template(
        tokenizer, model_args.model_name_or_path, script_args.dataset_name, chat_format=fmt
    )
    family = infer_chat_family(model_args.model_name_or_path) if fmt is None else None
    adopt_turn_ender_eos(tokenizer, model.generation_config, family, chat_format=fmt)
    if ref_model is not None:
        # Same served-metadata convention on the reference: it is never
        # saved, but its generation config must not disagree with the policy's.
        adopt_turn_ender_eos(tokenizer, ref_model.generation_config, family, chat_format=fmt)
    print(
        f"tokenizer eos={tokenizer.eos_token!r} pad={tokenizer.pad_token!r}; "
        f"generation eos={model.generation_config.eos_token_id}; "
        f"chat_format={fmt.name if fmt else None}; use_cache={model.config.use_cache}"
    )
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    dataset = _load_pair_dataset(
        script_args.dataset_name, training_args.seed, script_args.val_fraction
    )
    n_raw = len(dataset[script_args.dataset_train_split])
    expect_raw = script_args.expect_raw_rows
    if expect_raw is None:
        expect_raw = script_args.expect_train_rows
    if expect_raw is not None and n_raw != expect_raw:
        raise RuntimeError(f"expected {expect_raw} dataset rows but the dataset has {n_raw} rows")
    if rank0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        _write_split_manifests(training_args.output_dir, dataset)

    callbacks = [_StepGeometryCallback()]
    integrity = None
    if script_args.check_reference_frozen:
        integrity = _DpoIntegrityCallback()
        callbacks.append(integrity)
    if script_args.stop_after_step is not None:
        callbacks.append(_StopAfterStepCallback(script_args.stop_after_step))
    trainer_cls = _AuditedDPOTrainer if script_args.dump_first_batch else DPOTrainer
    trainer = trainer_cls(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    if integrity is not None:
        integrity.trainer = trainer
    # TRL has now tokenized and filtered the train set (rows whose prompt
    # alone fills max_length are dropped). This is the count that decides the
    # optimizer geometry, so it is the one the gate checks.
    n_processed = len(trainer.train_dataset)
    if script_args.expect_train_rows is not None and n_processed != script_args.expect_train_rows:
        raise RuntimeError(
            f"expect_train_rows={script_args.expect_train_rows} but TRL preprocessing left "
            f"{n_processed} rows (dropped {n_raw - n_processed}); refusing to train on a "
            "silently reduced dataset"
        )
    if rank0:
        _write_run_metadata(
            training_args.output_dir, script_args, training_args, model_args, model, tokenizer,
            fmt, dataset, trainer,
        )
        _append_json(
            os.path.join(training_args.output_dir, "run_metadata.json"),
            {"n_train_rows_after_trainer_preprocessing": n_processed,
             "base_revision": script_args.base_revision,
             "algo": "dpo"},
        )

    resume = _resolve_resume(training_args.resume_from_checkpoint, training_args.output_dir)
    print(f"resume_from_checkpoint -> {resume!r}")
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", train_result.metrics)
    observed = {
        "observed_global_step": trainer.state.global_step,
        "observed_max_steps": trainer.state.max_steps,
        "observed_warmup_steps": training_args.get_warmup_steps(trainer.state.max_steps),
        "observed_num_train_epochs": trainer.state.num_train_epochs,
    }
    problems = []
    if script_args.expect_max_steps is not None and trainer.state.max_steps != script_args.expect_max_steps:
        problems.append(f"max_steps {trainer.state.max_steps} != expected {script_args.expect_max_steps}")
    if script_args.expect_warmup_steps is not None and observed["observed_warmup_steps"] != script_args.expect_warmup_steps:
        problems.append(
            f"warmup steps {observed['observed_warmup_steps']} != expected {script_args.expect_warmup_steps}"
        )
    stopped_early = (
        script_args.stop_after_step is not None
        and trainer.state.global_step < trainer.state.max_steps
    )
    if not stopped_early and trainer.state.global_step != trainer.state.max_steps:
        problems.append(
            f"training ended at step {trainer.state.global_step}, scheduled {trainer.state.max_steps}"
        )
    if rank0:
        trainer.save_metrics("train", train_result.metrics)
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
        _append_json(os.path.join(training_args.output_dir, "run_metadata.json"), observed)
    if problems:
        raise RuntimeError("DPO geometry gate failed: " + "; ".join(problems))
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if stopped_early:
        print(
            f"stopped early after step {trainer.state.global_step}/{trainer.state.max_steps} "
            "(stop_after_step); no final save."
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return
    if _uses_fsdp(training_args) and "SHARDED_STATE_DICT" in str(
        trainer.accelerator.state.fsdp_plugin.state_dict_type
    ):
        # Trainer.save_model is a no-op for SHARDED_STATE_DICT (only the
        # resumable dcp shards exist); gather the full state dict on rank 0's
        # CPU and write the servable bf16 artifact at the output root.
        save_full_bf16(trainer, training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
    if rank0:
        if fmt is not None:
            write_sidecar(training_args.output_dir, fmt, tokenizer)
        _append_json(
            os.path.join(training_args.output_dir, "run_metadata.json"),
            {"wall_time_s": round(time.time() - t_start, 1), "final_save": "done"},
        )
    if integrity is not None and rank0:
        report = integrity.report
        if report.get("policy_changed") is False or report.get("reference_fixed") is False:
            raise RuntimeError(f"DPO integrity gate failed: {json.dumps(report, default=str)[:2000]}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(f"train_dpo complete: {training_args.output_dir}")


class _AuditedDPOTrainer:
    """Placeholder replaced at first use (DPOTrainer is imported lazily)."""

    def __new__(cls, *args, **kwargs):
        from trl import DPOTrainer

        class AuditedDPOTrainer(DPOTrainer):
            _dumped_first_batch = False

            def training_step(self, model, inputs, num_items_in_batch=None):
                if not self._dumped_first_batch and self.state.global_step == 0:
                    self._dumped_first_batch = True
                    if self.args.process_index == 0:
                        try:
                            dump_dpo_batch_audit(
                                inputs,
                                self.processing_class,
                                os.path.join(self.args.output_dir, "first_batch_audit.json"),
                            )
                        except Exception as exc:  # audit must never kill training
                            print(f"first-batch audit failed: {exc!r}")
                return super().training_step(model, inputs, num_items_in_batch)

        return AuditedDPOTrainer(*args, **kwargs)


def dump_dpo_batch_audit(inputs, tokenizer, path: str) -> None:
    """One collated DPO microbatch as the trainer sees it (rank-local).

    TRL's preference collator stacks chosen sequences in the first half of
    the batch and rejected in the second, each with a ``completion_mask``
    marking the loss positions; the audit decodes prompt/completion per row
    so the bytes the gradient lands on can be inspected.
    """
    ids = inputs["input_ids"].detach().cpu()
    attn = inputs["attention_mask"].detach().cpu()
    comp = inputs["completion_mask"].detach().cpu()
    n = ids.shape[0]
    half = n // 2
    rows = []
    for i in range(n):
        seq, am, cm = ids[i].tolist(), attn[i].tolist(), comp[i].tolist()
        prompt = [t for t, a, c in zip(seq, am, cm) if a == 1 and c == 0]
        completion = [t for t, a, c in zip(seq, am, cm) if a == 1 and c == 1]
        rows.append({
            "side": "chosen" if i < half else "rejected",
            "pair_index": i if i < half else i - half,
            "n_tokens": int(sum(am)),
            "n_padding": int(len(am) - sum(am)),
            "n_completion": len(completion),
            "completion_mask_on_padding": any(c == 1 and a == 0 for a, c in zip(am, cm)),
            "prompt_text": tokenizer.decode(prompt, skip_special_tokens=False),
            "completion_text": tokenizer.decode(completion, skip_special_tokens=False),
            "last_completion_token": tokenizer.convert_ids_to_tokens(completion[-1]) if completion else None,
            "input_ids": seq,
            "attention_mask": am,
            "completion_mask": cm,
        })
    with open(path, "w") as f:
        json.dump({"batch_shape": list(ids.shape), "rows": rows}, f, indent=1)
    print(f"first-batch audit written: {path} shape={list(ids.shape)}")


def parameter_fingerprint(model) -> dict:
    """Cross-rank fingerprint of a (possibly FSDP-sharded) model's parameters.

    Sums every local parameter shard in float64 (plus their squared sum and
    element count) and all-reduces, so the result is identical on every rank
    and independent of how the parameters are sharded. Two calls agree iff no
    parameter changed (up to float64 summation of the same values, which is
    deterministic for a fixed sharding).
    """
    import torch
    import torch.distributed as dist

    device = None
    total = torch.zeros((), dtype=torch.float64)
    sq = torch.zeros((), dtype=torch.float64)
    count = torch.zeros((), dtype=torch.float64)
    for p in model.parameters():
        data = p.detach()
        if hasattr(data, "to_local"):
            data = data.to_local()
        if data.numel() == 0:
            continue
        device = data.device
        d = data.to(torch.float64)
        total = total.to(device) + d.sum()
        sq = sq.to(device) + (d * d).sum()
        count = count.to(device) + float(d.numel())
    if dist.is_available() and dist.is_initialized():
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        total, sq, count = total.to(device), sq.to(device), count.to(device)
        for t in (total, sq, count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {"sum": float(total), "sum_sq": float(sq), "numel": int(count)}


class _DpoIntegrityCallback:
    """Fingerprint policy/reference at train begin and end; track finite losses.

    Set ``callback.trainer = trainer`` after the trainer exists (the callback
    only receives the policy through kwargs; the reference lives on the
    trainer). Writes ``dpo_integrity.json`` on rank 0 and exposes ``report``.
    """

    def __new__(cls):
        from transformers import TrainerCallback

        class DpoIntegrityCallback(TrainerCallback):
            def __init__(self):
                self.trainer = None
                self.report = {
                    "policy_begin": None, "policy_end": None,
                    "reference_begin": None, "reference_end": None,
                    "policy_changed": None, "reference_fixed": None,
                    "n_loss_logs": 0, "n_nonfinite_loss": 0, "n_nonfinite_grad_norm": 0,
                    "first_loss": None, "last_loss": None, "errors": [],
                }

            def _fingerprints(self, model, tag):
                try:
                    self.report[f"policy_{tag}"] = parameter_fingerprint(model)
                except Exception as exc:
                    self.report["errors"].append(f"policy_{tag}: {exc!r}")
                ref = getattr(self.trainer, "ref_model", None) if self.trainer is not None else None
                if ref is None:
                    self.report["errors"].append(f"reference_{tag}: trainer has no ref_model")
                    return
                try:
                    self.report[f"reference_{tag}"] = parameter_fingerprint(ref)
                except Exception as exc:
                    self.report["errors"].append(f"reference_{tag}: {exc!r}")

            def on_train_begin(self, args, state, control, **kwargs):
                self._fingerprints(kwargs.get("model"), "begin")
                self._write(args)

            def on_log(self, args, state, control, logs=None, **kwargs):
                import math

                if not logs or "loss" not in logs:
                    return
                self.report["n_loss_logs"] += 1
                loss = float(logs["loss"])
                if self.report["first_loss"] is None:
                    self.report["first_loss"] = loss
                self.report["last_loss"] = loss
                if not math.isfinite(loss):
                    self.report["n_nonfinite_loss"] += 1
                gn = logs.get("grad_norm")
                if gn is not None and not math.isfinite(float(gn)):
                    self.report["n_nonfinite_grad_norm"] += 1

            def on_train_end(self, args, state, control, **kwargs):
                self._fingerprints(kwargs.get("model"), "end")
                pb, pe = self.report["policy_begin"], self.report["policy_end"]
                rb, re_ = self.report["reference_begin"], self.report["reference_end"]
                if pb and pe:
                    self.report["policy_changed"] = (pb["sum"], pb["sum_sq"]) != (pe["sum"], pe["sum_sq"])
                if rb and re_:
                    self.report["reference_fixed"] = (rb["sum"], rb["sum_sq"]) == (re_["sum"], re_["sum_sq"])
                self.report["losses_finite"] = (
                    self.report["n_nonfinite_loss"] == 0 and self.report["n_nonfinite_grad_norm"] == 0
                )
                self._write(args)
                if args.process_index == 0:
                    print(f"dpo integrity: {json.dumps(self.report, default=str)[:3000]}")

            def _write(self, args):
                if args.process_index == 0:
                    with open(os.path.join(args.output_dir, "dpo_integrity.json"), "w") as f:
                        json.dump(self.report, f, indent=2, default=str)

        return DpoIntegrityCallback()


def train_sft(argv: list[str]) -> None:
    """SFT on the CHOSEN side of a DPO pair dataset (prompt/chosen/rejected).

    FORMAT PARITY NOTE: the dataset format decides the textual format in both
    trainers identically, so an SFT-then-DPO run always sees the same bytes in
    both stages. JSONL keeps prompt/chosen as message lists → the pinned
    family chat template is applied (matching eval); legacy CSVs carry
    *stringified* message lists that both trainers treat as raw text (no
    template). Loss is on the completion only in either case.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import (
        ModelConfig,
        SFTConfig,
        SFTTrainer,
        TrlParser,
        get_peft_config,
        get_quantization_config,
    )
    parser = TrlParser((_pair_script_arguments(), SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(argv)
    _set_seeds(training_args.seed)
    fmt = get_chat_format(script_args.chat_format) if script_args.chat_format else None
    rank0 = int(os.environ.get("RANK", "0")) == 0
    t_start = time.time()

    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=_resolve_device_map(training_args),
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    # Not a from_pretrained kwarg: newer model classes (Gemma3ForConditional-
    # Generation, Qwen3_5ForCausalLM on transformers 5) reject unknown kwargs
    # in __init__; OLMo/Llama merely tolerated it. False under Trainer
    # gradient checkpointing OR wrapper-level FSDP activation checkpointing.
    model.config.use_cache = resolve_use_cache(training_args)
    if _uses_fsdp(training_args) and isinstance(training_args.fsdp_config, dict):
        wanted = training_args.fsdp_config.get("transformer_layer_cls_to_wrap")
        if wanted:
            verified = fsdp_wrap_class(model.config)
            if verified not in list(wanted):
                raise ValueError(
                    f"fsdp_config.transformer_layer_cls_to_wrap={wanted} does not "
                    f"include the verified decoder class {verified!r} for "
                    f"{model.config.architectures}"
                )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _pin_chat_template(
        tokenizer, model_args.model_name_or_path, script_args.dataset_name, chat_format=fmt
    )
    adopt_turn_ender_eos(
        tokenizer,
        model.generation_config,
        infer_chat_family(model_args.model_name_or_path) if fmt is None else None,
        chat_format=fmt,
    )
    print(
        f"tokenizer eos={tokenizer.eos_token!r} pad={tokenizer.pad_token!r}; "
        f"generation eos={model.generation_config.eos_token_id}; "
        f"chat_format={fmt.name if fmt else None}; use_cache={model.config.use_cache}"
    )

    dataset = _load_pair_dataset(
        script_args.dataset_name, training_args.seed, script_args.val_fraction
    )

    def to_prompt_completion(ex):
        return {"prompt": ex["prompt"], "completion": ex["chosen"]}

    keep_cols = {"prompt", "row_id"}
    drop = [c for c in dataset["train"].column_names if c not in keep_cols]
    train_ds = dataset[script_args.dataset_train_split].map(
        to_prompt_completion, remove_columns=drop
    )
    holdout_raw = dataset[script_args.dataset_test_split]
    # The held-out split is always prepared (tokenized) so the final
    # diagnostic can score it; Trainer-driven evaluation stays governed by
    # eval_strategy (off by default -- the final checkpoint is prescribed, no
    # holdout-selected checkpoint, no early stopping).
    eval_ds = None
    if len(holdout_raw) > 0:
        eval_ds = holdout_raw.map(to_prompt_completion, remove_columns=drop)
        if script_args.holdout_max_rows is not None:
            eval_ds = eval_ds.select(range(min(script_args.holdout_max_rows, len(eval_ds))))

    truncation_stats = None
    if script_args.pretokenize:
        # Boundary-preserving length cap (see prepare_prompt_completion_dataset):
        # identical ids/labels to TRL's own preparation for rows within
        # max_length; over-long rows keep header, assistant boundary and the
        # supervised terminator instead of TRL's blind [:max_length] slice
        # (which drops the terminator, and drops prompt-only rows entirely).
        if training_args.max_length is None:
            raise ValueError("pretokenize requires max_length")
        train_ds, truncation_stats = prepare_prompt_completion_dataset(
            train_ds, tokenizer, training_args.max_length,
            min_completion=script_args.min_completion_tokens,
            num_proc=training_args.dataset_num_proc, desc="pretokenize train",
        )
        if eval_ds is not None:
            eval_ds, eval_trunc = prepare_prompt_completion_dataset(
                eval_ds, tokenizer, training_args.max_length,
                min_completion=script_args.min_completion_tokens,
                num_proc=training_args.dataset_num_proc, desc="pretokenize holdout",
            )
            truncation_stats = {"train": truncation_stats, "holdout": eval_trunc}
        else:
            truncation_stats = {"train": truncation_stats}
        for split, st in truncation_stats.items():
            if st["n_rows_without_supervised_terminator"] or st["n_rows_without_supervision"]:
                raise RuntimeError(f"{split}: rows lost their terminator/supervision: {st}")
        print("truncation policy: " + json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "truncated_rows"} for k, v in truncation_stats.items()}
        ))
    else:
        train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c == "row_id"])
        if eval_ds is not None:
            eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c == "row_id"])

    if rank0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        _write_split_manifests(training_args.output_dir, dataset)
        if truncation_stats is not None:
            with open(os.path.join(training_args.output_dir, "truncation_policy.json"), "w") as f:
                json.dump(truncation_stats, f, indent=1)

    callbacks = [_StepGeometryCallback()]
    if script_args.checkpoint_keep is not None:
        callbacks.append(_PruneBeforeSaveCallback(script_args.checkpoint_keep))
    is_moe = getattr(model.config, "num_experts", None) is not None
    if is_moe and script_args.moe_audit_steps:
        callbacks.append(_MoeGradAuditCallback(script_args.moe_audit_steps))
    if script_args.stop_after_step is not None:
        callbacks.append(_StopAfterStepCallback(script_args.stop_after_step))
    trainer_cls = _AuditedSFTTrainer if script_args.dump_first_batch else SFTTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks,
    )
    if is_moe:
        # TRL rewrites output_router_logits/router_aux_loss_coef from its own
        # SFTConfig.router_aux_loss_coef (default 0.001!). Make the outcome
        # visible; the config decides, this only records it.
        print(
            f"MoE router: aux_loss_enabled={trainer.aux_loss_enabled} "
            f"router_aux_loss_coef={model.config.router_aux_loss_coef} "
            f"output_router_logits={model.config.output_router_logits} "
            f"num_experts={model.config.num_experts} "
            f"num_experts_per_tok={model.config.num_experts_per_tok} "
            f"norm_topk_prob={model.config.norm_topk_prob}"
        )
    if rank0:
        _write_run_metadata(
            training_args.output_dir, script_args, training_args, model_args, model, tokenizer,
            fmt, dataset, trainer,
        )

    resume = _resolve_resume(training_args.resume_from_checkpoint, training_args.output_dir)
    print(f"resume_from_checkpoint -> {resume!r}")
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", train_result.metrics)
    if rank0:
        trainer.save_metrics("train", train_result.metrics)
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
    if training_args.eval_strategy != "no" and eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    stopped_early = (
        script_args.stop_after_step is not None
        and trainer.state.global_step < trainer.state.max_steps
    )
    if stopped_early:
        print(
            f"stopped early after step {trainer.state.global_step}/{trainer.state.max_steps} "
            "(stop_after_step); no final save/diagnostic -- resume with the same config."
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    if eval_ds is not None:
        holdout_diagnostic(
            trainer,
            os.path.join(training_args.output_dir, "holdout_diagnostic.json"),
            n_rows_total=len(holdout_raw),
        )

    if _uses_fsdp(training_args) and "SHARDED_STATE_DICT" in str(
        trainer.accelerator.state.fsdp_plugin.state_dict_type
    ):
        # Trainer.save_model is a no-op for SHARDED_STATE_DICT (only the
        # resumable dcp shards exist); gather the full state dict on rank 0's
        # CPU and write the servable bf16 artifact at the output root.
        save_full_bf16(trainer, training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
    if rank0:
        if fmt is not None:
            write_sidecar(training_args.output_dir, fmt, tokenizer)
        _append_json(
            os.path.join(training_args.output_dir, "run_metadata.json"),
            {"wall_time_s": round(time.time() - t_start, 1), "final_save": "done"},
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(f"train_sft complete: {training_args.output_dir}")


class _AuditedSFTTrainer:
    """Placeholder replaced at first use (SFTTrainer is imported lazily)."""

    def __new__(cls, *args, **kwargs):
        from trl import SFTTrainer

        class AuditedSFTTrainer(SFTTrainer):
            _dumped_first_batch = False

            def training_step(self, model, inputs, num_items_in_batch=None):
                if not self._dumped_first_batch and self.state.global_step == 0:
                    self._dumped_first_batch = True
                    if self.args.process_index == 0:
                        try:
                            dump_batch_audit(
                                inputs,
                                self.processing_class,
                                os.path.join(self.args.output_dir, "first_batch_audit.json"),
                            )
                        except Exception as exc:  # audit must never kill training
                            print(f"first-batch audit failed: {exc!r}")
                return super().training_step(model, inputs, num_items_in_batch)

        return AuditedSFTTrainer(*args, **kwargs)


def dump_batch_audit(inputs, tokenizer, path: str) -> None:
    """Write one collated microbatch as the trainer sees it (rank-local)."""
    ids = inputs["input_ids"].detach().cpu()
    labels = inputs["labels"].detach().cpu()
    attn = inputs.get("attention_mask")
    attn = attn.detach().cpu() if attn is not None else None
    rows = []
    for i in range(ids.shape[0]):
        seq = ids[i].tolist()
        lab = labels[i].tolist()
        am = attn[i].tolist() if attn is not None else [1] * len(seq)
        sup = [t for t, l in zip(seq, lab) if l != -100]
        rows.append({
            "n_tokens": int(sum(am)),
            "n_padding": int(len(am) - sum(am)),
            "n_supervised": len(sup),
            "padding_labels_all_ignored": all(
                l == -100 for l, a in zip(lab, am) if a == 0
            ),
            "supervised_text": tokenizer.decode(sup, skip_special_tokens=False),
            "prompt_text": tokenizer.decode(
                [t for t, l, a in zip(seq, lab, am) if l == -100 and a == 1],
                skip_special_tokens=False,
            ),
            "last_supervised_token": tokenizer.convert_ids_to_tokens(sup[-1]) if sup else None,
            "input_ids": seq,
            "labels": lab,
            "attention_mask": am,
        })
    with open(path, "w") as f:
        json.dump({"batch_shape": list(ids.shape), "rows": rows}, f, indent=1)
    print(f"first-batch audit written: {path} shape={list(ids.shape)}")


def _append_json(path: str, extra: dict) -> None:
    data = {}
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    data.update(extra)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_split_manifests(output_dir: str, dataset) -> None:
    """Ordered row-id manifests of the exact train/holdout membership."""
    for split, name in (("train", "train_row_ids.txt"), ("test", "holdout_row_ids.txt")):
        ds = dataset[split]
        if "row_id" in ds.column_names:
            ids = ds["row_id"]
        else:
            ids = [str(i) for i in range(len(ds))]
        with open(os.path.join(output_dir, name), "w") as f:
            f.write("\n".join(ids) + ("\n" if ids else ""))


def _write_run_metadata(
    output_dir, script_args, training_args, model_args, model, tokenizer, fmt, dataset, trainer
) -> None:
    import hashlib

    import accelerate
    import datasets as datasets_lib
    import torch
    import transformers
    import trl

    def sha256_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    cfg = model.config
    fsdp_cfg = training_args.fsdp_config if isinstance(training_args.fsdp_config, dict) else None
    meta = {
        "model_name_or_path": model_args.model_name_or_path,
        "model_revision": model_args.model_revision,
        "architectures": cfg.architectures,
        "model_config": {
            k: getattr(cfg, k)
            for k in (
                "model_type", "num_hidden_layers", "hidden_size", "vocab_size", "dtype",
                "use_cache", "num_experts", "num_experts_per_tok", "norm_topk_prob",
                "router_aux_loss_coef", "output_router_logits", "decoder_sparse_step",
                "moe_intermediate_size", "sliding_window", "layer_types", "rope_parameters",
                "max_position_embeddings",
            )
            if hasattr(cfg, k)
        },
        "dataset_name": script_args.dataset_name,
        "dataset_sha256": sha256_file(script_args.dataset_name)
        if os.path.isfile(script_args.dataset_name) else None,
        "val_fraction": script_args.val_fraction,
        "split_seed": training_args.seed,
        "n_train_rows": len(dataset["train"]),
        "n_holdout_rows": len(dataset["test"]),
        "chat_format": fmt.sidecar(tokenizer) if fmt else None,
        "chat_template_sha256": hashlib_sha256(tokenizer.chat_template or ""),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
        },
        "generation_config": {
            "eos_token_id": _eos_list(model.generation_config.eos_token_id),
            "pad_token_id": model.generation_config.pad_token_id,
        },
        "training_args": {
            k: getattr(training_args, k)
            for k in (
                "output_dir", "run_name", "seed", "data_seed", "num_train_epochs", "max_steps",
                "learning_rate", "lr_scheduler_type", "warmup_steps", "warmup_ratio", "optim",
                "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay", "max_grad_norm",
                "per_device_train_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "bf16", "fp16", "gradient_checkpointing",
                "max_length", "packing", "completion_only_loss", "assistant_only_loss",
                "truncation_mode", "padding_free", "eval_strategy", "save_strategy",
                "save_steps", "save_total_limit", "save_only_model", "logging_steps",
                "report_to", "fsdp", "dataloader_drop_last", "average_tokens_across_devices",
                "router_aux_loss_coef", "use_liger_kernel", "activation_offloading",
                "resume_from_checkpoint", "ddp_timeout",
            )
            if hasattr(training_args, k)
        },
        "fsdp_config": fsdp_cfg,
        "script_args": {
            "chat_format": script_args.chat_format,
            "holdout_max_rows": script_args.holdout_max_rows,
            "checkpoint_keep": script_args.checkpoint_keep,
            "moe_audit_steps": script_args.moe_audit_steps,
            **{
                k: getattr(script_args, k)
                for k in ("expect_raw_rows", "expect_train_rows", "expect_max_steps", "expect_warmup_steps",
                          "expect_world_size", "check_reference_frozen", "base_revision")
                if hasattr(script_args, k)
            },
        },
        "loss_type": getattr(training_args, "loss_type", None),
        "beta": getattr(training_args, "beta", None),
        "label_smoothing": getattr(training_args, "label_smoothing", None),
        "precompute_ref_log_probs": getattr(training_args, "precompute_ref_log_probs", None),
        "config_file_sha256": {
            p: sha256_file(p) for p in _config_files_from_argv(sys.argv) if os.path.isfile(p)
        },
        "fsdp_config_sha256": sha256_file(training_args.fsdp_config)
        if isinstance(training_args.fsdp_config, str) and os.path.isfile(training_args.fsdp_config)
        else None,
        "container_image": os.environ.get("VALUEGEN_CONTAINER_IMAGE"),
        "model_args": {
            "dtype": model_args.dtype, "attn_implementation": model_args.attn_implementation,
            "use_peft": model_args.use_peft,
        },
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "effective_global_batch": (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * int(os.environ.get("WORLD_SIZE", "1"))
        ),
        "versions": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "trl": trl.__version__,
            "accelerate": accelerate.__version__, "datasets": datasets_lib.__version__,
        },
        "rocm": getattr(torch.version, "hip", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if getattr(trainer, "aux_loss_enabled", None) is not None:
        meta["moe_aux_loss_enabled"] = bool(trainer.aux_loss_enabled)
    _append_json(os.path.join(output_dir, "run_metadata.json"), meta)


def _config_files_from_argv(argv: list[str]) -> list[str]:
    """``--config X`` / ``--config=X`` values in a TrlParser argv (for hashing)."""
    out = []
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif a.startswith("--config="):
            out.append(a.split("=", 1)[1])
    return out


class _StepGeometryCallback:
    """Log the resolved step geometry (and FSDP wrapping facts) at train start."""

    def __new__(cls):
        from transformers import TrainerCallback

        class StepGeometryCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                model = kwargs.get("model")
                train_dl = kwargs.get("train_dataloader")
                geometry = {
                    "world_size": args.world_size,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_global_batch": args.per_device_train_batch_size
                    * args.gradient_accumulation_steps * args.world_size,
                    "batches_per_epoch_per_rank": len(train_dl) if train_dl is not None else None,
                    "max_steps": state.max_steps,
                    "num_train_epochs": state.num_train_epochs,
                    "resolved_warmup_steps": args.get_warmup_steps(state.max_steps),
                    "warmup_steps_arg": args.warmup_steps,
                    # .value: HF SchedulerType/OptimizerNames are ExplicitEnum
                    # (str, Enum); under Python 3.12 str(enum) is the repr
                    # "SchedulerType.COSINE", not "cosine" — emit the canonical
                    # value so the geometry report and launch gates stay clean.
                    "lr_scheduler_type": str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)),
                    "final_scheduled_step": state.max_steps,
                    "global_step_at_train_begin": state.global_step,
                    "optim": str(getattr(args.optim, "value", args.optim)),
                }
                if train_dl is not None:
                    try:
                        geometry["samples_per_rank_per_epoch"] = len(train_dl.dataset) if not hasattr(
                            train_dl, "total_dataset_length"
                        ) else None
                        geometry["total_dataset_length"] = getattr(train_dl, "total_dataset_length", None)
                        geometry["total_batch_size_reported_by_loader"] = getattr(
                            train_dl, "total_batch_size", None
                        )
                    except Exception:
                        pass
                if model is not None:
                    geometry.update(fsdp_wrapping_report(model))
                    try:
                        cfg = getattr(model, "config", None)
                        if cfg is not None:
                            geometry["use_cache"] = cfg.use_cache
                    except Exception:
                        pass
                if args.process_index == 0:
                    with open(os.path.join(args.output_dir, "step_geometry.json"), "w") as f:
                        json.dump(geometry, f, indent=2, default=str)
                    print(f"step geometry: {json.dumps(geometry, default=str)}")

            def on_log(self, args, state, control, logs=None, **kwargs):
                if args.process_index == 0 and logs and "loss" in logs:
                    import torch

                    if torch.cuda.is_available():
                        logs["gpu_mem_alloc_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
                        logs["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

        return StepGeometryCallback()


def fsdp_wrapping_report(model) -> dict:
    """What the wrapped model really is: FSDP2 (fully_shard/DTensor) vs FSDP1,
    which classes are sharded, activation-checkpoint wrappers, offload."""
    import torch

    report = {"fsdp_version": None, "fsdp_units": 0, "fsdp_unit_classes": {},
              "activation_checkpoint_wrappers": 0, "ac_wrapped_classes": {},
              "dtensor_params": 0, "plain_params": 0, "param_dtypes": {},
              "cpu_offload": None, "sharded_all_trainable": None}
    try:
        from torch.distributed.fsdp import FSDPModule
    except Exception:
        FSDPModule = None
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
    except Exception:
        FSDP1 = None
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    except Exception:
        CheckpointWrapper = None
    fsdp1_units = 0
    for name, module in model.named_modules():
        if FSDPModule is not None and isinstance(module, FSDPModule):
            report["fsdp_units"] += 1
            inner = getattr(module, "_checkpoint_wrapped_module", module)
            cls = type(inner).__name__
            for base in type(module).__mro__:
                if base.__name__ not in ("FSDPModule",) and "FSDP" not in base.__name__ and base is not object:
                    cls = base.__name__
                    break
            report["fsdp_unit_classes"][cls] = report["fsdp_unit_classes"].get(cls, 0) + 1
            try:
                state = module._get_fsdp_state()
                pg = state._fsdp_param_group
                if pg is not None and report["cpu_offload"] is None:
                    report["cpu_offload"] = type(pg.offload_policy).__name__
            except Exception:
                pass
        if FSDP1 is not None and isinstance(module, FSDP1):
            fsdp1_units += 1
        if CheckpointWrapper is not None and isinstance(module, CheckpointWrapper):
            report["activation_checkpoint_wrappers"] += 1
            cls = type(module._checkpoint_wrapped_module).__name__
            report["ac_wrapped_classes"][cls] = report["ac_wrapped_classes"].get(cls, 0) + 1
    from torch.distributed.tensor import DTensor

    unsharded_trainable = []
    for name, p in model.named_parameters():
        if isinstance(p, DTensor):
            report["dtensor_params"] += 1
        else:
            report["plain_params"] += 1
            if p.requires_grad:
                unsharded_trainable.append(name)
        key = str(p.dtype)
        report["param_dtypes"][key] = report["param_dtypes"].get(key, 0) + 1
    report["unsharded_trainable_params"] = unsharded_trainable[:20]
    report["sharded_all_trainable"] = not unsharded_trainable
    if report["fsdp_units"] > 0:
        report["fsdp_version"] = 2
    elif fsdp1_units > 0:
        report["fsdp_version"] = 1
        report["fsdp_units"] = fsdp1_units
    return report


class _PruneBeforeSaveCallback:
    """Delete older checkpoint-* dirs right before a new save (storage cap)."""

    def __new__(cls, keep: int):
        from transformers import TrainerCallback

        class PruneBeforeSaveCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if not control.should_save:
                    return control
                if args.process_index == 0:
                    ckpts = []
                    for p in Path(args.output_dir).glob("checkpoint-*"):
                        if p.is_dir():
                            try:
                                ckpts.append((int(p.name.rsplit("-", 1)[-1]), p))
                            except ValueError:
                                continue
                    ckpts.sort()
                    n_keep = max(keep - 1, 0)
                    for _, p in ckpts[: len(ckpts) - n_keep] if n_keep < len(ckpts) else []:
                        print(f"pruning checkpoint before save (keep={keep}): {p}")
                        shutil.rmtree(p, ignore_errors=True)
                import torch

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                return control

        return PruneBeforeSaveCallback()


class _MoeGradAuditCallback:
    """Router / expert / shared / dense gradient audit at chosen optimizer steps.

    Runs in ``on_pre_optimizer_step`` (after clipping, before the update): for
    every rank the local DTensor shard of each parameter of interest is
    inspected -- expert tensors are sharded on dim 0 (one slice of experts per
    rank), so per-expert gradient norms come from the local shard and are
    gathered across ranks. Written by rank 0 to ``moe_grad_audit.json``.
    """

    def __new__(cls, steps_spec: str):
        from transformers import TrainerCallback

        class MoeGradAuditCallback(TrainerCallback):
            def __init__(self):
                self.spec = [s.strip() for s in steps_spec.split(",") if s.strip()]
                self.done = set()

            def _wanted(self, state):
                step = state.global_step + 1  # the optimizer step about to happen
                for s in self.spec:
                    if s == "first" and step == 1:
                        return True
                    if s == "last" and step == state.max_steps:
                        return True
                    if s.isdigit() and step == int(s):
                        return True
                return False

            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                step = state.global_step + 1
                if not self._wanted(state) or step in self.done:
                    return control
                self.done.add(step)
                model = kwargs.get("model")
                try:
                    report = moe_grad_report(model)
                except Exception as exc:
                    report = {"error": repr(exc)}
                if args.process_index == 0:
                    path = os.path.join(args.output_dir, "moe_grad_audit.json")
                    data = {}
                    if os.path.isfile(path):
                        with open(path) as f:
                            data = json.load(f)
                    data[f"step_{step}"] = report
                    with open(path, "w") as f:
                        json.dump(data, f, indent=1, default=str)
                    summary = {k: v for k, v in report.items() if k != "per_expert"}
                    print(f"MoE grad audit step {step}: {json.dumps(summary, default=str)[:4000]}")
                return control

        return MoeGradAuditCallback()


def moe_grad_report(model) -> dict:
    """Gradient statistics for router, expert, shared-expert and dense params."""
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    def local(t):
        return t.to_local() if isinstance(t, DTensor) else t

    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if world > 1 else 0
    groups = {"router": [], "experts": [], "shared_expert": [], "attention": [], "dense_mlp": [],
              "embeddings": [], "norms": [], "other": []}
    per_expert = {}  # layer -> {"gate_up": [norms], "down": [norms]} for local experts
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        entry = {"name": name, "shape": list(p.shape)}
        if g is None:
            entry["grad"] = None
        else:
            gl = local(g).float()
            entry["grad_norm"] = float(gl.norm())
            entry["finite"] = bool(torch.isfinite(gl).all())
            entry["nonzero_frac"] = float((gl != 0).float().mean())
        lname = name.lower()
        if ".gate.weight" in lname or "router" in lname:
            groups["router"].append(entry)
        elif "shared_expert" in lname:
            groups["shared_expert"].append(entry)
        elif "experts." in lname:
            groups["experts"].append(entry)
            if g is not None:
                gl = local(g).float()
                layer = name.split(".layers.")[1].split(".")[0] if ".layers." in name else "?"
                key = "gate_up" if "gate_up" in name else "down"
                norms = gl.reshape(gl.shape[0], -1).norm(dim=1).tolist()
                per_expert.setdefault(layer, {})[key] = norms
        elif "self_attn" in lname or "attn" in lname:
            groups["attention"].append(entry)
        elif "mlp" in lname and "experts" not in lname:
            groups["dense_mlp"].append(entry)
        elif "embed" in lname or "lm_head" in lname:
            groups["embeddings"].append(entry)
        elif "norm" in lname:
            groups["norms"].append(entry)
        else:
            groups["other"].append(entry)

    def summarize(entries):
        norms = [e["grad_norm"] for e in entries if e.get("grad_norm") is not None]
        return {
            "n_params": len(entries),
            "n_with_grad": len(norms),
            "n_finite": sum(1 for e in entries if e.get("finite")),
            "n_nonzero": sum(1 for e in entries if e.get("grad_norm")),
            "min_norm": min(norms) if norms else None,
            "max_norm": max(norms) if norms else None,
            "examples": [
                {k: e[k] for k in ("name", "grad_norm", "finite", "nonzero_frac") if k in e}
                for e in entries[:4]
            ],
        }

    # Gather per-expert local norms so rank 0 sees every expert of a few layers.
    gathered = [None] * world
    sample_layers = sorted(per_expert, key=lambda s: int(s) if s.isdigit() else 0)
    sample_layers = [l for i, l in enumerate(sample_layers) if i % max(1, len(sample_layers) // 4) == 0][:5]
    local_sample = {l: per_expert[l] for l in sample_layers}
    if world > 1:
        dist.all_gather_object(gathered, {"rank": rank, "experts": local_sample})
    else:
        gathered = [{"rank": 0, "experts": local_sample}]
    merged = {}
    for item in gathered:
        for layer, d in item["experts"].items():
            m = merged.setdefault(layer, {"gate_up": [], "down": []})
            for key in ("gate_up", "down"):
                m[key].extend(d.get(key, []))
    expert_summary = {}
    for layer, d in merged.items():
        gu = d["gate_up"]
        expert_summary[layer] = {
            "n_experts_seen": len(gu),
            "n_nonzero_grad_experts": sum(1 for x in gu if x > 0),
            "n_finite": sum(1 for x in gu if x == x and x not in (float("inf"),)),
            "min_norm": min(gu) if gu else None,
            "max_norm": max(gu) if gu else None,
        }
    return {
        "world_size": world,
        "groups": {k: summarize(v) for k, v in groups.items()},
        "expert_layers_sampled": expert_summary,
        "per_expert": {l: merged[l] for l in list(merged)[:2]},
    }


def holdout_diagnostic(trainer, out_path: str, n_rows_total: int | None = None) -> dict:
    """Completion-token NLL over the prepared held-out split, token-weighted.

    Deterministic sharding (rank r scores rows r::world), no sample
    duplication, sums all-reduced. Records loss = NLL sum / supervised
    tokens, exp(loss) guarded against overflow, token and row counts, and for
    MoE models the expert-selection statistics observed on these batches.
    """
    import math

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    eval_ds = trainer.eval_dataset
    if eval_ds is None:
        return {}
    model = trainer.model
    was_training = model.training
    model.eval()
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if world > 1 else 0
    idx = list(range(rank, len(eval_ds), world))
    shard = eval_ds.select(idx) if idx else eval_ds.select([])
    cols = [c for c in ("input_ids", "labels") if c in shard.column_names]
    shard = shard.select_columns(cols)
    loader = DataLoader(
        shard, batch_size=max(1, trainer.args.per_device_eval_batch_size), shuffle=False,
        collate_fn=trainer.data_collator,
    )
    device = trainer.args.device
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    n_tok = torch.zeros((), dtype=torch.float64, device=device)
    n_rows = torch.zeros((), dtype=torch.float64, device=device)
    router_counts = {}
    hooks = []
    cfg = getattr(model, "config", None)
    num_experts = getattr(cfg, "num_experts", None)
    if num_experts:
        def make_hook(layer_name):
            def hook(module, inputs, output):
                idxs = output[2] if isinstance(output, tuple) and len(output) >= 3 else None
                if idxs is None:
                    return
                counts = torch.bincount(idxs.reshape(-1).to(torch.int64), minlength=num_experts)
                router_counts[layer_name] = router_counts.get(layer_name, 0) + counts.detach().cpu()
            return hook
        for name, module in model.named_modules():
            if type(module).__name__.endswith("TopKRouter"):
                hooks.append(module.register_forward_hook(make_hook(name)))
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits[:, :-1, :].float()
            tgt = labels[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="sum"
            )
            loss_sum += loss.double()
            n_tok += (tgt != -100).sum().double()
            n_rows += float(input_ids.shape[0])
            del out, logits
    for h in hooks:
        h.remove()
    if world > 1:
        for t in (loss_sum, n_tok, n_rows):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    mean = float(loss_sum / n_tok) if float(n_tok) > 0 else float("nan")
    ppl = math.exp(mean) if mean == mean and mean < 700 else float("inf")
    result = {
        "completion_token_loss": mean,
        "perplexity": ppl,
        "n_supervised_tokens": int(n_tok.item()),
        "n_rows_scored": int(n_rows.item()),
        "n_rows_in_split": n_rows_total,
        "n_rows_prepared": len(eval_ds),
        "global_step": trainer.state.global_step,
        "seconds": round(time.time() - t0, 1),
        "loss_is_finite": mean == mean and abs(mean) != float("inf"),
    }
    if router_counts:
        # Every rank saw different rows; reduce the counts.
        stats = {}
        for layer, counts in sorted(router_counts.items()):
            c = counts.to(device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(c, op=dist.ReduceOp.SUM)
            total = float(c.sum())
            p = c / max(total, 1.0)
            ent = float(-(p[p > 0] * p[p > 0].log()).sum())
            stats[layer] = {
                "experts_hit": int((c > 0).sum()),
                "num_experts": int(num_experts),
                "max_load_share": float(p.max()),
                "min_load_share": float(p.min()),
                "normalized_entropy": ent / math.log(num_experts),
                "top_expert": int(c.argmax()),
            }
        shares = [s["experts_hit"] / s["num_experts"] for s in stats.values()]
        result["expert_selection"] = {
            "n_router_layers": len(stats),
            "min_frac_experts_hit": min(shares) if shares else None,
            "mean_frac_experts_hit": sum(shares) / len(shares) if shares else None,
            "min_normalized_entropy": min(s["normalized_entropy"] for s in stats.values()) if stats else None,
            "max_load_share_overall": max(s["max_load_share"] for s in stats.values()) if stats else None,
            "collapsed": any(s["experts_hit"] <= cfg.num_experts_per_tok for s in stats.values()) if stats else None,
            "per_layer": stats,
        }
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        summary = {k: v for k, v in result.items() if k != "expert_selection"}
        if "expert_selection" in result:
            summary["expert_selection"] = {
                k: v for k, v in result["expert_selection"].items() if k != "per_layer"
            }
        print(f"holdout diagnostic: {json.dumps(summary, default=str)}")
    if was_training:
        model.train()
    return result


def save_full_bf16(trainer, output_dir: str) -> None:
    """Final servable save for a SHARDED_STATE_DICT FSDP run.

    Gathers the full fp32 state dict to rank 0's CPU (every rank must call:
    it is a collective), casts to bf16 there, and writes sharded safetensors +
    config + generation_config + tokenizer at ``output_dir``. No GPU ever
    holds a full copy; rank 0 CPU peaks at ~fp32 + bf16 of the model.
    """
    import torch
    import torch.distributed as dist
    from transformers.modeling_utils import unwrap_model

    state_dict = trainer.accelerator.get_state_dict(trainer.model)  # collective; rank 0 gets it
    if trainer.args.process_index == 0:
        bf16 = {}
        for k, v in state_dict.items():
            bf16[k] = v.to(torch.bfloat16) if torch.is_floating_point(v) else v
        del state_dict
        unwrapped = unwrap_model(trainer.model)
        unwrapped.config.dtype = torch.bfloat16
        unwrapped.save_pretrained(output_dir, state_dict=bf16, safe_serialization=True)
        trainer.processing_class.save_pretrained(output_dir)
        unwrapped.generation_config.save_pretrained(output_dir)
        del bf16
        print(f"final bf16 save written: {output_dir}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _add_chat_family_arg(parser: argparse.ArgumentParser, flag: str) -> None:
    parser.add_argument(
        "--chat-family",
        choices=CHAT_FAMILIES,
        default=None,
        help=f"Explicit chat-template family. Required only when {flag} "
        "carries no recognizable family keyword; must agree with the path "
        "when it does.",
    )


def _check_chat_family(
    parser: argparse.ArgumentParser, flag: str, path: str, chat_family: str | None
) -> None:
    """The merged-dir naming refusal shared by ``merge`` and ``export``."""
    inferred = infer_chat_family(path)
    if inferred is None and chat_family is None:
        parser.error(
            f"{flag} {path!r} contains no chat-family keyword "
            f"({'/'.join(CHAT_FAMILIES)}); conflictscope selects the chat template "
            "by path substring, so this checkpoint would be served with the "
            "wrong template. Rename the dir or pass --chat-family to confirm."
        )
    if inferred is not None and chat_family not in (None, inferred):
        parser.error(
            f"--chat-family {chat_family} disagrees with the family the "
            f"path actually selects ({inferred}); the path substring wins at "
            f"eval time, so rename {flag} instead."
        )


def merge(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="valuegen.ground_truth.training merge",
        description="Merge the latest LoRA checkpoint into its base model.",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Trainer output_dir; the highest-step checkpoint-* inside is merged.",
    )
    parser.add_argument("--merged-path", required=True)
    _add_chat_family_arg(parser, "--merged-path")
    args = parser.parse_args(argv)
    _check_chat_family(parser, "--merged-path", args.merged_path, args.chat_family)

    merged_path = Path(args.merged_path)
    if (merged_path / "config.json").is_file():
        print(f"Merged model already exists, skipping: {merged_path}")
        return
    checkpoint = latest_checkpoint(args.checkpoint_root)
    print(f"Merging {checkpoint} -> {merged_path}")
    merge_lora(args.base_model, checkpoint, merged_path, chat_family=args.chat_family)


def export(argv: list[str]) -> None:
    """Full-FT counterpart of ``merge``: trainer output -> servable bf16 dir."""
    parser = argparse.ArgumentParser(
        prog="valuegen.ground_truth.training export",
        description="Re-save a full-FT trainer output as a bf16 servable dir.",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--trained",
        required=True,
        help="Trainer output_dir; the final save_model at its root is "
        "exported (never a checkpoint-* subdir — with save_strategy \"no\" "
        "the root is the only complete artifact, and with intermediate "
        "checkpointing it is still the final save).",
    )
    parser.add_argument("--out", required=True)
    _add_chat_family_arg(parser, "--out")
    parser.add_argument(
        "--chat-format",
        choices=sorted(CHAT_FORMATS),
        default=None,
        help="Explicit chat-format identity (chat_formats.CHAT_FORMATS). Wins "
        "over --chat-family/path keywords; the artifact carries it in a "
        "valuegen_chat_format.json sidecar.",
    )
    parser.add_argument("--base-revision", default=None, help="Hub revision of --base-model")
    parser.add_argument(
        "--verify", action="store_true",
        help="After export (or when it already exists), reload the dir independently and verify.",
    )
    parser.add_argument(
        "--prune-trained", action="store_true",
        help="Once the export verifies (implies --verify), drop --trained's weight "
        "shards and checkpoint-* dirs (prune_trainer_dir): a full-FT fp32 save is "
        "2x the export and nothing reads it again. trainer_state, configs and "
        "logs stay.",
    )
    args = parser.parse_args(argv)
    fmt = get_chat_format(args.chat_format) if args.chat_format else None
    if fmt is None:
        _check_chat_family(parser, "--out", args.out, args.chat_family)

    out = Path(args.out)
    trained = Path(args.trained)
    if (out / "config.json").is_file():
        print(f"Exported model already exists, skipping: {out}")
    else:
        if not (trained / "config.json").is_file():
            sys.exit(
                f"No config.json under {trained}: the trainer's final save_model "
                "never ran (training failed or was interrupted) — or this is a "
                "LoRA output dir, which the `merge` subcommand handles instead."
            )
        print(f"Exporting {trained} -> {out}")
        export_full_ft(
            args.base_model, trained, out, chat_family=args.chat_family, chat_format=fmt,
            base_revision=args.base_revision,
        )
    if args.verify or args.prune_trained:
        result = verify_export(out, chat_format=fmt)
        with open(out / "export_verification.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"export verified: {json.dumps(result, default=str)}")
    if args.prune_trained and prune_trainer_dir(trained, checkpoints=True):
        print(f"pruned trainer weights: {trained}")


def main() -> None:
    commands = {"dpo": train_dpo, "sft": train_sft, "merge": merge, "export": export}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(
            f"usage: python -m valuegen.ground_truth.training "
            f"{{{'|'.join(commands)}}} ..."
        )
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
