"""Explicit chat-format identities (no path-substring heuristics).

:mod:`training` selects the chat template of a run by a keyword in the model
path (``CHAT_FAMILIES``/``infer_chat_family``); conflictscope does the same at
eval time. That dispatch is wrong for the OLMo-3 collaborator template (the
un-hyphenated ``olmo3`` spelling lands on the OLMo-2/Tulu template, the
hyphenated one on plain ChatML with the wrong terminator) and fragile for
Qwen (``qwen3-`` names get a ``<think>`` scaffold). A :class:`ChatFormat` is
the explicit alternative: the training script, the export, the served
artifact and the evaluator all name the same identity, and the artifact
carries it in a ``valuegen_chat_format.json`` sidecar so nothing downstream
has to guess from the directory name.

Every template is a byte-exact copy of a published artifact's
``chat_template.jinja`` (see ``provenance``), pinned under
``configs/chat_templates/`` and hashed in run manifests.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES_DIR = REPO_ROOT / "configs" / "chat_templates"
SIDECAR_NAME = "valuegen_chat_format.json"


@dataclass(frozen=True)
class ChatFormat:
    name: str
    template_file: str
    # Token that closes the FINAL assistant turn: the completion is trained to
    # end in it and nothing else, so it is the served stop token.
    turn_ender: str
    # Further stop tokens the artifact's generation_config declares after the
    # turn-ender (secondary stops; the served set is [turn_ender, *extra_stops]).
    extra_stops: tuple[str, ...]
    # Pad token the artifact declares (None: keep the base tokenizer's).
    pad_token: str | None
    # Which conflictscope formatter renders the PROMPT bytes at eval time
    # ("qwen" == ChatML: <|im_start|>role\n...<|im_end|>\n, generation prompt
    # "<|im_start|>assistant\n", no thinking scaffold).
    conflictscope_formatter: str
    provenance: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def template_path(self) -> Path:
        return TEMPLATES_DIR / self.template_file

    @property
    def template(self) -> str:
        return self.template_path.read_text(encoding="utf-8")

    @property
    def template_sha256(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def stop_tokens(self) -> tuple[str, ...]:
        return (self.turn_ender, *self.extra_stops)

    def render(self, messages, add_generation_prompt: bool, **kwargs) -> str:
        """Render with the same Jinja settings transformers uses."""
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        # transformers exposes {% generation %} blocks via its own extension;
        # for a plain render they must simply be transparent.
        src = self.template.replace("{%- generation -%}", "").replace(
            "{%- endgeneration -%}", ""
        )
        return env.from_string(src).render(
            messages=copy.deepcopy(messages),
            add_generation_prompt=add_generation_prompt,
            tools=kwargs.pop("tools", None),
            eos_token=kwargs.pop("eos_token", self.turn_ender),
            **kwargs,
        )

    def sidecar(self, tokenizer=None) -> dict:
        d = {
            "chat_format": self.name,
            "template_file": str(self.template_path.relative_to(REPO_ROOT)),
            "template_sha256": self.template_sha256,
            "turn_ender": self.turn_ender,
            "extra_stops": list(self.extra_stops),
            "pad_token": self.pad_token,
            "conflictscope_formatter": self.conflictscope_formatter,
            "provenance": self.provenance,
        }
        if tokenizer is not None:
            d["stop_token_ids"] = [tokenizer.convert_tokens_to_ids(t) for t in self.stop_tokens()]
            d["pad_token_id"] = tokenizer.pad_token_id
        return d


CHAT_FORMATS: dict[str, ChatFormat] = {
    # Collaborator/published OLMo-3 ChatML: user/system turns are ChatML,
    # non-final assistant turns end "<|im_end|>\n", the FINAL assistant turn
    # ends with eos_token = <|endoftext|> (100257). {% generation %} markers
    # wrap assistant content + terminator. Tools-gated function-calling system
    # turn. Published generation_config: eos [100257], pad 100277 (<|pad|>) --
    # <|im_end|> is deliberately NOT a stop token (byte-level comparison with
    # the published artifact: generation_config.eos_token_id == [100257]).
    "olmo3_chatml": ChatFormat(
        name="olmo3_chatml",
        template_file="olmo3_chatml.jinja",
        turn_ender="<|endoftext|>",
        extra_stops=(),
        pad_token="<|pad|>",
        conflictscope_formatter="qwen",
        provenance=(
            "value-generalization/neutral-sft-v3-olmo3-7b@"
            "e93fa4d880af487009cb828b93665e88a428cc16:chat_template.jinja "
            "(sha256 51c7f8c700135ac224994f0d4a8b33562e9fe8edf9d33ad50909bed0a84ca9b6); "
            "generation_config eos [100257], pad 100277"
        ),
        notes=(
            "Intermediate assistant turns end <|im_end|>\\n; the final one ends <|endoftext|>.",
            "Not the legacy 'olmo' family (OLMo-2/Tulu <|user|>/<|assistant|> template).",
        ),
    ),
    # Plain Qwen ChatML: every turn "<|im_start|>role\ncontent<|im_end|>",
    # joined by "\n", generation prompt "\n<|im_start|>assistant\n"; NO
    # "<think>\n\n</think>\n\n" scaffold. Bytes identical to
    # training.CHAT_TEMPLATES["qwen"]. Published generation_config eos
    # [151645 (<|im_end|>), 151643 (<|endoftext|>)], pad 151643.
    "qwen_chatml": ChatFormat(
        name="qwen_chatml",
        template_file="qwen_chatml.jinja",
        turn_ender="<|im_end|>",
        extra_stops=("<|endoftext|>",),
        pad_token="<|endoftext|>",
        conflictscope_formatter="qwen",
        provenance=(
            "value-generalization/neutral-sft-v3-qwen3-8b@"
            "f1550fe714e01e2116cc4fdf3a1d510f8fd379c7:chat_template.jinja "
            "(sha256 333c256df3738c5351d58126b41fb323efae0514c1b7df837b11107d95441cdf); "
            "generation_config eos [151645, 151643], pad 151643"
        ),
        notes=("No <think> scaffolding at training, serving, or evaluation time.",),
    ),
}


def get_chat_format(name: str) -> ChatFormat:
    try:
        return CHAT_FORMATS[name]
    except KeyError:
        raise KeyError(
            f"Unknown chat format {name!r}; known: {sorted(CHAT_FORMATS)}"
        ) from None


def write_sidecar(out_dir: str | Path, fmt: ChatFormat, tokenizer=None) -> Path:
    path = Path(out_dir) / SIDECAR_NAME
    path.write_text(json.dumps(fmt.sidecar(tokenizer), indent=2) + "\n")
    return path


def read_sidecar(model_dir: str | Path) -> dict | None:
    path = Path(model_dir) / SIDECAR_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def check_model_dir(model_dir: str | Path, expect: str) -> dict:
    """Verify that a model dir carries the explicit chat-format ``expect``.

    Checks the ``valuegen_chat_format.json`` sidecar (name + template hash),
    the served ``chat_template`` (byte-identical to the pinned file), the
    tokenizer eos/pad tokens and the leading generation eos ids. Raises
    ``RuntimeError`` listing every mismatch; returns the verified summary.
    """
    from transformers import AutoTokenizer, GenerationConfig

    fmt = get_chat_format(expect)
    model_dir = Path(model_dir)
    problems = []
    sidecar = read_sidecar(model_dir)
    if sidecar is None:
        problems.append(f"no {SIDECAR_NAME} sidecar")
    else:
        if sidecar.get("chat_format") != fmt.name:
            problems.append(f"sidecar names {sidecar.get('chat_format')!r}, expected {fmt.name!r}")
        if sidecar.get("template_sha256") != fmt.template_sha256:
            problems.append("sidecar template_sha256 differs from the pinned template")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.chat_template != fmt.template:
        problems.append("served chat_template is not byte-identical to the pinned template")
    if tokenizer.eos_token != fmt.turn_ender:
        problems.append(f"tokenizer eos {tokenizer.eos_token!r} != {fmt.turn_ender!r}")
    if fmt.pad_token is not None and tokenizer.pad_token != fmt.pad_token:
        problems.append(f"tokenizer pad {tokenizer.pad_token!r} != {fmt.pad_token!r}")
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in fmt.stop_tokens()]
    gen_eos = None
    if (model_dir / "generation_config.json").is_file():
        gen = GenerationConfig.from_pretrained(str(model_dir))
        eos = gen.eos_token_id
        gen_eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        if gen_eos[: len(stop_ids)] != stop_ids:
            problems.append(f"generation eos {gen_eos} does not start with {stop_ids}")
    else:
        problems.append("no generation_config.json")
    summary = {
        "model_dir": str(model_dir),
        "chat_format": fmt.name,
        "template_sha256": fmt.template_sha256,
        "stop_token_ids": stop_ids,
        "generation_eos_token_id": gen_eos,
        "tokenizer_eos": tokenizer.eos_token,
        "tokenizer_pad": tokenizer.pad_token,
        "problems": problems,
    }
    if problems:
        raise RuntimeError(
            f"{model_dir}: chat-format check for {fmt.name} failed -- " + "; ".join(problems)
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m valuegen.ground_truth.chat_formats")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="verify a model dir's explicit chat-format identity")
    check.add_argument("--model-dir", required=True)
    check.add_argument("--expect", required=True, choices=sorted(CHAT_FORMATS))
    args = parser.parse_args(argv)
    if args.command == "check":
        summary = check_model_dir(args.model_dir, args.expect)
        print(f"chat format verified: {json.dumps(summary)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
