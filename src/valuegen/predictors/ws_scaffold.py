"""Render weight-steering SFT pairs into axolotl ``input_output`` segments.

The weight_steer base-model arm trains LoRAs on pretrained bases (Qwen3-8B-Base,
Olmo-3-1025-7B) whose chat tags are untrained, so the pairs must be rendered
through a ``training.SCAFFOLD_TEMPLATES`` scaffold (urial0) instead of the
tokenizer's chat template — the weight-space analog of persona/grad_proj's
``template=urial0`` param.

The fork's ``cs_prep_data.py`` writes ``{value}_{pol}.jsonl`` as bare
``{"messages": [user, assistant]}`` (the steering system prompt is stripped by
design — the ± answer contrast is the steering signal). This converter reads
those and writes ``{value}_{pol}.io.jsonl`` in axolotl's template-free format::

    {"segments": [{"label": false, "text": <prompt>}, {"label": true, "text": <answer>}]}

The prompt is ``render_chat_template(scaffold, messages_without_assistant,
add_generation_prompt=True)`` and the answer is the tail of the full render — the
same two calls (byte for byte) that ``grad_worker`` makes before tokenizing with
``add_special_tokens=False``, which is exactly how axolotl's ``input_output``
strategy tokenizes each segment. So the training text matches the activation
text the layer/vectors were read under. ``label=false`` on the prompt reproduces
the chat-template recipe's response-only masking (``roles_to_train:
[assistant]`` + ``train_on_inputs: false``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from valuegen.ground_truth.training import (
    SCAFFOLD_TEMPLATES,
    render_chat_template,
)


def to_segments(messages: list[dict], scaffold: str) -> dict:
    """One cs_prep_data row → an axolotl input_output segment record."""
    template = SCAFFOLD_TEMPLATES[scaffold]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            f"expected a trailing assistant turn, got roles "
            f"{[m.get('role') for m in messages]}"
        )
    prompt_msgs = messages[:-1]
    prompt = render_chat_template(template, prompt_msgs, add_generation_prompt=True)
    full = render_chat_template(template, messages, add_generation_prompt=False)
    if not full.startswith(prompt):
        # The scaffold must render the prompt as a literal prefix of the full
        # conversation, or the response span (and its mask) is wrong.
        raise ValueError(
            f"scaffold {scaffold!r} prompt is not a prefix of the full render; "
            "cannot split the response span"
        )
    answer = full[len(prompt):]
    return {
        "segments": [
            {"label": False, "text": prompt},
            {"label": True, "text": answer},
        ]
    }


def convert_file(src: Path, dst: Path, scaffold: str) -> int:
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    out = [to_segments(r["messages"], scaffold) for r in rows]
    dst.write_text("".join(json.dumps(r) + "\n" for r in out))
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scaffold", required=True, choices=sorted(SCAFFOLD_TEMPLATES))
    ap.add_argument("--sft-dir", required=True, type=Path,
                    help="cs_prep_data --out_dir (holds {value}_{pol}.jsonl)")
    ap.add_argument("--values", nargs="+", required=True)
    args = ap.parse_args()

    for value in args.values:
        for pol in ("pos", "neg"):
            src = args.sft_dir / f"{value}_{pol}.jsonl"
            dst = args.sft_dir / f"{value}_{pol}.io.jsonl"
            n = convert_file(src, dst, args.scaffold)
            print(f"{src.name} -> {dst.name}  ({n} rows, scaffold={args.scaffold})")


if __name__ == "__main__":
    main()
