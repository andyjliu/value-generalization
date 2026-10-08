"""Behavior-Embd: the persona predictor's behavior, read as text.

Persona takes the mean *activation* difference between a value's pos- and
neg-trait responses. Behavior-Embd keeps exactly those responses (the
``answer`` column of a default_llm pairs artifact) but embeds them with a
sentence encoder instead:

    vec[v] = mean_i enc(pos_answer_i) - mean_i enc(neg_answer_i)
    grid   = cosine(vec[a], vec[b])

so it separates "the behavior carries the signal" from "the model's
activations carry it". No GPU training, no generation, no API calls.

Writes ``<out>.npy`` and ``<out>_values.json`` (sorted value order). Name
the pairs by value set and pairs model (the one matching default_llm
artifact is used, and the grid lands in the store as
``data/similarity/sentemb_behavior/<artifact>/sentemb_behavior_respavg_diff``),
or pass ``--pairs``/``--out`` explicitly.

    .venvs/core/bin/python scripts/sentemb_behavior.py \\
        --value-set constitution_tenets_v3 --pairs-model neutral-sft-v3-qwen3-8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENCODER = "all-mpnet-base-v2"
ROOT = Path(__file__).resolve().parents[1]


def build_grid(pairs: Path, encoder: str = ENCODER) -> tuple[np.ndarray, list[str]]:
    from sentence_transformers import SentenceTransformer

    values = sorted(p.name[: -len("_pos.csv")] for p in pairs.glob("*_pos.csv"))
    if not values:
        raise SystemExit(f"no *_pos.csv in {pairs}")
    model = SentenceTransformer(encoder)
    vecs = []
    for v in values:
        pos = pd.read_csv(pairs / f"{v}_pos.csv")["answer"].astype(str).tolist()
        neg = pd.read_csv(pairs / f"{v}_neg.csv")["answer"].astype(str).tolist()
        ep = model.encode(pos, batch_size=64, show_progress_bar=False)
        en = model.encode(neg, batch_size=64, show_progress_bar=False)
        vecs.append(ep.mean(0) - en.mean(0))
    v = np.vstack(vecs)
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return vn @ vn.T, values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--value-set", help="with --pairs-model: find the pairs artifact")
    ap.add_argument("--pairs-model", help="short name of the model that wrote the pairs")
    ap.add_argument("--pairs", type=Path, help="pairs artifact directory")
    ap.add_argument("--out", type=Path, help="output stem (no .npy)")
    ap.add_argument("--encoder", default=ENCODER)
    args = ap.parse_args()
    if args.pairs is None:
        if not (args.value_set and args.pairs_model):
            ap.error("pass --pairs, or --value-set and --pairs-model")
        found = sorted((ROOT / "data/elicitation/pairs/default_llm" / args.value_set)
                       .glob(f"{args.pairs_model}-*"))
        if len(found) != 1:
            raise SystemExit(f"expected one {args.value_set} artifact from "
                             f"{args.pairs_model}, found {[p.name for p in found]}")
        args.pairs = found[0]
    if args.out is None:
        args.out = (ROOT / "data/similarity/sentemb_behavior" / args.pairs.name
                    / "sentemb_behavior_respavg_diff")
    grid, values = build_grid(args.pairs, args.encoder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{args.out}.npy", grid)
    Path(f"{args.out}_values.json").write_text(json.dumps(values, indent=2) + "\n")
    print(f"wrote {args.out}.npy ({len(values)} values)")


if __name__ == "__main__":
    main()
