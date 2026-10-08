"""Place the constitution tenets into the two external taxonomies RQ3 compares.

``vitw``: Values in the Wild (arXiv:2504.15236) built its hierarchy by
embedding value text with all-mpnet-base-v2 and clustering. Each tenet
description goes to its cosine-nearest VITW level-1 value (the 266 of
``value_sets/vitw_l1_266.json``) and inherits that value's level-2 and
level-3 categories from ``vitw_meta.json``. Deterministic, no API.

``litmus``: LitmusValues (arXiv:2505.14633, App. E step 3) classifies a value
into one of 16 classes by prompting a model with the class definitions. The
paper's instruction is reproduced verbatim; the tenet is the value. The
paper's model (claude-3-5-sonnet-20241022) is retired, so the published labels
came from the fallback, gpt-5.6-luna; the model actually used is recorded.
Completions are cached per (model, prompt) under ``--cache``.

Writes ``labels_vitw.json`` / ``labels_litmus16.json`` into ``--out`` (all 66
tenets; RQ3 uses the 49 trained ones).

    .venvs/core/bin/python scripts/taxonomy_labels.py vitw
    .venvs/core/bin/python scripts/taxonomy_labels.py litmus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MPNET = "sentence-transformers/all-mpnet-base-v2"
TENETS = ROOT / "value_sets" / "constitution_tenets_v3.json"
VITW_L1 = ROOT / "value_sets" / "vitw_l1_266.json"

# Verbatim classification instruction from LitmusValues App. E, step (3).
PAPER_INSTRUCTION = (
    "Based on the values and principle, can you classify the values? Find the most "
    "relevant class to do the classification. Here are the classification definitions:"
)
# Deviation (documented): the unit is a named value tenet, not a dilemma-action
# value label, and a single-label JSON reply makes parsing deterministic.
FORMAT = ('Classify the single value below into exactly one class. '
          'Reply ONLY with JSON: {"class": "<one class name exactly as written>"}.')


def vitw(out: Path, meta_path: Path) -> None:
    from sentence_transformers import SentenceTransformer

    tenets = json.loads(TENETS.read_text())
    l1 = json.loads(VITW_L1.read_text())
    meta = json.loads(meta_path.read_text())
    keys = [k for k in l1 if k in meta]
    enc = SentenceTransformer(MPNET)
    ten_keys = list(tenets)
    t_emb = enc.encode([tenets[k] for k in ten_keys], normalize_embeddings=True,
                       show_progress_bar=False)
    v_emb = enc.encode([l1[k] for k in keys], normalize_embeddings=True,
                       show_progress_bar=False)
    sims = t_emb @ v_emb.T
    labels = {}
    for i, tk in enumerate(ten_keys):
        j = int(np.argmax(sims[i]))
        m = meta[keys[j]]
        labels[tk] = {"l1": m["l1"], "l2": m["l2"], "l3": m["l3"],
                      "nearest_l1_key": keys[j], "cos": round(float(sims[i, j]), 4)}
    path = out / "labels_vitw.json"
    path.write_text(json.dumps({
        "_protocol": "all-mpnet-base-v2 nearest VITW-L1, inherit L2/L3 "
                     "(VITW arXiv:2504.15236 §2.3 encoder)",
        "encoder": MPNET, "labels": labels}, indent=1))
    print(f"wrote {path}; L3: {dict(Counter(v['l3'] for v in labels.values()))}")


def _call(model: str, prompt: str) -> str:
    if model.startswith("claude"):
        from anthropic import Anthropic

        r = Anthropic().messages.create(model=model, max_tokens=64,
                                        messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    from openai import OpenAI

    r = OpenAI().chat.completions.create(model=model, max_completion_tokens=512,
                                         messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def _parse(text: str, classes: dict) -> str | None:
    m = re.search(r'"class"\s*:\s*"([^"]+)"', text)
    cand = m.group(1).strip() if m else text.strip()
    low = {k.lower(): k for k in classes}
    if cand.lower() in low:
        return low[cand.lower()]
    return next((k for k in classes if k.lower() in cand.lower()), None)


def litmus(out: Path, classes_path: Path, cache: Path, model: str, fallback: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    classes = json.loads(classes_path.read_text())["classes"]
    tenets = json.loads(TENETS.read_text())
    defs = "\n".join(f"- {k}: {v}" for k, v in classes.items())
    cache.mkdir(parents=True, exist_ok=True)

    def prompt_of(tk):
        return (f"{PAPER_INSTRUCTION}\n{defs}\n\n{FORMAT}\n\n"
                f"Value name: {tk}\nValue (principle): {tenets[tk]}")

    def cache_file(m, tk):
        sha = hashlib.sha256((m + "\x00" + prompt_of(tk)).encode()).hexdigest()[:16]
        return cache / f"litmus_{sha}.json"

    # One model labels every tenet. A complete cache for the paper model, or
    # else for the fallback, is replayed without any API call; otherwise the
    # paper model is probed once (it is retired) and the fallback used.
    if all(cache_file(model, tk).exists() for tk in tenets):
        used = model
    elif all(cache_file(fallback, tk).exists() for tk in tenets):
        used = fallback
    else:
        used = model
        try:
            _call(model, "reply ok")
        except Exception as e:
            print(f"{model} unavailable ({str(e)[:60]}...) -> {fallback}")
            used = fallback
    labels = {}
    for tk in tenets:
        cf = cache_file(used, tk)
        if cf.exists():
            raw = json.loads(cf.read_text())["raw"]
        else:
            raw = _call(used, prompt_of(tk))
            cf.write_text(json.dumps({"model": used, "tenet": tk, "prompt": prompt_of(tk),
                                      "raw": raw}, indent=1))
        cls = _parse(raw, classes)
        if cls is None:
            raise SystemExit(f"unparseable class for {tk!r}: {raw!r}")
        labels[tk] = cls
    path = out / "labels_litmus16.json"
    path.write_text(json.dumps({
        "_protocol": "LitmusValues arXiv:2505.14633 App.E step 3 classification prompt",
        "model_requested": model, "model_used": used,
        "paper_instruction_sha12": hashlib.sha256(PAPER_INSTRUCTION.encode()).hexdigest()[:12],
        "labels": labels}, indent=1))
    print(f"wrote {path}; {len(set(labels.values()))} distinct classes, model {used}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="taxonomy", required=True)
    v = sub.add_parser("vitw")
    v.add_argument("--meta", type=Path, default=ROOT / "data/paper/rq3/vitw_meta.json",
                   help="VITW L1 -> {l1, l2, l3} hierarchy")
    lt = sub.add_parser("litmus")
    lt.add_argument("--classes", type=Path,
                    default=ROOT / "data/paper/rq3/litmus_value_classes.json",
                    help="the 16 LitmusValues class definitions (paper Table 2)")
    lt.add_argument("--cache", type=Path, default=ROOT / "data/taxonomy_labels/cache")
    lt.add_argument("--model", default="claude-3-5-sonnet-20241022")
    lt.add_argument("--fallback", default="gpt-5.6-luna")
    for p in (v, lt):
        p.add_argument("--out", type=Path, default=ROOT / "data/taxonomy_labels")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.taxonomy == "vitw":
        vitw(args.out, args.meta)
    else:
        litmus(args.out, args.classes, args.cache, args.model, args.fallback)


if __name__ == "__main__":
    main()
