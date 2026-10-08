"""Derived, generation-free cleaning of one immutable pair artifact.

Base models generating under a few-shot scaffold (``urial0``) sometimes
restart the scaffold mid-answer — emit a fresh ``# Answer:`` / ``# Query:``
block, invent a follow-up question, or fall into a pretraining reasoning-trace
register — and keep going for hundreds of words. Persona vectors average
activations over the response, so a long tail dilutes but does not dominate;
grad_proj's DPO-at-init loss is *summed* over response tokens, so the tail
carries proportionally more of the value vector. This adapter applies one
fixed, versioned cleaning protocol to every answer and drops pairs whose
either side collapses, so the predictors see the answer the base model wrote
*before* it wandered off. Judge scores were assigned to the uncleaned text at
the source's build time and are not recomputed; the effective-row filter
already ran there.

Protocol ``scaffold_restart_v1``:
  1. strip a duplicated leading ``# Answer:`` header (and a bare opening
     code fence directly after it);
  2. cut at the first restart marker found *after* the first character
     (``# Query:`` / ``# Answer:`` headings, a ``Question:``/``Answer:`` line,
     chat-template control tokens, or a reasoning-trace opener such as
     ``Okay, let's tackle``);
  3. cap at ``max_words`` whitespace tokens;
  4. drop the pair when either side has fewer than ``min_words`` left.
"""

from __future__ import annotations

import re

import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D

PROTOCOLS = ("scaffold_restart_v1",)

_LEADING_HEADER = re.compile(r"\A\s*# Answer:\s*\n(?:```[^\n]*\n)?")
_RESTART = re.compile(
    r"\n\s*(?:"
    r"# Query:|# Answer:|"
    r"Question:|Answer:|"
    r"<\|im_start\|>|<\|im_end\|>|<\|user\|>|<\|assistant\|>|<\|system\|>|"
    r"Okay, let'?s tackle|Okay, let me|Okay, so the user"
    r")"
)


def clean_answer(text: str, max_words: int, protocol: str = "scaffold_restart_v1") -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported clean protocol: {protocol!r}")
    if not isinstance(text, str):
        return ""
    text = _LEADING_HEADER.sub("", text, count=1)
    hit = _RESTART.search(text, 1)
    if hit:
        text = text[: hit.start()]
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.rstrip()


def _source(cfg: dict, cluster: ClusterConfig) -> D.Artifact:
    ref = cfg["source_artifact"]
    source = D.find_artifact_by_id(cluster, ref)
    if source is None:
        raise FileNotFoundError(f"pair_clean source artifact not found: {ref}")
    if not source.is_complete():
        raise RuntimeError(f"pair_clean source is incomplete: {source.root}")
    if source.artifact_type != "pairs":
        raise ValueError(
            f"pair_clean source {source.artifact_id} is "
            f"{source.artifact_type!r}, not 'pairs'"
        )
    return source


def clean_pairs(
    pos: pd.DataFrame, neg: pd.DataFrame, max_words: int, min_words: int, protocol: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if len(pos) != len(neg):
        raise ValueError(f"positional sides differ ({len(pos)} vs {len(neg)})")
    pos, neg = pos.copy(), neg.copy()
    stats = {"n_source": int(len(pos)), "n_cut": 0, "n_capped": 0, "n_dropped": 0}
    for frame in (pos, neg):
        raw = frame["answer"].fillna("").astype(str)
        cleaned = raw.map(lambda t: clean_answer(t, max_words, protocol))
        headless = raw.map(lambda t: _LEADING_HEADER.sub("", t, count=1))
        stats["n_cut"] += int(headless.map(lambda t: _RESTART.search(t, 1) is not None).sum())
        stats["n_capped"] += int((cleaned.str.split().str.len() >= max_words).sum())
        frame["answer_source"] = raw
        frame["answer"] = cleaned
    keep = (
        (pos["answer"].str.split().str.len() >= min_words)
        & (neg["answer"].str.split().str.len() >= min_words)
    )
    stats["n_dropped"] = int((~keep).sum())
    return pos[keep].reset_index(drop=True), neg[keep].reset_index(drop=True), stats


def build(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    only_values: list[str] | None = None,
) -> None:
    max_words, min_words = int(cfg["max_words"]), int(cfg["min_words"])
    protocol = cfg["clean_protocol"]
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported clean protocol: {protocol!r}")
    if max_words < min_words or min_words < 1:
        raise ValueError("pair_clean needs 1 <= min_words <= max_words")
    source = _source(cfg, cluster)
    values = only_values or artifact.pending_values()
    absent = [v for v in values if v not in source.values]
    if absent:
        raise ValueError(f"values {absent} are absent from source {source.artifact_id}")
    for value in values:
        if artifact.value_done(value):
            continue
        pos, neg = D.load_pairs(source, value)
        pos, neg, stats = clean_pairs(pos, neg, max_words, min_words, protocol)
        if len(pos) == 0:
            raise RuntimeError(f"{value}: every pair dropped by pair_clean")
        for frame in (pos, neg):
            frame["source_artifact"] = source.artifact_id
        D.write_pairs(artifact, value, pos, neg)
        print(
            f"  {value}: kept {len(pos)}/{stats['n_source']} pairs "
            f"(cut {stats['n_cut']} sides at a restart marker, capped {stats['n_capped']} "
            f"at {max_words} words, dropped {stats['n_dropped']} under {min_words} words)"
        )
