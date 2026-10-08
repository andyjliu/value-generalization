"""Appendix F: does the elicitation data matter?

Two figures, each scoring Persona and Gradient grids against the ds-metric GT
(off-diagonal Spearman rho, row-bootstrap 95% CIs, 2000 resamples, seed 0;
the json also carries paired cell-bootstrap deltas and the symmetric ceiling):

  * dpo_pairs_bars (DPO arms): generic elicitation data (the RQ1 grids:
    Gemini questions, the neutral-SFT model's +/- steered answers) vs the DPO
    training pairs themselves (data-proximity tier t1b: the 4,000-row DPO set,
    chosen = pos, rejected = neg). t1b has vectors only for the 49 trained
    tenets, so both are scored on the common 49x49 frame;
  * sft_base_bars (SFT arms): who reads and who writes the elicitation data.
    neutral = neutral-SFT model on its own answers (the DPO-arm RQ1 grids);
    base_reads_neutral = the pretrained base (urial0 scaffold) reading the same
    neutral-SFT answers; base_reads_base = the base reading answers it wrote
    itself (the RQ1 SFT-arm grids). Scored on RQ1's 49x66 frame.

Grids come from data/paper/rq1/grids (generic, neutral, base_reads_base) and
data/paper/appf/grids (dpo_pairs, base_reads_neutral). Writes
paper/out/appf_{dpo_pairs,sft_base}_bars.json and compares each byte-for-byte
with the figure inputs in paper/figures/appF_elicitation_data/inputs/.
Exit status 1 on any mismatch.

    .venvs/core/bin/python paper/reproduce/appf.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from _common import OUT, PAPER_DATA, ROOT, TARGETS, Checker, load_json, require_paper_data, write
from valuegen.analysis import correlate as C
from valuegen.analysis.steerability import load_target, load_values, offdiag_rho, symmetric_ceiling

RQ1 = PAPER_DATA / "rq1" / "grids"
APPF = PAPER_DATA / "appf" / "grids"
FIG_INPUTS = ROOT / "paper/figures/appF_elicitation_data/inputs"
NBOOT = 2000
PREDS = ["persona", "grad_proj"]

# The grids' original locations (recorded in sft_base_bars.json as provenance).
_SIM = "data/similarity"
_QN, _ON = "neutral-sft-v3-qwen3-8b-5a2708f91afb", "neutral-sft-v3-olmo3-7b-9e781b0f5a83"
_QB, _OB = "Qwen3-8B-Base-d16065ac5dae", "Olmo-3-1025-7B-f22fe71e8417"
SFT_SOURCE_PATHS = {
    "qwen_sft": {
        ("persona", "neutral"): f"{_SIM}/persona/default_llm/neutral-sft-v3-qwen3-8b/{_QN}/persona_response_avg_diff_L18.npy",
        ("grad_proj", "neutral"): f"{_SIM}/grad_proj/default_llm/neutral-sft-v3-qwen3-8b/{_QN}/gradproj_dpoinit_respavg_L18.npy",
        ("persona", "base_reads_neutral"): f"{_SIM}/persona/default_llm/Qwen3-8B-Base/{_QN}/persona_response_avg_diff_urial0_L18.npy",
        ("grad_proj", "base_reads_neutral"): f"{_SIM}/grad_proj/default_llm/Qwen3-8B-Base/{_QN}/gradproj_dpoinit_respavg_urial0_L18.npy",
        ("persona", "base_reads_base"): f"{_SIM}/persona/default_llm/Qwen3-8B-Base/{_QB}/persona_response_avg_diff_urial0_L18.npy",
        ("grad_proj", "base_reads_base"): f"{_SIM}/grad_proj/default_llm/Qwen3-8B-Base/{_QB}/gradproj_dpoinit_respavg_urial0_L18.npy",
    },
    "olmo_sft": {
        ("persona", "neutral"): f"{_SIM}/persona/default_llm/neutral-sft-v3-olmo3-7b/{_ON}/persona_response_avg_diff_L16.npy",
        ("grad_proj", "neutral"): f"{_SIM}/grad_proj/default_llm/neutral-sft-v3-olmo3-7b/{_ON}/gradproj_dpoinit_respavg_native_L16.npy",
        ("persona", "base_reads_neutral"): f"{_SIM}/persona/default_llm/Olmo-3-1025-7B/{_ON}/persona_response_avg_diff_urial0_L16.npy",
        ("grad_proj", "base_reads_neutral"): f"{_SIM}/grad_proj/default_llm/Olmo-3-1025-7B/{_ON}/gradproj_dpoinit_respavg_urial0_L16.npy",
        ("persona", "base_reads_base"): f"{_SIM}/persona/default_llm/Olmo-3-1025-7B/{_OB}/persona_response_avg_diff_urial0_L16.npy",
        ("grad_proj", "base_reads_base"): f"{_SIM}/grad_proj/default_llm/Olmo-3-1025-7B/{_OB}/gradproj_dpoinit_respavg_urial0_L16.npy",
    },
}


def load_grid(directory, name):
    m = np.load(directory / f"{name}.npy")
    rows, cols = load_values(directory / f"{name}_values.json")
    return m, rows, cols


def grid_for(arm: str, pred: str, src: str):
    """(matrix, rows, cols) of one bar's predictor grid."""
    family = arm.split("_")[0]
    if src in ("generic", "neutral"):          # neutral-SFT model, default_llm data
        return load_grid(RQ1 / f"{family}_dpo", pred)
    if src == "base_reads_base":               # base model, self-generated urial0 data
        return load_grid(RQ1 / f"{family}_sft", pred)
    return load_grid(APPF, f"{arm}_{pred}_{src}")


def row_boot(T, P, rows, cols, rng):
    out = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(rows), len(rows))
        b = offdiag_rho(T[idx], P[idx], [rows[i] for i in idx], cols)
        if np.isfinite(b):
            out.append(b)
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def dpo_pairs_bars() -> dict:
    arms, tiers = ["olmo_dpo", "qwen_dpo"], ["generic", "dpo_pairs"]
    res = {"frame": {}, "rho": {}, "paired": {}, "ceiling": {}}
    for arm in arms:
        T, tr, tc = load_target(TARGETS[arm], "ds")
        loaded = {(p, t): grid_for(arm, p, t) for p in PREDS for t in tiers}
        # common support: target rows/cols present in every grid
        rows = [v for v in tr if all(v in g[1] for g in loaded.values())]
        cols = [v for v in tc if v in rows]
        Tt = C.reindex(T, tr, tc, rows, cols)
        mask = C.offdiag_mask(rows, cols)
        res["frame"][arm] = {"rows": len(rows), "cols": len(cols), "n_cells": int(mask.sum())}
        res["ceiling"][arm] = float(symmetric_ceiling(Tt, rows, cols))
        P = {k: C.reindex(m, r, c, rows, cols) for k, (m, r, c) in loaded.items()}
        rng = np.random.default_rng(0)
        for pred in PREDS:
            for tier in tiers:
                rho = float(offdiag_rho(Tt, P[(pred, tier)], rows, cols))
                lo, hi = row_boot(Tt, P[(pred, tier)], rows, cols, rng)
                res["rho"][f"{arm}|{pred}|{tier}"] = {"rho": rho, "lo": lo, "hi": hi}
            d = C.paired_diff(P[(pred, "dpo_pairs")], P[(pred, "generic")], Tt, mask, n_boot=NBOOT)
            res["paired"][f"{arm}|{pred}"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                              for k, v in d.items()}
    return res


def sft_base_bars() -> dict:
    arms = ["olmo_sft", "qwen_sft"]
    srcs = ["neutral", "base_reads_neutral", "base_reads_base"]
    # paired steps: (a, b, label) -> diff = rho_a - rho_b
    steps = [("base_reads_neutral", "neutral", "reader"),
             ("base_reads_base", "base_reads_neutral", "author"),
             ("base_reads_base", "neutral", "total")]
    _, ROWS, COLS = load_target(TARGETS["qwen_dpo"], "ds")   # the RQ1 49x66 frame
    assert len(ROWS) == 49 and len(COLS) == 66
    mask = C.offdiag_mask(ROWS, COLS)
    res = {"frame": {"rows": len(ROWS), "cols": len(COLS), "n_cells": int(mask.sum())},
           "rho": {}, "paired": {}, "ceiling": {}, "paths": {}}
    for arm in arms:
        m, r, c = load_target(TARGETS[arm], "ds")
        T = C.reindex(m, r, c, ROWS, COLS)
        res["ceiling"][arm] = float(symmetric_ceiling(T, ROWS, COLS))
        P = {}
        for pred, src in SFT_SOURCE_PATHS[arm]:     # the original script's order (json key order)
            pm, pr, pc = grid_for(arm, pred, src)
            assert all(v in pr for v in COLS), f"{arm}/{pred}/{src} missing tenets"
            P[(pred, src)] = C.reindex(pm, pr, pc, ROWS, COLS)
            res["paths"][f"{arm}|{pred}|{src}"] = SFT_SOURCE_PATHS[arm][(pred, src)]
        rng = np.random.default_rng(0)
        for pred in PREDS:
            for src in srcs:
                rho = float(offdiag_rho(T, P[(pred, src)], ROWS, COLS))
                lo, hi = row_boot(T, P[(pred, src)], ROWS, COLS, rng)
                res["rho"][f"{arm}|{pred}|{src}"] = {"rho": rho, "lo": lo, "hi": hi}
            for a, b, lab in steps:
                d = C.paired_diff(P[(pred, a)], P[(pred, b)], T, mask, n_boot=NBOOT)
                res["paired"][f"{arm}|{pred}|{lab}"] = {
                    "a": a, "b": b,
                    **{k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating))}}
    return res


def main() -> int:
    require_paper_data()
    chk = Checker("Appendix F")
    for name, fn in (("dpo_pairs_bars.json", dpo_pairs_bars), ("sft_base_bars.json", sft_base_bars)):
        res = fn()
        out = f"appf_{name}"
        # Same serialization as the figure inputs (json.dump(indent=1), no newline).
        write(out, json.dumps(res, indent=1))
        chk.n += 1
        if (OUT / out).read_bytes() != (FIG_INPUTS / name).read_bytes():
            chk.failures.append(f"{out}: differs from {(FIG_INPUTS / name).relative_to(ROOT)}")
            chk.value(name, res, load_json(FIG_INPUTS / name))   # say where
        for k, v in res["rho"].items():
            print(f"  {k:36s} rho {v['rho']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
    return chk.finish()


if __name__ == "__main__":
    sys.exit(main())
