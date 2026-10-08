"""Appendix C: ConflictScope scenario generation and the steerability metric.

Reproduces:
  1. the scenario-generation comparison (agreement_polarization.csv, the
     figure's input): for each generating model, how often ten evaluated models
     agree on the binary choice (lower = more contested scenarios), the Likert
     polarization, the Likert difference rate and the mean Likert difference,
     from the evaluated models' judged answers in data/paper/appc/generation/
     (bootstrap CIs, 2000 draws, one seed-0 stream across generators);
  2. tab:metric-comparison: on the Qwen3-8B DPO 49×66 matrix, our ds metric
     (Likert and binary choice) against the ConflictScope metric and the raw
     difference, from the per-scenario evals in data/paper/evals/, with base
     and steered rates paired on the scenarios valid for both.

The generation figure's input in paper/figures/appC_conflictscope_metrics/inputs/
is checked byte-for-byte; the table and its numbers against paper/expected/appc/.

    .venvs/core/bin/python paper/reproduce/appc.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from _common import (EXPECTED, OUT, PAPER_DATA, ROOT, TARGETS, Checker, load_json,
                     require_paper_data, write)
from valuegen.analysis.steerability import load_values

FIG = ROOT / "paper/figures/appC_conflictscope_metrics/inputs"
GEN = PAPER_DATA / "appc" / "generation"


# ── 1. Scenario generation comparison ────────────────────────────────────────

# (directory, label); only each model's minimal / no-think generation run.
GENERATORS = [
    ("gpt-4.1", "gpt-4.1"), ("gpt-5", "gpt-5"), ("gpt-5.2", "gpt-5.2"), ("gpt-5.5", "gpt-5.5"),
    ("Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B"), ("Llama-3.3-70B-Instruct", "Llama-3.3-70B"),
    ("0611_claude35_pp", "claude-3.5-sonnet"), ("0611_qwen36_pp", "Qwen3.6-27B"),
]
N_BOOT_GEN = 2000


def observed_agreement(data: pd.DataFrame) -> float:
    """Pairwise fraction of evaluated-model pairs making the same A/B choice."""
    data["binary_choice"] = (data["choice"] != "A").astype(int)
    pivot = data.pivot_table(index="scenario_id", columns="model",
                             values="binary_choice", aggfunc="first").dropna()
    if len(pivot) == 0:
        return -1
    m = pivot.values
    n_s, n_m = m.shape
    total = n_s * (n_m * (n_m - 1)) // 2
    agree = sum(np.sum(m[:, i] == m[:, j]) for i in range(n_m) for j in range(i + 1, n_m))
    return agree / total if total > 0 else -1


def binomial_ci(successes, total):
    if total == 0:
        return 0, 0
    p = successes / total
    se = np.sqrt(p * (1 - p) / total)
    return max(0, p - 1.96 * se), min(1, p + 1.96 * se)


def load_generator(gen: str) -> pd.DataFrame:
    frames = []
    for f in sorted((GEN / gen / "model_evals").glob("*.csv")):
        df = pd.read_csv(f)
        for col in ("likert", "likert_a", "likert_b"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def generation_comparison() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for gen, label in GENERATORS:
        data = load_generator(gen)
        agree = observed_agreement(data.copy())
        n_models = data["model"].nunique()
        pivot = (data.assign(bc=(data["choice"] != "A").astype(int))
                 .pivot_table(index="scenario_id", columns="model", values="bc", aggfunc="first").dropna())
        n_scen = len(pivot)
        total = n_scen * n_models * (n_models - 1) // 2
        a_lo, a_hi = binomial_ci(int(round(agree * total)), total)

        lk = data["likert"].dropna().to_numpy()
        pol = np.abs(lk).mean()
        boot = np.array([np.abs(rng.choice(lk, size=len(lk), replace=True)).mean() for _ in range(N_BOOT_GEN)])
        p_lo, p_hi = np.percentile(boot, [2.5, 97.5])

        pair = data.dropna(subset=["likert_a", "likert_b"])
        n_pair = len(pair)
        diff_n = int((pair["likert_a"] != pair["likert_b"]).sum())
        ldr = diff_n / n_pair if n_pair else 0.0
        ldr_lo, ldr_hi = binomial_ci(diff_n, n_pair)

        adiff = (pair["likert_a"] - pair["likert_b"]).abs().to_numpy()
        ald = adiff.mean() if len(adiff) else 0.0
        bootd = np.array([rng.choice(adiff, size=len(adiff), replace=True).mean() for _ in range(N_BOOT_GEN)])
        ald_lo, ald_hi = np.percentile(bootd, [2.5, 97.5])

        rows.append(dict(generating_model=label, agreement=agree, agree_lo=a_lo, agree_hi=a_hi,
                         polarization=pol, pol_lo=p_lo, pol_hi=p_hi,
                         likert_diff_rate=ldr, ldr_lo=ldr_lo, ldr_hi=ldr_hi,
                         avg_likert_diff=ald, ald_lo=ald_lo, ald_hi=ald_hi,
                         n_scenarios=n_scen, n_models=n_models, n_likert=len(lk)))
    return pd.DataFrame(rows)


# ── 2. Steerability metric comparison ────────────────────────────────────────

# (row label, metric, granularity); ds/likert is ours, the reference row.
METRIC_ROWS = [
    (r"\textbf{Ours (Likert)}", "ds", "likert"),
    (r"Ours (binary)", "ds", "binary"),
    (r"ConflictScope \citep{liu2026generative}", "cs", "likert"),
    (r"Raw difference", "diff", "likert"),
]
REF = ("ds", "likert")


def aligned_scores(sub: pd.DataFrame, value: str) -> dict[str, np.ndarray]:
    """Per-scenario adherence to ``value`` in [0, 1] (NaN = invalid), as float32
    like the cache the published table was computed from."""
    is_v1 = (sub["value1"] == value).values
    lik = sub["likert"].to_numpy(dtype=float)
    likert = np.where(is_v1, (-lik + 1) / 2, (lik + 1) / 2)
    ch = sub["choice"].astype(str).values
    aligned = np.where(is_v1, "A", "B")
    binary = np.where(np.isin(ch, ["A", "B"]), (ch == aligned).astype(float), np.nan)
    return {"likert": likert.astype(np.float32), "binary": binary.astype(np.float32)}


def paired_rate_matrices(rows, cols):
    """{granularity: (F, B)}: steered rate and paired base rate per cell, over
    the scenarios valid for both the steered model and the base."""
    gt_id = TARGETS["qwen_dpo"].relative_to(ROOT / "data/gt").parts[0]
    ev = pd.read_parquet(PAPER_DATA / "evals" / f"{gt_id}.parquet")
    scen = pd.read_parquet(PAPER_DATA / "evals" / "scenarios.parquet")
    ev = ev.join(scen[["value1", "value2"]], on="scenario")
    by_model = {m: g.set_index("scenario") for m, g in ev.groupby("model", observed=True, sort=False)}
    base = by_model["base"]
    out = {g: (np.full((len(rows), len(cols)), np.nan), np.full((len(rows), len(cols)), np.nan))
           for g in ("likert", "binary")}
    for j, cv in enumerate(cols):
        sub_b = base[(base.value1 == cv) | (base.value2 == cv)]
        sids = sub_b.index
        sb = aligned_scores(sub_b, cv)
        st = {g: np.stack([aligned_scores(by_model[r].loc[sids], cv)[g] for r in rows])
              for g in ("likert", "binary")}
        for g in ("likert", "binary"):
            ft, b = st[g], sb[g]
            valid = ~np.isnan(ft) & ~np.isnan(b)[None]
            w = valid * np.ones(len(sids))[None]
            n = w.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.where(w, ft, 0).sum(1) / n
                bb = (w * np.nan_to_num(b)[None]).sum(1) / n
            f[n < 2] = np.nan
            bb[n < 2] = np.nan
            out[g][0][:, j], out[g][1][:, j] = f, bb
    return out


def metric(F, B, name):
    with np.errstate(invalid="ignore", divide="ignore"):
        d = F - B
        cs = d / (1 - B)
        return {"diff": d, "cs": cs, "ds": np.where(d >= 0, cs, d / B)}[name]


def metric_comparison() -> dict:
    rows, cols = load_values(f"{TARGETS['qwen_dpo']}_normalized_values.json")
    diag_idx = np.array([cols.index(r) for r in rows])
    off = np.array([[c != r for c in cols] for r in rows], dtype=bool)
    fb = paired_rate_matrices(rows, cols)
    mats = {}
    for _, m, g in METRIC_ROWS:
        M = metric(*fb[g], m)
        offv = M[off]
        mats[(m, g)] = dict(M_off=offv, diag_mean=float(np.nanmean(M[np.arange(len(M)), diag_idx])),
                            offdiag_abs_mean=float(np.nanmean(np.abs(offv))),
                            min=float(np.nanmin(offv[np.isfinite(offv)])))
    ref = mats[REF]["M_off"]
    out = {}
    for _, m, g in METRIC_ROWS:
        p = mats[(m, g)]
        ok = ~(np.isnan(p["M_off"]) | np.isnan(ref))
        rho = None if (m, g) == REF else float(stats.spearmanr(p["M_off"][ok], ref[ok])[0])
        out[f"{m}_{g}"] = {"rho_to_ours": rho, "diag_mean": p["diag_mean"],
                           "offdiag_abs_mean": p["offdiag_abs_mean"], "min": p["min"]}
    return out


def metric_tex(res: dict) -> str:
    lines = []
    for label, m, g in METRIC_ROWS:
        p = res[f"{m}_{g}"]
        rho = "---" if p["rho_to_ours"] is None else f"${p['rho_to_ours']:.2f}$"
        lines.append(f"{label} & {rho} & ${p['diag_mean']:.2f}$ & "
                     f"${p['offdiag_abs_mean']:.2f}$ & ${p['min']:.2f}$ \\\\")
    return r"""\begin{table}[t]
\centering
\begin{tabular}{lcccc}
\toprule
Metric & Spearman $\rho$ & Avg. & Avg. & Min. \\
 & to ours & diagonal & $|$off-diag.$|$ & cell \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{Comparison of different steerability metrics on the Qwen-3-8B DPO
generalization matrix. All metrics induce highly similar matrices (Spearman
$\rho \ge 0.96$). However, the ConflictScope metric's mean off-diagonal
magnitude is inflated by many strongly negative cells, which skews the matrix.
Our piecewise metric solves this while otherwise reporting highly similar
generalization to other metrics.}
\label{tab:metric-comparison}
\end{table}
""" % "\n".join(lines)


def main() -> int:
    require_paper_data()
    chk = Checker("Appendix C")

    gen = generation_comparison()
    gen.to_csv(OUT / "appc_agreement_polarization.csv", index=False)
    chk.n += 1
    if (OUT / "appc_agreement_polarization.csv").read_bytes() != (FIG / "agreement_polarization.csv").read_bytes():
        chk.failures.append("appc_agreement_polarization.csv: differs from the figure's "
                            "inputs/agreement_polarization.csv")

    res = metric_comparison()
    write("appc_metric_comparison.json", json.dumps(res, indent=1) + "\n")
    chk.value("metric_comparison", res, load_json(EXPECTED / "appc" / "metric_comparison.json"))
    write("appc_metric_comparison_table.tex", metric_tex(res))
    chk.tex("appc_metric_comparison_table.tex", "appc/metric_comparison_table.tex")
    return chk.finish()


if __name__ == "__main__":
    sys.exit(main())
