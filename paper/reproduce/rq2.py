"""RQ2: does persona similarity predict multi-value robustness?

The experiment is configs/experiments/multivalue/rq3_ew64_qwen8b.yaml (`rq3_` is the project's
earlier numbering for this experiment): 64 DPO models of
Qwen3-8B (neutral SFT), each trained on a 6-value set drawn evenly over the range
of persona tightness (mean pairwise persona cosine), then evaluated for prefill
robustness. Reproduces, from data/paper/rq2:

  1. metrics.csv — coverage and tightness of every set, from the embedding stores
     (`valuegen mv metrics`);
  2. outcomes.csv, joined.csv, correlations.csv — the per-arm prefill metrics from
     every checkpoint's judged rows, and their correlations with tightness
     (`valuegen mv analyze`); joined.csv is also the RQ2 figure's input;
  3. the preregistered analysis (design fixed before launch): the primary
     one-sided Spearman test, the cell-level models, the per-bin means and the
     no-top-bin result, slopes and the pool-reweighted rho;
  4. tab:rq2_metrics (appendix G).

The release code runs on a scratch copy of the experiment tree in
paper/out/rq2_work/ (inputs symlinked, outputs real), so data/paper stays
untouched. The judged rows ship as one compact table (evals/prefill_rows.parquet,
written by `scripts/export_paper_data.py rq2`) and are expanded there into the
per-checkpoint rows.jsonl that `mv analyze` reads. Every output is compared with paper/expected/ (csv/txt/tex
byte-for-byte; metrics.csv to 1e-12). Exit status 1 on any mismatch.

    .venvs/core/bin/python paper/reproduce/rq2.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from _common import EXPECTED, OUT, PAPER_DATA, ROOT, Checker, require_paper_data, write
from valuegen.config import ClusterConfig
from valuegen.multivalue import analyze as mvanalyze
from valuegen.multivalue.cells import MIN_DEN, prefill_cell_metrics, weighted_spearman, zscore
from valuegen.multivalue.config import load_config
from valuegen.multivalue.imports import all_arms
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.metrics_stage import write_metrics

D = PAPER_DATA / "rq2"
CONFIG = ROOT / "configs/experiments/multivalue/rq3_ew64_qwen8b.yaml"
EXP_ID = "rq3_ew64_qwen8b-9b66a6dd75a9"
WORK = OUT / "rq2_work"
X, Y = "persona.tightness", "prefill.retention_norm.estimate"
# Preregistered constants (PREREG.md): pool size, set size, seed, trim, bootstrap size,
# and the design's tightness range cut into 4 equal-width bins.
N_POOL, K, SEED, TRIM, N_BOOT = 100_000, 6, 42, 0.1, 10_000
EDGES = -0.075 + (0.459 - -0.075) / 4 * np.arange(5)


# ── 1–2. mv metrics + mv analyze on a scratch tree ───────────────────────────


def expand_evals(src: Path, dst: Path) -> None:
    """The evals/ tree `mv analyze` reads, with each checkpoint's rows.jsonl
    written back out of the compact prefill_rows.parquet."""
    dst.mkdir()
    os.symlink(src / "candidates.json", dst / "candidates.json")
    rows = pd.read_parquet(src / "prefill_rows.parquet")
    rows["suite"], rows["followup_id"] = "prefill", 0
    for ckpt, df in rows.groupby("checkpoint_id", observed=True, sort=False):
        d = dst / str(ckpt) / "prefill"
        d.mkdir(parents=True)
        os.symlink(src / str(ckpt) / "prefill" / "COMPLETE.json", d / "COMPLETE.json")
        df.to_json(d / "rows.jsonl", orient="records", lines=True)


def build_work_tree() -> tuple:
    """A repo-shaped tree where the release config resolves onto data/paper/rq2."""
    if WORK.exists():
        shutil.rmtree(WORK)
    data = WORK / "data"
    exp_root = data / "multivalue" / "rq3_ew64_qwen8b"
    src_root = D / "multivalue" / "rq3_ew64_qwen8b"
    exp = exp_root / EXP_ID
    (exp_root / "metrics").mkdir(parents=True)
    exp.mkdir()
    for name in ("configs", "value_sets"):
        os.symlink(ROOT / name, WORK / name)
    os.symlink(D / "embeddings", data / "embeddings")
    os.symlink(src_root / "sets.json", exp_root / "sets.json")
    for name in ("config.yaml", "eval_inputs"):
        os.symlink(src_root / EXP_ID / name, exp / name)
    expand_evals(src_root / EXP_ID / "evals", exp / "evals")

    cfg = load_config(CONFIG, root=WORK)
    cluster = ClusterConfig(
        envs={"default": ".venvs/core"}, repo=WORK, finetune_root=WORK / "finetune",
        finetune_root_legacy=WORK / "finetune", data=data, slurm_logs=WORK / "logs",
        mail_type="NONE", mail_user="", gpu_type=None, max_concurrent_gpus=1,
        default_time="1:00:00", default_mem="4G", cpu_partition="", cpu_qos="",
        source_path=CONFIG,
    )
    layout = Layout(cfg, cluster)
    if layout.exp_id != EXP_ID:
        raise SystemExit(f"{CONFIG.name} hashes to {layout.exp_id}, expected {EXP_ID}")
    return cfg, cluster, layout


def run_mv(chk: Checker) -> pd.DataFrame:
    cfg, cluster, layout = build_work_tree()

    # metrics.csv, recomputed from the embedding stores; it must agree with the
    # shipped one (float formatting aside), which analyze then reads.
    got = write_metrics(cfg, cluster, layout)
    want = pd.read_csv(EXPECTED / "rq2" / "metrics.csv")
    got = pd.read_csv(layout.metrics_csv)
    num = want.select_dtypes("number").columns
    if list(got.columns) != list(want.columns) or len(got) != len(want) or \
            not got.drop(columns=num).equals(want.drop(columns=num)) or \
            not np.allclose(got[num], want[num], rtol=0, atol=1e-12, equal_nan=True):
        chk.failures.append("metrics.csv: recomputed values differ from expected/rq2/metrics.csv")
    chk.n += 1
    shutil.copyfile(D / "multivalue" / "rq3_ew64_qwen8b" / "metrics" / "metrics.csv",
                    layout.metrics_csv)

    mvanalyze.analyze(cfg, cluster, layout, all_arms(layout), log=lambda *a, **k: None)
    for name in ("outcomes.csv", "joined.csv", "correlations.csv"):
        chk.n += 1
        if (layout.scores_dir / name).read_bytes() != (EXPECTED / "rq2" / name).read_bytes():
            chk.failures.append(f"{name}: differs from expected/rq2/{name}")
        shutil.copyfile(layout.scores_dir / name, OUT / f"rq2_{name}")
    return pd.read_csv(layout.scores_dir / "joined.csv"), layout


# ── 3. The preregistered analysis ────────────────────────────────────────────


def one_sided(x, y) -> str:
    r = stats.spearmanr(x, y, alternative="greater")
    return f"rho = {r.statistic:+.3f}  one-sided p = {r.pvalue:.2g}  n = {len(x)}"


def pool_shares(edges: np.ndarray) -> np.ndarray:
    """Share of a 100k random k=6 pool (the `mv sets` construction) in each design bin."""
    G = pd.read_csv(D / "cosine" / "persona.csv", index_col=0).to_numpy()
    rng = np.random.default_rng(SEED)
    pool = np.array([rng.choice(len(G), K, replace=False) for _ in range(N_POOL)])
    iu = np.triu_indices(K, 1)
    t = G[pool[:, :, None], pool[:, None, :]][:, iu[0], iu[1]].mean(1)
    inside = t[(t >= edges[0]) & (t <= edges[-1])]
    return np.histogram(inside, edges)[0] / len(inside)


def check_cosines() -> None:
    """The shipped cosine tables are the constitution stores' cosines."""
    for kind, npz in (("persona", "persona/qwen3-8b-neutral"), ("sentence", "sentence")):
        c = pd.read_csv(D / "cosine" / f"{kind}.csv", index_col=0)
        z = np.load(D / "embeddings" / npz / "constitution_tenets_v3.npz")
        v = np.stack([z[k].astype(np.float64) for k in c.index])
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        if np.abs(v @ v.T - c.to_numpy()).max() > 1e-11:
            raise SystemExit(f"cosine/{kind}.csv does not match the {kind} embedding store")


def prereg(J: pd.DataFrame, exp: Path) -> tuple[str, pd.DataFrame]:
    """The analyses of the preregistration, in its order (text report, cells)."""
    lines: list[str] = []
    say = lines.append
    arms = json.loads((D / "multivalue/rq3_ew64_qwen8b/sets.json").read_text())["families"]["ew6"]["arms"]
    J = J.copy()
    J["key"] = J.arm_id.str.removesuffix("_s42")
    A = J[J.key.isin(arms)].copy()
    A["bin"] = A.key.map(lambda k: arms[k]["bin"])
    excluded = A[~A["prefill.retention_norm.complete"].astype(bool)].arm_id.tolist()
    A = A[A["prefill.retention_norm.complete"].astype(bool)]
    say(f"arms: {len(A)} of {len(arms)} with a complete prefill eval; excluded: {excluded or 'none'}")

    say("\n=== PRIMARY (H1): Spearman persona.tightness -> retention_norm, one-sided, alpha .05 ===")
    say("  " + one_sided(A[X], A[Y]))

    say("\n=== Secondary 1: (arm, value) cells, value FE, WLS by denominator, SE clustered by arm ===")
    G = pd.read_csv(D / "cosine" / "persona.csv", index_col=0)
    Gs = pd.read_csv(D / "cosine" / "sentence.csv", index_col=0)
    cells = []
    for _, a in A.iterrows():
        c = pd.DataFrame(prefill_cell_metrics(
            pd.read_json(exp / f"evals/{a.key}_s42/prefill/rows.jsonl", lines=True)))
        assert abs(c.rn_num.sum() / c.rn_den.sum() - a[Y]) < 1e-6, f"{a.arm_id}: cells do not reproduce joined.csv"
        members = arms[a.key]["values"]
        c["aff_persona"] = [np.mean([G.loc[v, u] for u in members if u != v]) for v in c.value]
        c["aff_sentence"] = [np.mean([Gs.loc[v, u] for u in members if u != v]) for v in c.value]
        assert set(c.value) == set(members) and abs(c.aff_persona.mean() - a[X]) < 1e-6, \
            f"{a.arm_id}: affinity != tightness"
        cells.append(c.assign(arm_id=a.key, bin=a.bin, **{X: a[X]}))
    cells_df = pd.concat(cells, ignore_index=True)
    cells_df.loc[cells_df.rn_den < MIN_DEN, "retention_norm"] = np.nan
    say(f"  cells: {len(cells_df)} ({cells_df.value.nunique()} values); kept (den >= {MIN_DEN:g}): "
        f"{cells_df.retention_norm.notna().sum()}; "
        "cells aggregate to joined.csv and mean affinity = persona.tightness for every arm")
    Dc = cells_df.dropna(subset=["retention_norm"]).copy()
    say(f"  variance explained by value identity alone: R2 = "
        f"{smf.ols('retention_norm ~ C(value)', Dc).fit().rsquared:.2f}")
    for c in ["aff_persona", "aff_sentence", X]:
        Dc["z_" + c.replace(".", "_")] = zscore(Dc[c].clip(*Dc[c].quantile([0.01, 0.99])))
    for name, t in [("affinity(persona)", "z_aff_persona"), ("affinity(sentence)", "z_aff_sentence"),
                    ("arm tightness(persona)", "z_persona_tightness")]:
        for scope, d in [("all bins", Dc), ("no top bin", Dc[Dc.bin < 3])]:
            fit = smf.wls(f"retention_norm ~ C(value) + {t}", d, weights=d.rn_den).fit(
                cov_type="cluster", cov_kwds={"groups": d.arm_id})
            lo, hi = fit.conf_int().loc[t]
            say(f"  {name:24s} [{scope:10s}] b = {fit.params[t]:+.4f} per SD  [{lo:+.4f}, {hi:+.4f}]  "
                f"p = {fit.pvalues[t]:.2g}  n = {len(d)}")
    rhos = Dc.groupby("value").apply(
        lambda g: stats.spearmanr(g.aff_persona, g.retention_norm)[0] if len(g) >= 4 else np.nan,
        include_groups=False).dropna()
    say(f"  within-value Spearman(affinity, retention_norm): mean rho = {rhos.mean():+.3f}, "
        f"{(rhos > 0).sum()}/{len(rhos)} positive, t-test p = {stats.ttest_1samp(rhos, 0).pvalue:.2g}, "
        f"sign p = {stats.binomtest(int((rhos > 0).sum()), len(rhos)).pvalue:.2g}")

    say("\n=== Secondary 2: same test on the other prefill metrics ===")
    for m in ["maiya", "sturgeon", "retention"]:
        say(f"  {m:10s} " + one_sided(A[X], A[f"prefill.{m}.estimate"]))

    say("\n=== Secondary 3: shape ===")
    g = A.groupby("bin")[Y].agg(["count", "mean", "sem"])
    g["ci"] = g["sem"] * stats.t.ppf(0.975, g["count"] - 1)
    rng_x = A.groupby("bin")[X].agg(["min", "max"])
    for b, r in g.iterrows():
        say(f"  bin {b} (tightness {rng_x.loc[b, 'min']:+.3f}..{rng_x.loc[b, 'max']:+.3f}): mean = {r['mean']:.3f}  "
            f"95% CI [{r['mean'] - r.ci:.3f}, {r['mean'] + r.ci:.3f}]  n = {int(r['count'])}")
    for name, m in [("without top bin", A.bin < 3), ("lower two bins", A.bin < 2), ("upper two bins", A.bin >= 2)]:
        say(f"  {name:16s} " + one_sided(A[X][m], A[Y][m]))

    say("\n=== Secondary 4: effect size ===")
    x, y = A[X].to_numpy(), A[Y].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(A), (N_BOOT, len(A)))
    boot = np.array([np.polyfit(x[i], y[i], 1)[0] for i in idx])
    say(f"  OLS slope = {np.polyfit(x, y, 1)[0]:+.3f}  bootstrap 95% CI [{np.percentile(boot, 2.5):+.3f}, "
        f"{np.percentile(boot, 97.5):+.3f}]  ({N_BOOT} arm resamples)")
    m = (A.bin < 3).to_numpy()
    bootm = np.array([np.polyfit(x[m][i], y[m][i], 1)[0] for i in rng.integers(0, m.sum(), (N_BOOT, m.sum()))])
    say(f"  OLS slope without top bin = {np.polyfit(x[m], y[m], 1)[0]:+.3f}  bootstrap 95% CI "
        f"[{np.percentile(bootm, 2.5):+.3f}, {np.percentile(bootm, 97.5):+.3f}]")
    share = pool_shares(EDGES)
    say("  pool share per bin: " + "  ".join(f"{s:.3f}" for s in share))
    bins = A.bin.to_numpy()
    w = share[bins] / np.bincount(bins)[bins]
    rw = weighted_spearman(x, y, w)
    bw = []
    for _ in range(N_BOOT):  # resample within bin: the bin sizes are fixed by design
        i = np.concatenate([rng.choice(np.flatnonzero(bins == b), (bins == b).sum()) for b in range(4)])
        bw.append(weighted_spearman(x[i], y[i], w[i]))
    say(f"  pool-reweighted Spearman rho = {rw:+.3f}  bootstrap 95% CI [{np.percentile(bw, 2.5):+.3f}, "
        f"{np.percentile(bw, 97.5):+.3f}]  (within-bin resamples; effective n = {w.sum() ** 2 / (w ** 2).sum():.1f})")

    say("\n=== Secondary 5: sentence tightness (descriptive; not spread by the design) ===")
    r = stats.spearmanr(A["sentence.tightness"], A[Y])
    say(f"  rho = {r.statistic:+.3f}  two-sided p = {r.pvalue:.2g}  "
        f"(sentence tightness sd = {A['sentence.tightness'].std():.3f})")

    say("\n=== Secondary 6: pooled strat64 + ew64 (robustness check only) ===")
    S = pd.read_csv(D / "multivalue/rq3_strat64_qwen8b/joined.csv")
    S = S[(S.arm_id != "base") & S["prefill.retention_norm.complete"].astype(bool)]
    P = pd.concat([A[[X, Y]].assign(sweep="ew64"), S[[X, Y]].assign(sweep="strat64")], ignore_index=True)
    say("  strat64 alone     " + one_sided(S[X], S[Y]))
    say("  pooled            " + one_sided(P[X], P[Y]))
    fit = smf.ols("y ~ x + C(sweep)", P.rename(columns={X: "x", Y: "y"})).fit()
    say(f"  pooled OLS slope (sweep intercepts) = {fit.params['x']:+.3f}  "
        f"[{fit.conf_int().loc['x', 0]:+.3f}, {fit.conf_int().loc['x', 1]:+.3f}]")
    inter = smf.ols("y ~ x * C(sweep)", P.rename(columns={X: "x", Y: "y"})).fit()
    say(f"  slope difference strat64 - ew64 = {inter.params['x:C(sweep)[T.strat64]']:+.3f}  "
        f"p = {inter.pvalues['x:C(sweep)[T.strat64]']:.2g}")
    lowP = P[P[X] < EDGES[3]]
    say(f"  pooled, tightness below the top-bin edge ({EDGES[3]:+.3f}): " + one_sided(lowP[X], lowP[Y]))
    return "\n".join(lines) + "\n", cells_df


# ── 4. tab:rq2_metrics ───────────────────────────────────────────────────────


def metric_table(J: pd.DataFrame) -> str:
    ours = "prefill.retention_norm.estimate"
    outcomes = [(r"\textbf{Ours}", ours),
                (r"Prefill adherence \citep{oct}", "prefill.maiya.estimate"),
                (r"Paired defend rate \citep{sturgeon2026roleplaying}", "prefill.sturgeon.estimate")]
    predictors = ["persona.tightness", "sentence.tightness"]

    def fmt_p(p):
        return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"

    d = J[J["kind"] != "base"]
    d = d[d["prefill.retention_norm.complete"] == True]  # noqa: E712
    d = d.dropna(subset=[c for _, c in outcomes] + predictors)
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lccc}", r"\toprule",
         r"Robustness metric & Spearman $\rho$ & \persona & \sentence \\",
         r"& to ours & coherence & coherence \\", r"\midrule"]
    for label, col in outcomes:
        y = d[col].astype(float).values
        to_ours = "---" if col == ours else f"${stats.spearmanr(y, d[ours].astype(float).values)[0]:.2f}$"
        cells = []
        for pc in predictors:
            rho, p = stats.spearmanr(d[pc].astype(float).values, y)
            cells.append(f"${rho:.2f}$ (${fmt_p(p)}$)")
        L.append(f"{label} & {to_ours} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          (r"\caption{Spearman $\rho$ between three different ways to operationalize "
           r"prefill robustness, as well as the \Cref{fig:prefill-results} "
           r"correlation result computed under all three robustness metrics. Our "
           r"metric is highly correlated with metrics adapted from those used in "
           r"previous work, and the coherence-robustness relationship holds across "
           r"all choices of metric.}"),
          r"\label{tab:rq2_metrics}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    require_paper_data()
    chk = Checker("RQ2")
    J, layout = run_mv(chk)
    check_cosines()

    text, cells = prereg(J, layout.exp_dir)
    write("rq2_prereg_results.txt", text)
    cells.to_csv(OUT / "rq2_cells.csv", index=False)
    for got, want in (("rq2_prereg_results.txt", "prereg_results.txt"), ("rq2_cells.csv", "cells.csv")):
        chk.n += 1
        if (OUT / got).read_bytes() != (EXPECTED / "rq2" / want).read_bytes():
            chk.failures.append(f"{got}: differs from expected/rq2/{want}")

    write("rq2_metric_table.tex", metric_table(J))
    chk.tex("rq2_metric_table.tex", "rq2_metric_table.tex")
    print(text)
    return chk.finish()


if __name__ == "__main__":
    sys.exit(main())
