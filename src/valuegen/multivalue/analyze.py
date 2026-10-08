"""Analysis for multivalue experiments.

The command builds four artifacts:

* ``outcomes.csv`` -- long per-checkpoint suite estimates and paired deltas;
* ``joined.csv`` -- one wide row per arm (training seeds are averaged);
* ``correlations.csv`` -- bivariate correlations and, when applicable,
  covariate-adjusted OLS fits;
* figures and ``REPORT.md``.

Incomplete suite runs remain visible in the first two artifacts and in the
report, but are deliberately excluded from correlations.  An arm with more
than one configured training seed is complete only when every seed is
present and protocol-complete for the outcome in question.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from valuegen.analysis.plots import save, scatter_fit
from valuegen.config import ClusterConfig
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import eval_stage as ES
from valuegen.multivalue import outcomes as O
from valuegen.multivalue._hashing import sha256_json
from valuegen.multivalue.config import MultivalueConfig
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.sets import Arm

AnalyzeError = O.AnalyzeError


CORRELATION_COLUMNS = (
    "store", "metric", "outcome", "outcome_scale", "method", "n",
    "pearson_r", "spearman_rho", "perm_p", "bootstrap_ci_low", "bootstrap_ci_high",
    "coefficient", "coefficient_se", "coefficient_p", "coefficient_ci_low", "coefficient_ci_high",
    "partial_r", "covariates",
)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Publish a CSV without leaving a half-written final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def _finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _corr(x: np.ndarray, y: np.ndarray, kind: str = "pearson") -> float:
    """A warning-free correlation, NaN for fewer than 3 or constant data."""
    x, y = _finite_pair(x, y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    if kind == "spearman":
        from scipy.stats import rankdata

        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def correlate(x, y, n_perm: int, seed: int, n_boot: int = 2000) -> dict[str, float | int]:
    """Pearson/Spearman, two-sided label-permutation p, and Pearson bootstrap CI.

    The ``+1`` correction makes the Monte-Carlo p-value nonzero.  Resampling
    is over arms, which is the observation unit of this analysis.
    """
    x, y = _finite_pair(x, y)
    n = len(x)
    r, rho = _corr(x, y), _corr(x, y, "spearman")
    result: dict[str, float | int] = {
        "n": n, "pearson_r": r, "spearman_rho": rho,
        "perm_p": float("nan"), "bootstrap_ci_low": float("nan"),
        "bootstrap_ci_high": float("nan"),
    }
    if not np.isfinite(r):
        return result
    rng = np.random.default_rng(seed)
    ge = 0
    # Vectorized chunks keep the declared 20k permutations cheap even for
    # dozens of section outcomes, without allocating the full cube at once.
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    chunk = 1000
    for start in range(0, int(n_perm), chunk):
        size = min(chunk, int(n_perm) - start)
        order = np.argsort(rng.random((size, n)), axis=1)
        rp = yc[order] @ xc / denom
        ge += int((np.abs(rp) >= abs(r) - 1e-12).sum())
    result["perm_p"] = (ge + 1) / (int(n_perm) + 1)
    boots: list[float] = []
    for start in range(0, int(n_boot), chunk):
        size = min(chunk, int(n_boot) - start)
        ix = rng.integers(0, n, size=(size, n))
        xb, yb = x[ix], y[ix]
        xb = xb - xb.mean(axis=1, keepdims=True)
        yb = yb - yb.mean(axis=1, keepdims=True)
        den = np.linalg.norm(xb, axis=1) * np.linalg.norm(yb, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rb = (xb * yb).sum(axis=1) / den
        boots.extend(rb[np.isfinite(rb)].tolist())
    if boots:
        result["bootstrap_ci_low"], result["bootstrap_ci_high"] = map(
            float, np.quantile(boots, [0.025, 0.975])
        )
    return result


def _varying_covariates(frame: pd.DataFrame, names: Sequence[str]) -> list[str]:
    out = []
    for name in names:
        if name not in frame:
            continue
        x = pd.to_numeric(frame[name], errors="coerce")
        if x.notna().all() and x.nunique(dropna=True) > 1:
            out.append(name)
    return out


def ols_adjusted(frame: pd.DataFrame, x_col: str, y_col: str,
                 covariates: Sequence[str]) -> dict[str, Any] | None:
    """OLS coefficient of ``x_col`` and partial r after numeric covariates.

    Redundant covariates are skipped in declaration order.  ``None`` means
    that no covariate varied in this analysis slice, so no OLS row is due.
    Underpowered or rank-deficient fits still return a descriptive row with
    NaN inference fields rather than crashing the whole report.
    """
    cols = [x_col, y_col, *covariates]
    d = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    covs = _varying_covariates(d, covariates)
    if not covs:
        return None
    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    if len(d) < 3 or np.ptp(x) == 0:
        return {"n": len(d), "covariates": ",".join(covs)}

    # Keep only covariates that add rank; k and rows are often identical in
    # fixed-budget designs and including both would make the fit singular.
    zcols: list[str] = []
    design = np.column_stack([np.ones(len(d)), x])
    rank = np.linalg.matrix_rank(design)
    for c in covs:
        trial = np.column_stack([design, d[c].to_numpy(float)])
        new_rank = np.linalg.matrix_rank(trial)
        if new_rank > rank:
            zcols.append(c)
            design, rank = trial, new_rank
    if not zcols:
        # A configured covariate varies but is exactly collinear with the
        # predictor.  The requested adjusted model is not identifiable; keep
        # an explicit OLS row (with NaN estimates) instead of silently making
        # it look as though no adjusted analysis was requested.
        return {"n": len(d), "covariates": ",".join(covs)}

    out: dict[str, Any] = {"n": len(d), "covariates": ",".join(zcols)}
    if rank < design.shape[1] or len(d) <= design.shape[1]:
        return out
    try:
        import statsmodels.api as sm

        fit = sm.OLS(y, design).fit()
        ci = np.asarray(fit.conf_int(alpha=0.05), dtype=float)[1]
        out.update(coefficient=float(fit.params[1]), coefficient_se=float(fit.bse[1]),
                   coefficient_p=float(fit.pvalues[1]), coefficient_ci_low=float(ci[0]),
                   coefficient_ci_high=float(ci[1]))
    except Exception:
        # Point estimation remains useful in a minimal analysis environment;
        # statsmodels is a project dependency, so this mainly protects reports
        # from numerical failures in tiny/degenerate slices.
        out["coefficient"] = float(np.linalg.lstsq(design, y, rcond=None)[0][1])

    z = np.column_stack([np.ones(len(d)), *[d[c].to_numpy(float) for c in zcols]])
    xr = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    yr = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    out["partial_r"] = _corr(xr, yr)
    return out


def load_metrics(cfg: MultivalueConfig, layout: Layout, arms: Sequence[Arm]) -> pd.DataFrame:
    """Load and validate the metrics stage against this design and arm list."""
    path = layout.metrics_csv
    if not path.is_file():
        raise O.AnalyzeError(f"no metrics at {path}: run `valuegen mv metrics -c {cfg.path}`")
    df = pd.read_csv(path)
    required = {"arm_id", "family", "kind", "store", "coverage", "tightness", "k", "rows", "arms_sha256"}
    missing = required - set(df)
    if missing:
        raise O.AnalyzeError(f"{path}: missing columns {sorted(missing)}; rerun `valuegen mv metrics`")
    sets = mvdata.read_json(layout.sets_path)
    hashes = set(df["arms_sha256"].dropna().astype(str))
    if hashes != {str(sets["arms_sha256"])}:
        raise O.AnalyzeError(f"{path}: stale arms_sha256 {sorted(hashes)}; rerun `valuegen mv metrics`")
    expected_arms, expected_stores = {a.id for a in arms}, set(cfg.embeddings)
    got_arms, got_stores = set(df["arm_id"].astype(str)), set(df["store"].astype(str))
    if got_arms != expected_arms or got_stores != expected_stores:
        raise O.AnalyzeError(
            f"{path}: arms/stores differ from the current experiment "
            f"(missing arms {sorted(expected_arms - got_arms)}, extra arms {sorted(got_arms - expected_arms)}, "
            f"missing stores {sorted(expected_stores - got_stores)}, "
            f"extra stores {sorted(got_stores - expected_stores)}); "
            "rerun `valuegen mv metrics`"
        )
    counts = df.groupby(["arm_id", "store"], dropna=False).size()
    if (counts != 1).any() or len(counts) != len(expected_arms) * len(expected_stores):
        raise O.AnalyzeError(f"{path}: expected exactly one row per arm x store; rerun `valuegen mv metrics`")
    return df


def select_outcomes(cfg: MultivalueConfig, outcomes: pd.DataFrame) -> list[str]:
    """Expand configured ``suite.metric`` names, including ``*`` patterns."""
    available = sorted({f"{r.suite}.{r.metric}" for r in outcomes[["suite", "metric"]].itertuples(index=False)})
    if not cfg.analysis.outcomes:
        return available
    selected: list[str] = []
    for pattern in cfg.analysis.outcomes:
        matches = [name for name in available if fnmatch.fnmatchcase(name, pattern)]
        for name in matches:
            if name not in selected:
                selected.append(name)
    return selected


def _expected_checkpoints(candidates: Sequence) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for c in candidates:
        out.setdefault(c.arm.id, set()).add(c.ckpt_id)
    return out


def build_joined(cfg: MultivalueConfig, arms: Sequence[Arm], metrics: pd.DataFrame,
                 outcomes: pd.DataFrame, candidates: Sequence) -> tuple[pd.DataFrame, list[str]]:
    """One wide row per arm; return it and the expanded outcome names."""
    first_store = next(iter(cfg.embeddings))
    first = metrics[metrics["store"] == first_store].set_index("arm_id")
    rows = []
    for arm in arms:
        mr = first.loc[arm.id]
        rows.append({"arm_id": arm.id, "family": arm.family, "kind": arm.kind,
                     "k": int(mr["k"]), "rows": int(mr["rows"])})
    joined = pd.DataFrame(rows)
    for store in cfg.embeddings:
        d = metrics[metrics["store"] == store].set_index("arm_id")
        for metric in cfg.analysis.metrics:
            joined[f"{store}.{metric}"] = joined["arm_id"].map(d[metric])

    selected = select_outcomes(cfg, outcomes)
    expected = _expected_checkpoints(candidates)
    for outcome in selected:
        suite, metric = outcome.split(".", 1)
        d = outcomes[(outcomes["suite"] == suite) & (outcomes["metric"] == metric)]
        by_arm = {aid: g for aid, g in d.groupby("arm_id")}
        for suffix in ("estimate", "ci_low", "ci_high", "delta_vs_base", "delta_ci_low", "delta_ci_high"):
            joined[f"{outcome}.{suffix}"] = [
                float(pd.to_numeric(by_arm[a.id][suffix], errors="coerce").mean()) if a.id in by_arm else np.nan
                for a in arms
            ]
        joined[f"{outcome}.complete"] = [
            bool(a.id in by_arm
                 and set(by_arm[a.id]["ckpt_id"].astype(str)) == expected.get(a.id, set())
                 and by_arm[a.id]["complete"].astype(bool).all())
            for a in arms
        ]
        joined[f"{outcome}.n_checkpoints"] = [
            int(by_arm[a.id]["ckpt_id"].nunique()) if a.id in by_arm else 0 for a in arms
        ]
    return joined, selected


def _combo_seed(base: int, *parts: str) -> int:
    return int(sha256_json([int(base), *parts])[:16], 16) % (2**32)


def build_correlations(cfg: MultivalueConfig, joined: pd.DataFrame,
                       selected: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for store in cfg.embeddings:
        for metric in cfg.analysis.metrics:
            xcol = f"{store}.{metric}"
            for outcome in selected:
                complete_col = f"{outcome}.complete"
                for scale in ("estimate", "delta_vs_base"):
                    ycol = f"{outcome}.{scale}"
                    if xcol not in joined or ycol not in joined:
                        continue
                    keep = (joined[complete_col].astype(bool) if complete_col in joined
                            else pd.Series(False, index=joined.index))
                    d = joined.loc[keep, [xcol, ycol, *[c for c in cfg.analysis.covariates if c in joined]]].copy()
                    xy = d[[xcol, ycol]].apply(pd.to_numeric, errors="coerce")
                    d = d.loc[xy.notna().all(axis=1)].copy()
                    seed = _combo_seed(cfg.seed, store, metric, outcome, scale)
                    res = correlate(d[xcol], d[ycol], cfg.analysis.n_perm, seed, cfg.analysis.n_boot)
                    base = {"store": store, "metric": metric, "outcome": outcome,
                            "outcome_scale": scale, "method": "correlation", **res}
                    rows.append(base)
                    ols = ols_adjusted(d, xcol, ycol, cfg.analysis.covariates)
                    if ols is not None:
                        rows.append({"store": store, "metric": metric, "outcome": outcome,
                                     "outcome_scale": scale, "method": "ols", **ols})
    return pd.DataFrame(rows, columns=list(CORRELATION_COLUMNS))


def _slug(*parts: str) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_") for x in parts)


def _decorate_scatter(fig, ax, k: np.ndarray, base_y: float, base_lo: float, base_hi: float, scale: str) -> None:
    if ax.collections:
        points = ax.collections[0]
        points.set_array(np.asarray(k, dtype=float))
        points.set_cmap("viridis")
        if len(np.unique(k[np.isfinite(k)])) > 1:
            fig.colorbar(points, ax=ax, label="k")
    reference = 0.0 if scale == "delta_vs_base" else base_y
    if np.isfinite(reference):
        ax.axhline(reference, color="black", linestyle="--", linewidth=1, alpha=0.75, label="base")
    if scale == "estimate" and np.isfinite(base_lo) and np.isfinite(base_hi):
        ax.axhspan(base_lo, base_hi, color="black", alpha=0.08, label="base 95% bootstrap CI")
    if np.isfinite(reference):
        ax.legend(frameon=False, fontsize=8)


def write_figures(cfg: MultivalueConfig, layout: Layout, joined: pd.DataFrame,
                  selected: Sequence[str], log=print) -> list[Path]:
    """Scatter grid from the joined table; return all files written."""
    figdir = layout.scores_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    base = joined[joined["kind"] == "base"]
    for store in cfg.embeddings:
        for metric in cfg.analysis.metrics:
            xcol = f"{store}.{metric}"
            for outcome in selected:
                complete_col = f"{outcome}.complete"
                for scale in ("estimate", "delta_vs_base"):
                    ycol = f"{outcome}.{scale}"
                    keep = (joined[complete_col].astype(bool) if complete_col in joined
                            else pd.Series(False, index=joined.index))
                    d = joined.loc[keep, [xcol, ycol, "k"]].dropna()
                    if len(d) < 2:
                        continue
                    fig, ax = scatter_fit(d[xcol].to_numpy(float), d[ycol].to_numpy(float),
                                          xlabel=f"{store} {metric}", ylabel=f"{outcome} {scale}",
                                          title=f"{store} {metric} vs {outcome} ({scale})")
                    vals = base.iloc[0] if len(base) else {}
                    _decorate_scatter(fig, ax, d["k"].to_numpy(float),
                                      float(vals.get(f"{outcome}.estimate", np.nan)),
                                      float(vals.get(f"{outcome}.ci_low", np.nan)),
                                      float(vals.get(f"{outcome}.ci_high", np.nan)), scale)
                    path = figdir / (_slug(store, metric, outcome, scale) + ".png")
                    save(fig, path)
                    written.append(path)

    log(f"figures: {len(written)} -> {figdir}")
    return written


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int | None = None) -> str:
    columns = [c for c in columns if c in df]
    if not columns or df.empty:
        return "_(no data)_"
    d = df.loc[:, columns]
    omitted = 0
    if max_rows is not None and len(d) > max_rows:
        omitted = len(d) - max_rows
        d = d.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(_fmt(v) for v in row) + " |" for row in d.itertuples(index=False, name=None))
    if omitted:
        lines.append(f"\n_({omitted} additional rows are in the CSV.)_")
    return "\n".join(lines)


def write_report(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm],
                 joined: pd.DataFrame, correlations: pd.DataFrame, completeness: pd.DataFrame,
                 selected: Sequence[str], figures: Sequence[Path]) -> Path:
    sets = mvdata.read_json(layout.sets_path)
    complete_n = int((completeness["state"].isin(["done", "reusable"])).sum()) if len(completeness) else 0
    total_n = len(completeness)
    family_lines = []
    for fid, fam in sets.get("families", {}).items():
        bins = [r.get("bin") for r in fam.get("arms", {}).values() if r.get("bin") is not None]
        bin_counts = {int(b): bins.count(b) for b in sorted(set(bins))}
        bin_summary = f", bins={bin_counts}" if bins else ""
        family_lines.append(f"- `{fid}`: {fam.get('n')} arms, k={fam.get('k') or 'varies'}, "
                            f"scheme={fam.get('scheme')}{bin_summary}")
    design_stats = []
    family_ids = [fid for fid in sets.get("families", {})]
    for family in family_ids:
        d = joined[joined["family"] == family]
        for store in cfg.embeddings:
            for metric in cfg.analysis.metrics:
                values = pd.to_numeric(d[f"{store}.{metric}"], errors="coerce").dropna()
                if values.empty:
                    continue
                q = values.quantile([0, .25, .5, .75, 1])
                design_stats.append({"family": family, "store": store, "metric": metric, "n": len(values),
                                     "min": q.loc[0], "q25": q.loc[.25], "median": q.loc[.5],
                                     "q75": q.loc[.75], "max": q.loc[1], "iqr": q.loc[.75] - q.loc[.25]})
    design_stats_df = pd.DataFrame(design_stats)
    corr_view = correlations[correlations["method"] == "correlation"] if len(correlations) else correlations
    ols_view = correlations[correlations["method"] == "ols"] if len(correlations) else correlations
    cfg_arg = shlex.quote(str(cfg.path)) if cfg.path else "CONFIG.yaml"
    cluster_arg = f" --cluster {shlex.quote(str(cluster.source_path))}" if cluster.source_path else ""
    outcome_cols = ["arm_id", "family", "kind", "k", "rows"]
    for name in selected:
        outcome_cols += [f"{name}.estimate", f"{name}.delta_vs_base", f"{name}.complete"]
    lines = [
        f"# Multivalue analysis: {cfg.name}", "",
        f"Generated {mvdata.now()}. Experiment `{layout.exp_id}`.", "",
        "## Design", "",
        f"- Base model: `{cfg.train.base_model}`" + (f" at `{cfg.train.revision}`" if cfg.train.revision else ""),
        f"- Frozen sets: `{layout.sets_path}` (`arms_sha256={sets.get('arms_sha256')}`)",
        f"- Sampling: {sets.get('scheme')} on {sets.get('store')}.{sets.get('metric')}; "
        f"candidate pool={sets.get('n_candidates')}, bins={sets.get('bins')}, seed={sets.get('seed')}",
        f"- Fixed configured training budget: {cfg.budget.rows} rows per generated arm; "
        f"training seeds={list(cfg.train.seeds)}",
        *family_lines, "",
        "### Frozen-arm metric readouts", "",
        markdown_table(design_stats_df,
                       ["family", "store", "metric", "n", "min", "q25", "median", "q75", "max", "iqr"]), "",
        "## Completeness", "",
        f"{complete_n}/{total_n} candidate-suite evaluations satisfy the current protocol. "
        "Rows from incomplete runs are summarized for diagnosis but excluded from correlations.", "",
        markdown_table(completeness, ["arm_id", "ckpt_id", "kind", "suite", "state", "rows"], max_rows=100), "",
        "## Per-arm outcomes", "",
        "Each row is one arm. When multiple training seeds are configured, point estimates and interval endpoints "
        "are averaged; `.complete` requires every expected seed.", "",
        markdown_table(joined, outcome_cols, max_rows=100), "",
        "## Correlations", "",
        f"Pearson r, Spearman rho, two-sided label-permutation p ({cfg.analysis.n_perm} draws), and a 95% "
        f"arm-bootstrap interval for Pearson r ({cfg.analysis.n_boot} draws).", "",
        markdown_table(corr_view, ["store", "metric", "outcome", "outcome_scale", "n", "pearson_r",
                                   "spearman_rho", "perm_p", "bootstrap_ci_low", "bootstrap_ci_high"]), "",
        "## Covariate-adjusted OLS", "",
        "OLS is emitted only when at least one configured covariate varies and adds independent rank. "
        "The coefficient is in raw metric units; partial r residualizes both predictor and outcome on the same "
        "covariates.", "",
        markdown_table(ols_view, ["store", "metric", "outcome", "outcome_scale", "n", "coefficient",
                                  "coefficient_se", "coefficient_p", "coefficient_ci_low", "coefficient_ci_high",
                                  "partial_r", "covariates"]), "",
        "## Figures", "",
        *([f"- [{p.name}](scores/figures/{p.name})" for p in figures]
          or ["_(No figure had at least two complete observations.)_"]),
        "", "## Limitations", "",
        "- Bootstrap intervals quantify evaluation-item sampling, not training-seed uncertainty. With one training "
        "seed, optimization variability is unmeasured.",
        "- Correlations are observational across selected value sets; they do not identify a causal effect of "
        "coverage or tightness.",
        "- Prefill scores depend on the configured judge and prompts. The base band reflects "
        "evaluation sampling only.",
        "- Imported arms may differ in provenance; any allowed base mismatch should be treated as exploratory "
        "rather than directly comparable.",
        "", "## Reproduce", "", "```bash",
        f"valuegen mv metrics -c {cfg_arg}{cluster_arg}",
        f"valuegen mv analyze -c {cfg_arg}{cluster_arg}",
        f"valuegen mv status -c {cfg_arg}{cluster_arg}",
        "```", "",
        "Artifacts: `scores/outcomes.csv`, `scores/joined.csv`, `scores/correlations.csv`, and "
        "`scores/completeness.csv`.",
    ]
    layout.report_path.parent.mkdir(parents=True, exist_ok=True)
    layout.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return layout.report_path


def analyze(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm], *,
            log=print) -> dict[str, Any]:
    """Run the complete local analysis stage and return its artifacts."""
    metrics = load_metrics(cfg, layout, arms)
    outcomes, completeness = O.compute_outcomes(cfg, cluster, layout, arms, log=log)
    O.write_outcomes(layout, outcomes, completeness)
    candidates = ES.plan_candidates(cfg, cluster, layout, arms)
    joined, selected = build_joined(cfg, arms, metrics, outcomes, candidates)
    correlations = build_correlations(cfg, joined, selected)
    _write_csv(joined, layout.scores_dir / "joined.csv")
    _write_csv(correlations, layout.scores_dir / "correlations.csv")
    figures = write_figures(cfg, layout, joined, selected, log=log)
    report = write_report(cfg, cluster, layout, arms, joined, correlations, completeness, selected, figures)
    log(f"analysis: {len(outcomes)} outcomes, {len(joined)} arms, {len(correlations)} statistical rows")
    log(f"wrote {layout.scores_dir / 'outcomes.csv'}")
    log(f"wrote {layout.scores_dir / 'joined.csv'}")
    log(f"wrote {layout.scores_dir / 'correlations.csv'}")
    log(f"wrote {report}")
    return {"outcomes": outcomes, "completeness": completeness, "joined": joined,
            "correlations": correlations, "figures": figures, "report": report}
