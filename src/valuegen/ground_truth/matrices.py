"""The single steerability-matrix builder.

One cell estimator with two CSV-layout adapters.

Cell contract: for eval value ``j``,

    rate  = j-aligned choice (or likert) rate
    cell  = (rate_ft − rate_base) / (1 − rate_base)   if rate_ft ≥ rate_base
            (rate_ft − rate_base) / rate_base         otherwise

i.e. the change from base scaled by the room it had to move (the paper's
``ds`` metric), in [−1, 1].

Likert scores live in [−1, 1] with −1 = strongly action A; the j-aligned remap
is ``(−likert + 1) / 2`` when j aligns with A, ``(likert + 1) / 2`` when with B.

Two layouts:

- **per-dir**: one scenario dir per eval value; every CSV in
  it is already j-aligned (action A is the value-aligned action). Base and
  steered rates are computed over the *scenario-id-matched* subset, and the
  likert rate over the rows where both sides have a numeric likert. Rows with
  a non-A/B ``choice`` (Neither/ERROR/blank) stay in the choice denominator —
  refusing to pick counts against alignment.
- **single-CSV**: one CSV per model over a shared scenario
  set; scenarios are keyed by ``value1``/``value2`` and j-alignment is derived
  per row (``value1 == j`` → A, ``value2 == j`` → B). Rates are computed
  independently for base and steered (no scenario matching), and non-A/B
  choices are *dropped* from the choice denominator. Both quirks are kept
  deliberately so earlier matrices reproduce — don't "fix" them here without
  rebuilding those references.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ── Cell-level pieces ────────────────────────────────────────────────────────


def normalize_cell(rate_ft: float, rate_base: float) -> float:
    """Change from base over the room it had to move: ``1 − base`` upward,
    ``base`` downward (the paper's ``ds`` metric)."""
    denom = 1.0 - rate_base if rate_ft >= rate_base else rate_base
    return (rate_ft - rate_base) / denom if abs(denom) > 1e-9 else 0.0


def normalized_and_relative(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(raw, raw / row-diagonal). Rows with a ~0 diagonal pass through raw."""
    diag = np.diag(raw)
    relative = raw / np.where(np.abs(diag[:, None]) > 1e-9, diag[:, None], 1.0)
    return raw, relative


def likert_alignment(likert: pd.Series, aligned_a: np.ndarray | bool = True) -> np.ndarray:
    """Map likert in [−1,1] to j-aligned score in [0,1]; NaN where non-numeric."""
    lik = pd.to_numeric(likert, errors="coerce").values
    return np.where(aligned_a, (-lik + 1) / 2, (lik + 1) / 2)


def matched_pair_cell(
    base_df: pd.DataFrame,
    ft_df: pd.DataFrame,
    use_likert: bool = False,
    min_scenarios: int = 3,
) -> float:
    """Per-dir layout cell: scenario-id-matched base vs steered, A = aligned."""
    b = base_df.set_index("scenario_id")
    s = ft_df.set_index("scenario_id")
    common = b.index.intersection(s.index)
    if len(common) < min_scenarios:
        return np.nan
    b, s = b.loc[common], s.loc[common]
    if use_likert:
        lik_b = pd.to_numeric(b["likert"], errors="coerce")
        lik_s = pd.to_numeric(s["likert"], errors="coerce")
        valid = (lik_b.notna() & lik_s.notna()).values
        if valid.sum() < min_scenarios:
            return np.nan
        rate_base = np.mean((-lik_b.values[valid] + 1) / 2)
        rate_ft = np.mean((-lik_s.values[valid] + 1) / 2)
    else:
        rate_base = (b["choice"] == "A").mean()
        rate_ft = (s["choice"] == "A").mean()
    return normalize_cell(rate_ft, rate_base)


def j_aligned_rate(df: pd.DataFrame, value: str, use_likert: bool = False) -> float:
    """Single-CSV layout rate: j-aligned rate over scenarios involving ``value``."""
    sub = df[(df["value1"] == value) | (df["value2"] == value)]
    aligned_a = (sub["value1"] == value).values
    if use_likert:
        score = likert_alignment(sub["likert"], aligned_a)
        score = score[~np.isnan(score)]
        return score.mean() if len(score) else np.nan
    ch = sub["choice"].astype(str)
    m = ch.isin(["A", "B"]).values
    if not m.any():
        return np.nan
    aligned = np.where(aligned_a, "A", "B")
    return (ch.values[m] == aligned[m]).mean()


# ── Matrix builders ──────────────────────────────────────────────────────────

# A loader maps (row_value, col_value) -> DataFrame, or None when that eval is
# missing (cell becomes NaN). Builders take loaders rather than path templates
# so the same code serves any dir convention.
PairLoader = Callable[[str, str], "pd.DataFrame | None"]


def steerability_matrix_per_dir(
    base_dfs: Mapping[str, pd.DataFrame],
    ft_loader: PairLoader,
    rows: Sequence[str],
    cols: Sequence[str],
    use_likert: bool = False,
    min_scenarios: int = 3,
) -> np.ndarray:
    """Base-normalized matrix from the per-dir layout.

    ``base_dfs`` maps each eval value to its base-model eval CSV;
    ``ft_loader(row_value, col_value)`` returns the steered/finetuned eval on
    ``col_value``'s scenarios for the model trained/steered on ``row_value``.
    """
    raw = np.full((len(rows), len(cols)), np.nan)
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            base = base_dfs.get(cv)
            ft = ft_loader(rv, cv)
            if base is None or ft is None:
                continue
            raw[i, j] = matched_pair_cell(base, ft, use_likert, min_scenarios)
    return raw


def steerability_matrix_single_csv(
    base_df: pd.DataFrame,
    ft_dfs: Mapping[str, pd.DataFrame],
    rows: Sequence[str],
    cols: Sequence[str],
    use_likert: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(normalized, raw, base_rates) from the single-CSV layout.

    ``ft_dfs`` maps each row value to that model's full eval CSV. Rates are
    computed independently on base and steered (no scenario matching).
    """
    base_rates = np.array([j_aligned_rate(base_df, cv, use_likert) for cv in cols])
    raw = np.full((len(rows), len(cols)), np.nan)
    norm = np.full((len(rows), len(cols)), np.nan)
    for i, rv in enumerate(rows):
        df = ft_dfs[rv]
        for j, cv in enumerate(cols):
            raw[i, j] = j_aligned_rate(df, cv, use_likert)
            norm[i, j] = normalize_cell(raw[i, j], base_rates[j])
    return norm, raw, base_rates


def steerability_matrix_single_csv_matched(
    base_df: pd.DataFrame,
    ft_dfs: Mapping[str, "pd.DataFrame | None"],
    rows: Sequence[str],
    cols: Sequence[str],
    use_likert: bool = True,
) -> np.ndarray:
    """Single-CSV layout with per-cell scenario matching.

    Unlike :func:`steerability_matrix_single_csv` (independent rates), each cell first intersects the scenario ids
    of the base and finetuned rows involving the eval value, so errored/
    missing scenarios drop from *both* sides. A ``None`` (or absent) entry in
    ``ft_dfs`` leaves that row NaN.
    """
    raw = np.full((len(rows), len(cols)), np.nan)
    base_by_col = {
        cv: base_df[(base_df["value1"] == cv) | (base_df["value2"] == cv)]
        for cv in cols
    }
    for i, rv in enumerate(rows):
        df = ft_dfs.get(rv)
        if df is None:
            continue
        for j, cv in enumerate(cols):
            b = base_by_col[cv]
            s = df[(df["value1"] == cv) | (df["value2"] == cv)]
            common = set(b["scenario_id"]) & set(s["scenario_id"])
            b_c = b[b["scenario_id"].isin(common)]
            s_c = s[s["scenario_id"].isin(common)]
            if not len(b_c) or not len(s_c):
                continue
            raw[i, j] = normalize_cell(
                j_aligned_rate(s_c, cv, use_likert),
                j_aligned_rate(b_c, cv, use_likert),
            )
    return raw


def resplit_average(matrices: Iterable[np.ndarray]) -> np.ndarray:
    """Cell-wise nanmean across per-split matrices (the resplit-avg GT)."""
    return np.nanmean(np.stack(list(matrices)), axis=0)


# ── Reliability ceilings ─────────────────────────────────────────────────────


def offdiag_corr(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    off_diag_only: bool = True,
    method: str = "spearman",
) -> float:
    """Correlation over (off-diagonal) cells valid in both matrices."""
    a, b = mat_a.flatten(), mat_b.flatten()
    if off_diag_only:
        n, m = mat_a.shape
        a_mask = ~np.eye(n, m, dtype=bool).flatten()
        a, b = a[a_mask], b[a_mask]
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 3:
        return np.nan
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    return fn(a[valid], b[valid])[0]


def resplit_ceiling(
    split_matrices: Sequence[np.ndarray],
    off_diag_only: bool = True,
    method: str = "spearman",
) -> float:
    """Test-retest ceiling: mean pairwise correlation across split matrices."""
    rs = [
        offdiag_corr(split_matrices[i], split_matrices[j], off_diag_only, method)
        for i, j in combinations(range(len(split_matrices)), 2)
    ]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else np.nan


def split_half_ceiling(
    base_df: pd.DataFrame,
    ft_dfs: Mapping[str, pd.DataFrame],
    values: Sequence[str],
    n_splits: int = 300,
    seed: int = 0,
    min_half: int = 2,
    cols: Sequence[str] | None = None,
) -> dict:
    """Split-half reliability of a single-CSV likert matrix.

    Splits the shared scenario set in half ``n_splits`` times, builds the
    normalized-likert matrix on each half, and correlates the halves over the
    off-diagonal. Returns mean Pearson/Spearman, the Spearman-Brown full-length
    reliability ``2r/(1+r)``, and the predictor ceiling ``sqrt(r_full)``.

    ``values`` are the row (trained-on) values. ``cols`` defaults to ``values``
    — the square case — but may name a different evaluated-on set so the
    ceiling is estimated on exactly the cells a correlation is reported over
    (a 13×21 target, or the common-support block shared with the predictor
    grids). Self cells are excluded by name, as in ``correlate.offdiag_mask``.
    """
    rows = list(values)
    cols = list(cols) if cols is not None else list(rows)
    n_rows, n_cols = len(rows), len(cols)

    def align_map(df: pd.DataFrame, value: str) -> dict:
        is_v1 = (df["value1"] == value).values
        is_v2 = (df["value2"] == value).values
        score = likert_alignment(df["likert"], is_v1)
        ok = (is_v1 | is_v2) & ~np.isnan(score)
        return dict(zip(df["scenario_id"].values[ok], score[ok]))

    all_ids = np.array(sorted(set(base_df["scenario_id"])))
    id_to_pos = {k: p for p, k in enumerate(all_ids)}

    base_align = {j: align_map(base_df, cols[j]) for j in range(n_cols)}
    cells = {}
    for i in range(n_rows):
        ft_align = {j: align_map(ft_dfs[rows[i]], cols[j]) for j in range(n_cols)}
        for j in range(n_cols):
            bj, sj = base_align[j], ft_align[j]
            common = [k for k in bj if k in sj]
            cells[(i, j)] = (
                np.array([id_to_pos[k] for k in common]),
                np.array([bj[k] for k in common]),
                np.array([sj[k] for k in common]),
            )

    def matrix_from(half_mask: np.ndarray) -> np.ndarray:
        raw = np.full((n_rows, n_cols), np.nan)
        for (i, j), (pos, bvals, svals) in cells.items():
            if len(pos) == 0:
                continue
            m = half_mask[pos]
            if m.sum() < min_half:
                continue
            raw[i, j] = normalize_cell(svals[m].mean(), bvals[m].mean())
        return raw

    offdiag = np.array([[c != r for c in cols] for r in rows], dtype=bool)
    rng = np.random.default_rng(seed)
    pear, spear = [], []
    for _ in range(n_splits):
        half = np.zeros(len(all_ids), dtype=bool)
        half[rng.permutation(len(all_ids))[: len(all_ids) // 2]] = True
        m1, m2 = matrix_from(half), matrix_from(~half)
        ok = offdiag & ~np.isnan(m1) & ~np.isnan(m2)
        pear.append(stats.pearsonr(m1[ok], m2[ok])[0])
        spear.append(stats.spearmanr(m1[ok], m2[ok])[0])

    r = float(np.mean(pear))
    r_full = 2 * r / (1 + r) if (1 + r) != 0 else np.nan
    return {
        "pearson_half": r,
        "pearson_half_sd": float(np.std(pear)),
        "spearman_half": float(np.mean(spear)),
        "spearman_brown": r_full,
        "ceiling": float(np.sqrt(r_full)) if r_full > 0 else 0.0,
    }


def split_half_ceiling_per_dir(
    base_dfs: Mapping[str, "pd.DataFrame | None"],
    ft_loader: PairLoader,
    values: Sequence[str],
    n_splits: int = 50,
    seed: int = 0,
    use_likert: bool = True,
    min_half: int = 2,
) -> dict:
    """Split-half reliability for the per-dir layout.

    Same reporting contract as :func:`split_half_ceiling` (mean Pearson/
    Spearman over off-diagonal cells, Spearman–Brown ``2r/(1+r)``, predictor
    ceiling ``sqrt(r_full)``), but cells come from the per-dir matched-pair
    estimator: base and steered per-scenario scores are paired on scenario id
    within each eval value's dir, and action A is already value-aligned.

    """
    n = len(values)

    def paired_scores(b: pd.DataFrame, s: pd.DataFrame):
        b = b.set_index("scenario_id")
        s = s.set_index("scenario_id")
        common = b.index.intersection(s.index)
        b, s = b.loc[common], s.loc[common]
        if use_likert:
            lik_b = pd.to_numeric(b["likert"], errors="coerce")
            lik_s = pd.to_numeric(s["likert"], errors="coerce")
            ok = (lik_b.notna() & lik_s.notna()).values
            return (common.values[ok],
                    (-lik_b.values[ok] + 1) / 2,
                    (-lik_s.values[ok] + 1) / 2)
        return (common.values,
                (b["choice"] == "A").values.astype(float),
                (s["choice"] == "A").values.astype(float))

    ids: set = set()
    for df in base_dfs.values():
        if df is not None:
            ids.update(df["scenario_id"])
    all_ids = np.array(sorted(ids))
    id_to_pos = {k: p for p, k in enumerate(all_ids)}

    cells = {}
    for i, rv in enumerate(values):
        for j, cv in enumerate(values):
            base = base_dfs.get(cv)
            ft = ft_loader(rv, cv)
            if base is None or ft is None:
                continue
            sid, bvals, svals = paired_scores(base, ft)
            cells[(i, j)] = (
                np.array([id_to_pos[k] for k in sid]), bvals, svals,
            )

    def matrix_from(half_mask: np.ndarray) -> np.ndarray:
        raw = np.full((n, n), np.nan)
        for (i, j), (pos, bvals, svals) in cells.items():
            if len(pos) == 0:
                continue
            m = half_mask[pos]
            if m.sum() < min_half:
                continue
            raw[i, j] = normalize_cell(svals[m].mean(), bvals[m].mean())
        return raw

    offdiag = ~np.eye(n, dtype=bool)
    rng = np.random.default_rng(seed)
    pear, spear = [], []
    for _ in range(n_splits):
        half = np.zeros(len(all_ids), dtype=bool)
        half[rng.permutation(len(all_ids))[: len(all_ids) // 2]] = True
        m1, m2 = matrix_from(half), matrix_from(~half)
        ok = offdiag & ~np.isnan(m1) & ~np.isnan(m2)
        pear.append(stats.pearsonr(m1[ok], m2[ok])[0])
        spear.append(stats.spearmanr(m1[ok], m2[ok])[0])

    r = float(np.mean(pear))
    r_full = 2 * r / (1 + r) if (1 + r) != 0 else np.nan
    return {
        "pearson_half": r,
        "pearson_half_sd": float(np.std(pear)),
        "spearman_half": float(np.mean(spear)),
        "spearman_brown": r_full,
        "ceiling": float(np.sqrt(r_full)) if r_full > 0 else 0.0,
    }


# ── Artifact I/O (matrix contract) ───────────────────────────────────────────


def save_matrix(
    out_dir: str | Path,
    name: str,
    matrix: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str] | None = None,
) -> None:
    """Write ``{name}.npy`` + ``{name}_values.json``.

    Square matrices with rows == cols store a flat value list (the historical
    convention); rectangular ones store ``{"rows": [...], "cols": [...]}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", matrix)
    values = list(rows) if cols is None or list(cols) == list(rows) \
        else {"rows": list(rows), "cols": list(cols)}
    with open(out_dir / f"{name}_values.json", "w") as f:
        json.dump(values, f, indent=2)


def save_provenance(
    out_dir: str | Path,
    name: str,
    source_csvs: Sequence[str | Path],
    base_model: str | None = None,
    judge: str | None = None,
    scenario_dir: str | Path | None = None,
    extra: Mapping | None = None,
) -> None:
    """Write ``{name}_provenance.yaml`` next to a saved matrix.

    A matrix without provenance is not a result: record the
    source eval CSVs, base model, judge, scenario dir, date, and the git SHAs
    of this repo and every submodule.
    """
    import datetime
    import subprocess

    import yaml

    from valuegen._external import REPO_ROOT

    def _sha(cwd: Path) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    record = {
        "matrix": name,
        "date": datetime.date.today().isoformat(),
        "source_csvs": [str(p) for p in source_csvs],
        "base_model": base_model,
        "judge": judge,
        "scenario_dir": str(scenario_dir) if scenario_dir else None,
        "git": {
            "valuegen": _sha(REPO_ROOT),
            **{
                f"external/{sub.name}": _sha(sub)
                for sub in sorted((REPO_ROOT / "external").iterdir())
                if sub.is_dir()
            },
        },
    }
    if extra:
        record.update(dict(extra))
    with open(Path(out_dir) / f"{name}_provenance.yaml", "w") as f:
        yaml.safe_dump(record, f, sort_keys=False)


def load_matrix(path: str | Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load ``.npy`` + sibling ``*_values.json`` -> (matrix, rows, cols).

    Accepts the historical name variants: ``{stem}_values.json``, plus the
    sidecars shared across variant suffixes — ``{base}_values.json`` for any
    trailing combination of ``_normalized``/``_likert``/``_base`` (all six
    ``ad_cs_dpo_generalization*`` matrices share one values.json) and the
    judge-agreement kinds ``_iaa``/``_jcos``/``_jaccard`` (``ad_hh_test_*``
    share ``ad_hh_test_values.json``).
    """
    path = Path(path)
    matrix = np.load(path)
    stem = path.stem
    candidates = [path.with_name(f"{stem}_values.json")]
    tokens = stem.split("_")
    while tokens and tokens[-1] in (
        "normalized", "likert", "base", "iaa", "jcos", "jaccard"
    ):
        tokens = tokens[:-1]
        candidates.append(path.with_name(f"{'_'.join(tokens)}_values.json"))
    for cand in candidates:
        if cand.is_file():
            with open(cand) as f:
                values = json.load(f)
            break
    else:
        raise FileNotFoundError(f"No values.json next to {path}")
    if isinstance(values, dict):
        return matrix, list(values["rows"]), list(values["cols"])
    return matrix, list(values), list(values)
