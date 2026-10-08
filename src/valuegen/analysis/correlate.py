"""Correlate generalization matrices: masks, CIs, ceilings, leaderboards.

The core call is :func:`correlate(matrix_a, matrix_b, mask)` over any two
arrays sharing a shape — GT-vs-predictor (leaderboards), GT-vs-GT
(prompt-steer as a reference for DPO, cross-model grid similarity), and
half-vs-half (split-half ceilings) are all the same call; there is no
special-cased ceiling or cross-model path.

Matrices carry value labels (``.npy + values.json``); :func:`reindex` maps a
predictor's square grid onto a target's (rows, cols) frame by *name*, so a
21×21 similarity matrix lines up under a 13×21 GT grid without hand-kept
index arrays. :func:`offdiag_mask` drops the self column — square and
rectangular grids alike — because the off-diagonal transfer cells are the
generalization question; the diagonal is the direct effect.

Numeric conventions: Pearson bootstrap with
``np.random.default_rng(seed)`` and 5000 resamples, percentile 95% CI,
Spearman–Brown ``2r/(1+r)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
from scipy import stats

# ── Masks and label alignment ────────────────────────────────────────────────


def offdiag_mask(rows: Sequence[str], cols: Sequence[str] | None = None) -> np.ndarray:
    """True on cross-value (transfer) cells; the self column is dropped.

    Square (``cols`` omitted or equal to ``rows``): ``~eye``. Rectangular
    (e.g. 13 trained × 21 eval): cell ``(i, j)`` is masked out when
    ``cols[j] == rows[i]`` — the predict_dpo self-column drop.
    """
    rows = list(rows)
    cols = list(cols) if cols is not None else rows
    return np.array([[c != r for c in cols] for r in rows], dtype=bool)


def reindex(
    matrix: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str],
    new_rows: Sequence[str],
    new_cols: Sequence[str],
) -> np.ndarray:
    """Re-frame ``matrix`` onto (``new_rows``, ``new_cols``) by label; NaN
    where a label is missing (the ad_12 NaN-pad convention). Subsetting a
    21×21 similarity grid to a 13×21 target frame is the common case."""
    ri = {v: i for i, v in enumerate(rows)}
    ci = {v: i for i, v in enumerate(cols)}
    out = np.full((len(new_rows), len(new_cols)), np.nan)
    for i, rv in enumerate(new_rows):
        if rv not in ri:
            continue
        for j, cv in enumerate(new_cols):
            if cv in ci:
                out[i, j] = matrix[ri[rv], ci[cv]]
    return out


# ── The core call ────────────────────────────────────────────────────────────


@dataclass
class CorrResult:
    pearson_r: float
    pearson_p: float
    ci_lo: float
    ci_hi: float
    spearman_rho: float
    spearman_p: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


_EMPTY = CorrResult(*([np.nan] * 6), n=0)


def correlate(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    mask: np.ndarray | None = None,
    n_boot: int = 5000,
    seed: int = 0,
) -> CorrResult:
    """Pearson + Spearman over the masked, finite-in-both cells.

    ``mask`` selects cells (True = keep); cells NaN in either matrix are
    always dropped on top of it. The 95% CI is a percentile bootstrap of the
    Pearson r over cells (``n_boot`` resamples); pass ``n_boot=0`` to skip.
    """
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape} — reindex first")
    keep = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    x, y = a[keep], b[keep]
    if len(x) < 3:
        return _EMPTY
    r, rp = stats.pearsonr(x, y)
    rho, sp = stats.spearmanr(x, y)
    lo = hi = np.nan
    if n_boot:
        rng = np.random.default_rng(seed)
        # Constant draws are an expected feature of bootstrap sampling, not a
        # warning-worthy condition for the caller.  They are filtered below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            boot = np.asarray([
                stats.pearsonr(x[idx], y[idx])[0]
                for idx in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))
            ])
        # Small matrices can yield a constant bootstrap draw (for example,
        # choosing the same one of three cells repeatedly).  scipy correctly
        # reports that draw's r as NaN; it should not poison an otherwise
        # valid CI.  The historical large grids virtually never hit this, but
        # filtering preserves the percentile-bootstrap definition for every
        # finite resample and makes the public helper useful on smoke grids.
        boot = boot[np.isfinite(boot)]
        if len(boot):
            lo, hi = np.percentile(boot, [2.5, 97.5])
    return CorrResult(float(r), float(rp), float(lo), float(hi),
                      float(rho), float(sp), int(len(x)))


def spearman_brown(r_half: float) -> float:
    """Full-length reliability from a split-half correlation: ``2r/(1+r)``."""
    return 2 * r_half / (1 + r_half) if (1 + r_half) != 0 else float("nan")


def paired_diff(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "spearman",
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Test whether ``pred_a`` beats ``pred_b`` at predicting ``target``.

    Two point estimates side by side are not a comparison: the two
    correlations share the same target cells, so their sampling errors are
    correlated and the difference has a much tighter distribution than either
    margin suggests. This resamples *cells* (the shared unit) with
    replacement and recomputes both correlations on each draw, giving a CI on
    ``rho_a - rho_b`` and a two-sided bootstrap p.

    All three matrices must already share a frame; cells non-finite in any of
    them are dropped, so both predictors are scored on identical support.
    """
    a, b, t = (np.asarray(m, dtype=float) for m in (pred_a, pred_b, target))
    if not (a.shape == b.shape == t.shape):
        raise ValueError(f"shape mismatch: {a.shape}, {b.shape}, {t.shape} — reindex first")
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    x, y, z = a[keep], b[keep], t[keep]
    if len(z) < 3:
        return {"method": method, "n": int(len(z)), "diff": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                "corr_a": np.nan, "corr_b": np.nan}

    corr = (lambda u, v: stats.spearmanr(u, v)[0]) if method == "spearman" \
        else (lambda u, v: stats.pearsonr(u, v)[0])
    obs_a, obs_b = corr(x, z), corr(y, z)

    rng = np.random.default_rng(seed)
    diffs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        for _ in range(n_boot):
            idx = rng.integers(0, len(z), len(z))
            d = corr(x[idx], z[idx]) - corr(y[idx], z[idx])
            if np.isfinite(d):
                diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan, np.nan))
    # Two-sided: how often the resampled difference lands on the other side of
    # zero from the observed one.
    p = (2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan)
    return {
        "method": method,
        "n": int(len(z)),
        "corr_a": float(obs_a),
        "corr_b": float(obs_b),
        "diff": float(obs_a - obs_b),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": float(min(p, 1.0)),
    }


# ── Leaderboards ─────────────────────────────────────────────────────────────


def leaderboard(
    target: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str],
    predictors: Mapping[str, tuple[np.ndarray, Sequence[str], Sequence[str]]],
    off_diag_only: bool = True,
    n_boot: int = 5000,
    seed: int = 0,
    ceiling: float | None = None,
) -> list[dict]:
    """Correlate every predictor against one target, on the target's frame.

    ``predictors`` maps display name -> (matrix, rows, cols); each is
    reindexed onto the target's (rows, cols) before the shared mask is
    applied. ``ceiling`` (e.g. a Spearman–Brown split-half reliability) adds a
    ``ceiling_frac = pearson_r / ceiling`` column. Rows come back in input
    order — sort by whatever column the analysis ranks on.
    """
    mask = offdiag_mask(rows, cols) if off_diag_only else None
    out = []
    for name, (matrix, p_rows, p_cols) in predictors.items():
        aligned = reindex(matrix, p_rows, p_cols, rows, cols)
        res = correlate(aligned, target, mask, n_boot=n_boot, seed=seed)
        entry = {"predictor": name, **res.as_dict()}
        if ceiling is not None:
            entry["ceiling"] = ceiling
            entry["ceiling_frac"] = (
                res.pearson_r / ceiling if np.isfinite(res.pearson_r) and ceiling else np.nan
            )
        out.append(entry)
    return out


def format_leaderboard(entries: Sequence[Mapping], ceiling: float | None = None) -> str:
    """Fixed-width table in the predict_dpo house style."""
    width = max([len("predictor")] + [len(e["predictor"]) for e in entries]) + 2
    header = (f"{'predictor':{width}s} {'Pearson r':>10s} {'95% CI':>18s} "
              f"{'Spearman':>9s} {'n':>5s}")
    if ceiling is not None:
        header += f" {'r/ceil':>7s}"
    lines = [header]
    for e in entries:
        line = (f"{e['predictor']:{width}s} {e['pearson_r']:+10.3f}   "
                f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]   "
                f"{e['spearman_rho']:+9.3f}  {e['n']:5d}")
        if ceiling is not None:
            frac = e.get("ceiling_frac", np.nan)
            line += f" {frac:7.2f}" if np.isfinite(frac) else f" {'—':>7s}"
        lines.append(line)
    return "\n".join(lines)


# ── Persona layers ───────────────────────────────────────────────────────────
#
# There is deliberately no `best_layer_by_corr` here. Choosing the persona layer
# whose cosine grid best correlates with the ground-truth matrix, and then
# reporting that correlation, fits the target: the number stops being a
# prediction. Layers are chosen GT-independently by the steer-and-judge sweep on
# a held-out eval pool (`predictors.persona.run_sweep` / `best_layer`).


def sim_at_layer(vectors: Mapping[str, np.ndarray], values: Sequence[str], layer: int) -> np.ndarray:
    """Cosine grid over ``values`` at one layer of stacked vectors (NaN-padded
    for values without a vector)."""
    from valuegen.predictors.store import cosine_grid

    sliced = {v: vec[layer] for v, vec in vectors.items()}
    return cosine_grid(sliced, list(values))


# ── Matrix resolution (CLI sugar) ────────────────────────────────────────────


def resolve_matrix(
    spec: str,
    cluster=None,
    model: str | None = None,
    data_method: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], str]:
    """Resolve a matrix spec to ``(matrix, rows, cols, label)``.

    ``spec`` is either a path to a ``.npy`` (with its sibling
    ``*_values.json``) or predictor sugar
    ``predictor[:data_method[:model[:name]]]`` resolved through the store —
    e.g. ``persona``, ``persona:default_llm:allenai/OLMo-2-1124-7B-SFT``, or
    ``persona:default_llm:...:persona_response_avg_diff_L15``. Omitted parts
    fall back to the ``model`` / ``data_method`` arguments, then to the
    predictor's registered default data method. When the keyed similarity dir
    holds a single ``.npy`` the trailing name is optional.
    """
    path = Path(spec)
    if spec.endswith(".npy") or path.is_file():
        from valuegen.ground_truth.matrices import load_matrix

        matrix, rows, cols = load_matrix(path)
        return matrix, rows, cols, path.stem

    from valuegen.predictors import PREDICTORS
    from valuegen.predictors import store

    parts = spec.split(":")
    name = parts[0]
    if name not in PREDICTORS:
        raise FileNotFoundError(
            f"{spec!r} is neither a matrix file nor a predictor name "
            f"({sorted(PREDICTORS)})"
        )
    if cluster is None:
        from valuegen.config import load_cluster

        cluster = load_cluster()
    data_method = (parts[1] if len(parts) > 1 and parts[1] else None) \
        or data_method or PREDICTORS[name].default_data
    model = (parts[2] if len(parts) > 2 and parts[2] else None) or model
    sim_name = parts[3] if len(parts) > 3 and parts[3] else None
    if not model:
        raise SystemExit(
            f"--pred {spec!r} needs a model to key the store "
            "(pass --model, or spell it predictor:data_method:model)"
        )
    sim_dir = store.similarity_dir(cluster, name, data_method, model)
    # A matrix without provenance is not a result: the sidecar is
    # what distinguishes a registered matrix from a fork's raw output, which
    # some predictors have the fork write into the same directory.
    candidates = [
        p for p in sorted(sim_dir.glob("*/*.npy"))
        if p.with_name(f"{p.stem}_provenance.yaml").is_file()
    ]
    if sim_name:
        candidates = [
            p for p in candidates
            if p.stem == sim_name or str(p.relative_to(sim_dir))[: -len(".npy")] == sim_name
        ]
    if not candidates:
        raise FileNotFoundError(
            f"no similarity matrix under {sim_dir}"
            + (f" matching {sim_name!r}" if sim_name else "")
            + f" — build it with `valuegen predict -m {name} ...`"
        )
    if len(candidates) > 1:
        rels = [str(p.relative_to(sim_dir))[: -len(".npy")] for p in candidates]
        raise SystemExit(
            f"{sim_dir} holds several matrices ({rels}); "
            f"disambiguate as {name}:{data_method}:{model}:<name> "
            "(name may include the artifact-ID directory)"
        )
    matrix, values = store.load_similarity(candidates[0])
    label = f"{name}/{data_method}/{store._short(model)}/{candidates[0].stem}"
    return matrix, values, values, label
