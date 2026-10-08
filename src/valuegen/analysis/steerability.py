"""The paper's steerability analyses: the ds metric, the symmetric ceiling,
the cross-arm aggregate, movement geometry and the taxonomy z test.

Ported from the analysis notebooks that produced the paper's numbers
(``0917/7b-final/make_plots.py`` for RQ1,
``0924/iclr_plots/rq3_functional_taxonomy`` for RQ3); the numerics are kept
identical, including RNG call order, so ``paper/reproduce`` matches the
published values exactly.

**ds metric.** A GT run stores two variants of the Likert choice rate: ``raw``
(``f``, the steered model's rate) and ``normalized``, which is ``ds``: the
change from the base rate ``b`` scaled by the room it had to move, upward
moves by ``1 - b`` and downward moves by ``b``, so every cell lies in
``[-1, 1]``. ``b`` is recovered from the pair and must be constant down each
column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .correlate import offdiag_mask

# ── Loading ──────────────────────────────────────────────────────────────────


def load_values(path: str | Path) -> tuple[list[str], list[str]]:
    """(rows, cols) from a ``*_values.json``: a ``{"rows", "cols"}`` dict, or
    a flat list for a square grid."""
    v = json.loads(Path(path).read_text())
    if isinstance(v, dict):
        return list(v["rows"]), list(v["cols"])
    return list(v), list(v)


def base_rate(raw: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    """The unsteered base rate per column, shape ``(1, n_cols)``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(normalized >= 0, (raw - normalized) / (1 - normalized), raw / (1 + normalized))
    if np.nanstd(b, axis=0).max() >= 1e-9:
        raise ValueError("base rate is not column-constant: raw/normalized are not a pair")
    return np.nanmean(b, axis=0)[None, :]


def load_target(stem: str | Path, metric: str = "ds") -> tuple[np.ndarray, list[str], list[str]]:
    """A GT matrix under ``metric`` (``ds`` or ``diff``) from its stem, i.e.
    the matrix path without ``_raw.npy`` / ``_normalized.npy``."""
    stem = str(stem)
    raw = np.load(stem + "_raw.npy")
    norm = np.load(stem + "_normalized.npy")
    if metric == "ds":
        m = norm
    elif metric == "diff":
        m = raw - base_rate(raw, norm)
    else:
        raise ValueError(f"metric {metric!r}; expected ds or diff")
    rows, cols = load_values(stem + "_raw_values.json")
    return m, rows, cols


# ── RQ1: off-diagonal rank correlation, ceiling, aggregate ───────────────────


def offdiag_rho(target: np.ndarray, pred: np.ndarray, rows: Sequence[str],
                cols: Sequence[str]) -> float:
    """Spearman ρ over off-diagonal cells finite in both matrices."""
    m = offdiag_mask(rows, cols) & np.isfinite(target) & np.isfinite(pred)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(rankdata(target[m]), rankdata(pred[m]))[0, 1])


def symmetric_part(m: np.ndarray, rows: Sequence[str], cols: Sequence[str]) -> np.ndarray:
    """``(M + Mᵀ)/2`` over value pairs measured in both directions; a cell whose
    reverse was not measured (a column that is not a trained row) is kept."""
    s = m.copy()
    ri = {v: i for i, v in enumerate(rows)}
    ci = {v: j for j, v in enumerate(cols)}
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            if a != b and b in ri and a in ci:
                s[i, j] = (m[i, j] + m[ri[b], ci[a]]) / 2
    return s


def symmetric_ceiling(m: np.ndarray, rows: Sequence[str], cols: Sequence[str]) -> float:
    """The best off-diagonal ρ any symmetric grid (e.g. a cosine predictor) can
    reach: ρ between the target and its symmetric part. Under off-diagonal
    Spearman the other cosine-grid constraints are vacuous (``I + εA`` realizes
    any symmetric ``A``'s ranks). In-sample, so slightly optimistic."""
    return offdiag_rho(m, symmetric_part(m, rows, cols), rows, cols)


def _rho_on_rows(t, p, off, idx):
    ts, ps, ms = t[idx], p[idx], off[idx]
    m = ms & np.isfinite(ts) & np.isfinite(ps)
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(rankdata(ts[m]), rankdata(ps[m]))[0, 1]


def aggregate_bootstrap(
    targets: Mapping[str, np.ndarray],
    preds: Mapping[tuple[str, str], np.ndarray | None],
    off: np.ndarray,
    pred_order: Sequence[str],
    arm_order: Sequence[str],
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[dict, dict, dict]:
    """Per predictor, the mean off-diagonal ρ across arms with a joint row
    bootstrap (the same resampled rows in every arm, so cross-arm dependence
    is kept). ``preds[(p, arm)]`` may be None; that arm drops out of p's mean.
    Returns (point, (lo, hi) 95% percentile CI, number of arms covered)."""
    n_rows = off.shape[0]
    rng = np.random.default_rng(seed)
    full = np.arange(n_rows)

    def arms_for(p):
        return [a for a in arm_order if preds.get((p, a)) is not None]

    def mean_over(p, idx):
        vals = [_rho_on_rows(targets[a], preds[(p, a)], off, idx) for a in arms_for(p)]
        return np.nanmean(vals) if vals else np.nan

    point = {p: float(mean_over(p, full)) for p in pred_order}
    n_arms = {p: len(arms_for(p)) for p in pred_order}
    boot = {p: np.empty(n_boot) for p in pred_order}
    for b in range(n_boot):
        idx = rng.integers(0, n_rows, n_rows)
        for p in pred_order:
            boot[p][b] = mean_over(p, idx)
    ci = {p: (float(np.nanpercentile(boot[p], 2.5)), float(np.nanpercentile(boot[p], 97.5)))
          for p in pred_order}
    return point, ci, n_arms


# ── RQ3: movement geometry and the taxonomy z test ───────────────────────────


def movement_distance(m: np.ndarray, rows: Sequence[str], cols: Sequence[str],
                      keep_rows: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Cosine distance between steered rows' co-movement profiles.

    The self-steer cell is dropped, then each column's mean (the column
    effect) is removed, estimated on *all* rows; only then are the rows
    restricted to ``keep_rows`` (in that order), so an arm with extra rows
    still centers on its own full set."""
    x = m.astype(float).copy()
    cset = {c: k for k, c in enumerate(cols)}
    for i, r in enumerate(rows):
        if r in cset:
            x[i, cset[r]] = np.nan
    x = x - np.nanmean(x, axis=0, keepdims=True)
    rows = list(rows)
    if keep_rows is None:
        keep_rows = rows
    idx = [rows.index(r) for r in keep_rows]
    xs = x[idx]
    n = len(idx)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ok = np.isfinite(xs[i]) & np.isfinite(xs[j])
            a, b = xs[i, ok], xs[j, ok]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            d[i, j] = d[j, i] = 1.0 - float(a @ b) / (na * nb) if na > 0 and nb > 0 else 1.0
    return d, list(keep_rows)


def label_vector(assign: Mapping[str, object], rows: Sequence[str]) -> np.ndarray:
    """Integer labels for ``rows``, classes numbered in sorted-by-str order."""
    uniq = {c: i for i, c in enumerate(sorted({assign[r] for r in rows}, key=str))}
    return np.array([uniq[assign[r]] for r in rows])


def taxonomy_z(labels: np.ndarray, dists: Mapping[str, np.ndarray],
               arm_order: Sequence[str], n_perm: int = 5000, seed: int = 0) -> dict[str, float]:
    """How much better than chance a labeling groups values that co-move.

    Per arm: z of the precomputed-distance silhouette against ``n_perm``
    size-matched relabelings. Aggregate (``"agg"``): the *same* relabelings
    are applied to every arm and the null is the cross-arm mean of the per-arm
    standardized silhouettes, which absorbs the arms' strong correlation (a
    correlation-corrected Stouffer, not the naive mean of z's)."""
    from sklearn.metrics import silhouette_score

    n = len(labels)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(n_perm)]
    per_z, null_z = [], []
    for arm in arm_order:
        d = dists[arm]
        obs = silhouette_score(d, labels, metric="precomputed")
        nl = np.array([silhouette_score(d, labels[p], metric="precomputed") for p in perms])
        per_z.append((obs - nl.mean()) / nl.std())
        null_z.append((nl - nl.mean()) / nl.std())
    per_z = np.array(per_z)
    agg_null = np.array(null_z).mean(0)
    z_agg = (per_z.mean() - agg_null.mean()) / agg_null.std()
    return {arm: float(z) for arm, z in zip(arm_order, per_z)} | {"agg": float(z_agg)}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def transport_labels(assign: Mapping[str, int], source_vecs: Mapping[str, np.ndarray],
                     target_vecs: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Carry a clustering of source values onto target values: each target
    joins the cluster whose (unit-normalized mean) centroid it is most
    cosine-similar to."""
    cl_ids = sorted(set(assign.values()))
    cmat = np.stack([
        _unit(np.stack([np.asarray(source_vecs[v], dtype=np.float64)
                        for v, c in assign.items() if c == cl]).mean(0))
        for cl in cl_ids
    ])
    return {t: cl_ids[int(np.argmax(cmat @ _unit(np.asarray(target_vecs[t], dtype=np.float64))))]
            for t in target_vecs}


# ── Cross-model agreement of similarity geometry ─────────────────────────────


def rank_matrix(d: np.ndarray) -> np.ndarray:
    """Symmetric matrix of the upper-triangle entries' ranks (diagonal 0)."""
    iu = np.triu_indices(len(d), 1)
    r = np.zeros_like(d, dtype=float)
    r[iu] = rankdata(d[iu])
    return r + r.T


def mantel(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
           n_perm: int = 2000) -> tuple[float, float]:
    """Spearman Mantel statistic of two distance matrices, permuting ``b``'s
    rows and columns jointly; one-sided p (null ≥ observed). ``rng`` is
    passed in so successive pairs share one stream."""
    n = len(a)
    iu = np.triu_indices(n, 1)

    def z(x):
        return (x - x.mean()) / x.std()

    za = z(rank_matrix(a)[iu])
    rb = rank_matrix(b)

    def stat(x):
        return float(za @ z(x) / len(za))

    obs = stat(rb[iu])
    null = np.array([stat(rb[p][:, p][iu]) for p in (rng.permutation(n) for _ in range(n_perm))])
    return obs, (1 + int((null >= obs).sum())) / (n_perm + 1)


# ── Scenario bootstrap (eval-sampling uncertainty) ───────────────────────────


def rate_operator(evals, n_scenarios: int, scenario_values, rows: Sequence[str],
                  cols: Sequence[str]) -> dict:
    """Sparse operator so that ``rates(op, W)`` gives every model's Likert
    choice rate per column under scenario weights ``W`` [b, S].

    ``evals`` is one GT run's compact table (columns model, scenario, likert;
    models ``"base"`` then ``rows``); ``scenario_values`` the per-scenario
    ``(value1, value2)`` frame. For column ``j`` the rate is the weighted mean
    over scenarios that pit ``j`` against another value of the score aligned
    with ``j``: ``(1 - likert)/2`` when ``j`` is value1, ``(1 + likert)/2``
    when it is value2; NaN Likerts are dropped. Unit weights reproduce the
    published ``_raw`` matrix and base rate."""
    import scipy.sparse as sp

    cidx = {c: k for k, c in enumerate(cols)}
    j_by = {col: scenario_values[col].map(cidx) for col in ("value1", "value2")}
    s_len, c_len = n_scenarios, len(cols)
    ps, qs = [], []
    for m in ["base", *rows]:
        df = evals[evals.model == m]
        if len(df) != s_len:
            raise ValueError(f"model {m!r}: {len(df)} scenarios, expected {s_len}")
        s = df.scenario.to_numpy()
        lik = df.likert.to_numpy(dtype=float)
        ok = ~np.isnan(lik)
        r_, c_, pv = [], [], []
        for col, score in (("value1", (-lik + 1) / 2), ("value2", (lik + 1) / 2)):
            j = j_by[col].to_numpy()[s]
            keep = ok & ~np.isnan(j.astype(float))
            r_.append(s[keep])
            c_.append(j[keep].astype(int))
            pv.append(score[keep])
        r_, c_ = np.concatenate(r_), np.concatenate(c_)
        ps.append(sp.csr_matrix((np.concatenate(pv), (r_, c_)), shape=(s_len, c_len)))
        qs.append(sp.csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(s_len, c_len)))
    return {"P": sp.hstack(ps).tocsc(), "Q": sp.hstack(qs).tocsc(),
            "R": len(rows), "C": c_len, "rows": list(rows), "cols": list(cols)}


def rates(op: dict, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scenario weights ``w`` [b, S] → (base [b, C], raw [b, R, C])."""
    num, den = np.asarray(w @ op["P"]), np.asarray(w @ op["Q"])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (num / den).reshape(len(w), op["R"] + 1, op["C"])
    return r[:, 0], r[:, 1:]


def ds_from_base(raw: np.ndarray, base: np.ndarray) -> np.ndarray:
    b = base[None, :]
    return np.where(raw >= b, (raw - b) / (1 - b), (raw - b) / b)


def movement_distance_fast(m: np.ndarray, rows: Sequence[str], cols: Sequence[str],
                           keep_idx: Sequence[int]) -> np.ndarray:
    """Vectorized :func:`movement_distance` (equal to ~1e-10) for bootstraps."""
    cset = {c: k for k, c in enumerate(cols)}
    x = m.astype(float).copy()
    for i, r in enumerate(rows):
        if r in cset:
            x[i, cset[r]] = np.nan
    x = (x - np.nanmean(x, axis=0, keepdims=True))[list(keep_idx)]
    fin = np.isfinite(x)
    x0 = np.where(fin, x, 0.0)
    dot = x0 @ x0.T
    sq = x0 ** 2
    na2 = sq @ fin.T.astype(float)
    nb2 = fin.astype(float) @ sq.T
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1.0 - dot / np.sqrt(na2 * nb2)
    np.fill_diagonal(d, 0.0)
    return d


def silhouette_batch(d: np.ndarray, labs: np.ndarray) -> np.ndarray:
    """Mean silhouette for each labeling in ``labs`` [P, n] on precomputed
    distances ``d``; matches sklearn (singleton clusters score 0)."""
    p_, n = labs.shape
    k = int(labs.max()) + 1
    oh = np.zeros((p_, n, k))
    oh[np.arange(p_)[:, None], np.arange(n)[None, :], labs] = 1.0
    sums = np.einsum("ij,pjk->pik", d, oh)
    sizes = oh.sum(1)
    own_size = np.take_along_axis(sizes, labs, 1)
    own_sum = np.take_along_axis(sums, labs[..., None], 2)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = own_sum / (own_size - 1)
        mean_other = sums / sizes[:, None, :]
    mean_other[oh.astype(bool)] = np.inf
    mean_other[np.broadcast_to(sizes[:, None, :] == 0, mean_other.shape)] = np.inf
    b = mean_other.min(2)
    s = (b - a) / np.maximum(a, b)
    s[own_size == 1] = 0.0
    return s.mean(1)


def bootstrap_taxonomy_z(ops: Mapping[str, dict], keep_rows: Sequence[str],
                         labels: Mapping[str, np.ndarray], arm_order: Sequence[str],
                         n_boot: int = 1000, n_perm: int = 1000, chunk: int = 50,
                         seed: int = 0, progress=None) -> dict[str, dict[str, np.ndarray]]:
    """Scenario bootstrap of :func:`taxonomy_z`: each replicate draws one
    multinomial reweighting of the scenarios, shared by every model of every
    arm (they were evaluated on the same pool, so base/steered stay paired),
    rebuilds each arm's ds co-movement distances and rescores every labeling
    against its own ``n_perm`` relabelings (shared across arms). Returns
    ``{taxonomy: {arm | "agg": z per replicate}}``."""
    s_len = ops[arm_order[0]]["P"].shape[0]
    idx = {a: [ops[a]["rows"].index(r) for r in keep_rows] for a in arm_order}
    rng = np.random.default_rng(seed)
    n = len(keep_rows)
    cols = [*arm_order, "agg"]
    out = {t: {c: [] for c in cols} for t in labels}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        w = rng.multinomial(s_len, np.full(s_len, 1.0 / s_len), size=b).astype(float)
        per_arm = {a: rates(ops[a], w) for a in arm_order}
        for k in range(b):
            ds_ = {a: movement_distance_fast(ds_from_base(per_arm[a][1][k], per_arm[a][0][k]),
                                             ops[a]["rows"], ops[a]["cols"], idx[a])
                   for a in arm_order}
            perms = np.stack([rng.permutation(n) for _ in range(n_perm)])
            for t, lab in labels.items():
                per_z, null_z = [], []
                for a in arm_order:
                    obs = silhouette_batch(ds_[a], lab[None])[0]
                    nl = silhouette_batch(ds_[a], lab[perms])
                    per_z.append((obs - nl.mean()) / nl.std())
                    null_z.append((nl - nl.mean()) / nl.std())
                for a, z in zip(arm_order, per_z):
                    out[t][a].append(z)
                out[t]["agg"].append(np.mean(per_z) / np.array(null_z).mean(0).std())
        done += b
        if progress:
            progress(done, n_boot)
    return {t: {c: np.array(v) for c, v in d.items()} for t, d in out.items()}
