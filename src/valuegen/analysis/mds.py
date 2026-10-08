"""Dissimilarities and MDS maps of similarity matrices.

Clustering and the scores that judge it live in :mod:`valuegen.analysis.cluster`;
this module only turns a similarity grid into distances and into a 2D layout.

The numeric conventions (seeds, n_init, baseline sampling) are fixed so maps
reproduce exactly.

Dissimilarities are the **raw** ``1 − cos`` (:func:`cos_to_dist`); nothing
rescales, standardizes, or de-identities them before SMACOF. Read the maps
accordingly: weight-steering task vectors are near-orthogonal, so ``1 − cos``
is ≈0.97 for nearly every off-diagonal pair, and metric SMACOF spends its
layout budget reproducing that constant baseline rather than the small
fluctuations that carry the relational signal. ``metric_raw`` on such a grid
is expected to look like a structureless blob at a deceptively low stress —
reproducing a constant is easy. ``nonmetric`` fits its own monotone transform
and so keeps only the rank order of pairs.

See :mod:`valuegen.analysis.cluster` for the silhouette-vs-k *no-clusters*
check: a curve at or below its random baseline means the geometry is a
continuous gradient, so read the maps as relational layouts, not blobs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# ── Dissimilarities ──────────────────────────────────────────────────────────


def offdiag_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def cos_to_dist(sim: np.ndarray) -> np.ndarray:
    """``1 − cos``, symmetrized, clipped at 0, zero diagonal."""
    d = 1.0 - np.asarray(sim, dtype=float)
    np.fill_diagonal(d, 0.0)
    return np.clip(0.5 * (d + d.T), 0.0, None)


def drop_nan_values(sim: np.ndarray, values: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Return the largest finite induced similarity grid.

    Predictor stores NaN-pad an unavailable value in *both* its row and
    column.  Therefore testing whether an original row is wholly finite would
    incorrectly discard every available row as well: each contains the
    unavailable value's NaN column.  Start with finite-diagonal values and
    repeatedly retain only rows that are finite within that candidate frame.
    This drops entries such as ``ad_12`` while preserving the complete grid
    over the remaining values before an embedding is attempted.
    """
    sim = np.asarray(sim, dtype=float)
    values = list(values)
    if sim.shape != (len(values), len(values)):
        raise ValueError("similarity matrix must be square and match values")
    keep = [i for i in range(len(values)) if np.isfinite(sim[i, i])]
    while keep:
        finite_rows = np.isfinite(sim[np.ix_(keep, keep)]).all(axis=1)
        if finite_rows.all():
            break
        keep = [i for i, finite in zip(keep, finite_rows) if finite]
    return sim[np.ix_(keep, keep)], [values[i] for i in keep]


# ── Embeddings ───────────────────────────────────────────────────────────────


def mds_coords(
    dist: np.ndarray,
    metric: bool = True,
    seed: int = 42,
    n_init: int = 8,
    max_iter: int = 500,
) -> tuple[np.ndarray, float]:
    """2D SMACOF on a precomputed dissimilarity → (coords, stress)."""
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              metric=metric, normalized_stress="auto", n_init=n_init,
              max_iter=max_iter)
    coords = mds.fit_transform(dist)
    return coords, float(mds.stress_)


MDS_VARIANTS = ("metric_raw", "nonmetric")


def mds_variant(sim: np.ndarray, variant: str, seed: int = 42) -> tuple[np.ndarray, float]:
    """SMACOF on the raw ``1 − cos`` dissimilarity, metric or nonmetric."""
    if variant == "metric_raw":
        return mds_coords(cos_to_dist(sim), metric=True, seed=seed)
    if variant == "nonmetric":
        return mds_coords(cos_to_dist(sim), metric=False, seed=seed)
    raise ValueError(f"unknown MDS variant {variant!r}; known: {MDS_VARIANTS}")
