"""Embedding-geometry metrics over value sets. Pure numpy.

- ``coverage(members, external)``: mean over external items of the max
  cosine similarity to any set member (facility-location coverage of the
  external map by the set).
- ``tightness(members)``: mean pairwise cosine similarity within the set
  (``nan`` for k < 2).

Both take row-vector matrices; :func:`score_pool` evaluates either metric
for many candidate sets at once from the precomputed cosine grids.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

METRICS = ("coverage", "tightness")


def unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def cosine_grid(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``[len(A), len(B)]`` cosine similarities."""
    return unit(A) @ unit(B).T


def coverage(members: np.ndarray, external: np.ndarray) -> float:
    members = np.atleast_2d(members)
    if members.shape[0] == 0:
        return float("nan")
    S = cosine_grid(members, external)  # [k, n_ext]
    return float(S.max(axis=0).mean())


def per_external_max(members: np.ndarray, external: np.ndarray) -> np.ndarray:
    """Per external item, the max cosine to any member (``[n_ext]``)."""
    return cosine_grid(np.atleast_2d(members), external).max(axis=0)


def tightness(members: np.ndarray) -> float:
    members = np.atleast_2d(members)
    k = members.shape[0]
    if k < 2:
        return float("nan")
    G = cosine_grid(members, members)
    off = G[~np.eye(k, dtype=bool)]
    return float(off.mean())


def score_pool(pool: np.ndarray, metric: str, S_ue: np.ndarray | None = None,
               G_uu: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """Score every candidate set in ``pool`` (``[n, k]`` universe indices).

    ``S_ue`` is the universe x external cosine grid (needed for coverage),
    ``G_uu`` the universe x universe grid (needed for tightness).
    """
    pool = np.asarray(pool, dtype=int)
    if pool.ndim != 2:
        raise ValueError("pool must be [n, k]")
    n, k = pool.shape
    out = np.empty(n, dtype=float)
    if metric == "coverage":
        if S_ue is None:
            raise ValueError("coverage needs S_ue")
        for s in range(0, n, chunk):
            block = pool[s:s + chunk]  # [b, k]
            out[s:s + chunk] = S_ue[block].max(axis=1).mean(axis=1)  # [b, k, E] -> [b]
    elif metric == "tightness":
        if G_uu is None:
            raise ValueError("tightness needs G_uu")
        if k < 2:
            out[:] = np.nan
            return out
        mask = ~np.eye(k, dtype=bool)
        for s in range(0, n, chunk):
            block = pool[s:s + chunk]
            sub = G_uu[block[:, :, None], block[:, None, :]]  # [b, k, k]
            out[s:s + chunk] = sub[:, mask].mean(axis=1)
    else:
        raise ValueError(f"unknown metric {metric!r}; known: {METRICS}")
    return out


def set_metrics(values: Sequence[str], universe_names: Sequence[str], U: np.ndarray, E: np.ndarray) -> dict:
    """Both metrics for one named set (rows of ``U`` aligned to ``universe_names``)."""
    pos = {v: i for i, v in enumerate(universe_names)}
    idx = np.asarray([pos[v] for v in values], dtype=int)
    M = U[idx]
    return {"coverage": coverage(M, E), "tightness": tightness(M), "k": int(len(idx))}
