"""Clustering on precomputed distances, and the scores that judge it.

Three clustering algorithms:

``silhouette_curve``            kmeans on 2D MDS coordinates
``silhouette_curve_upgma``      UPGMA (average linkage) on the raw distances
``silhouette_curve_kmedoids``   k-medoids on the raw distances

and three ways to score a partition, which do different jobs:

``silhouette``   average similarity of items within a cluster to all other items in the same cluster.
``silhouette_z`` silhouette in sds above that grid's own random partition.
``hubert_gamma`` rank correlation of pair distance against a different cluster.

The baseline must match the clustering method: :func:`random_baseline` 
for coordinate methods, :func:`random_partition_stats` for methods on a
precomputed distance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def silhouette_curve(coords: np.ndarray, ks: Sequence[int], seed: int = 42) -> list[float]:
    """KMeans-on-2D-coords silhouette per k (the WS-MDS diagnostic)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    return [
        float(silhouette_score(
            coords,
            KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(coords),
        ))
        for k in ks
    ]


def silhouette_curve_upgma(dist: np.ndarray, ks: Sequence[int]) -> list[float]:
    """UPGMA on a precomputed distance matrix — the multimodel variant, which
    skips the MDS step entirely.

    UPGMA (Unweighted Pair Group Method with Arithmetic mean; Sokal & Michener
    1958) is hierarchical agglomerative clustering with *average* linkage: start
    with every item its own cluster, repeatedly merge the two closest, stop at
    k. "Closest" between two groups is the mean distance over all cross-pairs.
    sklearn spells it ``linkage="average"``.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    out = []
    for k in ks:
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(dist)
        out.append(float(silhouette_score(dist, labels, metric="precomputed")))
    return out


def kmedoids_partition(
    dist: np.ndarray, k: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """FasterPAM on a precomputed distance → (labels, medoid indices).

    A medoid is one of the items, unlike a k-means centroid, which is a point in
    space that corresponds to no value. Anything downstream that has to *name* a
    representative value per cluster wants these indices.
    """
    import kmedoids as _km

    result = _km.fasterpam(np.asarray(dist, dtype=float), k, random_state=seed)
    return np.asarray(result.labels), np.asarray(result.medoids)


def silhouette_curve_kmedoids(
    dist: np.ndarray, ks: Sequence[int], seed: int = 0
) -> list[float]:
    """k-medoids silhouette per k on a precomputed distance — same null as
    :func:`silhouette_curve_upgma` (:func:`random_partition_stats`)."""
    from sklearn.metrics import silhouette_score

    return [
        float(silhouette_score(dist, kmedoids_partition(dist, k, seed)[0],
                               metric="precomputed"))
        for k in ks
    ]


def random_baseline(
    n: int, ks: Sequence[int], n_rand: int = 500, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± sd silhouette of ``n`` uniform points in the unit square per k
    (chance band for :func:`silhouette_curve`)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    means, sds = [], []
    for k in ks:
        scores = [
            silhouette_score(
                X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
            )
            for X in (rng.uniform(0, 1, (n, 2)) for _ in range(n_rand))
        ]
        means.append(np.mean(scores))
        sds.append(np.std(scores))
    return np.array(means), np.array(sds)


def random_partition(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random labeling of ``n`` items into exactly ``k`` non-empty clusters."""
    while True:
        lab = rng.integers(0, k, size=n)
        if len(np.unique(lab)) == k:
            return lab


def random_partition_stats(
    dist: np.ndarray, ks: Sequence[int], n_rand: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, sd and 95th percentile of random-partition silhouettes per k.

    The sd is what :func:`silhouette_z` needs; :func:`random_partition_baseline`
    wraps this for the plot band.
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    means, sds, p95s = [], [], []
    for k in ks:
        scores = [
            silhouette_score(dist, random_partition(n, k, rng), metric="precomputed")
            for _ in range(n_rand)
        ]
        means.append(np.mean(scores))
        sds.append(np.std(scores))
        p95s.append(np.percentile(scores, 95))
    return np.array(means), np.array(sds), np.array(p95s)


def relabel_null_stats(
    dist: np.ndarray, labels: Sequence[int], n_rand: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Mean and sd of the silhouette under the random label hypothesis: the
    observed ``labels`` are permuted ``n_rand`` times with ``dist`` fixed, so
    every null draw keeps the partition's cluster sizes.

    Unlike :func:`random_partition_stats` (uniform k-labelings, one null per k)
    this null is specific to the partition, so an unbalanced partition is not
    flattered by a comparison against roughly balanced random ones. Feed the
    pair to :func:`score_partition`.
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    scores = [
        silhouette_score(dist, rng.permutation(lab), metric="precomputed")
        for _ in range(n_rand)
    ]
    return float(np.mean(scores)), float(np.std(scores))


def random_partition_baseline(
    dist: np.ndarray, ks: Sequence[int], n_rand: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 95th-percentile silhouette of random partitions on a
    precomputed distance (chance band for
    :func:`silhouette_curve_upgma`)."""
    mean, _, p95 = random_partition_stats(dist, ks, n_rand=n_rand, seed=seed)
    return mean, p95


def silhouette_z(
    sils: Sequence[float], null_mean: np.ndarray, null_sd: np.ndarray
) -> np.ndarray:
    """Silhouette in sds above its own chance mean.

    ``null_mean``/``null_sd`` come from :func:`random_baseline` (coords) or
    :func:`random_partition_stats` (precomputed distances) — the null must match
    the clustering method, as in :func:`silhouette_curve` (kmeans on MDS coords)
    vs :func:`silhouette_curve_upgma` (UPGMA on raw distances).
    """
    return (np.asarray(sils, dtype=float) - null_mean) / null_sd


def hubert_gamma(
    dist: np.ndarray, labels: Sequence[int], rank: bool = True
) -> float:
    """Hubert's Γ: correlation between pair distance and "are they in different
    clusters", over all off-diagonal pairs.

    ``rank=True`` (default) replaces the distances by their ranks first. This
    matters: the plain Pearson form is invariant to an affine rescaling of the
    distances but *not* to a monotone reshaping of them. Holding the clustering
    and the rank order of distances fixed on the qwen3 persona grid, raw
    silhouette spans 0.061-0.534 and Pearson Γ spans 0.400-0.573 across
    d -> 2d+5, d^2, sqrt(d), d^4, exp(3d)-1 — while the ranked form is exactly
    0.580 throughout. Only the ranked form is safe for comparing grids whose
    distances are distributed differently, which is the usual case here
    (weight-steering task vectors are near-orthogonal, MCQ profiles bunch near
    zero). Pass ``rank=False`` only to reproduce the Pearson variant.
    """
    lab = np.asarray(labels)
    iu = np.triu_indices(len(lab), 1)
    diff = (lab[:, None] != lab[None, :]).astype(float)[iu]
    if diff.min() == diff.max():
        return float("nan")            # one cluster: no between-pairs
    d = np.asarray(dist, dtype=float)[iu]
    if rank:
        from scipy.stats import rankdata

        d = rankdata(d)
    return float(np.corrcoef(d, diff)[0, 1])


def gamma_curve_upgma(
    dist: np.ndarray, ks: Sequence[int], linkage: str = "average",
    rank: bool = True,
) -> list[float]:
    """Hubert's Γ per k, same UPGMA clustering as
    :func:`silhouette_curve_upgma`."""
    from sklearn.cluster import AgglomerativeClustering

    return [
        hubert_gamma(dist, AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage=linkage).fit_predict(dist),
            rank=rank)
        for k in ks
    ]


# ── One entry point per method, so callers need no per-method branching ──────

CLUSTER_METHODS = ("upgma", "kmedoids", "mds", "hdbscan")


def hdbscan_labels(dist: np.ndarray, min_cluster_size: int = 4) -> np.ndarray:
    """HDBSCAN on a precomputed distance; ``-1`` marks noise.

    ``copy=True`` is essential, not cosmetic: with ``metric="precomputed"`` the
    sklearn default (``copy=False``) mutates the distance matrix in place, so
    reusing one matrix across a min_cluster_size sweep silently runs every
    iteration after the first on corrupted input.
    """
    import warnings

    from sklearn.cluster import HDBSCAN

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed",
                       copy=True).fit_predict(np.asarray(dist, dtype=float))


def cluster_labels(
    dist: np.ndarray,
    k: int,
    method: str = "upgma",
    coords: np.ndarray | None = None,
    seed: int = 0,
    min_cluster_size: int = 4,
) -> np.ndarray:
    """Labels from one of :data:`CLUSTER_METHODS`.

    ``upgma`` and ``kmedoids`` cluster ``dist`` directly. ``mds`` clusters
    ``coords`` (2D MDS positions) with kmeans and so needs them passed in — it
    is the only method that sees a flattened view rather than the real
    distances. ``hdbscan`` ignores ``k`` entirely, picks its own count, and can
    return ``-1`` for noise.
    """
    if method == "upgma":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(dist)
    if method == "kmedoids":
        return kmedoids_partition(dist, k, seed=seed)[0]
    if method == "mds":
        from sklearn.cluster import KMeans

        if coords is None:
            raise ValueError("method 'mds' needs coords= (2D MDS positions)")
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(coords)
    if method == "hdbscan":
        return hdbscan_labels(dist, min_cluster_size)
    raise ValueError(f"unknown method {method!r}; known: {CLUSTER_METHODS}")


def medoids_of(dist: np.ndarray, labels: Sequence[int]) -> dict[int, int]:
    """``{cluster: index}`` of each cluster's medoid — the member with the
    smallest total distance to its own cluster-mates.

    A medoid is one of the items, so it reads as the cluster's name. Noise
    (``-1``) is skipped.
    """
    lab = np.asarray(labels)
    out = {}
    for c in sorted(set(lab.tolist())):
        if c == -1:
            continue
        idx = np.where(lab == c)[0]
        out[int(c)] = int(idx[np.argmin(dist[np.ix_(idx, idx)].sum(axis=1))])
    return out


def score_partition(
    dist: np.ndarray,
    labels: Sequence[int],
    null_mean: float | None = None,
    null_sd: float | None = None,
) -> dict[str, float]:
    """``{silhouette, z, gamma, n_clusters, n_noise}`` for one partition.

    Noise points are dropped before scoring, so a method that declines to
    cluster most of the data is graded only on what it kept — which flatters it.
    ``z`` is therefore returned as NaN whenever anything was dropped, because
    the null was drawn over the full set and no longer applies.
    """
    from sklearn.metrics import silhouette_score

    lab = np.asarray(labels)
    real = lab != -1
    n_noise = int((~real).sum())
    n_cl = len(set(lab[real].tolist()))
    out = {"n_clusters": n_cl, "n_noise": n_noise,
           "silhouette": float("nan"), "z": float("nan"), "gamma": float("nan")}
    if n_cl < 2 or real.sum() <= n_cl:
        return out
    ix = np.where(real)[0]
    sub = dist[np.ix_(ix, ix)]
    out["silhouette"] = float(silhouette_score(sub, lab[real], metric="precomputed"))
    out["gamma"] = hubert_gamma(sub, lab[real])
    if null_mean is not None and null_sd is not None and n_noise == 0:
        out["z"] = float((out["silhouette"] - null_mean) / null_sd)
    return out


def upgma_linkage(dist: np.ndarray):
    """SciPy linkage matrix for UPGMA — the tree behind
    :func:`silhouette_curve_upgma`, showing every k at once.

    Cutting it at ``k`` reproduces that function's labels exactly, so the
    dendrogram is a view of the same clustering rather than a second one.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    return linkage(squareform(np.asarray(dist, dtype=float), checks=False),
                   method="average")
