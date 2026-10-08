"""valuegen.analysis.steerability: the paper's RQ1/RQ3 statistics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from valuegen.analysis import steerability as S


def test_normalized_scales_by_room_to_move_and_recovers_base():
    from valuegen.ground_truth.matrices import normalize_cell

    base = np.array([0.2, 0.5])
    raw = np.array([[0.6, 0.25], [0.2, 0.75]])
    norm = np.array([[normalize_cell(f, b) for f, b in zip(row, base)] for row in raw])
    np.testing.assert_allclose(norm, [[0.5, -0.5], [0.0, 0.5]], atol=1e-12)
    np.testing.assert_allclose(norm, S.ds_from_base(raw, base), atol=1e-12)
    np.testing.assert_allclose(S.base_rate(raw, norm), base[None, :], atol=1e-12)


def test_base_rate_rejects_mismatched_pair():
    raw = np.array([[0.6, 0.3], [0.2, 0.9]])
    with pytest.raises(ValueError, match="column-constant"):
        S.base_rate(raw, np.zeros_like(raw) + np.array([[0.1], [0.2]]))


def test_symmetric_part_averages_only_measured_pairs():
    rows, cols = ["a", "b"], ["a", "b", "c"]
    m = np.array([[9.0, 1.0, 5.0], [3.0, 9.0, 7.0]])
    s = S.symmetric_part(m, rows, cols)
    np.testing.assert_allclose(s, [[9.0, 2.0, 5.0], [2.0, 9.0, 7.0]])


def test_symmetric_ceiling_is_one_for_symmetric_target():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    vals = list("abcdef")
    assert S.symmetric_ceiling(a + a.T, vals, vals) == pytest.approx(1.0)


def test_movement_distance_fast_matches_reference():
    rng = np.random.default_rng(1)
    rows = [f"v{i}" for i in range(7)]
    cols = rows + ["x", "y"]
    m = rng.normal(size=(7, 9))
    keep = ["v3", "v0", "v5", "v6"]
    d, order = S.movement_distance(m, rows, cols, keep_rows=keep)
    assert order == keep
    fast = S.movement_distance_fast(m, rows, cols, [rows.index(r) for r in keep])
    np.testing.assert_allclose(fast, d, atol=1e-10)


def test_silhouette_batch_matches_sklearn():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 3))
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    labs = np.stack([rng.integers(0, 3, 12) for _ in range(5)])
    labs[0] = [0] * 11 + [1]                 # a singleton cluster scores 0
    want = [silhouette_score(d, lab, metric="precomputed") for lab in labs]
    np.testing.assert_allclose(S.silhouette_batch(d, labs), want, atol=1e-12)


def test_taxonomy_z_finds_planted_groups():
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(3, 5)) * 4
    lab = np.repeat([0, 1, 2], 6)
    x = centers[lab] + rng.normal(size=(18, 5))
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    z = S.taxonomy_z(lab, {"a": d, "b": d}, ["a", "b"], n_perm=200, seed=0)
    assert z["a"] > 3 and z["b"] == z["a"] and z["agg"] > 3
    z_rand = S.taxonomy_z(rng.permutation(lab), {"a": d}, ["a"], n_perm=200, seed=0)
    assert abs(z_rand["a"]) < 3


def test_transport_labels_nearest_centroid():
    src = {"p": np.array([1.0, 0.0]), "q": np.array([0.9, 0.1]), "r": np.array([0.0, 1.0])}
    assign = {"p": 7, "q": 7, "r": 2}
    tgt = {"t1": np.array([0.2, 1.0]), "t2": np.array([5.0, 0.1])}
    assert S.transport_labels(assign, src, tgt) == {"t1": 2, "t2": 7}


def test_rate_operator_rebuilds_likert_rates():
    scen = pd.DataFrame({"value1": ["a", "a", "b"], "value2": ["b", "c", "c"]})
    lik = {"base": [0.0, 1.0, -1.0], "a": [-1.0, np.nan, 0.5]}
    evals = pd.DataFrame([{"model": m, "scenario": s, "likert": v}
                          for m, vs in lik.items() for s, v in enumerate(vs)])
    op = S.rate_operator(evals, 3, scen, ["a"], ["a", "b", "c"])
    base, raw = S.rates(op, np.ones((1, 3)))
    # column a: scenarios 0, 1 as value1 -> (1 - lik)/2
    # column b: s0 as value2 ((1+lik)/2), s2 as value1; column c: s1, s2 as value2
    np.testing.assert_allclose(base[0], [(0.5 + 0.0) / 2, (0.5 + 1.0) / 2, (1.0 + 0.0) / 2])
    np.testing.assert_allclose(raw[0, 0], [1.0, (0.0 + 0.25) / 2, 0.75])
    # weights reweight scenarios
    _, raw_w = S.rates(op, np.array([[2.0, 0.0, 1.0]]))
    np.testing.assert_allclose(raw_w[0, 0], [1.0, (2 * 0.0 + 0.25) / 3, 0.75])


def test_mantel_identical_matrices():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 2))
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    rho, p = S.mantel(d, d, np.random.default_rng(0), n_perm=99)
    assert rho == pytest.approx(1.0) and p == pytest.approx(0.01)
