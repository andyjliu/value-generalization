"""Unit tests for matrix analysis and the ``analyze`` CLI.

These deliberately use small synthetic matrices, so the API contracts are
pinned independently of any large shipped artifact.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import pytest

from valuegen import cli
from valuegen.analysis import correlate as C
from valuegen.analysis import mds as M
from valuegen.analysis import cluster as CL
from valuegen.ground_truth.matrices import save_matrix


def test_offdiag_mask_and_reindex_handle_rectangular_label_frames():
    rows = ["train_b", "train_a"]
    cols = ["train_a", "held_out", "train_b"]
    mask = C.offdiag_mask(rows, cols)
    np.testing.assert_array_equal(
        mask, [[True, True, False], [False, True, True]]
    )

    source = np.array([[11.0, 12.0], [21.0, 22.0]])
    aligned = C.reindex(
        source, ["train_a", "train_b"], ["held_out", "train_a"], rows, cols
    )
    expected = np.array([[22.0, 21.0, np.nan], [12.0, 11.0, np.nan]])
    np.testing.assert_allclose(aligned, expected, equal_nan=True)


def test_correlate_respects_mask_nan_filter_and_seeded_bootstrap():
    a = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    b = np.array([[2.0, 4.0, 99.0], [8.0, 10.0, -6.0]])
    mask = np.array([[True, True, True], [True, False, False]])

    first = C.correlate(a, b, mask, n_boot=100, seed=17)
    second = C.correlate(a, b, mask, n_boot=100, seed=17)
    assert first.n == 3
    assert first.pearson_r == pytest.approx(1.0)
    assert first.spearman_rho == pytest.approx(1.0)
    assert (first.ci_lo, first.ci_hi) == pytest.approx((1.0, 1.0))
    assert first == second

    too_small = C.correlate(a, b, np.array([[True, False, False], [True, False, False]]))
    assert too_small.n == 0 and np.isnan(too_small.pearson_r)
    with pytest.raises(ValueError, match="shape mismatch"):
        C.correlate(a, np.ones((3, 2)))


def test_leaderboard_reindexes_and_drops_self_cells():
    rows = ["a", "b", "c"]
    target = np.array([[100.0, 1.0, 2.0], [3.0, 100.0, 4.0], [5.0, 6.0, 100.0]])
    # Source is deliberately in a different value order; its off-diagonal cells
    # exactly match target, while the diagonal does not.
    pred_values = ["c", "a", "b"]
    pred = C.reindex(target, rows, rows, pred_values, pred_values)
    np.fill_diagonal(pred, -5.0)
    entries = C.leaderboard(
        target, rows, rows, {"perfect": (pred, pred_values, pred_values)},
        n_boot=0, ceiling=0.8,
    )
    entry = entries[0]
    assert entry["n"] == 6
    assert entry["pearson_r"] == pytest.approx(1.0)
    assert entry["ceiling_frac"] == pytest.approx(1.25)
    assert "r/ceil" in C.format_leaderboard(entries, ceiling=0.8)
    assert C.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert np.isnan(C.spearman_brown(-1.0))


def test_sim_at_layer_slices_and_pads_missing_values():
    vectors = {
        "a": np.array([[1.0, 0.0], [1.0, 0.0]]),
        "b": np.array([[0.0, 1.0], [1.0, 0.5]]),
        "c": np.array([[-1.0, 0.0], [0.0, 1.0]]),
    }
    values = ["a", "b", "c"]
    # layer 0: a ⟂ b; layer 1: a·b > 0 — the slice is what changes the grid.
    assert C.sim_at_layer(vectors, values, 0)[0, 1] == pytest.approx(0.0)
    assert C.sim_at_layer(vectors, values, 1)[0, 1] > 0
    # ``c`` has no vector and must stay NaN-padded rather than drop a row.
    sim = C.sim_at_layer({k: v for k, v in vectors.items() if k != "c"}, values, 1)
    assert np.isnan(sim[2]).all() and np.isnan(sim[:, 2]).all()


def test_no_gt_dependent_layer_selection_is_exposed():
    # Layer selection must be GT-independent (steer-and-judge sweep on held-out
    # questions). A helper that picks the layer best-correlating with the target
    # fits the target, so analysis must not offer one.
    assert not hasattr(C, "best_layer_by_corr")


def test_mds_dissimilarities_embeddings_and_baselines_are_deterministic():
    sim = np.array([
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.8],
        [0.1, 0.2, 0.8, 1.0],
    ])
    dist = M.cos_to_dist(sim)
    assert np.allclose(dist, dist.T) and np.allclose(np.diag(dist), 0)
    # SMACOF consumes the raw 1−cos distances; no rescaling variant may creep
    # back in, since a rescaled map's distances no longer read as cosines.
    assert M.MDS_VARIANTS == ("metric_raw", "nonmetric")
    assert not hasattr(M, "zscore_dist") and not hasattr(M, "spectral_embed")

    metric, stress = M.mds_variant(sim, "metric_raw", seed=3)
    assert metric.shape == (4, 2) and stress >= 0
    nonmetric, stress = M.mds_variant(sim, "nonmetric", seed=3)
    assert nonmetric.shape == (4, 2) and stress >= 0
    with pytest.raises(ValueError, match="unknown MDS variant"):
        M.mds_variant(sim, "not-a-variant")

    padded = sim.copy()
    padded[3, :] = np.nan
    padded[:, 3] = np.nan
    compact, values = M.drop_nan_values(padded, ["a", "b", "c", "missing"])
    assert values == ["a", "b", "c"] and compact.shape == (3, 3)
    means_a, sds_a = CL.random_baseline(6, [2, 3], n_rand=8, seed=9)
    means_b, sds_b = CL.random_baseline(6, [2, 3], n_rand=8, seed=9)
    np.testing.assert_allclose(means_a, means_b)
    np.testing.assert_allclose(sds_a, sds_b)


def test_resolve_matrix_ignores_fork_output_without_provenance(cluster):
    """weight_steer has the fork write its raw ``similarity_matrix.npy`` into
    the same store dir as the registered matrix; only the latter is a result."""
    from valuegen.predictors import store

    values = ["a", "b"]
    sim = np.array([[1.0, 0.5], [0.5, 1.0]])
    sim_dir = store.similarity_dir(
        cluster, "weight_steer", "default_llm", "org/M", "aid-1"
    )
    sim_dir.mkdir(parents=True)
    np.save(sim_dir / "similarity_matrix.npy", sim)
    (sim_dir / "similarity_values.json").write_text(json.dumps(values))

    # Raw output alone: the build never finished, so nothing is resolvable.
    with pytest.raises(FileNotFoundError, match="no similarity matrix"):
        C.resolve_matrix("weight_steer", cluster, "org/M", "default_llm")

    store.save_similarity(sim_dir, "weight_steer_cos", sim, values,
                          provenance={"predictor": "weight_steer"})
    matrix, rows, cols, label = C.resolve_matrix(
        "weight_steer", cluster, "org/M", "default_llm"
    )
    np.testing.assert_allclose(matrix, sim)
    assert rows == cols == values
    assert label.endswith("weight_steer_cos")


def test_analyze_correlate_cli_writes_label_aligned_leaderboard(
    cluster, monkeypatch, tmp_path, capsys
):
    rows = ["a", "b", "c"]
    target = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
    pred = target.copy()
    half_a = target.copy()
    half_b = target * 2
    for name, matrix in {
        "target": target, "pred": pred, "half_a": half_a, "half_b": half_b,
    }.items():
        save_matrix(tmp_path, name, matrix, rows)
    out = tmp_path / "leaderboard.csv"
    monkeypatch.setattr(cli, "load_cluster", lambda _=None: cluster)

    args = argparse.Namespace(
        verb="correlate", target=str(tmp_path / "target.npy"),
        pred=[str(tmp_path / "pred.npy")], model=None, data_method=None,
        all_cells=False, n_boot=0, seed=0, ceiling=None,
        ceiling_halves=[str(tmp_path / "half_a.npy"), str(tmp_path / "half_b.npy")],
        out=str(out), cluster=None,
    )
    assert cli.cmd_analyze(args) == 0
    assert "Spearman–Brown ceiling" in capsys.readouterr().out
    with open(out, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["predictor"] == "pred" and float(row["pearson_r"]) == pytest.approx(1.0)
    assert float(row["ceiling_frac"]) == pytest.approx(1.0)


def test_relabel_null_stats_permutes_the_observed_labels():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(0, 0.1, (10, 2)), rng.normal(3, 0.1, (10, 2))])
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    lab = [0] * 10 + [1] * 10
    mu, sd = CL.relabel_null_stats(dist, lab, n_rand=200)
    assert abs(mu) < 0.2 and sd > 0
    assert CL.relabel_null_stats(dist, lab, n_rand=200) == (mu, sd)  # seeded
