import itertools
import json

import numpy as np
import pytest

from valuegen.multivalue import cli as mv_cli
from valuegen.multivalue import sets as mvsets
from valuegen.multivalue.embeddings import load_stores
from valuegen.multivalue.layout import NotFrozenError


def _stores(env, only=None):
    return load_stores(env["cfg"], env["source_values"], env["ext_names"], only=only)


def test_candidate_pool_enumerates_or_samples():
    rng = np.random.default_rng(1)
    pool = mvsets.candidate_pool(6, 3, 1000, rng)
    assert pool.shape == (20, 3) and len({tuple(r) for r in pool}) == 20
    big = mvsets.candidate_pool(40, 5, 500, rng)
    assert big.shape == (500, 5) and len({tuple(r) for r in big}) == 500
    assert np.all(np.diff(big, axis=1) > 0)  # sorted, distinct within a row


def test_allocate_extremes_first_and_deficit():
    assert mvsets.allocate(8, 4, [10, 10, 10, 10]) == [2, 2, 2, 2]
    assert mvsets.allocate(10, 4, [10, 10, 10, 10]) == [3, 2, 2, 3]
    assert mvsets.allocate(8, 4, [1, 10, 10, 10]) == [1, 2, 2, 3]
    with pytest.raises(mvsets.SetsError):
        mvsets.allocate(8, 2, [3, 3])


def test_stratified_draw_covers_bins():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    picked, bins, edges = mvsets.sample_stratified(scores, 16, 8, rng)
    assert len(picked) == 16 and len(set(picked.tolist())) == 16
    assert [int((bins == b).sum()) for b in range(8)] == [2] * 8
    assert np.all(np.diff(scores[picked]) >= 0)


def test_width_bins_trim_and_even_spread():
    rng = np.random.default_rng(0)
    scores = rng.exponential(size=5000)  # right-skewed, like persona tightness
    bin_of, edges = mvsets.width_bins(scores, 4, trim=1.0)
    lo, hi = np.percentile(scores, [1.0, 99.0])
    assert edges[0] == pytest.approx(lo) and edges[-1] == pytest.approx(hi)
    assert np.allclose(np.diff(edges), (hi - lo) / 4)
    assert set(np.unique(bin_of)) == {-1, 0, 1, 2, 3}
    assert np.all((scores[bin_of == -1] < lo) | (scores[bin_of == -1] > hi))
    picked, bins, _ = mvsets.sample_stratified(scores, 16, 4, rng, binning="width", trim=1.0)
    assert [int((bins == b).sum()) for b in range(4)] == [4] * 4
    assert np.all((scores[picked] >= lo) & (scores[picked] <= hi))
    # a quarter of the picks land in each quarter of the range, whatever the pool's shape
    assert np.histogram(scores[picked], bins=edges)[0].tolist() == [4] * 4
    with pytest.raises(mvsets.SetsError):
        mvsets.assign_bins(scores, 4, "quantile", trim=1.0)
    with pytest.raises(mvsets.SetsError):
        mvsets.assign_bins(scores, 4, "nope")


def test_random_scheme_is_deterministic_and_distinct(mv_env):
    env = mv_env
    stores = _stores(env)
    rec1, ro1 = mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="random")
    rec2, _ = mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="random")
    assert rec1["arms_sha256"] == rec2["arms_sha256"]
    arms = rec1["families"]["sub3"]["arms"]
    assert len(arms) == 8 and all(len(a["values"]) == 3 for a in arms.values())
    assert len({tuple(a["values"]) for a in arms.values()}) == 8
    assert rec1["families"]["probe"]["scheme"] == "explicit"
    assert rec1["families"]["probe"]["arms"]["probe_01"]["values"] == sorted(env["source_values"][2:5])
    rec3, _ = mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="random", seed=8)
    assert rec3["arms_sha256"] != rec1["arms_sha256"]
    # readouts present for both stores and both metrics
    assert set(ro1["sub3"]["pool"]) == {"sentence", "persona"}
    assert "coverage" in ro1["sub3"]["picked"]["persona"]
    assert ro1["sub3"]["design_coverage"]["max"] >= 1
    assert "_external_reach" in ro1 and "coverage_ceiling" in ro1["_external_reach"]["sentence"]
    text = mvsets.format_readouts(rec1, ro1)
    assert "family sub3" in text and "family probe" in text


def test_stratified_scheme_records_scores_and_bins(mv_env):
    env = mv_env
    stores = _stores(env)
    rec, ro = mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="stratified",
                                metric="tightness", store="persona", n_candidates=500, bins=4)
    arms = rec["families"]["sub3"]["arms"]
    assert rec["store"] == "persona" and rec["metric"] == "tightness"
    assert "binning" not in rec  # default cut leaves the record (hence exp_id) as before
    rec_w, ro_w = mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="stratified", metric="tightness",
                                    store="persona", n_candidates=500, bins=4, binning="width", trim=1.0)
    assert rec_w["binning"] == "width" and rec_w["trim"] == 1.0
    assert ro_w["sub3"]["bins"]["picked_occupancy"] == [2, 2, 2, 2]
    assert "width, trim 1%" in mvsets.format_readouts(rec_w, ro_w)
    assert [a["bin"] for a in arms.values()] == sorted(a["bin"] for a in arms.values())
    assert ro["sub3"]["bins"]["picked_occupancy"] == [2, 2, 2, 2]
    assert ro["sub3"]["pool_size"] == 220  # C(12,3) < 500 -> full enumeration
    # recorded score equals the metric recomputed from the store
    s = stores["persona"]
    from valuegen.multivalue.metrics import set_metrics
    for a in arms.values():
        assert a["score"] == pytest.approx(set_metrics(a["values"], s.universe_names, s.universe, s.external)["tightness"])
    assert any(k.startswith("tightness:persona~sentence") or k.startswith("tightness:sentence~persona")
               for k in ro["sub3"]["store_agreement_spearman"])
    with pytest.raises(mvsets.SetsError, match="not loaded"):
        mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="stratified", store="nope")


def test_freeze_semantics_and_identity(mv_env):
    env, layout = mv_env, mv_env["layout"]
    stores = _stores(env)
    rec, _ = mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="random")
    with pytest.raises(NotFrozenError):
        layout.exp_id
    mvsets.write_sets(layout.sets_path, rec)
    exp1 = layout.exp_id
    assert exp1.startswith("mvtest-")
    mvsets.write_sets(layout.sets_path, rec)  # identical: no-op
    assert layout.exp_id == exp1
    other, _ = mvsets.build_sets(env["cfg"], env["source_values"], stores, scheme="random", seed=99)
    with pytest.raises(mvsets.SetsError, match="--force"):
        mvsets.write_sets(layout.sets_path, other)
    mvsets.write_sets(layout.sets_path, other, force=True)
    assert layout.exp_id != exp1
    arms = mvsets.arms_from_record(mvsets.load_sets(layout.sets_path))
    assert arms[0].id == "base" and arms[0].kind == "base" and not arms[0].trained and arms[0].k == 0
    assert len(mvsets.trained_arms(arms)) == 10
    assert layout.checkpoint_dir("sub3_00", 42).name == "sub3_00_s42"
    assert layout.mix_dir("probe_01").parent == layout.exp_dir / "mixes"


def test_full_reference_arm(mv_env):
    rec = {"schema_version": 1, "universe": ["a", "b", "c"], "reference": ["base", "full"],
           "families": {"f": {"scheme": "random", "arms": {"f_00": {"values": ["a", "b"]}}}}}
    arms = mvsets.arms_from_record(rec)
    assert [a.id for a in arms] == ["base", "full", "f_00"]
    assert arms[1].values == ("a", "b", "c") and arms[1].kind == "full"


def test_cli_sets_preview_freeze_and_metrics(mv_env, monkeypatch, capsys):
    env = mv_env
    monkeypatch.setattr(mv_cli, "load_cluster", lambda _p=None: env["cluster"])
    argv = ["sets", "-c", str(env["cfg_path"]), "--scheme", "stratified", "--candidates", "300", "--bins", "4"]
    assert mv_cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "not frozen" in out and not env["layout"].frozen
    assert any(env["layout"].preview_dir.glob("stratified_coverage_sentence_s7.json"))
    assert mv_cli.main(argv + ["--freeze"]) == 0
    assert env["layout"].frozen
    rec = json.loads(env["layout"].sets_path.read_text())
    assert rec["scheme"] == "stratified" and rec["store"] == "sentence" and rec["bins"] == 4
    # a different draw refuses to overwrite without --force
    with pytest.raises(mvsets.SetsError, match="--force"):
        mv_cli.main(argv + ["--freeze", "--seed", "3"])
    assert mv_cli.main(["metrics", "-c", str(env["cfg_path"])]) == 0
    import pandas as pd
    df = pd.read_csv(env["layout"].metrics_csv)
    assert set(df["store"]) == {"sentence", "persona"}
    assert len(df) == 2 * (1 + 8 + 2)
    base = df[df["arm_id"] == "base"]
    assert base["coverage"].isna().all() and (base["rows"] == 0).all()
    sub = df[(df["arm_id"] == "sub3_00") & (df["store"] == "sentence")].iloc[0]
    assert sub["k"] == 3 and sub["rows"] == 60 and sub["sampling_score"] == pytest.approx(sub["coverage"])
    assert (df["arms_sha256"] == rec["arms_sha256"]).all()
    assert mv_cli.main(["train", "-c", str(env["cfg_path"])]) == 1  # frozen but no mixes yet -> refused
