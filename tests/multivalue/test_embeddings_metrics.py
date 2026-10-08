import itertools
import json

import numpy as np
import pytest

from valuegen.multivalue import data as mvdata
from valuegen.multivalue.config import StoreConfig
from valuegen.multivalue.embeddings import EmbeddingError, load_store, load_stores, load_vectors
from valuegen.multivalue.metrics import coverage, cosine_grid, score_pool, set_metrics, tightness, unit


def test_load_vectors_every_format(tmp_path):
    names = ["a", "b", "c"]
    M = np.arange(12, dtype=float).reshape(3, 4)
    np.savez(tmp_path / "x.npz", labels=np.array(names), vectors=M)
    np.savez(tmp_path / "y.npz", **{n: M[i] for i, n in enumerate(names)})
    np.save(tmp_path / "z.npy", M)
    (tmp_path / "z.labels.json").write_text(json.dumps(names))
    (tmp_path / "w.json").write_text(json.dumps({n: M[i].tolist() for i, n in enumerate(names)}))
    import torch
    torch.save({n: torch.tensor(M[i]) for i, n in enumerate(names)}, tmp_path / "v.pt")
    for f in ("x.npz", "y.npz", "z.npy", "w.json", "v.pt"):
        vecs = load_vectors(tmp_path / f)
        assert set(vecs) == set(names)
        assert np.allclose(vecs["b"], M[1]), f
    with pytest.raises(EmbeddingError, match="sidecar"):
        np.save(tmp_path / "nolabels.npy", M)
        load_vectors(tmp_path / "nolabels.npy")
    (tmp_path / "x.csv").write_text("a,1\n")
    with pytest.raises(EmbeddingError, match="unsupported"):
        load_vectors(tmp_path / "x.csv")


def test_store_layer_selection_and_alignment(mv_env):
    cfg, env = mv_env["cfg"], mv_env
    uni = env["source_values"]
    stores = load_stores(cfg, uni, env["ext_names"])
    s = stores["sentence"]
    assert s.universe.shape == (len(uni), 16) and s.external.shape == (6, 16)
    assert np.allclose(s.universe[3], env["uni_vecs"][uni[3]])
    assert stores["persona"].universe.shape == (len(uni), 16)  # layer 2 of [4, 16]
    # 2-D vectors without a layer
    bad = StoreConfig(name="persona", universe=cfg.embeddings["persona"].universe,
                      external=cfg.embeddings["persona"].external, model=cfg.train.base_model, layer=None)
    with pytest.raises(EmbeddingError, match="set `layer:`"):
        load_store(bad, uni, env["ext_names"])
    with pytest.raises(EmbeddingError, match="have no vector"):
        load_store(cfg.embeddings["sentence"], uni + ["not_a_value"], env["ext_names"])


def test_coverage_and_tightness_by_hand():
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    members = np.stack([e1, np.array([1.0, 1.0])])
    external = np.stack([e1, e2, np.array([-1.0, 0.0])])
    # max cos per external: e1 -> 1.0 ; e2 -> cos(45deg)=0.7071 ; -e1 -> -0.7071 (vs [1,1])
    assert coverage(members, external) == pytest.approx((1.0 + np.sqrt(0.5) - np.sqrt(0.5)) / 3)
    assert tightness(members) == pytest.approx(np.sqrt(0.5))
    assert np.isnan(tightness(e1[None, :]))
    assert tightness(np.stack([e1, e1, e1])) == pytest.approx(1.0)


def test_score_pool_matches_per_set(mv_env):
    env = mv_env
    uni = env["source_values"]
    s = load_stores(env["cfg"], uni, env["ext_names"], only=["sentence"])["sentence"]
    S_ue, G_uu = cosine_grid(s.universe, s.external), cosine_grid(s.universe, s.universe)
    pool = np.asarray(list(itertools.combinations(range(len(uni)), 3)))
    cov = score_pool(pool, "coverage", S_ue=S_ue)
    tig = score_pool(pool, "tightness", G_uu=G_uu, chunk=7)
    for i in [0, 5, len(pool) - 1]:
        m = set_metrics([uni[j] for j in pool[i]], uni, s.universe, s.external)
        assert cov[i] == pytest.approx(m["coverage"]) and tig[i] == pytest.approx(m["tightness"])
    assert np.all(np.isnan(score_pool(pool[:, :1], "tightness", G_uu=G_uu)))
    with pytest.raises(ValueError):
        score_pool(pool, "nope", S_ue=S_ue)


def test_universe_and_external_resolution(mv_env):
    cfg, cluster = mv_env["cfg"], mv_env["cluster"]
    names, desc = mvdata.resolve_universe(cfg, cluster)
    assert names == mv_env["source_values"] and set(desc) == set(names)
    ext, edesc = mvdata.load_external(cfg)
    assert ext == sorted(mv_env["ext_names"]) and set(edesc) == set(ext)
    import shutil
    shutil.rmtree(mv_env["datasets_dir"])
    with pytest.raises(mvdata.SourceError, match="valuegen gt intervene"):
        mvdata.resolve_universe(cfg, cluster)
