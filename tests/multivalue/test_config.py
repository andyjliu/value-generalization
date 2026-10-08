import copy

import pytest
import yaml

from valuegen.multivalue.config import ConfigError, load_config, parse_config


def _reload(env, mutate):
    raw = copy.deepcopy(env["raw"])
    mutate(raw)
    p = env["cfg_path"].with_name("mut.yaml")
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(p)


def test_loads_and_types(mv_env):
    cfg = mv_env["cfg"]
    assert cfg.name == "mvtest" and cfg.seed == 7
    assert set(cfg.embeddings) == {"sentence", "persona"}
    assert cfg.embeddings["persona"].layer == 2
    assert [f.id for f in cfg.arms.families] == ["sub3", "probe"]
    assert cfg.family("sub3").k == 3 and cfg.family("sub3").n == 8
    assert cfg.family("probe").explicit and cfg.family("probe").size == 2
    assert cfg.budget.rows == 60
    assert cfg.train.seeds == (42,) and cfg.train.resources["gpus"] == 2
    assert cfg.evals.grader("toy_panel") == "google/gemini-3.7-flash"
    assert cfg.evals.grader("toy_items") == "openai/gpt-5.5"
    assert cfg.evals.enabled() == ["toy_panel", "toy_items"]
    assert cfg.evals.suites["toy_items"].subsample == {"n": 10, "seed": 3}
    assert cfg.evals.candidate["max_tokens"] == 4096  # default filled
    assert cfg.analysis.n_boot == 50


def test_training_identity_excludes_scheduling(mv_env):
    a = mv_env["cfg"].training_identity()
    b = _reload(mv_env, lambda r: r["train"].update({"concurrent": 9, "resources": {"gpus": 8, "time": "9:00:00"}})).training_identity()
    assert a == b
    c = _reload(mv_env, lambda r: r["budget"].update({"rows": 61})).training_identity()
    assert a != c
    d = _reload(mv_env, lambda r: r["evals"]["suites"]["toy_panel"].update({"repeats": 50})).training_identity()
    assert a == d  # evals are outside the training identity


def test_missing_embedding_file_is_an_error(mv_env):
    with pytest.raises(ConfigError, match="no embedding file"):
        _reload(mv_env, lambda r: r["embeddings"]["sentence"].update({"external": "/nonexistent.npz"}))


def test_persona_store_must_declare_matching_model(mv_env):
    with pytest.raises(ConfigError, match="must declare `model:`"):
        _reload(mv_env, lambda r: r["embeddings"]["persona"].pop("model"))
    with pytest.raises(ConfigError, match="base-matched"):
        _reload(mv_env, lambda r: r["embeddings"]["persona"].update({"model": "other/model"}))
    # sentence stores are model-free
    _reload(mv_env, lambda r: r["embeddings"]["sentence"].pop("model", None))


def test_family_validation(mv_env):
    with pytest.raises(ConfigError, match="explicit family"):
        _reload(mv_env, lambda r: r["arms"]["families"][1].update({"k": 2}))
    with pytest.raises(ConfigError, match="duplicate ids"):
        _reload(mv_env, lambda r: r["arms"]["families"].append({"id": "sub3", "k": 2, "n": 2}))
    with pytest.raises(ConfigError, match="collides with a reference"):
        _reload(mv_env, lambda r: r["arms"]["families"].append({"id": "base", "k": 2, "n": 2}))
    with pytest.raises(ConfigError, match="must be a list containing 'base'"):
        _reload(mv_env, lambda r: r["arms"].update({"reference": ["full"]}))


def test_suite_and_grader_validation(mv_env):
    with pytest.raises(ConfigError, match="unknown suite"):
        _reload(mv_env, lambda r: r["evals"]["suites"].update({"xstest": {}}))
    with pytest.raises(ConfigError, match="provider/model"):
        _reload(mv_env, lambda r: r["evals"]["suites"]["toy_panel"].update({"grader": "flash"}))
    with pytest.raises(ConfigError, match="unknown keys"):
        _reload(mv_env, lambda r: r.update({"conflictscope": {}}))


def test_require_files_false_still_checks_base_match(mv_env, tmp_path):
    raw = copy.deepcopy(mv_env["raw"])
    raw["embeddings"]["sentence"]["universe"] = "/nonexistent.npz"
    parse_config(raw, require_files=False)  # ok
    raw["embeddings"]["persona"]["model"] = "wrong"
    with pytest.raises(ConfigError, match="base-matched"):
        parse_config(raw, require_files=False)
