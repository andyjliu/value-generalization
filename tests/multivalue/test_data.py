"""Per-arm mixes (`mv data`) and external arms (`mv import`)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from valuegen.multivalue import _hashing as H
from valuegen.multivalue import cli as mv_cli
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import sets as mvsets
from valuegen.multivalue.embeddings import load_stores


def _freeze(env, **kw):
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    universe, _ = mvdata.resolve_universe(cfg, cluster)
    external, _ = mvdata.load_external(cfg)
    stores = load_stores(cfg, universe, external)
    record, _ = mvsets.build_sets(cfg, universe, stores, scheme="random", n_candidates=200, **kw)
    mvsets.write_sets(layout.sets_path, record)
    return mvsets.arms_from_record(record)


def _buildable(arms):
    # the fixture has 20 rows per value and budget 60: a 2-value set would need 30 each
    return [a.id for a in arms if a.trained and a.k >= 3]


# ── source + selection primitives ────────────────────────────────────────────


def test_load_source_and_audit(mv_env):
    env = mv_env
    vals = env["source_values"][:4]
    rows = mvdata.load_source(env["datasets_dir"], vals, "rev-x")
    assert {v: len(r) for v, r in rows.items()} == {v: 20 for v in vals}
    r0 = rows[vals[0]][0]
    assert r0.identity_sha256 == H.sha256_text(r0.identity)
    assert json.loads(r0.identity)[0] == "rev-x"
    rep = mvdata.audit_source(rows, env["datasets_dir"], min_per_value=20)
    assert rep["n_rows"] == 80 and not rep["problems"] and set(rep["file_sha256"]) == set(vals)
    with pytest.raises(mvdata.SourceError, match="need at least 21"):
        mvdata.audit_source(rows, env["datasets_dir"], min_per_value=21)
    with pytest.raises(mvdata.SourceError, match="no source rows"):
        mvdata.load_source(env["datasets_dir"], ["nope"], "rev-x")


def test_selection_is_nested_and_seeded(mv_env):
    env = mv_env
    vals = env["source_values"][:3]
    rows = mvdata.load_source(env["datasets_dir"], vals, "rev")
    small, q_small = mvdata.select_arm(rows, "a", vals[:2], seed=7, n_total=20)
    big, q_big = mvdata.select_arm(rows, "b", vals, seed=7, n_total=45)
    assert q_small == {vals[0]: 10, vals[1]: 10} and sum(q_big.values()) == 45 and set(q_big.values()) == {15}
    sel = {"a": mvdata.selected_identities_by_value(small, 7), "b": mvdata.selected_identities_by_value(big, 7)}
    assert mvdata.nested_selection_check(sel)["ok"]
    # a different seed reorders the per-value ranking
    other, _ = mvdata.select_arm(rows, "a", vals[:2], seed=8, n_total=20)
    assert {r.identity for r in other} != {r.identity for r in small}
    # the mix shuffle depends on the arm id, not the row ordering
    m1 = mvdata.shuffle_mix(small, 7, "a")
    m2 = mvdata.shuffle_mix(list(reversed(small)), 7, "a")
    assert [r.identity for r in m1] == [r.identity for r in m2]
    assert [r.identity for r in mvdata.shuffle_mix(small, 7, "z")] != [r.identity for r in m1]
    bad = {"a": {"v": ["1", "2"]}, "b": {"v": ["2", "1", "3"]}}
    assert mvdata.nested_selection_check(bad) == {"ok": False, "violations": [{"arms": ["a", "b"], "value": "v"}]}
    with pytest.raises(mvdata.SourceError, match="exceeds"):
        mvdata.select_arm(rows, "a", vals[:1], seed=7, n_total=21)


def test_write_mix_is_atomic_and_content_addressed(mv_env, tmp_path):
    env = mv_env
    vals = env["source_values"][:2]
    rows = mvdata.load_source(env["datasets_dir"], vals, "rev")
    out = tmp_path / "mix" / "arm"
    mixed, comp = mvdata.build_arm(rows, "arm", vals, seed=1, n_total=10, out_dir=out)
    assert (out / "dataset.jsonl").is_file() and (out / "rows.jsonl").is_file()
    assert comp["n_rows"] == 10 and comp["dataset_sha256"] == H.sha256_file(out / "dataset.jsonl")
    assert not list(out.parent.glob("*.partial-*"))
    recs = [json.loads(l) for l in (out / "dataset.jsonl").read_text().splitlines()]
    assert all(set(r) == set(mvdata.DATASET_KEYS) for r in recs)
    prov = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines()]
    assert [p["position"] for p in prov] == list(range(10))
    # identical rebuild keeps the published record
    _, again = mvdata.build_arm(rows, "arm", vals, seed=1, n_total=10, out_dir=out)
    assert again["dataset_sha256"] == comp["dataset_sha256"] and again["built_at"] == comp["built_at"]
    # a different selection refuses to overwrite
    with pytest.raises(mvdata.SourceError, match="refusing to overwrite"):
        mvdata.build_arm(rows, "arm", vals, seed=2, n_total=10, out_dir=out)
    assert json.loads((out / "composition.json").read_text())["dataset_sha256"] == comp["dataset_sha256"]


# ── token audit with a stand-in tokenizer ────────────────────────────────────

EOS = 999


class _Tok:
    """Deterministic stand-in: one id per character, a generation marker,
    and EOS after every assistant turn (so the completion ends with EOS)."""

    eos_token_id = EOS

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_dict=True,
                            chat_template=None):
        ids = []
        for m in messages:
            if m["role"] == "assistant":
                ids += [1, *[ord(c) for c in m["content"]], EOS]
            else:
                ids += [ord(c) for c in m["content"]]
        if add_generation_prompt:
            ids.append(1)
        return {"input_ids": ids}


def _row(prompt: str, chosen: str, rejected: str, value="v", i=0):
    return mvdata.SourceRow(
        value=value, source="s", scenario_id=f"{value}-{i}", score=None,
        prompt=({"role": "user", "content": prompt},), chosen=({"role": "assistant", "content": chosen},),
        rejected=({"role": "assistant", "content": rejected},), triple_sha256="t", identity=f"id{i}",
        identity_sha256="h", file_index=i)


def test_token_audit_counts_drops_truncation_and_gates():
    rows = [
        _row("ab", "cde", "fg", i=0),                      # prompt 3 (+gen) fits: 3+4=7 / 3+3=6 tokens
        _row("abcdefgh", "x", "y", i=1),                   # prompt 9 >= max_length 8: TRL drops it
        _row("abcd", "wxyz", "wxyq", i=2),                 # 5+5=10 > 8: both sides truncated to 8
        _row("abcdef", "p", "q", i=3),                     # prompt 7 < 8; 7+2=9: completion cut to 1 token each
    ]
    audit = mvdata.token_audit(rows, _Tok(), max_length=8, chat_template="t")
    assert audit["n_rows"] == 4 and audit["n_dropped_by_trl_filter"] == 1
    assert audit["dropped_by_trl_filter"][0]["index"] == 1
    assert audit["n_truncated_pairs"] == 3  # rows 1, 2, 3
    assert audit["n_prompt_mismatch"] == 0 and audit["n_missing_terminator_before_truncation"] == 0
    # row 2: "wxyz"+EOS vs "wxyq"+EOS both cut to "wxy" -> identical after truncation
    assert audit["n_identical_completions_after_truncation"] == 1
    assert audit["n_empty_completion_after_truncation"] == 0
    assert audit["lengths"]["prompt"]["max"] == 9 and audit["by_value"]["v"]["rows"] == 4
    fails, warns = mvdata.audit_gates("arm", audit, accept_drops=False)
    assert len(fails) == 1 and "TRL would drop 1" in fails[0]
    assert any("identical" in w for w in warns)
    fails, _ = mvdata.audit_gates("arm", audit, accept_drops=True)
    assert fails == []
    assert mvdata.step_geometry(12000, 6, 16) == {"expect_raw_rows": 12000, "expect_train_rows": 11994,
                                                   "expect_max_steps": 750, "expect_warmup_steps": 75}


def test_overlap_audit_exact_and_normalized():
    rows = [_row("What is  love?", "a", "b", i=0), _row("Other", "a", "b", i=1)]
    rep = mvdata.overlap_audit(rows, {"toy_panel": {"q1": "what is love?", "q2": "Other", "q3": "none"}})
    assert rep["suites"]["toy_panel"] == {"n_inputs": 3, "exact_matches": 1, "normalized_matches": 2}
    assert not rep["clean"] and {m["input_id"] for m in rep["suspected_matches"]} == {"q1", "q2"}


# ── the stage ────────────────────────────────────────────────────────────────


def test_build_mixes_stage(mv_env):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms = _freeze(env)
    only = _buildable(arms)
    assert len(only) == 8 + 1  # sub3 x 8 + probe_01 (k=3); probe_00 (k=2) cannot take 30 rows/value
    manifest = mvdata.build_mixes(cfg, cluster, layout, arms, token_audit_enabled=False, only=only, log=lambda *_: None)
    assert set(manifest["arms"]) == set(only) and manifest["nesting_ok"] and manifest["gate_failures"] == []
    assert manifest["effective_batch"] == 2  # recipe has no batch keys: 1 x 1 x 2 gpus
    assert layout.config_record.is_file() and (layout.mixes_dir / "source_audit.json").is_file()
    rec = yaml.safe_load(layout.config_record.read_text())
    assert rec["exp_id"] == layout.exp_id and rec["identity"] == layout.identity()
    for aid in only:
        comp = json.loads((layout.mix_dir(aid) / "composition.json").read_text())
        assert comp["n_rows"] == 60 and comp["quota"] == {v: 20 for v in comp["values"]}
        assert comp["exp_id"] == layout.exp_id and comp["arms_sha256"] == mvsets.load_sets(layout.sets_path)["arms_sha256"]
        n_lines = sum(1 for _ in open(layout.mix_dir(aid) / "dataset.jsonl"))
        assert n_lines == 60 and manifest["arms"][aid]["expect"]["expect_max_steps"] == 30
    nesting = json.loads((layout.mixes_dir / "nesting.json").read_text())
    assert nesting["ok"] and nesting["seed"] == cfg.seed
    # rows for a value are identical across arms that share it (nested prefixes at equal quota)
    by_arm = {}
    for aid in only:
        for line in open(layout.mix_dir(aid) / "rows.jsonl"):
            r = json.loads(line)
            by_arm.setdefault(aid, {}).setdefault(r["value"], set()).add(r["identity"])
    shared = [(a, b, v) for a in only for b in only if a < b for v in set(by_arm[a]) & set(by_arm[b])]
    assert shared and all(by_arm[a][v] == by_arm[b][v] for a, b, v in shared)
    # rerun: identical, no rewrite
    again = mvdata.build_mixes(cfg, cluster, layout, arms, token_audit_enabled=False, only=only, log=lambda *_: None)
    assert {a: m["dataset_sha256"] for a, m in again["arms"].items()} == {a: m["dataset_sha256"] for a, m in manifest["arms"].items()}
    # the 2-value probe arm exceeds the per-value supply
    with pytest.raises(mvdata.SourceError, match="need at least 30"):
        mvdata.build_mixes(cfg, cluster, layout, arms, token_audit_enabled=False, log=lambda *_: None)


def test_build_mixes_token_audit_uses_cache_and_gates(mv_env, monkeypatch):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms = _freeze(env)
    only = _buildable(arms)[:2]
    # recipe: max_length small enough that nothing is dropped (prompts are ~14 chars)
    recipe = yaml.safe_load(cfg.train.recipe.read_text())
    recipe.update({"max_length": 64, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 3})
    cfg.train.recipe.write_text(yaml.safe_dump(recipe))

    class _Fmt:
        name, template, template_sha256 = "qwen_chatml", "tpl", "abc"

    calls = []
    monkeypatch.setattr(mvdata, "load_tokenizer", lambda _cfg: (calls.append(1), (_Tok(), _Fmt()))[1])
    m = mvdata.build_mixes(cfg, cluster, layout, arms, only=only, log=lambda *_: None)
    assert calls == [1] and m["effective_batch"] == 12 and m["max_length"] == 64
    for aid in only:
        audit = json.loads((layout.mix_dir(aid) / "token_audit.json").read_text())
        assert audit["n_dropped_by_trl_filter"] == 0 and audit["chat_format"] == "qwen_chatml"
        assert m["arms"][aid]["expect"] == {"expect_raw_rows": 60, "expect_train_rows": 60,
                                            "expect_max_steps": 5, "expect_warmup_steps": 1}
    # second run: audits cached, tokenizer not loaded again
    mvdata.build_mixes(cfg, cluster, layout, arms, only=only, log=lambda *_: None)
    assert calls == [1]
    # a max_length that drops rows fails the gate unless the recipe accepts drops
    recipe["max_length"] = 12
    cfg.train.recipe.write_text(yaml.safe_dump(recipe))
    with pytest.raises(mvdata.GateError, match="TRL would drop"):
        mvdata.build_mixes(cfg, cluster, layout, arms, only=only, log=lambda *_: None)
    recipe["accept_trl_prompt_drops"] = True
    cfg.train.recipe.write_text(yaml.safe_dump(recipe))
    m = mvdata.build_mixes(cfg, cluster, layout, arms, only=only, log=lambda *_: None)
    assert all(a["expect"]["expect_train_rows"] < 60 for a in m["arms"].values())


# ── imports ──────────────────────────────────────────────────────────────────


def _fake_rq3_experiment(tmp_path, env, name="rq3_fake", base_repo=None, revision="rev-a", chat_format="qwen_chatml",
                         arms=None):
    """An RQ3-driver-shaped experiment: config.yaml (layout.out_root),
    out/runs.jsonl, mixes with composition.json, exports, one completed suite."""
    base_repo = base_repo or env["cfg"].train.base_model
    exp = tmp_path / name
    out = exp / "out_root"
    exp.mkdir()
    (exp / "config.yaml").write_text(yaml.safe_dump({"name": name, "layout": {"out_root": str(out)}}))
    out.mkdir()
    arms = arms or {"ext_a": env["source_values"][:3], "ext_b": env["source_values"][2:6]}
    runs = [{"kind": "neutral", "run_id": None, "checkpoint_id": "neutral_x", "identity": {
        "base_repo": base_repo, "base_revision": revision, "chat_format": chat_format, "template_sha256": "t"},
        "arm_id": "neutral", "values": [], "model_dir": str(out / "models" / "neutral_x")}]
    (out / "models" / "neutral_x").mkdir(parents=True)
    for aid, values in arms.items():
        ckpt = f"{aid}_s42_deadbeef"
        mix = out / "mixes" / aid / "seed_42"
        mix.mkdir(parents=True)
        (mix / "dataset.jsonl").write_text("{}\n" * 30)
        (mix / "composition.json").write_text(json.dumps(
            {"arm_id": aid, "values": values, "n_rows": 30, "dataset_sha256": f"sha-{aid}"}))
        export = out / "ckpt" / ckpt / "export"
        export.mkdir(parents=True)
        if aid != "ext_b":
            (export / "export_verification.json").write_text("{}")
        suite = out / "evals" / "toy_panel" / ckpt
        suite.mkdir(parents=True)
        (suite / "COMPLETE.json").write_text("{}")
        runs.append({"kind": "trained", "run_id": ckpt, "checkpoint_id": ckpt,
                     "identity": {"experiment": name, "arm_id": aid, "base_revision": revision,
                                  "chat_format": chat_format, "template_sha256": "t"},
                     "arm_id": aid, "arm_kind": "subset", "training_seed": 42, "values": values,
                     "mix_dir": str(mix), "mix_dataset_sha256": f"sha-{aid}", "model_dir": str(export)})
    with open(out / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    return exp


def test_import_experiment_registers_arms(mv_env, tmp_path):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    exp = _fake_rq3_experiment(tmp_path, env)
    universe, _ = mvdata.resolve_universe(cfg, cluster)
    res = mvimports.import_experiment(cfg, cluster, layout, exp, universe=universe, log=lambda *_: None)
    assert res["added"] == ["ext_a", "ext_b"] and res["experiment"] == "rq3_fake"
    rec = mvimports.load_imports_record(layout)
    a = rec["arms"]["ext_a"]
    assert a["k"] == 3 and a["rows"] == 30 and a["dataset_sha256"] == "sha-ext_a" and a["export_verified"]
    assert set(a["evals"]) == {"toy_panel"} and a["base"]["revision"] == "rev-a"
    assert not rec["arms"]["ext_b"]["export_verified"]
    assert rec["base_model_dirs"] == {f"{cfg.train.base_model}@rev-a": str(exp / "out_root" / "models" / "neutral_x")}
    # idempotent; the experiment dir, its out root and the runs file all resolve
    for src in (exp, exp / "out_root", exp / "out_root" / "runs.jsonl"):
        res = mvimports.import_experiment(cfg, cluster, layout, src if src.is_dir() else src.parent,
                                          universe=universe, log=lambda *_: None)
        assert res["added"] == [] and res["unchanged"] == ["ext_a", "ext_b"]
    arms = mvimports.load_imports(layout)
    assert [a.id for a in arms] == ["ext_a", "ext_b"] and all(a.kind == "imported" and a.trained for a in arms)
    assert arms[0].family == "import:rq3_fake" and arms[0].extra["model_dir"].endswith("export")
    # unfrozen: all_arms is just the imports; frozen: reference + families + imports
    assert [a.id for a in mvimports.all_arms(layout)] == ["ext_a", "ext_b"]
    _freeze(env)
    ids = [a.id for a in mvimports.all_arms(layout)]
    assert ids[0] == "base" and ids[-2:] == ["ext_a", "ext_b"] and len(ids) == 1 + 8 + 2 + 2


def test_import_exclude_skips_named_source_arms(mv_env, tmp_path):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    universe, _ = mvdata.resolve_universe(cfg, cluster)
    exp = _fake_rq3_experiment(tmp_path, env)
    res = mvimports.import_experiment(cfg, cluster, layout, exp, universe=universe, prefix="p_",
                                      exclude=["ext_b"], log=lambda *_: None)
    assert res["added"] == ["p_ext_a"] and res["skipped"] == ["ext_b"]
    assert list(mvimports.load_imports_record(layout)["arms"]) == ["p_ext_a"]
    # an exclude that names no run is a typo, not a no-op
    with pytest.raises(mvimports.ImportError_, match="no run for: \\['ext_zz'\\]"):
        mvimports.import_experiment(cfg, cluster, layout, exp, universe=universe, prefix="p_",
                                    exclude=["ext_zz"], log=lambda *_: None)


def test_import_refuses_mismatch_collision_and_rewrites(mv_env, tmp_path):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    universe, _ = mvdata.resolve_universe(cfg, cluster)
    other = _fake_rq3_experiment(tmp_path, env, name="rq3_olmo", base_repo="org/other-base", revision="rev-b")
    with pytest.raises(mvimports.ImportError_, match="base model org/other-base"):
        mvimports.import_experiment(cfg, cluster, layout, other, universe=universe, log=lambda *_: None)
    res = mvimports.import_experiment(cfg, cluster, layout, other, universe=universe, allow_base_mismatch=True,
                                      prefix="olmo_", log=lambda *_: None)
    assert res["added"] == ["olmo_ext_a", "olmo_ext_b"]
    rec = mvimports.load_imports_record(layout)
    assert rec["arms"]["olmo_ext_a"]["base_mismatch"] and rec["base_model_dirs"]
    # same ids from a different source with different content: refused unless --force
    changed = _fake_rq3_experiment(tmp_path, env, name="rq3_olmo2", base_repo="org/other-base", revision="rev-b",
                                   arms={"ext_a": env["source_values"][:4]})
    with pytest.raises(mvimports.ImportError_, match="--force"):
        mvimports.import_experiment(cfg, cluster, layout, changed, universe=universe, allow_base_mismatch=True,
                                    prefix="olmo_", log=lambda *_: None)
    res = mvimports.import_experiment(cfg, cluster, layout, changed, universe=universe, allow_base_mismatch=True,
                                      prefix="olmo_", force=True, log=lambda *_: None)
    assert res["added"] == ["olmo_ext_a"] and mvimports.load_imports_record(layout)["arms"]["olmo_ext_a"]["k"] == 4
    # values outside the universe
    alien = _fake_rq3_experiment(tmp_path, env, name="rq3_alien", arms={"ext_z": ["not_a_value"]})
    with pytest.raises(mvimports.ImportError_, match="outside the universe"):
        mvimports.import_experiment(cfg, cluster, layout, alien, universe=universe, log=lambda *_: None)
    # collision with a frozen arm id
    _freeze(env)
    clash = _fake_rq3_experiment(tmp_path, env, name="rq3_clash", arms={"sub3_00": env["source_values"][:3]})
    with pytest.raises(mvimports.ImportError_, match="collides with a frozen arm"):
        mvimports.import_experiment(cfg, cluster, layout, clash, universe=universe, log=lambda *_: None)
    with pytest.raises(mvimports.ImportError_, match="no runs.jsonl"):
        mvimports.locate_runs(tmp_path / "nowhere")


def test_cli_import_data_metrics_flow(mv_env, tmp_path, monkeypatch, capsys):
    env = mv_env
    layout = env["layout"]
    monkeypatch.setattr(mv_cli, "load_cluster", lambda _p=None: env["cluster"])
    c = ["-c", str(env["cfg_path"])]
    exp = _fake_rq3_experiment(tmp_path, env)
    assert mv_cli.main(["import", *c, "--from", str(exp)]) == 0
    assert "2 arms imported" in capsys.readouterr().out
    assert mv_cli.main(["sets", *c, "--scheme", "random", "--candidates", "200", "--freeze"]) == 0
    arms = mvsets.arms_from_record(mvsets.load_sets(layout.sets_path))
    only = [x for a in _buildable(arms) for x in ("--arm", a)]
    assert mv_cli.main(["data", *c, "--skip-token-audit", *only]) == 0
    out = capsys.readouterr().out
    assert "9 mixes under" in out and (layout.mixes_dir / "manifest.json").is_file()
    assert not (layout.mixes_dir / "ext_a").exists()  # imported arms are never rebuilt
    assert mv_cli.main(["metrics", *c]) == 0
    df = pd.read_csv(layout.metrics_csv)
    imp = df[df["arm_id"] == "ext_a"]
    assert len(imp) == 2 and set(imp["kind"]) == {"imported"} and (imp["rows"] == 30).all() and (imp["k"] == 3).all()
    assert imp["coverage"].notna().all()
    assert (df[df["arm_id"] == "sub3_00"]["rows"] == 60).all()
