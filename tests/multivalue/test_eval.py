"""The eval stage (`mv eval`): frozen suite inputs, protocol hashes, candidates,
sbatch rendering, completion markers (incl. reuse of imported runs), base staging, worker CLI."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from valuegen._external import REPO_ROOT
from valuegen.config import GcsConfig
from valuegen.multivalue import _hashing as H
from valuegen.multivalue import cli as mv_cli
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import eval_stage as ES
from valuegen.multivalue import eval_worker
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import train as mvtrain
from valuegen.multivalue.evals import get_suite, subsample
from valuegen.multivalue.evals import runner as R
from valuegen.multivalue.evals.base import EvalError

from test_data import _freeze
from test_train import _fake_finished_run, _prepared

_INPUTS: dict[str, dict] = {}
GRADER_VARS = ("GOOGLE_API_KEY", "OPENAI_API_KEY")


def _seed_inputs(env) -> dict[str, dict]:
    """Frozen inputs for both toy suites; built once per session, then
    copied into each test's layout (ensure_inputs reuses matching params)."""
    cfg, layout = env["cfg"], env["layout"]
    for s in ("toy_panel", "toy_items"):
        if s in _INPUTS:
            mvdata.write_json(R.inputs_path(layout, s), _INPUTS[s])
        else:
            _INPUTS[s] = R.ensure_inputs(cfg, layout, s, log=lambda *_: None)
    return {s: _INPUTS[s] for s in ("toy_panel", "toy_items")}


def _with_suite(cfg, name, **changes):
    suites = {**cfg.evals.suites, name: dataclasses.replace(cfg.evals.suites[name], **changes)}
    return dataclasses.replace(cfg, evals=dataclasses.replace(cfg.evals, suites=suites))


def _servable_imports(layout):
    """Give the fake experiment's exports a config.json so they count as servable."""
    for a in mvimports.load_imports(layout):
        if a.extra.get("model_dir"):
            Path(a.extra["model_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(a.extra["model_dir"]) / "config.json").write_text("{}")


def _marker(cfg, suite, inputs, *, repeats=None, grader=None, extra=None) -> dict:
    """An RQ3-driver-shaped completion marker (no multivalue protocol hash)."""
    sc = cfg.evals.suites[suite]
    n = repeats if repeats is not None else sc.repeats
    units = {u: {"expected": k * n, "n_scored": k * n, "n_failed": 0}
             for u, k in get_suite(suite).unit_expectations(inputs).items()}
    return {"suite": suite, "complete": True, "epochs": n, "grader": grader or cfg.evals.grader(suite),
            "candidate": {k: cfg.evals.candidate[k] for k in ("temperature", "top_p", "max_tokens")},
            "units": units, "problems": [], **(extra or {})}


def _grader_env(monkeypatch, present: bool):
    for v in GRADER_VARS:
        if present:
            monkeypatch.setenv(v, "test-key")
        else:
            monkeypatch.delenv(v, raising=False)


# ── subsampling, inputs, protocol ────────────────────────────────────────────


def test_subsample_is_seeded_and_order_preserving():
    ids = [f"i{n:02d}" for n in range(20)]
    s = subsample(ids, 5, 3)
    assert len(s) == 5 and s == [i for i in ids if i in set(s)]  # caller order
    assert subsample(ids, 5, 3) == s and subsample(list(reversed(ids)), 5, 3) == list(reversed(s))
    assert subsample(ids, 5, 4) != s
    assert subsample(ids, None, 3) == ids and subsample(ids, 20, 3) == ids and subsample(ids, 99, 3) == ids
    with pytest.raises(EvalError, match="duplicate"):
        subsample(["a", "a"], 1, 0)


def test_inputs_are_frozen_and_hashed(mv_env, monkeypatch):
    env = mv_env
    cfg, layout = env["cfg"], env["layout"]
    _freeze(env)
    inputs = _seed_inputs(env)
    ag, ms = inputs["toy_panel"], inputs["toy_items"]
    assert ag["n_panel"] == 6 and ag["n_conditions"] == 6 and ag["params"] == {"subsample": None}
    assert ag["conditions"][0]["condition_id"] == "s0_replacement"
    assert ag["inputs_sha256"] == H.sha256_json(ag["conditions"])
    assert ms["n_total"] == 20 and ms["n_samples"] == 10
    assert ms["params"] == {"subsample": {"n": 10, "seed": 3}}
    assert ms["inputs_sha256"] == H.sha256_json(ms["sample_ids"]) and len(ms["smoke_sample_ids"]) == 5
    assert R.inputs_path(layout, "toy_items").is_file()
    # matching params -> the frozen file is reused without rebuilding
    with monkeypatch.context() as m:
        m.setattr(type(get_suite("toy_items")), "build_inputs",
                  lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
        assert R.ensure_inputs(cfg, layout, "toy_items", log=lambda *_: None)["inputs_sha256"] == ms["inputs_sha256"]
    # a changed subsample rebuilds (and is a different protocol)
    cfg5 = _with_suite(cfg, "toy_items", subsample={"n": 5, "seed": 3})
    lines = []
    ms5 = R.ensure_inputs(cfg5, layout, "toy_items", log=lines.append)
    assert ms5["n_samples"] == 5 and set(ms5["sample_ids"]) < set(ms["sample_ids"]) and any("rebuilding" in ln for ln in lines)
    assert R.protocol_sha(cfg5, "toy_items", ms5) != R.protocol_sha(cfg, "toy_items", ms)
    # protocol hash: repeats, grader, smoke matter; connection limits do not
    base = R.protocol_sha(cfg, "toy_panel", ag)
    assert R.protocol_sha(_with_suite(cfg, "toy_panel", repeats=3), "toy_panel", ag) != base
    assert R.protocol_sha(_with_suite(cfg, "toy_panel", grader="openai/gpt-5.5"), "toy_panel", ag) != base
    assert R.protocol_sha(cfg, "toy_panel", ag, smoke=True) != base
    assert R.protocol_sha(_with_suite(cfg, "toy_panel", extra={"grader_max_connections": 2}), "toy_panel", ag) == base
    with pytest.raises(EvalError, match="unknown keys"):
        get_suite("toy_panel").options(dataclasses.replace(cfg.evals.suites["toy_panel"], extra={"bogus": 1}))


def test_suite_selection(mv_env):
    cfg = mv_env["cfg"]
    assert ES.select_suites(cfg, None) == ["toy_panel", "toy_items"]
    assert ES.select_suites(cfg, ["toy_items", "toy_items"]) == ["toy_items"]
    assert ES.select_suites(_with_suite(cfg, "prefill", enabled=True), None) == ["toy_panel", "toy_items", "prefill"]
    with pytest.raises(EvalError, match="disabled"):
        ES.select_suites(cfg, ["prefill"])
    with pytest.raises(EvalError, match="unknown suite"):
        ES.select_suites(cfg, ["nope"])
    with pytest.raises(EvalError, match="disabled"):
        ES.select_suites(_with_suite(cfg, "toy_panel", enabled=False), ["toy_panel"])
    pre = get_suite("prefill")
    with pytest.raises(EvalError, match="layout"):
        pre.build_inputs(cfg, cfg.evals.suites["prefill"], REPO_ROOT)
    assert pre.boot_unit == "scenario_id" and pre.per_candidate and "recant.by_bucket.*" in pre.metrics


def test_summarizers():
    ag = get_suite("toy_panel").summarize([
        {"valid": True, "harmful": 1.0, "urgency_type": "replacement", "scenario": "s0"},
        {"valid": True, "harmful": 0.0, "urgency_type": "replacement", "scenario": "s1"},
        {"valid": True, "harmful": 0.0, "urgency_type": "none", "scenario": "s0"},
        {"valid": False, "harmful": None, "urgency_type": "none", "scenario": "s2"},
    ])
    assert ag["harmful_rate"] == pytest.approx(1 / 3) and ag["replacement_harmful_rate"] == 0.5
    assert ag["none_harmful_rate"] == 0.0
    assert ag["by_scenario.s0"] == 0.5 and "by_scenario.s2" not in ag and ag["n_valid"] == 3
    ms = get_suite("toy_items").summarize([
        {"valid": True, "compliant": 1.0, "compliance_rating": 8, "top_level_section": "A"},
        {"valid": True, "compliant": 0.0, "compliance_rating": 3, "top_level_section": "B"},
        {"valid": True, "compliant": 1.0, "compliance_rating": None, "top_level_section": "B"},
    ])
    assert ms["overall_compliance"] == pytest.approx(2 / 3) and ms["mean_rating"] == 5.5
    assert ms["by_section.A"] == 1.0 and ms["by_section.B"] == 0.5
    pf = get_suite("prefill").summarize([
        {"valid": True, "condition": "injected", "pro_value": 1.0, "likert_toward_value": 6.0, "ambiguous": 0.0,
         "bucket": "on_target"},
        {"valid": True, "condition": "baseline", "pro_value": 0.0, "likert_toward_value": 2.0, "ambiguous": 0.0,
         "bucket": "on_target"},
        {"valid": False, "condition": "injected", "pro_value": None},
    ])
    assert pf["recant_rate"] == 1.0 and pf["baseline_rate"] == 0.0 and pf["recant.by_bucket.on_target"] == 1.0
    assert pf["recant_likert.by_bucket.on_target"] == 6.0 and pf["ambiguous.by_condition.baseline"] == 0.0
    assert pf["n_valid"] == 2 and "recant.by_bucket.off_target" not in pf


# ── candidates ───────────────────────────────────────────────────────────────


def test_plan_candidates(mv_env, tmp_path):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env, tmp_path)
    _servable_imports(layout)
    cands = ES.plan_candidates(cfg, cluster, layout, arms)
    by_id = {c.ckpt_id: c for c in cands}
    assert cands[0].ckpt_id == "base" and cands[0].kind == "base" and cands[0].model_source == "staged"
    assert cands[0].model_dir == layout.base_export_dir and not cands[0].export_verified
    trained = [c for c in cands if c.kind == "trained"]
    assert len(trained) == len(mvtrain.trainable_arms(arms)) and not any(c.export_verified for c in trained)
    assert by_id["sub3_00_s42"].model_dir == layout.checkpoint_dir("sub3_00", 42) / "export"
    assert by_id["ext_a_s42"].kind == "imported" and by_id["ext_a_s42"].export_verified and by_id["ext_a_s42"].seed == 42
    assert set(by_id["ext_a_s42"].external_suites) == {"toy_panel"}
    assert not by_id["ext_b_s42"].export_verified  # no export_verification.json in the fake
    assert [c.kind for c in cands] == ["base"] + ["trained"] * len(trained) + ["imported"] * 2
    with pytest.raises(EvalError, match="unknown arm"):
        ES.plan_candidates(cfg, cluster, layout, arms, only=["nope"])
    assert [c.ckpt_id for c in ES.plan_candidates(cfg, cluster, layout, arms, only=["base", "ext_a"])] == ["base", "ext_a_s42"]
    # a recorded, verified run becomes servable
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    _fake_finished_run(cfg, run)
    mvtrain.record_run(cfg, cluster, layout, run)
    assert {c.ckpt_id: c.export_verified for c in ES.plan_candidates(cfg, cluster, layout, arms)}[run.ckpt_id]
    # a stale record (another exp_id) does not
    rec = json.loads(run.record_path.read_text())
    rec["identity"]["exp_id"] = "other-000000000000"
    mvdata.write_json(run.record_path, rec)
    assert not {c.ckpt_id: c.export_verified for c in ES.plan_candidates(cfg, cluster, layout, arms)}[run.ckpt_id]
    # the base is served from an import's verified base copy at the pinned repo+revision
    imp = mvimports.load_imports_record(layout)
    neutral = Path(imp["base_model_dirs"][f"{cfg.train.base_model}@rev-a"])
    pinned = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, revision="rev-a"))
    assert ES.resolve_base_model_dir(pinned, layout, imp) == (layout.base_export_dir, "staged")  # not verified yet
    (neutral / "config.json").write_text("{}")
    (neutral / "export_verification.json").write_text('{"problems": []}')
    assert ES.resolve_base_model_dir(pinned, layout, imp) == (neutral, "imported")
    base = ES.plan_candidates(pinned, cluster, layout, arms, only=["base"])[0]
    assert base.model_dir == neutral and base.model_source == "imported" and base.export_verified


# ── rendering ────────────────────────────────────────────────────────────────


def test_eval_dry_run_renders_the_stage(mv_env, tmp_path, monkeypatch):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env, tmp_path)
    _servable_imports(layout)
    inputs = _seed_inputs(env)
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    _fake_finished_run(cfg, run)
    mvtrain.record_run(cfg, cluster, layout, run)
    _grader_env(monkeypatch, False)
    lines = []
    assert ES.evaluate(cfg, cluster, layout, arms, dry_run=True, log=lines.append) == 0
    out = "\n".join(lines)
    assert "eval: 3 candidates x ['toy_panel', 'toy_items'], 3 pending" in out
    assert "not scheduled" in out and "ext_b_s42" in out and "sub3_01_s42" in out and "(--all schedules" in out
    assert "WARNING: grader google/gemini-3.7-flash for toy_panel: GOOGLE_API_KEY" in out
    assert "WARNING: grader openai/gpt-5.5 for toy_items: OPENAI_API_KEY" in out
    assert "dry run: rendered" in out
    script = (layout.slurm_jobs_dir / "eval.sbatch").read_text()
    assert "#SBATCH --array=0-2" in script and "#SBATCH --gres=gpu:A6000:1" in script
    assert 'export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"' in script
    common = f"-c {cfg.path} --cluster {cluster.source_path}"
    base_dir = layout.base_export_dir
    assert (f"if [ ! -f {base_dir}/export_verification.json ]; then\n"
            f"    python -m valuegen.multivalue.eval_worker stage-base {common}\nfi") in script
    assert f"chat_formats check --model-dir {base_dir} --expect qwen_chatml" in script
    for name, model_dir in (("base", base_dir), (run.ckpt_id, run.export_dir),
                            ("ext_a_s42", Path(arms[-2].extra["model_dir"]))):
        assert f"vllm serve {model_dir} --served-model-name {name} " in script
        for s in ("toy_panel", "toy_items"):
            assert (f"python -m valuegen.multivalue.eval_worker run {common} --ckpt {name} --suite {s} "
                    f'--base-url "$CAND_BASE_URL"') in script
    assert "--max-model-len 16384 --max-num-seqs 64" in script and "--seed 10000" in script
    assert "--gpu-memory-utilization 0.9" in script
    assert script.count("stage-base") == 1 and "gcloud" not in script and "--smoke" not in script
    hostfile = layout.candidate_dir(run.ckpt_id) / "serve" / "cand_host.txt"
    assert str(hostfile) in script
    task_map = json.loads((layout.slurm_jobs_dir / "eval_task_map.json").read_text())
    assert [task_map[str(i)] for i in range(3)] == ["eval/base", f"eval/{run.ckpt_id}", "eval/ext_a_s42"]
    rec = json.loads(layout.candidates_record.read_text())
    assert rec["suites"] == ["toy_panel", "toy_items"] and rec["candidates"]["base"]["scheduled"]
    assert not rec["candidates"]["ext_b_s42"]["scheduled"] and rec["candidates"][run.ckpt_id]["values"] == list(run.arm.values)
    # --suite restricts the workers; --all schedules unverified runs too
    lines.clear()
    assert ES.evaluate(cfg, cluster, layout, arms, dry_run=True, suites=["toy_panel"], include_unverified=True,
                       log=lines.append) == 0
    script = (layout.slurm_jobs_dir / "eval.sbatch").read_text()
    assert "--suite toy_items" not in script and "--suite toy_panel" in script
    assert f"#SBATCH --array=0-{len(mvtrain.trainable_arms(arms)) + 1}" in script  # base + every run + ext_a
    assert "not scheduled: ['ext_b_s42']" in "\n".join(lines)
    with pytest.raises(EvalError, match="--after requires --no-wait"):
        ES.evaluate(cfg, cluster, layout, arms, dry_run=True, after="afterany:1", log=lines.append)
    # a GCS cluster stages the export in at run time and serves it through a link
    gcs_cluster = dataclasses.replace(cluster, gcs=GcsConfig(bucket="gs://bucket/mv", stage_dir=Path("/scratch/mv")))
    cmd = ES.eval_command(cfg, gcs_cluster, layout, ES.plan_candidates(cfg, gcs_cluster, layout, arms, only=[run.arm.id])[0],
                          ["toy_panel"])
    link = layout.candidate_dir(run.ckpt_id) / "serve" / "model"
    assert "gcloud storage rsync" in cmd and f'ln -sfn "$MODEL_DIR" {link}' in cmd
    assert f"vllm serve {link} --served-model-name {run.ckpt_id}" in cmd
    assert "gcloud" not in ES.eval_command(cfg, gcs_cluster, layout, ES.plan_candidates(cfg, gcs_cluster, layout, arms, only=["base"])[0], ["toy_panel"])
    # --smoke: base only, one repeat, a few units, its own tree
    lines.clear()
    assert ES.evaluate(cfg, cluster, layout, arms, dry_run=True, smoke=True, log=lines.append) == 0
    assert "smoke eval: 1 candidates" in "\n".join(lines)
    script = (layout.slurm_jobs_dir / "eval_smoke.sbatch").read_text()
    assert script.count("--smoke") == 2 and "--served-model-name base" in script and run.ckpt_id not in script
    assert (layout.smoke_evals_dir / "candidates.json").is_file()
    smoke_run = R.make_run(cfg, layout, ES.plan_candidates(cfg, cluster, layout, arms, only=["base"])[0], "toy_panel",
                           inputs["toy_panel"], "http://127.0.0.1:1/v1", smoke=True)
    assert smoke_run.epochs == 1 and len(smoke_run.units) == 2
    ms_run = R.make_run(cfg, layout, ES.plan_candidates(cfg, cluster, layout, arms, only=["base"])[0], "toy_items",
                        inputs["toy_items"], "http://127.0.0.1:1/v1")
    assert ms_run.epochs == 1 and ms_run.extra["sample_ids"] == inputs["toy_items"]["sample_ids"]
    assert ms_run.grader_name == "openai/gpt-5.5" and ms_run.candidate.temperature == 0.7
    assert ms_run.log_root == layout.suite_dir("base", "toy_items")


# ── markers / done ───────────────────────────────────────────────────────────


def test_markers_gate_done_and_external_reuse(mv_env, tmp_path):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env, tmp_path)
    _servable_imports(layout)
    inputs = _seed_inputs(env)
    suites = ["toy_panel", "toy_items"]
    ext = next(c for c in ES.plan_candidates(cfg, cluster, layout, arms) if c.ckpt_id == "ext_a_s42")
    ext_dir = Path(ext.external_suites["toy_panel"])
    # the fake's "{}" marker is not complete -> nothing to reuse
    assert R.suite_state(cfg, layout, ext, "toy_panel", inputs["toy_panel"]) == "pending"
    mvdata.write_json(ext_dir / "COMPLETE.json", _marker(cfg, "toy_panel", inputs["toy_panel"], repeats=3))
    ok, why = R.marker_accepts(mvdata.read_json(ext_dir / "COMPLETE.json"), cfg, "toy_panel", inputs["toy_panel"])
    assert not ok and "epochs 3 != repeats 2" in why
    mvdata.write_json(ext_dir / "COMPLETE.json", _marker(cfg, "toy_panel", inputs["toy_panel"], grader="google/other"))
    assert R.marker_accepts(mvdata.read_json(ext_dir / "COMPLETE.json"), cfg, "toy_panel", inputs["toy_panel"])[1].startswith("grader")
    mvdata.write_json(ext_dir / "COMPLETE.json", _marker(cfg, "toy_panel", inputs["toy_panel"]))
    assert R.marker_accepts(mvdata.read_json(ext_dir / "COMPLETE.json"), cfg, "toy_panel", inputs["toy_panel"]) == (True, "compatible")
    assert R.suite_state(cfg, layout, ext, "toy_panel", inputs["toy_panel"]) == "reusable"
    lines = []
    assert ES.evaluate(cfg, cluster, layout, arms, dry_run=True, log=lines.append) == 0
    assert any("ext_a_s42/toy_panel: reusing the imported run" in ln for ln in lines)
    linked = layout.suite_dir("ext_a_s42", "toy_panel")
    assert linked.is_symlink() and linked.resolve() == ext_dir.resolve()
    assert R.suite_state(cfg, layout, ext, "toy_panel", inputs["toy_panel"]) == "done"
    assert R.suite_state(cfg, layout, ext, "toy_items", inputs["toy_items"]) == "pending"
    assert not ES.candidate_done(cfg, layout, ext, suites, inputs)
    # a subsampled protocol never matches a full external run
    sub = _with_suite(cfg, "toy_panel", subsample={"n": 4, "seed": 1})
    sub_inputs = R.ensure_inputs(sub, layout, "toy_panel", log=lambda *_: None)
    assert sub_inputs["n_conditions"] == 4
    assert R.marker_accepts(mvdata.read_json(ext_dir / "COMPLETE.json"), sub, "toy_panel", sub_inputs)[1] == "6 units != 4 expected"
    # our own markers: protocol hash gates done; other hash -> stale; logs without marker -> partial
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    _fake_finished_run(cfg, run)
    mvtrain.record_run(cfg, cluster, layout, run)
    cand = next(c for c in ES.plan_candidates(cfg, cluster, layout, arms) if c.ckpt_id == run.ckpt_id)
    stage = ES.build_stage(cfg, cluster, layout, [cand], suites, inputs)
    assert stage.pending() == stage.tasks
    for s in suites:
        d = layout.suite_dir(cand.ckpt_id, s)
        d.mkdir(parents=True)
        mvdata.write_json(d / "COMPLETE.json", {"complete": True, "protocol_sha256": R.protocol_sha(cfg, s, inputs[s])})
    assert ES.candidate_states(cfg, layout, cand, suites, inputs) == {"toy_panel": "done", "toy_items": "done"}
    assert stage.pending() == []
    mvdata.write_json(layout.suite_dir(cand.ckpt_id, "toy_panel") / "COMPLETE.json",
                      {"complete": True, "protocol_sha256": "0" * 64})
    assert R.suite_state(cfg, layout, cand, "toy_panel", inputs["toy_panel"]) == "stale" and stage.pending() == stage.tasks
    (layout.suite_dir(cand.ckpt_id, "toy_panel") / "COMPLETE.json").unlink()
    (layout.suite_dir(cand.ckpt_id, "toy_panel") / "conditions" / "x").mkdir(parents=True)
    (layout.suite_dir(cand.ckpt_id, "toy_panel") / "conditions" / "x" / "log.eval").write_bytes(b"")
    assert R.suite_state(cfg, layout, cand, "toy_panel", inputs["toy_panel"]) == "partial"
    # a smoke marker lives in its own tree and never satisfies the real protocol
    mvdata.write_json(layout.suite_dir(cand.ckpt_id, "toy_panel", smoke=True) / "COMPLETE.json",
                      {"complete": True, "protocol_sha256": R.protocol_sha(cfg, "toy_panel", inputs["toy_panel"], smoke=True)})
    assert R.suite_state(cfg, layout, cand, "toy_panel", inputs["toy_panel"], smoke=True) == "done"
    assert R.suite_state(cfg, layout, cand, "toy_panel", inputs["toy_panel"]) == "partial"
    # once every suite is done the stage reports nothing pending
    lines.clear()
    for s in suites:
        for c in (cand, ES.plan_candidates(cfg, cluster, layout, arms, only=["base"])[0]):
            d = layout.suite_dir(c.ckpt_id, s)
            d.mkdir(parents=True, exist_ok=True)
            mvdata.write_json(d / "COMPLETE.json", {"complete": True, "protocol_sha256": R.protocol_sha(cfg, s, inputs[s])})
    assert ES.evaluate(cfg, cluster, layout, arms, dry_run=True, only=["base", run.arm.id], log=lines.append) == 0
    assert "all evals done" in lines[-1]


# ── base staging + worker ────────────────────────────────────────────────────


def _local_base(tmp_path) -> Path:
    src = tmp_path / "base_model"
    src.mkdir()
    (src / "config.json").write_text('{"architectures": ["Test"]}')
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "tokenizer.json").write_text("{}")
    return src


def test_local_grader_is_served_in_job_and_costs_one_packed_slot(mv_env, tmp_path):
    """A grader with no API provider (the prefill judge) is served once per eval
    element on the GPU the reduced ``pack`` leaves free."""
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env, tmp_path)
    suites = ES.select_suites(cfg, None)
    inputs = {s: R.ensure_inputs(cfg, layout, s, log=lambda *_: None) for s in suites}
    cands = ES.plan_candidates(cfg, cluster, layout, arms)[:3]
    node = dataclasses.replace(cluster, gpus_per_node=8)

    # API graders only: nothing is served, packing is left to the cluster default
    assert ES.needs_local_judge(cfg, suites) is None
    plain = ES.build_stage(cfg, node, layout, cands, suites, inputs)
    assert plain.pack == 0 and plain.node_setup == ""

    # a local grader on any enabled suite pulls in the in-job judge
    local = _with_suite(cfg, suites[0], grader="Qwen/Qwen3.6-27B")
    assert ES.needs_local_judge(local, suites) == "Qwen/Qwen3.6-27B"
    stage = ES.build_stage(local, node, layout, cands, suites, inputs)
    assert stage.pack == 7  # 8 GPUs - 1 for the judge
    assert stage.gpu_headroom == 0  # the judge is inside the element, not beside it
    setup = stage.node_setup
    assert "HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7" in setup  # the slot pack leaves free
    assert "--tensor-parallel-size 1" in setup and "vllm serve Qwen/Qwen3.6-27B" in setup
    assert 'export PREFILL_JUDGE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"' in setup
    assert "export PREFILL_JUDGE_MODEL=Qwen/Qwen3.6-27B" in setup and "PREFILL_JUDGE_NO_THINK=1" in setup
    assert "trap " in setup  # the server is reaped with the element

    # the rendered element runs the judge once, before the packed tasks fan out
    from valuegen.slurm import Orchestrator

    txt = Orchestrator(name="t", stages=[stage], cluster=node, dry_run=True).render(stage, stage.tasks)
    assert "parallel --jobs 7" in txt  # the chunk fans out over the candidate GPUs only
    assert txt.index("judge READY") < txt.index("case ${SLURM_ARRAY_TASK_ID")
    # a single-candidate element is not packed, and still gets its judge
    solo = Orchestrator(name="t", stages=[stage], cluster=node, dry_run=True).render(stage, stage.tasks[:1])
    assert "judge READY" in solo

    # a judge wider than the node has no room beside a candidate
    wide = _with_suite(local, "prefill", extra={**local.evals.suites["prefill"].extra, "judge_gpus": 8})
    with pytest.raises(EvalError, match="leaves no room"):
        ES.build_stage(wide, node, layout, cands, suites, inputs)


def test_stage_base_export(mv_env, tmp_path, monkeypatch):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    _freeze(env)
    src = _local_base(tmp_path)
    local = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, base_model=str(src)))
    import valuegen.ground_truth.training as T

    calls = []
    monkeypatch.setattr(T, "align_chat_format", lambda d, fmt: calls.append(("align", Path(d).name, fmt.name)) or {})
    monkeypatch.setattr(T, "verify_export", lambda d, chat_format=None: {"problems": ["dtype"]})
    with pytest.raises(EvalError, match="failed verification"):
        ES.stage_base_export(local, layout, log=lambda *_: None)
    dst = layout.base_export_dir
    assert not dst.exists() and dst.with_name("export.partial").is_dir()
    monkeypatch.setattr(T, "verify_export", lambda d, chat_format=None: {"problems": []})
    rec = ES.stage_base_export(local, layout, log=lambda *_: None)
    assert rec == {"problems": []} and dst.is_dir() and not dst.with_name("export.partial").exists()
    assert (dst / "model.safetensors").is_symlink() and (dst / "model.safetensors").resolve() == src / "model.safetensors"
    assert (dst / "config.json").is_file() and not (dst / "config.json").is_symlink()
    meta = json.loads((dst / "run_metadata.json").read_text())
    assert meta["kind"] == "base" and meta["origin"] == "local" and meta["chat_format"] == "qwen_chatml"
    assert calls == [("align", "export.partial", "qwen_chatml")] * 2
    # idempotent: a staged base is returned without touching anything
    monkeypatch.setattr(T, "verify_export", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-staged")))
    assert ES.stage_base_export(local, layout) == {"problems": []}
    base = ES.plan_candidates(local, cluster, layout, mvimports.all_arms(layout), only=["base"])[0]
    assert base.export_verified and base.model_source == "staged"


def test_eval_worker_cli(mv_env, tmp_path, monkeypatch, capsys):
    env = mv_env
    cluster = env["cluster"]
    _freeze(env)
    src = _local_base(tmp_path)
    import yaml

    raw = dict(env["raw"])
    raw["train"] = {**raw["train"], "base_model": str(src)}
    raw["embeddings"] = {**raw["embeddings"], "persona": {**raw["embeddings"]["persona"], "model": str(src)}}
    cfg_path = tmp_path / "local_base.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    import valuegen.ground_truth.training as T

    monkeypatch.setattr(T, "align_chat_format", lambda d, fmt: {})
    monkeypatch.setattr(T, "verify_export", lambda d, chat_format=None: {"problems": []})
    monkeypatch.setattr(eval_worker, "load_cluster", lambda _p=None: cluster)
    c = ["-c", str(cfg_path), "--cluster", str(cluster.source_path)]
    assert eval_worker.main(["stage-base", *c]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])  # after the "base staged at" line
    assert out["problems"] == [] and Path(out["base_export_dir"]).is_dir()
    assert eval_worker.main(["run", *c, "--ckpt", "nope", "--suite", "toy_panel", "--base-url", "http://127.0.0.1:1/v1"]) == 1
    assert "not a candidate" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        eval_worker.main(["run", *c, "--ckpt", "base"])  # --suite/--base-url required


def test_cli_eval(mv_env, tmp_path, monkeypatch, capsys):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    monkeypatch.setattr(mv_cli, "load_cluster", lambda _p=None: cluster)
    c = ["-c", str(env["cfg_path"])]
    assert mv_cli.main(["eval", *c, "--dry-run"]) == 1
    assert "mv sets" in capsys.readouterr().err  # not frozen
    _prepared(env, tmp_path)
    _seed_inputs(env)
    _grader_env(monkeypatch, True)
    assert mv_cli.main(["eval", *c, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run: rendered" in out and "WARNING" not in out and (layout.slurm_jobs_dir / "eval.sbatch").is_file()
    assert mv_cli.main(["eval", *c, "--build", "toy_panel"]) == 0
    assert "'n_conditions': 6" in capsys.readouterr().out
    assert mv_cli.main(["eval", *c, "--build", "prefill"]) == 1
    err = capsys.readouterr().err  # the fixture's 12-value universe never matches the real pool (or has no csv)
    assert "expected_scenarios" in err or "does not exist" in err
    assert mv_cli.main(["eval", *c, "--dry-run", "--suite", "prefill"]) == 1
    assert "disabled" in capsys.readouterr().err
    assert mv_cli.main(["eval", *c, "--after", "afterany:1"]) == 1
    assert "--after requires --no-wait" in capsys.readouterr().err
    # a missing grader key refuses a real submission (and no job is ever launched from tests)
    _grader_env(monkeypatch, False)
    assert mv_cli.main(["eval", *c]) == 1
    assert "GOOGLE_API_KEY" in capsys.readouterr().err
    _grader_env(monkeypatch, True)
    monkeypatch.setattr(ES, "sbatch_available", lambda: False)
    assert mv_cli.main(["eval", *c]) == 1
    assert "sbatch not found" in capsys.readouterr().out
