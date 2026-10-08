"""`mv run`: train -> eval waves, the export-deletion tombstone, the self-resubmitting driver."""

from __future__ import annotations

import dataclasses
import json

import pytest

from valuegen.multivalue import cli as mv_cli
from valuegen.multivalue import config as mvconfig
from valuegen.multivalue import eval_stage as ES
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import status as S
from valuegen.multivalue import train as mvtrain
from valuegen.multivalue import waves as W
from valuegen.multivalue.evals import runner as R
from valuegen.multivalue.layout import Layout
from valuegen.slurm import Orchestrator

from test_analyze_status import _inputs, _write_eval
from test_train import _fake_finished_run, _prepared


def _cand(cfg, cluster, layout, arms, ckpt_id):
    return {c.ckpt_id: c for c in ES.plan_candidates(cfg, cluster, layout, arms)}[ckpt_id]


# ── config ───────────────────────────────────────────────────────────────────


def test_schedule_config_is_validated_and_outside_the_identity(mv_env, tmp_path):
    env = mv_env
    cfg, raw = env["cfg"], env["raw"]
    assert cfg.schedule == mvconfig.ScheduleConfig(wave=None, delete_exports=False, order="contiguous")
    with_sched = mvconfig.parse_config({**raw, "schedule": {"wave": 8, "delete_exports": True}}, path=cfg.path)
    assert with_sched.schedule.wave == 8 and with_sched.schedule.delete_exports
    assert with_sched.training_identity() == cfg.training_identity()
    for bad, msg in (({"wave": 0}, "positive integer"), ({"wave": True}, "positive integer"),
                     ({"delete_exports": "yes"}, "boolean"), ({"waves": 2}, "unknown keys"),
                     ({"order": "random"}, "schedule.order")):
        with pytest.raises(mvconfig.ConfigError, match=msg):
            mvconfig.parse_config({**raw, "schedule": bad}, path=cfg.path)
    # CLI overrides win; the block is the fallback
    assert W.resolve_schedule(with_sched, None, None) == (8, True)
    assert W.resolve_schedule(with_sched, 3, False) == (3, False)
    assert W.resolve_schedule(cfg, None, True) == (None, True)
    with pytest.raises(W.WaveError):
        W.resolve_schedule(cfg, 0, None)


# ── wave cutting ─────────────────────────────────────────────────────────────


def test_cut_waves_and_wave_candidates(mv_env):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env)
    runs = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)
    assert len(runs) == 9
    waves = W.cut_waves(runs, 4)
    assert [len(w) for w in waves] == [4, 4, 1] and [r for w in waves for r in w] == runs
    assert W.cut_waves(runs, None) == [runs] and W.cut_waves(runs, 100) == [runs] and W.cut_waves([], 4) == [[]]
    with pytest.raises(W.WaveError):
        W.cut_waves(runs, 0)
    # stride: same runs, no wave over W, every wave spans the run list end to end
    strided = W.cut_waves(runs, 4, "stride")
    assert strided == [runs[0::3], runs[1::3], runs[2::3]] and max(len(w) for w in strided) <= 4
    assert W.cut_waves(runs, None, "stride") == [runs] and W.cut_waves([], 4, "stride") == [[]]
    with pytest.raises(W.WaveError, match="unknown wave order"):
        W.cut_waves(runs, 4, "random")
    # base joins the first eval wave only; unverified trained runs are never candidates of a wave
    _fake_finished_run(cfg, runs[0])
    mvtrain.record_run(cfg, cluster, layout, runs[0])
    cands = ES.plan_candidates(cfg, cluster, layout, arms)
    first = W.wave_candidates(cands, [runs[0]], first=True)
    assert [c.ckpt_id for c in first] == ["base", runs[0].ckpt_id]
    assert [c.ckpt_id for c in W.wave_candidates(cands, [runs[0]], first=False)] == [runs[0].ckpt_id]
    assert W.wave_candidates(cands, [], first=False) == []


# ── the tombstone ────────────────────────────────────────────────────────────


def test_delete_export_needs_every_enabled_suite_and_leaves_a_tombstone(mv_env, monkeypatch, capsys):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env)
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    assert W.resolve_schedule(cfg, None, None) == (None, False)  # off by default
    ok, why = mvtrain.delete_export(cfg, cluster, layout, run)
    assert not ok and "not done" in why
    _fake_finished_run(cfg, run)
    mvtrain.record_run(cfg, cluster, layout, run)
    ok, why = mvtrain.delete_export(cfg, cluster, layout, run)
    assert not ok and "no frozen inputs" in why and run.export_dir.is_dir()
    inputs = _inputs(env)
    cand = _cand(cfg, cluster, layout, arms, run.ckpt_id)
    assert cand.export_verified and not cand.export_deleted
    ok, why = mvtrain.delete_export(cfg, cluster, layout, run)
    assert not ok and "toy_panel is pending" in why
    _write_eval(env, cand, inputs, 1)
    # a --suite subset cannot cause a premature delete: toy_items is checked even if the wave ran only toy_panel
    ms = layout.suite_dir(run.ckpt_id, "toy_items") / "COMPLETE.json"
    ms.unlink()
    ok, why = mvtrain.delete_export(cfg, cluster, layout, run)
    assert not ok and "toy_items is pending" in why
    _write_eval(env, cand, inputs, 1)
    sha_before = json.loads(run.record_path.read_text())["export_sha256"]
    ok, why = mvtrain.delete_export(cfg, cluster, layout, run)
    assert ok and "export deleted" in why
    assert not run.export_dir.exists() and (run.trainer_dir / "config.json").is_file()
    rec = json.loads(run.record_path.read_text())
    assert rec["export_sha256"] == sha_before and rec["export_verified"]
    assert rec["export_deleted"]["suites"] == ["toy_panel", "toy_items"] and rec["export_deleted"]["by"] == "mv run --delete-exports"
    assert mvtrain.export_tombstone(run) == rec["export_deleted"]
    # done-check, state, stage, planner, evaluate, status all understand the tombstone
    assert mvtrain.run_done(layout, cluster, run) and mvtrain.run_state(layout, cluster, run) == "pruned"
    assert mvtrain.build_stage(cfg, cluster, layout, [run]).pending() == []
    cand = _cand(cfg, cluster, layout, arms, run.ckpt_id)
    assert cand.export_deleted and not cand.export_verified and cand.to_json()["export_deleted"]
    ok, why = mvtrain.delete_export(cfg, cluster, layout, run)
    assert not ok and "already deleted" in why
    lines = []
    assert ES.evaluate(cfg, cluster, layout, arms, dry_run=True, include_unverified=True, log=lines.append) == 0
    out = "\n".join(lines)
    assert "1 candidates have their exports deleted" in out and "0 of them are incomplete" in out
    script = (layout.slurm_jobs_dir / "eval.sbatch").read_text()
    assert run.ckpt_id not in script
    crec = json.loads(layout.candidates_record.read_text())
    assert crec["candidates"][run.ckpt_id]["export_deleted"] and not crec["candidates"][run.ckpt_id]["scheduled"]
    # enabling a new suite: the deleted candidate is reported as needing a retrain, never scheduled
    lines.clear()
    prefill_on = {**cfg.evals.suites, "prefill": dataclasses.replace(cfg.evals.suites["prefill"], enabled=True)}
    cfg_p = dataclasses.replace(cfg, evals=dataclasses.replace(cfg.evals, suites=prefill_on))
    with monkeypatch.context() as m:
        m.setattr(R, "ensure_inputs", lambda c, l, s, **k: inputs.get(s, {"suite": s}))
        m.setattr(ES, "candidate_done", lambda *a, **k: False)
        assert ES.evaluate(cfg_p, cluster, layout, arms, dry_run=True, suites=["prefill"], log=lines.append) == 0
    assert f"1 of them are incomplete for ['prefill'] -- a new suite needs a retrain" in "\n".join(lines)
    # a stale identity still wins over the tombstone
    stale = dict(json.loads(run.record_path.read_text()))
    stale["identity"] = {**stale["identity"], "dataset_sha256": "f" * 64}
    from valuegen.multivalue import data as mvdata
    mvdata.write_json(run.record_path, stale)
    assert not mvtrain.run_done(layout, cluster, run) and mvtrain.run_state(layout, cluster, run) == "stale"
    mvdata.write_json(run.record_path, rec)
    frame = S.status_frame(cfg, cluster, layout, arms)
    assert frame.set_index("arm_id").loc[run.arm.id, "train"] == "pruned"
    assert "pruned=1" in S.format_status(cfg, layout, frame)
    # base and imported arms are refused by kind
    base_run = dataclasses.replace(run, arm=dataclasses.replace(run.arm, kind="base"))
    assert not mvtrain.delete_export(cfg, cluster, layout, base_run)[0]


def test_delete_export_refuses_on_a_gcs_cluster(mv_env):
    # A cluster that stages exports to a bucket has already push-and-pruned them;
    # a tombstone would hide the remote copy, so --delete-exports must refuse (before
    # any local rmtree) even for a finished, fully-scored run.
    from valuegen.config import GcsConfig
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env)
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    _fake_finished_run(cfg, run)
    mvtrain.record_run(cfg, cluster, layout, run)
    for suite in cfg.evals.enabled():
        _write_eval(env, _cand(cfg, cluster, layout, arms, run.ckpt_id), _inputs(env), 1)
    gcs = dataclasses.replace(cluster, gcs=GcsConfig(bucket="gs://b/prefix", stage_dir=layout.root / "stage"))
    ok, why = mvtrain.delete_export(cfg, gcs, layout, run)
    assert not ok and "gcs:" in why and run.export_dir.is_dir()
    assert mvtrain.export_tombstone(run) is None


# ── the loop ─────────────────────────────────────────────────────────────────


class _StubOrchestrator:
    """Runs every pending task in-process: a train task lays down a finished
    run, an eval task writes complete markers; keys in ``fail`` do nothing."""

    calls: list[tuple[str, list[str]]] = []
    fail: set[str] = set()
    complete = None  # set by the test: key -> None

    def __init__(self, name, stages, cluster, dry_run=False, **_):
        self.name, self.cluster, self.dry_run = name, cluster, dry_run
        self.script_dir = cluster.repo / "slurm_jobs" / name
        self.log_dir = cluster.slurm_logs / name

    def run_stage(self, stage, wait=True, dependency=None):
        keys = [t.key for t in stage.pending()]
        type(self).calls.append((stage.name, keys))
        for t in stage.pending():
            if t.key not in self.fail:
                type(self).complete(t.key)
        return (not stage.pending()), None

    _directives = Orchestrator._directives
    _preamble = Orchestrator._preamble
    _placement = Orchestrator._placement
    _walltime = Orchestrator._walltime
    _clamp_noted = set()


def test_run_waves_alternates_skips_failures_and_deletes_after_eval(mv_env, monkeypatch):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env)
    inputs = _inputs(env)
    runs = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)
    by_ckpt = {r.ckpt_id: r for r in runs}
    monkeypatch.setattr(W, "sbatch_available", lambda: True)
    for var in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "test-key")  # a real (non-dry) run refuses to start without grader keys

    def complete(key):
        kind, ckpt = key.split("/", 1)
        if kind == "train":
            _fake_finished_run(cfg, by_ckpt[ckpt])
            mvtrain.record_run(cfg, cluster, layout, by_ckpt[ckpt])
        else:
            _write_eval(env, _cand(cfg, cluster, layout, arms, ckpt), inputs, 1)

    _StubOrchestrator.calls, _StubOrchestrator.complete = [], staticmethod(complete)
    _StubOrchestrator.fail = {f"train/{runs[1].ckpt_id}"}
    lines = []
    rc = W.run_waves(cfg, cluster, layout, arms, wave=4, delete_exports=True, only=only, log=lines.append,
                     orchestrator=_StubOrchestrator)
    assert rc == 1
    names = [n for n, _ in _StubOrchestrator.calls]
    assert names == ["train_w0", "eval_w0", "train_w1", "eval_w1", "train_w2", "eval_w2"]
    calls = dict(_StubOrchestrator.calls)
    assert calls["train_w0"] == [f"train/{r.ckpt_id}" for r in runs[:4]]
    # the failed run is not evaluated; base joins the first eval wave only
    assert calls["eval_w0"] == ["eval/base"] + [f"eval/{r.ckpt_id}" for r in (runs[0], runs[2], runs[3])]
    assert calls["eval_w1"] == [f"eval/{r.ckpt_id}" for r in runs[4:8]] and calls["eval_w2"] == [f"eval/{runs[8].ckpt_id}"]
    out = "\n".join(lines)
    assert f"[w0] 1 runs not trained; their evals are skipped this pass" in out and runs[1].ckpt_id in out
    assert "8/9 runs trained, 8 exports deleted, 1 runs failed" in out and "all waves complete" not in out
    # exports of every evaluated checkpoint are gone, tombstoned, and still done
    for r in runs:
        if r is runs[1]:
            assert mvtrain.run_state(layout, cluster, r) == "pending"
        else:
            assert not r.export_dir.exists() and mvtrain.run_state(layout, cluster, r) == "pruned"
    rec = json.loads(layout.candidates_record.read_text())
    assert rec["candidates"]["base"]["scheduled"] and not rec["candidates"][runs[1].ckpt_id]["scheduled"]
    assert (layout.exp_dir / "train" / "runs.json").is_file()
    # the next pass retries only the failed run in its original wave, then completes
    _StubOrchestrator.calls, _StubOrchestrator.fail = [], set()
    lines.clear()
    rc = W.run_waves(cfg, cluster, layout, arms, wave=4, delete_exports=True, only=only, log=lines.append,
                     orchestrator=_StubOrchestrator)
    assert rc == 0
    assert _StubOrchestrator.calls == [("train_w0", [f"train/{runs[1].ckpt_id}"]), ("eval_w0", [f"eval/{runs[1].ckpt_id}"])]
    assert "9/9 runs trained, 1 exports deleted, 0 runs failed" in "\n".join(lines) and "all waves complete" in "\n".join(lines)
    assert all(mvtrain.run_state(layout, cluster, r) == "pruned" for r in runs)
    # a third pass submits nothing
    _StubOrchestrator.calls = []
    assert W.run_waves(cfg, cluster, layout, arms, wave=4, only=only, log=lines.append, orchestrator=_StubOrchestrator) == 0
    assert _StubOrchestrator.calls == []
    # without the flag nothing is deleted (fresh checkpoint)
    (runs[0].root / "run_record.json").unlink()
    _StubOrchestrator.calls = []
    assert W.run_waves(cfg, cluster, layout, arms, wave=4, only=only, log=lines.append, orchestrator=_StubOrchestrator) == 0
    assert _StubOrchestrator.calls == [("train_w0", [f"train/{runs[0].ckpt_id}"])]  # its evals are already done
    assert runs[0].export_dir.is_dir() and mvtrain.run_state(layout, cluster, runs[0]) == "done"


# ── dry run + driver ─────────────────────────────────────────────────────────


def test_dry_run_renders_every_wave_and_the_driver(mv_env, monkeypatch):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env)
    _inputs(env)
    runs = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)
    for var in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    lines = []
    assert W.run_waves(cfg, cluster, layout, arms, wave=4, delete_exports=True, only=only, dry_run=True,
                       log=lines.append) == 0
    out = "\n".join(lines)
    assert "run: 9 runs in 3 wave(s) of 4" in out and "dry run: rendered 3 wave(s)" in out
    assert "WARNING: grader" in out  # missing keys only warn under --dry-run
    d = layout.slurm_jobs_dir
    for k in range(3):
        assert (d / f"train_w{k}.sbatch").is_file() and (d / f"eval_w{k}.sbatch").is_file()
    t0 = (d / "train_w0.sbatch").read_text()
    assert f"--job-name=multivalue/{layout.exp_id}_train_w0" in t0 and "#SBATCH --array=0-3%2" in t0
    assert all(r.ckpt_id in t0 for r in runs[:4]) and runs[4].ckpt_id not in t0
    e0, e1, e2 = ((d / f"eval_w{k}.sbatch").read_text() for k in range(3))
    assert "--served-model-name base" in e0 and "--served-model-name base" not in e1
    assert "#SBATCH --array=0-4" in e0 and "#SBATCH --array=0-3" in e1 and "--array" not in e2  # base + 4, 4, 1
    assert f"--served-model-name {runs[8].ckpt_id}" in e2
    for s in ("toy_panel", "toy_items"):
        assert f"--suite {s}" in e0
    # the driver
    drv = (d / W.DRIVER_SCRIPT).read_text()
    assert f"--job-name=multivalue/{layout.exp_id}_run" in drv and "#SBATCH --partition=cpu\n#SBATCH --qos=cpu_qos" in drv
    assert "#SBATCH --time=1:00:00" in drv and "--gres" not in drv and "--array" not in drv
    assert f"NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID {d / W.DRIVER_SCRIPT})" in drv
    assert (f"python -m valuegen.multivalue.cli run -c {cfg.path} --cluster {cluster.source_path} --wave 4 "
            f"--delete-exports " + " ".join(f"--arm {a}" for a in only)) in drv
    assert drv.index("NEXT=$(sbatch") < drv.index("valuegen.multivalue.cli run")
    assert 'if [ "$rc" -eq 0 ] && [ -n "$NEXT" ]; then scancel "$NEXT"; fi\nexit $rc' in drv
    assert f'source "{cluster.venv("default")}/bin/activate"' in drv
    # cluster-specific placement: controller partition/qos + walltime cap + mandatory GPU
    ctrl = dataclasses.replace(cluster, controller_partition="ctrl", controller_qos="cq", cpu_stage_gpus=1,
                               partition_max_time={"ctrl": "2-00:00:00"})
    script = W.render_driver(cfg, ctrl, Layout(cfg, ctrl), wave=None, delete_exports=False, suites=["toy_panel"])
    drv = script.read_text()
    assert "#SBATCH --partition=ctrl\n#SBATCH --qos=cq\n#SBATCH --gres=gpu:1" in drv and "#SBATCH --time=2-00:00:00" in drv
    assert "--wave" not in drv and "--delete-exports" not in drv and "--suite toy_panel" in drv
    # --submit: refuses without sbatch, refuses a second live chain, never submits from the test suite
    lines.clear()
    assert W.submit_driver(cfg, cluster, layout, wave=4, dry_run=True, log=lines.append) == 0
    assert "not submitted" in "\n".join(lines)
    monkeypatch.setattr(W, "sbatch_available", lambda: False)
    assert W.submit_driver(cfg, cluster, layout, log=lines.append) == 1
    monkeypatch.setattr(W, "sbatch_available", lambda: True)
    monkeypatch.setattr(Orchestrator, "_find_active_job", lambda self, stage: 4242)
    assert W.submit_driver(cfg, cluster, layout, log=lines.append) == 1
    assert "already live (job 4242)" in "\n".join(lines)


def test_cli_run(mv_env, monkeypatch, capsys):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    monkeypatch.setattr(mv_cli, "load_cluster", lambda _p=None: cluster)
    c = ["-c", str(env["cfg_path"])]
    assert mv_cli.main(["run", *c, "--dry-run"]) == 1
    assert "mv sets" in capsys.readouterr().err  # not frozen
    arms, only, manifest = _prepared(env)
    _inputs(env)
    sel = [x for a in only for x in ("--arm", a)]
    assert mv_cli.main(["run", *c, "--dry-run", "--wave", "4", "--delete-exports", *sel]) == 0
    out = capsys.readouterr().out
    assert "3 wave(s) of 4" in out and "delete_exports=True" in out
    assert (layout.slurm_jobs_dir / "eval_w2.sbatch").is_file() and (layout.slurm_jobs_dir / W.DRIVER_SCRIPT).is_file()
    assert mv_cli.main(["run", *c, "--submit", "--dry-run", "--wave", "4", *sel]) == 0
    assert "not submitted" in capsys.readouterr().out
    assert mv_cli.main(["run", *c, "--dry-run", "--suite", "prefill", *sel]) == 1
    assert "disabled" in capsys.readouterr().err
    assert mv_cli.main(["run", *c, "--dry-run", "--wave", "0", *sel]) == 1
    assert "positive integer" in capsys.readouterr().err
