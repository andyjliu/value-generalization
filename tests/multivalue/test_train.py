"""The train stage (`mv train`): planning, sbatch rendering, run records, done predicates."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from valuegen.config import GcsConfig
from valuegen.ground_truth.chat_formats import get_chat_format
from valuegen.multivalue import _hashing as H
from valuegen.multivalue import cli as mv_cli
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import train as mvtrain
from valuegen.multivalue.sets import Arm

from test_data import _buildable, _fake_rq3_experiment, _freeze


def _prepared(env, tmp_path=None):
    """Frozen design + mixes for the buildable arms (+ one imported experiment when tmp_path is given)."""
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms = _freeze(env)
    only = _buildable(arms)
    if tmp_path is not None:
        universe, _ = mvdata.resolve_universe(cfg, cluster)
        mvimports.import_experiment(cfg, cluster, layout, _fake_rq3_experiment(tmp_path, env), universe=universe,
                                    log=lambda *_: None)
    arms = mvimports.all_arms(layout)
    manifest = mvdata.build_mixes(cfg, cluster, layout, arms, token_audit_enabled=False, only=only, log=lambda *_: None)
    return arms, only, manifest


def _fake_finished_run(cfg, run: mvtrain.Run, *, dataset_sha256=None, world_size=None, steps=None):
    """Lay down the files a finished train+export leaves, consistent with the run's expectations."""
    fmt = get_chat_format(cfg.train.chat_format)
    exp = run.expect
    steps = exp["expect_max_steps"] if steps is None else steps
    run.trainer_dir.mkdir(parents=True, exist_ok=True)
    run.export_dir.mkdir(parents=True, exist_ok=True)
    (run.trainer_dir / "config.json").write_text("{}")
    (run.trainer_dir / "model-00001-of-00001.safetensors").write_bytes(b"trainer-weights")
    (run.trainer_dir / "model.safetensors.index.json").write_text("{}")
    mvdata.write_json(run.trainer_dir / "run_metadata.json", {
        "final_save": "done", "dataset_sha256": dataset_sha256 or run.dataset_sha256,
        "n_train_rows": exp["expect_raw_rows"], "n_train_rows_after_trainer_preprocessing": exp["expect_train_rows"],
        "observed_max_steps": steps, "observed_global_step": steps, "observed_warmup_steps": exp["expect_warmup_steps"],
        "world_size": world_size if world_size is not None else int(cfg.train.resources["gpus"]), "seed": run.seed,
        "chat_format": {"chat_format": fmt.name, "template_sha256": fmt.template_sha256},
        "base_revision": cfg.train.revision,
    })
    mvdata.write_json(run.trainer_dir / "step_geometry.json", {"max_steps": steps})
    mvdata.write_json(run.trainer_dir / "dpo_integrity.json",
                      {"policy_changed": True, "reference_fixed": True, "losses_finite": True})
    (run.export_dir / "config.json").write_text("{}")
    (run.export_dir / "model.safetensors").write_bytes(b"export-weights")
    mvdata.write_json(run.export_dir / "export_verification.json",
                      {"problems": [], "sidecar": {"chat_format": fmt.name, "template_sha256": fmt.template_sha256}})


def test_plan_runs_skips_base_and_imported_and_checks_mixes(mv_env, tmp_path):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env, tmp_path)
    kinds = {a.kind for a in arms}
    assert {"base", "imported", "sampled", "explicit"} <= kinds
    assert {a.id for a in mvtrain.trainable_arms(arms)} == {a.id for a in arms if a.kind not in ("base", "imported")}
    runs = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)
    assert [r.ckpt_id for r in runs] == [f"{a}_s42" for a in sorted(only, key=lambda x: (x.split("_")[0] == "sub3", x))]
    r0 = runs[0]
    assert r0.root == layout.checkpoint_dir(r0.arm.id, 42) and r0.trainer_dir.name == "trainer"
    assert r0.expect == {"expect_raw_rows": 60, "expect_train_rows": 60, "expect_max_steps": 30, "expect_warmup_steps": 3}
    # the arm without a mix, the base arm and an imported arm are refused by name
    with pytest.raises(mvtrain.TrainError, match="not in the mixes manifest"):
        mvtrain.plan_runs(cfg, layout, arms, manifest, only=["probe_00"])
    with pytest.raises(mvtrain.TrainError, match="not trainable"):
        mvtrain.plan_runs(cfg, layout, arms, manifest, only=["base"])
    with pytest.raises(mvtrain.TrainError, match="not trainable"):
        mvtrain.plan_runs(cfg, layout, arms, manifest, only=["ext_a"])
    with pytest.raises(mvtrain.TrainError, match="unknown arm"):
        mvtrain.plan_runs(cfg, layout, arms, manifest, only=["nope"])
    with pytest.raises(mvtrain.TrainError, match="not in train.seeds"):
        mvtrain.plan_runs(cfg, layout, arms, manifest, only=only, seeds=[7])
    # a mix rebuilt behind the manifest's back is refused
    comp_path = layout.mix_dir(only[0]) / "composition.json"
    comp = json.loads(comp_path.read_text())
    comp["dataset_sha256"] = "0" * 64
    comp_path.write_text(json.dumps(comp))
    with pytest.raises(mvtrain.TrainError, match="differs from the manifest"):
        mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)


def test_manifest_gates(mv_env):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    _freeze(env)
    with pytest.raises(mvtrain.TrainError, match="mv data"):
        mvtrain.load_manifest(layout)
    arms = mvimports.all_arms(layout)
    mvdata.build_mixes(cfg, cluster, layout, arms, token_audit_enabled=False, only=_buildable(arms), log=lambda *_: None)
    m = mvtrain.load_manifest(layout)
    m["gate_failures"] = ["sub3_00: 3 TRL drops"]
    mvdata.write_json(layout.mixes_dir / "manifest.json", m)
    with pytest.raises(mvtrain.TrainError, match="gate failures"):
        mvtrain.load_manifest(layout)
    m["gate_failures"], m["exp_id"] = [], "other-000000000000"
    mvdata.write_json(layout.mixes_dir / "manifest.json", m)
    with pytest.raises(mvtrain.TrainError, match="exp_id"):
        mvtrain.load_manifest(layout)


def test_train_dry_run_renders_the_stage(mv_env, tmp_path):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env, tmp_path)
    lines = []
    assert mvtrain.train(cfg, cluster, layout, arms, dry_run=True, only=only, log=lines.append) == 0
    assert any("9 runs (9 arms x 1 seeds), 9 pending" in ln for ln in lines)
    script = (layout.slurm_jobs_dir / "train.sbatch").read_text()
    assert "#SBATCH --array=0-8%2" in script and "#SBATCH --gres=gpu:A6000:2" in script
    assert 'export WANDB_MODE="disabled"' in script
    assert "torchrun --nproc_per_node=2" in script
    assert f"--config {Path(cfg.train.recipe).resolve()}" in script
    assert f"--model_name_or_path {cfg.train.base_model}" in script
    assert "--dataset_name " + str(layout.mix_dir("sub3_00") / "dataset.jsonl") in script
    assert f"--output_dir {layout.checkpoint_dir('sub3_00', 42) / 'trainer'}" in script
    assert "--seed 42" in script and "--chat_format qwen_chatml" in script
    assert f"--fsdp_config {Path(cfg.train.fsdp_config).resolve()}" in script
    assert "--expect_raw_rows 60" in script and "--expect_train_rows 60" in script
    assert "--expect_max_steps 30" in script and "--expect_warmup_steps 3" in script and "--expect_world_size 2" in script
    assert "--model_revision" not in script and "--base_revision" not in script  # no revision pinned in the fixture
    assert "valuegen.ground_truth.training export" in script and "--chat-format qwen_chatml" in script and "--verify" in script
    assert f"-c {cfg.path} --record sub3_00_s42 --cluster {cluster.source_path}" in script
    assert "gcloud" not in script
    task_map = json.loads((layout.slurm_jobs_dir / "train_task_map.json").read_text())
    assert task_map["0"] == "train/probe_01_s42" and task_map["1"] == "train/sub3_00_s42"
    runs_rec = json.loads((layout.exp_dir / "train" / "runs.json").read_text())
    assert set(runs_rec["runs"]) == {f"{a}_s42" for a in only}
    assert runs_rec["runs"]["sub3_00_s42"]["export_dir"] == str(layout.checkpoint_dir("sub3_00", 42) / "export")
    assert runs_rec["runs"]["sub3_00_s42"]["gcs_uri"] is None
    # a pinned revision rides as both the load revision and the recorded base revision
    pinned = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, revision="abc123"))
    cmd = mvtrain.train_command(pinned, cluster, layout, mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0])
    assert "--model_revision abc123" in cmd and "--base_revision abc123" in cmd and "--base-revision abc123" in cmd


def test_record_run_verifies_prunes_and_gates_the_done_predicate(mv_env):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms, only, manifest = _prepared(env)
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    stage = mvtrain.build_stage(cfg, cluster, layout, [run])
    assert mvtrain.run_state(layout, cluster, run) == "pending" and stage.pending() == stage.tasks

    _fake_finished_run(cfg, run)
    assert mvtrain.run_state(layout, cluster, run) == "exported" and not mvtrain.run_done(layout, cluster, run)
    rec = mvtrain.record_run(cfg, cluster, layout, run)
    assert rec["export_verified"] and rec["problems"] == [] and rec["identity"] == run.identity(layout.exp_id)
    assert rec["export_sha256"] == {"model.safetensors": H.sha256_file(run.export_dir / "model.safetensors")}
    assert rec["trainer_save_pruned"] and not (run.trainer_dir / "model-00001-of-00001.safetensors").exists()
    assert (run.trainer_dir / "model.safetensors.index.json.pruned").is_file()
    assert (run.trainer_dir / "config.json").is_file()  # the finished-trainer guard still holds
    assert mvtrain.run_done(layout, cluster, run) and mvtrain.run_state(layout, cluster, run) == "done"
    assert stage.pending() == []

    # a record from another identity (a different design or dataset) never counts
    stale = dict(json.loads(run.record_path.read_text()))
    stale["identity"] = {**stale["identity"], "dataset_sha256": "f" * 64}
    mvdata.write_json(run.record_path, stale)
    assert not mvtrain.run_done(layout, cluster, run) and mvtrain.run_state(layout, cluster, run) == "stale"

    # inconsistent trainer output -> problems, not done, and no pruning
    _fake_finished_run(cfg, run, dataset_sha256="e" * 64, world_size=8, steps=31)
    no_prune = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, prune_trainer_save=False))
    rec = mvtrain.record_run(no_prune, cluster, layout, run)
    assert not rec["export_verified"] and not rec["trainer_save_pruned"]
    assert any("trained dataset sha" in p for p in rec["problems"])
    assert any("world size 8" in p for p in rec["problems"])
    assert any("steps 31/31" in p for p in rec["problems"]) and any("step geometry" in p for p in rec["problems"])
    assert (run.trainer_dir / "model-00001-of-00001.safetensors").is_file()
    assert mvtrain.run_state(layout, cluster, run) == "failed-verify"
    # missing files are reported by name
    _fake_finished_run(cfg, run)
    (run.trainer_dir / "dpo_integrity.json").unlink()
    (run.export_dir / "export_verification.json").unlink()
    problems, _ = mvtrain.verify_run(cfg, layout, run)
    assert "no dpo_integrity.json" in problems and "no export_verification.json" in problems


def test_gcs_cluster_pushes_the_export_and_accepts_the_marker(mv_env):
    env = mv_env
    cfg, layout = env["cfg"], env["layout"]
    cluster = dataclasses.replace(env["cluster"], gcs=GcsConfig(bucket="gs://bucket/mv", stage_dir=Path("/scratch/mv")))
    arms, only, manifest = _prepared(env)
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    cmd = mvtrain.train_command(cfg, cluster, layout, run)
    marker = str(run.export_dir) + ".gcs"
    assert "gcloud storage rsync" in cmd and f"gs://bucket/mv/merged/{run.export_dir.name}" in cmd
    assert cmd.index("--record") < cmd.index("gcloud storage rsync")  # record (hashes the export) before pushing
    assert f"[ ! -s {marker} ]" in cmd  # the whole train/export/record block is skipped once pushed
    _fake_finished_run(cfg, run)
    rec = mvtrain.record_run(cfg, cluster, layout, run)
    assert rec["export_verified"]
    # simulate push-and-prune: export gone, marker present
    for f in run.export_dir.iterdir():
        f.unlink()
    run.export_dir.rmdir()
    assert not mvtrain.run_done(layout, cluster, run)
    Path(marker).write_text(f"gs://bucket/mv/merged/{run.export_dir.name}\n")
    assert mvtrain.run_done(layout, cluster, run) and mvtrain.run_state(layout, cluster, run) == "done"
    assert not mvtrain.run_done(layout, env["cluster"], run)  # without a gcs block the marker is not evidence
    rec_path = mvtrain.write_runs_record(layout, [run], cluster)
    assert json.loads(rec_path.read_text())["runs"][run.ckpt_id]["gcs_uri"] == f"gs://bucket/mv/merged/{run.export_dir.name}"


def test_cli_train(mv_env, tmp_path, monkeypatch, capsys):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    monkeypatch.setattr(mv_cli, "load_cluster", lambda _p=None: cluster)
    c = ["-c", str(env["cfg_path"])]
    assert mv_cli.main(["train", *c, "--dry-run"]) == 1
    assert "mv sets" in capsys.readouterr().err  # not frozen
    arms, only, manifest = _prepared(env, tmp_path)
    assert mv_cli.main(["train", *c, "--dry-run"]) == 1
    assert "probe_00: not in the mixes manifest" in capsys.readouterr().err
    sel = [x for a in only for x in ("--arm", a)]
    assert mv_cli.main(["train", *c, "--dry-run", *sel]) == 0
    out = capsys.readouterr().out
    assert "9 pending" in out and "dry run: rendered" in out and (layout.slurm_jobs_dir / "train.sbatch").is_file()
    # --record on a fabricated finished run, then the stage reports it done
    run = mvtrain.plan_runs(cfg, layout, arms, manifest, only=only)[0]
    _fake_finished_run(cfg, run)
    assert mv_cli.main(["train", *c, "--record", run.ckpt_id]) == 0
    assert json.loads(capsys.readouterr().out)["export_verified"] is True
    assert mv_cli.main(["train", *c, "--record", "sub3_00_s7"]) == 1
    assert "not a run of this experiment" in capsys.readouterr().err
    assert mv_cli.main(["train", *c, "--dry-run", *sel]) == 0
    out = capsys.readouterr().out
    assert "8 pending" in out and f"{run.ckpt_id}" not in out.split("dry run")[0].split("pending")[1]
    # submitting without sbatch is refused (no job is ever launched from the test suite)
    monkeypatch.setattr(mvtrain, "sbatch_available", lambda: False)
    assert mv_cli.main(["train", *c, *sel]) == 1
    assert "sbatch not found" in capsys.readouterr().out
