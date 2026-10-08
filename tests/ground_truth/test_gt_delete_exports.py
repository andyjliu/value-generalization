"""`gt run` delete-after-wave (``schedule.delete_exports``): tombstone +
readiness split, the delete gates, the wave hook, and the ``gt eval`` guard."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from pathlib import Path

import pytest

from valuegen import cli
from valuegen.config import (
    ClusterConfig,
    GcsConfig,
    ensure_config_record,
    canonical_experiment,
    gt_id,
    intervention_id,
    load_experiment,
)
from valuegen.ground_truth import common, gcs, interventions, training
from valuegen.ground_truth.evals import conflictscope as conflictscope_eval
from valuegen.slurm import Orchestrator, Stage, Task

MTAG = "olmo7b_sft"
VALUES = ["calibrated_uncertainty", "respecting_user_autonomy", "hard_constraint_fidelity"]


@pytest.fixture
def cluster(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    return ClusterConfig(
        envs={"default": "test"}, repo=repo, finetune_root=tmp_path / "finetune",
        finetune_root_legacy=tmp_path / "legacy", data=tmp_path / "data",
        slurm_logs=tmp_path / "logs", mail_type="END", mail_user="test@example.com",
        gpu_type="A6000", max_concurrent_gpus=8, default_time="1:00:00",
        default_mem="4G", cpu_partition="cpu", cpu_qos="cpu_qos",
        source_path=tmp_path / "cluster.yaml",
    )


def _write_config(tmp_path, name="del_test", schedule="", judge="gpt-4.1") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"name: {name}\n"
        "intervention:\n"
        "  method: label_subset\n"
        "  value_set: constitution_tenets_v3\n"
        f"  models: {{{MTAG}: allenai/OLMo-2-1124-7B-SFT}}\n"
        f"  values: [{', '.join(VALUES)}]\n"
        "  labels: {dir: data/labels}\n"
        "evaluation:\n"
        "  method: conflictscope\n"
        "  scenarios: data/scenarios/const_v3_cs\n"
        f"  judge: {{model: {judge}, api_base: http://judge/v1}}\n"
        + schedule
    )
    return path


@pytest.fixture
def cfg(tmp_path):
    return load_experiment(_write_config(tmp_path))


def _entries(cfg, cluster) -> dict[str, dict]:
    manifest = interventions.build_manifest(cfg, cluster)
    return {e["value"]: e for e in manifest["interventions"]}


def _train(cfg, cluster, value, *, lora=False) -> tuple[Path, Path]:
    """Fake a finished train+finish: merged dir + trainer dir with shards."""
    merged = interventions.merged_path(cfg, cluster, MTAG, value)
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")
    (merged / "model-00001-of-00001.safetensors").write_bytes(b"w" * 64)
    out = interventions.train_output_dir(cfg, cluster, MTAG, value)
    (out / "checkpoint-10").mkdir(parents=True)
    (out / "checkpoint-10" / "optimizer.pt").write_bytes(b"o" * 32)
    (out / "global_step10").mkdir()
    (out / "trainer_state.json").write_text("{}")
    (out / "config.json").write_text("{}")
    if lora:
        (out / "adapter_model.safetensors").write_bytes(b"a" * 8)
        (out / "adapter_config.json").write_text("{}")
    else:
        (out / "model-00001-of-00001.safetensors").write_bytes(b"w" * 64)
        (out / "model.safetensors.index.json").write_text("{}")
        (out / "pytorch_model.bin").write_bytes(b"w" * 16)
    return merged, out


def _eval_stages(cfg, cluster) -> list[Stage]:
    manifest = interventions.build_manifest(cfg, cluster)
    return interventions.stamp(
        conflictscope_eval.stages(cfg, cluster, manifest),
        gt_id(cfg),
        common.gt_record_path(cfg, cluster),
    )


def _eval_task(cfg, cluster, value) -> Task:
    tasks = {t.key: t for s in _eval_stages(cfg, cluster) for t in s.tasks}
    return tasks[f"{MTAG}_{value}"]


def _score(cfg, cluster, value) -> None:
    """Fake a finished, gt_id-stamped eval of one checkpoint."""
    ensure_config_record(
        common.gt_record_path(cfg, cluster), gt_id(cfg), canonical_experiment(cfg)
    )
    task = _eval_task(cfg, cluster, value)
    task.done.parent.mkdir(parents=True, exist_ok=True)
    task.done.write_text("scenario,score\ns,1\n")
    assert task.is_done()


# ── schedule validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("block,match", [
    ("schedule: {wave: 2, delete_export: true}\n", "unknown keys"),
    ("schedule: {wave: 2, delete_exports: 1}\n", "must be a boolean"),
    ("schedule: {delete_exports: true}\n", "requires schedule.wave"),
    ("schedule: {wave: 2, stop_after_wave: 0}\n", "positive integer"),
    ("schedule: {wave: 2, stop_after_wave: true}\n", "positive integer"),
    ("schedule: {stop_after_wave: 1}\n", "requires schedule.wave"),
])
def test_schedule_delete_exports_is_validated(tmp_path, block, match):
    with pytest.raises(ValueError, match=match):
        load_experiment(_write_config(tmp_path, schedule=block))


def test_schedule_delete_exports_is_accepted_and_never_hashed(tmp_path, cfg):
    on = load_experiment(
        _write_config(tmp_path, schedule="schedule: {wave: 2, delete_exports: true}\n")
    )
    assert on["schedule"] == {"wave": 2, "delete_exports": True}
    assert gt_id(on) == gt_id(cfg)
    # false without a wave is inert, not an error
    load_experiment(_write_config(tmp_path, schedule="schedule: {delete_exports: false}\n"))


def _args(config, verb="run", **kw) -> argparse.Namespace:
    base = dict(verb=verb, config=str(config), cluster=None, dry_run=False,
                no_wait=False, delete_exports=None)
    return argparse.Namespace(**{**base, **kw})


def test_cli_flag_overrides_yaml_and_needs_a_wave(tmp_path, cluster):
    waved = load_experiment(_write_config(tmp_path, schedule="schedule: {wave: 2}\n"))
    on = load_experiment(
        _write_config(tmp_path, schedule="schedule: {wave: 2, delete_exports: true}\n")
    )
    plain = load_experiment(_write_config(tmp_path))
    assert not cli._delete_exports(waved, _args("x"), cluster)
    assert cli._delete_exports(waved, _args("x", delete_exports=True), cluster)
    assert cli._delete_exports(on, _args("x"), cluster)
    assert not cli._delete_exports(on, _args("x", delete_exports=False), cluster)
    with pytest.raises(SystemExit, match="needs schedule.wave"):
        cli._delete_exports(plain, _args("x", delete_exports=True), cluster)
    gcs_cluster = dataclasses.replace(
        cluster, gcs=GcsConfig(bucket="gs://b/v", stage_dir=Path("/mnt/ssd"))
    )
    with pytest.raises(SystemExit, match="push-and-pruned"):
        cli._delete_exports(on, _args("x"), gcs_cluster)


def test_gt_run_parser_takes_both_flag_spellings():
    parser = argparse.ArgumentParser()
    cli._gt_parser(parser.add_subparsers(dest="command"))
    with pytest.raises(SystemExit):  # the flag belongs to `run` alone
        parser.parse_args(["gt", "eval", "-c", "x", "--delete-exports"])
    assert parser.parse_args(["gt", "run", "-c", "x"]).delete_exports is None
    assert parser.parse_args(["gt", "run", "-c", "x", "--delete-exports"]).delete_exports is True
    assert parser.parse_args(["gt", "run", "-c", "x", "--no-delete-exports"]).delete_exports is False


def test_delete_exports_is_refused_with_no_wait(tmp_path, cluster, monkeypatch):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    monkeypatch.setattr(interventions, "build_data", lambda cfg, cluster: None)
    config = _write_config(tmp_path, schedule="schedule: {wave: 2, delete_exports: true}\n")
    with pytest.raises(SystemExit, match="drop\\s+--no-wait"):
        cli.cmd_gt(_args(config, no_wait=True, dry_run=True))


def test_submit_renders_a_self_resubmitting_controller(tmp_path, cluster, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    config = _write_config(tmp_path, schedule="schedule: {wave: 2, delete_exports: true}\n")
    args = _args(config, submit=True, dry_run=True, stop_after_wave=1, delete_exports=False,
                 until="train")
    assert cli.cmd_gt(args) == 0
    assert "not submitted" in capsys.readouterr().out
    name = gt_id(load_experiment(config))
    script = cluster.repo / "slurm_jobs" / name / cli.DRIVER_SCRIPT
    body = script.read_text()
    assert f"--job-name={name}_{cli.DRIVER_STAGE}" in body
    assert f"--dependency=afterany:$SLURM_JOB_ID {script}" in body
    assert f"gt run -c {config.resolve()}" in body
    assert "--no-delete-exports --stop-after-wave 1 --until train" in body
    assert "--submit" not in body and "--gres" not in body
    # The controller returning on its own -- pass or fail -- ends the chain.
    assert 'if [ -n "$NEXT" ]; then scancel "$NEXT"; fi' in body
    with pytest.raises(SystemExit, match="drop --no-wait"):
        cli.cmd_gt(_args(config, submit=True, no_wait=True, stop_after_wave=None))


# ── tombstone + readiness split ──────────────────────────────────────────────


def test_tombstone_reads_trained_but_not_loadable(cfg, cluster):
    entry = _entries(cfg, cluster)["calibrated_uncertainty"]
    merged = Path(entry["path"])
    (train,) = interventions.train_stages(cfg, cluster)
    task = next(t for t in train.tasks if t.key == f"{MTAG}/calibrated_uncertainty")
    assert not task.is_done() and not interventions.entry_ready(entry)
    assert interventions.tombstone(entry) is None

    merged.parent.mkdir(parents=True)
    gcs.deleted_marker_path(merged).write_text(json.dumps({"gt_id": "g"}))
    assert gcs.checkpoint_trained(merged) and not gcs.checkpoint_loadable(merged)
    assert task.is_done()  # never retrains — without a gcs: block too
    assert not interventions.entry_ready(entry)
    assert interventions.tombstone(entry) == {"gt_id": "g"}
    assert not gcs.ready(merged)  # mv's notion of servable is unchanged


# ── delete gates ─────────────────────────────────────────────────────────────


def test_delete_refuses_until_this_runs_eval_is_done(cfg, cluster):
    entry = _entries(cfg, cluster)["calibrated_uncertainty"]
    merged, _ = _train(cfg, cluster, "calibrated_uncertainty")
    task = _eval_task(cfg, cluster, "calibrated_uncertainty")

    deleted, why = interventions.delete_checkpoint(cfg, cluster, entry, [task])
    assert not deleted and "eval not done" in why
    deleted, why = interventions.delete_checkpoint(cfg, cluster, entry, [])
    assert not deleted and "no eval task" in why

    # An eval CSV stamped by a *different* GT run does not count.
    task.done.parent.mkdir(parents=True)
    task.done.write_text("scenario,score\ns,1\n")
    common.gt_record_path(cfg, cluster).write_text("config_id: someone-else\n")
    deleted, why = interventions.delete_checkpoint(cfg, cluster, entry, [task])
    assert not deleted and "eval not done" in why
    assert (merged / "config.json").is_file()
    assert interventions.tombstone(entry) is None


def test_delete_refuses_gcs_prompts_and_references(cfg, cluster):
    entry = _entries(cfg, cluster)["calibrated_uncertainty"]
    merged, _ = _train(cfg, cluster, "calibrated_uncertainty")
    _score(cfg, cluster, "calibrated_uncertainty")
    task = _eval_task(cfg, cluster, "calibrated_uncertainty")

    gcs_cluster = dataclasses.replace(
        cluster, gcs=GcsConfig(bucket="gs://b/v", stage_dir=Path("/mnt/ssd"))
    )
    deleted, why = interventions.delete_checkpoint(cfg, gcs_cluster, entry, [task])
    assert not deleted and "push-and-pruned" in why

    prompt = {**entry, "kind": "system_prompt"}
    deleted, why = interventions.delete_checkpoint(cfg, cluster, prompt, [task])
    assert not deleted and "never deleted" in why

    referenced = {**cfg, "intervention": None}
    deleted, why = interventions.delete_checkpoint(referenced, cluster, entry, [task])
    assert not deleted and "referenced" in why
    assert (merged / "config.json").is_file()


def test_delete_removes_merged_and_shards_keeps_small_files(cfg, cluster):
    entry = _entries(cfg, cluster)["calibrated_uncertainty"]
    merged, out = _train(cfg, cluster, "calibrated_uncertainty")
    _score(cfg, cluster, "calibrated_uncertainty")
    task = _eval_task(cfg, cluster, "calibrated_uncertainty")

    deleted, why = interventions.delete_checkpoint(cfg, cluster, entry, [task])
    assert deleted, why
    assert not merged.exists()
    assert not merged.with_name(merged.name + ".deleting").exists()
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json", "model.safetensors.index.json.pruned", "trainer_state.json",
    ]
    record = interventions.tombstone(entry)
    assert record["gt_id"] == gt_id(cfg) and record["eval_tasks"] == [task.key]
    assert record["by"] == "gt run --delete-exports"
    # merged (2 + 64) + trainer shards (64 + 16) + checkpoint-10 (32)
    assert record["freed_bytes"] == 2 + 64 + 64 + 16 + 32

    # Idempotent: the second call is a no-op, the tombstone is untouched.
    again, why = interventions.delete_checkpoint(cfg, cluster, entry, [task])
    assert not again and "already deleted" in why
    assert interventions.tombstone(entry) == record


def test_delete_finishes_a_delete_cut_short(cfg, cluster):
    entry = _entries(cfg, cluster)["calibrated_uncertainty"]
    merged, out = _train(cfg, cluster, "calibrated_uncertainty")
    # Crash after the tombstone, before the rename ...
    gcs.deleted_marker_path(merged).write_text(json.dumps({"gt_id": gt_id(cfg)}))
    deleted, _ = interventions.delete_checkpoint(cfg, cluster, entry, [])
    assert not deleted and not merged.exists()
    assert not list(out.glob("*.safetensors")) and not (out / "checkpoint-10").exists()
    # ... or after the rename, mid-rmtree.
    doomed = merged.with_name(merged.name + ".deleting")
    doomed.mkdir()
    (doomed / "model-00001-of-00001.safetensors").write_bytes(b"w")
    interventions.delete_checkpoint(cfg, cluster, entry, [])
    assert not doomed.exists()


def test_trainer_prune_keeps_the_lora_adapter(cfg, cluster, tmp_path):
    _, out = _train(cfg, cluster, "calibrated_uncertainty", lora=True)
    assert training.prune_trainer_dir(out, checkpoints=True)
    assert sorted(p.name for p in out.iterdir()) == [
        "adapter_config.json", "adapter_model.safetensors", "config.json",
        "trainer_state.json",
    ]
    # An adapter that only a checkpoint-* holds is lifted to the root first.
    only_ckpt = tmp_path / "lora_only_ckpt"
    (only_ckpt / "checkpoint-5").mkdir(parents=True)
    (only_ckpt / "checkpoint-5" / "adapter_model.safetensors").write_bytes(b"a")
    (only_ckpt / "checkpoint-5" / "adapter_config.json").write_text("{}")
    training.prune_trainer_dir(only_ckpt, checkpoints=True)
    assert sorted(p.name for p in only_ckpt.iterdir()) == [
        "adapter_config.json", "adapter_model.safetensors",
    ]
    # mv's call (no checkpoints=) leaves checkpoint dirs alone.
    _, out2 = _train(cfg, cluster, "respecting_user_autonomy")
    training.prune_trainer_dir(out2)
    assert (out2 / "checkpoint-10").is_dir() and not list(out2.glob("*.safetensors"))
    assert not training.prune_trainer_dir(tmp_path / "absent")


# ── trainer-save prune inside the training task ──────────────────────────────


@pytest.mark.parametrize("schedule,pruned", [
    ("", False),
    ("schedule: {wave: 2}\n", False),
    # Deleting after the evals implies nothing reads the trainer save again.
    ("schedule: {wave: 2, delete_exports: true}\n", True),
    ("schedule: {wave: 2, delete_exports: true, prune_trainer_save: false}\n", False),
    ("schedule: {prune_trainer_save: true}\n", True),
])
def test_train_task_prunes_the_trainer_save_after_export(tmp_path, cluster, schedule, pruned):
    on = load_experiment(_write_config(tmp_path, schedule=schedule))
    (stage,) = interventions.train_stages(on, cluster)
    assert all(("--prune-trained" in t.command) == pruned for t in stage.tasks)
    # Pacing only: the knob never reaches an identity.
    assert gt_id(on) == gt_id(load_experiment(_write_config(tmp_path)))


def test_prune_trainer_save_must_be_a_boolean(tmp_path):
    with pytest.raises(ValueError, match="must be a boolean"):
        load_experiment(_write_config(tmp_path, schedule="schedule: {prune_trainer_save: 1}\n"))


def test_export_prunes_only_after_the_export_verifies(cfg, cluster, monkeypatch):
    merged, out = _train(cfg, cluster, "calibrated_uncertainty")
    argv = ["--base-model", "base", "--trained", str(out), "--out", str(merged),
            "--prune-trained"]

    def failing(out_dir, chat_format=None):
        raise RuntimeError("export verification failed")

    monkeypatch.setattr(training, "verify_export", failing)
    with pytest.raises(RuntimeError):
        training.export(argv)
    assert list(out.glob("*.safetensors")) and (out / "checkpoint-10").is_dir()

    monkeypatch.setattr(training, "verify_export", lambda out_dir, chat_format=None: {"problems": []})
    training.export(argv)  # --prune-trained implies the verify
    assert (merged / "export_verification.json").is_file()
    assert not list(out.glob("*.safetensors")) and not (out / "checkpoint-10").exists()
    assert (out / "trainer_state.json").is_file()


# ── wave hook ────────────────────────────────────────────────────────────────


def _waved(cfg, cluster, delete=True, stop_after=None) -> list[Stage]:
    manifest = interventions.build_manifest(cfg, cluster)
    interventions.claim(cfg, cluster)  # the record train done-checks validate against
    train = interventions.stamp(
        interventions.train_stages(cfg, cluster),
        intervention_id(cfg), interventions.record_path(cfg, cluster),
    )
    hook = functools.partial(interventions.delete_checkpoint, cfg, cluster) if delete else None
    return interventions.interleave_waves(
        train, _eval_stages(cfg, cluster), manifest, wave=2, delete=hook,
        stop_after=stop_after,
    )


def test_wave_hook_covers_only_its_waves_values(cfg, cluster, capsys):
    for value in VALUES:
        _train(cfg, cluster, value)
        _score(cfg, cluster, value)
    by_name = {s.name: s for s in _waved(cfg, cluster)}
    assert all(s.after is None for n, s in by_name.items() if not n.startswith("eval_"))
    assert all(s.after is None for s in _waved(cfg, cluster, delete=False))

    by_name[f"eval_{MTAG}_w0"].after()
    entries = _entries(cfg, cluster)
    assert [v for v in VALUES if interventions.tombstone(entries[v])] == VALUES[:2]
    assert (Path(entries["hard_constraint_fidelity"]["path"]) / "config.json").is_file()
    out = capsys.readouterr().out
    assert out.count("checkpoint deleted") == 2 and f"{MTAG}_base" not in out

    # The last wave carries the base eval task, which is never a target.
    assert f"{MTAG}_base" in {t.key for t in by_name[f"eval_{MTAG}_w1"].tasks}
    by_name[f"eval_{MTAG}_w1"].after()
    assert interventions.tombstone(entries["hard_constraint_fidelity"]) is not None
    assert capsys.readouterr().out.count("delete_exports:") == 1


def test_stop_after_wave_is_a_prefix_of_the_run_and_never_hashed(tmp_path, cfg, cluster):
    stopped = load_experiment(
        _write_config(tmp_path, schedule="schedule: {wave: 2, stop_after_wave: 1}\n")
    )
    assert gt_id(stopped) == gt_id(cfg)
    assert intervention_id(stopped) == intervention_id(cfg)

    full = [s for s in _waved(cfg, cluster) if not s.server]
    pilot = [s for s in _waved(cfg, cluster, stop_after=1) if not s.server]
    assert [s.name for s in pilot] == [f"train_{MTAG}_w0", f"eval_{MTAG}_w0"]
    # Same train tasks as the full run's first wave; the eval wave also
    # carries the base task, which otherwise waits for the last wave.
    assert [t.key for t in pilot[0].tasks] == [t.key for t in full[0].tasks]
    assert [t.key for t in pilot[1].tasks] == (
        [t.key for t in full[1].tasks] + [f"{MTAG}_base"]
    )
    assert pilot[1].after is not None
    # A stop past the end is the whole run.
    assert [s.name for s in _waved(cfg, cluster, stop_after=99)] == [
        s.name for s in _waved(cfg, cluster)
    ]


def test_restart_refires_the_hook_without_double_delete(cfg, cluster, capsys):
    for value in VALUES[:2]:
        _train(cfg, cluster, value)
        _score(cfg, cluster, value)
    entries = _entries(cfg, cluster)

    def run_first_wave() -> list[str]:
        calls = []
        stages = [s for s in _waved(cfg, cluster) if s.name.endswith("_w0")]
        for s in stages:
            if s.after is not None:
                s.after = (lambda f=s.after: (calls.append(1), f())[1])
        # Train + eval of wave 0 are done: run_stage completes with nothing
        # pending and never reaches sbatch.
        assert Orchestrator("t", stages, cluster).run()
        return calls

    assert run_first_wave() == [1]
    records = {v: interventions.tombstone(entries[v]) for v in VALUES[:2]}
    assert all(records.values())
    capsys.readouterr()
    assert run_first_wave() == [1]  # restarted controller: hook re-fires ...
    assert capsys.readouterr().out.count("already deleted") == 2  # ... as a no-op
    assert {v: interventions.tombstone(entries[v]) for v in VALUES[:2]} == records


def test_dry_run_never_fires_the_hook(cluster):
    fired = []
    stage = Stage(name="eval_x_w0", tasks=[], time="1:00:00", mem="4G",
                  after=lambda: fired.append(1))
    assert Orchestrator("t", [stage], cluster, dry_run=True).run()
    assert not fired
    assert Orchestrator("t", [stage], cluster).run()
    assert fired == [1]


def test_hook_does_not_fire_for_an_incomplete_stage(cluster, monkeypatch):
    fired = []
    stage = Stage(name="eval_x_w0", tasks=[Task("k", "true", done=lambda: False)],
                  time="1:00:00", mem="4G", after=lambda: fired.append(1))
    orch = Orchestrator("t", [stage], cluster)
    monkeypatch.setattr(orch, "run_stage", lambda s: (False, None))
    assert not orch.run()
    assert not fired


# ── gt eval guard + status ───────────────────────────────────────────────────


def _delete_all(cfg, cluster) -> None:
    for value, entry in _entries(cfg, cluster).items():
        _train(cfg, cluster, value)
        _score(cfg, cluster, value)
        deleted, why = interventions.delete_checkpoint(
            cfg, cluster, entry, [_eval_task(cfg, cluster, value)]
        )
        assert deleted, why


def test_gt_eval_passes_when_tombstoned_checkpoints_are_all_scored(
    tmp_path, cluster, monkeypatch
):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    ran = []
    monkeypatch.setattr(
        cli, "_run_orchestrator", lambda args, name, stages, cluster: ran.append(name) or 0
    )
    config = _write_config(tmp_path)
    cfg = load_experiment(config)
    _delete_all(cfg, cluster)
    assert cli.cmd_gt(_args(config, "eval")) == 0
    assert ran == [gt_id(cfg)]


def test_gt_eval_refuses_tombstoned_checkpoints_with_pending_evals(
    tmp_path, cluster, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    monkeypatch.setattr(
        cli, "_run_orchestrator", lambda *a: pytest.fail("submitted unscoreable evals")
    )
    monkeypatch.setattr(interventions, "build_data", lambda cfg, cluster: None)
    cfg = load_experiment(_write_config(tmp_path))
    _delete_all(cfg, cluster)

    # Same intervention (same id, same tombstones), a different eval config.
    other = _write_config(tmp_path, name="del_test", judge="gpt-4.1-mini")
    other_cfg = load_experiment(other)
    assert gt_id(other_cfg) != gt_id(cfg)
    for verb in ("eval", "run"):  # `run` must not read the tombstones as "train me"
        assert cli.cmd_gt(_args(other, verb)) == 1
        err = capsys.readouterr().err
        assert f"deleted after gt_id {gt_id(cfg)}" in err
        assert ".deleted" in err and "gt intervene" in err


def test_gt_status_lists_pruned_checkpoints(tmp_path, cluster, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    config = _write_config(tmp_path, schedule="schedule: {wave: 2, delete_exports: true}\n")
    cfg = load_experiment(config)
    _delete_all(cfg, cluster)
    assert cli.cmd_gt(_args(config, "status")) == 0
    out = capsys.readouterr().out
    assert "pruned (checkpoint deleted after its evals): 3/3" in out
    assert f"- {MTAG}_calibrated_uncertainty" in out
    # Tombstoned train tasks read done, so every train wave is complete.
    assert f"train_{MTAG}_w0" in out and "OK 2/2" in out
