"""Unit tests for opt-in GCS checkpoint storage (config, snippets, wiring)."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from valuegen.config import ClusterConfig, GcsConfig, load_cluster, load_experiment
from valuegen.ground_truth import gcs, interventions
from valuegen.ground_truth.evaluation import EvalSpec, eval_command, eval_tasks

REPO = Path(__file__).resolve().parents[2]

GCS = GcsConfig(bucket="gs://test-bucket/valuegen", stage_dir=Path("/mnt/localssd/u"))


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


@pytest.fixture
def experiment(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: gcs_test\n"
        "intervention:\n"
        "  method: label_subset\n"
        "  value_set: constitution_tenets_v3\n"
        "  models: {olmo7b_sft: allenai/OLMo-2-1124-7B-SFT}\n"
        "  values: [calibrated_uncertainty]\n"
        "evaluation:\n"
        "  method: conflictscope\n"
        "  scenarios: data/scenarios/const_v3_cs\n"
    )
    return load_experiment(path)


# ── cluster.yaml surface ─────────────────────────────────────────────────────


def _write_cluster_yaml(tmp_path, gcs_block=None) -> Path:
    raw = {
        "envs": {"default": ".venvs/core"},
        "paths": {
            "repo": str(tmp_path), "finetune_root": str(tmp_path / "ft"),
            "finetune_root_legacy": str(tmp_path / "ft"), "data": "data",
            "slurm_logs": "logs",
        },
        "slurm": {
            "mail_type": "END", "mail_user": "t@example.com", "gpu_type": "A100",
            "max_concurrent_gpus": 8, "default_time": "1:00:00",
            "default_mem": "4G", "cpu_partition": "cpu", "cpu_qos": "cpu_qos",
        },
    }
    if gcs_block is not None:
        raw["gcs"] = gcs_block
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_gcs_block_is_opt_in(tmp_path):
    assert load_cluster(_write_cluster_yaml(tmp_path)).gcs is None
    disabled = {"enabled": False, "bucket": "gs://b", "stage_dir": "/scratch"}
    assert load_cluster(_write_cluster_yaml(tmp_path, disabled)).gcs is None
    enabled = {"enabled": True, "bucket": "gs://b/prefix/", "stage_dir": "/scratch"}
    loaded = load_cluster(_write_cluster_yaml(tmp_path, enabled)).gcs
    assert loaded == GcsConfig(bucket="gs://b/prefix", stage_dir=Path("/scratch"))


def test_gcs_block_validates_bucket_and_stage_dir(tmp_path):
    with pytest.raises(ValueError, match="gs://"):
        load_cluster(_write_cluster_yaml(
            tmp_path, {"enabled": True, "bucket": "s3://b", "stage_dir": "/s"}))
    with pytest.raises(ValueError, match="stage_dir is required"):
        load_cluster(_write_cluster_yaml(
            tmp_path, {"enabled": True, "bucket": "gs://b"}))
    with pytest.raises(ValueError, match="absolute"):
        load_cluster(_write_cluster_yaml(
            tmp_path, {"enabled": True, "bucket": "gs://b", "stage_dir": "rel"}))


# ── snippet rendering ────────────────────────────────────────────────────────


def test_push_orders_rsync_then_marker_then_prune(tmp_path):
    merged = tmp_path / "iv_olmo7b_sft_calibrated_uncertainty_merged"
    snippet = gcs.push_and_prune_sh(GCS, merged)
    marker = str(gcs.marker_path(merged))
    assert snippet.startswith(f"if [ ! -s {marker} ]")  # idempotent on retry
    uri = "gs://test-bucket/valuegen/merged/iv_olmo7b_sft_calibrated_uncertainty_merged"
    rsync = snippet.index(f"{merged} {uri}")
    write_marker = snippet.index(f"{uri} > {marker}")
    prune = snippet.index(f"rm -rf {merged}")
    assert rsync < write_marker < prune


def test_push_from_scratch_keeps_marker_canonical(tmp_path):
    merged = tmp_path / "gt" / "iv_olmo7b_sft_calibrated_uncertainty_merged"
    local = gcs.staged_path(GCS, merged)
    snippet = gcs.push_and_prune_sh(GCS, merged, local_dir=local)
    uri = gcs.remote_uri(GCS, merged)
    marker = str(gcs.marker_path(merged))
    assert f"{local} {uri}" in snippet
    # Nothing else creates the canonical parent when the export lands on
    # scratch, and the marker must be written there before the prune.
    assert snippet.index(f"mkdir -p {merged.parent}") < snippet.index(f"> {marker}")
    assert f"rm -rf {local}" in snippet and f"rm -rf {merged}\n" not in snippet


def test_stage_in_preserves_basename_and_prefers_local(tmp_path):
    merged = tmp_path / "iv_olmo7b_sft_calibrated_uncertainty_merged"
    snippet = gcs.stage_in_sh(GCS, merged)
    # Chat templates are selected by path substring at eval time, so the
    # staged dir must keep the merged dir's name.
    assert "/mnt/localssd/u/merged/iv_olmo7b_sft_calibrated_uncertainty_merged" in snippet
    assert snippet.index(f"if [ -f {merged}/config.json ]") < snippet.index("rsync")


def test_ready_accepts_local_dir_or_upload_marker(tmp_path):
    merged = tmp_path / "m_merged"
    assert not gcs.ready(merged)
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    assert gcs.ready(merged)
    (merged / "config.json").unlink()
    gcs.marker_path(merged).write_text("gs://b/merged/m_merged\n")
    assert gcs.ready(merged)


# ── training-side wiring ─────────────────────────────────────────────────────


def test_train_stages_are_unchanged_with_gcs_off(cluster, experiment):
    (stage,) = interventions.train_stages(experiment, cluster)
    (task,) = stage.tasks
    assert "gcloud" not in task.command
    merged = interventions.merged_path(experiment, cluster, "olmo7b_sft", "calibrated_uncertainty")
    # Done = merged/config.json on disk (or a delete_exports tombstone, see
    # test_gt_delete_exports) — no upload marker involved.
    assert not task.is_done()
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")
    assert task.is_done()


def test_train_stages_push_and_guard_with_gcs_on(cluster, experiment):
    cluster = dataclasses.replace(cluster, gcs=GCS)
    (stage,) = interventions.train_stages(experiment, cluster)
    (task,) = stage.tasks
    merged = interventions.merged_path(experiment, cluster, "olmo7b_sft", "calibrated_uncertainty")
    # Train+merge is guarded so a retry after a failed upload doesn't retrain.
    assert task.command.index("if [ ! -s") < task.command.index("training dpo")
    assert "gcloud storage rsync" in task.command
    assert not task.is_done()
    merged.parent.mkdir(parents=True)
    gcs.marker_path(merged).write_text(gcs.remote_uri(GCS, merged) + "\n")
    assert task.is_done()


def test_train_finish_writes_to_scratch_with_gcs_on(cluster, experiment):
    cluster = dataclasses.replace(cluster, gcs=GCS)
    (stage,) = interventions.train_stages(experiment, cluster)
    (task,) = stage.tasks
    merged = interventions.merged_path(experiment, cluster, "olmo7b_sft", "calibrated_uncertainty")
    staged = gcs.staged_path(GCS, merged)
    assert f"--out {staged}" in task.command  # full-FT export (use_peft unset)
    assert f"{staged} {gcs.remote_uri(GCS, merged)}" in task.command
    # An HF-id init is passed through literally, with no stage-in prelude.
    assert "MODEL_DIR" not in task.command


def test_train_stages_stage_in_a_checkpoint_init(cluster, tmp_path):
    init = tmp_path / "finetune" / "neutral_sft_v3_olmo-3_7b_merged"
    path = tmp_path / "exp_init.yaml"
    path.write_text(
        "name: gcs_init_test\n"
        "intervention:\n"
        "  method: label_subset\n"
        "  value_set: constitution_tenets_v3\n"
        f"  models: {{neutral_olmo-3_7b: {init}}}\n"
        "  values: [calibrated_uncertainty]\n"
        "evaluation:\n"
        "  method: conflictscope\n"
        "  scenarios: data/scenarios/const_v3_cs\n"
    )
    experiment = load_experiment(path)
    (plain,) = interventions.train_stages(experiment, cluster)
    assert f"--model_name_or_path {init}" in plain.tasks[0].command
    assert "MODEL_DIR" not in plain.tasks[0].command

    cluster = dataclasses.replace(cluster, gcs=GCS)
    (stage,) = interventions.train_stages(experiment, cluster)
    command = stage.tasks[0].command
    # The init may already be pruned to the bucket: resolve it at run time,
    # before the trainer, and have trainer and merge both read $MODEL_DIR.
    assert command.index("MODEL_DIR=") < command.index("training dpo")
    assert "--model_name_or_path \"$MODEL_DIR\"" in command
    assert "--base-model \"$MODEL_DIR\"" in command
    assert gcs.remote_uri(GCS, init) in command


def test_entry_ready_accepts_marker_only_checkpoints(tmp_path):
    merged = tmp_path / "m_merged"
    entry = {"kind": "checkpoint", "path": str(merged)}
    assert not interventions.entry_ready(entry)
    gcs.marker_path(merged).write_text("gs://b/merged/m_merged\n")
    assert interventions.entry_ready(entry)


# ── eval-side wiring ─────────────────────────────────────────────────────────


def test_eval_interventions_carry_stage_in_only_with_gcs(cluster, experiment):
    manifest = interventions.build_manifest(experiment, cluster)
    plain = interventions.eval_interventions(manifest, "olmo7b_sft", cluster)
    assert "stage_in" not in plain["olmo7b_sft_calibrated_uncertainty"]
    with_gcs = interventions.eval_interventions(
        manifest, "olmo7b_sft", dataclasses.replace(cluster, gcs=GCS)
    )
    assert "gcloud storage rsync" in with_gcs["olmo7b_sft_calibrated_uncertainty"]["stage_in"]


def test_eval_command_stages_in_and_loads_model_dir(tmp_path):
    spec = EvalSpec(tmp_path / "scenarios", tmp_path / "out")
    merged = tmp_path / "iv_olmo7b_sft_calibrated_uncertainty_merged"
    stage_in = gcs.stage_in_sh(GCS, merged)
    tasks = eval_tasks(
        spec, "base",
        {"ft": {"model": str(merged), "stage_in": stage_in}},
        "base",
    )
    staged = tasks[1].command
    assert '-m "$MODEL_DIR"' in staged
    assert staged.index("MODEL_DIR=") < staged.index("evaluate_models.py")
    # Base task (an HF name, never remote) keeps the literal -m argument.
    assert "-m base" in tasks[0].command
    plain = eval_command(spec, str(merged), "ft.csv")
    assert f"-m {merged}" in plain and "MODEL_DIR" not in plain


def test_base_slot_checkpoint_is_staged_in(cluster, experiment, tmp_path):
    from valuegen.ground_truth.evals import conflictscope

    spec = EvalSpec(tmp_path / "scenarios", tmp_path / "out")
    init = tmp_path / "finetune" / "neutral_sft_v3_qwen3_8b_merged"
    plain = conflictscope._shared_base_task(
        spec, experiment, cluster, "neutral_qwen3_8b", str(init)
    )
    assert "MODEL_DIR" not in plain.command
    staged = conflictscope._shared_base_task(
        spec, experiment, dataclasses.replace(cluster, gcs=GCS),
        "neutral_qwen3_8b", str(init),
    )
    assert '-m "$MODEL_DIR"' in staged.command
    assert gcs.remote_uri(GCS, init) in staged.command
    # An HF id in the base slot is never remote.
    hf = conflictscope._shared_base_task(
        spec, experiment, dataclasses.replace(cluster, gcs=GCS),
        "olmo7b_sft", "allenai/OLMo-2-1124-7B-SFT",
    )
    assert "MODEL_DIR" not in hf.command
