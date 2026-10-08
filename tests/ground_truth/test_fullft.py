"""Unit tests for first-class full fine-tuning (``use_peft: false``) support.

Everything here is render-time or CLI-validation logic — no GPUs, no torch.
The GPU truths (FSDP wiring, fp32 master weights surviving mixed precision)
are covered operationally by the sweep controller's smoke job and first-row
drift check, not here.
"""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from valuegen.config import ClusterConfig, GcsConfig, load_experiment
from valuegen.ground_truth import gcs, interventions, training

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
        gpu_type=None, max_concurrent_gpus=16, default_time="1:00:00",
        default_mem="4G", cpu_partition="cpu", cpu_qos="cpu_qos",
        source_path=tmp_path / "cluster.yaml",
    )


def _experiment(tmp_path, train_block: str = "", resources: str = ""):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: fullft_test\n"
        "intervention:\n"
        "  method: label_subset\n"
        "  value_set: constitution_tenets_v3\n"
        "  models: {olmo7b_sft: allenai/OLMo-2-1124-7B-SFT}\n"
        "  values: [calibrated_uncertainty]\n"
        + train_block
        + resources
        + "evaluation:\n"
        "  method: conflictscope\n"
        "  scenarios: data/scenarios/const_v3_cs\n"
    )
    return load_experiment(path)


FULL_FT_TRAIN = "  train:\n    use_peft: false\n    learning_rate: 5.0e-7\n"
LORA_TRAIN = "  train:\n    use_peft: true\n    lora_r: 32\n"
TORCHRUN_RES = "  resources:\n    train: {gpus: 8, nproc: 8}\n"


def test_full_ft_renders_export_instead_of_merge(cluster, tmp_path):
    cfg = _experiment(tmp_path, FULL_FT_TRAIN, TORCHRUN_RES)
    (stage,) = interventions.train_stages(cfg, cluster)
    (task,) = stage.tasks
    assert "valuegen.ground_truth.training export" in task.command
    assert "training merge" not in task.command
    assert "torchrun --nproc_per_node=8" in task.command
    # Port 0 rendezvous: two trainings on one node must not fight over 29500.
    assert "--rdzv-endpoint=localhost:0" in task.command
    # The export reads the trainer output root and lands at the merged_path
    # identity, so manifest/done/GCS plumbing never fork on the recipe.
    out_dir = interventions.train_output_dir(cfg, cluster, "olmo7b_sft", "calibrated_uncertainty")
    merged = interventions.merged_path(cfg, cluster, "olmo7b_sft", "calibrated_uncertainty")
    assert f"--trained {out_dir}" in task.command
    assert f"--out {merged}" in task.command
    assert not task.is_done()
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")
    assert task.is_done()


def test_absent_use_peft_means_full_ft(cluster, tmp_path):
    # The train block is forwarded verbatim to TRL, whose ModelConfig.use_peft
    # defaults to False — the stage list must agree with the trainer.
    cfg = _experiment(tmp_path)
    (stage,) = interventions.train_stages(cfg, cluster)
    assert "valuegen.ground_truth.training export" in stage.tasks[0].command


def test_train_chat_format_is_pinned_on_the_export(cluster, tmp_path):
    # train.chat_format reaches the trainer via the verbatim train block; the
    # export must carry the same explicit identity (and verify it), never the
    # path-substring family.
    cfg = _experiment(tmp_path, FULL_FT_TRAIN + "    chat_format: olmo3_chatml\n")
    (stage,) = interventions.train_stages(cfg, cluster)
    (task,) = stage.tasks
    assert "--chat-format olmo3_chatml" in task.command
    assert "--verify" in task.command
    plain = _experiment(tmp_path, FULL_FT_TRAIN)
    (stage,) = interventions.train_stages(plain, cluster)
    assert "--chat-format" not in stage.tasks[0].command
    assert "--verify" not in stage.tasks[0].command


def test_lora_still_renders_merge(cluster, tmp_path):
    cfg = _experiment(tmp_path, LORA_TRAIN)
    (stage,) = interventions.train_stages(cfg, cluster)
    (task,) = stage.tasks
    out_dir = interventions.train_output_dir(cfg, cluster, "olmo7b_sft", "calibrated_uncertainty")
    merged = interventions.merged_path(cfg, cluster, "olmo7b_sft", "calibrated_uncertainty")
    assert training.merge_command("allenai/OLMo-2-1124-7B-SFT", out_dir, merged) in task.command
    assert "training export" not in task.command


def test_nproc_gpus_mismatch_is_refused(cluster, tmp_path):
    cfg = _experiment(
        tmp_path, FULL_FT_TRAIN, "  resources:\n    train: {gpus: 2, nproc: 4}\n"
    )
    with pytest.raises(ValueError, match="nproc=4 but gpus=2"):
        interventions.train_stages(cfg, cluster)


# ── export CLI (validation precedes any torch import) ────────────────────────


def test_export_refuses_out_path_without_family_keyword(tmp_path):
    with pytest.raises(SystemExit):
        training.export([
            "--base-model", "base", "--trained", str(tmp_path / "raw"),
            "--out", str(tmp_path / "anonymous_dir"),
        ])


def test_export_refuses_disagreeing_chat_family(tmp_path):
    with pytest.raises(SystemExit):
        training.export([
            "--base-model", "base", "--trained", str(tmp_path / "raw"),
            "--out", str(tmp_path / "olmo7b_sft_calibrated_uncertainty_merged"),
            "--chat-family", "qwen",
        ])


def test_export_errors_on_missing_trainer_output(tmp_path):
    trained = tmp_path / "raw"
    trained.mkdir()
    with pytest.raises(SystemExit, match="No config.json"):
        training.export([
            "--base-model", "base", "--trained", str(trained),
            "--out", str(tmp_path / "olmo7b_sft_calibrated_uncertainty_merged"),
        ])


def test_export_is_idempotent(tmp_path, capsys):
    out = tmp_path / "olmo7b_sft_calibrated_uncertainty_merged"
    out.mkdir()
    (out / "config.json").write_text("{}")
    training.export([
        "--base-model", "base", "--trained", str(tmp_path / "raw"),
        "--out", str(out),
    ])
    assert "skipping" in capsys.readouterr().out


# ── checkpoint resolution and device placement ───────────────────────────────


def test_latest_checkpoint_falls_back_to_adapter_at_root(tmp_path):
    # save_strategy "no" leaves no checkpoint-* dirs; the final save_model
    # still lands a complete adapter at the root, which merge must accept.
    (tmp_path / "adapter_config.json").write_text("{}")
    assert training.latest_checkpoint(tmp_path) == tmp_path
    (tmp_path / "checkpoint-5").mkdir()
    assert training.latest_checkpoint(tmp_path).name == "checkpoint-5"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        training.latest_checkpoint(empty)


def test_resolve_device_map(monkeypatch):
    args = lambda **kw: SimpleNamespace(**{"deepspeed": None, "fsdp": None, **kw})
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert training._resolve_device_map(args()) == "auto"
    assert training._resolve_device_map(args(deepspeed="zero3.json")) is None
    # transformers parses `fsdp:` into a non-empty list of options.
    assert training._resolve_device_map(args(fsdp=["full_shard", "auto_wrap"])) is None
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert training._resolve_device_map(args()) is None
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert training._resolve_device_map(args()) == "auto"
