from pathlib import Path
import subprocess
import shutil
import tempfile

import pytest

from valuegen.config import ClusterConfig


@pytest.fixture(scope="session")
def staged_qwen_model():
    """One self-contained copy per test process; resolve cache symlinks onto scratch.

    Downloads the pinned ~1.5 GB checkpoint on first use. Everything after this
    fixture runs offline against the staged copy.
    """
    from huggingface_hub import snapshot_download

    try:
        source = snapshot_download("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
    except Exception as error:
        pytest.fail(
            "could not fetch the Qwen/Qwen3-0.6B test checkpoint "
            f"({type(error).__name__}: {error}). On a node without network access, "
            "run the suite once from a node that has it (same HF_HOME) so the "
            "snapshot is cached, or deselect these tests with "
            "-m 'not model_smoke and not e2e'.", pytrace=False)
    with tempfile.TemporaryDirectory(prefix="valuegen-qwen-") as directory:
        base = Path(directory) / "qwen3-base"
        shutil.copytree(source, base)
        yield base


@pytest.fixture(autouse=True)
def isolated_slurm(monkeypatch):
    """Scheduler tests must supply responses instead of contacting real SLURM."""
    real_run = subprocess.run

    def run(args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and args:
            command = Path(args[0]).name
            if command in {"squeue", "sacct", "scancel"}:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            if command == "sbatch":
                pytest.fail("unexpected SLURM submission; mock the scheduler in this test")
            if list(args) == ["which", "sbatch"]:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return real_run(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


@pytest.fixture
def cluster(tmp_path: Path) -> ClusterConfig:
    """Small, fully isolated cluster config for elicitation/predictor contract tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    real_repo = Path(__file__).resolve().parents[1]
    (repo / "value_sets").symlink_to(real_repo / "value_sets", target_is_directory=True)
    source = tmp_path / "cluster.yaml"
    source.write_text("test cluster\n")
    return ClusterConfig(
        envs={
            "default": ".venvs/test",
            "eval_api": ".venvs/test",
            "persona": ".venvs/persona-test",
            "weight_steering": ".venvs/test",
            "ws_train": ".venvs/ws-train-test",
        },
        repo=repo,
        finetune_root=tmp_path / "finetune",
        finetune_root_legacy=tmp_path / "legacy-finetune",
        data=tmp_path / "data",
        slurm_logs=tmp_path / "logs",
        mail_type="END",
        mail_user="test@example.com",
        gpu_type="A6000",
        max_concurrent_gpus=8,
        default_time="1:00:00",
        default_mem="4G",
        cpu_partition="cpu",
        cpu_qos="cpu_qos",
        exclude=["slow-a", "slow-b"],
        source_path=source,
    )
