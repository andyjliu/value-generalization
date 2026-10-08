"""CLI command-line safety and dispatch tests."""

from __future__ import annotations

import argparse
import sys

import pytest

from valuegen import cli
from valuegen.elicitation import datasets as D
from valuegen.elicitation import registry


def test_parse_params_yaml_coercion_and_validation():
    assert cli._parse_params(["n_pairs=5", "flag=true", "name=hello"]) == {
        "n_pairs": 5, "flag": True, "name": "hello",
    }
    with pytest.raises(SystemExit, match="expects k=v"):
        cli._parse_params(["broken"])


def test_cost_gate_requires_explicit_approval_noninteractively(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="cold cache"):
        cli._confirm(["2 GPU tasks"], False, "test build")
    assert "will submit SLURM jobs" in capsys.readouterr().out
    cli._confirm(["2 GPU tasks"], True, "test build")


def test_predict_rejects_wrong_artifact_type(cluster, monkeypatch, tmp_path):
    cfg = registry.resolve_config(
        "descriptions", "constitution_tenets_v3", values=["hard_constraint_fidelity"]
    )
    artifact = D.resolve_artifact(cluster, cfg)
    artifact.root.mkdir(parents=True)
    artifact.descriptions_path.write_text("value,description\nno_affirm_delusions,test\n")
    D.persist_data_config(artifact.root, cfg)
    D.write_manifest(artifact)
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    args = argparse.Namespace(
        method="persona", value_set="constitution_tenets_v3", model="org/base",
        data=str(artifact.root), values=None, param=None, layer=None, pooling=None,
        encoder=None, gpus=1, build=False, dry_run=False, cluster=None,
    )
    with pytest.raises(SystemExit, match="requires a 'pairs' artifact"):
        cli.cmd_predict(args)


@pytest.fixture
def fork_dry_run_dir(tmp_path, monkeypatch, cluster):
    """Run the real, stdlib-only fork renderer from an isolated script copy."""
    import shutil
    import shlex

    from valuegen import _external

    root = tmp_path / "persona-fork"
    root.mkdir()
    source = _external.persona_vectors_root()
    shutil.copy2(source / "run_pipeline.py", root / "run_pipeline.py")
    # The parser reads this source file to enumerate template choices.
    (root / "eval").mkdir()
    shutil.copy2(source / "eval/model_utils.py", root / "eval/model_utils.py")
    monkeypatch.setattr(_external, "PERSONA_VECTORS", root)
    # The delegate activates this environment before invoking `python`.
    # Use the current test interpreter instead of the caller's shell PATH.
    bin_dir = cluster.venv("persona") / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    (bin_dir / "activate").write_text(f"export PATH={shlex.quote(str(bin_dir))}:\"$PATH\"\n")
    return "pytest_dry_run"


def test_predict_dry_run_auto_builds_without_cost_confirmation(
    cluster, monkeypatch, fork_dry_run_dir
):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    monkeypatch.setattr(cli, "_confirm", lambda *args: pytest.fail("dry-run confirmed cost"))
    args = argparse.Namespace(
        method="persona", value_set="constitution_tenets_v3", model="org/base",
        data="default_llm", values=["hard_constraint_fidelity"], param=[f"experiment={fork_dry_run_dir}"],
        layer=1, pooling=None, encoder=None, gpus=1, build=False, dry_run=True,
        cluster=None,
    )
    assert cli.cmd_predict(args) == 0
    script = next((cluster.repo / "slurm_jobs").glob("**/persona_extract.sbatch"))
    text = script.read_text()
    assert "--threshold 0" in text and ".venvs/persona-test/bin/activate" in text
    from valuegen._external import persona_vectors_root

    assert (persona_vectors_root() / "slurm_jobs" / fork_dry_run_dir / "base" / "stage1_generate.sh").is_file()


def test_main_exposes_elicitation_and_predict_commands():
    with pytest.raises(SystemExit) as parser_exit:
        cli.main(["data", "build", "--help"])
    assert parser_exit.value.code == 0
    with pytest.raises(SystemExit) as predict_exit:
        cli.main(["predict", "--help"])
    assert predict_exit.value.code == 0


@pytest.mark.parametrize("verb", ["run", "intervene", "eval", "status", "matrices"])
def test_main_exposes_split_ground_truth_commands(verb):
    with pytest.raises(SystemExit) as parser_exit:
        cli.main(["gt", verb, "--help"])
    assert parser_exit.value.code == 0


def _gt_args(config, verb="eval"):
    return argparse.Namespace(
        verb=verb, config=str(config), cluster=None, dry_run=False, no_wait=False
    )


def _write_experiment(tmp_path, name="dpo_eval", evaluation_extra=""):
    config = tmp_path / f"{name}.yaml"
    config.write_text(
        f"name: {name}\n"
        "intervention:\n"
        "  method: label_subset\n"
        "  value_set: constitution_tenets_v3\n"
        "  values: [calibrated_uncertainty]\n"
        "  models: {olmo7b_sft: allenai/OLMo-2-1124-7B-SFT}\n"
        "  labels: {dir: data/labels}\n"
        "evaluation:\n"
        "  method: conflictscope\n"
        "  scenarios: data/scenarios/const_v3_cs\n"
        "  judge: {model: gpt-4.1, api_base: http://judge/v1}\n"
        + evaluation_extra
    )
    return config


def test_gt_eval_never_builds_a_missing_intervention(
    cluster, monkeypatch, tmp_path, capsys
):
    # The headline guard of the split: `gt eval` scores an existing artifact.
    # Un-built payloads are an error pointing at `intervene`, never a build.
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    config = _write_experiment(tmp_path)

    assert cli.cmd_gt(_gt_args(config, "eval")) == 1
    err = capsys.readouterr().err
    assert "payloads missing" in err and "gt intervene" in err
    assert not (cluster.data / "interventions").exists()


def test_gt_eval_dangling_reference_raises(cluster, monkeypatch, tmp_path):
    # A reference is a promise the artifact exists; a dangling one must fail
    # loudly, not resolve to an implicit rebuild of someone else's work.
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    config = tmp_path / "ref.yaml"
    config.write_text(
        "name: reeval\n"
        "evaluation:\n"
        "  method: conflictscope\n"
        "  value_set: constitution_tenets_v3\n"
        "  intervention: nonexistent-0123456789ab\n"
        "  scenarios: data/scenarios/const_v3_cs\n"
        "  judge: {model: gpt-4.1, api_base: http://judge/v1}\n"
    )
    with pytest.raises(FileNotFoundError, match="gt intervene"):
        cli.cmd_gt(_gt_args(config, "eval"))



def test_gt_until_truncates_after_the_named_stage(cluster, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    pending = [argparse.Namespace(name=n) for n in ("serve_gen", "generate", "train", "eval")]
    monkeypatch.setattr(cli, "_sequence", lambda *a, **k: list(pending))
    # The inline dataset build is not under test (it would read judgment CSVs).
    monkeypatch.setattr(cli.interventions, "build_data", lambda cfg, cluster: None)
    seen = []
    monkeypatch.setattr(
        cli, "_run_orchestrator",
        lambda args, name, stages, cluster: seen.append([s.name for s in stages]) or 0,
    )
    config = _write_experiment(tmp_path)
    args = _gt_args(config, "intervene")
    args.until = "generate"
    assert cli.cmd_gt(args) == 0
    assert seen.pop() == ["serve_gen", "generate"]

    args.until = "no_such_stage"
    assert cli.cmd_gt(args) == 1
    assert "no such pending stage" in capsys.readouterr().err
    assert not seen


def test_controller_dry_run_renders_on_controller_partition(cluster, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    cluster.controller_partition, cluster.controller_qos = "ctl_part", "ctl_qos"
    script = tmp_path / "steps.sh"
    script.write_text("set -euo pipefail\nvaluegen gt run -c x.yaml\n")
    assert cli.main(["controller", str(script), "--time", "2-00:00:00", "--dry-run"]) == 0
    sbatch = (cluster.repo / "slurm_jobs" / "controller" / "steps.sbatch").read_text()
    assert "#SBATCH --partition=ctl_part" in sbatch
    assert "#SBATCH --qos=ctl_qos" in sbatch
    assert "#SBATCH --time=2-00:00:00" in sbatch
    assert f"bash {script}" in sbatch


def test_smoke_dry_run_places_the_gpu_job_from_cluster_yaml(cluster, monkeypatch):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    cluster.gpu_partition, cluster.gpu_qos, cluster.account = "gpu_part", "gpu_qos", "grant"
    assert cli.main(["smoke", "--dry-run"]) == 0
    sbatch = (cluster.repo / "slurm_jobs" / "smoke" / "cuda_smoke.sbatch").read_text()
    assert "#SBATCH --partition=gpu_part" in sbatch
    assert "#SBATCH --qos=gpu_qos" in sbatch
    assert "#SBATCH --account=grant" in sbatch
    assert "#SBATCH --gres=gpu:A6000:1" in sbatch
    assert f"bash {cluster.repo / 'tests/fixtures/cuda_smoke.sh'}" in sbatch
    assert "-m pytest" not in sbatch

    assert cli.main(["smoke", "--tests", "--dry-run"]) == 0
    sbatch = (cluster.repo / "slurm_jobs" / "smoke" / "smoke_tests.sbatch").read_text()
    assert "#SBATCH --time=01:30:00" in sbatch
    assert sbatch.rstrip().endswith("python -B -m pytest -p no:cacheprovider -rs tests/")


def test_controller_missing_script(cluster, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    with pytest.raises(SystemExit, match="no such file"):
        cli.main(["controller", str(tmp_path / "nope.sh"), "--dry-run"])
