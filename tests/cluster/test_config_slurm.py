"""Unit tests for the deterministic config/SLURM command and orchestration seams."""

import dataclasses
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import subprocess
import yaml

from valuegen.config import (
    ClusterConfig, canonical_experiment, config_hash, ensure_config_record,
    intervention_id, load_cluster, load_experiment,
)
from valuegen.ground_truth import training
from valuegen.ground_truth.evaluation import (
    EvalSpec, eval_command, eval_tasks, model_view, serve_stage,
)
from valuegen.ground_truth.evals import conflictscope as conflictscope_eval
from valuegen.slurm import Orchestrator, Stage, Task


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cluster(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "value_sets").symlink_to(REPO / "value_sets", target_is_directory=True)
    return ClusterConfig(
        envs={"default": "test", "eval_api": "test-api", "persona": "persona", "weight_steering": "ws"},
        repo=repo, finetune_root=tmp_path / "finetune", finetune_root_legacy=tmp_path / "legacy",
        data=tmp_path / "data", slurm_logs=tmp_path / "logs", mail_type="END",
        mail_user="test@example.com", gpu_type="A6000", max_concurrent_gpus=8,
        default_time="1:00:00", default_mem="4G", cpu_partition="cpu", cpu_qos="cpu_qos",
        exclude=["slow-a", "slow-b"], source_path=tmp_path / "cluster.yaml",
    )


@pytest.mark.parametrize("external_data", [False, True], ids=["relative-data", "external-data"])
@pytest.mark.parametrize("separate_eval", [False, True], ids=["shared-env", "separate-env"])
def test_config_contract(tmp_path, monkeypatch, external_data, separate_eval):
    repo = tmp_path / "repo"
    data = tmp_path / "large-disk" / "data" if external_data else repo / "data"
    envs = {"default": ".venvs/core", "eval_api": ".venvs/core", "ws_train": ".venvs/ws-train"}
    if separate_eval:
        envs["eval_api"] = str(tmp_path / "eval-env")
    raw = {
        "envs": envs,
        "paths": {
            "repo": str(repo), "data": str(data) if external_data else "data",
            "finetune_root": str(tmp_path / "finetune"),
            "finetune_root_legacy": str(tmp_path / "legacy"), "slurm_logs": "logs",
        },
        "slurm": {
            "mail_type": "END", "mail_user": "test@example.com", "gpu_type": "A100",
            "max_concurrent_gpus": 8, "default_time": "1:00:00", "default_mem": "4G",
            "cpu_partition": "cpu", "cpu_qos": "cpu_qos", "exclude": ["slow-a", "slow-b"],
        },
    }
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_cluster(path)
    assert cfg.repo == repo
    assert cfg.data == data
    assert cfg.slurm_logs == repo / "logs"
    excludes = raw["slurm"]["exclude"]
    assert cfg.exclude == excludes
    assert cfg.exclude_arg == ",".join(excludes)
    monkeypatch.setenv("VALUEGEN_CLUSTER", str(path))
    expected_eval = tmp_path / "eval-env" if separate_eval else repo / ".venvs/core"
    assert load_cluster().venv("eval_api") == expected_eval
    assert load_cluster().venv("default") == repo / ".venvs/core"
    assert load_cluster().venv("ws_train") == repo / ".venvs/ws-train"
    assert load_experiment(REPO / "configs/experiments/ground_truth/const_v3_dpo_full49.yaml")["name"] == "const_v3_dpo_full49"


def test_config_hash_uses_resolved_semantics_and_validates_reuse(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        "# formatting is not identity\nname: hash_test\n"
        "intervention:\n  method: none\n  value_set: constitution_tenets_v3\n"
        "  models: {olmo7b_sft: allenai/OLMo-2-1124-7B-SFT}\n"
        "evaluation:\n  method: conflictscope\n"
        "  scenarios: data/scenarios/const_v3_cs\n"
    )
    second.write_text(
        "name: hash_test  # same semantics\n"
        "intervention: {method: none, value_set: constitution_tenets_v3, models: {olmo7b_sft: allenai/OLMo-2-1124-7B-SFT}}\n"
        "evaluation: {method: conflictscope, scenarios: data/scenarios/const_v3_cs}\n"
    )
    cfg1, cfg2 = load_experiment(first), load_experiment(second)
    assert config_hash(cfg1) == config_hash(cfg2)
    assert intervention_id(cfg1).startswith("hash_test-")
    changed = dict(cfg1)
    changed["intervention"] = dict(cfg1["intervention"], values=["calibrated_uncertainty"])
    assert config_hash(changed) != config_hash(cfg1)

    record = ensure_config_record(
        tmp_path / "run/resolved_config.yaml",
        intervention_id(cfg1),
        canonical_experiment(cfg1["intervention"]),
    )
    output = tmp_path / "done.csv"
    output.write_text("complete")
    task = Task("done", "true", output, intervention_id(cfg1), record)
    assert task.is_done()
    record.write_text("config_id: wrong\n")
    assert not task.is_done()


def test_training_commands_and_checkpoint_selection(tmp_path):
    assert training.infer_chat_family("tulu-3-8b-llama") == "tulu"
    assert training.infer_chat_family("unrelated") is None
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-17").mkdir()
    assert training.latest_checkpoint(tmp_path).name == "checkpoint-17"
    with pytest.raises(FileNotFoundError):
        training.latest_checkpoint(tmp_path / "empty")
    one = training.train_command("dpo", "train.yaml", "pairs.csv", "base", "out", "run")
    many = training.train_command("sft", "train.yaml", "pairs.csv", "base", "out", "run", nproc=4)
    assert one.startswith("python -m valuegen.ground_truth.training dpo")
    assert "torchrun --nproc_per_node=4" in many
    assert "-m valuegen.ground_truth.training sft" in many
    assert "--merged-path merged" in training.merge_command("base", "out", "merged")


def test_eval_commands_preserve_n_plus_one(tmp_path):
    spec = EvalSpec(tmp_path / "scenarios", tmp_path / "out", temperature=0, max_tokens=99,
                    max_scenarios=7, user_model="user", judge_model="judge",
                    assistant_api_base="http://assistant", wait_for={"judge": tmp_path / "judge.host"})
    command = eval_command(spec, "base", "base.csv", tmp_path / "steer.txt")
    for flag in ("wait_ready()", "--temperature 0", "--max-tokens 99", "--max-scenarios 7", "--cache", "--filter", "--steer-prompt"):
        assert flag in command
    tasks = eval_tasks(spec, "base", {"ft": {"model": "ft"}, "prompt": {"steer_prompt": tmp_path / "p.txt"}}, "base")
    assert [t.key for t in tasks] == ["base", "ft", "prompt"]
    assert "-m ft" in tasks[1].command
    assert "--assistant-scaffold" not in command
    # A scaffold needs an in-process assistant; the fork refuses a served one.
    with pytest.raises(ValueError, match="in-process assistant"):
        eval_command(spec, "base", "base.csv", scaffold=tmp_path / "urial0.json")
    in_process = replace(spec, assistant_api_base=None)
    scaffolded = eval_command(in_process, "base", "base.csv", scaffold=tmp_path / "urial0.json")
    assert f"--assistant-scaffold {tmp_path / 'urial0.json'}" in scaffolded



def test_eval_command_seeds_the_first_turn_cache_without_clobbering(tmp_path):
    seed = tmp_path / "scenarios" / "cache.json"
    seed.parent.mkdir()
    seed.write_text('{"seed": 1}')
    out = tmp_path / "out"
    spec = EvalSpec(seed.parent, out, cache_seed=seed)
    command = eval_command(spec, "base", "base.csv")
    snippet = command[command.index("mkdir -p"):command.index("python ")]
    run = lambda: subprocess.run(["bash", "-c", snippet], capture_output=True, text=True)
    run()
    assert (out / "cache.json").read_text() == '{"seed": 1}'
    (out / "cache.json").write_text('{"extended": 1}')  # a task grew the cache
    run()
    assert (out / "cache.json").read_text() == '{"extended": 1}'
    assert not list(out.glob(".cache.json.seed*"))
    # No seed and no cache: warn (first turns would be regenerated), write nothing.
    (out / "cache.json").unlink()
    seed.unlink()
    assert "WARNING" in run().stderr and not (out / "cache.json").exists()
    # Not interactive, or cache off: no seeding at all.
    assert "cache.json" not in eval_command(replace(spec, cache=False), "base", "base.csv")


def test_fetch_eval_inputs_verifies_hashes(tmp_path):
    import hashlib
    import importlib.util

    mod_spec = importlib.util.spec_from_file_location(
        "fetch_eval_inputs", REPO / "scripts" / "fetch_eval_inputs.py")
    fetch = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(fetch)
    blobs = {hub: f"payload {i}".encode() for i, hub in enumerate(fetch.FILES)}
    src = tmp_path / "hub"
    src.mkdir()
    for i, (hub, data) in enumerate(blobs.items()):
        (src / str(i)).write_bytes(data)
    paths = {hub: str(src / str(i)) for i, hub in enumerate(blobs)}
    pinned = {hub: (name, hashlib.sha256(blobs[hub]).hexdigest())
              for hub, (name, _) in fetch.FILES.items()}
    fetch.FILES = pinned
    dest = tmp_path / "const_v3_cs"
    assert len(fetch.fetch(dest, download=paths.__getitem__)) == 2
    assert fetch.fetch(dest, download=paths.__getitem__) == []  # idempotent
    (dest / "cache.json").write_text("tampered")
    with pytest.raises(SystemExit, match="--force"):
        fetch.fetch(dest, download=paths.__getitem__)
    fetch.fetch(dest, force=True, download=paths.__getitem__)
    fetch.FILES = {h: (n, "0" * 64) for h, (n, _) in pinned.items()}
    with pytest.raises(SystemExit, match="expected"):
        fetch.fetch(tmp_path / "fresh", download=paths.__getitem__)


def test_checkpoint_paths_that_misread_as_huge_models_get_a_symlink_view(tmp_path):
    """conflictscope sizes vLLM by regexing ``(\\d+)b`` out of the model path.

    A 12-hex artifact ID can start with digits-then-``b``
    (``...n4000-94b43cf7f484_olmo7b_base_...`` reads 94B before 7B), which
    asked tp=5 on a 1-GPU eval task and killed every task in the array. Such a
    path is evaluated through a symlink view; paths that already read correctly
    keep their exact command, so finished runs are untouched.
    """
    base = "allenai/OLMo-2-1124-7B"
    root = tmp_path / "merged"
    bad = root / "aft_sanitized13_n4000-94b43cf7f484_olmo7b_base_respecting_user_autonomy_merged"
    good = root / "aft_full-3962e482167e_olmo7b_base_calibrated_uncertainty_merged"

    viewed, prelude = model_view(str(bad), base)
    assert prelude is not None
    assert "94b" not in viewed.lower()
    assert f"ln -sfn {bad} {viewed}" in prelude
    # The view resolves to the base model's own size, not the hash's.
    from valuegen.ground_truth.evaluation import _inferred_gpus
    assert _inferred_gpus(viewed) == _inferred_gpus(base) == 1

    assert model_view(str(good), base) == (str(good), None)
    assert model_view(base, base) == (base, None)

    spec = EvalSpec(tmp_path / "scenarios", tmp_path / "out")
    tasks = eval_tasks(spec, base, {"respecting_user_autonomy": {"model": str(bad)},
                                    "calibrated_uncertainty": {"model": str(good)}}, "base")
    assert "ln -sfn" in tasks[1].command and f"-m {viewed}" in tasks[1].command
    assert "ln -sfn" not in tasks[2].command and f"-m {good}" in tasks[2].command


def test_serve_stage_can_load_local_snapshot_under_stable_model_name(tmp_path):
    snapshot = tmp_path / "snapshot"
    stage = serve_stage(
        "serve_judge", "org/model", tmp_path / "host", load_path=snapshot,
    )
    command = stage.tasks[0].command
    assert f"vllm serve {snapshot}" in command
    assert "--served-model-name org/model" in command

    hub_stage = serve_stage("serve_judge", "org/model", tmp_path / "host")
    assert "vllm serve org/model" in hub_stage.tasks[0].command
    assert "--served-model-name" not in hub_stage.tasks[0].command


def test_orchestrator_renders_gpu_and_cpu_stages(cluster, tmp_path):
    gpu = Stage("gpu", [Task("a", "echo gpu", tmp_path / "a"), Task("b", "echo gpu", tmp_path / "b")], "1:00:00", "8G", gpus=2, gpu_headroom=2)
    cpu = Stage("cpu", [Task("c", "echo cpu", tmp_path / "c")], "1:00:00", "1G")
    orch = Orchestrator("gt_test", [gpu, cpu], cluster, dry_run=True)
    assert orch.run()
    gpu_script = (cluster.repo / "slurm_jobs/gt_test/gpu.sbatch").read_text()
    cpu_script = (cluster.repo / "slurm_jobs/gt_test/cpu.sbatch").read_text()
    assert "#SBATCH --gres=gpu:A6000:2" in gpu_script
    assert "#SBATCH --exclude=slow-a,slow-b" in gpu_script
    assert "#SBATCH --array=0-1%3" in gpu_script
    assert "#SBATCH --partition=cpu" in cpu_script
    assert "--gres" not in cpu_script
    assert "--exclude" not in cpu_script


def test_gpu_only_cluster_attaches_gpu_to_cpu_stages(cluster, tmp_path):
    """A GPU-only cluster: every partition needs >=1 GPU, so 0-GPU stages
    get ``cpu_stage_gpus`` untyped GPUs, the excludes, and a GPU throttle."""
    cluster = replace(cluster, cpu_partition="general", cpu_qos="normal", cpu_stage_gpus=1)
    tasks = [Task(k, "echo", tmp_path / k) for k in "abcdefghijkl"]
    cpu = Stage("cpu", tasks, "1:00:00", "1G")
    orch = Orchestrator("gt_gpuonly", [cpu], cluster, dry_run=True)
    assert orch.run()
    script = (cluster.repo / "slurm_jobs/gt_gpuonly/cpu.sbatch").read_text()
    assert "#SBATCH --partition=general" in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --gres=gpu:1\n" in script  # untyped: any GPU will do
    assert "#SBATCH --exclude=slow-a,slow-b" in script
    assert "#SBATCH --array=0-11%8" in script


def test_multiple_gpu_types_render_constraint(cluster, tmp_path):
    cluster = replace(cluster, gpu_type=["L40S", "A6000"])
    gpu = Stage("gpu", [Task("a", "echo", tmp_path / "a")], "1:00:00", "8G", gpus=2)
    orch = Orchestrator("gt_multi", [gpu], cluster, dry_run=True)
    assert orch.run()
    script = (cluster.repo / "slurm_jobs/gt_multi/gpu.sbatch").read_text()
    assert "#SBATCH --gres=gpu:2\n" in script
    assert "#SBATCH --constraint=L40S|A6000" in script
    assert cluster.gpu_types == ["L40S", "A6000"]


def test_untyped_gres_omits_the_type_segment(cluster, tmp_path):
    # Clusters that report `GresTypes = gpu` advertise a bare `gpu:N` on every
    # node; a typed `--gres=gpu:<type>:N` matches nothing there and every GPU
    # job fails with "Requested node configuration is not available".
    untyped = dataclasses.replace(cluster, gpu_type=None)
    stage = Stage("gpu", [Task("a", "echo gpu", tmp_path / "a")], "1:00:00", "8G", gpus=2)
    assert Orchestrator("gt_test", [stage], untyped, dry_run=True).run()
    script = (untyped.repo / "slurm_jobs/gt_test/gpu.sbatch").read_text()
    assert "#SBATCH --gres=gpu:2" in script
    assert "gpu:None" not in script


def test_preamble_falls_back_to_hf_api_key_for_hf_token(cluster, tmp_path):
    # .env names the Hub secret HF_API_KEY; the hub client only reads
    # HF_TOKEN, so a job resolving a private base repo by id 401'd. The
    # export must come after activate() (which sources .env) and must not
    # clobber an HF_TOKEN that is already set.
    stage = Stage("hf", [Task("a", "echo", tmp_path / "a")], "1:00:00", "8G", gpus=1)
    assert Orchestrator("hf_token", [stage], cluster, dry_run=True).run()
    script = (cluster.repo / "slurm_jobs/hf_token/hf.sbatch").read_text()
    line = 'export HF_TOKEN="${HF_TOKEN:-${HF_API_KEY:-}}"'
    assert line in script
    assert script.index(cluster.activate(stage.env)) < script.index(line)
    for env, want in [({"HF_API_KEY": "k"}, "k"), ({"HF_TOKEN": "t", "HF_API_KEY": "k"}, "t"), ({}, "")]:
        out = subprocess.run(
            ["bash", "-c", f'set -u; {line}; printf %s "$HF_TOKEN"'],
            env={"PATH": os.environ["PATH"], **env}, capture_output=True, text=True, check=True,
        ).stdout
        assert out == want


def test_nccl_flags_are_opt_in_per_stage_or_cluster(cluster, tmp_path):
    # NCCL_P2P_DISABLE/NCCL_IB_DISABLE used to be emitted on every multi-GPU
    # stage; on NVLink nodes they roughly halve FSDP training and TP serving
    # throughput, so they are now opt-in (per stage, or cluster-wide for
    # hardware where P2P is actually broken).
    stages = [
        Stage("plain", [Task("a", "echo", tmp_path / "a")], "1:00:00", "8G", gpus=2),
        Stage("opted", [Task("b", "echo", tmp_path / "b")], "1:00:00", "8G", gpus=2, nccl_conservative=True),
        Stage("single", [Task("c", "echo", tmp_path / "c")], "1:00:00", "8G", gpus=1, nccl_conservative=True),
    ]
    assert Orchestrator("nccl_stage", stages, cluster, dry_run=True).run()
    scripts = cluster.repo / "slurm_jobs/nccl_stage"
    assert "NCCL_P2P_DISABLE" not in (scripts / "plain.sbatch").read_text()
    opted = (scripts / "opted.sbatch").read_text()
    assert "export NCCL_P2P_DISABLE=1" in opted
    assert "export NCCL_IB_DISABLE=1" in opted
    # The flags only matter for multi-GPU collectives; a 1-GPU stage never
    # emits them, opted in or not.
    assert "NCCL_P2P_DISABLE" not in (scripts / "single.sbatch").read_text()

    conservative = dataclasses.replace(cluster, nccl_conservative=True)
    stage = Stage("plain", [Task("d", "echo", tmp_path / "d")], "1:00:00", "8G", gpus=2)
    assert Orchestrator("nccl_cluster", [stage], conservative, dry_run=True).run()
    assert "export NCCL_P2P_DISABLE=1" in (
        conservative.repo / "slurm_jobs/nccl_cluster/plain.sbatch"
    ).read_text()


def test_no_wait_submission_wires_stage_dependencies(cluster, tmp_path, monkeypatch):
    stages = [
        Stage("first", [Task("first", "echo first", tmp_path / "first")], "1:00:00", "1G"),
        Stage("server", [Task("server", "echo server", lambda: False)], "1:00:00", "1G", server=True),
        Stage("last", [Task("last", "echo last", tmp_path / "last")], "1:00:00", "1G"),
    ]
    orch = Orchestrator("dependency_test", stages, cluster)
    dependencies = []
    ids = iter([101, 102, 103])

    def fake_sbatch(script, dependency=None):
        dependencies.append(dependency)
        return next(ids)

    monkeypatch.setattr(orch, "_sbatch", fake_sbatch)
    orch.submit()
    assert dependencies == [None, "afterok:101", "after:102"]


def test_no_wait_chains_a_reaper_with_afterany(cluster, tmp_path, monkeypatch):
    # A reaper must fire whether or not the eval it follows succeeded, so it
    # chains afterany (not afterok) — otherwise a failed eval leaks the judge.
    stages = [
        Stage("server", [Task("server", "echo server", lambda: False)], "1:00:00", "1G", server=True),
        Stage("eval", [Task("eval", "echo eval", tmp_path / "eval")], "1:00:00", "1G"),
        Stage("reap", [Task("reap", "scancel -n x", lambda: False)], "0:05:00", "1G", reaper=True),
    ]
    orch = Orchestrator("reaper_test", stages, cluster)
    dependencies = []
    ids = iter([501, 502, 503])
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: dependencies.append(dependency) or next(ids))
    orch.submit()
    # server (no dep) -> eval (after:server) -> reap (afterany:eval)
    assert dependencies == [None, "after:501", "afterany:502"]


def test_polling_run_skips_the_reaper_stage(cluster, tmp_path, monkeypatch):
    # run() reaps servers itself, so the reaper job must never be submitted there.
    submitted = []
    stages = [
        Stage("work", [Task("work", "echo", tmp_path / "work")], "1:00:00", "1G"),
        Stage("reap", [Task("reap", "scancel -n x", lambda: False)], "0:05:00", "1G", reaper=True),
    ]
    (tmp_path / "work").write_text("done")  # work already complete -> no submit
    orch = Orchestrator("reaper_skip", stages, cluster)
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: submitted.append(Path(script).stem) or 1)
    assert orch.run()
    assert "reap" not in submitted


def test_no_wait_chains_sibling_train_stages_with_aftercorr(cluster, tmp_path, monkeypatch):
    # train_stages() keys tasks "{mtag}/{value}" and iterates the same values
    # list per model, so index i means the same value in every train_{mtag}
    # array — aftercorr lets model B's value i start as soon as model A's
    # value i finishes, instead of waiting for model A's stragglers.
    stages = [
        Stage("train_a", [Task("a/v1", "echo", tmp_path / "a1"), Task("a/v2", "echo", tmp_path / "a2")], "1:00:00", "1G"),
        Stage("train_b", [Task("b/v1", "echo", tmp_path / "b1"), Task("b/v2", "echo", tmp_path / "b2")], "1:00:00", "1G"),
    ]
    orch = Orchestrator("aftercorr_test", stages, cluster)
    dependencies = []
    ids = iter([201, 202])
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: dependencies.append(dependency) or next(ids))
    orch.submit()
    assert dependencies == [None, "aftercorr:201"]


def test_no_wait_falls_back_to_afterok_when_pending_values_diverge(cluster, tmp_path, monkeypatch):
    # A resumed run can leave sibling train_{mtag} stages with different
    # values still pending (e.g. model A is missing v1, model B is missing
    # v2) — same array length, but index i no longer means the same value.
    # aftercorr would silently miswire that; must fall back to afterok.
    stages = [
        Stage("train_a", [Task("a/v2", "echo", tmp_path / "a2")], "1:00:00", "1G"),
        Stage("train_b", [Task("b/v1", "echo", tmp_path / "b1")], "1:00:00", "1G"),
    ]
    orch = Orchestrator("divergent_resume", stages, cluster)
    dependencies = []
    ids = iter([301, 302])
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: dependencies.append(dependency) or next(ids))
    orch.submit()
    assert dependencies == [None, "afterok:301"]


def test_no_wait_does_not_use_aftercorr_outside_train_stages(cluster, tmp_path, monkeypatch):
    # Same-shape, same-key-suffix arrays that aren't train_{mtag} stages
    # (e.g. two eval_{mtag} stages, which prepend a base task the train
    # stages don't have) must not opt into aftercorr just by coincidence.
    stages = [
        Stage("eval_a", [Task("a/v1", "echo", tmp_path / "a1")], "1:00:00", "1G"),
        Stage("eval_b", [Task("b/v1", "echo", tmp_path / "b1")], "1:00:00", "1G"),
    ]
    orch = Orchestrator("non_train_test", stages, cluster)
    dependencies = []
    ids = iter([401, 402])
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: dependencies.append(dependency) or next(ids))
    orch.submit()
    assert dependencies == [None, "afterok:401"]


def test_no_wait_dry_run_renders_every_stage(cluster, tmp_path):
    stages = [
        Stage("one", [Task("one", "echo one", tmp_path / "one")], "1:00:00", "1G"),
        Stage("two", [Task("two", "echo two", tmp_path / "two")], "1:00:00", "1G"),
    ]
    orch = Orchestrator("dependency_dry_run", stages, cluster, dry_run=True)
    orch.submit()
    assert (cluster.repo / "slurm_jobs/dependency_dry_run/one.sbatch").is_file()
    assert (cluster.repo / "slurm_jobs/dependency_dry_run/two.sbatch").is_file()


def test_finished_task_is_not_resubmitted_when_its_output_lags(cluster, tmp_path, monkeypatch):
    # squeue drops a job before NFS necessarily shows what it wrote. A done-check
    # fired the instant the job ends must not resubmit the array for nothing.
    visible_after = {"calls": 0}

    def done() -> bool:
        visible_after["calls"] += 1
        return visible_after["calls"] > 3  # the write "appears" on the 4th look

    stage = Stage("lagging", [Task("t", "echo t", done)], "1:00:00", "1G")
    submitted = []
    orch = Orchestrator("settle", [stage], cluster)
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: submitted.append(script) or 1)
    monkeypatch.setattr(orch, "_job_active", staticmethod(lambda job_id: False))
    monkeypatch.setattr("valuegen.slurm.time.sleep", lambda _: None)

    assert orch.run()
    assert len(submitted) == 1, "a lagging output triggered a redundant resubmission"


def test_a_genuinely_failed_task_is_still_retried(cluster, tmp_path, monkeypatch):
    stage = Stage("failing", [Task("t", "false", lambda: False)], "1:00:00", "1G", max_retries=2)
    submitted = []
    orch = Orchestrator("retry", [stage], cluster)
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: submitted.append(script) or 1)
    monkeypatch.setattr(orch, "_job_active", staticmethod(lambda job_id: False))
    monkeypatch.setattr("valuegen.slurm.time.sleep", lambda _: None)

    assert not orch.run()
    assert len(submitted) == 2  # both attempts, then give up


def _server_revival_orch(cluster, monkeypatch, alive_after_first_check: bool):
    """A judge server + a dependent eval stage whose task never completes.

    ``_job_active`` is asked about two kinds of job: the server (id 900) and
    the finished eval arrays (anything else, always inactive so the retry loop
    advances).
    """
    server = Stage(
        "serve_judge", [Task("serve", "vllm serve", lambda: False)],
        "1:00:00", "1G", server=True, hostfile=cluster.repo / "judge_host.txt",
    )
    evals = Stage(
        "eval_m", [Task("t", "eval", lambda: False)], "1:00:00", "1G",
        max_retries=3, needs_servers=("serve_judge",),
    )
    orch = Orchestrator("revive", [server, evals], cluster)

    submitted: list[str] = []
    ids = iter(range(900, 999))

    def fake_sbatch(script, dependency=None):
        submitted.append(Path(script).stem)
        return next(ids)

    checks = {"n": 0}

    def job_active(job_id):
        if job_id != 900:
            return False  # eval arrays are always "finished"
        checks["n"] += 1
        return alive_after_first_check or checks["n"] <= 1

    monkeypatch.setattr(orch, "_sbatch", fake_sbatch)
    monkeypatch.setattr(orch, "_job_active", job_active)
    monkeypatch.setattr("valuegen.slurm.time.sleep", lambda _: None)
    return orch, submitted


def test_a_dead_server_is_resubmitted_before_the_next_retry(cluster, monkeypatch):
    # A past failure: the judge was SIGKILLed mid-array, so
    # attempts 2 and 3 polled a stale hostfile until timeout and ran no work.
    orch, submitted = _server_revival_orch(cluster, monkeypatch, alive_after_first_check=False)
    hostfile = cluster.repo / "judge_host.txt"

    assert not orch.run()  # the eval task never completes; that is not what we assert on
    # attempt 1 finds the server alive; attempts 2 and 3 each revive it first.
    assert submitted.count("serve_judge") == 3
    assert submitted.count("eval_m") == 3
    assert not hostfile.exists(), "a revived server must clear the stale hostfile"


def test_a_live_server_is_not_resubmitted_between_retries(cluster, monkeypatch):
    orch, submitted = _server_revival_orch(cluster, monkeypatch, alive_after_first_check=True)

    assert not orch.run()
    assert submitted.count("serve_judge") == 1, "a healthy server was needlessly restarted"
    assert submitted.count("eval_m") == 3


# ── exclusive-node clusters (AMD AUP): no GRES/mem, packing, caps, containers ──


def _exclusive(cluster, **over):
    """The test cluster reshaped like an exclusive-node cluster: whole 8-GPU nodes,
    no GRES, no --mem, 12h caps, 16-GPU (2-node) concurrency."""
    fields = dict(
        gpu_type=None, request_gres=False, request_mem=False,
        gpus_per_node=8, max_concurrent_gpus=16,
        gpu_partition="mi3508x", gpu_qos="alloc_q", cpu_partition="mi2101x",
        cpu_qos="alloc_q", cpu_stage_gpus=1, account="myaccount",
        partition_max_time={"mi3508x": "12:00:00", "mi2101x": "12:00:00"},
        exclude=[],
    )
    fields.update(over)
    return replace(cluster, **fields)


def _tasks(tmp_path, n, prefix="t"):
    return [Task(f"{prefix}{i}", f"echo {i}", tmp_path / f"{prefix}{i}.done") for i in range(n)]


def test_exclusive_cluster_omits_gres_and_mem(cluster, tmp_path):
    ex = _exclusive(cluster)
    gpu = Stage("train", _tasks(tmp_path, 1), "05:00:00", "48G", gpus=2)
    cpu = Stage("merge", _tasks(tmp_path, 1), "01:00:00", "8G")
    assert Orchestrator("ex", [gpu, cpu], ex, dry_run=True).run()
    gpu_script = (ex.repo / "slurm_jobs/ex/train.sbatch").read_text()
    cpu_script = (ex.repo / "slurm_jobs/ex/merge.sbatch").read_text()
    for script in (gpu_script, cpu_script):
        assert "--gres" not in script
        assert "--mem" not in script
        assert "#SBATCH --account=myaccount" in script
    assert "#SBATCH --partition=mi3508x\n#SBATCH --qos=alloc_q" in gpu_script
    # CPU stages still go to the cheap single-GPU partition; the mandatory
    # cpu_stage_gpus only ever mattered for --gres and the throttle.
    assert "#SBATCH --partition=mi2101x\n#SBATCH --qos=alloc_q" in cpu_script


def test_single_gpu_arrays_pack_gpus_per_node_tasks_per_element(cluster, tmp_path):
    ex = _exclusive(cluster)
    tasks = _tasks(tmp_path, 10)
    # Headroom of a nominal tp=2 judge: on an exclusive cluster it holds a
    # whole node, so 16 - 8 leaves one node's worth for the array.
    stage = Stage("eval_m", tasks, "04:00:00", "32G", gpus=1, gpu_headroom=2)
    orch = Orchestrator("pack", [stage], ex, dry_run=True)
    assert orch.run()
    script = (ex.repo / "slurm_jobs/pack/eval_m.sbatch").read_text()
    assert "#SBATCH --array=0-1%1" in script  # 10 tasks -> 2 nodes, 1 at a time
    assert "parallel --jobs 8" in script
    assert "CUDA_VISIBLE_DEVICES=$(({#} - 1)) HIP_VISIBLE_DEVICES=$(({#} - 1)) bash {}" in script
    assert "--gres" not in script and "--mem" not in script
    # Chunk 0 runs t0..t7, chunk 1 runs t8, t9 -- each via its own script.
    task_map = json.loads((ex.repo / "slurm_jobs/pack/eval_m_task_map.json").read_text())
    assert task_map == {"0": [f"t{i}" for i in range(8)], "1": ["t8", "t9"]}
    t3 = (ex.repo / "slurm_jobs/pack/eval_m/t3.sh").read_text()
    assert t3 == "set -euo pipefail\necho 3\n"
    assert "1)  # t8, t9" in script


def test_packing_is_off_without_gpus_per_node_and_for_multi_gpu_stages(cluster, tmp_path):
    ex = _exclusive(cluster, gpus_per_node=None)
    stage = Stage("eval_m", _tasks(tmp_path, 3), "04:00:00", "32G", gpus=1)
    assert Orchestrator("nopack", [stage], ex, dry_run=True).run()
    script = (ex.repo / "slurm_jobs/nopack/eval_m.sbatch").read_text()
    assert "parallel" not in script and "#SBATCH --array=0-2%16" in script
    assert not (ex.repo / "slurm_jobs/nopack/eval_m").exists()
    # A 2-GPU stage on an exclusive cluster: one element per task, each
    # holding a node, so the 16-GPU cap means two at a time.
    two = Stage("train_m", _tasks(tmp_path, 3, "u"), "05:00:00", "48G", gpus=2)
    assert Orchestrator("two", [two], _exclusive(cluster), dry_run=True).run()
    script = (cluster.repo / "slurm_jobs/two/train_m.sbatch").read_text()
    assert "parallel" not in script and "#SBATCH --array=0-2%2" in script
    # pack=1 opts a single-GPU stage out even where the cluster packs.
    one = Stage("eval_x", _tasks(tmp_path, 3, "v"), "04:00:00", "32G", gpus=1, pack=1)
    assert Orchestrator("one", [one], _exclusive(cluster), dry_run=True).run()
    assert "parallel" not in (cluster.repo / "slurm_jobs/one/eval_x.sbatch").read_text()


def test_packed_mem_scales_with_chunk_where_mem_is_requested(cluster, tmp_path):
    # Exclusive-node clusters: gres + mem still requested; a chunk
    # asks for pack x per-task mem.
    ex = _exclusive(cluster, request_gres=True, request_mem=True, gpu_type="A6000", gpus_per_node=4)
    stage = Stage("eval_m", _tasks(tmp_path, 5), "04:00:00", "32G", gpus=1)
    assert Orchestrator("mem", [stage], ex, dry_run=True).run()
    script = (ex.repo / "slurm_jobs/mem/eval_m.sbatch").read_text()
    assert "#SBATCH --mem=128G" in script
    assert "#SBATCH --gres=gpu:A6000:4" in script
    assert "parallel --jobs 4" in script


def test_walltime_is_clamped_to_the_partition_cap_and_retries_scale(cluster, tmp_path, monkeypatch):
    ex = _exclusive(cluster)
    long = Stage("label", _tasks(tmp_path, 2), "2-00:00:00", "24G")  # cpu: 12h cap
    fine = Stage("train", _tasks(tmp_path, 1, "u"), "05:00:00", "48G", gpus=1)
    uncapped = Stage("gen", _tasks(tmp_path, 1, "w"), "2-00:00:00", "8G", server=True, gpus=0)
    orch = Orchestrator("cap", [long, fine, uncapped], ex, dry_run=True)
    assert orch._walltime(long) == ("12:00:00", 4)
    assert orch._walltime(fine) == ("05:00:00", 1)
    assert orch._walltime(uncapped) == ("2-00:00:00", 1)  # cluster-default partition: no cap known
    assert orch.run()
    assert "#SBATCH --time=12:00:00" in (ex.repo / "slurm_jobs/cap/label.sbatch").read_text()
    # A clamped stage gets max_retries x ceil(requested / cap) attempts.
    attempts = []
    orch = Orchestrator("cap2", [long], ex)
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: attempts.append(1) or 1)
    monkeypatch.setattr(orch, "_wait", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_settle", lambda stage: None)
    monkeypatch.setattr(orch, "_find_active_job", lambda stage: None)
    assert orch.run_stage(long) == (False, None)
    assert len(attempts) == 3 * 4


def test_find_active_job_adopts_the_array_not_one_running_element(cluster, tmp_path, monkeypatch):
    """SLURM gives a *running* array element its own job id (``%A``); only
    ``%F`` names the array. Adopting an element would end the wait as soon as
    that one task finished."""
    import valuegen.slurm as S

    ex = _exclusive(cluster)
    stage = Stage("train_w0", _tasks(tmp_path, 8), "01:30:00", "0", gpus=1)
    orch = Orchestrator("mv", [stage], ex)
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = list(argv)
        fmt = argv[argv.index("-o") + 1]
        # what squeue prints for an array with 2 elements running and 6 queued
        rows = ([("421479", "PENDING"), ("421501", "RUNNING"), ("421503", "RUNNING")] if fmt.startswith("%A")
                else [("421479", "PENDING"), ("421479", "RUNNING"), ("421479", "RUNNING")])
        out = "\n".join(f"{a} {b}" for a, b in rows) + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(S.subprocess, "run", fake_run)
    assert orch._find_active_job(stage) == 421479
    assert "%F %T" in seen["argv"]
    assert "-n" in seen["argv"] and "mv_train_w0" in seen["argv"]


def test_slurm_time_seconds_parses_every_sbatch_form():
    from valuegen.slurm import slurm_time_seconds

    assert slurm_time_seconds("30") == 30 * 60
    assert slurm_time_seconds("30:00") == 30 * 60
    assert slurm_time_seconds("04:00:00") == 4 * 3600
    assert slurm_time_seconds("1-00:00:00") == 86400
    assert slurm_time_seconds("2-12") == 60 * 3600
    assert slurm_time_seconds("2-12:30") == 60 * 3600 + 30 * 60


def test_a_live_job_with_the_stage_name_is_adopted_not_duplicated(cluster, tmp_path, monkeypatch):
    hostfile = tmp_path / "judge_host.txt"
    hostfile.write_text("node1:8600\n")
    server = serve_stage("serve_judge", "m", hostfile, gpus=2)
    stage = Stage("eval_m", _tasks(tmp_path, 2), "04:00:00", "32G", gpus=1, needs_servers=("serve_judge",))
    orch = Orchestrator("adopt", [server, stage], cluster)
    submitted = []
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: submitted.append(script.name) or 99)
    monkeypatch.setattr(orch, "_find_active_job", lambda s: {"serve_judge": 41, "eval_m": 42}.get(s.name))
    waited = []
    monkeypatch.setattr(orch, "_wait", lambda job_id, label, servers=(): waited.append((job_id, tuple(servers))))
    monkeypatch.setattr(orch, "_settle", lambda s: None)
    monkeypatch.setattr(orch, "_job_active", lambda job_id: True)
    assert orch.run_stage(server) == (True, 41)
    assert hostfile.exists()  # the live server's hostfile is kept
    assert orch._server_jobs == {"serve_judge": 41}
    # The array is adopted too: waited on (with its servers re-checked mid-wait),
    # never resubmitted, and each adoption consumes one attempt.
    assert orch.run_stage(stage) == (False, None)
    assert submitted == []
    assert waited == [(42, ("serve_judge",))] * 3


def test_a_server_dead_on_arrival_is_not_revived_forever(cluster, tmp_path, monkeypatch):
    hostfile = tmp_path / "judge_host.txt"
    server = serve_stage("serve_judge", "m", hostfile, gpus=2)
    server.max_retries = 2
    stage = Stage("eval_m", _tasks(tmp_path, 1), "04:00:00", "32G", gpus=1, needs_servers=("serve_judge",))
    orch = Orchestrator("doa", [server, stage], cluster)
    submitted = []
    monkeypatch.setattr(orch, "_sbatch", lambda script, dependency=None: submitted.append(script.name) or len(submitted))
    monkeypatch.setattr(orch, "_find_active_job", lambda s: None)
    monkeypatch.setattr(orch, "_job_active", lambda job_id: False)  # every server job is gone at once
    monkeypatch.setattr(orch, "_settle", lambda s: None)
    # _wait polls twice per attempt, reviving servers each poll.
    def fake_wait(job_id, label, servers=()):
        for _ in range(2):
            orch._ensure_servers(servers)
    monkeypatch.setattr(orch, "_wait", fake_wait)
    assert orch.run_stage(stage) == (False, None)
    # 3 attempts x (1 pre-attempt + 2 in-wait) = 9 revival opportunities, but
    # only max_retries=2 server submissions happen; the array itself is
    # submitted once per attempt.
    assert submitted.count("serve_judge.sbatch") == 2
    assert submitted.count("eval_m.sbatch") == 3


def test_conflictscope_judge_honors_prefix_caching_and_serve_args(cluster, tmp_path):
    cfg = {"name": "judge_policy", "intervention": {}, "evaluation": {"judge": {
        "model": "Qwen/Qwen3.6-27B", "serve": "local", "gpus": 2,
        "enable_prefix_caching": False, "serve_args": ["--gdn-prefill-backend triton"],
    }}}
    command = conflictscope_eval.judge_stage(cfg, cluster).tasks[0].command
    assert "--no-enable-prefix-caching" in command
    assert "--gdn-prefill-backend triton" in command
    default = serve_stage("serve_judge", "m", tmp_path / "h.txt").tasks[0].command
    assert "--enable-prefix-caching" in default and "--no-enable-prefix-caching" not in default
    
    
def test_gres_lines_stage_override_narrows_to_one_type(cluster, tmp_path):
    cluster = replace(cluster, gpu_type=["L40S", "A6000"])
    assert cluster.gres_lines(4) == [
        "#SBATCH --gres=gpu:4", "#SBATCH --constraint=L40S|A6000",
    ]
    assert cluster.gres_lines(4, gpu_type="L40S") == ["#SBATCH --gres=gpu:L40S:4"]
    with pytest.raises(ValueError, match="not among"):
        cluster.gres_lines(4, gpu_type="H100")
    # and the rendered sbatch of a pinned stage carries the typed gres
    gpu = Stage("gpu", [Task("a", "echo", tmp_path / "a")], "1:00:00", "8G",
                gpus=4, gpu_type="L40S")
    orch = Orchestrator("gt_pinned", [gpu], cluster, dry_run=True)
    assert orch.run()
    script = (cluster.repo / "slurm_jobs/gt_pinned/gpu.sbatch").read_text()
    assert "#SBATCH --gres=gpu:L40S:4\n" in script
    assert "--constraint" not in script


def test_wait_ready_watches_a_cohosted_server_pid():
    from valuegen.ground_truth.evaluation import wait_ready_sh

    assert "kill -0" not in wait_ready_sh("/tmp/hf", "judge")
    sh = wait_ready_sh("/tmp/hf", "generator", pid_var="VLLM_PID")
    assert 'kill -0 "${VLLM_PID}"' in sh and "died before becoming ready" in sh
