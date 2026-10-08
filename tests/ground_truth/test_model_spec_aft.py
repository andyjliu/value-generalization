"""Contract tests for the ``model_spec_aft`` intervention method.

Everything here is offline: config identity, spec rendering, the fork-chat →
pair-format dataset conversion, and the declared manifest grid. The fork
subprocess itself is exercised by the smoke experiment, not unit tests.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from valuegen.config import intervention_id, resolve_experiment
from valuegen.ground_truth import interventions, model_spec_aft


def _cfg(**generation) -> dict:
    return {
        "name": "aft_test",
        "intervention": {
            "method": "model_spec_aft",
            "value_set": "constitution_tenets_v3",
            "values": ["calibrated_uncertainty", "hard_constraint_fidelity"],
            "models": {"olmo7b_base": "allenai/OLMo-2-1124-7B"},
            "generation": dict(generation),
        },
        "evaluation": {
            "method": "conflictscope",
            "scenarios": "data/scenarios/const_v3_cs",
            "judge": {"model": "Qwen/Qwen3.6-27B"},
        },
    }


def test_resolution_defaults_algo_sft_and_materializes_generation():
    cfg = resolve_experiment(_cfg(n_samples=100))
    iv = cfg["intervention"]
    assert iv["algo"] == "sft"
    assert iv["schema_version"] == 1
    gen = iv["generation"]
    assert gen["n_samples"] == 100          # explicit value kept
    assert gen["strip_cot"] is True         # default materialized
    assert gen["disable_thinking"] is False
    assert gen["api_base"] is None


def test_resolution_rejects_non_sft_algo():
    raw = _cfg()
    raw["intervention"]["algo"] = "dpo"
    with pytest.raises(ValueError, match="algo must be 'sft'"):
        resolve_experiment(raw)


def test_eval_knobs_never_move_the_intervention_id():
    a = resolve_experiment(_cfg())
    b = resolve_experiment(_cfg())
    b["evaluation"]["judge"]["model"] = "some/other-judge"
    assert intervention_id(a) == intervention_id(b)

    c = resolve_experiment(_cfg(n_samples=999))
    assert intervention_id(a) != intervention_id(c)


def test_build_specs_renders_full_description_once(cluster):
    cfg = resolve_experiment(_cfg())
    paths = model_spec_aft.build_specs(cfg, cluster)
    assert set(paths) == {"calibrated_uncertainty", "hard_constraint_fidelity"}
    text = paths["calibrated_uncertainty"].read_text()
    assert "acknowledging uncertainty and knowledge limits" in text
    assert "{" not in text  # no unfilled placeholders reach the fork
    # Idempotent: a second call returns the same files without rewriting.
    again = model_spec_aft.build_specs(cfg, cluster)
    assert again == paths


def test_convert_dataset_maps_chat_to_pair_format(tmp_path):
    src = tmp_path / "dataset.jsonl"
    src.write_text(
        json.dumps({"messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]}) + "\n"
    )
    dst = tmp_path / "out" / "dataset.jsonl"
    assert model_spec_aft._convert_dataset(src, dst) == 1
    row = json.loads(dst.read_text())
    assert row["prompt"] == [{"role": "user", "content": "q1"}]
    assert row["chosen"] == [{"role": "assistant", "content": "a1"}]


def test_convert_dataset_rejects_unexpected_roles(tmp_path):
    src = tmp_path / "dataset.jsonl"
    src.write_text(
        json.dumps({"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
        ]}) + "\n"
    )
    with pytest.raises(ValueError, match="user, assistant"):
        model_spec_aft._convert_dataset(src, tmp_path / "out.jsonl")


def test_generation_command_and_served_api_base(tmp_path, monkeypatch):
    monkeypatch.delenv("VALUEGEN_MSM_API_BASE", raising=False)
    cfg = resolve_experiment(_cfg(n_samples=100))
    spec = tmp_path / "calibrated_uncertainty.txt"
    cmd = model_spec_aft._generation_command(cfg, "calibrated_uncertainty", spec)
    assert "src.aft.generate_chat" in cmd
    assert f"--spec_name {spec}" in cmd
    assert "--n_samples 100" in cmd
    assert "--skip_existing true" in cmd
    assert "--vllm_num_threads 20" in cmd  # == generation.max_concurrent default
    assert "--disable_thinking false" in cmd
    assert "--vllm_base_url" not in cmd

    cfg["intervention"]["generation"]["api_base"] = "http://stable:8000/v1"
    cmd = model_spec_aft._generation_command(cfg, "calibrated_uncertainty", spec)
    assert "--use_vllm_if_model_not_found true" in cmd
    assert "--vllm_base_url http://stable:8000/v1" in cmd

    # Ephemeral endpoints ride in via env, without touching the config hash.
    cfg["intervention"]["generation"]["api_base"] = None
    monkeypatch.setenv("VALUEGEN_MSM_API_BASE", "http://node:8000/v1")
    cmd = model_spec_aft._generation_command(cfg, "calibrated_uncertainty", spec)
    assert "--vllm_base_url http://node:8000/v1" in cmd

    cfg["intervention"]["generation"]["disable_thinking"] = True
    cmd = model_spec_aft._generation_command(cfg, "calibrated_uncertainty", spec)
    assert "--disable_thinking true" in cmd


def test_manifest_declares_the_checkpoint_grid(cluster):
    cfg = resolve_experiment(_cfg())
    manifest = interventions.build_manifest(cfg, cluster)
    entries = manifest["interventions"]
    assert [e["tag"] for e in entries] == ["olmo7b_base_calibrated_uncertainty", "olmo7b_base_hard_constraint_fidelity"]
    assert all(e["kind"] == "checkpoint" for e in entries)
    assert all("_merged" in e["path"] for e in entries)


def test_stages_are_sft_train_arrays(cluster):
    # No use_peft in the train block means full FT (TRL's ModelConfig
    # default), whose finishing step is export; a LoRA block gets the merge.
    cfg = resolve_experiment(_cfg())
    stages = model_spec_aft.stages(cfg, cluster)
    assert [s.name for s in stages] == ["train_olmo7b_base"]
    command = stages[0].tasks[0].command
    assert "valuegen.ground_truth.training sft" in command
    assert "training export" in command

    lora = _cfg()
    lora["intervention"]["train"] = {"use_peft": True, "lora_r": 32}
    cfg = resolve_experiment(lora)
    command = model_spec_aft.stages(cfg, cluster)[0].tasks[0].command
    assert "training merge" in command


def test_serve_local_emits_server_and_generate_stages(cluster):
    raw = _cfg(model_id="Qwen/Qwen3.6-27B", serve="local")
    cfg = resolve_experiment(raw)
    cfg["_path"] = "/tmp/aft_test.yaml"
    stages = model_spec_aft.stages(cfg, cluster)
    assert [s.name for s in stages] == ["serve_gen", "generate", "train_olmo7b_base"]

    serve, generate = stages[0], stages[1]
    assert "vllm serve" in serve.tasks[0].command
    assert generate.cancel_servers == ("serve_gen",)
    assert [t.key for t in generate.tasks] == ["generate_calibrated_uncertainty", "generate_hard_constraint_fidelity"]
    command = generate.tasks[0].command
    assert "wait_ready" in command
    assert "model_spec_aft generate -c /tmp/aft_test.yaml --value calibrated_uncertainty" in command

    # build_data must not try to generate inline in served mode (no server
    # is up); it only writes the specs and defers to the stages.
    model_spec_aft.build_data(cfg, cluster)
    assert not (
        interventions.datasets_dir(cfg, cluster) / "calibrated_uncertainty" / "dataset.jsonl"
    ).exists()

    # Once the datasets exist, the server/generate stages disappear.
    for value in ("calibrated_uncertainty", "hard_constraint_fidelity"):
        dst = interventions.datasets_dir(cfg, cluster) / value / "dataset.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text('{"prompt": [], "chosen": []}\n')
    assert [s.name for s in model_spec_aft.stages(cfg, cluster)] == [
        "train_olmo7b_base"
    ]


def test_generation_command_forwards_backfill_block_only_when_present(tmp_path):
    spec = tmp_path / "calibrated_uncertainty.txt"
    plain = resolve_experiment(_cfg(n_samples=100))
    assert "backfill" not in model_spec_aft._generation_command(plain, "calibrated_uncertainty", spec)

    raw = _cfg(n_samples=100, backfill={
        "max_rounds": 3, "pass_rate_scaled": True, "until_full": True, "dedup": True,
    })
    cfg = resolve_experiment(raw)
    # The block survives resolution and re-keys the artifact.
    assert cfg["intervention"]["generation"]["backfill"]["until_full"] is True
    assert intervention_id(cfg) != intervention_id(plain)
    cmd = model_spec_aft._generation_command(cfg, "calibrated_uncertainty", spec)
    assert "--backfill_max_rounds 3" in cmd
    assert "--backfill_pass_rate_scaled true" in cmd
    assert "--backfill_until_full true" in cmd
    assert "--backfill_dedup true" in cmd


def test_dataset_complete_requires_rows_only_under_until_full(tmp_path):
    path = tmp_path / "dataset.jsonl"
    plain = resolve_experiment(_cfg(n_samples=3))
    full = resolve_experiment(_cfg(n_samples=3, backfill={
        "max_rounds": 3, "pass_rate_scaled": True, "until_full": True, "dedup": True,
    }))
    assert not model_spec_aft.dataset_complete(plain, path)
    path.write_text("")
    assert not model_spec_aft.dataset_complete(plain, path)
    path.write_text('{"messages": []}\n' * 2)
    # historical loop: an existing file is complete; until_full: it is not
    assert model_spec_aft.dataset_complete(plain, path)
    assert not model_spec_aft.dataset_complete(full, path)
    path.write_text('{"messages": []}\n' * 3)
    assert model_spec_aft.dataset_complete(full, path)


def test_short_until_full_dataset_keeps_its_generate_task_pending(cluster):
    raw = _cfg(model_id="Qwen/Qwen3.6-27B", serve="local", n_samples=3, backfill={
        "max_rounds": 3, "pass_rate_scaled": True, "until_full": True, "dedup": True,
    })
    cfg = resolve_experiment(raw)
    cfg["_path"] = "/tmp/aft_test.yaml"
    data_root = interventions.datasets_dir(cfg, cluster)
    for value in ("calibrated_uncertainty", "hard_constraint_fidelity"):
        (data_root / value).mkdir(parents=True, exist_ok=True)
    (data_root / "calibrated_uncertainty" / "dataset.jsonl").write_text('{"m": 1}\n' * 3)
    (data_root / "hard_constraint_fidelity" / "dataset.jsonl").write_text('{"m": 1}\n' * 2)  # short
    generate = model_spec_aft.stages(cfg, cluster)[1]
    assert [t.key for t in generate.tasks] == ["generate_hard_constraint_fidelity"]
    (data_root / "hard_constraint_fidelity" / "dataset.jsonl").write_text('{"m": 1}\n' * 3)
    assert generate.tasks[0].is_done()


def _served_cfg(**gen):
    cfg = resolve_experiment(_cfg(model_id="Qwen/Qwen3.6-27B", serve="local", **gen))
    cfg["_path"] = "/tmp/aft_test.yaml"
    return cfg


def test_gpu_only_cluster_self_hosts_generation_in_one_job(cluster):
    # With cpu_stage_gpus > 0 a CPU client array would cost a GPU per HTTP
    # client, so generation is one job that co-hosts the server.
    gpu_only = replace(cluster, cpu_stage_gpus=1)
    cfg = _served_cfg()
    stages = model_spec_aft.stages(cfg, gpu_only)
    assert [s.name for s in stages] == ["generate", "train_olmo7b_base"]
    generate = stages[0]
    assert generate.gpus == int(cfg["intervention"]["generation"]["gpus"])
    assert not generate.server and generate.cancel_servers == ()
    assert [t.key for t in generate.tasks] == ["generate"]
    command = generate.tasks[0].command
    assert "vllm serve" in command and "VLLM_PID=$!" in command
    assert 'kill -0 "${VLLM_PID}"' in command  # dead server fails fast
    assert "printf \"%s\\n\" calibrated_uncertainty hard_constraint_fidelity" in command
    assert "model_spec_aft generate -c /tmp/aft_test.yaml --value _V_" in command
    assert "--max-num-seqs 128" in command
    assert "--max-num-batched-tokens 8192" in command


def test_self_hosted_generation_shards_values_across_servers(cluster):
    gpu_only = replace(cluster, cpu_stage_gpus=1)
    generate = model_spec_aft.stages(_served_cfg(servers=2), gpu_only)[0]
    assert [t.key for t in generate.tasks] == ["generate_0", "generate_1"]
    assert "hostfile_gen_0" in generate.tasks[0].command
    assert "printf \"%s\\n\" calibrated_uncertainty " in generate.tasks[0].command
    assert "printf \"%s\\n\" hard_constraint_fidelity " in generate.tasks[1].command


def test_generation_gpu_type_pins_the_stage_but_not_the_hash(cluster):
    plain, pinned = _served_cfg(), _served_cfg(gpu_type="L40S")
    assert intervention_id(pinned) == intervention_id(plain)
    # served path: the server stage carries the pin
    assert model_spec_aft.stages(plain, cluster)[0].gpu_type is None
    assert model_spec_aft.stages(pinned, cluster)[0].gpu_type == "L40S"
    # self-hosted path: the generate job carries it
    gpu_only = replace(cluster, cpu_stage_gpus=1)
    generate = model_spec_aft.stages(pinned, gpu_only)[0]
    assert generate.name == "generate" and generate.gpu_type == "L40S"
