"""Split intervention/evaluation GT construction and replication tests."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from valuegen import values as V
from valuegen.config import (
    ClusterConfig,
    gt_id,
    intervention_id,
    load_experiment,
    pool_id,
    resolve_experiment,
)
from valuegen.ground_truth import (
    common,
    conflictscope,
    exclusive_assignment,
    interventions,
    label_subset,
    scenario_pool,
)
from valuegen.ground_truth.evals import conflictscope as conflictscope_eval
from valuegen.slurm import Orchestrator


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cluster(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "value_sets").symlink_to(REPO / "value_sets", target_is_directory=True)
    return ClusterConfig(
        envs={"default": ".venvs/test", "eval_api": ".venvs/test", "persona": ".venvs/persona", "weight_steering": ".venvs/test", "ws_train": ".venvs/ws-train"},
        repo=repo, finetune_root=tmp_path / "finetune", finetune_root_legacy=tmp_path / "legacy",
        data=tmp_path / "data", slurm_logs=tmp_path / "logs", mail_type="END", mail_user="test@example.com",
        gpu_type="A6000", max_concurrent_gpus=8, default_time="1:00:00", default_mem="4G",
        cpu_partition="cpu", cpu_qos="cpu_qos", exclude=["slow-a", "slow-b"], source_path=tmp_path / "cluster.yaml",
    )


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


def _cfg(name):
    return load_experiment(FIXTURES / name)


def test_label_subset_training_and_eval_are_separate_registries(cluster):
    cfg = _cfg("label_subset_grid.yaml")
    manifest = interventions.build_manifest(cfg, cluster)
    training = interventions.stages(cfg, cluster)
    evaluation = conflictscope_eval.stages(cfg, cluster, manifest)

    assert len(training) == 5
    assert all(s.name.startswith("train_") and len(s.tasks) == 13 for s in training)
    assert [s.name for s in evaluation[:1]] == ["serve_judge"]
    evals = evaluation[1:]
    assert len(evals) == 5
    assert all(len(s.tasks) == 14 for s in evals)
    assert all(s.tasks[0].key.endswith("_base") for s in evals)
    # Every eval stage must name the servers its tasks block on, or the
    # orchestrator cannot revive one that dies mid-array (2026-07-27).
    assert all(s.needs_servers == ("serve_judge",) for s in evals)
    assert all(t.config_id == intervention_id(cfg) for s in training for t in s.tasks)
    assert all(t.config_id is None for s in evaluation for t in s.tasks)
    for stage in training + evals:
        script = Orchestrator(cfg["name"], [stage], cluster, dry_run=True).render(stage, stage.tasks)
        assert "#SBATCH --exclude=slow-a,slow-b" in script


def test_conflictscope_completions_do_not_wait_for_final_judge(cluster):
    cfg = _cfg("conflictscope_pairs_smoke.yaml")
    pool_stages = scenario_pool.pending_stages(cfg, cluster)
    assert len(pool_stages) == 3
    assert [stage.name.rsplit("_", 1)[-1] for stage in pool_stages] == [
        "scenarios", "dedup", "split",
    ]
    assert " -d" not in pool_stages[0].tasks[0].command
    assert "scenario_pool dedup" in pool_stages[1].tasks[0].command
    assert all(
        task.config_id == pool_id(cfg["intervention"]["pool"])
        for stage in pool_stages for task in stage.tasks
    )
    stages = conflictscope.stages(cfg, cluster)
    completion_stage = next(s for s in stages if s.name == "completions")
    assert len(completion_stage.tasks) == 2
    for task in completion_stage.tasks:
        assert "--user-model gpt-4.1" in task.command
        assert "--judge-model gpt-4.1" in task.command
        assert "judge_host.txt" not in task.command
        assert "wait_ready" not in task.command


def test_conflictscope_completions_samples_fan_out(cluster):
    cfg = _cfg("conflictscope_pairs_smoke.yaml")
    comp = cfg["intervention"]["completions"]
    comp["samples"] = 3
    comp["temperature"] = 0.8
    stages = conflictscope.stages(cfg, cluster)
    stage = next(s for s in stages if s.name == "completions")
    stems = [f"{m}-{k}" for m in ("gpt-4o-mini", "gpt-4.1") for k in range(3)]
    assert [t.key for t in stage.tasks] == [f"completions_{s}" for s in stems]
    for task, stem in zip(stage.tasks, stems):
        assert f"--output-name {stem}.csv" in task.command
        assert "--temperature 0.8" in task.command
        assert task.done.name == f"{stem}.csv"


def test_conflictscope_samples_require_positive_temperature(cluster):
    cfg = _cfg("conflictscope_pairs_smoke.yaml")
    cfg["intervention"]["completions"]["samples"] = 3
    with pytest.raises(ValueError, match="temperature"):
        conflictscope.stages(cfg, cluster)
    cfg["intervention"]["completions"]["temperature"] = 0.0
    with pytest.raises(ValueError, match="temperature"):
        conflictscope.stages(cfg, cluster)


def test_reference_model_stems_expand_with_samples():
    comp = {"reference_models": ["m1", "m2"], "samples": 2, "temperature": 1.0}
    assert conflictscope.reference_model_stems(comp) == ["m1-0", "m1-1", "m2-0", "m2-1"]
    assert conflictscope.reference_model_stems({"reference_models": ["m1"]}) == ["m1"]
    assert conflictscope.reference_model_stems({}) is None


@pytest.mark.parametrize("max_pairs, expected_count", [(None, 4), (2, 2)])
def test_label_subset_build_data_with_mocked_sources(
    cluster, monkeypatch, max_pairs, expected_count
):
    cfg = resolve_experiment({
        "name": "labels_test",
        "intervention": {
            "method": "label_subset", "value_set": "constitution_tenets_v3",
            "values": ["calibrated_uncertainty"], "models": {"olmo7b_sft": "allenai/OLMo-2-1124-7B-SFT"},
            "max_pairs": max_pairs,
            "labels": {"dir": "gt_labels", "tau": 0.5, "seed": 42},
        },
        "evaluation": {
            "method": "conflictscope", "scenarios": "data/scenarios/const_v3_cs",
            "judge": {"model": "gpt-4.1"},
        },
    })
    labels = cluster.repo / "gt_labels"; labels.mkdir(exist_ok=True)
    try:
        scores = pd.DataFrame({"row_id": [0, 1], "principle_id": [4, 4], "prob_A": [0.9, 0.1], "prob_B": [0.1, 0.9]})
        scores.to_csv(labels / label_subset.HH_LABELS, index=False)
        scores.to_csv(labels / label_subset.PKU_LABELS, index=False)
        text = pd.DataFrame({"prompt": ["q0", "q1"], "A": ["a0", "a1"], "B": ["b0", "b1"]})
        monkeypatch.setitem(label_subset.SOURCE_LOADERS, "hh", lambda: text)
        monkeypatch.setitem(label_subset.SOURCE_LOADERS, "pku", lambda: text)
        label_subset.build_data(cfg, cluster)
        result = pd.read_json(
            interventions.datasets_dir(cfg, cluster) / "calibrated_uncertainty/dataset.jsonl",
            lines=True,
        )
        assert len(result) == expected_count
        # Real message lists (not stringified reprs): the conversational
        # DPOTrainer path depends on this shape surviving the round trip.
        assert all(x[0]["role"] == "user" for x in result.prompt)
        assert all(x[0]["role"] == "assistant" for x in result.chosen)
    finally:
        for path in labels.iterdir(): path.unlink()
        labels.rmdir()


def test_resolve_models_rejects_family_mismatch():
    # Training picks the template from the HF name, eval from the mtag-derived
    # merged-dir name; a disagreement would silently split the two formats.
    with pytest.raises(ValueError, match="different formats"):
        interventions.resolve_models({"qwen_custom": "allenai/OLMo-2-1124-7B-SFT"})
    # Agreement (and keyword-less local paths, decided at train time) pass.
    assert interventions.resolve_models(
        {"olmo_custom": "allenai/OLMo-2-1124-7B-SFT"}
    ) == {"olmo_custom": "allenai/OLMo-2-1124-7B-SFT"}



def test_resolve_models_trusts_a_pinned_chat_format():
    # The HF name's spelling (`olmo3`) infers OLMo-2's family while the mtag
    # (`olmo-3`) infers ChatML; an explicit train.chat_format pins training's
    # template, so the name no longer decides anything and the pair is fine.
    models = {"neutral_olmo-3_7b": "value-generalization/neutral-sft-v3-olmo3-7b"}
    with pytest.raises(ValueError, match="different formats"):
        interventions.resolve_models(models)
    assert interventions.resolve_models(models, "olmo3_chatml") == models


def test_dataset_file_prefers_existing_legacy_csv(tmp_path):
    # Fresh dirs get JSONL; legacy dirs keep their CSV so the checkpoints
    # they produced stay reproducible (raw-text training, no chat template).
    assert interventions.dataset_file(tmp_path) == tmp_path / "dataset.jsonl"
    (tmp_path / "dataset.csv").touch()
    assert interventions.dataset_file(tmp_path) == tmp_path / "dataset.csv"


def test_conflictscope_pairs_and_resplits(tmp_path):
    rows = []
    for sid in range(6):
        for model, likert, response in (("m1", 1, "best"), ("m2", -1, "worst")):
            rows.append({"scenario_id": f"s{sid}", "value1": "honesty", "value2": "helpfulness", "likert": likert, "model": model, "conversation": f"USER: ask {sid}\nASSISTANT: {response}"})
    completions = tmp_path / "completions"; completions.mkdir()
    frame = pd.DataFrame(rows)
    frame[frame.model == "m1"].to_csv(completions / "m1.csv", index=False)
    frame[frame.model == "m2"].to_csv(completions / "m2.csv", index=False)
    pairs = tmp_path / "pairs.jsonl"
    conflictscope.generate_pairs(completions, ["honesty", "helpfulness"], pairs)
    result = pd.read_json(pairs, lines=True)
    # The target ranking prefers value1, so the legacy builder flips Likert
    # signs and selects m2's -1 response as the preferred completion.
    assert len(result) == 6 and set(result.chosen_model) == {"m2"}
    assert all(x[0]["role"] == "assistant" for x in result.chosen)
    capped = tmp_path / "pairs_capped.jsonl"
    conflictscope.generate_pairs(
        completions, ["honesty", "helpfulness"], capped, seed=42, max_pairs=2
    )
    assert (
        pd.read_json(capped, lines=True).scenario_id.tolist()
        == result.scenario_id.iloc[:2].tolist()
    )
    # The legacy resplit path still runs on stringified-repr CSVs.
    pairs_csv = tmp_path / "pairs.csv"
    conflictscope.generate_pairs(completions, ["honesty", "helpfulness"], pairs_csv)
    scenarios = tmp_path / "scenarios.csv"
    pd.DataFrame({"scenario_id": [f"s{i}" for i in range(6)]}).to_csv(scenarios, index=False)
    conflictscope.resplit_steered(pairs_csv, scenarios, tmp_path / "resplit", seeds=[1, 2], test_size=2)
    assert len(pd.read_csv(tmp_path / "resplit_split1/test_outputs/gpt-4.1.csv")) == 2


def test_conflictscope_global_dedup_and_stratified_split(tmp_path):
    source = tmp_path / "candidates"
    (source / "outputs").mkdir(parents=True)
    rows = pd.DataFrame({
        "scenario_id": [f"s{i}" for i in range(8)],
        "value1": ["a"] * 8,
        "value2": ["b"] * 4 + ["c"] * 4,
        "description": ["a0", "a0-copy", "a1", "a2", "b0", "b0-copy", "b1", "b2"],
    })
    rows.to_csv(source / "outputs/model.csv", index=False)
    vectors = {
        "a0": [1, 0, 0, 0, 0, 0], "a0-copy": [1, 0, 0, 0, 0, 0],
        "a1": [0, 1, 0, 0, 0, 0], "a2": [0, 0, 1, 0, 0, 0],
        "b0": [0, 0, 0, 1, 0, 0], "b0-copy": [0, 0, 0, 1, 0, 0],
        "b1": [0, 0, 0, 0, 1, 0], "b2": [0, 0, 0, 0, 0, 1],
    }
    encoder = lambda texts: np.array([vectors[text.split()[0]] for text in texts])
    deduped = tmp_path / "deduped"
    report = scenario_pool.global_deduplicate(
        source, deduped, threshold=0.99, encoder=encoder
    )
    assert report["rejected_ids"] == ["s1", "s5"]
    assert report["existing_train_sources"] == []

    train, test = tmp_path / "train", tmp_path / "test"
    scenario_pool.split_dataset(deduped, train, test, test_size=2, seed=42)
    test_df = pd.read_csv(test / "outputs/model.csv")
    assert set(zip(test_df.value1, test_df.value2)) == {("a", "b"), ("a", "c")}
    import yaml
    report = yaml.safe_load((test / "split_report.yaml").read_text())
    assert report["strategy"] == "stratified_value_pair"


def test_conflictscope_dedup_against_immutable_training(tmp_path):
    existing = tmp_path / "existing.csv"
    pd.DataFrame({
        "scenario_id": ["train-1"], "value1": ["a"], "value2": ["b"],
        "description": ["training"],
    }).to_csv(existing, index=False)
    original = existing.read_bytes()
    candidates = tmp_path / "candidates"
    (candidates / "outputs").mkdir(parents=True)
    pd.DataFrame({
        "scenario_id": ["eval-train-copy", "eval-unique", "eval-unique-copy"],
        "value1": ["a"] * 3, "value2": ["b"] * 3,
        "description": ["training-copy", "unique", "unique-copy"],
    }).to_csv(candidates / "outputs/model.csv", index=False)
    vectors = {
        "training": [1, 0], "training-copy": [1, 0],
        "unique": [0, 1], "unique-copy": [0, 1],
    }
    encoder = lambda texts: np.array([vectors[text.split()[0]] for text in texts])
    deduped = tmp_path / "deduped"
    report = scenario_pool.global_deduplicate(
        candidates, deduped, threshold=0.99,
        existing_train=[existing], encoder=encoder,
    )
    assert report["rejected_ids"] == ["eval-train-copy", "eval-unique-copy"]
    assert existing.read_bytes() == original

    train, test = tmp_path / "train", tmp_path / "test"
    scenario_pool.split_dataset(deduped, train, test, existing_train=[existing])
    assert pd.read_csv(train / "outputs/model.csv").scenario_id.tolist() == ["train-1"]
    assert pd.read_csv(test / "outputs/model.csv").scenario_id.tolist() == ["eval-unique"]


def test_intervention_and_evaluation_hashes_are_separated():
    cfg = _cfg("label_subset_grid.yaml")

    eval_changed = copy.deepcopy(cfg)
    eval_changed["evaluation"]["judge"]["model"] = "Qwen/another-judge"
    assert intervention_id(eval_changed) == intervention_id(cfg)
    assert gt_id(eval_changed) != gt_id(cfg)

    train_changed = copy.deepcopy(cfg)
    train_changed["intervention"]["train"]["seed"] = 7
    assert intervention_id(train_changed) != intervention_id(cfg)
    assert gt_id(train_changed) != gt_id(cfg)


def test_manifest_round_trip_and_foreign_refusal(cluster):
    cfg = _cfg("label_subset_grid.yaml")
    interventions.claim(cfg, cluster)
    manifest = interventions.load_manifest(cfg, cluster)
    persisted = yaml.safe_load(interventions.manifest_path(cfg, cluster).read_text())

    assert persisted == manifest
    assert len(manifest["interventions"]) == 5 * 13
    assert {entry["kind"] for entry in manifest["interventions"]} == {"checkpoint"}
    # Declared up front, not yet trained: every entry is pending.
    assert len(interventions.missing_entries(manifest)) == 5 * 13

    persisted["interventions"][0]["path"] = "/foreign/checkpoint"
    interventions.manifest_path(cfg, cluster).write_text(
        yaml.safe_dump(persisted, sort_keys=False)
    )
    with pytest.raises(RuntimeError, match="different intervention config"):
        interventions.claim(cfg, cluster)


def test_explicit_reference_reuses_intervention_across_eval_configs(cluster):
    owner = _cfg("label_subset_grid.yaml")
    interventions.claim(owner, cluster)
    iid = intervention_id(owner)

    def referenced(max_scenarios):
        evaluation = copy.deepcopy(owner["evaluation"])
        evaluation["intervention"] = iid
        evaluation["max_scenarios"] = max_scenarios
        return resolve_experiment({"name": owner["name"], "evaluation": evaluation})

    first, second = referenced(10), referenced(20)
    assert intervention_id(first) == intervention_id(second) == iid
    assert gt_id(first) != gt_id(second)
    assert interventions.load_manifest(first, cluster) == interventions.load_manifest(
        second, cluster
    )
    assert interventions.stages(first, cluster) == []


def test_base_eval_is_shared_across_intervention_artifacts(cluster):
    first = _cfg("label_subset_grid.yaml")
    first["intervention"]["models"] = {"olmo7b_sft": "allenai/OLMo-2-1124-7B-SFT"}
    second = copy.deepcopy(first)
    second["intervention"]["values"] = ["calibrated_uncertainty"]
    assert intervention_id(first) != intervention_id(second)
    assert gt_id(first) != gt_id(second)
    assert conflictscope_eval.base_eval_id(first, cluster) == conflictscope_eval.base_eval_id(
        second, cluster
    )

    first_manifest = interventions.build_manifest(first, cluster)
    second_manifest = interventions.build_manifest(second, cluster)
    first_base = conflictscope_eval.stages(first, cluster, first_manifest)[-1].tasks[0]
    second_base = conflictscope_eval.stages(second, cluster, second_manifest)[-1].tasks[0]
    shared = conflictscope_eval.base_evals_dir(first, cluster)
    assert str(shared) in first_base.command
    assert str(shared) in second_base.command
    assert first_base.done != second_base.done  # per-run links, one shared payload



def test_base_eval_id_does_not_depend_on_where_the_repo_lives(cluster, tmp_path):
    import dataclasses

    cfg = _cfg("label_subset_grid.yaml")
    # Another location, and another layout (data root inside the repo).
    for repo, data in [(tmp_path / "clone2" / "repo", tmp_path / "clone2" / "data"),
                       (tmp_path / "clone3", tmp_path / "clone3" / "data")]:
        elsewhere = dataclasses.replace(cluster, repo=repo, data=data)
        assert conflictscope_eval.base_eval_id(cfg, cluster) == (
            conflictscope_eval.base_eval_id(cfg, elsewhere)
        )
    # ...but it still tracks which scenarios are evaluated.
    moved = copy.deepcopy(cfg)
    moved["evaluation"]["scenarios"] = "data/scenarios/some_other_set"
    assert conflictscope_eval.base_eval_id(moved, cluster) != (
        conflictscope_eval.base_eval_id(cfg, cluster)
    )


def test_base_scaffold_renames_the_shared_csv_and_writes_its_spec(cluster):
    import json

    served = _cfg("base_only_served.yaml")
    served["evaluation"]["scaffolds"] = {"olmo7b_sft": "urial0"}
    # A scaffold needs an in-process assistant: refused at plan time when the
    # experiment serves one.
    with pytest.raises(ValueError, match="in-process assistant"):
        conflictscope_eval.stages(
            served, cluster, interventions.build_manifest(served, cluster)
        )

    plain = _cfg("label_subset_grid.yaml")
    plain["intervention"]["models"] = {"olmo7b_sft": "allenai/OLMo-2-1124-7B-SFT"}
    cfg = copy.deepcopy(plain)
    cfg["evaluation"]["scaffolds"] = {"olmo7b_sft": "urial0"}
    # A per-model knob, not an eval knob: the GT run moves, the shared
    # store does not -- the scaffold rides in the CSV name instead.
    assert gt_id(plain) != gt_id(cfg)
    assert conflictscope_eval.base_eval_id(plain, cluster) == (
        conflictscope_eval.base_eval_id(cfg, cluster)
    )
    manifest = interventions.build_manifest(cfg, cluster)
    tasks = conflictscope_eval.stages(cfg, cluster, manifest)[-1].tasks
    base = tasks[0]
    store = conflictscope_eval.base_evals_dir(cfg, cluster)
    assert "OLMo-2-1124-7B-SFT_urial0_base.csv" in base.command
    assert f"--assistant-scaffold {store / 'scaffolds' / 'urial0.json'}" in base.command
    assert str(base.done).endswith("olmo7b_sft_base.csv")  # link name unchanged
    spec = json.loads((store / "scaffolds" / "urial0.json").read_text())
    assert spec["chat_template"].startswith("{{- '# Instruction")
    assert spec["stop"] == ["\n# Query:", "\n# Instruction"]
    # Base slot only: the intervention rows keep their trained chat template.
    assert all("--assistant-scaffold" not in t.command for t in tasks[1:])

    cfg["evaluation"]["scaffolds"] = {"olmo7b_sft": "urial9"}
    with pytest.raises(KeyError, match="Unknown scaffold"):
        conflictscope_eval.stages(cfg, cluster, manifest)


def test_served_assistant_rejects_checkpoints_and_multiple_models(cluster):
    # A shared served assistant only fits base-only evals: checkpoint
    # interventions are each their own assistant, and one server serves one
    # base model. Both misconfigurations fail before submitting anything.
    dpo = _cfg("label_subset_grid.yaml")
    dpo["evaluation"]["assistant"]["serve"] = "local"
    with pytest.raises(ValueError, match="checkpoint interventions"):
        conflictscope_eval.stages(
            dpo, cluster, interventions.build_manifest(dpo, cluster)
        )

    steer = _cfg("base_only_served.yaml")
    steer["intervention"]["models"] = {
        "olmo7b_sft": "allenai/OLMo-2-1124-7B-SFT",
        "tulu8b_sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    }
    with pytest.raises(ValueError, match="one base model"):
        conflictscope_eval.stages(
            steer, cluster, interventions.build_manifest(steer, cluster)
        )


def test_scenario_pool_identity_ignores_scheduling_but_not_generation_knobs():
    cfg = _cfg("conflictscope_pairs_smoke.yaml")
    training_pool = cfg["intervention"]["pool"]
    eval_pool = cfg["evaluation"]["scenarios"]["pool"]
    assert pool_id(training_pool) == pool_id(eval_pool)

    scheduling = copy.deepcopy(training_pool)
    scheduling["scenario_gen"]["time"] = "00:10:00"
    assert pool_id(scheduling) == pool_id(training_pool)

    behavioral = copy.deepcopy(training_pool)
    behavioral["scenario_gen"]["num_scenarios"] += 1
    assert pool_id(behavioral) != pool_id(training_pool)


# ── Oracle labeling (label_subset generate mode) ─────────────────────────────


def _oracle_cfg(cluster, **oracle):
    """A resolved generate-mode config over a 1-value slice of the tenets."""
    cfg = resolve_experiment({
        "name": "labels_gen",
        "intervention": {
            "method": "label_subset",
            "value_set": "constitution_tenets_v3",
            "values": ["accurate_overall_impressions"],
            "models": {"olmo7b_sft": "allenai/OLMo-2-1124-7B-SFT"},
            "labels": {"sources": ["hh"], "oracle": {
                "model": "Qwen/Qwen3.6-27B", "serve": "local",
                "shards": {"hh": 2}, **oracle,
            }},
        },
        "evaluation": {
            "method": "conflictscope", "scenarios": "data/scenarios/x",
            "judge": {"model": "gpt-4.1"},
        },
    })
    cfg["_path"] = "/tmp/labels_gen.yaml"
    return cfg


def test_prelabeled_config_does_not_materialize_oracle_defaults():
    cfg = _cfg("label_subset_grid.yaml")
    assert "oracle" not in cfg["intervention"]["labels"]
    assert not label_subset.generates_labels(cfg)


def test_label_id_tracks_judgment_knobs_not_scheduling_knobs(cluster):
    base = label_subset.label_id(_oracle_cfg(cluster), cluster)
    # Scheduling: same judgments, same store.
    assert base == label_subset.label_id(
        _oracle_cfg(cluster, shards={"hh": 8}, concurrency=8), cluster
    )
    # Judgment-affecting: a different oracle must not write into that store.
    for changed in (
        {"model": "Qwen/Qwen2.5-7B-Instruct"},
        {"orderings": 1},
        {"max_chars": 500},
        {"template": "Consider {description}."},
    ):
        assert label_subset.label_id(_oracle_cfg(cluster, **changed), cluster) != base


def test_label_claim_refuses_a_foreign_manifest(cluster):
    cfg = _oracle_cfg(cluster)
    manifest = label_subset.claim(cfg, cluster)
    label_subset.claim(cfg, cluster)  # idempotent
    record = yaml.safe_load(manifest.read_text())
    record["identity"]["model"] = "some/other-judge"
    manifest.write_text(yaml.safe_dump(record))
    with pytest.raises(RuntimeError, match="different oracle config"):
        label_subset.claim(cfg, cluster)


def test_generate_mode_stages_release_the_oracle_before_training(cluster):
    cfg = _oracle_cfg(cluster)
    stages = label_subset.stages(cfg, cluster)
    names = [s.name for s in stages]
    assert names[:4] == ["serve_oracle", "label", "merge_labels", "datasets"]
    assert names.index("label") < names.index("train_olmo7b_sft")
    label_stage = stages[1]
    assert label_stage.gpus == 0  # talks HTTP to the server; holds no GPU itself
    assert label_stage.cancel_servers == ("serve_oracle",)
    assert [t.key for t in label_stage.tasks] == ["label_hh_0", "label_hh_1"]
    assert "wait_ready" in label_stage.tasks[0].command
    assert "--source hh --shard 1 --shards 2" in label_stage.tasks[1].command

    # Existing judgments are reused, not re-judged: no server, no label array.
    directory = label_subset.labels_dir(cfg, cluster)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hh.csv").write_text("row_id,principle_id,order,prob_A,prob_B,prob_NA,cost\n")
    assert [s.name for s in label_subset.stages(cfg, cluster)][0] == "datasets"


def test_generate_mode_reuses_complete_prebuilt_datasets(cluster):
    cfg = _oracle_cfg(cluster)
    dataset = interventions.datasets_dir(cfg, cluster) / "accurate_overall_impressions" / "dataset.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text('{"prompt": [], "chosen": [], "rejected": []}\n')

    stages = label_subset.stages(cfg, cluster)
    names = [stage.name for stage in stages]
    assert "serve_oracle" not in names
    assert "label" not in names
    assert names == ["datasets", "train_olmo7b_sft"]
    assert not stages[0].pending()


def test_principle_subset_keeps_positional_ids(cluster):
    # A value set carries values no run trains on; judging them is real GPU
    # time. A subset drops their rows without renumbering the
    # rest — ids stay positional, so subset and full CSVs stay comparable.
    cfg = _oracle_cfg(cluster, principles=["accurate_overall_impressions", "balanced_ethical_perspectives"])
    rows = label_subset.labeled_principles(cfg, cluster)
    assert [(r["id"], r["key"]) for r in rows] == [(5, "accurate_overall_impressions"), (7, "balanced_ethical_perspectives")]
    # A different set of judgments -> a different store.
    assert label_subset.label_id(cfg, cluster) != label_subset.label_id(_oracle_cfg(cluster), cluster)
    with pytest.raises(KeyError, match="not in the value set"):
        label_subset.labeled_principles(_oracle_cfg(cluster, principles=["nope"]), cluster)


def test_training_a_value_the_oracle_does_not_judge_raises(cluster):
    # values: ["accurate_overall_impressions"] — labeling only
    # balanced_ethical_perspectives would train on
    # an empty dataset, so it fails before the oracle spends a GPU-hour.
    cfg = _oracle_cfg(cluster, principles=["balanced_ethical_perspectives"])
    with pytest.raises(ValueError, match="not judged by the oracle"):
        label_subset.stages(cfg, cluster)


def test_principle_ids_follow_value_set_order(cluster):
    rows = label_subset.principle_statements({"accurate_overall_impressions": "not telling white lies"})
    assert rows == [{"id": 0, "key": "accurate_overall_impressions",
                     "statement": "The AI should prioritize not telling white lies."}]


class _FakeChoice:
    def __init__(self, tops):
        self.logprobs = type("L", (), {"content": [type("T", (), {"top_logprobs": tops})()]})()


class _FakeClient:
    """A judge that always favors whichever response is shown in slot A."""

    def __init__(self):
        self.completions = self
        self.chat = self

    def create(self, **kwargs):
        tops = [
            type("E", (), {"token": "A", "logprob": np.log(0.8)})(),
            type("E", (), {"token": "B", "logprob": np.log(0.2)})(),
        ]
        return type("R", (), {"choices": [_FakeChoice(tops)]})()


def test_label_shard_averages_orderings_and_unswaps(cluster, monkeypatch):
    cfg = _oracle_cfg(cluster)
    text = pd.DataFrame({"prompt": ["q0", "q1"], "A": ["a0", "a1"], "B": ["b0", "b1"]})
    monkeypatch.setitem(label_subset.SOURCE_LOADERS, "hh", lambda: text)
    judge = label_subset.Judge("http://x/v1", "Qwen/Qwen3.6-27B", client=_FakeClient())

    out = label_subset.label_shard(cfg, cluster, "hh", shard=0, shards=2, judge=judge)
    df = pd.read_csv(out)
    # shard 0 of 2 -> row 0 only; one row per principle, orderings collapsed.
    n_principles = len(V.load_value_set(REPO / "value_sets/constitution_tenets_v3.json"))
    assert list(df.row_id.unique()) == [0]
    assert len(df) == n_principles
    assert set(df.order) == {0}
    # A position-biased judge that always picks slot A is exactly cancelled by
    # averaging the two orderings: 0.8 shown first, 0.2 when swapped back.
    assert np.allclose(df.prob_A, 0.5) and np.allclose(df.prob_B, 0.5)


def test_merge_source_requires_every_shard(cluster, monkeypatch):
    cfg = _oracle_cfg(cluster)
    text = pd.DataFrame({"prompt": ["q0", "q1"], "A": ["a0", "a1"], "B": ["b0", "b1"]})
    monkeypatch.setitem(label_subset.SOURCE_LOADERS, "hh", lambda: text)
    judge = label_subset.Judge("http://x/v1", "Qwen/Qwen3.6-27B", client=_FakeClient())

    label_subset.label_shard(cfg, cluster, "hh", shard=0, shards=2, judge=judge)
    with pytest.raises(FileNotFoundError, match="1/2 shards missing"):
        label_subset.merge_source(cfg, cluster, "hh", shards=2)

    label_subset.label_shard(cfg, cluster, "hh", shard=1, shards=2, judge=judge)
    merged = pd.read_csv(label_subset.merge_source(cfg, cluster, "hh", shards=2))
    n_principles = len(V.load_value_set(REPO / "value_sets/constitution_tenets_v3.json"))
    assert len(merged) == 2 * n_principles
    assert list(merged.columns) == [
        "row_id", "principle_id", "order", "prob_A", "prob_B", "prob_NA", "cost"
    ]


def _assign_inputs(rows):
    """rows: (source, row_id, prompt, A, B, {pid: score}) -> (scores, keys, prompt_keys)."""
    long, text_rows = [], []
    for src, rid, prompt, a, b, sc in rows:
        text_rows.append((src, rid, prompt, a, b))
        for pid, s in sc.items():
            long.append({"source": src, "row_id": rid, "principle_id": pid, "score": s})
    text = pd.DataFrame(text_rows, columns=["source", "row_id", "prompt", "A", "B"])
    idx = pd.MultiIndex.from_frame(text[["source", "row_id"]])
    keys = pd.Series(exclusive_assignment.triple_key(text).values, index=idx)
    prompt_keys = pd.Series(exclusive_assignment.prompt_key(text).values, index=idx)
    return pd.DataFrame(long), keys, prompt_keys


def test_assignment_unit_prompt_caps_one_row_per_prompt():
    # One prompt ("shared") carries two eligible pairs, one strong for each
    # value; "alt2"/"alt3" are weaker fallbacks so unit=prompt stays feasible.
    rows = [
        ("s", 0, "shared", "a0", "b0", {10: 0.9}),
        ("s", 1, "shared", "a1", "b1", {20: 0.9}),
        ("s", 2, "alt2", "a2", "b2", {20: 0.8}),
        ("s", 3, "alt3", "a3", "b3", {10: 0.8}),
    ]
    scores, keys, prompt_keys = _assign_inputs(rows)
    kw = dict(rule="mincost_flow", tau=0.5, cap=1, floor=0, seed=0)

    # unit=triple (default): the two strongest rows both come from "shared"
    # (distinct {A,B} pairs), so one prompt trains both values.
    triple = exclusive_assignment.assign(scores, keys, [10, 20], **kw)
    tri_pk = prompt_keys.loc[list(zip(triple.source, triple.row_id))]
    assert tri_pk.tolist() == ["shared", "shared"]

    # unit=prompt: the prompt layer (cap 1) forbids that; each value still
    # fills its cap, but from two distinct prompts.
    pr = exclusive_assignment.assign(
        scores, keys, [10, 20], unit="prompt", prompt_keys=prompt_keys, **kw)
    pr_pk = prompt_keys.loc[list(zip(pr.source, pr.row_id))]
    assert len(pr) == 2
    assert not pr_pk.duplicated().any()
    assert sorted(pr.principle_id) == [10, 20]


def test_assignment_unit_prompt_requires_flow_and_keys():
    scores, keys, prompt_keys = _assign_inputs(
        [("s", 0, "p", "a", "b", {10: 0.9})])
    with pytest.raises(ValueError, match="mincost_flow"):
        exclusive_assignment.assign(
            scores, keys, [10], rule="argmax_bounded", tau=0.5, cap=1, floor=0,
            seed=0, unit="prompt", prompt_keys=prompt_keys)
    with pytest.raises(ValueError, match="prompt_keys"):
        exclusive_assignment.assign(
            scores, keys, [10], rule="mincost_flow", tau=0.5, cap=1, floor=0,
            seed=0, unit="prompt")


def test_schedule_wave_interleaves_train_and_eval_per_model(cluster):
    cfg = _cfg("label_subset_grid.yaml")  # 5 models x 13 values, local judge
    manifest = interventions.build_manifest(cfg, cluster)
    training = interventions.stages(cfg, cluster)
    evaluation = conflictscope_eval.stages(cfg, cluster, manifest)

    waved = interventions.interleave_waves(training, evaluation, manifest, wave=5)

    mtags = list(manifest["models"])
    # Judge registered first but ON DEMAND (served by the first eval wave's
    # needs_servers, never submitted ahead of the first training wave), then
    # train/eval pairs per model in manifest order: 13 -> 5, 5, 3.
    assert waved[0].name == "serve_judge" and waved[0].server
    assert waved[0].on_demand
    assert not evaluation[0].on_demand  # the un-waved stage list still serves up front
    expected = ["serve_judge"]
    for m in mtags:
        for k in range(3):
            expected += [f"train_{m}_w{k}", f"eval_{m}_w{k}"]
    assert [s.name for s in waved] == expected

    m0 = mtags[0]
    values = interventions.values_for(manifest, m0)
    by_name = {s.name: s for s in waved}
    assert [t.key for t in by_name[f"train_{m0}_w0"].tasks] == [f"{m0}/{v}" for v in values[:5]]
    assert [t.key for t in by_name[f"eval_{m0}_w0"].tasks] == [f"{m0}_{v}" for v in values[:5]]
    # The base eval is not done in this fresh tmp cluster -> rides in the LAST
    # wave, after that wave's steered values; never in the first.
    last = by_name[f"eval_{m0}_w2"]
    assert [t.key for t in last.tasks] == [f"{m0}_{v}" for v in values[10:]] + [f"{m0}_base"]
    assert f"{m0}_base" not in {t.key for t in by_name[f"eval_{m0}_w0"].tasks}
    # Every eval wave releases the judge; the train stages keep their spec.
    for s in waved[1:]:
        if s.name.startswith("eval_"):
            assert s.needs_servers == ("serve_judge",)
            assert s.cancel_servers == ("serve_judge",)
        else:
            assert s.cancel_servers == () and s.gpus == training[0].gpus

    # Nothing lost, nothing duplicated: the waves are a permutation of the
    # original task sets.
    orig = sorted(t.key for st in training + evaluation if not st.server for t in st.tasks)
    now = sorted(t.key for st in waved if not st.server for t in st.tasks)
    assert orig == now
    # A done base task is dropped from the last wave rather than re-run.
    done_base = evaluation[1].tasks[0]
    done_base.done = lambda: True
    waved2 = interventions.interleave_waves(training, evaluation, manifest, wave=5)
    assert f"{m0}_base" not in {t.key for s in waved2 for t in s.tasks}


def test_schedule_block_is_validated_and_never_hashed():
    from valuegen.config import canonical_experiment

    cfg = _cfg("label_subset_grid.yaml")
    cfg_w = copy.deepcopy(cfg)
    cfg_w["schedule"] = {"wave": 8}
    assert gt_id(cfg_w) == gt_id(cfg)
    assert intervention_id(cfg_w) == intervention_id(cfg)
    assert "schedule" not in canonical_experiment(cfg_w)

    raw = yaml.safe_load((FIXTURES / "label_subset_grid.yaml").read_text())
    for bad in ({"wave": 0}, {"wave": "8"}, {"waves": 8}, [8]):
        raw["schedule"] = bad
        with pytest.raises(ValueError):
            resolve_experiment(raw)
    raw["schedule"] = {"wave": 8}
    assert resolve_experiment(raw)["schedule"] == {"wave": 8}
