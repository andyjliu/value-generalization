"""Elicitation artifact, adapter, and registry contracts."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from valuegen import values as V
from valuegen.elicitation import datasets as D
from valuegen.elicitation import registry
from valuegen.elicitation.adapters import (
    conflictscope,
    descriptions,
    pair_clean,
    persona,
)


VALUES = ["hard_constraint_fidelity", "accurate_overall_impressions"]


def _pair_frame(value: str, polarity: str, answers=("a0", "a1")) -> pd.DataFrame:
    return pd.DataFrame({
        "question": ["q0", "q1"],
        "system_prompt": ["s0", "s1"],
        "answer": list(answers),
        "polarity": [polarity, polarity],
        "value": [value, value],
        "pair_id": [10, 11],
    })


def _cfg(method="conflictscope_action_prompt", **overrides):
    cfg = {
        "method": method,
        "artifact": "pairs",
        "schema_version": 1,
        "value_set": "constitution_tenets_v3",
        "values": list(VALUES),
        "legacy": False,
        "model": "org/model",
        "n_pairs": 2,
        "seed": 42,
        "temperature": 0.7,
        "max_tokens": 1000,
        "gpus": 1,
        "scenarios_dir": "/tmp/scenarios",
    }
    cfg.update(overrides)
    return cfg


def test_resolve_config_materializes_identity_and_aliases(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    cfg = registry.resolve_config(
        "conflictscope_v2", "constitution_tenets_v3", model="org/model",
        values=VALUES, params={"scenarios_dir": str(scenarios)},
    )
    assert cfg["method"] == "conflictscope_action_prompt"
    assert cfg["scenarios_dir"] == str(scenarios.resolve())
    assert cfg["temperature"] == 0.7 and cfg["schema_version"] == 1
    assert registry.needs_slurm(cfg)

    api = dict(cfg, model="gpt-4.1")
    assert not registry.needs_slurm(api)
    desc = registry.resolve_config("none", "constitution_tenets_v3", model="ignored")
    assert desc["model"] is None and desc["artifact"] == "descriptions"
    assert D.data_artifact_id(desc).startswith("shared-")


def test_resolve_config_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="missing required"):
        registry.resolve_config("conflictscope_v2", "constitution_tenets_v3", model="m")
    with pytest.raises(KeyError, match="unknown params"):
        registry.resolve_config(
            "conflictscope_v2", "constitution_tenets_v3", model="m",
            params={"scenarios_dir": str(tmp_path), "bogus": 1},
        )
    with pytest.raises(KeyError, match="not in value set"):
        registry.resolve_config(
            "conflictscope_v2", "constitution_tenets_v3", model="m", values=["missing"],
            params={"scenarios_dir": str(tmp_path)},
        )


def test_resolve_config_only_accepts_value_sets_from_the_registry(tmp_path):
    external = tmp_path / "private_values.json"
    external.write_text(json.dumps({"custom_a": "A private value"}))
    with pytest.raises(ValueError, match="outside the registry"):
        registry.resolve_config("descriptions", external)

    # The dangerous case: an external file whose stem collides with a registry
    # name. Keeping the stem would silently build the registry's spec_tenets under a
    # config hash identical to a plain-spec_tenets config.
    collide = tmp_path / "spec_tenets.json"
    collide.write_text(json.dumps({k: f"REWORDED: {v}" for k, v in V.load_value_set("spec_tenets").items()}))
    with pytest.raises(ValueError, match="outside the registry"):
        registry.resolve_config("descriptions", collide)

    # A path *into* the registry is still a legitimate way to name a set.
    registry_path = V.value_sets_dir() / "spec_tenets.json"
    relative = Path(os.path.relpath(registry_path))
    for ref in ("spec_tenets", registry_path, relative):
        assert registry.resolve_config("descriptions", ref)["value_set"] == "spec_tenets"


def test_artifact_identity_changes_with_behavioral_config(cluster):
    cfg = _cfg()
    assert D.data_artifact_id(cfg) == D.data_artifact_id(dict(cfg))
    assert D.data_artifact_id(cfg) != D.data_artifact_id(dict(cfg, seed=7))
    assert D.data_artifact_id(cfg) != D.data_artifact_id(dict(cfg, schema_version=2))
    legacy = dict(cfg, legacy=True)
    assert D.data_artifact_id(legacy).startswith("legacy-")
    assert D.resolve_artifact(cluster, cfg).root == D.artifact_dir(cluster, cfg)


def test_config_claim_completion_manifest_and_discovery(cluster):
    artifact = D.resolve_artifact(cluster, _cfg())
    D.persist_data_config(artifact.root, artifact.cfg)
    assert D.config_id_at(artifact.root) == artifact.artifact_id
    with pytest.raises(RuntimeError, match="Config-ID mismatch"):
        D.persist_data_config(artifact.root, dict(artifact.cfg, seed=9))

    D.write_pairs(
        artifact, VALUES[0], _pair_frame(VALUES[0], "pos"),
        _pair_frame(VALUES[0], "neg", ("n0", "n1")),
    )
    assert artifact.value_done(VALUES[0])
    assert not artifact.value_done(VALUES[1])
    assert not artifact.is_complete()
    D.write_pairs(
        artifact, VALUES[1], _pair_frame(VALUES[1], "pos"),
        _pair_frame(VALUES[1], "neg", ("n0", "n1")),
    )
    D.write_manifest(artifact, {"filter": {"threshold": 50}})
    assert artifact.is_complete()
    assert D.read_manifest(artifact.root)["n_pairs_per_value"] == {
        VALUES[0]: 2, VALUES[1]: 2,
    }
    assert D.find_artifact_by_id(cluster, artifact.artifact_id).root == artifact.root
    assert D.find_artifact_by_id(cluster, str(artifact.root)).artifact_id == artifact.artifact_id
    assert [a.artifact_id for a in D.list_artifacts(cluster)] == [artifact.artifact_id]


def test_write_pairs_validates_schema_and_pairing(cluster):
    artifact = D.resolve_artifact(cluster, _cfg(values=[VALUES[0]]))
    pos = _pair_frame(VALUES[0], "pos")
    neg = _pair_frame(VALUES[0], "neg")
    with pytest.raises(ValueError, match="row counts differ"):
        D.write_pairs(artifact, VALUES[0], pos, neg.iloc[:1])
    with pytest.raises(ValueError, match="missing schema"):
        D.write_pairs(artifact, VALUES[0], pos.drop(columns="answer"), neg)


def test_generation_conversion_filter_and_paired_subsample():
    source = pd.DataFrame({
        "scenario_id": [10, 11, 12],
        "prompt": ["q0", "q1", "q2"],
        "pos_response": ["p0", "", "p2"],
        "neg_response": ["n0", "n1", "n2"],
        "pos_system": ["ps0", "ps1", "ps2"],
        "neg_system": ["ns0", "ns1", "ns2"],
    })
    pos, neg = D.pair_frames_from_generation(source, "v")
    assert pos.pair_id.tolist() == neg.pair_id.tolist() == [10, 11, 12]
    assert pos.system_prompt.tolist() == ["ps0", "ps1", "ps2"]
    pos, neg = D.drop_empty_pairs(pos, neg)
    assert pos.pair_id.tolist() == neg.pair_id.tolist() == [10, 12]
    pos, neg = D.subsample_pairs(pos, neg, 1, seed=42)
    assert len(pos) == len(neg) == 1 and pos.pair_id.iloc[0] == neg.pair_id.iloc[0]


def test_persona_extract_roundtrip_preserves_effective_fork_inputs(cluster, tmp_path):
    value = VALUES[0]
    pos_src = pd.DataFrame({
        "question": ["q0", "q1", "q2", "q3"],
        "prompt": ["P0", "P1", "P2", "P3"],
        "answer": ["a0", "a1", "a2", None],
        value: [100, 49, 100, 100],
        "coherence": [100, 100, 40, 100],
    })
    neg_src = pd.DataFrame({
        "question": ["q0", "q1", "q2", "q3"],
        "prompt": ["N0", "N1", "N2", "N3"],
        "answer": ["b0", "b1", "b2", "b3"],
        value: [0, 0, 0, 0],
        "coherence": [100, 100, 100, 100],
    })
    pos_path, neg_path = tmp_path / "pos.csv", tmp_path / "neg.csv"
    pos_src.to_csv(pos_path, index=False); neg_src.to_csv(neg_path, index=False)
    pos, neg = D.from_persona_extract(pos_path, neg_path, value, threshold=50)
    assert pos.answer.tolist() == ["a0"] and neg.answer.tolist() == ["b0"]

    artifact = D.resolve_artifact(cluster, _cfg(values=[value]))
    D.write_pairs(artifact, value, pos, neg)
    compat = D.build_fork_compat(artifact, "org/model")
    compat_pos = pd.read_csv(compat / f"{value}_pos_instruct.csv")
    compat_neg = pd.read_csv(compat / f"{value}_neg_instruct.csv")
    assert compat_pos.prompt.tolist() == ["P0"]
    assert compat_neg.prompt.tolist() == ["N0"]
    assert compat_pos[value].tolist() == [100]
    assert compat_neg[value].tolist() == [0]
    assert compat_pos.coherence.tolist() == compat_neg.coherence.tolist() == [100]


def test_fork_compat_formats_bare_questions_lazily(cluster, monkeypatch):
    value = VALUES[0]
    artifact = D.resolve_artifact(cluster, _cfg(values=[value]))
    D.write_pairs(
        artifact, value, _pair_frame(value, "pos"), _pair_frame(value, "neg")
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"CHAT:{messages[0]['content']}"

    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    compat = D.build_fork_compat(artifact, "org/model")
    assert pd.read_csv(compat / f"{value}_pos_instruct.csv").prompt.tolist() == [
        "CHAT:q0", "CHAT:q1"
    ]


def test_descriptions_build_is_complete_and_shared(cluster):
    cfg = registry.resolve_config("descriptions", "constitution_tenets_v3", values=VALUES)
    artifact = registry.build_inline(cfg, cluster)
    frame = pd.read_csv(artifact.descriptions_path)
    assert frame.value.tolist() == VALUES
    assert frame.description.str.len().min() > 0
    assert artifact.is_complete()


def test_action_generation_preserves_orientation_and_system_prompts(monkeypatch):
    calls = []

    class Client:
        def generate(self, messages):
            calls.append(messages)
            return f"response-{len(calls)}"

    monkeypatch.setattr(conflictscope, "_client", lambda cfg: Client())
    scenarios = pd.DataFrame([
        {"scenario_id": "s0", "user_prompt": "{'persona': 'P', 'goal': 'G'}",
         "value1": "v", "value2": "x", "action1": "support-v", "action2": "support-x"},
        {"scenario_id": "s1", "user_prompt": "plain", "value1": "x", "value2": "v",
         "action1": "support-x-2", "action2": "support-v-2"},
    ])
    cfg = _cfg(values=["v"], n_pairs=2)
    out = conflictscope._generate_steered(cfg, "v", scenarios, conflictscope._action_systems("v"))
    assert out.prompt.tolist() == ["P G", "plain"]
    assert "support-v" in out.pos_system.iloc[0] and "support-x" in out.neg_system.iloc[0]
    assert "support-v-2" in out.pos_system.iloc[1]
    assert calls[0][0]["content"] == out.pos_system.iloc[0]
    assert calls[0][1] == {"role": "user", "content": "P G"}


def test_legacy_action_conversion_orients_chosen_by_value(cluster, tmp_path):
    value = VALUES[0]
    source = tmp_path / "legacy"; source.mkdir()
    pd.DataFrame([
        {"scenario_id": "s0", "prompt": repr([{"content": "q0"}]),
         "chosen": repr([{"content": "chosen0"}]), "rejected": repr([{"content": "reject0"}]),
         "chosen_value": value},
        {"scenario_id": "s1", "prompt": repr([{"content": "q1"}]),
         "chosen": repr([{"content": "chosen1"}]), "rejected": repr([{"content": "reject1"}]),
         "chosen_value": "other"},
    ]).to_csv(source / f"{value}.csv", index=False)
    cfg = _cfg(
        "legacy_conflictscope_action_pairs", values=[value], legacy=True,
        source_dir=str(source), n_pairs=None,
    )
    artifact = D.resolve_artifact(cluster, cfg)
    conflictscope.build_legacy_action_pairs(cfg, cluster, artifact)
    pos, neg = D.load_pairs(artifact, value)
    assert pos.answer.tolist() == ["chosen0", "reject1"]
    assert neg.answer.tolist() == ["reject0", "chosen1"]


def test_generation_stage_uses_persisted_config_cluster_and_two_file_done(cluster):
    cfg = _cfg(values=[VALUES[0]])
    stage = registry.generation_stage(cfg, cluster)
    artifact = D.resolve_artifact(cluster, cfg)
    task = stage.tasks[0]
    assert str(artifact.root / D.DATA_CONFIG) in task.command
    assert f"--cluster {cluster.source_path}" in task.command
    assert stage.gpus == 1 and stage.env == "default"
    artifact.root.mkdir(parents=True, exist_ok=True)
    artifact.neg_path(VALUES[0]).write_text("not enough")
    assert not task.is_done()
    artifact.pos_path(VALUES[0]).write_text("now complete")
    assert task.is_done()


def test_default_llm_delegate_injects_all_cluster_knobs(cluster, monkeypatch):
    cfg = registry.resolve_config(
        "default_llm", "constitution_tenets_v3", model="org/model", values=VALUES,
        params={"experiment": "existing", "gpus": 2},
    )
    artifact = D.resolve_artifact(cluster, cfg)
    orch = persona.run_pipeline_orchestrator(cfg, cluster, artifact, dry_run=True)
    cmd = orch._command("run")
    assert cmd[cmd.index("--array-throttle") + 1] == "4"
    assert cmd[cmd.index("--exclude") + 1] == "slow-a,slow-b"
    # Partition/QOS/GPU facts come from cluster.yaml, never the fork's defaults.
    flag = lambda name: cmd[cmd.index(name) + 1]
    assert (flag("--cpu-partition"), flag("--cpu-qos")) == ("cpu", "cpu_qos")
    assert (flag("--gpu-partition"), flag("--gpu-qos")) == ("", "")  # account default
    assert (flag("--gpu-type"), flag("--gpu-constraint")) == ("A6000", "")
    assert flag("--cpu-stage-gpus") == "0"
    import dataclasses
    multi = dataclasses.replace(cluster, gpu_type=["L40S", "A6000"],
                                array_partition="preempt", array_qos="preempt_qos")
    cmd = persona.run_pipeline_orchestrator(cfg, multi, artifact, dry_run=True)._command("run")
    assert (flag("--gpu-partition"), flag("--gpu-qos")) == ("preempt", "preempt_qos")
    assert (flag("--gpu-type"), flag("--gpu-constraint")) == ("", "L40S|A6000")
    for flag in ("--method", "--target", "--threshold", "--generate-model", "--gpus"):
        assert flag in cmd
    value_set = json.loads((artifact.root / "value_set.json").read_text())
    assert list(value_set) == VALUES


def test_default_llm_optional_knobs_pass_through_without_moving_hashes(cluster):
    base = registry.resolve_config(
        "default_llm", "constitution_tenets_v3", model="org/model", values=VALUES)
    assert "judge_model" not in base  # unset optionals stay out of the identity
    cfg = registry.resolve_config(
        "default_llm", "constitution_tenets_v3", model="org/model", values=VALUES,
        params={"generate_provider": "gemini", "judge_model": "gemini-x",
                "judge_scoring_protocol": "gemini_fulltext_0_100_v1",
                "layer_start": 18, "layer_end": 18},
    )
    assert cfg["experiment"] != base["experiment"]
    orch = persona.run_pipeline_orchestrator(
        cfg, cluster, D.resolve_artifact(cluster, cfg), dry_run=True)
    cmd = orch._command("run")
    assert cmd[cmd.index("--generate-provider") + 1] == "gemini"
    assert cmd[cmd.index("--judge-scoring-protocol") + 1] == "gemini_fulltext_0_100_v1"
    assert cmd[cmd.index("--layer-end") + 1] == "18"
    assert "--chat-template-family" not in cmd
    assert "fixed-layer evaluation (layer 18)" in "\n".join(
        persona.default_llm_plan(cfg, cluster, None))
    with pytest.raises(KeyError, match="unknown params"):
        registry.resolve_config(
            "default_llm", "constitution_tenets_v3", model="org/model", values=VALUES,
            params={"judge_modle": "typo"})


def test_pair_clean_truncates_restarts_and_drops_collapsed_pairs(cluster):
    values = [VALUES[0]]
    src_cfg = _cfg(values=values)
    source = D.resolve_artifact(cluster, src_cfg)
    good = "A clear answer with plenty of words in it to survive the minimum length filter easily."
    restarted = good + "\n\n# Query:\n```\nInvented follow-up question\n```\n\n# Answer:\n```\nmore text"
    headed = "# Answer:\n```\n" + good
    collapsed = "short\n\nQuestion: what now?\n" + good
    pos = _pair_frame(values[0], "pos", (restarted, headed))
    neg = _pair_frame(values[0], "neg", (good, collapsed))
    D.persist_data_config(source.root, src_cfg)
    D.write_pairs(source, values[0], pos, neg)
    D.write_manifest(source)

    cfg = registry.resolve_config(
        "pair_clean", "constitution_tenets_v3", values=values,
        params={"source_artifact": source.artifact_id, "min_words": 5, "max_words": 12},
    )
    assert cfg["clean_protocol"] == "scaffold_restart_v1"
    artifact = D.resolve_artifact(cluster, cfg)
    assert artifact.artifact_id != source.artifact_id
    pair_clean.build(cfg, cluster, artifact)
    cpos, cneg = D.load_pairs(artifact, values[0])
    # pair 1 (headed / collapsed) is dropped: neg side cut to "short" (< min_words)
    assert len(cpos) == len(cneg) == 1
    assert cpos.answer[0] == " ".join(good.split()[:12])  # restart cut, then capped
    assert cneg.answer[0] == " ".join(good.split()[:12])
    assert cpos.source_artifact[0] == source.artifact_id
    assert cpos.answer_source[0] == restarted
    # protocol details
    assert pair_clean.clean_answer(headed, 1000) == good
    assert pair_clean.clean_answer("real\nOkay, let's tackle this.\nmore", 1000) == "real"
    assert pair_clean.clean_answer("x <|im_start|>user", 1000) == "x <|im_start|>user"  # no newline: kept
    assert pair_clean.clean_answer("x\n<|im_start|>user", 1000) == "x"
    with pytest.raises(ValueError):
        pair_clean.clean_answer("x", 10, protocol="nope")
