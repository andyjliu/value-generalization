"""Predictor store, stage, and numerical contract tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from valuegen.elicitation import datasets as D
from valuegen.predictors import get_predictor
from valuegen.predictors import persona, sentence_emb, store, weight_steer


VALUES = ["alpha", "beta", "missing"]


def _artifact(tmp_path: Path, method="test_pairs", artifact_type="pairs") -> D.Artifact:
    cfg = {
        "method": method,
        "artifact": artifact_type,
        "schema_version": 1,
        "value_set": "test",
        "values": list(VALUES),
        "legacy": False,
        "model": None if artifact_type == "descriptions" else "policy/model",
    }
    return D.Artifact(tmp_path / "artifact", cfg)


def test_predictor_registry_contract():
    assert get_predictor("persona").requires == "pairs"
    assert get_predictor("weight_steer").default_data == "default_llm"
    assert get_predictor("sentence_emb").requires == "descriptions"
    with pytest.raises(KeyError, match="Unknown predictor"):
        get_predictor("unknown")


def test_cosine_grid_handles_missing_zero_and_nonfinite_vectors():
    vectors = {
        "alpha": np.array([1.0, 0.0]),
        "beta": np.array([1.0, 1.0]),
        "zero": np.zeros(2),
        "bad": np.array([np.inf, 0.0]),
    }
    values = ["alpha", "beta", "missing", "zero", "bad"]
    sim = store.cosine_grid(vectors, values)
    assert np.allclose(sim[:2, :2], [[1, 2 ** -0.5], [2 ** -0.5, 1]])
    assert np.isnan(sim[2:, :]).all() and np.isnan(sim[:, 2:]).all()


def test_nan_pad_reindexes_matrix():
    raw = np.array([[1.0, 0.25], [0.25, 1.0]])
    out = store.nan_pad(raw, ["beta", "alpha"], VALUES)
    assert np.allclose(out[:2, :2], [[1.0, 0.25], [0.25, 1.0]])
    assert np.isnan(out[2]).all() and np.isnan(out[:, 2]).all()


def test_store_claim_refuses_cross_artifact_reuse(tmp_path):
    directory = tmp_path / "vectors"
    store.claim_dir(directory, "artifact-a", {"extra": 1})
    store.claim_dir(directory, "artifact-a")
    assert store.read_claim(directory)["extra"] == 1
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        store.claim_dir(directory, "artifact-b")
    # An ``extra`` is part of the claim, not decoration: a sweep keyed by its
    # pairs artifact is only valid for the eval pool it steered on, and mixing
    # two pools under one key would make the layer choices incomparable.
    with pytest.raises(RuntimeError, match="was built with extra=1"):
        store.claim_dir(directory, "artifact-a", {"extra": 2})
    assert store.read_claim(directory)["extra"] == 1


def test_store_paths_are_isolated_by_data_config(cluster, tmp_path):
    """Was a strict xfail in the old known-gaps suite: predictor outputs keyed by
    (predictor, data_method, model) alone, so two data configs sharing those
    three overwrote each other's matrix + values sidecars. The nested artifact-ID amendment
    nests outputs under the source artifact ID, which is what separates them."""
    subset = _artifact(tmp_path)
    subset.cfg["values"] = VALUES[:2]
    full = _artifact(tmp_path)
    assert subset.artifact_id != full.artifact_id

    dirs = [
        store.similarity_dir(
            cluster, "weight_steer", a.method, "org/base", a.artifact_id
        )
        for a in (subset, full)
    ]
    assert dirs[0] != dirs[1]
    # The manifest guard is belt-and-braces behind the path level, not the only
    # thing standing between two data configs.
    for directory, artifact in zip(dirs, (subset, full)):
        store.claim_dir(directory, artifact.artifact_id)
    assert store.read_claim(dirs[0])["source_artifact_id"] == subset.artifact_id


def test_similarity_roundtrip_and_provenance(tmp_path):
    matrix = np.array([[1.0, np.nan], [np.nan, 1.0]])
    path = store.save_similarity(
        tmp_path, "cos", matrix, ["a", "b"], {"data_artifact": "id"}
    )
    loaded, values = store.load_similarity(path)
    assert values == ["a", "b"]
    assert np.allclose(loaded, matrix, equal_nan=True)
    provenance = yaml.safe_load((tmp_path / "cos_provenance.yaml").read_text())
    assert provenance["matrix"] == "cos" and provenance["data_artifact"] == "id"


def test_similarity_refuses_rewrite_under_a_different_predictor_config(tmp_path):
    """The {artifact_id} path level separates data configs; identity separates
    predictor configs sharing one matrix name (weight_steer's SIM_NAME)."""
    matrix = np.eye(2)
    store.save_similarity(
        tmp_path, "ws_cos", matrix, ["a", "b"], identity={"recipe": "lora-sft.yml"}
    )
    # An identical rerun still overwrites: resume must stay idempotent.
    store.save_similarity(
        tmp_path, "ws_cos", matrix, ["a", "b"], identity={"recipe": "lora-sft.yml"}
    )
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        store.save_similarity(
            tmp_path, "ws_cos", matrix, ["a", "b"],
            identity={"recipe": "full-sft.yml"},
        )
    # A distinct name is the way out, and legacy matrices without a recorded
    # identity stay writable.
    store.save_similarity(
        tmp_path, "ws_cos_full", matrix, ["a", "b"], identity={"recipe": "full-sft.yml"}
    )
    store.save_similarity(tmp_path, "legacy_cos", matrix, ["a", "b"])
    store.save_similarity(
        tmp_path, "legacy_cos", matrix, ["a", "b"], identity={"recipe": "lora-sft.yml"}
    )


def test_load_vectors_slices_layers_and_skips_missing(tmp_path):
    torch.save(torch.arange(6).reshape(2, 3), tmp_path / "a_kind.pt")
    torch.save(torch.tensor([4.0, 5.0]), tmp_path / "b_kind.pt")
    with pytest.raises(ValueError, match="layer is required"):
        store.load_vectors(tmp_path, ["a"], "kind")
    loaded = store.load_vectors(tmp_path, ["a", "b", "c"], "kind", layer=1)
    assert loaded["a"].tolist() == [3.0, 4.0, 5.0]
    assert loaded["b"].tolist() == [4.0, 5.0]
    assert "c" not in loaded


def test_persona_best_layer_and_explicit_layer(cluster, tmp_path):
    sweep = tmp_path / "sweep.csv"
    pd.DataFrame({"layer": [0, 1, 2], "mean_trait_score": [0.1, 0.9, 0.4]}).to_csv(sweep, index=False)
    assert persona.best_layer(sweep) == 1
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES}
    assert persona.resolve_layer({**cfg, "layer": 7}, cluster, artifact) == 7
    # No sweep and no --layer: refuse rather than fall back to anything that
    # would look at the target matrix.
    with pytest.raises(ValueError, match="No layer available"):
        persona.resolve_layer(cfg, cluster, artifact)


def test_persona_resolves_layer_from_its_own_sweep(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES}
    sweep_dir = store.sweeps_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    sweep_dir.mkdir(parents=True)
    pd.DataFrame({"layer": [0, 1, 2], "mean_trait_score": [0.1, 0.4, 0.9]}).to_csv(
        sweep_dir / persona.AGGREGATED_SWEEP, index=False
    )
    assert persona.resolve_layer(cfg, cluster, artifact) == 2
    assert not persona._needs_sweep(cfg, cluster, artifact)


def test_persona_aggregate_sweep_matches_fork_stage4(tmp_path):
    # mean trait score / coherence per layer, across values (the fork's stage-4
    # groupby). Explicit value list, so an existing aggregate in the directory
    # cannot be folded back in as a pseudo-value.
    for value, scores in (("alpha", [10.0, 90.0]), ("beta", [30.0, 50.0])):
        pd.DataFrame({
            "layer": [0, 1], "trait_mean": scores, "coherence_mean": [80.0, 60.0],
        }).to_csv(tmp_path / f"{value}_layer_sweep.csv", index=False)
    agg = pd.read_csv(persona.aggregate_sweep(tmp_path, ["alpha", "beta"]))
    assert agg["mean_trait_score"].tolist() == [20.0, 70.0]
    assert agg["n_traits"].tolist() == [2, 2]
    assert persona.best_layer(tmp_path / persona.AGGREGATED_SWEEP) == 1


def test_persona_requested_pooling_controls_resume(cluster, tmp_path):
    # Every "is this done?" check used to key on response_avg_diff while run()
    # loaded cfg["pooling"], so asking for a pooling that is not on disk skipped
    # extraction and then silently wrote an all-NaN similarity matrix
    # (load_vectors finds nothing, cosine_grid pads it, exit 0).
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": ["alpha"], "pooling": "prompt_last_diff"}
    vec_dir = store.vectors_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    vec_dir.mkdir(parents=True)
    torch.save(torch.ones(2, 3), vec_dir / "alpha_response_avg_diff.pt")

    assert persona._needs_extraction(cfg, cluster, artifact)
    stage = persona._extraction_stage(cfg, cluster, artifact)
    assert stage.tasks[0].done == vec_dir / "alpha_prompt_last_diff.pt"

    torch.save(torch.ones(2, 3), vec_dir / "alpha_prompt_last_diff.pt")
    assert not persona._needs_extraction(cfg, cluster, artifact)


def test_persona_torn_extraction_write_is_not_complete(cluster, tmp_path):
    # generate_vec.py saves the three poolings in three non-atomic torch.save
    # calls, response_avg_diff second. A crash before the third leaves exactly
    # this state, which the old response_avg_diff completion check called done —
    # permanently, since resume re-checks the same proxy.
    artifact = _artifact(tmp_path)
    vec_dir = store.vectors_dir(
        cluster, "persona", artifact.method, "org/base", artifact.artifact_id
    )
    vec_dir.mkdir(parents=True)
    for kind in ("prompt_avg_diff", "response_avg_diff"):
        torch.save(torch.ones(2, 3), vec_dir / f"alpha_{kind}.pt")

    default = {"model": "org/base", "values": ["alpha"]}
    assert not persona._needs_extraction(default, cluster, artifact)
    torn = {**default, "pooling": "prompt_last_diff"}
    assert persona._needs_extraction(torn, cluster, artifact)


def test_persona_fork_vectors_lacking_requested_pooling_are_reextracted(
    cluster, tmp_path, monkeypatch
):
    # generate_vec.py's sentence-transformer branch writes no prompt_last_diff at
    # all. Copying whatever the fork has must not count as having what was asked
    # for, or the copy "succeeds" into a directory missing the requested pooling.
    artifact = _artifact(tmp_path, method="default_llm")
    artifact.cfg["experiment"] = "exp"
    fork = tmp_path / "fork-vectors"
    fork.mkdir()
    for kind in ("prompt_avg_diff", "response_avg_diff"):
        torch.save(torch.ones(2, 3), fork / f"alpha_{kind}.pt")
    monkeypatch.setattr(persona, "_fork_vec_dir", lambda cfg, artifact: fork)

    cfg = {"model": "org/base", "values": ["alpha"], "pooling": "prompt_last_diff"}
    assert persona._needs_extraction(cfg, cluster, artifact)
    assert persona._copy_fork_vectors(cfg, cluster, artifact) == ["alpha"]


def test_persona_extraction_stage_uses_compat_threshold_and_gpu(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:2], "gpus": 2}
    stage = persona._extraction_stage(cfg, cluster, artifact)
    assert stage.env == "persona" and stage.gpus == 2 and len(stage.tasks) == 2
    for value, task in zip(VALUES, stage.tasks):
        assert f"--trait {value}" in task.command
        assert "--threshold 0" in task.command
        assert str(D.fork_compat_dir(artifact, cfg["model"])) in task.command


def test_persona_needs_slurm_accounts_for_fork_native_vectors(
    cluster, tmp_path, monkeypatch
):
    artifact = _artifact(tmp_path, method="default_llm")
    artifact.cfg["experiment"] = "exp"
    cfg = {"model": "org/base", "values": VALUES[:2]}
    fork = tmp_path / "fork-vectors"
    fork.mkdir()
    monkeypatch.setattr(persona, "_fork_vec_dir", lambda cfg, artifact: fork)
    assert persona.needs_slurm(cfg, cluster, artifact)

    # run_pipeline's vectors (stage 2) are copied into the store, so seeing them
    # drops the extraction array. The sweep (stages 3-4) is *not* borrowed from
    # the fork, even though run_pipeline left one in sweep_results/: a layer may
    # only come from a sweep in the store, keyed by this artifact ID. Importing
    # copies the fork's sweeps in with a manifest, so a fork run worth reusing is
    # already here — and a config with no sweep of its own gets a real one rather
    # than silently inheriting whatever layer the fork last landed on.
    for value in VALUES[:2]:
        torch.save(torch.ones(2, 3), fork / f"{value}_{persona.DEFAULT_POOLING}.pt")
    assert persona.needs_slurm(cfg, cluster, artifact)  # the sweep, not extraction
    with pytest.raises(ValueError, match="No layer available"):
        persona.resolve_layer(cfg, cluster, artifact)

    sweep_dir = store.sweeps_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"layer": [0, 1], "mean_trait_score": [0.2, 0.8]}).to_csv(
        sweep_dir / persona.AGGREGATED_SWEEP, index=False
    )
    assert not persona.needs_slurm(cfg, cluster, artifact)
    assert persona.resolve_layer(cfg, cluster, artifact) == 1


def test_persona_needs_slurm_for_the_layer_sweep_when_no_layer_is_known(
    cluster, tmp_path
):
    # Vectors present but no layer and no sweep: the sweep is real GPU work and
    # must surface in the cost gate rather than being silently skipped (or, as
    # it once was, replaced by a fit against the ground-truth matrix).
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:2]}
    vec_dir = store.vectors_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    vec_dir.mkdir(parents=True)
    for value in VALUES[:2]:
        torch.save(torch.ones(2, 3), vec_dir / f"{value}_{persona.DEFAULT_POOLING}.pt")

    assert not persona._needs_extraction(cfg, cluster, artifact)
    assert persona.needs_slurm(cfg, cluster, artifact)
    plan = "\n".join(persona.plan(cfg, cluster, artifact))
    assert "layer sweep" in plan and "sweep_layers.py" in plan


def test_persona_run_builds_similarity_from_existing_vectors(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {
        "model": "org/base", "values": VALUES, "layer": 1,
        "pooling": "response_avg_diff", "predictor": "persona", "schema_version": 1,
    }
    vec_dir = store.vectors_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    store.claim_dir(vec_dir, artifact.artifact_id)
    torch.save(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), vec_dir / "alpha_response_avg_diff.pt")
    torch.save(torch.tensor([[1.0, 0.0], [1.0, 1.0]]), vec_dir / "beta_response_avg_diff.pt")
    # A present but unusable zero vector exercises cosine_grid's NaN padding
    # without making the predictor correctly schedule a missing extraction.
    torch.save(torch.zeros(2, 2), vec_dir / "missing_response_avg_diff.pt")
    out = persona.run(cfg, cluster, artifact)
    matrix, values = store.load_similarity(out)
    assert values == VALUES
    assert np.allclose(matrix[:2, :2], [[1, 2 ** -0.5], [2 ** -0.5, 1]])
    assert np.isnan(matrix[2]).all()
    provenance = yaml.safe_load(out.with_name(out.stem + "_provenance.yaml").read_text())
    assert provenance["resolved_config"] == cfg


def test_weight_stage_graph_and_commands(cluster, tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:2], "predictor": "weight_steer"}
    template = tmp_path / "lora.yml"; template.write_text("template")
    accel = tmp_path / "accelerate.yml"; accel.write_text("accelerate")
    monkeypatch.setattr(weight_steer, "_template_paths", lambda cfg: (template, accel))
    monkeypatch.setattr(weight_steer, "_ws_root", lambda: tmp_path / "fork")
    stages = weight_steer.stages(cfg, cluster, artifact)
    assert [s.name for s in stages] == ["ws_prep", "ws_train", "ws_build", "ws_gather"]
    prep, train, build, gather = stages
    assert len(prep.tasks) == 1 and len(train.tasks) == 4 and len(build.tasks) == 2
    assert train.gpus == build.gpus == 1
    assert build.mem == gather.mem == "120G" and gather.cpu_partition
    assert "--persona_extract_dir" in prep.tasks[0].command
    assert "--threshold 50" in prep.tasks[0].command
    assert "--build_index 0" in build.tasks[0].command
    assert "--from_cache" in gather.tasks[0].command
    assert "--ground_truth" not in gather.tasks[0].command


def test_weight_similarity_coverage_is_subset_aware(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:2]}
    paths = weight_steer._paths(cfg, cluster, artifact)
    paths["sim"].mkdir(parents=True)
    np.save(paths["sim"] / "similarity_matrix.npy", np.eye(1))
    (paths["sim"] / "similarity_values.json").write_text(json.dumps(["alpha"]))
    assert weight_steer._sim_covers(paths, ["alpha"])
    assert not weight_steer._sim_covers(paths, ["alpha", "beta"])
    assert weight_steer.needs_slurm(cfg, cluster, artifact)


def test_weight_similarity_corrupt_metadata_is_incomplete(cluster, tmp_path):
    """A sidecar truncated by a killed gather task means rebuild, not crash."""
    artifact = _artifact(tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:1]}
    paths = weight_steer._paths(cfg, cluster, artifact)
    paths["sim"].mkdir(parents=True)
    np.save(paths["sim"] / "similarity_matrix.npy", np.eye(1))
    (paths["sim"] / "similarity_values.json").write_text("not-json")
    assert weight_steer._sim_covers(paths, ["alpha"]) is False
    assert weight_steer.needs_slurm(cfg, cluster, artifact)


def test_weight_run_registers_and_nan_pads_cached_matrix(cluster, tmp_path):
    artifact = _artifact(tmp_path)
    cfg = {
        "model": "org/base", "values": VALUES, "predictor": "weight_steer",
        "schema_version": 1,
    }
    paths = weight_steer._paths(cfg, cluster, artifact)
    for value in VALUES:
        for polarity in ("pos", "neg"):
            frame = pd.DataFrame({
                "question": ["q"], "system_prompt": [""], "answer": ["a"],
                "polarity": [polarity], "value": [value], "prompt": ["formatted"],
            })
            if polarity == "pos":
                pos = frame
            else:
                D.write_pairs(artifact, value, pos, frame)
    paths["sim"].mkdir(parents=True)
    np.save(
        paths["sim"] / "similarity_matrix.npy",
        np.array([[1.0, 0.2, np.nan], [0.2, 1.0, np.nan], [np.nan, np.nan, np.nan]]),
    )
    (paths["sim"] / "similarity_values.json").write_text(json.dumps(VALUES))
    out = weight_steer.run(cfg, cluster, artifact)
    matrix, values = store.load_similarity(out)
    assert values == VALUES and np.allclose(matrix[:2, :2], [[1, .2], [.2, 1]])
    assert np.isnan(matrix[2]).all()


def test_sentence_embedding_run_matches_direct_cosine(cluster, tmp_path, monkeypatch):
    artifact = _artifact(tmp_path, method="descriptions", artifact_type="descriptions")
    artifact.root.mkdir(parents=True)
    pd.DataFrame({
        "value": ["alpha", "beta"],
        "description": ["first", "second"],
    }).to_csv(artifact.descriptions_path, index=False)

    class FakeEncoder:
        def __init__(self, name):
            assert name == "encoder/test"

        def encode(self, texts):
            assert texts == ["first", "second"]
            return np.array([[1.0, 0.0], [1.0, 1.0]])

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEncoder)
    cfg = {
        "model": "ignored", "encoder": "encoder/test", "values": VALUES,
        "predictor": "sentence_emb", "schema_version": 1,
    }
    out = sentence_emb.run(cfg, cluster, artifact)
    matrix, values = store.load_similarity(out)
    assert values == VALUES
    assert np.allclose(matrix[:2, :2], [[1, 2 ** -0.5], [2 ** -0.5, 1]])
    assert np.isnan(matrix[2]).all()


def test_persona_sweep_renders_with_the_artifact_chat_template_family(cluster, tmp_path):
    """Steered generation must format prompts the way extraction did, so the
    sweep reads the formatter from the data artifact rather than a knob."""
    artifact = _artifact(tmp_path)
    pool = _artifact(tmp_path / "pool", method="eval_pool", artifact_type="eval_pool")
    cfg = {"model": "org/neutral-sft-qwen3", "values": VALUES[:1], "gpus": 1}
    cmd = persona._sweep_stage(cfg, cluster, artifact, pool).tasks[0].command
    assert "--chat_template_family" not in cmd
    artifact.cfg["chat_template_family"] = "tokenizer_nothink"
    cmd = persona._sweep_stage(cfg, cluster, artifact, pool).tasks[0].command
    assert "--chat_template_family tokenizer_nothink" in cmd


def _fake_base(tmp_path, params: float, vocab=150_000, hidden=4096):
    """A base-model dir whose bf16 safetensors bytes imply ``params``.
    The shard is sparse, so nothing is actually written."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"vocab_size": vocab, "hidden_size": hidden}))
    with open(base / "model.safetensors", "wb") as f:
        f.truncate(int(params * 2))
    return base


def test_weight_steer_resources_keep_the_7b_calibration(cluster, tmp_path, monkeypatch):
    # An unreadable base (an HF name, no local files) keeps the old requests,
    # and so does a base at the calibration size.
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(weight_steer, "_template_paths",
                        lambda cfg: (tmp_path / "t.yml", tmp_path / "a.yml"))
    monkeypatch.setattr(weight_steer, "_ws_root", lambda: tmp_path / "fork")
    cfg = {"model": "org/base", "values": VALUES[:2], "predictor": "weight_steer"}
    _, train, build, gather = weight_steer.stages(cfg, cluster, artifact)
    assert (train.time, train.mem) == ("01:30:00", "32G")
    assert (build.time, build.mem) == ("01:00:00", "120G")
    assert gather.mem == "120G" and gather.time == "08:00:00"
    # Single-GPU training keeps its exact accelerate command.
    assert "CUDA_VISIBLE_DEVICES=0 accelerate launch" in train.tasks[0].command


def test_weight_steer_resources_scale_with_base_size(cluster, tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(weight_steer, "_template_paths",
                        lambda cfg: (tmp_path / "t.yml", tmp_path / "a.yml"))
    monkeypatch.setattr(weight_steer, "_ws_root", lambda: tmp_path / "fork")
    base = _fake_base(tmp_path, params=30e9)
    cfg = {"model": str(base), "values": VALUES, "predictor": "weight_steer", "gpus": 2}
    _, train, build, gather = weight_steer.stages(cfg, cluster, artifact)
    scale = 30e9 / weight_steer.REF_PARAMS
    assert train.gpus == 2
    assert train.mem == f"{math.ceil(32 * scale)}G"
    assert int(build.mem[:-1]) >= 16 * 30e9 / 2**30  # ~16 bytes/param peak
    assert gather.time == weight_steer._hms(min(8.0 * scale, 47.0))
    # Gather memory is linear in the value count (vocab x hidden fp32 per value).
    gib = len(VALUES) * 150_000 * 4096 * 4 / 2**30
    assert gather.mem == f"{max(120, math.ceil(gib * 1.3) + 16)}G"
    # Model-parallel training: both GPUs visible, launched without accelerate
    # (accelerate nulls the recipe's device_map).
    cmd = train.tasks[0].command
    assert "CUDA_VISIBLE_DEVICES=0,1 python -m axolotl.cli.train" in cmd
    assert "accelerate launch" not in cmd
