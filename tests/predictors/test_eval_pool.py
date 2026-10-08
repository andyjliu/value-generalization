"""The held-out eval pool, the layer sweep it feeds, and its cost gate.

The eval pool is the anti-circularity mechanism for layer selection: a
persona layer is chosen by steering on questions the vectors were never
extracted from. Every seam here can invert silently — the halves can swap, the
sweep can be pointed at the extraction data, the pool can be generated per
method instead of per value set, or its API spend can fire ungated — and none of
those failures raise on their own. They are what these tests pin.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from valuegen import cli
from valuegen.elicitation import datasets as D
from valuegen.elicitation import registry
from valuegen.elicitation.adapters import eval_pool
from valuegen.predictors import persona, store


VALUES = ["hard_constraint_fidelity", "accurate_overall_impressions"]
VALUE_SET = "constitution_tenets_v3"

# 8 scenarios, 25% held out: an eval half of 2 and an extract half of 6 are
# distinguishable, so an output_dir/eval_dir swap changes the counts. A 50/50
# split (the default) would make the inversion invisible.
TARGET = 8
EVAL_FRAC = 0.25


def _pool_cfg(values=VALUES, **params) -> dict:
    return registry.resolve_config(
        "eval_pool", VALUE_SET, values=list(values),
        params={"target": TARGET, "eval_frac": EVAL_FRAC, **params},
    )


def _write_trait_json(path: Path, questions: list[str]) -> None:
    path.write_text(json.dumps({
        "questions": questions,
        "eval_prompt": "Q: {question}\nA: {answer}\nScore 0-100.",
    }))


def _fake_fork(monkeypatch, tmp_path, calls: list[dict]) -> Path:
    """Stand in for ``generate_constraint_traits.py``, honoring its contract:
    ``--train_split`` of a fresh pool lands in ``--output_dir``, the remainder in
    ``--eval_dir``, and a value whose output_dir JSON already exists is skipped
    (the fork's resume)."""
    root = tmp_path / "persona_fork"
    root.mkdir(exist_ok=True)

    # The adapter refuses to spend a subprocess without a reachable generator key
    # (the fork swallows per-value errors and still exits 0). The fork is faked
    # here, so nothing is spent and any value will do — but the key must be set,
    # or the guard fires before the code under test runs. It used to be ambient:
    # conda carried the keys in `conda env config vars`. Under uv they live in the
    # repo's .env, which `cluster.activate()` sources for SBATCH and subprocesses
    # and a bare `pytest` never sees.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    import valuegen._external as ext
    monkeypatch.setattr(ext, "persona_vectors_root", lambda: root)
    real_run = subprocess.run

    def fake_run(cmd, cwd=None, **kwargs):
        # ``subprocess`` is one module: the manifest's git-SHA lookup comes
        # through here too, and only the fork's `bash -c` is ours to fake.
        if list(cmd[:2]) != ["bash", "-c"]:
            return real_run(cmd, cwd=cwd, **kwargs)
        inner = cmd[-1]
        argv = shlex.split(inner)
        flags = {
            argv[i]: argv[i + 1]
            for i in range(len(argv) - 1)
            if argv[i].startswith("--")
        }
        out_dir = Path(flags["--output_dir"])
        eval_dir = Path(flags["--eval_dir"])
        value_set = json.loads(Path(flags["--value_set"]).read_text())
        train_split = float(flags["--train_split"])
        target = int(flags["--target"])
        calls.append({"inner": inner, "flags": flags, "cwd": cwd,
                      "value_set": value_set,
                      "extract_present": sorted(p.name for p in out_dir.glob("*.json"))})

        out_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        for value in value_set:
            if (out_dir / f"{value}.json").is_file():
                continue
            pool = [f"{value}-q{i}" for i in range(target)]
            n_train = round(train_split * target)
            _write_trait_json(out_dir / f"{value}.json", pool[:n_train])
            _write_trait_json(eval_dir / f"{value}.json", pool[n_train:])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(eval_pool.subprocess, "run", fake_run)
    return root


def _questions(path: Path) -> list[str]:
    return json.loads(path.read_text())["questions"]


# ── the adapter: which half is registered, and what the fork is told ──────────


def test_eval_pool_registers_the_eval_half_disjoint_from_the_extraction_half(
    cluster, tmp_path, monkeypatch
):
    # The seam: --train_split is the *extraction* fraction and --output_dir is
    # the extraction half, while the artifact itself is --eval_dir. Swap the two
    # directories (or read eval_frac as the train fraction) and the sweep steers
    # on the questions the vectors were extracted from — layer selection becomes
    # circular and nothing anywhere fails.
    calls = []
    root = _fake_fork(monkeypatch, tmp_path, calls)
    cfg = _pool_cfg()
    artifact = registry.build(cfg, cluster)

    flags = calls[0]["flags"]
    assert calls[0]["cwd"] == root
    assert flags["--method"] == "conflictscope"
    assert flags["--output_dir"] == str(artifact.root / "extract_half")
    assert flags["--eval_dir"] == str(artifact.root)
    assert flags["--train_split"] == str(1.0 - EVAL_FRAC)
    assert flags["--target"] == str(TARGET)
    assert flags["--model"] == cfg["generate_model"]
    assert "--filter_model" not in flags  # unset by default, so not passed
    assert f"{cluster.venv('persona')}/bin/activate" in calls[0]["inner"]

    assert artifact.is_complete()
    for value in VALUES:
        held_out = _questions(artifact.eval_path(value))
        extracted = _questions(artifact.root / "extract_half" / f"{value}.json")
        # eval_frac of the pool is held out — not 1 - eval_frac.
        assert len(held_out) == 2 and len(extracted) == 6
        assert not set(held_out) & set(extracted)
        # And it is readable as fork trait data (questions + judge rubric), so
        # sweep_layers.py --trait_data_dir eats the artifact with no compat view.
        assert D.load_eval_pool(artifact, value)["questions"] == held_out


def test_eval_pool_passes_the_filter_model_only_when_configured(
    cluster, tmp_path, monkeypatch
):
    calls = []
    _fake_fork(monkeypatch, tmp_path, calls)
    registry.build(_pool_cfg(filter_model="gpt-4.1-mini"), cluster)
    assert calls[0]["flags"]["--filter_model"] == "gpt-4.1-mini"


def test_eval_pool_hands_the_fork_the_full_value_set_on_resume(
    cluster, tmp_path, monkeypatch
):
    # Conflictscope-mode generation pits each value against opponents drawn from
    # the set it is handed, so a resume that passed only the *pending* values
    # would generate different scenarios than a fresh run — the pool would stop
    # being one coherent artifact. The fork's own resume (skip values with an
    # extract JSON) is what keeps the finished values from being regenerated.
    calls = []
    _fake_fork(monkeypatch, tmp_path, calls)
    cfg = _pool_cfg()
    artifact = D.resolve_artifact(cluster, cfg)
    done, pending = VALUES

    artifact.root.mkdir(parents=True, exist_ok=True)
    (artifact.root / "extract_half").mkdir(exist_ok=True)
    _write_trait_json(artifact.eval_path(done), ["kept-eval"])
    _write_trait_json(artifact.root / "extract_half" / f"{done}.json", ["kept-extract"])
    assert artifact.pending_values() == [pending]

    registry.build(cfg, cluster)
    assert calls[0]["value_set"] and list(calls[0]["value_set"]) == VALUES
    # The finished value is untouched; only the pending one is generated.
    assert _questions(artifact.eval_path(done)) == ["kept-eval"]
    assert len(_questions(artifact.eval_path(pending))) == 2


def test_eval_pool_value_set_json_rejects_values_outside_the_set(cluster):
    cfg = _pool_cfg()
    cfg["values"] = [*VALUES, "not_a_constraint"]
    artifact = D.resolve_artifact(cluster, cfg)
    with pytest.raises(KeyError, match="not in value set"):
        eval_pool._value_set_json(cfg, artifact)


def test_eval_pool_clears_a_torn_extract_half_for_a_pending_value(
    cluster, tmp_path, monkeypatch
):
    # A run killed between writing the extract half and the eval half leaves an
    # extract JSON with no eval JSON. The fork resumes by skipping values whose
    # extract JSON exists, so without the unlink the value is skipped forever and
    # its eval half never appears — the build just keeps failing.
    calls = []
    _fake_fork(monkeypatch, tmp_path, calls)
    cfg = _pool_cfg()
    artifact = D.resolve_artifact(cluster, cfg)
    torn = artifact.root / "extract_half" / f"{VALUES[0]}.json"
    torn.parent.mkdir(parents=True, exist_ok=True)
    _write_trait_json(torn, ["stale-extract"])

    artifact = registry.build(cfg, cluster)
    # The fork saw no extract JSON for the torn value, so it regenerated it.
    assert calls[0]["extract_present"] == []
    assert artifact.is_complete()
    assert _questions(torn) != ["stale-extract"]
    assert not set(_questions(torn)) & set(_questions(artifact.eval_path(VALUES[0])))


def test_eval_pool_build_reports_a_fork_that_produced_nothing(
    cluster, tmp_path, monkeypatch
):
    # A fork that exits 0 without writing anything (its filter passed nothing, a
    # generation silently truncated): say so, don't register an empty artifact
    # that a sweep would later steer on.
    #
    # This one hand-rolls its fork stub rather than using ``_fake_fork``, so it
    # needs the generator key for the same reason (see there): without it the
    # missing-key guard raises first and this never reaches the path it asserts.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    import valuegen._external as ext
    monkeypatch.setattr(ext, "persona_vectors_root", lambda: tmp_path)
    real_run = subprocess.run
    monkeypatch.setattr(eval_pool.subprocess, "run", lambda cmd, cwd=None, **kw: (
        subprocess.CompletedProcess(cmd, 0) if list(cmd[:2]) == ["bash", "-c"]
        else real_run(cmd, cwd=cwd, **kw)
    ))
    cfg = _pool_cfg()
    artifact = D.resolve_artifact(cluster, cfg)
    with pytest.raises(RuntimeError, match="produced nothing"):
        eval_pool.build(cfg, cluster, artifact)


def test_eval_pool_is_keyed_by_value_set_not_by_data_method(cluster, tmp_path):
    # The comparability claim: every data method's vectors are swept on the same
    # questions, so their layer choices can be compared. That only holds if the
    # pool a predictor resolves depends on the value set alone.
    def _artifact(method: str) -> D.Artifact:
        return D.Artifact(tmp_path / method, {
            "method": method, "artifact": "pairs", "schema_version": 1,
            "value_set": VALUE_SET, "values": list(VALUES), "legacy": False,
            "model": "org/policy",
        })

    cfg = {"model": "org/base", "values": VALUES}
    pools = [
        persona.eval_pool_artifact(cfg, cluster, _artifact(method))
        for method in ("conflictscope_action_prompt", "default_llm")
    ]
    assert pools[0].root == pools[1].root
    assert pools[0].artifact_id == pools[1].artifact_id
    assert pools[0].cfg["model"] is None  # no model in the identity


# ── the sweep stage that consumes the pool ───────────────────────────────────


def _pairs_artifact(cluster, tmp_path, values=VALUES) -> D.Artifact:
    cfg = registry.resolve_config(
        "conflictscope_action_prompt", VALUE_SET, model="org/policy",
        values=list(values), params={"scenarios_dir": str(tmp_path)},
    )
    artifact = D.resolve_artifact(cluster, cfg)
    for value in values:
        frames = [
            pd.DataFrame({
                "question": ["q"], "system_prompt": [""], "answer": ["a"],
                "polarity": [polarity], "value": [value],
            })
            for polarity in ("pos", "neg")
        ]
        D.write_pairs(artifact, value, *frames)
    D.persist_data_config(artifact.root, cfg)
    D.write_manifest(artifact)
    return artifact


def test_persona_sweep_stage_steers_on_the_held_out_pool(cluster, tmp_path):
    artifact = _pairs_artifact(cluster, tmp_path)
    cfg = {"model": "org/base", "values": VALUES, "gpus": 2,
           "pooling": "prompt_avg_diff"}
    pool = persona.eval_pool_artifact(cfg, cluster, artifact)
    vec_dir = persona._vectors_dir(cfg, cluster, artifact)
    sweep_dir = persona._sweeps_dir(cfg, cluster, artifact)

    stage = persona._sweep_stage(cfg, cluster, artifact, pool)
    assert stage.env == "persona" and stage.gpus == 2
    assert [task.key for task in stage.tasks] == VALUES
    for value, task in zip(VALUES, stage.tasks):
        assert task.done == sweep_dir / f"{value}_layer_sweep.csv"
        assert f"--trait_data_dir {pool.root}" in task.command
        assert f"--vector_path {vec_dir}/{value}_prompt_avg_diff.pt" in task.command
        assert f"--trait {value}" in task.command
        # The pool is already fork-shaped trait data: no compat view here.
        assert D.FORK_COMPAT not in task.command
        # layer_end defaults to None = sweep to the model's last layer. The
        # fork's run_pipeline hardcodes 32, which silently truncates a 32B.
        assert "--layer_end" not in task.command
        assert "--layer_start 10" in task.command

    # Resume: a value with a sweep CSV drops out of the array.
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / f"{VALUES[0]}_layer_sweep.csv").write_text("layer\n0\n")
    assert persona._pending_sweeps(cfg, cluster, artifact) == [VALUES[1]]
    resumed = persona._sweep_stage(cfg, cluster, artifact, pool)
    assert [task.key for task in resumed.tasks] == [VALUES[1]]


def test_persona_sweep_stage_passes_an_explicit_layer_end(cluster, tmp_path):
    artifact = _pairs_artifact(cluster, tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:1], "layer_end": 24}
    pool = persona.eval_pool_artifact(cfg, cluster, artifact)
    stage = persona._sweep_stage(cfg, cluster, artifact, pool)
    assert "--layer_end 24" in stage.tasks[0].command


def test_persona_sweep_refuses_to_mix_two_eval_pools_under_one_key(
    cluster, tmp_path
):
    # Layer choices are only comparable if the vectors under one key were all
    # judged on the same questions, so the sweep's claim binds *both* its pairs
    # artifact and its eval pool.
    sweep_dir = tmp_path / "sweeps"
    store.claim_dir(sweep_dir, "pairs-a", extra={"eval_pool_artifact_id": "pool-a"})
    store.claim_dir(sweep_dir, "pairs-a", extra={"eval_pool_artifact_id": "pool-a"})
    with pytest.raises(RuntimeError, match="eval_pool_artifact_id"):
        store.claim_dir(sweep_dir, "pairs-a", extra={"eval_pool_artifact_id": "pool-b"})


def test_persona_run_sweep_demands_a_pool_rather_than_inventing_a_layer(
    cluster, tmp_path
):
    artifact = _pairs_artifact(cluster, tmp_path)
    cfg = {"model": "org/base", "values": VALUES}
    with pytest.raises(RuntimeError, match="No eval pool"):
        persona.run_sweep(cfg, cluster, artifact)


# ── the cost gate on the pool's API spend ────────────────────────────────────


def _predict_args(artifact: D.Artifact, **overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        method="persona", value_set=VALUE_SET, model="org/base",
        data=str(artifact.root), values=None, param=None, layer=None, pooling=None,
        encoder=None, gpus=1, build=False, dry_run=False, cluster=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_predict_gates_the_eval_pool_generation(
    cluster, tmp_path, monkeypatch, capsys
):
    # ensure_aux_data spends API credits, not SLURM time — but it is still spend,
    # and predictor-built data goes through the same gate as any other. A
    # cold `valuegen predict -m persona` must not fire a generation job silently.
    artifact = _pairs_artifact(cluster, tmp_path)
    built = []
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    monkeypatch.setattr(registry, "build", lambda *a, **k: built.append(a[0]))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit, match="cold cache"):
        cli.cmd_predict(_predict_args(artifact))
    out = capsys.readouterr().out
    assert "will spend API credits" in out
    assert "eval pool" in out
    assert built == []  # nothing was generated


def test_predict_builds_the_eval_pool_once_approved(
    cluster, tmp_path, monkeypatch
):
    artifact = _pairs_artifact(cluster, tmp_path)
    built = []
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    monkeypatch.setattr(registry, "build", lambda cfg, cluster, **k: built.append(cfg))
    monkeypatch.setattr(cli, "sbatch_available", lambda: True)
    monkeypatch.setattr(persona, "run", lambda *a, **k: tmp_path / "sim.npy")

    assert cli.cmd_predict(_predict_args(artifact, build=True)) == 0
    assert [cfg["method"] for cfg in built] == ["eval_pool"]
    assert built[0]["value_set"] == VALUE_SET and built[0]["values"] == VALUES


def test_predict_skips_the_eval_pool_when_the_layer_is_pinned(
    cluster, tmp_path, monkeypatch
):
    # An explicit --layer means no sweep, so there is nothing to steer on and no
    # generation to pay for. (A pinned layer is also the only path left for
    # legacy_conflictscope_action_pairs, which has no recoverable eval half.)
    artifact = _pairs_artifact(cluster, tmp_path)
    monkeypatch.setattr(cli, "load_cluster", lambda path=None: cluster)
    monkeypatch.setattr(
        registry, "build", lambda *a, **k: pytest.fail("built an unneeded eval pool")
    )
    monkeypatch.setattr(cli, "sbatch_available", lambda: True)
    monkeypatch.setattr(persona, "run", lambda *a, **k: tmp_path / "sim.npy")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    cfg = {"model": "org/base", "values": VALUES, "layer": 12}
    assert not persona._needs_sweep(cfg, cluster, artifact)
    assert cli.cmd_predict(_predict_args(artifact, layer=12, build=True)) == 0


# ── generator knobs (provider, trait method) and empty-pool resume ───────────


def test_eval_pool_generator_knobs_are_optional_and_identity_preserving(cluster):
    # Unset knobs never enter the resolved config, so every existing
    # anthropic/conflictscope pool keeps its artifact ID ...
    base = _pool_cfg()
    for key in ("generate_provider", "disable_thinking", "generation_max_tokens",
                "trait_method"):
        assert key not in base
    # ... while setting one is identity-affecting.
    gemini = _pool_cfg(generate_provider="gemini")
    assert (D.resolve_artifact(cluster, gemini).artifact_id
            != D.resolve_artifact(cluster, base).artifact_id)


def test_eval_pool_passes_provider_and_default_method_flags(
    cluster, tmp_path, monkeypatch
):
    calls = []
    _fake_fork(monkeypatch, tmp_path, calls)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    registry.build(_pool_cfg(
        generate_provider="gemini", trait_method="default",
        disable_thinking=True, generation_max_tokens=2000,
    ), cluster)
    flags, inner = calls[0]["flags"], calls[0]["inner"]
    assert flags["--provider"] == "gemini"
    assert flags["--method"] == "default"
    # One `target` knob sizes the pool in both modes.
    assert flags["--target_questions"] == str(TARGET)
    assert flags["--generation_max_tokens"] == "2000"
    assert "--disable_thinking" in shlex.split(inner)


def test_eval_pool_default_command_is_unchanged_apart_from_the_provider(
    cluster, tmp_path, monkeypatch
):
    calls = []
    _fake_fork(monkeypatch, tmp_path, calls)
    registry.build(_pool_cfg(), cluster)
    flags, argv = calls[0]["flags"], shlex.split(calls[0]["inner"])
    assert flags["--provider"] == "anthropic"
    assert "--target_questions" not in flags
    assert "--disable_thinking" not in argv and "--generation_max_tokens" not in flags


def test_eval_pool_requires_the_selected_providers_key(cluster, tmp_path, monkeypatch):
    _fake_fork(monkeypatch, tmp_path, [])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cluster, "repo", tmp_path)  # no .env to fall back on
    with pytest.raises(RuntimeError, match="no GEMINI_API_KEY for the gemini generator"):
        registry.build(_pool_cfg(generate_provider="gemini"), cluster)


def test_eval_pool_with_zero_questions_reads_as_pending(cluster):
    # A thinking model that burns max_tokens returns empty completions, but the
    # fork still writes the JSON scaffold; that value must stay pending.
    cfg = _pool_cfg()
    artifact = D.resolve_artifact(cluster, cfg)
    artifact.root.mkdir(parents=True, exist_ok=True)
    _write_trait_json(artifact.eval_path(VALUES[0]), [])
    _write_trait_json(artifact.eval_path(VALUES[1]), ["q"])
    assert artifact.pending_values() == [VALUES[0]]


# ── persona sweeps: pooling-scoped directory, partition override ─────────────


def test_persona_sweep_dir_is_scoped_by_non_default_pooling(cluster, tmp_path):
    # The sweep steers with the pooled vector, so a non-default pooling must
    # not find (and silently reuse) the default pooling's sweep CSVs.
    artifact = _pairs_artifact(cluster, tmp_path)
    default = persona._sweeps_dir({"model": "org/base"}, cluster, artifact)
    assert default == store.sweeps_dir(
        cluster, "persona", artifact.method, "org/base", artifact.artifact_id
    )
    last = persona._sweeps_dir(
        {"model": "org/base", "pooling": "prompt_last_diff"}, cluster, artifact
    )
    assert last == default / "pool_prompt_last_diff"


def test_persona_sweep_partition_is_overridable(cluster, tmp_path):
    artifact = _pairs_artifact(cluster, tmp_path)
    cfg = {"model": "org/base", "values": VALUES[:1]}
    pool = persona.eval_pool_artifact(cfg, cluster, artifact)
    assert persona._sweep_stage(cfg, cluster, artifact, pool).array_partition is True
    cfg["sweep_array_partition"] = False
    assert persona._sweep_stage(cfg, cluster, artifact, pool).array_partition is False
