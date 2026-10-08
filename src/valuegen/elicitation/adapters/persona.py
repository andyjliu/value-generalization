"""Persona-fork data methods.

- ``default_llm``: persona-instruction contrastive pairs, on-policy. The fork
  is used as given — ``run_pipeline.py`` (its own SLURM orchestrator) runs
  stage 1 (trait/question generation) and stage 2 (steered pos/neg completion
  generation + judging), and this adapter *registers* the stage-2
  ``eval_persona_extract`` CSVs as the standard artifact, applying the fork's
  own effective-row quality filter (see ``datasets.from_persona_extract``).
  run_pipeline also computes vectors and layer sweeps in the same pass — that
  is by design: the extract/sweep outputs are the persona
  *predictor's* inputs, and other consumers (weight steering) read only the
  registered completions.

(An ``off_policy`` method — fork trait JSONs + closed-source-model completions
via ModelWrapper — was dropped 2026-07-10: its extraction model entered the
artifact identity without affecting generation. Revive from git history with
``needs_model=False`` + ``off_policy_model`` in the hash if needed.)
"""

from __future__ import annotations

import json
from pathlib import Path

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D


def _fork_paths(cfg: dict, cluster: ClusterConfig) -> dict[str, Path]:
    """Where run_pipeline puts things for (model, experiment)."""
    from valuegen._external import persona_vectors_root

    root = persona_vectors_root()
    short = D.model_short(cfg["model"])
    experiment = cfg["experiment"]
    return {
        "root": root,
        "extract_csvs": root / "eval_persona_extract" / short / experiment,
        "vectors": root / "persona_vectors" / short / experiment,
        "sweeps": root / "sweep_results" / short / experiment,
    }


def _pipeline_value_set(cfg: dict, artifact: D.Artifact) -> Path:
    """Write the (possibly trimmed) value set run_pipeline will iterate.

    Trimming matters: e.g. AD runs drop ad_12 (no label data; its adversarial
    seed questions also defang in trait generation), which is exactly the
    ``values:`` subset in the data config.
    """
    from valuegen.values import load_value_set

    full = load_value_set(cfg["value_set"])
    missing = [v for v in cfg["values"] if v not in full]
    if missing:
        raise KeyError(f"values {missing} not in value set {cfg['value_set']}")
    path = artifact.root / "value_set.json"
    artifact.root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({v: full[v] for v in cfg["values"]}, indent=2)
    if not path.is_file() or path.read_text() != payload:
        path.write_text(payload)
    return path


def run_pipeline_orchestrator(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact,
                              dry_run: bool = False):
    """The fork delegate, with cluster facts injected from cluster.yaml."""
    from valuegen.slurm import RunPipelineOrchestrator

    extra = [
        "--method", str(cfg["trait_method"]),
        "--target", str(cfg["target"]),
        "--target-questions", str(cfg.get("target_questions", 40)),
        "--instruction-pairs", str(cfg.get("instruction_pairs", 5)),
        "--threshold", str(cfg["threshold"]),
        "--generate-model", str(cfg["generate_model"]),
        "--generate-provider", str(cfg.get("generate_provider", "anthropic")),
        "--generation-max-tokens", str(cfg.get("generation_max_tokens", 8000)),
        "--judge-model", str(cfg.get("judge_model", "gpt-4.1-mini-2025-04-14")),
        "--judge-scoring-protocol", str(cfg.get(
            "judge_scoring_protocol", "legacy_logprob_0_100")),
        "--layer-start", str(cfg.get("layer_start", 10)),
        "--layer-end", str(cfg.get("layer_end", 32)),
        "--gpus", str(cfg["gpus"]),
    ]
    if cfg.get("chat_template_family"):
        extra += ["--chat-template-family", str(cfg["chat_template_family"])]
    return RunPipelineOrchestrator(
        cluster=cluster,
        value_set_path=_pipeline_value_set(cfg, artifact),
        model=cfg["model"],
        experiment=cfg["experiment"],
        extra_args=extra,
        gpus=int(cfg["gpus"]),
        dry_run=dry_run,
    )


def default_llm_plan(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    """Human-readable job plan for the cost gate."""
    n = len(cfg["values"])
    layer_start = int(cfg.get("layer_start", 10))
    layer_end = int(cfg.get("layer_end", 32))
    layer_plan = (
        f"fixed-layer evaluation (layer {layer_start})"
        if layer_start == layer_end
        else f"layer sweep ({layer_start}..{layer_end})"
    )
    return [
        f"persona_vectors run_pipeline.py (experiment={cfg['experiment']}, "
        f"model={cfg['model']}):",
        f"  stage 1  generate traits    1 cpu job (API: {cfg['generate_model']})",
        f"  stage 2  extract            {n}-task GPU array ({cfg['gpus']} gpu/task)",
        f"  stage 3  {layer_plan:<18} {n}-task GPU array",
        "  stage 4  aggregate          1 cpu job",
    ]


def build_default_llm(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    only_values: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Run (or resume) the fork pipeline, then register the completions."""
    paths = _fork_paths(cfg, cluster)
    pending_csvs = [
        v for v in cfg["values"]
        if not (paths["extract_csvs"] / f"{v}_pos_instruct.csv").is_file()
        or not (paths["extract_csvs"] / f"{v}_neg_instruct.csv").is_file()
    ]
    if pending_csvs or dry_run:
        orch = run_pipeline_orchestrator(cfg, cluster, artifact, dry_run=dry_run)
        ok = orch.run()
        if dry_run:
            return
        if not ok:
            raise RuntimeError(
                "persona_vectors run_pipeline did not complete; rerun "
                "`valuegen data build` to resume (it is idempotent)"
            )
    for value in only_values or artifact.pending_values():
        if artifact.value_done(value):
            continue
        pos_df, neg_df = D.from_persona_extract(
            paths["extract_csvs"] / f"{value}_pos_instruct.csv",
            paths["extract_csvs"] / f"{value}_neg_instruct.csv",
            value,
            threshold=cfg["threshold"],
        )
        D.write_pairs(artifact, value, pos_df, neg_df)
        print(f"  {value}: registered {len(pos_df)} effective pairs")
