"""Shared helpers for the split ground-truth pipeline.

Ground truth decomposes into two sides joined only by the intervention
manifest (see ``interventions.py``): the training side owns datasets,
checkpoints, and steer prompts under an *intervention* identity; the
evaluation side owns eval CSVs and matrices under a *GT run* identity
(``evals/``). This module keeps only what both sides share — value-set path
resolution, config-templated paths, and the GT run's directory layout::

- intervention artifacts: ``data/interventions/{method}/{value_set}/{intervention_id}/``
- scenario pools:         ``data/scenarios/{value_set}/{pool_id}/``
- GT runs:                ``data/gt/{gt_id}/{model_evals,matrices}/``
- shared base evals:      ``data/gt/base_evals/{base_eval_id}/``
"""

from __future__ import annotations

from pathlib import Path

from valuegen.config import ClusterConfig, gt_id, intervention_id


def value_set_path(value_set, cluster: ClusterConfig) -> Path:
    """Resolve a ``value_set:`` entry (bare registry name or path) to a file."""
    text = str(value_set)
    path = Path(text)
    if path.suffix == ".json":
        return path if path.is_absolute() else cluster.repo / path
    return cluster.repo / "value_sets" / f"{text}.json"


def configured_path(value: str | Path, cfg: dict, cluster: ClusterConfig) -> Path:
    """Resolve repo-relative paths; ``{gt_id}``/``{intervention_id}``/
    ``{experiment}`` templates render to the stable identities."""
    rendered = str(value).format(
        gt_id=gt_id(cfg),
        intervention_id=intervention_id(cfg),
        experiment=cfg["name"],
    )
    path = Path(rendered)
    return path if path.is_absolute() else cluster.repo / path


# ── GT run layout (the evaluation side's output home) ────────────────────────


def gt_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    return cluster.data / "gt" / gt_id(cfg)


def gt_record_path(cfg: dict, cluster: ClusterConfig) -> Path:
    return gt_dir(cfg, cluster) / "resolved_config.yaml"


def eval_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    out = cfg["evaluation"].get("output_dir")
    return configured_path(out, cfg, cluster) if out else gt_dir(cfg, cluster) / "model_evals"


def matrices_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    out = cfg["evaluation"].get("matrices", {}).get("output_dir")
    return configured_path(out, cfg, cluster) if out else gt_dir(cfg, cluster) / "matrices"


def script_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    """Eval-side hostfiles etc., keyed by the GT run."""
    return cluster.repo / "slurm_jobs" / gt_id(cfg)
