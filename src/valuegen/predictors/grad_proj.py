"""Gradient-projection predictor: DPO-at-init activation gradients + cosine grids.

For each value, render its elicitation pairs exactly as the DPO trainer would
(``training.CHAT_TEMPLATES``), take the gradient of the DPO loss *at
initialization* with respect to the residual-stream activations at every layer,
pool over response tokens, and average over pairs — one ``[n_layers+1, hidden]``
update-direction stack per value. The grid is cosine between value stacks at
one layer: the first-order analog of ``weight_steer`` (first gradient step read
in activation space instead of trained ΔW), and Kowal et al.'s (2026)
probe-filter approximation of Concept Influence applied value-to-value.

**Why no reference model**: at θ₀ the policy equals the DPO reference, so the
sigmoid factor in the DPO gradient is exactly ½ and constant across pairs.
Up to that scale (irrelevant under cosine) the per-pair loss is
``−[log p(answer_pos | ctx) − log p(answer_neg | ctx)]``, one forward+backward,
no reference in memory. The approximation this signs up for: it predicts the
*initial* update direction, not converged multi-epoch training.

**Layer indexing matches the persona fork**: stacks are indexed like
``outputs.hidden_states`` — ``[0]`` is the embedding output, ``[k]`` the output
of decoder layer ``k``, ``num_hidden_layers + 1`` entries total — because the
layer is *inherited from the persona sweep* for the same (model, artifact).
That keeps layer selection GT-independent (the sweep steers on the held-out
eval pool and never sees the target matrix) without running a second sweep;
``resolve_layer`` raises when no persona sweep exists and no ``--layer`` is
pinned. Never pick the layer by correlating grids against the ground truth.

The math lives in :mod:`valuegen.predictors.grad_worker` (run as a SLURM array
in the core env, one GPU task per value). Like ``sentence_emb``, the method has
no fork — this repo owns it, so the fork-ownership rule doesn't apply.

Similarity: cosine over ``{value}_{KIND}.pt`` at the inherited layer,
NaN-padded for values without vectors, saved under
``data/similarity/grad_proj/{data_method}/{model_short}/{artifact_id}/``.
"""

from __future__ import annotations

from pathlib import Path

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D
from valuegen.predictors import persona, store

LOSS = "dpoinit"
POOLING = "respavg"
KIND = f"{LOSS}_{POOLING}_grad"

EXTRACT_TIME, EXTRACT_MEM = "02:00:00", "48G"

# `valuegen predict --predictor-param template=native` renders pairs with the
# tokenizer's own template (thinking off) instead of the pinned family one —
# for checkpoints trained under their native template (the neutral SFT
# models). It changes the rendered bytes, so it is part of the grid identity.
PREDICTOR_PARAMS = ("template",)
DEFAULT_TEMPLATE = "family"


def template_of(cfg: dict) -> str:
    from valuegen.predictors.grad_worker import TEMPLATE_MODES

    mode = str(cfg.get("template") or DEFAULT_TEMPLATE)
    if mode not in TEMPLATE_MODES:
        raise ValueError(f"grad_proj template {mode!r}; expected one of {TEMPLATE_MODES}")
    return mode


def _vector_kind(cfg: dict) -> str:
    """Vector filename stem: the template mode is a suffix when non-default,
    so native and family stacks for one model never share a file."""
    mode = template_of(cfg)
    return KIND if mode == DEFAULT_TEMPLATE else f"{KIND}_{mode}"


def _vectors_dir(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> Path:
    return store.vectors_dir(
        cluster, "grad_proj", artifact.method, cfg["model"], artifact.artifact_id
    )


def _pending_vectors(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    vec_dir = _vectors_dir(cfg, cluster, artifact)
    kind = _vector_kind(cfg)
    return [v for v in cfg["values"] if not (vec_dir / f"{v}_{kind}.pt").is_file()]


def needs_slurm(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> bool:
    return bool(_pending_vectors(cfg, cluster, artifact))


def resolve_layer(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> int:
    """The layer for the cosine grid. Never derived from the target matrix.

    Inherited from the *persona* sweep for the same (data, model, artifact) —
    the sweep dir lookup keys on ``model_short``, so a mismatched model fails
    to find a sweep rather than silently borrowing a foreign layer.
    """
    if cfg.get("layer") is not None:
        return int(cfg["layer"])

    swept = store.sweeps_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    ) / persona.AGGREGATED_SWEEP
    if swept.is_file():
        layer = persona.best_layer(swept)
        print(f"layer {layer} inherited from persona sweep {swept}")
        return layer

    raise ValueError(
        "No layer available for this (model, data): no persona sweep to "
        "inherit from and no --layer was given. Run `valuegen predict -m "
        "persona ... --build` first (its GT-independent sweep chooses the "
        "layer), or pass --layer explicitly. Do not pick the layer by "
        "correlating against the ground-truth matrix — that fits the target."
    )


def _extraction_stage(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact):
    from valuegen.slurm import Stage, Task

    vec_dir = _vectors_dir(cfg, cluster, artifact)
    kind, mode = _vector_kind(cfg), template_of(cfg)
    tasks = []
    for value in cfg["values"]:
        out = vec_dir / f"{value}_{kind}.pt"
        tasks.append(
            Task(
                key=value,
                command=(
                    f"python -m valuegen.predictors.grad_worker \\\n"
                    f"    --model {cfg['model']} \\\n"
                    f"    --pos {artifact.pos_path(value)} \\\n"
                    f"    --neg {artifact.neg_path(value)} \\\n"
                    f"    --template {mode} \\\n"
                    f"    --out {out}"
                ),
                done=out,
            )
        )
    return Stage(
        name="grad_extract",
        tasks=tasks,
        time=EXTRACT_TIME,
        mem=EXTRACT_MEM,
        gpus=int(cfg.get("gpus", 1)),
        env="default",
        # Per-value, idempotent, minutes long — same rationale as
        # persona_extract's opt-in.
        array_partition=True,
        # Varying-shape activation buffers fragment the caching allocator.
        extra_exports={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


def plan(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    pending = _pending_vectors(cfg, cluster, artifact)
    if not pending:
        return []
    return [
        f"grad_proj extraction (model={cfg['model']}, data={artifact.artifact_id}):",
        f"  1 GPU array, {len(pending)} grad_worker tasks "
        f"({cfg.get('gpus', 1)} gpu/task, core env): {pending}",
    ]


def run(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    dry_run: bool = False,
) -> Path | None:
    # Fail on a missing layer source *before* spending GPU time: extraction
    # computes all layers, but a run that can't finish should say so up front.
    if not dry_run:
        layer = resolve_layer(cfg, cluster, artifact)

    vec_dir = _vectors_dir(cfg, cluster, artifact)
    if not dry_run:
        store.claim_dir(vec_dir, artifact.artifact_id)

    missing = _pending_vectors(cfg, cluster, artifact)
    if missing or dry_run:
        from valuegen.slurm import Orchestrator

        orch = Orchestrator(
            name=f"predict_grad_{artifact.artifact_id}{D.job_suffix(cfg, artifact)}",
            stages=[_extraction_stage(cfg, cluster, artifact)],
            cluster=cluster,
            dry_run=dry_run,
        )
        if not orch.run():
            raise RuntimeError("grad_proj extraction incomplete; rerun to resume")
        if dry_run:
            return None

    mode = template_of(cfg)
    vectors = store.load_vectors(vec_dir, cfg["values"], _vector_kind(cfg), layer=layer)
    absent = [v for v in cfg["values"] if v not in vectors]
    if absent:
        print(f"note: no vectors for {absent}; their rows/cols are NaN")
    sim = store.cosine_grid(vectors, cfg["values"])

    sim_dir = store.similarity_dir(
        cluster, "grad_proj", artifact.method, cfg["model"], artifact.artifact_id
    )
    store.claim_dir(sim_dir, artifact.artifact_id)
    suffix = "" if mode == DEFAULT_TEMPLATE else f"_{mode}"
    name = f"gradproj_{LOSS}_{POOLING}{suffix}_L{layer}"
    out = store.save_similarity(
        sim_dir, name, sim, cfg["values"],
        # Cells depend on the vectors (loss + pooling + rendered bytes) and
        # the layer — not on how the layer was chosen (same rationale as
        # persona's identity).
        # The template enters only when non-default so the grids written
        # before it existed (all `family`) still accept an identical rerun.
        identity={"loss": LOSS, "pooling": POOLING, "layer": layer,
                  **({"template": mode} if mode != DEFAULT_TEMPLATE else {})},
        provenance={
            "predictor": "grad_proj",
            "data_artifact": artifact.artifact_id,
            "model": cfg["model"],
            "template": mode,
            "layer": layer,
            "layer_source": "pinned" if cfg.get("layer") is not None
            else "persona_sweep",
            "loss": LOSS,
            "pooling": POOLING,
            "resolved_config": dict(cfg),
        },
    )
    print(f"wrote {out}")
    return out
