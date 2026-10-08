"""Data-method registry: names, defaults, identity, and build dispatch.

Methods (older names are kept as aliases):

===============================  =============  ========  =====================
method                           artifact       build     source of math
===============================  =============  ========  =====================
conflictscope_action_prompt      pairs          generate  conflictscope submodule
conflictscope_ranking_prompt     pairs          generate  conflictscope submodule
default_llm                      pairs          delegate  persona_vectors fork
pair_clean                       pairs          inline    cleaned derived artifact
descriptions                     descriptions   inline    value_sets/*.json
eval_pool                        eval_pool      fork_sub  persona_vectors fork
legacy_conflictscope_action_pairs pairs         inline    archived 0409 CSVs
===============================  =============  ========  =====================

Build kinds: ``inline`` runs in-process; ``generate`` runs in-process for API
policy models but needs a GPU SLURM array for local ones (the worker module is
the array task body); ``delegate`` hands the whole build to the persona fork's
``run_pipeline.py``, which submits its own SLURM arrays; ``fork_subprocess``
shells out to a fork script in its own conda env without touching SLURM (API
work only). Anything that submits SLURM is cost-gated at the CLI (job plan +
confirmation / ``--build``).

``eval_pool`` is the odd one out: it is not predictor input data but the
held-out questions the persona layer sweep steers on, keyed by value set alone
(see ``adapters/eval_pool``). It exists so that layer selection is
GT-independent — nothing may pick a layer by correlating against the target.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from valuegen.config import ClusterConfig, config_hash
from valuegen.elicitation import datasets as D

DATA_SCHEMA_VERSIONS = {
    "conflictscope_action_prompt": 1,
    "conflictscope_ranking_prompt": 1,
    "default_llm": 1,
    "pair_clean": 1,
    "descriptions": 1,
    "eval_pool": 1,
    "legacy_conflictscope_action_pairs": 1,
}

ALIASES = {
    "conflictscope_v1": "conflictscope_ranking_prompt",
    "conflictscope_v2": "conflictscope_action_prompt",
    "none": "descriptions",
}

_API_PREFIXES = ("gpt-", "claude-", "gemini-", "o1", "o3", "o4", "grok-")


def is_api_model(model: str | None) -> bool:
    return model is not None and model.lower().startswith(_API_PREFIXES)


@dataclass(frozen=True)
class DataMethod:
    name: str
    artifact: str  # "pairs" | "descriptions"
    build_kind: str  # "inline" | "generate" | "delegate"
    defaults: dict = field(default_factory=dict)
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    needs_model: bool = True
    legacy: bool = False


DATA_METHODS = {
    "conflictscope_action_prompt": DataMethod(
        name="conflictscope_action_prompt",
        artifact="pairs",
        build_kind="generate",
        defaults={
            "n_pairs": 50, "seed": 42, "temperature": 0.7, "max_tokens": 1000,
            "gpus": 1,
        },
        required=("scenarios_dir",),
    ),
    "conflictscope_ranking_prompt": DataMethod(
        name="conflictscope_ranking_prompt",
        artifact="pairs",
        build_kind="generate",
        defaults={
            "n_pairs": 50, "seed": 42, "temperature": 0.7, "max_tokens": 1000,
            "gpus": 1,
        },
        required=("scenarios_dir",),
    ),
    "default_llm": DataMethod(
        name="default_llm",
        artifact="pairs",
        build_kind="delegate",
        defaults={
            "trait_method": "conflictscope",
            "target": 40,
            "threshold": 50,
            "generate_model": "claude-sonnet-4-6",
            "gpus": 1,
            "experiment": None,  # None -> derived vg_{hash} (see resolve_config)
        },
        optional=(
            "target_questions", "instruction_pairs", "generate_provider",
            "generation_max_tokens", "judge_model", "judge_scoring_protocol",
            "layer_start", "layer_end",
            # Fork prompt formatter for the policy model: "tokenizer" (its own
            # template), "tokenizer_nothink" (own template, enable_thinking
            # off — Qwen3-template checkpoints), "olmo" (frozen OLMo-2).
            "chat_template_family",
        ),
    ),
    "pair_clean": DataMethod(
        # Generation-free cleaning of one pairs artifact (scaffold-restart
        # truncation + length cap); identity is source + protocol + caps.
        name="pair_clean",
        artifact="pairs",
        build_kind="inline",
        defaults={"clean_protocol": "scaffold_restart_v1", "max_words": 400, "min_words": 10},
        required=("source_artifact",),
        needs_model=False,
    ),
    "descriptions": DataMethod(
        name="descriptions",
        artifact="descriptions",
        build_kind="inline",
        needs_model=False,
    ),
    "eval_pool": DataMethod(
        name="eval_pool",
        artifact="eval_pool",
        build_kind="fork_subprocess",
        defaults={
            "target": 40,
            # 50/50, the fork's --train_split default and the setting behind
            # every existing sweep_results run.
            "eval_frac": 0.5,
            "generate_model": "claude-sonnet-4-6",
            "filter_model": None,
        },
        # Optional so existing (anthropic-built) pools keep their hashes; a
        # non-anthropic generator is identity-affecting and must be explicit.
        # generation_max_tokens matters for thinking models (Gemini burns the
        # fork's 8000 default on reasoning and returns empty completions).
        # trait_method: the fork's generator — unset = "conflictscope" (the
        # historical default, keeps existing hashes). "default" generates the
        # questions in one call with no scenario filter chain; on Gemini the
        # conflictscope filters passed ~5% of scenarios (2026-08-27 smoke).
        optional=("generate_provider", "disable_thinking", "generation_max_tokens",
                  "trait_method"),
        # The questions probe a *value*, not a policy: one pool serves every
        # model and every data method, so no model enters the identity.
        needs_model=False,
    ),
    "legacy_conflictscope_action_pairs": DataMethod(
        name="legacy_conflictscope_action_pairs",
        artifact="pairs",
        build_kind="inline",
        defaults={
            "n_pairs": None, "seed": 42,
            # The archived pairs are OLMo action-steered completions.
            "model": "allenai/OLMo-2-1124-7B-SFT",
        },
        required=("source_dir",),
        legacy=True,
    ),
}


def resolve_method(name: str) -> DataMethod:
    canonical = ALIASES.get(name, name)
    if canonical != name:
        print(f"note: data method {name!r} is a legacy alias for {canonical!r}")
    if canonical not in DATA_METHODS:
        raise KeyError(
            f"Unknown data method {name!r}; known: {sorted(DATA_METHODS)} "
            f"(aliases: {sorted(ALIASES)})"
        )
    return DATA_METHODS[canonical]


def resolve_config(
    method_name: str,
    value_set: str | Path,
    model: str | None = None,
    values: list[str] | None = None,
    params: dict | None = None,
) -> dict:
    """Fully resolve a data config: defaults materialized, values expanded.

    The returned dict is the identity payload — every behavioral knob is
    explicit before hashing. ``value_set`` is stored as the registry name
    (paths are location-dependent, names are not).
    """
    from valuegen.values import load_value_set, registry_name

    method = resolve_method(method_name)
    params = dict(params or {})
    accepted = set(method.defaults) | set(method.required) | set(method.optional)
    unknown = set(params) - accepted
    if unknown:
        raise KeyError(
            f"{method.name}: unknown params {sorted(unknown)}; "
            f"accepted: {sorted(accepted)}"
        )
    missing = [k for k in method.required if not params.get(k)]
    if missing:
        raise ValueError(f"{method.name}: missing required params {missing}")

    # Load through the normalized name, not the caller's argument: builders will
    # only ever see the name, so validating anything else can pass against a
    # dict nobody builds from.
    vs_name = registry_name(value_set)
    value_dict = load_value_set(vs_name)
    resolved_values = list(values) if values else list(value_dict.keys())
    unknown_values = [v for v in resolved_values if v not in value_dict]
    if unknown_values:
        raise KeyError(f"values {unknown_values} not in value set {vs_name}")

    cfg = {
        "method": method.name,
        "artifact": method.artifact,
        "schema_version": DATA_SCHEMA_VERSIONS[method.name],
        "value_set": vs_name,
        "values": resolved_values,
        "legacy": method.legacy,
        **copy.deepcopy(method.defaults),
        **params,
    }
    if method.needs_model:
        cfg.setdefault("model", model)
        if model is not None:
            cfg["model"] = model
        if not cfg.get("model"):
            raise ValueError(f"{method.name}: a policy --model is required")
    else:
        cfg["model"] = None

    # Paths in the identity payload are canonicalized (absolute) so the same
    # config hashes the same regardless of the caller's cwd.
    for key in ("scenarios_dir", "source_dir"):
        if cfg.get(key):
            cfg[key] = str(Path(cfg[key]).resolve())

    # default_llm: the fork keys its output dirs by an experiment tag. Derive
    # a config-unique tag when not pinned explicitly (pin it to register an
    # existing fork run, e.g. experiment: ad).
    if cfg.get("experiment", "") is None:
        cfg["experiment"] = f"vg_{config_hash(cfg)}"
    return cfg


def needs_slurm(cfg: dict) -> bool:
    method = DATA_METHODS[cfg["method"]]
    if method.build_kind == "delegate":
        return True
    if method.build_kind == "generate":
        return not is_api_model(cfg["model"])
    return False


def build_plan(cfg: dict, cluster: ClusterConfig) -> list[str]:
    """Human-readable job plan (shown by the CLI cost gate)."""
    artifact = D.resolve_artifact(cluster, cfg)
    pending = artifact.pending_values()
    method = DATA_METHODS[cfg["method"]]
    if method.build_kind == "delegate":
        from valuegen.elicitation.adapters import persona

        return persona.default_llm_plan(cfg, cluster, artifact)
    if method.build_kind == "fork_subprocess":
        from valuegen.elicitation.adapters import eval_pool

        return eval_pool.plan(cfg, cluster, artifact)
    if needs_slurm(cfg):
        return [
            f"pair generation ({cfg['method']}, model={cfg['model']}):",
            f"  1 GPU array, {len(pending)} tasks ({cfg.get('gpus', 1)} gpu/task), "
            f"one per value: {pending}",
        ]
    return [f"inline build ({cfg['method']}): values {pending}"]


def _builder(method_name: str):
    from valuegen.elicitation.adapters import (
        conflictscope,
        descriptions,
        eval_pool,
        pair_clean,
        persona,
    )

    return {
        "conflictscope_action_prompt": conflictscope.build_action_prompt,
        "conflictscope_ranking_prompt": conflictscope.build_ranking_prompt,
        "legacy_conflictscope_action_pairs": conflictscope.build_legacy_action_pairs,
        "default_llm": persona.build_default_llm,
        "pair_clean": pair_clean.build,
        "descriptions": descriptions.build,
        "eval_pool": eval_pool.build,
    }[method_name]


def build_inline(
    cfg: dict,
    cluster: ClusterConfig,
    only_values: list[str] | None = None,
    finalize: bool = True,
) -> D.Artifact:
    """Run a build in-process (the worker path and every inline method)."""
    artifact = D.resolve_artifact(cluster, cfg)
    D.persist_data_config(artifact.root, cfg)
    _builder(cfg["method"])(cfg, cluster, artifact, only_values=only_values)
    if finalize and not artifact.pending_values():
        D.write_manifest(artifact)
    return artifact


def generation_stage(cfg: dict, cluster: ClusterConfig):
    """SLURM stage for local-model pair generation: one array task per value."""
    from valuegen.slurm import Stage, Task

    artifact = D.resolve_artifact(cluster, cfg)
    D.persist_data_config(artifact.root, cfg)
    config_path = artifact.root / D.DATA_CONFIG
    # The worker must resolve paths from the *same* cluster config as the
    # controller, or a --cluster override writes under one data root while
    # the controller polls another.
    cluster_arg = f" --cluster {cluster.source_path}" if cluster.source_path else ""
    tasks = [
        Task(
            key=value,
            command=(
                f"python -m valuegen.elicitation.worker "
                f"--config {config_path} --value {value}{cluster_arg}"
            ),
            done=(lambda v=value: artifact.value_done(v)),
        )
        for value in artifact.values
    ]
    return Stage(
        name="generate_pairs",
        tasks=tasks,
        time="8:00:00",
        mem="48G",
        gpus=int(cfg.get("gpus", 1)),
        env="default",
    )


def build(
    cfg: dict,
    cluster: ClusterConfig,
    dry_run: bool = False,
) -> D.Artifact:
    """Build (or resume) an artifact. The CLI has already applied the cost
    gate for anything that reaches SLURM."""
    artifact = D.resolve_artifact(cluster, cfg)
    if artifact.is_complete():
        print(f"artifact {artifact.artifact_id} already built at {artifact.root}")
        return artifact

    method = DATA_METHODS[cfg["method"]]
    if method.build_kind == "delegate":
        from valuegen.elicitation.adapters import persona

        D.persist_data_config(artifact.root, cfg)
        persona.build_default_llm(cfg, cluster, artifact, dry_run=dry_run)
        if not dry_run and not artifact.pending_values():
            D.write_manifest(artifact)
        return artifact

    if method.build_kind == "fork_subprocess":
        from valuegen.elicitation.adapters import eval_pool

        D.persist_data_config(artifact.root, cfg)
        eval_pool.build(cfg, cluster, artifact, dry_run=dry_run)
        if not dry_run and not artifact.pending_values():
            D.write_manifest(artifact)
        return artifact

    if needs_slurm(cfg):
        from valuegen.slurm import Orchestrator

        orch = Orchestrator(
            name=f"data_{artifact.artifact_id}",
            stages=[generation_stage(cfg, cluster)],
            cluster=cluster,
            dry_run=dry_run,
        )
        if not orch.run():
            raise RuntimeError("pair generation did not complete; rerun to resume")
        if not dry_run and not artifact.pending_values():
            D.write_manifest(artifact)
        return artifact

    if dry_run:
        print(f"[dry-run] would build inline: {artifact.artifact_id}")
        return artifact
    return build_inline(cfg, cluster)
