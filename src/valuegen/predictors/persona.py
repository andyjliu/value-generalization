"""Persona-vector predictor: mean-activation-diff vectors + cosine grids.

The persona fork owns the math (hidden-state extraction, activation-diff
construction, steering evaluation); this module resolves inputs, schedules the
fork, and registers outputs.

Two paths to vectors:

- **Fork-native** (``--data default_llm``): the fork's ``run_pipeline.py``
  already produced ``persona_vectors/{model}/{experiment}/{trait}_*.pt`` while
  building the elicitation artifact — they are copied into the store unchanged.
- **Generic** (any pairs artifact): the artifact is rendered into the fork's
  input shape (``datasets.build_fork_compat`` — pass-through scores make the
  fork's effective-row filter a no-op, since filtering happened at artifact
  build), and ``generate_vec.py`` runs as a SLURM array in the persona env,
  one task per value.

**Layer selection is GT-independent, always.** A vector is a
``[n_layers, hidden]`` stack, and the cosine grid needs one layer. We pick it
the way the fork does: steer with the vector at each layer, generate answers to
the value's *held-out* eval questions (the ``eval_pool`` artifact), judge trait
expression 0-100, and take the layer with the highest mean trait score across
values. Nothing in that loop touches the ground-truth matrix. Selecting the
layer by correlating cosine grids against the target — which this module used
to allow — makes the resulting correlation a fitted quantity, not a prediction;
don't reintroduce it.

Similarity: cosine over ``{value}_{pooling}.pt`` at the chosen layer,
NaN-padded for values without vectors, saved under
``data/similarity/persona/{data_method}/{model_short}/{artifact_id}/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D
from valuegen.predictors import store

POOLINGS = ("response_avg_diff", "prompt_avg_diff", "prompt_last_diff")
DEFAULT_POOLING = "response_avg_diff"
VEC_KINDS = POOLINGS

AGGREGATED_SWEEP = "aggregated_layer_sweep.csv"
BEST_LAYER = "best_layer.json"

# Sweep knobs, defaulted to the fork's run_pipeline settings (the ones behind
# every existing sweep_results run). layer_end None = sweep to the model's last
# layer; run_pipeline's 32 is a 7-8B assumption that is wrong for a 32B.
SWEEP_DEFAULTS = {
    "coef": 1.5,
    "n_per_question": 3,
    "layer_start": 10,
    "layer_end": None,
    "layer_step": 1,
    "judge_model": "gpt-4.1-mini-2025-04-14",
    # The fork's judge protocol. Logprob 0-100 needs single-token numerals
    # (OpenAI tokenizers); gemini-* judges must use "0_100_text".
    "judge_eval_type": "0_100",
}

# Keys `valuegen predict --predictor-param` may set on the persona cfg: the
# sweep knobs above, plus the eval-pool resolution keys — eval_pool_artifact
# must resolve the same artifact ID as the explicit `data build` that made the
# pool, or ensure_aux_data will try to build a second, default-generator pool.
PREDICTOR_PARAMS = tuple(SWEEP_DEFAULTS) + (
    "target", "eval_frac", "generate_model", "generate_provider",
    "generation_max_tokens", "filter_model", "trait_method",
    "chat_template_family",
    # `template=<training.SCAFFOLD_TEMPLATES name>`: re-render the pairs
    # through a plain-text scaffold instead of the artifact's chat-formatted
    # prompt — for pretrained bases whose chat tags are untrained. Changes the
    # activation bytes, so it keys the vectors dir, the matrix name and the
    # identity; the fork's steer-and-judge sweep cannot generate under a
    # scaffold, so a layer must be pinned.
    "template",
)


def template_of(cfg: dict) -> str | None:
    template = cfg.get("template")
    if not template:
        return None
    from valuegen.ground_truth.training import SCAFFOLD_TEMPLATES

    if template not in SCAFFOLD_TEMPLATES:
        raise ValueError(
            f"persona template {template!r}; expected one of {sorted(SCAFFOLD_TEMPLATES)}"
        )
    return str(template)


def best_layer(sweep_csv: str | Path) -> int:
    """GT-independent layer choice: argmax mean trait score in an
    ``aggregated_layer_sweep.csv`` (the fork's aggregation)."""
    df = pd.read_csv(sweep_csv)
    return int(df.loc[df["mean_trait_score"].idxmax(), "layer"])


def _fork_vec_dir(cfg: dict, artifact: D.Artifact) -> Path | None:
    """Where run_pipeline left vectors, if this artifact is fork-native."""
    if artifact.method != "default_llm" or template_of(cfg) is not None:
        # Fork-native vectors were extracted under the chat template, never a
        # scaffold — nothing to borrow.
        return None
    from valuegen._external import persona_vectors_root

    root = persona_vectors_root()
    experiment = artifact.cfg.get("experiment")
    if not experiment:
        return None
    return root / "persona_vectors" / D.model_short(cfg["model"]) / experiment


def _vectors_dir(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> Path:
    base = store.vectors_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    template = template_of(cfg)
    return base / template if template else base


def pooling_of(cfg: dict) -> str:
    return cfg.get("pooling") or DEFAULT_POOLING


def _pending_vectors(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    """Values whose *requested* pooling is not on disk.

    Keyed on the requested pooling, not on ``DEFAULT_POOLING``: one
    ``generate_vec.py`` run writes all three poolings, so response_avg_diff is
    normally a fine proxy for "extracted" — but only normally. Its three
    ``torch.save`` calls are not atomic, so a crash between them leaves
    response_avg_diff (saved second) without prompt_last_diff, and a proxy check
    would call that complete forever. Checking the pooling we are about to load
    is the same cost and cannot go stale.
    """
    vec_dir = _vectors_dir(cfg, cluster, artifact)
    pooling = pooling_of(cfg)
    return [
        v for v in cfg["values"]
        if not (vec_dir / f"{v}_{pooling}.pt").is_file()
    ]


def _copy_fork_vectors(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    """Copy fork-native vectors into the store; returns values still missing."""
    fork_dir = _fork_vec_dir(cfg, artifact)
    vec_dir = _vectors_dir(cfg, cluster, artifact)
    pending = _pending_vectors(cfg, cluster, artifact)
    if fork_dir is None or not fork_dir.is_dir():
        return pending
    vec_dir.mkdir(parents=True, exist_ok=True)
    pooling = pooling_of(cfg)
    still_missing = []
    for value in pending:
        for kind in VEC_KINDS:
            kind_src = fork_dir / f"{value}_{kind}.pt"
            if kind_src.is_file():
                shutil.copy2(kind_src, vec_dir / f"{value}_{kind}.pt")
        # Copying whatever the fork happens to have is not the same as having
        # what was asked for: a fork dir written by the sentence-transformer
        # branch has no prompt_last_diff at all. Re-extract rather than declare
        # victory and hand load_vectors a file that is not there.
        if not (vec_dir / f"{value}_{pooling}.pt").is_file():
            still_missing.append(value)
    return still_missing


def _extraction_stage(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact):
    """SLURM array: fork ``generate_vec.py`` per value, persona env.

    ``--threshold 0`` is deliberate: the compat CSVs carry pass-through
    scores, so the fork's effective-row mask keeps every (already-filtered)
    row.
    """
    from valuegen._external import persona_vectors_root
    from valuegen.slurm import Stage, Task

    root = persona_vectors_root()
    compat = D.fork_compat_dir(artifact, cfg["model"], template_of(cfg))
    vec_dir = _vectors_dir(cfg, cluster, artifact)
    pooling = pooling_of(cfg)
    tasks = []
    for value in cfg["values"]:
        tasks.append(
            Task(
                key=value,
                command=(
                    f"cd {root}\n"
                    f"python generate_vec.py \\\n"
                    f"    --model_name {cfg['model']} \\\n"
                    f"    --pos_path {compat}/{value}_pos_instruct.csv \\\n"
                    f"    --neg_path {compat}/{value}_neg_instruct.csv \\\n"
                    f"    --trait {value} \\\n"
                    f"    --save_dir {vec_dir}/ \\\n"
                    f"    --threshold 0"
                ),
                done=vec_dir / f"{value}_{pooling}.pt",
            )
        )
    return Stage(
        name="persona_extract",
        tasks=tasks,
        time="6:00:00",
        mem="48G",
        gpus=int(cfg.get("gpus", 1)),
        env="persona",
        # One value's vectors per task, each writing its own .pt files: a
        # preemption costs a single re-extraction, so this belongs in the
        # array pool rather than competing with training for the default one.
        array_partition=True,
    )


def _needs_extraction(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> bool:
    pending = _pending_vectors(cfg, cluster, artifact)
    if not pending:
        return False
    fork_dir = _fork_vec_dir(cfg, artifact)
    if fork_dir is not None and fork_dir.is_dir():
        # Borrowable only if the fork has the pooling we actually want.
        pooling = pooling_of(cfg)
        pending = [
            v for v in pending
            if not (fork_dir / f"{v}_{pooling}.pt").is_file()
        ]
    return bool(pending)


def _needs_sweep(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> bool:
    """A sweep is needed only when we must choose a layer and have no sweep to
    read it from — an explicit ``--layer`` or an existing sweep short-circuits.

    "An existing sweep" means one in the store, keyed by artifact ID. The fork's
    own ``sweep_results/`` does not count: importing copies those into the store
    with a manifest naming the eval questions they used, so a fork run that
    matters is already here. Reading the fork's copy directly would let a config
    with no sweep of its own silently borrow a layer from whatever the fork last
    ran on this model — including a config built precisely to test that a fresh
    sweep happens.
    """
    if cfg.get("layer") is not None:
        return False
    if template_of(cfg) is not None:
        # The fork's sweep generates under the model's chat template, not a
        # scaffold; resolve_layer raises so the layer is pinned explicitly.
        return False
    if (_sweeps_dir(cfg, cluster, artifact) / AGGREGATED_SWEEP).is_file():
        return False
    return True


def needs_slurm(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> bool:
    return _needs_extraction(cfg, cluster, artifact) or _needs_sweep(cfg, cluster, artifact)


def plan(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    lines = []
    if _needs_extraction(cfg, cluster, artifact):
        pending = _pending_vectors(cfg, cluster, artifact)
        lines += [
            f"persona extraction (model={cfg['model']}, data={artifact.artifact_id}):",
            f"  1 GPU array, {len(pending)} generate_vec.py tasks "
            f"({cfg.get('gpus', 1)} gpu/task, persona env): {pending}",
        ]
    if _needs_sweep(cfg, cluster, artifact):
        lines += sweep_plan(cfg, cluster, artifact)
    return lines


# ── Layer sweep (steer on held-out eval questions, judge trait expression) ───


def _sweeps_dir(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> Path:
    base = store.sweeps_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    # The sweep steers with the *pooled* vector, so the chosen layer is
    # pooling-specific. The store path is not pooling-keyed for historical
    # reasons (every existing sweep used DEFAULT_POOLING), so nest any
    # non-default pooling under its own dir: without this a prompt_last_diff
    # sweep would find the response_avg_diff `{value}_layer_sweep.csv` files,
    # skip the sweep as "done", and re-aggregate the wrong pooling's data into
    # a best_layer.json stamped with the new pooling. grad_proj inherits the
    # canonical layer via store.sweeps_dir directly, so it is unaffected.
    pooling = pooling_of(cfg)
    if pooling != DEFAULT_POOLING:
        base = base / f"pool_{pooling}"
    return base


def _sweep_cfg(cfg: dict) -> dict:
    sweep = {**SWEEP_DEFAULTS, **{k: cfg[k] for k in SWEEP_DEFAULTS if k in cfg}}
    if (str(sweep["judge_model"]).startswith("gemini-")
            and sweep["judge_eval_type"] not in ("0_100_text", "binary_text")):
        # Fail here, not mid-array: the fork's OpenAiJudge raises the same
        # constraint only after the GPU task has loaded the model.
        raise ValueError(
            f"gemini judge {sweep['judge_model']!r} cannot use eval type "
            f"{sweep['judge_eval_type']!r} (no single-token 0-100 logprobs); "
            "pass judge_eval_type=0_100_text"
        )
    return sweep


def eval_pool_artifact(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> D.Artifact:
    """The held-out questions this artifact's vectors are swept on.

    Keyed by value set only, so every data method's vectors for a given value
    set are judged on identical questions and their layer choices stay
    comparable. Built by ``valuegen data build --method eval_pool``.
    """
    from valuegen.elicitation import registry as R

    params = {
        k: cfg[k]
        for k in ("target", "eval_frac", "generate_model", "generate_provider",
                  "generation_max_tokens", "filter_model", "trait_method")
        if k in cfg
    }
    pool_cfg = R.resolve_config(
        "eval_pool",
        artifact.cfg["value_set"],
        values=list(cfg["values"]),
        params=params,
    )
    return D.resolve_artifact(cluster, pool_cfg)


def ensure_aux_data(cfg, cluster, artifact, gate, dry_run: bool = False) -> None:
    """Build the eval pool the sweep needs, if a sweep is going to happen.

    Called by the CLI (the ``predict`` auto-build path). ``gate(plan, reason)``
    is the cost gate — generation is API spend, not SLURM, but it is spend.
    """
    from valuegen.elicitation import registry as R

    if not _needs_sweep(cfg, cluster, artifact):
        return
    pool = eval_pool_artifact(cfg, cluster, artifact)
    if pool.is_complete():
        return
    gate(
        R.build_plan(pool.cfg, cluster),
        f"eval pool {pool.artifact_id} (held-out questions for the layer sweep)",
    )
    R.build(pool.cfg, cluster, dry_run=dry_run)


def _pending_sweeps(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    sweep_dir = _sweeps_dir(cfg, cluster, artifact)
    return [
        v for v in cfg["values"]
        if not (sweep_dir / f"{v}_layer_sweep.csv").is_file()
    ]


def _sweep_stage(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact, pool: D.Artifact):
    """SLURM array: fork ``sweep_layers.py`` per value, persona env.

    ``--trait_data_dir`` points straight at the eval-pool artifact: its
    ``{value}.json`` files are already in the fork's trait-data shape, so there
    is no compat view to render here.
    """
    from valuegen._external import persona_vectors_root
    from valuegen.slurm import Stage, Task

    root = persona_vectors_root()
    sweep = _sweep_cfg(cfg)
    vec_dir = _vectors_dir(cfg, cluster, artifact)
    sweep_dir = _sweeps_dir(cfg, cluster, artifact)
    pooling = pooling_of(cfg)

    tasks = []
    for value in _pending_sweeps(cfg, cluster, artifact):
        out = sweep_dir / f"{value}_layer_sweep.csv"
        command = (
            f"cd {root}\n"
            f"python sweep_layers.py \\\n"
            f"    --model {cfg['model']} \\\n"
            f"    --vector_path {vec_dir}/{value}_{pooling}.pt \\\n"
            f"    --trait {value} \\\n"
            f"    --output_path {out}.partial \\\n"
            f"    --trait_data_dir {pool.root} \\\n"
            f"    --coef {sweep['coef']} \\\n"
            f"    --n_per_question {sweep['n_per_question']} \\\n"
            f"    --judge_model {sweep['judge_model']} \\\n"
            f"    --judge_eval_type {sweep['judge_eval_type']} \\\n"
            f"    --layer_start {sweep['layer_start']} \\\n"
            f"    --layer_step {sweep['layer_step']}"
        )
        if sweep["layer_end"] is not None:
            command += f" \\\n    --layer_end {sweep['layer_end']}"
        # Steered generation must render prompts exactly as extraction did;
        # the formatter is a data-artifact knob, so read it from there.
        # A predictor-param override exists for activation-only runs that read
        # one model's pairs through another model (e.g. Qwen3-30B-A3B on the
        # neutral-qwen3-8b artifact, which needs tokenizer_nothink so the
        # steered generations do not think).
        family = cfg.get("chat_template_family") or artifact.cfg.get("chat_template_family")
        if family:
            command += f" \\\n    --chat_template_family {family}"
        # The fork appends one row per layer as it goes; a preempted task
        # would leave a partial CSV that `done=out` mistakes for a finished
        # sweep. Write to .partial and publish atomically on success.
        command = command.replace(
            f"cd {root}\n", f"cd {root}\nrm -f {out}.partial\n", 1
        )
        command += f" \\\n    && mv {out}.partial {out}"
        tasks.append(Task(key=value, command=command, done=out))

    return Stage(
        name="persona_sweep",
        tasks=tasks,
        time="12:00:00",
        mem="48G",
        gpus=int(cfg.get("gpus", 1)),
        env="persona",
        # Short per-value tasks (minutes per layer) and each CSV row is
        # appended as it completes, so a preempted task loses little; the
        # preempt pool avoids general's throttle. Overridable to general
        # (sweep_array_partition=False) when the caller wants no preemption
        # and is willing to take general's 10-job cap — a scheduling knob
        # only, never in any artifact identity.
        array_partition=bool(cfg.get("sweep_array_partition", True)),
    )


def aggregate_sweep(sweep_dir: Path, values: list[str]) -> Path:
    """Per-value sweeps -> ``aggregated_layer_sweep.csv`` (the fork's stage-4
    aggregation: mean trait score and coherence per layer, across values)."""
    frames = []
    for value in values:
        path = sweep_dir / f"{value}_layer_sweep.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df["trait_name"] = value
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no per-value sweeps to aggregate under {sweep_dir}")

    combined = pd.concat(frames, ignore_index=True)
    agg = combined.groupby("layer").agg(
        mean_trait_score=("trait_mean", "mean"),
        mean_coherence=("coherence_mean", "mean"),
        n_traits=("trait_name", "nunique"),
    ).reset_index()
    out = sweep_dir / AGGREGATED_SWEEP
    agg.to_csv(out, index=False)
    return out


def run_sweep(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    dry_run: bool = False,
) -> Path | None:
    """Sweep layers for this (model, artifact) and record the chosen layer."""
    import json

    pool = eval_pool_artifact(cfg, cluster, artifact)
    if not pool.is_complete():
        raise RuntimeError(
            f"No eval pool for value set {artifact.cfg['value_set']!r} "
            f"(expected {pool.artifact_id} at {pool.root}). Build it first:\n"
            f"  valuegen data build --method eval_pool "
            f"--value-set {artifact.cfg['value_set']}\n"
            "The layer sweep steers on its held-out questions; without it the "
            "only way to pick a layer is to pass --layer explicitly."
        )

    sweep_dir = _sweeps_dir(cfg, cluster, artifact)
    if not dry_run:
        store.claim_dir(
            sweep_dir,
            artifact.artifact_id,
            extra={"eval_pool_artifact_id": pool.artifact_id},
        )

    if _pending_sweeps(cfg, cluster, artifact) or dry_run:
        from valuegen.slurm import Orchestrator

        orch = Orchestrator(
            name=f"sweep_persona_{artifact.artifact_id}{D.job_suffix(cfg, artifact)}",
            stages=[_sweep_stage(cfg, cluster, artifact, pool)],
            cluster=cluster,
            dry_run=dry_run,
        )
        if not orch.run():
            raise RuntimeError("persona layer sweep incomplete; rerun to resume")
        if dry_run:
            return None

    aggregate_sweep(sweep_dir, list(cfg["values"]))
    layer = best_layer(sweep_dir / AGGREGATED_SWEEP)
    (sweep_dir / BEST_LAYER).write_text(json.dumps({
        "layer": layer,
        "selection": "argmax mean_trait_score over the aggregated sweep",
        "model": cfg["model"],
        "data_artifact": artifact.artifact_id,
        "eval_pool_artifact": pool.artifact_id,
        "pooling": pooling_of(cfg),
        **_sweep_cfg(cfg),
    }, indent=2))
    print(f"swept layer {layer} (held-out eval pool {pool.artifact_id})")
    return sweep_dir / BEST_LAYER


def sweep_plan(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    sweep = _sweep_cfg(cfg)
    pending = _pending_sweeps(cfg, cluster, artifact)
    end = sweep["layer_end"] if sweep["layer_end"] is not None else "last"
    return [
        f"persona layer sweep (model={cfg['model']}, data={artifact.artifact_id}):",
        f"  1 GPU array, {len(pending)} sweep_layers.py tasks "
        f"(layers {sweep['layer_start']}..{end} step {sweep['layer_step']}, "
        f"coef {sweep['coef']}, {sweep['n_per_question']}/question, "
        f"judge {sweep['judge_model']} [{sweep['judge_eval_type']}]): {pending}",
    ]


def resolve_layer(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> int:
    """The layer for the cosine grid. Never derived from the target matrix."""
    if cfg.get("layer") is not None:
        return int(cfg["layer"])
    if template_of(cfg) is not None:
        raise ValueError(
            f"template={cfg['template']} needs an explicit --layer: the fork's "
            "steer-and-judge sweep generates under the chat template, not a "
            "scaffold, so no GT-independent sweep exists for these vectors. "
            "Inherit the layer from the same architecture's chat-template sweep."
        )

    swept = _sweeps_dir(cfg, cluster, artifact) / AGGREGATED_SWEEP
    if swept.is_file():
        layer = best_layer(swept)
        print(f"layer {layer} from sweep {swept}")
        return layer

    raise ValueError(
        "No layer available for this (model, data): no sweep has been run and "
        "no --layer was given. Re-run `valuegen predict persona ... --build` to "
        "run the GT-independent sweep (steer on the held-out eval pool, judge "
        "trait expression), or pass --layer explicitly. Do not pick the layer "
        "by correlating against the ground-truth matrix — that fits the target."
    )


def run(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    dry_run: bool = False,
) -> Path | None:
    vec_dir = _vectors_dir(cfg, cluster, artifact)
    if not dry_run:
        store.claim_dir(vec_dir, artifact.artifact_id)

    missing = _copy_fork_vectors(cfg, cluster, artifact) if not dry_run \
        else _pending_vectors(cfg, cluster, artifact)
    if missing or dry_run:
        if not dry_run:
            D.build_fork_compat(
                artifact, cfg["model"], values=missing, template=template_of(cfg)
            )
        from valuegen.slurm import Orchestrator

        orch = Orchestrator(
            name=f"predict_persona_{artifact.artifact_id}{D.job_suffix(cfg, artifact)}",
            stages=[_extraction_stage(cfg, cluster, artifact)],
            cluster=cluster,
            dry_run=dry_run,
        )
        if not orch.run():
            raise RuntimeError("persona extraction incomplete; rerun to resume")
        if dry_run:
            return None

    # The vectors exist; now choose a layer. The sweep is a second GPU array,
    # so it is gated by the caller (needs_slurm/plan cover it) and skipped
    # entirely when --layer is pinned or a sweep already exists.
    if _needs_sweep(cfg, cluster, artifact):
        if run_sweep(cfg, cluster, artifact, dry_run=dry_run) is None and dry_run:
            return None

    layer = resolve_layer(cfg, cluster, artifact)
    pooling = pooling_of(cfg)
    template = template_of(cfg)
    vectors = store.load_vectors(vec_dir, cfg["values"], pooling, layer=layer)
    absent = [v for v in cfg["values"] if v not in vectors]
    if absent:
        print(f"note: no vectors for {absent}; their rows/cols are NaN")
    sim = store.cosine_grid(vectors, cfg["values"])

    sim_dir = store.similarity_dir(
        cluster, "persona", artifact.method, cfg["model"], artifact.artifact_id
    )
    store.claim_dir(sim_dir, artifact.artifact_id)
    suffix = f"_{template}" if template else ""
    name = f"persona_{pooling}{suffix}_L{layer}"
    out = store.save_similarity(
        sim_dir, name, sim, cfg["values"],
        # The grid depends on the vectors (rendered bytes included), the
        # pooling and the layer — not on how the layer was chosen, so sweep
        # knobs stay out (a different sweep that lands on the same layer
        # produces the same cells, and one that lands elsewhere gets a
        # different name). The template enters only when set so grids written
        # before it existed still accept an identical rerun.
        identity={"pooling": pooling, "layer": layer,
                  **({"template": template} if template else {})},
        provenance={
            "predictor": "persona",
            "data_artifact": artifact.artifact_id,
            "model": cfg["model"],
            "template": template,
            "layer": layer,
            "layer_source": "pinned" if cfg.get("layer") is not None else "sweep",
            "pooling": pooling,
            "resolved_config": dict(cfg),
        },
    )
    print(f"wrote {out}")
    return out
