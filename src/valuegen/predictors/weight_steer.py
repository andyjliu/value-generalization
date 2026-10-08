"""Weight-steering predictor: contrastive weight task vectors + cosine grids.

The weight-steering fork owns the method (pairs → SFT formatting via
``cs_prep_data.py``, the axolotl LoRA recipe in ``axolotl_configs/multimodel``,
task-vector construction ``t = ΔW_pos − ΔW_neg`` incl. the dense
``embed_tokens``/``lm_head`` diffs, and weight-space cosine via
``cs_task_vectors.py``). This module does the staging: it generates the model × value ×
polarity task maps, schedules the fork's CLIs on SLURM, and registers the
similarity artifact. **No localization ablation** (full/lora_only/
lm_head_only was cut from scope), and task-vector generation takes no GT
matrix — correlation happens in ``analysis/correlate``.

Stage layout (single base model per invocation; multi-model = one ``predict``
per model, matching how 32B is isolated everywhere else):

    prep (1 cpu task)  →  train (2N GPU tasks)  →  build (N GPU tasks, 120G)
                       →  gather (1 cpu task, mem scales with N)

Input is any *pairs* artifact through its ``fork_compat`` view —
``cs_prep_data`` reads the persona-extract shape as given, and its
threshold filter is a no-op on the pass-through scores (filtering already
happened at artifact build). This kills the old hardcoded
``persona_extract`` reach-in.

Costs to know (measured with sacct): each cached task vector is ~25–29 GB on
``finetune_root`` — delete ``weight_steering_taskvecs/{artifact_id}`` after
the gather; adapters are small, keep them. Train tasks are tiny (≤~350 pairs,
1 epoch, 5–8 min each).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D
from valuegen.predictors import store

SIM_NAME = "weight_steer_cos"
RECIPE = "multimodel/lora-sft.yml"  # the fork's LoRA recipe; part of matrix identity
# Template-free recipe for the base-model + scaffold arm: the pairs are
# pre-rendered to axolotl ``input_output`` segments (valuegen owns the byte
# rendering), so the recipe carries no chat_template. Selected automatically
# when ``scaffold`` is set; see ``_recipe`` and the ws_prep render step.
SCAFFOLD_RECIPE = "multimodel/lora-sft-inputoutput.yml"

# Accepted --predictor-param keys. ``scaffold`` re-renders the training text
# through ``training.SCAFFOLD_TEMPLATES`` (e.g. urial0) for pretrained bases
# whose chat tags are untrained — the weight-space analog of persona/grad_proj's
# ``template=`` param. ``recipe``/``recipe_variant`` override the LoRA recipe and
# its output namespace.
PREDICTOR_PARAMS = ("scaffold", "recipe", "recipe_variant", "wandb_project")

# Stage resources, calibrated on the 7-8B runs (sacct, 2026-09-04/05) and scaled
# by parameter count for larger bases (see ``_model_params`` / ``_scale``):
#   ws_train   13.8-14.6 GB GPU peak, 100-200 s per LoRA (1 GPU)
#   ws_build   103 GiB RSS (fp32 base load + dense vector), 10-14 min
#   ws_gather  148 GiB RSS, 102 min for n=66
TRAIN_TIME, TRAIN_MEM = "01:30:00", "32G"
BUILD_TIME, BUILD_MEM = "01:00:00", "120G"  # dense vocab-matrix diff
GATHER_TIME = "04:00:00"
GATHER_MEM_FLOOR_GB = 120  # what n<=~40 needs; below this the request is pointless
REF_PARAMS = 7.5e9  # the calibration point above; scale factors are relative to it


def _model_params(cfg: dict) -> float | None:
    """Parameter count of the base, from its safetensors bytes (bf16 -> /2).

    Summing the shard files works for single-file and indexed checkpoints alike
    (model views symlink the shards, and ``os.stat`` follows symlinks). An fp32
    checkpoint overestimates 2x, which only makes the requests conservative.
    """
    try:
        files = list(Path(cfg["model"]).glob("*.safetensors"))
        total = sum(f.stat().st_size for f in files)
    except OSError:
        return None
    return total / 2 if total else None


def _scale(cfg: dict) -> float:
    """Resource multiplier vs the 7-8B calibration; never below 1."""
    params = _model_params(cfg)
    if not params:
        return 1.0
    return max(1.0, params / REF_PARAMS)


def _hms(hours: float) -> str:
    total = int(math.ceil(hours * 3600))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _train_resources(cfg: dict) -> tuple[str, str]:
    """(time, mem) for one LoRA task. Wall time grows with the forward cost;
    CPU RAM is only shard loading. Both scale linearly, with headroom for the
    slower 8-bit expert loop on MoE bases."""
    k = _scale(cfg)
    return _hms(1.5 * k), f"{math.ceil(32 * k)}G"


def _build_resources(cfg: dict) -> tuple[str, str]:
    """(time, mem) for one dense task-vector build.

    ``cs_task_vectors --build_index`` holds, all in fp32 on the CPU: the dense
    pos deltas (every LoRA module expanded to a full dW, so ~4 bytes * params),
    the base model while it is loaded to diff the modules_to_save weights (its
    bf16 shards plus the fp32 copy transiently, ~6 bytes * params), then the
    neg deltas and the pos-neg vector (4 bytes * params each). Peak is the
    base-load phase, ~10 bytes * params, or the three-vector phase, ~12: the
    2026-09-23 smoke on the 30.5B MoE was OOM-killed at 352 GiB *during the
    base load* under a 336G request, so the old 8 bytes * params estimate was
    short. 7.3B observed 103 GiB. The write is ~4 bytes * params to the NAS,
    so time scales too.
    """
    params = _model_params(cfg)
    if not params:
        return BUILD_TIME, BUILD_MEM
    gib = 16 * params / 2**30
    mem = max(120, math.ceil(gib * 1.1) + 40)
    return _hms(1.0 * _scale(cfg) * 1.5), f"{mem}G"


def _gather_time(cfg: dict) -> str:
    """Gather streams n vectors of 4 bytes * params each off the NAS: 102 min at
    7.3B. Params alone under-predicts MoE bases, whose per-expert modules make the
    walk far longer: Qwen3-30B-A3B (18.7k modules) timed out at 88% of a 16.3 h
    request on 2026-10-01. So ~8 h x scale, capped under general's 2-day limit;
    the fork's gather also checkpoints, so a timeout resumes rather than restarts."""
    return _hms(min(8.0 * _scale(cfg), 47.0))


def _gather_mem(cfg: dict, values: list[str]) -> str:
    """Peak gather RSS is linear in the value count, so the request must be too.

    ``cs_task_vectors --from_cache`` accumulates the Gram matrix one module at a
    time, materializing ``X = torch.empty((n, numel), float32)`` for each. The
    embedding/lm_head modules dominate at ``vocab_size * hidden_size``, so peak
    bytes ~= n * vocab * hidden * 4. A fixed 120G silently covered n=21 (49 GiB)
    and OOM-killed n=66 (153 GiB observed as 180 GiB RSS, job 10301978), which
    cost a controller run on 2026-09-03. Read the shape from the base config and
    add 30% for the transient Gram plus interpreter overhead.
    """
    try:
        conf = json.loads((Path(cfg["model"]) / "config.json").read_text())
        numel = int(conf["vocab_size"]) * int(conf["hidden_size"])
    except (OSError, KeyError, ValueError, TypeError):
        return f"{GATHER_MEM_FLOOR_GB}G"  # unreadable config: keep the old default
    gib = len(values) * numel * 4 / 2**30
    return f"{max(GATHER_MEM_FLOOR_GB, math.ceil(gib * 1.3) + 16)}G"


def _ws_root() -> Path:
    from valuegen._external import weight_steering_root

    return weight_steering_root()


def _scaffold(cfg: dict) -> str | None:
    """SCAFFOLD_TEMPLATES name the training text is re-rendered through, or None.

    Set for the base-model arm (pretrained bases + urial0), mirroring
    persona/grad_proj's ``template=`` param. When present, the pairs go out as
    axolotl ``input_output`` segments and the recipe defaults to the
    template-free ``SCAFFOLD_RECIPE`` rather than the chat_template one.
    """
    return cfg.get("scaffold")


def _recipe(cfg: dict) -> str:
    if "recipe" in cfg:
        return cfg["recipe"]
    return SCAFFOLD_RECIPE if _scaffold(cfg) else RECIPE


def _variant(cfg: dict) -> str | None:
    """Optional output namespace for a non-default training recipe.

    Defaults to the scaffold name so the base+scaffold grids nest under e.g.
    ``.../urial0/`` — matching persona/grad_proj — and never collide with the
    chat-template (instruct) grid built from the same pairs artifact.
    """
    return cfg.get("recipe_variant") or _scaffold(cfg)


def _identity(cfg: dict) -> dict:
    identity = {"recipe": _recipe(cfg)}
    if _variant(cfg):
        identity["recipe_variant"] = _variant(cfg)
    if _scaffold(cfg):
        identity["scaffold"] = _scaffold(cfg)
    return identity


def _paths(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> dict[str, Path]:
    aid = artifact.artifact_id
    cache = (cluster.finetune_root / "weight_steering_taskvecs" / aid
             / D.model_short(cfg["model"]))
    sim = store.similarity_dir(cluster, "weight_steer", artifact.method,
                               cfg["model"], aid)
    if _variant(cfg):
        cache = cache / _variant(cfg)
        sim = sim / _variant(cfg)
    return {
        "sft": artifact.root / "ws_sft" / D.model_short(cfg["model"]),
        "adapters": cluster.finetune_root / "weight_steering",
        "cache": cache,
        "sim": sim,
    }


def _adapter_tmpl(artifact: D.Artifact, model: str, variant: str | None = None) -> str:
    suffix = f"_{variant}" if variant else ""
    return f"ws_{artifact.artifact_id}_{D.model_short(model)}{suffix}_{{value}}_{{pol}}"


def _template_paths(cfg: dict) -> tuple[Path, Path]:
    root = _ws_root()
    template = root / "axolotl_configs" / _recipe(cfg)
    accel = root / "axolotl_configs" / "multimodel" / "accelerate_single.yaml"
    for p in (template, accel):
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} missing — the weight-steering fork needs the multimodel "
                "axolotl template (see external/weight-steering)"
            )
    return template, accel


def stages(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list:
    from valuegen.slurm import Stage, Task

    root = _ws_root()
    paths = _paths(cfg, cluster, artifact)
    compat = D.fork_compat_dir(artifact, cfg["model"])
    template, accel = _template_paths(cfg)
    values = list(cfg["values"])
    tmpl = _adapter_tmpl(artifact, cfg["model"], _variant(cfg))
    scaffold = _scaffold(cfg)
    dataset_ext = "io.jsonl" if scaffold else "jsonl"

    # prep: fork cs_prep_data over the compat view (pandas-only, one cpu task).
    prep_cmds = [
        f"python {root}/cs_prep_data.py --value {v} "
        f"--persona_extract_dir {compat} --out_dir {paths['sft']} --threshold 50"
        for v in values
    ]
    if scaffold:
        # Re-render cs_prep_data's bare-question messages into axolotl
        # input_output segments through training.SCAFFOLD_TEMPLATES, byte-matched
        # to how persona/grad_proj render the same scaffold (response-only via the
        # segment label). valuegen owns the render, so this runs in the *core*
        # python explicitly — the ws_prep stage's own env (weight_steering) has no
        # valuegen. cs_prep_data drops the steering system prompt by design; the
        # ± answer contrast is weight_steer's steering signal, so the scaffold sees
        # exactly the bare user turn, same as the chat_template arm.
        core_py = cluster.venv("default") / "bin" / "python"
        prep_cmds.append(
            f"{core_py} -m valuegen.predictors.ws_scaffold "
            f"--scaffold {scaffold} --sft-dir {paths['sft']} "
            f"--values {' '.join(values)}"
        )

    def _prep_done() -> bool:
        return all(
            (paths["sft"] / f"{v}_{pol}.{dataset_ext}").is_file()
            for v in values for pol in ("pos", "neg")
        )

    prep = Stage(
        name="ws_prep",
        tasks=[Task(key="prep_all", command="\n".join(prep_cmds), done=_prep_done)],
        time="00:30:00",
        mem="8G",
        env="weight_steering",
    )

    # train: 2N LoRA SFTs (value × polarity), axolotl, ``gpus`` GPUs each (one
    # process; >1 GPU is naive model parallelism through the recipe's
    # ``gpu_memory_limit`` -> accelerate device_map, numerically the same
    # training as one GPU -- the recipe must set it, or the extra GPUs idle).
    # The config is rendered at task runtime (sed on the fork template) so
    # --dry-run touches nothing on the finetune root.
    gpus = int(cfg.get("gpus", 1))
    cvd = ",".join(str(i) for i in range(gpus))
    # Launcher: `accelerate launch` (the 7B runs, byte-identical) for one GPU.
    # For >1 GPU the trainer must be started WITHOUT it: accelerate exports
    # ACCELERATE_USE_* into the child, and axolotl's normalize_config then
    # nulls cfg.device_map ("let accelerate place the model"), so the 8-bit
    # quantizer drops the whole model on GPU 0 (smoke 2026-09-23: 29-31 GB
    # on card 0 under `device_map: balanced`, OOM in the loss; the same
    # recipe through axolotl's loader without the launcher splits 13/16 GB).
    # Single-process either way, so it is the same training loop.
    launcher = (
        f"accelerate launch --config_file {accel} --num_processes 1 -m axolotl.cli.train"
        if gpus == 1 else "python -m axolotl.cli.train"
    )
    train_time, train_mem = _train_resources(cfg)
    build_time, build_mem = _build_resources(cfg)
    train_tasks = []
    for value in values:
        for pol in ("pos", "neg"):
            out_dir = paths["adapters"] / tmpl.format(value=value, pol=pol)

            def _done(d: Path = out_dir) -> bool:
                return (d / "adapter_model.safetensors").is_file() or \
                    (d / "adapter_model.bin").is_file()

            train_tasks.append(
                Task(
                    key=f"{value}_{pol}",
                    command=(
                        f"OUTPUT_DIR={out_dir}\n"
                        f'mkdir -p "$OUTPUT_DIR"\n'
                        f"sed -e 's|__BASE_MODEL__|{cfg['model']}|g' \\\n"
                        f"    -e 's|__DATASET__|{paths['sft']}/{value}_{pol}.{dataset_ext}|g' \\\n"
                        f"    -e \"s|__OUTPUT_DIR__|$OUTPUT_DIR|g\" "
                        f"{template} > \"$OUTPUT_DIR/config.yml\"\n"
                        f'CUDA_VISIBLE_DEVICES={cvd} {launcher} "$OUTPUT_DIR/config.yml"\n'
                        f'test -f "$OUTPUT_DIR/adapter_model.safetensors" || '
                        f'test -f "$OUTPUT_DIR/adapter_model.bin"'
                    ),
                    done=_done,
                )
            )
    train = Stage(
        name="ws_train",
        tasks=train_tasks,
        time=train_time,
        mem=train_mem,
        gpus=gpus,
        # The only stage that needs the pinned axolotl env. prep/build/gather just
        # read adapters (raw safetensors + regex, no peft) and run in core.
        env="ws_train",
        extra_exports={"WANDB_PROJECT": cfg.get("wandb_project", "valuegen-ws")},
    )

    # build: one task vector per value, cached as safetensors for the
    # shard-streaming gather (parallel-build pattern).
    values_arg = " ".join(values)
    build_tasks = [
        Task(
            key=value,
            command=(
                f"python {root}/cs_task_vectors.py \\\n"
                f"    --adapters_dir {paths['adapters']} \\\n"
                f"    --adapter_tmpl '{tmpl}' \\\n"
                f"    --base_model {cfg['model']} \\\n"
                f"    --values {values_arg} \\\n"
                f"    --cache_dir {paths['cache']} \\\n"
                f"    --build_index {i}"
            ),
            done=paths["cache"] / f"{value}.safetensors",
        )
        for i, value in enumerate(values)
    ]
    build = Stage(
        name="ws_build",
        tasks=build_tasks,
        time=build_time,
        mem=build_mem,
        gpus=1,
        env="weight_steering",
    )

    # gather: stream the cached vectors into one cosine matrix. No
    # --ground_truth: task vectors are GT-decoupled by design. Done only
    # when the built matrix covers this run's value subset — a matrix built
    # for a smaller subset must not be reused with NaN holes.
    gather_mem = _gather_mem(cfg, values)
    gather = Stage(
        name="ws_gather",
        tasks=[
            Task(
                key="gather",
                command=(
                    f"python {root}/cs_task_vectors.py \\\n"
                    f"    --adapters_dir UNUSED_IN_CACHE_MODE \\\n"
                    f"    --adapter_tmpl '{tmpl}' \\\n"
                    f"    --base_model {cfg['model']} \\\n"
                    f"    --values {values_arg} \\\n"
                    f"    --cache_dir {paths['cache']} \\\n"
                    f"    --from_cache \\\n"
                    f"    --output_dir {paths['sim']}"
                ),
                done=lambda: _sim_covers(paths, values),
            )
        ],
        time=_gather_time(cfg),
        mem=gather_mem,
        env="weight_steering",
        cpu_partition=True,
    )
    return [prep, train, build, gather]


def _sim_covers(paths: dict[str, Path], values: list[str]) -> bool:
    """Whether the fork's built matrix exists *and* covers ``values``.

    Unreadable resume state (a sidecar truncated by a killed gather task) counts
    as incomplete, not as an error: the gather stage rebuilds it.
    """
    values_json = paths["sim"] / "similarity_values.json"
    if not (paths["sim"] / "similarity_matrix.npy").is_file() \
            or not values_json.is_file():
        return False
    try:
        built = json.loads(values_json.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"warning: unreadable {values_json}; rebuilding weight-steering matrix")
        return False
    if not isinstance(built, list):
        return False
    return set(values) <= set(built)


def needs_slurm(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> bool:
    paths = _paths(cfg, cluster, artifact)
    return not _sim_covers(paths, list(cfg["values"]))


def plan(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    n = len(cfg["values"])
    return [
        f"weight_steer (model={cfg['model']}, data={artifact.artifact_id}, "
        f"recipe={_recipe(cfg)}):",
        "  ws_prep    1 cpu task (cs_prep_data over fork_compat pairs)",
        f"  ws_train   {2 * n}-task GPU array (LoRA SFT, {cfg.get('gpus', 1)} gpu/task, "
        f"{_train_resources(cfg)[1]}/{_train_resources(cfg)[0]} each)",
        f"  ws_build   {n}-task GPU array ({_build_resources(cfg)[1]}/"
        f"{_build_resources(cfg)[0]} — dense embed/lm_head diffs)",
        f"  ws_gather  1 cpu task ({_gather_mem(cfg, cfg['values'])}/{_gather_time(cfg)}; "
        f"~{4 * (_model_params(cfg) or REF_PARAMS) / 1e9:.0f} GB/vector cached on "
        "finetune_root, delete after)",
    ]


def run(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    dry_run: bool = False,
) -> Path | None:
    paths = _paths(cfg, cluster, artifact)
    if not dry_run:
        store.claim_dir(paths["sim"], artifact.artifact_id)
        D.build_fork_compat(artifact, cfg["model"])

    if needs_slurm(cfg, cluster, artifact) or dry_run:
        from valuegen.slurm import Orchestrator

        orch = Orchestrator(
            name=f"predict_ws_{artifact.artifact_id}{D.job_suffix(cfg, artifact)}",
            stages=stages(cfg, cluster, artifact),
            cluster=cluster,
            dry_run=dry_run,
        )
        if not orch.run():
            raise RuntimeError("weight-steering pipeline incomplete; rerun to resume")
        if dry_run:
            return None

    # Register the fork's raw output in the standard similarity convention,
    # NaN-padded onto the full requested value order.
    raw = np.load(paths["sim"] / "similarity_matrix.npy")
    built_values = json.loads((paths["sim"] / "similarity_values.json").read_text())
    sim = store.nan_pad(raw, built_values, cfg["values"])
    out = store.save_similarity(
        paths["sim"], SIM_NAME, sim, cfg["values"],
        # SIM_NAME is a constant, so this is the only thing keeping two runs
        # with different LoRA recipes from overwriting each other's cells.
        identity=_identity(cfg),
        provenance={
            "predictor": "weight_steer",
            "data_artifact": artifact.artifact_id,
            "model": cfg["model"],
            "adapter_tmpl": _adapter_tmpl(
                artifact, cfg["model"], _variant(cfg)
            ),
            "cache_dir": str(paths["cache"]),
            "resolved_config": dict(cfg),
        },
    )
    print(f"wrote {out}")
    print(f"note: task-vector cache at {paths['cache']} "
          f"(~{4 * (_model_params(cfg) or REF_PARAMS) / 1e9:.0f} GB/value) can "
          "be deleted now; adapters are small and stay.")
    return out
