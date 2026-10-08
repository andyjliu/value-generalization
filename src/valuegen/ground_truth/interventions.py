"""Intervention artifacts: the seam between training and evaluation.

An *intervention* is anything that steers a base model toward a value — a DPO
or SFT checkpoint (``label_subset``, ``conflictscope_pairs``,
``model_spec_aft``), or nothing (``none``, base models only). The
training side of a GT experiment produces an intervention **artifact**; the
evaluation side consumes it through its manifest and never looks at how it
was made. That seam is what lets one set of checkpoints be scored under many
eval configs, and eval-only methods skip training entirely.

Artifact layout::

    data/interventions/{method}/{value_set}/{intervention_id}/
        resolved_config.yaml    # config_id = intervention_id, resolved block
        manifest.yaml           # models + per-tag payloads (see below)
        datasets/{value}/dataset.jsonl      # training methods (legacy: .csv)
    {finetune_root}/interventions/{intervention_id}/{mtag}_{value}/   # trainer out
    {finetune_root}/merged/{intervention_id}_{mtag}_{value}_merged    # checkpoints

The manifest lists every intervention as ``{tag, mtag, value, kind, path}``
with ``kind: checkpoint`` (path = merged model dir) or ``kind: system_prompt``
(path = prompt file). It is written up front, at claim time — declaring the
grid — and an entry is *ready* when its payload exists on disk, so a manifest
doubles as the completion check for a referenced artifact.

``gt run`` with ``schedule.delete_exports`` removes a checkpoint once the GT
run's eval of it is done (``delete_checkpoint``), leaving a ``<merged>.deleted``
tombstone: the checkpoint still counts as *trained* (nothing retrains) but is
no longer *ready* (nothing can load it).

Method driver surface (each method module):

- ``build_data(cfg, cluster)`` — inline, idempotent data build.
- ``stages(cfg, cluster) -> list[Stage]`` — the training-side SLURM stages
  (empty for eval-time methods).
- ``manifest_entries(cfg, cluster) -> list[dict]`` — the declared grid.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from valuegen.config import (
    ClusterConfig,
    canonical_experiment,
    ensure_config_record,
    gt_id,
    intervention_id,
)
from valuegen.ground_truth import gcs, training
from valuegen.slurm import Stage, Task

INTERVENTION_METHODS = {
    "label_subset": "valuegen.ground_truth.label_subset",
    "conflictscope_pairs": "valuegen.ground_truth.conflictscope",
    "model_spec_aft": "valuegen.ground_truth.model_spec_aft",
    "none": None,
}


def get_method(name: str):
    """Intervention method name -> driver module (lazy import; None for none)."""
    if name not in INTERVENTION_METHODS:
        raise KeyError(
            f"Unknown intervention method {name!r}; known: "
            f"{sorted(INTERVENTION_METHODS)}"
        )
    module = INTERVENTION_METHODS[name]
    return importlib.import_module(module) if module else None


def resolve_models(entries, chat_format: str | None = None) -> dict[str, str]:
    """Intervention ``models:`` block -> ordered ``{mtag: hf_name}``.

    Takes a mapping of explicit ``mtag: hf_name`` pairs. MTAGs must contain a
    chat-family keyword (olmo/qwen/tulu/...) — enforced here because the
    merged-dir name inherits it and selects the chat template at eval time.
    With an explicit ``train.chat_format`` the trainer's template is pinned,
    so the model name's own family no longer has to agree with the tag's.
    """
    if not isinstance(entries, dict):
        raise TypeError(
            "intervention.models must be a mapping of mtag: hf_name, got "
            f"{type(entries).__name__}"
        )
    resolved = {str(k): str(v) for k, v in entries.items()}
    for mtag, hf_name in resolved.items():
        mtag_family = training.infer_chat_family(mtag)
        if mtag_family is None:
            raise ValueError(
                f"Model tag {mtag!r} carries no chat-family keyword "
                f"({'/'.join(training.CHAT_FAMILIES)}); merged-dir names derive "
                "from it and select the chat template by substring at eval time."
            )
        hf_family = None if chat_format else training.infer_chat_family(hf_name)
        if hf_family is not None and hf_family != mtag_family:
            raise ValueError(
                f"Model tag {mtag!r} selects the {mtag_family} chat template at "
                f"eval time (via the merged-dir name), but its model "
                f"{hf_name!r} selects {hf_family} at training time — the run "
                "would train and evaluate under different formats. Rename the "
                "tag to match the model's family."
            )
    return resolved


def train_chat_format(iv: dict) -> str | None:
    """The intervention's explicit ``train.chat_format``, if any."""
    return (iv.get("train") or {}).get("chat_format")


# ── Artifact paths ───────────────────────────────────────────────────────────


def _value_set_name(value_set) -> str:
    """A path-component-safe name for the value set (stem of a .json path)."""
    path = Path(str(value_set))
    return path.stem if path.suffix == ".json" else str(value_set)


def intervention_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    iv = cfg["intervention"]
    return (
        cluster.data / "interventions" / iv["method"]
        / _value_set_name(iv["value_set"]) / intervention_id(cfg)
    )


def record_path(cfg: dict, cluster: ClusterConfig) -> Path:
    return intervention_dir(cfg, cluster) / "resolved_config.yaml"


def manifest_path(cfg: dict, cluster: ClusterConfig) -> Path:
    return intervention_dir(cfg, cluster) / "manifest.yaml"


def datasets_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    return intervention_dir(cfg, cluster) / "datasets"


def dataset_file(value_dir: Path) -> Path:
    """The per-value training dataset inside ``value_dir``.

    Fresh builds write ``dataset.jsonl`` — real message lists, so DPOTrainer's
    conversational path applies the pinned chat template and training sees the
    same token contexts the eval serves. Legacy dirs keep their
    ``dataset.csv`` (stringified message reprs trained as raw text, no
    template); ``training._load_pair_dataset`` dispatches on the extension, so
    an existing CSV wins here and keeps its checkpoints reproducible. Never
    convert one format to the other in place.
    """
    csv = value_dir / "dataset.csv"
    return csv if csv.is_file() else value_dir / "dataset.jsonl"


def script_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    """Training-side hostfiles etc. — keyed by intervention, not GT run, so
    the same path holds whichever verb (run/intervene) submitted the work."""
    return cluster.repo / "slurm_jobs" / intervention_id(cfg)


def train_output_dir(cfg: dict, cluster: ClusterConfig, mtag: str, value: str) -> Path:
    return (
        cluster.finetune_root / "interventions" / intervention_id(cfg)
        / f"{mtag}_{value}"
    )


def merged_path(cfg: dict, cluster: ClusterConfig, mtag: str, value: str) -> Path:
    return (
        cluster.finetune_root / "merged"
        / f"{intervention_id(cfg)}_{mtag}_{value}_merged"
    )


# ── Manifest ─────────────────────────────────────────────────────────────────


def checkpoint_entries(cfg: dict, cluster: ClusterConfig) -> list[dict]:
    """The declared grid for training methods: one merged checkpoint per
    (model, value). Shared by label_subset and conflictscope_pairs."""
    iv = cfg["intervention"]
    return [
        {
            "tag": f"{mtag}_{value}",
            "mtag": mtag,
            "value": value,
            "kind": "checkpoint",
            "path": str(merged_path(cfg, cluster, mtag, value)),
        }
        for mtag in resolve_models(iv["models"], train_chat_format(iv))
        for value in iv["values"]
    ]


def build_manifest(cfg: dict, cluster: ClusterConfig) -> dict:
    """The manifest as this config declares it (pure; nothing written)."""
    iv = cfg["intervention"]
    method = get_method(iv["method"])
    entries = [] if method is None else method.manifest_entries(cfg, cluster)
    return {
        "intervention_id": intervention_id(cfg),
        "method": iv["method"],
        "value_set": str(iv["value_set"]),
        "schema_version": iv["schema_version"],
        "models": resolve_models(iv["models"], train_chat_format(iv)),
        "interventions": entries,
    }


def claim(cfg: dict, cluster: ClusterConfig) -> dict:
    """Write (or verify) the artifact's record + manifest, refusing foreign ones.

    The id is a hash of the resolved intervention block, so a manifest that
    regenerates differently under the same id means a moved cluster root, a
    hash collision, or a hand-edited file — either way the payloads on disk
    are not this config's, and mixing them would corrupt the artifact.
    """
    manifest = build_manifest(cfg, cluster)
    directory = intervention_dir(cfg, cluster)
    path = directory / "manifest.yaml"
    if path.is_file():
        existing = yaml.safe_load(path.read_text()) or {}
        if existing != manifest:
            raise RuntimeError(
                f"{path} was written by a different intervention config; "
                "refusing to mix payloads. Delete the directory deliberately "
                "if you really mean to rebuild."
            )
        return manifest
    ensure_config_record(
        directory / "resolved_config.yaml",
        intervention_id(cfg),
        canonical_experiment(cfg["intervention"]),
    )
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def find_manifest(ref: str, cluster: ClusterConfig) -> Path:
    """Locate a referenced intervention artifact by its ID."""
    root = cluster.data / "interventions"
    hits = sorted(root.glob(f"*/*/{ref}/manifest.yaml"))
    if not hits:
        raise FileNotFoundError(
            f"No intervention artifact {ref!r} under {root}. Referenced "
            "interventions are never built implicitly — run `valuegen gt "
            "intervene` on the config that owns it first."
        )
    if len(hits) > 1:
        raise RuntimeError(f"Intervention id {ref!r} is ambiguous: {hits}")
    return hits[0]


def load_manifest(cfg: dict, cluster: ClusterConfig) -> dict:
    """The manifest this experiment evaluates — read-only.

    A referenced intervention is loaded from disk (it must exist); an inline
    block is regenerated in memory, so status/dry-run never write.
    """
    ref = cfg["evaluation"].get("intervention")
    if ref:
        return yaml.safe_load(find_manifest(str(ref), cluster).read_text()) or {}
    return build_manifest(cfg, cluster)


def entry_ready(entry: dict) -> bool:
    path = Path(entry["path"])
    if entry["kind"] == "checkpoint":
        # On disk, or pushed to GCS (upload marker; never written when the
        # gcs block is off, so this stays a pure local-config.json check).
        # A tombstoned checkpoint is trained but not loadable: not ready.
        return gcs.checkpoint_loadable(path)
    return path.is_file() and path.stat().st_size > 0


def tombstone(entry: dict) -> dict | None:
    """The ``<merged>.deleted`` record ``delete_checkpoint`` left, if any."""
    if entry["kind"] != "checkpoint":
        return None
    marker = gcs.deleted_marker_path(entry["path"])
    if not (marker.is_file() and marker.stat().st_size > 0):
        return None
    try:
        record = json.loads(marker.read_text())
    except ValueError:
        record = None
    return record if isinstance(record, dict) else {}


def missing_entries(manifest: dict) -> list[dict]:
    return [e for e in manifest["interventions"] if not entry_ready(e)]


def eval_interventions(
    manifest: dict, mtag: str, cluster: ClusterConfig | None = None
) -> dict[str, dict]:
    """One model's manifest entries in the shape ``evaluation.eval_tasks``
    consumes: ``{tag: {"model": path} | {"steer_prompt": path}}``.

    With ``cluster.gcs`` set, checkpoint entries also carry a ``stage_in``
    shell prelude that re-materializes a pushed-and-pruned checkpoint from
    GCS into node-local scratch at run time (``gcs.stage_in_sh``).
    """
    out = {}
    for entry in manifest["interventions"]:
        if entry["mtag"] != mtag:
            continue
        if entry["kind"] == "checkpoint":
            out[entry["tag"]] = {"model": entry["path"]}
            if cluster is not None and cluster.gcs is not None:
                out[entry["tag"]]["stage_in"] = gcs.stage_in_sh(
                    cluster.gcs, entry["path"]
                )
        elif entry["kind"] == "system_prompt":
            out[entry["tag"]] = {"steer_prompt": entry["path"]}
        else:
            raise ValueError(f"Unknown intervention kind {entry['kind']!r}")
    return out


def values_for(manifest: dict, mtag: str) -> list[str]:
    """The steered values for one model, in manifest order (= matrix rows)."""
    return [e["value"] for e in manifest["interventions"] if e["mtag"] == mtag]


# ── Stage assembly (training side) ───────────────────────────────────────────


def stamp(stages: list[Stage], config_id: str, record: Path) -> list[Stage]:
    """Bind every task's done-check to a config identity record. With split
    identities the orchestrator no longer stamps globally — training tasks
    validate against the intervention record, eval tasks against the GT run's."""
    for stage in stages:
        for task in stage.tasks:
            task.config_id = config_id
            task.config_record = record
    return stages


def write_train_config(cfg: dict, cluster: ClusterConfig) -> Path:
    """Dump the intervention's ``train:`` block verbatim for ``TrlParser --config``."""
    train_block = dict(cfg["intervention"].get("train") or {})
    out = script_dir(cfg, cluster)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train_config.yaml"
    path.write_text(yaml.safe_dump(train_block, sort_keys=False))
    return path


def train_stages(cfg: dict, cluster: ClusterConfig) -> list[Stage]:
    """One train+finish array per base model (kept per-model to stay under
    the normal-QOS array submit cap, like the old controller).

    The finishing step depends on the recipe: LoRA (``use_peft: true``) merges
    the adapter into its base; full FT exports the trainer output as a bf16
    servable dir. Both land at ``merged_path``, so everything downstream —
    manifest, done-checks, GCS push/stage-in, eval — never cares which recipe
    produced the artifact.
    """
    iv = cfg["intervention"]
    models = resolve_models(iv["models"], train_chat_format(iv))
    train_values = list(iv["values"])
    res = iv.get("resources", {}).get("train", {})
    train_cfg = write_train_config(cfg, cluster)
    data_root = datasets_dir(cfg, cluster)
    nproc = int(res.get("nproc", 1))
    gpus = int(res.get("gpus", 1))
    if nproc > 1 and nproc != gpus:
        raise ValueError(
            f"resources.train sets nproc={nproc} but gpus={gpus}: torchrun "
            "binds one rank per GPU, so a multi-process training must request "
            "exactly nproc GPUs."
        )
    # The train block is forwarded verbatim to TRL, whose ModelConfig.use_peft
    # defaults to False — so an absent key trains full-FT, and the finishing
    # step here must agree with what the trainer will actually do.
    full_ft = not (iv.get("train") or {}).get("use_peft", False)
    # An explicit ``train.chat_format`` already pins the trainer's template
    # (forwarded verbatim above); the export must pin the same identity, or
    # the servable dir falls back to the path-substring family (OLMo-3 has
    # none here and would be exported under the OLMo-2 tulu template).
    chat_format = (iv.get("train") or {}).get("chat_format")
    # Unhashed pacing knob (config._resolve_schedule): the fp32 trainer save
    # goes as soon as its export verifies rather than with the wave's deletion.
    schedule = cfg.get("schedule") or {}
    prune_trained = bool(
        schedule.get("prune_trainer_save", schedule.get("delete_exports", False))
    )
    run_id = intervention_id(cfg)

    stages = []
    for mtag, hf_name in models.items():
        tasks = []
        for value in train_values:
            out_dir = train_output_dir(cfg, cluster, mtag, value)
            merged = merged_path(cfg, cluster, mtag, value)
            finish_target, init, prelude = merged, hf_name, ""
            if cluster.gcs is not None:
                # With remote checkpoints the finishing step writes to
                # node-local scratch and the push rsyncs straight from there,
                # so the shared filesystem never holds a model-sized
                # transient. The trainer output dir stays put (its metadata
                # is what export/prune/delete read).
                finish_target = gcs.staged_path(cluster.gcs, merged)
                # An init that is itself a checkpoint under finetune_root
                # (e.g. a neutral-SFT model, possibly already pruned to the
                # bucket) gets the same run-time stage-in prelude eval tasks
                # use; trainer and finish then read $MODEL_DIR. The staged dir
                # keeps the basename, so the chat-family keyword survives.
                if Path(hf_name).is_absolute():
                    prelude = gcs.stage_in_sh(cluster.gcs, hf_name) + "\n"
                    init = '"$MODEL_DIR"'
            finish = (
                training.export_command(
                    init, out_dir, finish_target,
                    chat_format=chat_format, verify=bool(chat_format),
                    base_revision=(iv.get("train") or {}).get("model_revision"),
                    prune_trained=prune_trained,
                )
                if full_ft
                else training.merge_command(init, out_dir, finish_target)
            )
            command = (
                prelude
                + training.train_command(
                    algo=iv.get("algo", "dpo"),
                    train_config=train_cfg,
                    dataset=dataset_file(data_root / value),
                    model=init,
                    output_dir=out_dir,
                    run_name=f"{run_id}_{mtag}_{value}",
                    nproc=nproc,
                )
                + "\n"
                + finish
            )
            # Trained = on disk, pushed, or deleted on purpose after its evals
            # (tombstone) — the last must hold without GCS too.
            done = (lambda m=merged: gcs.checkpoint_trained(m))
            if cluster.gcs is not None:
                command = (
                    gcs.guarded_train_sh(command, merged)
                    + "\n"
                    + gcs.push_and_prune_sh(
                        cluster.gcs, merged, local_dir=finish_target
                    )
                )
            tasks.append(
                Task(
                    key=f"{mtag}/{value}",
                    command=command,
                    done=done,
                )
            )
        stages.append(
            Stage(
                name=f"train_{mtag}",
                tasks=tasks,
                time=res.get("time", "05:00:00"),
                mem=res.get("mem", "48G"),
                gpus=gpus,
                extra_exports={
                    "WANDB_PROJECT": iv.get("wandb_project", "valuegen")
                },
            )
        )
    return stages


# ── Deletion (schedule.delete_exports) ───────────────────────────────────────


def _tree_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _finish_delete(merged: Path, trainer_dir: Path) -> None:
    """The destructive half of ``delete_checkpoint`` — idempotent, so a
    controller killed mid-delete is completed by the next run's hook."""
    doomed = merged.with_name(merged.name + ".deleting")
    if merged.is_dir():
        if doomed.is_dir():
            shutil.rmtree(doomed)
        # Rename first: a half-removed tree must never sit at the path whose
        # config.json reads as "loadable".
        merged.rename(doomed)
    if doomed.is_dir():
        shutil.rmtree(doomed)
    training.prune_trainer_dir(trainer_dir, checkpoints=True)


def delete_checkpoint(
    cfg: dict,
    cluster: ClusterConfig,
    entry: dict,
    eval_tasks: list[Task],
    by: str = "gt run --delete-exports",
) -> tuple[bool, str]:
    """Remove one trained checkpoint once this GT run has scored it, leaving
    a tombstone the training done-check accepts.

    Returns ``(deleted, reason)``. Nothing is removed unless the entry is a
    ``kind: checkpoint`` of this config's own inline intervention (never a
    base model, a steering prompt, or a referenced artifact) and *every* task
    in ``eval_tasks`` — this GT run's evals of the tag — passes ``is_done()``,
    gt_id stamp included. The payload is the merged dir plus the trainer dir's
    weight/optimizer shards (``training.prune_trainer_dir``); trainer_state,
    logs, configs and the LoRA adapter stay.

    The tombstone is written *first*, so every crash point leaves a state the
    next run completes (``_finish_delete``) instead of a half-deleted dir that
    still looks loadable. Scoring the same intervention under another eval
    config afterwards needs a retrain: remove ``<merged>.deleted`` and rerun
    ``gt intervene``.
    """
    tag = entry["tag"]
    if entry["kind"] != "checkpoint":
        return False, f"{tag}: {entry['kind']} interventions are never deleted"
    if cfg.get("intervention") is None:
        return False, f"{tag}: referenced artifacts are never deleted"
    if cluster.gcs is not None:
        return False, (
            f"{tag}: cluster stages checkpoints to {cluster.gcs.bucket} (gcs: "
            "block); local copies are already push-and-pruned, so deletion is "
            "a no-op"
        )
    merged = Path(entry["path"])
    trainer_dir = train_output_dir(cfg, cluster, entry["mtag"], entry["value"])
    if tombstone(entry) is not None:
        _finish_delete(merged, trainer_dir)  # no-op unless a delete was cut short
        return False, f"{tag}: checkpoint already deleted"
    if not eval_tasks:
        return False, f"{tag}: no eval task in this GT run"
    pending = [t.key for t in eval_tasks if not t.is_done()]
    if pending:
        return False, f"{tag}: eval not done ({', '.join(pending)})"
    if not gcs.checkpoint_loadable(merged):
        return False, f"{tag}: no checkpoint at {merged}"

    freed = _tree_bytes(merged) + sum(
        _tree_bytes(sub) if sub.is_dir() else sub.stat().st_size
        for pattern in ("*.safetensors", "pytorch_model*.bin", "checkpoint-*", "global_step*")
        for sub in trainer_dir.glob(pattern)
        if sub.name != "adapter_model.safetensors"
    )
    record = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by": by,
        "gt_id": gt_id(cfg),
        "eval_tasks": [t.key for t in eval_tasks],
        "freed_bytes": freed,
    }
    gcs.deleted_marker_path(merged).write_text(json.dumps(record, indent=2) + "\n")
    _finish_delete(merged, trainer_dir)
    return True, f"{tag}: checkpoint deleted ({freed / 2**30:.1f} GiB freed)"


def _delete_hook(
    delete: Callable[[dict, list[Task]], tuple[bool, str]],
    targets: list[tuple[dict, list[Task]]],
) -> Callable[[], None]:
    def after() -> None:
        for entry, eval_tasks in targets:
            _, reason = delete(entry, eval_tasks)
            print(f"  delete_exports: {reason}", flush=True)

    return after


def interleave_waves(
    train_stages: list[Stage],
    eval_stages: list[Stage],
    manifest: dict,
    wave: int,
    delete: Callable[[dict, list[Task]], tuple[bool, str]] | None = None,
    stop_after: int | None = None,
) -> list[Stage]:
    """Alternate training and evaluation in waves of ``wave`` values per model.

    ``gt run`` normally trains every value before the first eval starts
    (``train_{mtag}`` then ``eval_{mtag}``). With ``schedule.wave: W`` the
    same tasks are re-cut into ``train_{mtag}_w{k}`` / ``eval_{mtag}_w{k}``
    pairs of at most W values each, in manifest order, so the first W
    checkpoints are scored while the rest are still queued. Nothing about the
    tasks changes — same commands, same done-checks, same artifacts — only
    the order the orchestrator walks them in.

    The judge (any server an eval stage ``needs_servers``) is marked
    ``on_demand`` so the orchestrator does not submit it up front: the first
    eval wave's ``needs_servers`` check serves it, and every eval wave
    releases it via ``cancel_servers``; the next wave's eval re-serves it the
    same way. On a shared GPU budget a resident judge would otherwise halve
    training concurrency — in the first wave as much as between waves. The base eval task (``{mtag}_base``) joins the *last* wave,
    and only if its done-check does not already pass (a shared base store
    usually has it), so it never delays the first wave's signal.

    ``delete`` (``schedule.delete_exports``; ``delete_checkpoint`` bound to
    the config) is attached to every eval wave as its ``Stage.after`` hook,
    over exactly that wave's manifest entries — each with its own eval task
    as the gate — so a wave's checkpoints are freed before the next wave
    trains. ``{mtag}_base`` is never a target. A restarted controller walks
    completed waves again (nothing pending, the stage completes at once) and
    the hook re-fires; the tombstone gate makes that a no-op.

    ``stop_after`` (``schedule.stop_after_wave``) keeps only the first N waves
    of each model — a pilot pass of the real run. The base eval then joins
    wave N, the last one scheduled. A later run without it walks the finished
    waves as no-ops and carries on from wave N+1.

    Stages that are not a per-model train/eval pair (scenario pools, server
    stages, the pooled ``eval_base`` of method: none panels) pass through
    unchanged: pools and servers in front, in their original order, the rest
    after the waves.
    """
    if wave < 1:
        raise ValueError(f"wave must be >= 1, got {wave}")
    if stop_after is not None and stop_after < 1:
        raise ValueError(f"stop_after must be >= 1, got {stop_after}")
    trains = {s.name: s for s in train_stages if s.name.startswith("train_")}
    evals = {s.name: s for s in eval_stages if s.name.startswith("eval_")}
    servers = [
        dataclasses.replace(s, on_demand=True) for s in eval_stages if s.server
    ]
    front = [s for s in train_stages if s.name not in trains]
    tail = [s for s in eval_stages if s.name not in evals and not s.server]

    result: list[Stage] = list(front) + list(servers)
    for mtag in manifest["models"]:
        train = trains.pop(f"train_{mtag}", None)
        ev = evals.pop(f"eval_{mtag}", None)
        if train is None or ev is None:
            # One side only (eval-time methods, referenced artifacts, a model
            # with no rows): nothing to interleave, keep whatever exists.
            result += [s for s in (train, ev) if s is not None]
            continue
        train_by_value = {t.key.split("/", 1)[1]: t for t in train.tasks}
        eval_by_tag = {t.key: t for t in ev.tasks}
        entries = {
            e["value"]: e for e in manifest["interventions"] if e["mtag"] == mtag
        }
        values = list(entries)
        missing = [v for v in values if v not in train_by_value or f"{mtag}_{v}" not in eval_by_tag]
        if missing:
            raise ValueError(
                f"interleave_waves: {mtag} has manifest values without a "
                f"matching train+eval task: {missing[:5]}"
            )
        chunks = [values[i : i + wave] for i in range(0, len(values), wave)]
        chunks = chunks[:stop_after]
        base = eval_by_tag.get(f"{mtag}_base")
        for k, chunk in enumerate(chunks):
            eval_tasks = [eval_by_tag[f"{mtag}_{v}"] for v in chunk]
            if k == len(chunks) - 1 and base is not None and not base.is_done():
                eval_tasks.append(base)
            result.append(
                dataclasses.replace(
                    train,
                    name=f"{train.name}_w{k}",
                    tasks=[train_by_value[v] for v in chunk],
                )
            )
            result.append(
                dataclasses.replace(
                    ev,
                    name=f"{ev.name}_w{k}",
                    tasks=eval_tasks,
                    cancel_servers=tuple(ev.needs_servers),
                    after=None if delete is None else _delete_hook(
                        delete,
                        [(entries[v], [eval_by_tag[f"{mtag}_{v}"]]) for v in chunk],
                    ),
                )
            )
    # Train/eval stages for models the manifest does not list (should not
    # happen; keep them rather than drop work silently).
    result += list(trains.values()) + list(evals.values()) + tail
    return result


def build_data(cfg: dict, cluster: ClusterConfig) -> None:
    """Inline data build + artifact claim for an inline intervention block."""
    if cfg.get("intervention") is None:
        return  # referenced artifact: nothing to build here, ever
    claim(cfg, cluster)
    method = get_method(cfg["intervention"]["method"])
    if method is not None:
        method.build_data(cfg, cluster)


def stages(cfg: dict, cluster: ClusterConfig) -> list[Stage]:
    """The training-side stage list, stamped with the intervention identity.

    Empty for the eval-only method (none) and for referenced
    artifacts — a reference is a promise the work exists, not a request to
    rebuild it.
    """
    if cfg.get("intervention") is None:
        return []
    method = get_method(cfg["intervention"]["method"])
    if method is None:
        return []
    return stamp(
        method.stages(cfg, cluster),
        intervention_id(cfg),
        record_path(cfg, cluster),
    )
