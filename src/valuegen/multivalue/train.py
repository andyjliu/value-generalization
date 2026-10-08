"""``mv train``: one SLURM task per trained arm x seed (train -> export -> record).

    valuegen mv train -c CFG --dry-run            # render slurm_jobs/multivalue/{exp_id}/train.sbatch, submit nothing
    valuegen mv train -c CFG                      # submit and poll (Orchestrator.run: retries, filesystem-checked)
    valuegen mv train -c CFG --no-wait            # submit once and return
    valuegen mv train -c CFG --record CKPT_ID     # (called by the job) verify a finished run, write run_record.json

Inputs are the frozen design (``sets.json``) and the published mixes
(``mixes/manifest.json`` from ``mv data``): a run is refused when its mix is
missing, carries gate failures, or no longer matches the manifest. Each run
lives under ``{finetune_root}/multivalue/{exp_id}/{arm}_s{seed}/``:

    trainer/            TRL output (final save only: ``save_strategy: no``)
    export/             ``training export --verify`` output (the checkpoint evals serve)
    run_record.json     verification against the manifest's expected geometry

Restart-from-base semantics as in the RQ3 drivers: a partial trainer output
(no ``config.json``) is removed and training restarts from the pinned base
with the same seed; a finished trainer output is reused; the export is
rebuilt only when its verification record is missing. A run is *done* when
its record carries this experiment's identity (exp_id, arm, seed, dataset
sha) with ``export_verified: true`` and the export is present locally or, on
clusters with a ``gcs:`` block, pushed and pruned (marker file), or deleted
on purpose by ``mv run --delete-exports`` (an ``export_deleted`` tombstone in
the record, see :func:`delete_export`). Trainer weight shards are deleted
after a verified export unless ``train.prune_trainer_save: false``.

Base and imported arms are never trained here.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from valuegen.config import ClusterConfig
from valuegen.ground_truth import gcs
from valuegen.ground_truth import training as T
from valuegen.multivalue import data as mvdata
from valuegen.multivalue._hashing import sha256_file
from valuegen.multivalue.config import MultivalueConfig
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.sets import Arm
from valuegen.slurm import Orchestrator, Stage, Task, sbatch_available

RUN_RECORD_VERSION = 1
EXPECT_KEYS = ("expect_raw_rows", "expect_train_rows", "expect_max_steps", "expect_warmup_steps")


class TrainError(RuntimeError):
    pass


def q(x) -> str:
    return shlex.quote(str(x))


# ── runs ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Run:
    arm: Arm
    seed: int
    ckpt_id: str
    root: Path
    mix: dict  # the arm's manifest entry

    @property
    def trainer_dir(self) -> Path:
        return self.root / "trainer"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def record_path(self) -> Path:
        return self.root / "run_record.json"

    @property
    def dataset_sha256(self) -> str:
        return self.mix["dataset_sha256"]

    @property
    def expect(self) -> dict:
        return {k: int(self.mix["expect"][k]) for k in EXPECT_KEYS}

    def identity(self, exp_id: str) -> dict:
        return {"exp_id": exp_id, "arm_id": self.arm.id, "seed": self.seed, "dataset_sha256": self.dataset_sha256}


def trainable_arms(arms: Sequence[Arm]) -> list[Arm]:
    """Arms this stage trains: everything but the untrained base and imports."""
    return [a for a in arms if a.kind not in ("base", "imported")]


def load_manifest(layout: Layout) -> dict:
    path = layout.mixes_dir / "manifest.json"
    if not path.is_file():
        raise TrainError(f"no mixes manifest at {path}: run `valuegen mv data` first")
    manifest = mvdata.read_json(path)
    if manifest.get("exp_id") != layout.exp_id:
        raise TrainError(f"{path}: exp_id {manifest.get('exp_id')} != {layout.exp_id} (rebuild with `mv data`)")
    if manifest.get("gate_failures"):
        raise TrainError(f"{path}: mixes carry gate failures; nothing is trained: {manifest['gate_failures'][:3]}")
    if not manifest.get("nesting_ok", False):
        raise TrainError(f"{path}: nested selection check failed")
    return manifest


def check_arm_mix(layout: Layout, manifest: dict, arm: Arm) -> dict:
    """The manifest entry for ``arm``, verified against the mix on disk."""
    entry = manifest["arms"].get(arm.id)
    if entry is None:
        raise TrainError(f"{arm.id}: not in the mixes manifest (run `valuegen mv data --arm {arm.id}`)")
    if tuple(entry["values"]) != tuple(arm.values):
        raise TrainError(f"{arm.id}: manifest values differ from the frozen design")
    comp_path = Path(entry["mix_dir"]) / "composition.json"
    if not comp_path.is_file() or not (Path(entry["mix_dir"]) / "dataset.jsonl").is_file():
        raise TrainError(f"{arm.id}: mix at {entry['mix_dir']} is incomplete")
    comp = mvdata.read_json(comp_path)
    if comp.get("dataset_sha256") != entry["dataset_sha256"]:
        raise TrainError(f"{arm.id}: composition.json dataset_sha256 differs from the manifest (rebuild with `mv data`)")
    # A mix built with --skip-token-audit carries geometry that assumes zero TRL
    # drops; the trainer's own expect_* gates still abort on a mismatch.
    return entry


def plan_runs(cfg: MultivalueConfig, layout: Layout, arms: Sequence[Arm], manifest: dict, *,
              only: Sequence[str] | None = None, seeds: Sequence[int] | None = None) -> list[Run]:
    """One :class:`Run` per trainable arm x seed, in (family, arm, seed) order."""
    todo = trainable_arms(arms)
    if only:
        known = {a.id for a in arms}
        unknown = sorted(set(only) - known)
        if unknown:
            raise TrainError(f"unknown arm ids: {unknown}")
        skipped = sorted(set(only) - {a.id for a in todo})
        if skipped:
            raise TrainError(f"not trainable here (base or imported): {skipped}")
        todo = [a for a in todo if a.id in set(only)]
    use_seeds = tuple(seeds) if seeds else cfg.train.seeds
    bad = sorted(set(use_seeds) - set(cfg.train.seeds))
    if bad:
        raise TrainError(f"seeds {bad} are not in train.seeds {list(cfg.train.seeds)}")
    runs = []
    for a in sorted(todo, key=lambda a: (a.family, a.id)):
        entry = check_arm_mix(layout, manifest, a)
        for s in use_seeds:
            runs.append(Run(arm=a, seed=int(s), ckpt_id=layout.ckpt_id(a.id, int(s)),
                            root=layout.checkpoint_dir(a.id, int(s)), mix=entry))
    return runs


# ── commands ─────────────────────────────────────────────────────────────────


def _nproc(cfg: MultivalueConfig) -> int:
    res = cfg.train.resources
    return int(res.get("nproc") or res.get("gpus", 1))


def _world_size(cfg: MultivalueConfig) -> int:
    return _nproc(cfg)


def record_command(cfg: MultivalueConfig, cluster: ClusterConfig, run: Run) -> str:
    cmd = f"python -m valuegen.multivalue.cli train -c {q(cfg.path)} --record {q(run.ckpt_id)}"
    if cluster.source_path is not None:
        cmd += f" --cluster {q(cluster.source_path)}"
    return cmd


def train_command(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, run: Run) -> str:
    """Shell for one task: guarded train -> guarded export+verify -> record
    (-> GCS push-and-prune when the cluster has a ``gcs:`` block)."""
    tr = cfg.train
    train = T.train_command(
        algo="dpo", train_config=Path(tr.recipe).resolve(), dataset=Path(run.mix["mix_dir"]) / "dataset.jsonl",
        model=tr.base_model, output_dir=run.trainer_dir, run_name=f"mv_{layout.exp_id}_{run.ckpt_id}",
        nproc=_nproc(cfg),
    )
    train += f" \\\n    --seed {run.seed}"
    train += f" \\\n    --chat_format {tr.chat_format}"
    if tr.fsdp_config is not None:
        train += f" \\\n    --fsdp_config {q(Path(tr.fsdp_config).resolve())}"
    if tr.revision:
        if not Path(tr.base_model).is_dir():
            train += f" \\\n    --model_revision {tr.revision}"
        train += f" \\\n    --base_revision {tr.revision}"
    for k, v in run.expect.items():
        train += f" \\\n    --{k} {v}"
    train += f" \\\n    --expect_world_size {_world_size(cfg)}"
    export = T.export_command(
        tr.base_model, run.trainer_dir, run.export_dir, chat_format=tr.chat_format,
        base_revision=tr.revision, verify=True,
    )
    body = f"""mkdir -p {q(run.root)}
# No periodic checkpoints in this recipe: a partial trainer output (no final
# config.json) is not resumable, so it is removed and training restarts from
# the pinned base with the same seed. A finished trainer output is reused.
if [ ! -f {q(run.trainer_dir / "config.json")} ]; then
    rm -rf {q(run.trainer_dir)}
    {train}
fi
if [ ! -f {q(run.export_dir / "export_verification.json")} ]; then
    rm -rf {q(run.export_dir)} {q(str(run.export_dir) + ".partial")}
    {export}
fi
{record_command(cfg, cluster, run)}"""
    if cluster.gcs is not None:
        body = gcs.guarded_train_sh(body, run.export_dir) + "\n" + gcs.push_and_prune_sh(cluster.gcs, run.export_dir)
    return body


# ── verification / done ──────────────────────────────────────────────────────


def _read(path: Path):
    return mvdata.read_json(path) if path.is_file() else None


def verify_run(cfg: MultivalueConfig, layout: Layout, run: Run) -> tuple[list[str], dict]:
    """Problems with a finished run's artifacts, plus the records read."""
    from valuegen.ground_truth.chat_formats import get_chat_format

    fmt = get_chat_format(cfg.train.chat_format)
    expect = run.expect
    world = _world_size(cfg)
    meta = _read(run.trainer_dir / "run_metadata.json")
    geometry = _read(run.trainer_dir / "step_geometry.json")
    integrity = _read(run.trainer_dir / "dpo_integrity.json")
    verification = _read(run.export_dir / "export_verification.json")
    problems: list[str] = []
    if meta is None:
        problems.append("no run_metadata.json")
    else:
        if meta.get("final_save") != "done":
            problems.append("trainer final save not recorded")
        if meta.get("dataset_sha256") != run.dataset_sha256:
            problems.append(f"trained dataset sha {str(meta.get('dataset_sha256'))[:12]} != mix {run.dataset_sha256[:12]}")
        if meta.get("n_train_rows") not in (None, expect["expect_raw_rows"]):
            problems.append(f"raw rows {meta.get('n_train_rows')} != expected {expect['expect_raw_rows']}")
        if meta.get("n_train_rows_after_trainer_preprocessing") != expect["expect_train_rows"]:
            problems.append(f"rows after preprocessing {meta.get('n_train_rows_after_trainer_preprocessing')} "
                            f"!= expected {expect['expect_train_rows']}")
        if (meta.get("observed_max_steps") != expect["expect_max_steps"]
                or meta.get("observed_global_step") != expect["expect_max_steps"]):
            problems.append(f"steps {meta.get('observed_global_step')}/{meta.get('observed_max_steps')} "
                            f"!= expected {expect['expect_max_steps']}")
        if meta.get("observed_warmup_steps") != expect["expect_warmup_steps"]:
            problems.append(f"warmup {meta.get('observed_warmup_steps')} != expected {expect['expect_warmup_steps']}")
        if meta.get("world_size") != world:
            problems.append(f"world size {meta.get('world_size')} != {world}")
        if meta.get("seed") not in (None, run.seed):
            problems.append(f"trainer seed {meta.get('seed')} != {run.seed}")
        cf = meta.get("chat_format") or {}
        if cf.get("chat_format") != fmt.name or cf.get("template_sha256") != fmt.template_sha256:
            problems.append("trainer chat format/template differs from the pinned format")
        if cfg.train.revision and meta.get("base_revision") != cfg.train.revision:
            problems.append("base revision not recorded as pinned")
    if geometry is None:
        problems.append("no step_geometry.json")
    elif geometry.get("max_steps") != expect["expect_max_steps"]:
        problems.append(f"step geometry max_steps {geometry.get('max_steps')}")
    if integrity is None:
        problems.append("no dpo_integrity.json")
    else:
        if integrity.get("policy_changed") is not True:
            problems.append("policy parameters did not change")
        if integrity.get("reference_fixed") is not True:
            problems.append("reference parameters were not fixed")
        if integrity.get("losses_finite") is not True:
            problems.append("non-finite loss or grad norm observed")
    if verification is None:
        problems.append("no export_verification.json")
    elif verification.get("problems"):
        problems.append(f"export verification problems {verification['problems']}")
    elif (verification.get("sidecar") or {}).get("chat_format") != fmt.name:
        problems.append("export sidecar chat format mismatch")
    if not (run.export_dir / "config.json").is_file():
        problems.append("export has no config.json")
    return problems, {"run_metadata": meta, "step_geometry": geometry, "dpo_integrity": integrity,
                      "export_verification": verification}


def record_run(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, run: Run) -> dict:
    """Verify a finished run and write ``run_record.json`` (the done evidence).
    Prunes trainer weight shards after a verified export when configured."""
    problems, files = verify_run(cfg, layout, run)
    shas = ({f.name: sha256_file(f) for f in sorted(run.export_dir.glob("*.safetensors"))}
            if run.export_dir.is_dir() and not problems else None)
    record = {
        "schema_version": RUN_RECORD_VERSION, "ckpt_id": run.ckpt_id, "identity": run.identity(layout.exp_id),
        "arm": {"id": run.arm.id, "family": run.arm.family, "kind": run.arm.kind, "values": list(run.arm.values)},
        "trainer_dir": str(run.trainer_dir), "export_dir": str(run.export_dir),
        "expect": run.expect, "world_size": _world_size(cfg),
        "export_verified": not problems, "problems": problems, **files,
        "export_sha256": shas, "trainer_save_pruned": False, "recorded_at": mvdata.now(),
    }
    mvdata.write_json(run.record_path, record)
    if not problems and cfg.train.prune_trainer_save:
        T.prune_trainer_dir(run.trainer_dir)
        record["trainer_save_pruned"] = True
        mvdata.write_json(run.record_path, record)
    return record


def export_tombstone(run: Run) -> dict | None:
    """The ``export_deleted`` record written by :func:`delete_export`, if any."""
    rec = _read(run.record_path)
    tomb = (rec or {}).get("export_deleted")
    return tomb if isinstance(tomb, dict) else None


def export_present(run: Run, cluster: ClusterConfig) -> bool:
    """The export is on disk, (GCS clusters) pushed and pruned, or deleted
    on purpose after its evals completed (tombstone)."""
    if (run.export_dir / "config.json").is_file():
        return True
    if cluster.gcs is not None and gcs.ready(run.export_dir):
        return True
    return export_tombstone(run) is not None


def run_done(layout: Layout, cluster: ClusterConfig, run: Run) -> bool:
    rec = _read(run.record_path)
    if not rec or rec.get("export_verified") is not True:
        return False
    if rec.get("identity") != run.identity(layout.exp_id):
        return False
    return export_present(run, cluster)


def run_state(layout: Layout, cluster: ClusterConfig, run: Run) -> str:
    """One word per run for status tables. ``pruned`` is a done state whose
    export was deleted after its evals completed."""
    if run_done(layout, cluster, run):
        return "pruned" if export_tombstone(run) is not None else "done"
    rec = _read(run.record_path)
    if rec is not None and rec.get("identity") == run.identity(layout.exp_id) and rec.get("problems"):
        return "failed-verify"
    if rec is not None and rec.get("identity") != run.identity(layout.exp_id):
        return "stale"
    if (run.export_dir / "export_verification.json").is_file():
        return "exported"
    if (run.trainer_dir / "config.json").is_file():
        return "trained"
    if run.trainer_dir.is_dir():
        return "partial"
    return "pending"


# ── deletion ─────────────────────────────────────────────────────────────────


def delete_export(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, run: Run, *,
                  by: str = "mv run --delete-exports") -> tuple[bool, str]:
    """Remove a verified run's ``export/`` once every enabled suite has
    scored it, leaving a tombstone the done-check accepts.

    Returns ``(deleted, reason)``. Nothing is removed unless the run is done
    with this experiment's identity and *every* suite in ``evals.suites``
    that is enabled has frozen inputs and a ``done`` marker for this
    checkpoint -- regardless of which ``--suite`` subset a wave ran. Base and
    imported arms are never touched (their exports are not ours). The
    tombstone (``export_deleted: {at, suites, by}``) keeps ``run_done`` true
    so nothing retrains; ``export_sha256`` stays as the fingerprint of what
    was served. Enabling a new suite later needs an explicit retrain: remove
    the run dir and ``mv train --arm ID``.
    """
    from valuegen.multivalue.evals import runner as R
    from valuegen.multivalue.evals.base import Candidate

    if cluster.gcs is not None:
        return False, (f"{run.ckpt_id}: cluster stages exports to {cluster.gcs.bucket} "
                       "(gcs: block); local exports are already push-and-pruned, so deletion is a no-op")
    if run.arm.kind in ("base", "imported"):
        return False, f"{run.ckpt_id}: {run.arm.kind} arm exports are never deleted"
    if export_tombstone(run) is not None:
        return False, f"{run.ckpt_id}: export already deleted"
    if not run_done(layout, cluster, run):
        return False, f"{run.ckpt_id}: run is not done ({run_state(layout, cluster, run)})"
    enabled = cfg.evals.enabled()
    if not enabled:
        return False, f"{run.ckpt_id}: no enabled suites"
    cand = Candidate(ckpt_id=run.ckpt_id, arm=run.arm, seed=run.seed, kind="trained", model_dir=run.export_dir,
                     export_verified=True, model_source="export")
    for suite in enabled:
        path = R.inputs_path(layout, suite)
        if not path.is_file():
            return False, f"{run.ckpt_id}: suite {suite} has no frozen inputs at {path}"
        state = R.suite_state(cfg, layout, cand, suite, mvdata.read_json(path))
        if state != "done":
            return False, f"{run.ckpt_id}: suite {suite} is {state}, not done"
    if run.export_dir.is_dir():
        shutil.rmtree(run.export_dir)
    partial = run.export_dir.with_name(run.export_dir.name + ".partial")
    if partial.is_dir():
        shutil.rmtree(partial)
    rec = _read(run.record_path) or {}
    rec["export_deleted"] = {"at": mvdata.now(), "suites": list(enabled), "by": by}
    mvdata.write_json(run.record_path, rec)
    return True, f"{run.ckpt_id}: export deleted after {enabled}"


# ── stage ────────────────────────────────────────────────────────────────────


def build_stage(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, runs: Sequence[Run], *,
                name: str = "train") -> Stage:
    res = cfg.train.resources
    tasks = [Task(key=f"train/{r.ckpt_id}", command=train_command(cfg, cluster, layout, r),
                  done=(lambda r=r: run_done(layout, cluster, r))) for r in runs]
    return Stage(name=name, tasks=tasks, time=str(res["time"]), mem=str(res.get("mem") or cluster.default_mem),
                 gpus=int(res["gpus"]), env=cfg.train.env, throttle=int(cfg.train.concurrent),
                 extra_exports={"WANDB_MODE": "disabled"})


def orchestrator_name(layout: Layout) -> str:
    """Scripts under ``slurm_jobs/multivalue/{exp_id}/`` (= ``layout.slurm_jobs_dir``)."""
    return f"multivalue/{layout.exp_id}"


def write_runs_record(layout: Layout, runs: Sequence[Run], cluster: ClusterConfig) -> Path:
    """``{exp_dir}/train/runs.json``: the checkpoint ids and export dirs later stages serve."""
    path = layout.exp_dir / "train" / "runs.json"
    rec = {"schema_version": 1, "exp_id": layout.exp_id, "runs": {
        r.ckpt_id: {"arm_id": r.arm.id, "family": r.arm.family, "kind": r.arm.kind, "values": list(r.arm.values),
                    "seed": r.seed, "root": str(r.root), "export_dir": str(r.export_dir),
                    "dataset_sha256": r.dataset_sha256, "expect": r.expect,
                    "gcs_uri": gcs.remote_uri(cluster.gcs, r.export_dir) if cluster.gcs is not None else None}
        for r in runs}, "written_at": mvdata.now()}
    mvdata.write_json(path, rec)
    return path


def train(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm], *,
          dry_run: bool = False, no_wait: bool = False, only: Sequence[str] | None = None,
          seeds: Sequence[int] | None = None, log=print) -> int:
    """The ``mv train`` entry point (minus ``--record``). Returns an exit code."""
    manifest = load_manifest(layout)
    runs = plan_runs(cfg, layout, arms, manifest, only=only, seeds=seeds)
    if not runs:
        log("nothing to train: no trainable arms (base and imported arms are skipped)")
        return 0
    write_runs_record(layout, runs, cluster)
    stage = build_stage(cfg, cluster, layout, runs)
    pending = stage.pending()
    log(f"train: {len(stage.tasks)} runs ({len(runs) // max(1, len(set(r.seed for r in runs)))} arms x "
        f"{len(set(r.seed for r in runs))} seeds), {len(pending)} pending, "
        f"{int(cfg.train.resources['gpus'])} GPUs x {cfg.train.resources['time']} each, {cfg.train.concurrent} concurrent")
    for r in runs:
        state = run_state(layout, cluster, r)
        if state != "done":
            log(f"  - {r.ckpt_id}: {state}  (k={r.arm.k}, {r.mix['n_rows']} rows, {r.expect['expect_max_steps']} steps)")
    if not pending:
        log("all runs done")
        return 0
    orch = Orchestrator(name=orchestrator_name(layout), stages=[stage], cluster=cluster, dry_run=dry_run)
    if dry_run:
        orch.run()
        log(f"dry run: rendered {layout.slurm_jobs_dir / 'train.sbatch'}")
        return 0
    if not sbatch_available():
        log("sbatch not found; run from the SLURM login node or use --dry-run")
        return 1
    if no_wait:
        orch.submit()
        return 0
    return 0 if orch.run() else 1


def record_main(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm],
                ckpt_id: str, log=print) -> int:
    manifest = load_manifest(layout)
    match = [(a, s) for a in trainable_arms(arms) for s in cfg.train.seeds if layout.ckpt_id(a.id, s) == ckpt_id]
    if len(match) != 1:
        raise TrainError(f"{ckpt_id}: not a run of this experiment")
    arm, seed = match[0]
    (run,) = plan_runs(cfg, layout, arms, manifest, only=[arm.id], seeds=[seed])
    rec = record_run(cfg, cluster, layout, run)
    log(json.dumps({"ckpt_id": ckpt_id, "export_verified": rec["export_verified"], "problems": rec["problems"],
                    "trainer_save_pruned": rec["trainer_save_pruned"]}, indent=2))
    return 0 if rec["export_verified"] else 1


if __name__ == "__main__":  # pragma: no cover
    print("use `valuegen mv train`", file=sys.stderr)
    sys.exit(2)
