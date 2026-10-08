"""Every on-disk path of a multivalue experiment, derived from (config, cluster).

    {cluster.data}/multivalue/{name}/
        sets.json                     frozen arms (the training identity input)
        sets_preview/                 unfrozen previews from `mv sets`
        metrics/metrics.csv           arm x store -> coverage, tightness, k, rows
        {exp_id}/                     everything that depends on the identity
            config.yaml               resolved-config record
            mixes/{arm}/              dataset.jsonl, rows.jsonl, composition.json
            train/runs.json           checkpoint ids + export dirs of the train stage
            eval_inputs/{suite}/inputs.json  frozen suite inputs (rendered panel, sample ids, prefill data)
            evals/candidates.json     every served checkpoint of the eval stage
            evals/{ckpt_id}/serve/    vLLM hostfile + log of the serving job
            evals/{ckpt_id}/{suite}/  Inspect logs per unit, grades.jsonl, rows.jsonl, COMPLETE.json
                                      (a symlink into the source experiment for a reused imported run)
            evals_smoke/              the same layout for `mv eval --smoke`
            scores/                   outcomes.csv, joined.csv, correlations.csv, figures/
            REPORT.md
    {cluster.finetune_root}/multivalue/{exp_id}/{arm}_s{seed}/
        trainer/  export/  run_record.json  (checkpoints)
    {cluster.finetune_root}/multivalue/{exp_id}/base/export/
        the untrained base staged as a servable export (weights symlinked into the HF cache)

``exp_id = f"{name}-{hash}"`` where the hash covers the frozen ``sets.json``
bytes plus :meth:`MultivalueConfig.training_identity`. It exists only once
``sets.json`` is frozen; the identity-free paths (sets, previews, metrics)
never need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from valuegen.config import ClusterConfig
from valuegen.multivalue._hashing import sha256_file, short_hash
from valuegen.multivalue.config import MultivalueConfig


class NotFrozenError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    cfg: MultivalueConfig
    cluster: ClusterConfig

    # ── identity-free ──
    @property
    def root(self) -> Path:
        return self.cluster.data / "multivalue" / self.cfg.name

    @property
    def sets_path(self) -> Path:
        return self.root / "sets.json"

    @property
    def preview_dir(self) -> Path:
        return self.root / "sets_preview"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def metrics_csv(self) -> Path:
        return self.metrics_dir / "metrics.csv"

    @property
    def frozen(self) -> bool:
        return self.sets_path.is_file()

    # ── identity ──
    def identity(self) -> dict:
        if not self.frozen:
            raise NotFrozenError(
                f"{self.sets_path} does not exist: run `valuegen mv sets ... --freeze` first"
            )
        return {"sets_sha256": sha256_file(self.sets_path), **self.cfg.training_identity()}

    @property
    def exp_id(self) -> str:
        return f"{self.cfg.name}-{short_hash(self.identity())}"

    @property
    def exp_dir(self) -> Path:
        return self.root / self.exp_id

    @property
    def config_record(self) -> Path:
        return self.exp_dir / "config.yaml"

    @property
    def mixes_dir(self) -> Path:
        return self.exp_dir / "mixes"

    def mix_dir(self, arm_id: str) -> Path:
        return self.mixes_dir / arm_id

    @property
    def checkpoint_root(self) -> Path:
        return self.cluster.finetune_root / "multivalue" / self.exp_id

    def ckpt_id(self, arm_id: str, seed: int) -> str:
        return f"{arm_id}_s{seed}"

    def checkpoint_dir(self, arm_id: str, seed: int) -> Path:
        return self.checkpoint_root / self.ckpt_id(arm_id, seed)

    @property
    def base_export_dir(self) -> Path:
        """The servable copy of the untrained base this package stages."""
        return self.checkpoint_root / "base" / "export"

    @property
    def evals_dir(self) -> Path:
        return self.exp_dir / "evals"

    @property
    def smoke_evals_dir(self) -> Path:
        return self.exp_dir / "evals_smoke"

    def candidate_dir(self, ckpt_id: str, *, smoke: bool = False) -> Path:
        return (self.smoke_evals_dir if smoke else self.evals_dir) / ckpt_id

    def suite_dir(self, ckpt_id: str, suite: str, *, smoke: bool = False) -> Path:
        return self.candidate_dir(ckpt_id, smoke=smoke) / suite

    @property
    def candidates_record(self) -> Path:
        return self.evals_dir / "candidates.json"

    @property
    def eval_inputs_dir(self) -> Path:
        return self.exp_dir / "eval_inputs"

    @property
    def scores_dir(self) -> Path:
        return self.exp_dir / "scores"

    @property
    def report_path(self) -> Path:
        return self.exp_dir / "REPORT.md"

    @property
    def slurm_jobs_dir(self) -> Path:
        return self.cluster.repo / "slurm_jobs" / "multivalue" / self.exp_id

    # ── source ──
    @property
    def source_datasets_dir(self) -> Path:
        """``data/interventions/{method}/{value_set}/{artifact}/datasets``."""
        from valuegen.multivalue.data import source_datasets_dir

        return source_datasets_dir(self.cfg, self.cluster)
