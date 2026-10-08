"""``mv import --from DIR``: register externally trained arms.

An import is a run record from one of the RQ3 experiment drivers
(``{out_root}/runs.jsonl``: one line per trained checkpoint plus the neutral
base entry). Each trained run becomes an arm of kind ``imported`` with its
values, rows, checkpoint and any completed suite directories, so ``mv
metrics`` scores it, ``mv data``/``mv train`` skip it, and ``mv eval``
serves its export.

Imports are identity-free (``{root}/imports.json`` beside ``sets.json``):
they can be registered before the design is frozen, and their checkpoints
are never rebuilt. The importer refuses a run whose base model, revision or
chat format differs from ``train`` (``--allow-base-mismatch`` to override,
for exploratory analysis only) and an arm id that collides with a frozen
arm; re-importing an identical run is a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import yaml

from valuegen.multivalue import data as mvdata
from valuegen.multivalue.config import SUITES
from valuegen.multivalue.sets import Arm

IMPORTS_SCHEMA_VERSION = 1


class ImportError_(RuntimeError):
    pass


def imports_path(layout) -> Path:
    return layout.root / "imports.json"


def _empty() -> dict:
    return {"schema_version": IMPORTS_SCHEMA_VERSION, "arms": {}, "base_model_dirs": {}, "base_evals": {}}


def load_imports_record(layout) -> dict:
    p = imports_path(layout)
    if not p.is_file():
        return _empty()
    rec = mvdata.read_json(p)
    if rec.get("schema_version") != IMPORTS_SCHEMA_VERSION:
        raise ImportError_(f"{p}: schema_version {rec.get('schema_version')} != {IMPORTS_SCHEMA_VERSION}")
    return rec


def load_imports(layout) -> list[Arm]:
    """Imported arms (kind ``imported``, family ``import:{experiment}``);
    ``extra`` carries the full import record (rows, model_dir, evals ...)."""
    rec = load_imports_record(layout)
    return [Arm(id=aid, family=f"import:{r['source']['experiment']}", values=tuple(r["values"]),
                kind="imported", extra=dict(r))
            for aid, r in sorted(rec["arms"].items())]


def all_arms(layout, sets_record: dict | None = None) -> list[Arm]:
    """Frozen arms (if frozen) followed by imported arms."""
    from valuegen.multivalue import sets as mvsets

    arms: list[Arm] = []
    if sets_record is not None or layout.frozen:
        arms = mvsets.arms_from_record(sets_record or mvsets.load_sets(layout.sets_path))
    imported = load_imports(layout)
    clash = {a.id for a in arms} & {a.id for a in imported}
    if clash:
        raise ImportError_(f"imported arm ids collide with frozen arms: {sorted(clash)[:5]}")
    return arms + imported


# ── reading an RQ3 experiment ────────────────────────────────────────────────


def locate_runs(from_dir: str | Path) -> tuple[Path, Path]:
    """``(runs.jsonl, out_root)`` for an experiment directory, its ``out``
    symlink, or the out root itself."""
    d = Path(from_dir).expanduser().resolve()
    candidates = []
    cfg = d / "config.yaml"
    if cfg.is_file():  # the driver's own out root first (the `out` symlink points there)
        try:
            raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            out_root = (raw.get("layout") or {}).get("out_root")
            if out_root:
                candidates.append(Path(out_root) / "runs.jsonl")
        except yaml.YAMLError:
            pass
    candidates += [d / "runs.jsonl", d / "out" / "runs.jsonl"]
    for c in candidates:
        if c.is_file():
            return c, c.parent
    raise ImportError_(f"no runs.jsonl under {d} (looked at {[str(c) for c in candidates]})")


def read_runs(path: Path) -> list[dict]:
    return [r for _, r in mvdata.iter_jsonl(path)]


def _find_suite_dirs(out_root: Path, checkpoint_id: str) -> dict[str, str]:
    """Completed suite directories for a checkpoint under either layout
    (``evals/{suite}/{ckpt}`` for the RQ3 drivers, ``evals/{ckpt}/{suite}`` here)."""
    found = {}
    for suite in SUITES:
        for d in (out_root / "evals" / suite / checkpoint_id, out_root / "evals" / checkpoint_id / suite):
            if (d / "COMPLETE.json").is_file():
                found[suite] = str(d)
                break
    return found


def _base_of(runs: Sequence[dict]) -> dict:
    neutral = [r for r in runs if r.get("kind") == "neutral"]
    if len(neutral) != 1:
        return {}
    ident = neutral[0].get("identity") or {}
    return {"repo": ident.get("base_repo"), "revision": ident.get("base_revision"),
            "chat_format": ident.get("chat_format"), "template_sha256": ident.get("template_sha256"),
            "model_dir": neutral[0].get("model_dir"), "checkpoint_id": neutral[0].get("checkpoint_id")}


def check_base(cfg, base: dict, run_ident: dict) -> list[str]:
    problems = []
    repo = base.get("repo")
    if repo and repo != cfg.train.base_model:
        problems.append(f"base model {repo} != train.base_model {cfg.train.base_model}")
    rev = run_ident.get("base_revision") or base.get("revision")
    if cfg.train.revision and rev and rev != cfg.train.revision:
        problems.append(f"base revision {rev[:12]} != train.revision {cfg.train.revision[:12]}")
    fmt = run_ident.get("chat_format") or base.get("chat_format")
    if fmt and fmt != cfg.train.chat_format:
        problems.append(f"chat_format {fmt} != train.chat_format {cfg.train.chat_format}")
    return problems


def import_record(run: dict, experiment: str, runs_path: Path, out_root: Path, base: dict) -> dict:
    ident = run.get("identity") or {}
    mix_dir = run.get("mix_dir")
    comp = None
    if mix_dir and (Path(mix_dir) / "composition.json").is_file():
        comp = mvdata.read_json(Path(mix_dir) / "composition.json")
    model_dir = run.get("model_dir") or run.get("export_dir")
    values = list(run.get("values") or (comp or {}).get("values") or [])
    if not values:
        raise ImportError_(f"{experiment}/{run.get('arm_id')}: no values in the run record or its composition")
    ckpt = run.get("checkpoint_id") or run.get("run_id")
    return {
        "source": {"experiment": experiment, "runs": str(runs_path), "out_root": str(out_root),
                   "run_id": run.get("run_id"), "checkpoint_id": ckpt, "arm_id": run.get("arm_id"),
                   "arm_kind": run.get("arm_kind")},
        "values": sorted(values), "k": len(values),
        "rows": int((comp or {}).get("n_rows") or run.get("expect", {}).get("expect_raw_rows") or 0),
        "seed": run.get("training_seed"),
        "model_dir": model_dir,
        "export_verified": bool(model_dir and (Path(model_dir) / "export_verification.json").is_file()),
        "mix_dir": mix_dir,
        "dataset_sha256": (comp or {}).get("dataset_sha256") or run.get("mix_dataset_sha256"),
        "base": {"repo": base.get("repo"), "revision": ident.get("base_revision") or base.get("revision"),
                 "chat_format": ident.get("chat_format") or base.get("chat_format"),
                 "template_sha256": ident.get("template_sha256") or base.get("template_sha256")},
        "identity": ident,
        "evals": _find_suite_dirs(out_root, ckpt) if ckpt else {},
        "imported_at": mvdata.now(),
    }


def _same_import(a: dict, b: dict) -> bool:
    keys = ("values", "dataset_sha256", "model_dir")
    return all(a.get(k) == b.get(k) for k in keys) and a["source"]["checkpoint_id"] == b["source"]["checkpoint_id"]


def import_experiment(cfg, cluster, layout, from_dir: str | Path, *, universe: Sequence[str],
                      allow_base_mismatch: bool = False, force: bool = False, prefix: str | None = None,
                      exclude: Sequence[str] | None = None, log=print) -> dict:
    """Register every trained run of an RQ3 experiment. ``exclude`` names
    source arm ids (before ``prefix``) to leave out, e.g. a driver's k=49
    controls that a k=6 sweep has no use for. Returns
    ``{"added": [...], "unchanged": [...], "skipped": [...], "record": imports_record}``."""
    runs_path, out_root = locate_runs(from_dir)
    runs = read_runs(runs_path)
    experiment = next((r.get("identity", {}).get("experiment") for r in runs if r.get("identity", {}).get("experiment")),
                      None) or Path(from_dir).resolve().name
    base = _base_of(runs)
    rec = load_imports_record(layout)
    frozen_ids = set()
    if layout.frozen:
        from valuegen.multivalue import sets as mvsets

        frozen_ids = {a.id for a in mvsets.arms_from_record(mvsets.load_sets(layout.sets_path))}
    universe_set = set(universe)
    excluded = set(exclude or ())
    added, unchanged, skipped = [], [], []
    for run in runs:
        if run.get("kind") != "trained":
            continue
        if str(run.get("arm_id")) in excluded:
            skipped.append(str(run["arm_id"]))
            continue
        problems = check_base(cfg, base, run.get("identity") or {})
        if problems and not allow_base_mismatch:
            raise ImportError_(f"{experiment}/{run.get('arm_id')}: " + "; ".join(problems)
                               + " (pass --allow-base-mismatch to import anyway)")
        entry = import_record(run, experiment, runs_path, out_root, base)
        if problems:
            entry["base_mismatch"] = problems
        outside = sorted(set(entry["values"]) - universe_set)
        if outside:
            raise ImportError_(f"{experiment}/{run.get('arm_id')}: values outside the universe: {outside[:5]}")
        aid = f"{prefix}{run['arm_id']}" if prefix else str(run["arm_id"])
        if aid in frozen_ids:
            raise ImportError_(f"imported arm id {aid!r} collides with a frozen arm (use --prefix)")
        prev = rec["arms"].get(aid)
        if prev is not None:
            if _same_import(prev, entry):
                unchanged.append(aid)
                continue
            if not force:
                raise ImportError_(f"{aid}: already imported from {prev['source']['experiment']} with a different "
                                   f"checkpoint/values/rows (pass --force to replace)")
        rec["arms"][aid] = entry
        added.append(aid)
        log(f"  {aid}: k={entry['k']} rows={entry['rows']} export={'ok' if entry['export_verified'] else 'MISSING'} "
            f"evals={sorted(entry['evals']) or '-'}")
    if base.get("repo") and base.get("revision") and base.get("model_dir"):
        key = f"{base['repo']}@{base['revision']}"
        if not check_base(cfg, base, {}) or allow_base_mismatch:
            rec["base_model_dirs"][key] = base["model_dir"]
            # the source experiment's own untrained-base evals: the base
            # candidate reuses them like any imported arm's (eval_stage)
            if base.get("checkpoint_id"):
                found = _find_suite_dirs(out_root, base["checkpoint_id"])
                if found:
                    rec.setdefault("base_evals", {})[key] = found
    unknown = sorted(excluded - {str(r.get("arm_id")) for r in runs})
    if unknown:
        raise ImportError_(f"{experiment}: --exclude names arms it has no run for: {unknown}")
    if added:
        layout.root.mkdir(parents=True, exist_ok=True)
        mvdata.write_json(imports_path(layout), rec)
    return {"added": added, "unchanged": unchanged, "skipped": skipped, "record": rec, "experiment": experiment}
