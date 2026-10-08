"""Read-only arm-by-stage status for ``valuegen mv status``."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import eval_stage as ES
from valuegen.multivalue import train as T
from valuegen.multivalue.config import MultivalueConfig
from valuegen.multivalue.evals import runner as R
from valuegen.multivalue.evals.base import EvalError
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.sets import Arm


def _read_json(path: Path) -> dict | None:
    try:
        return mvdata.read_json(path) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _collapse(states: Mapping[str, str]) -> str:
    """A compact aggregate that preserves per-seed differences."""
    if not states:
        return "pending"
    values = list(states.values())
    if len(set(values)) == 1:
        return values[0]
    return ",".join(f"{key}:{value}" for key, value in states.items())


def _metrics_states(cfg: MultivalueConfig, layout: Layout, arms: Sequence[Arm]) -> dict[str, str]:
    if not layout.metrics_csv.is_file():
        return {a.id: "pending" for a in arms}
    try:
        df = pd.read_csv(layout.metrics_csv)
        sets = _read_json(layout.sets_path) or {}
        if "arms_sha256" not in df or set(df["arms_sha256"].dropna().astype(str)) != {str(sets.get("arms_sha256"))}:
            return {a.id: "stale" for a in arms}
        out = {}
        for a in arms:
            d = df[df["arm_id"].astype(str) == a.id]
            stores = set(d["store"].astype(str)) if "store" in d else set()
            out[a.id] = "done" if len(d) == len(cfg.embeddings) and stores == set(cfg.embeddings) else "stale"
        return out
    except Exception:
        return {a.id: "invalid" for a in arms}


def _mix_state(layout: Layout, arm: Arm, manifest: dict | None) -> str:
    if arm.kind == "base":
        return "n/a"
    if arm.kind == "imported":
        return "imported"
    entry = (manifest or {}).get("arms", {}).get(arm.id)
    comp = _read_json(layout.mix_dir(arm.id) / "composition.json")
    dataset = layout.mix_dir(arm.id) / "dataset.jsonl"
    if entry is None and comp is None:
        return "pending"
    if entry is None or comp is None or not dataset.is_file():
        return "partial"
    if (manifest or {}).get("exp_id") != layout.exp_id:
        return "stale"
    if (tuple(entry.get("values") or ()) != arm.values
            or entry.get("dataset_sha256") != comp.get("dataset_sha256")
            or comp.get("exp_id") != layout.exp_id):
        return "stale"
    if (manifest or {}).get("gate_failures"):
        return "gate-failed"
    return "done"


def _train_state(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arm: Arm,
                 manifest: dict | None) -> str:
    if arm.kind == "base":
        ready = (layout.base_export_dir / "export_verification.json").is_file()
        return "ready" if ready else "staged-on-eval"
    if arm.kind == "imported":
        model = Path(arm.extra["model_dir"]) if arm.extra.get("model_dir") else None
        ready = bool(arm.extra.get("export_verified") and model and (model / "config.json").is_file())
        return "imported" if ready else "missing-export"
    entry = (manifest or {}).get("arms", {}).get(arm.id)
    if entry is None:
        return "pending"
    states = {}
    for seed in cfg.train.seeds:
        run = T.Run(arm=arm, seed=int(seed), ckpt_id=layout.ckpt_id(arm.id, int(seed)),
                    root=layout.checkpoint_dir(arm.id, int(seed)), mix=entry)
        try:
            states[f"s{seed}"] = T.run_state(layout, cluster, run)
        except (KeyError, TypeError, ValueError):
            states[f"s{seed}"] = "stale"
    return _collapse(states)


def _eval_states(cfg: MultivalueConfig, layout: Layout, candidates: Sequence) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    enabled = cfg.evals.enabled()
    inputs = {s: _read_json(R.inputs_path(layout, s)) for s in enabled}
    by_arm: dict[str, list] = {}
    for cand in candidates:
        by_arm.setdefault(cand.arm.id, []).append(cand)
    for aid, cands in by_arm.items():
        for suite in enabled:
            if inputs[suite] is None:
                out[(aid, suite)] = "inputs-missing"
                continue
            states = {}
            for cand in cands:
                try:
                    states[cand.ckpt_id] = R.suite_state(cfg, layout, cand, suite, inputs[suite])
                except (EvalError, KeyError, TypeError, ValueError):
                    states[cand.ckpt_id] = "unavailable"
            out[(aid, suite)] = _collapse(states)
    return out


def _analysis_states(layout: Layout, arms: Sequence[Arm]) -> dict[str, str]:
    required = [layout.scores_dir / "outcomes.csv", layout.scores_dir / "joined.csv",
                layout.scores_dir / "correlations.csv", layout.report_path]
    if not any(p.exists() for p in required):
        return {a.id: "pending" for a in arms}
    if not all(p.is_file() for p in required):
        return {a.id: "partial" for a in arms}
    try:
        joined = pd.read_csv(layout.scores_dir / "joined.csv")
        got = set(joined["arm_id"].astype(str))
        return {a.id: "done" if a.id in got else "stale" for a in arms}
    except Exception:
        return {a.id: "invalid" for a in arms}


def status_frame(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout,
                 arms: Sequence[Arm]) -> pd.DataFrame:
    """Build the arm x stage table without creating or modifying artifacts."""
    manifest = _read_json(layout.mixes_dir / "manifest.json")
    metric_states = _metrics_states(cfg, layout, arms)
    candidates = ES.plan_candidates(cfg, cluster, layout, arms)
    eval_states = _eval_states(cfg, layout, candidates)
    analysis_states = _analysis_states(layout, arms)
    rows = []
    for arm in arms:
        n_rows = 0 if arm.kind == "base" else int(arm.extra.get("rows", cfg.budget.rows))
        row = {"arm_id": arm.id, "family": arm.family, "kind": arm.kind, "k": arm.k, "rows": n_rows,
               "sets": "done", "metrics": metric_states[arm.id], "data": _mix_state(layout, arm, manifest),
               "train": _train_state(cfg, cluster, layout, arm, manifest)}
        for suite in cfg.evals.enabled():
            row[f"eval.{suite}"] = eval_states.get((arm.id, suite), "pending")
        row["analyze"] = analysis_states[arm.id]
        rows.append(row)
    return pd.DataFrame(rows)


def format_status(cfg: MultivalueConfig, layout: Layout, frame: pd.DataFrame) -> str:
    stage_cols = [c for c in frame if c not in ("arm_id", "family", "kind", "k", "rows")]
    lines = [f"multivalue {cfg.name} ({layout.exp_id})", frame.to_string(index=False)]
    for col in stage_cols:
        counts = Counter(frame[col].astype(str))
        lines.append(f"{col}: " + ", ".join(f"{state}={n}" for state, n in sorted(counts.items())))
    return "\n".join(lines)


def status(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout,
           arms: Sequence[Arm], *, log=print) -> pd.DataFrame:
    frame = status_frame(cfg, cluster, layout, arms)
    log(format_status(cfg, layout, frame))
    return frame
