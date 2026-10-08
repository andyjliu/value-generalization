"""Per-candidate outcome scalars with paired bootstrap CIs and deltas vs base
(``scores/outcomes.csv``).

Every suite metric is the mean over valid rows of that row's contribution
(:meth:`Suite.row_metrics`), so a candidate reduces to per-unit sums and
counts over the suite's ``boot_unit``. The bootstrap resamples units with
replacement -- the *same* unit draws for every candidate of a suite (one
seeded index matrix per suite over the frozen inputs' unit list) -- so ``delta = candidate - base`` has a paired
percentile CI. A unit absent from a candidate simply contributes nothing to
that draw.

A per-candidate suite (prefill) has no shared unit list: each candidate is
bootstrapped over its own selected units, and the base is compared on those
same draws after its rows are re-bucketed under the candidate's value set
(``Suite.rebucket``) and restricted to the candidate's samples
(``Suite.sample_key``). ``Suite.prepare_rows`` runs on each candidate's rows
(the base's after re-bucketing and restriction) before ``row_metrics``. A
suite's ``derived`` metrics (a difference, a linear combination or a ratio
of summary metrics; :func:`evals.base.eval_derived`) are evaluated on the
same draws as their inputs and emitted as extra metric rows; their deltas vs
base are differences of the derived values on the shared draws.

Rows come from ``evals/{ckpt}/{suite}/rows.jsonl`` when it exists, else from
an imported arm's compatible external run (read in place; nothing is linked
here). ``complete`` records whether the suite's marker is accepted under the
current protocol; incomplete rows are still summarized but flagged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import eval_stage as ES
from valuegen.multivalue.config import BASE_ARM, MultivalueConfig
from valuegen.multivalue.evals import get_suite
from valuegen.multivalue.evals import runner as R
from valuegen.multivalue.evals.base import Candidate, EvalError, eval_derived
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.sets import Arm

OUTCOME_COLUMNS = ("arm_id", "family", "kind", "seed", "ckpt_id", "suite", "metric", "estimate", "ci_low", "ci_high",
                   "n_valid", "n_rows", "n_units", "complete", "state", "delta_vs_base", "delta_ci_low",
                   "delta_ci_high", "base_ckpt")
BOOT_CHUNK = 200


class AnalyzeError(RuntimeError):
    pass


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_suite_rows(cfg: MultivalueConfig, layout: Layout, cand: Candidate, suite: str,
                       inputs: Mapping[str, Any]) -> tuple[Path | None, str]:
    """``(rows.jsonl, state)`` for a candidate's suite: the layout dir when it
    holds rows, else an imported arm's compatible external run."""
    state = R.suite_state(cfg, layout, cand, suite, inputs)
    root = R.suite_root(layout, cand, suite)
    if state == "reusable":
        root = Path(cand.external_suites[suite])
    p = root / "rows.jsonl"
    return (p if p.is_file() else None), state


@dataclass
class SuiteTerms:
    """One candidate's per-unit sums/counts for every metric."""
    metrics: list[str]
    S: np.ndarray  # [n_units, n_metrics]
    C: np.ndarray
    estimate: dict[str, float]
    n_valid: int
    n_rows: int
    n_units: int  # units with at least one valid row


def unit_terms(suite_name: str, rows: Sequence[Mapping[str, Any]], units: Sequence[str],
               metrics: Sequence[str], *, ignore_unknown: bool = False) -> SuiteTerms:
    """Rows whose unit is outside ``units`` raise, or (``ignore_unknown``) are
    left out -- a per-candidate suite's rows from an earlier selection."""
    suite = get_suite(suite_name)
    upos = {u: i for i, u in enumerate(units)}
    mpos = {m: j for j, m in enumerate(metrics)}
    S = np.zeros((len(units), len(metrics)))
    C = np.zeros_like(S)
    n_valid = 0
    seen = set()
    for r in suite.prepare_rows(rows):
        contrib = suite.row_metrics(r)
        if not contrib:
            continue
        u = str(r.get(suite.boot_unit))
        if u not in upos:
            if ignore_unknown:
                continue
            raise AnalyzeError(f"{suite_name}: row unit {u!r} is not in the frozen inputs' {suite.boot_unit} list")
        n_valid += 1
        seen.add(u)
        for m, v in contrib.items():
            if v is None or m not in mpos:
                continue
            S[upos[u], mpos[m]] += float(v)
            C[upos[u], mpos[m]] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        est = S.sum(0) / C.sum(0)
    return SuiteTerms(list(metrics), S, C, {m: float(est[j]) for j, m in enumerate(metrics)},
                      n_valid, len(rows), len(seen))


def boot_indices(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(n_boot, n_units)) if n_units else np.zeros((n_boot, 0), int)


def bootstrap(terms: SuiteTerms, idx: np.ndarray) -> np.ndarray:
    """``[n_boot, n_metrics]`` resampled means (NaN where a draw has no rows)."""
    out = np.full((idx.shape[0], len(terms.metrics)), np.nan)
    for start in range(0, idx.shape[0], BOOT_CHUNK):
        ix = idx[start:start + BOOT_CHUNK]
        s = terms.S[ix].sum(1)
        c = terms.C[ix].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:start + BOOT_CHUNK] = s / c
    return out


def ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def metric_names(suite_name: str, per_candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    """Fixed metric names first, then every ``by_x.*`` name any candidate produced."""
    suite = get_suite(suite_name)
    names = list(suite.fixed_metrics())
    extra = set()
    for rows in per_candidate_rows.values():
        for r in suite.prepare_rows(rows):
            extra.update(k for k in suite.row_metrics(r) if k not in names)
    return names + sorted(extra)


def base_terms_for(suite_name: str, base_rows: Sequence[Mapping[str, Any]], cand: Candidate,
                   cand_rows: Sequence[Mapping[str, Any]], units: Sequence[str], metrics: Sequence[str]) -> SuiteTerms:
    """The base's terms on a per-candidate suite: its rows re-bucketed under
    ``cand``'s value set and restricted to the samples ``cand`` answered."""
    suite = get_suite(suite_name)
    keys = {suite.sample_key(r) for r in cand_rows}
    rows = [suite.rebucket(r, cand.arm) for r in base_rows if suite.sample_key(r) in keys]
    return unit_terms(suite_name, rows, units, metrics, ignore_unknown=True)


def outcome_rows(cand: Candidate, suite_name: str, metrics: Sequence[str], t: SuiteTerms, b: np.ndarray,
                 tb: SuiteTerms | None, bb: np.ndarray | None, *, done: bool, state: str,
                 base_key: str | None) -> list[dict]:
    """One outcome row per metric (plus the suite's derived metrics) for a
    candidate; deltas against the base's terms/draws when given."""
    suite = get_suite(suite_name)
    mpos = {m: j for j, m in enumerate(metrics)}
    est = dict(t.estimate)
    boot = {m: b[:, j] for m, j in mpos.items()}
    best = dict(tb.estimate) if tb is not None else {}
    bboot = {m: bb[:, j] for m, j in mpos.items()} if bb is not None else {}
    nan_boot = np.full(b.shape[0], np.nan)
    for name, spec in suite.derived.items():
        est[name] = eval_derived(spec, est)
        boot[name] = eval_derived(spec, boot, nan=nan_boot)
        if tb is not None:
            best[name] = eval_derived(spec, best)
            bboot[name] = eval_derived(spec, bboot, nan=nan_boot)
    out = []
    for m in [*metrics, *suite.derived]:
        lo, hi = ci(boot[m])
        row = {"arm_id": cand.arm.id, "family": cand.arm.family, "kind": cand.kind, "seed": cand.seed,
               "ckpt_id": cand.ckpt_id, "suite": suite_name, "metric": m, "estimate": est[m],
               "ci_low": lo, "ci_high": hi, "n_valid": t.n_valid, "n_rows": t.n_rows, "n_units": t.n_units,
               "complete": done, "state": state,
               "delta_vs_base": float("nan"), "delta_ci_low": float("nan"), "delta_ci_high": float("nan"),
               "base_ckpt": base_key}
        if tb is not None:
            dlo, dhi = ci(boot[m] - bboot[m])
            row.update(delta_vs_base=est[m] - best[m], delta_ci_low=dlo, delta_ci_high=dhi)
        out.append(row)
    return out


def compute_outcomes(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout, arms: Sequence[Arm], *,
                     suites: Sequence[str] | None = None, n_boot: int | None = None, seed: int | None = None,
                     log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(outcomes, completeness)``: one outcome row per candidate x suite x
    metric; one completeness row per candidate x suite (state, rows path)."""
    names = list(suites or cfg.evals.enabled())
    n_boot = int(n_boot or cfg.analysis.n_boot)
    seed = cfg.seed if seed is None else int(seed)
    cands = ES.plan_candidates(cfg, cluster, layout, arms)
    base = next((c for c in cands if c.kind == "base"), None)
    out_rows: list[dict] = []
    completeness: list[dict] = []
    for s_i, suite_name in enumerate(names):
        suite = get_suite(suite_name)
        ip = R.inputs_path(layout, suite_name)
        if not ip.is_file():
            raise AnalyzeError(f"no frozen inputs for {suite_name} at {ip}: run `valuegen mv eval --build {suite_name}`")
        inputs = mvdata.read_json(ip)
        rows_by: dict[str, list[dict]] = {}
        states: dict[str, str] = {}
        for c in cands:
            path, state = resolve_suite_rows(cfg, layout, c, suite_name, inputs)
            states[c.ckpt_id] = state
            completeness.append({"ckpt_id": c.ckpt_id, "arm_id": c.arm.id, "kind": c.kind, "suite": suite_name,
                                 "state": state, "rows": str(path) if path else None})
            if path is not None:
                rows_by[c.ckpt_id] = read_rows(path)
        if not rows_by:
            log(f"{suite_name}: no rows for any candidate")
            continue
        metrics = metric_names(suite_name, rows_by)
        base_key = base.ckpt_id if base is not None and base.ckpt_id in rows_by else None
        boot_seed = seed * 1000 + s_i
        if suite.per_candidate:
            opts = suite.options(cfg.evals.suites[suite_name])
            log(f"{suite_name}: {len(rows_by)} candidates with rows (per-candidate units), {len(metrics)} metrics, "
                f"{n_boot} bootstrap draws" + ("" if base_key else " (no base rows: deltas are NaN)"))
            for c in cands:
                rows = rows_by.get(c.ckpt_id)
                if rows is None:
                    continue
                sel = suite.select(inputs, c, opts, cfg.seed)
                units = suite.boot_units(inputs, selection=sel)
                idx = boot_indices(len(units), n_boot, boot_seed)
                t = unit_terms(suite_name, rows, units, metrics, ignore_unknown=True)
                tb = bb = None
                if base_key and c.ckpt_id != base_key:
                    tb = base_terms_for(suite_name, rows_by[base_key], c, rows, units, metrics)
                    bb = bootstrap(tb, idx)
                out_rows += outcome_rows(c, suite_name, metrics, t, bootstrap(t, idx), tb, bb,
                                         done=states[c.ckpt_id] in ("done", "reusable"), state=states[c.ckpt_id],
                                         base_key=base_key)
            continue
        units = suite.boot_units(inputs)
        idx = boot_indices(len(units), n_boot, boot_seed)
        terms = {k: unit_terms(suite_name, rows, units, metrics) for k, rows in rows_by.items()}
        boots = {k: bootstrap(t, idx) for k, t in terms.items()}
        log(f"{suite_name}: {len(terms)} candidates with rows, {len(units)} units, {len(metrics)} metrics, "
            f"{n_boot} bootstrap draws" + ("" if base_key else " (no base rows: deltas are NaN)"))
        for c in cands:
            t = terms.get(c.ckpt_id)
            if t is None:
                continue
            paired = bool(base_key and c.ckpt_id != base_key)
            out_rows += outcome_rows(c, suite_name, metrics, t, boots[c.ckpt_id],
                                     terms[base_key] if paired else None, boots[base_key] if paired else None,
                                     done=states[c.ckpt_id] in ("done", "reusable"), state=states[c.ckpt_id],
                                     base_key=base_key)
    outcomes = pd.DataFrame(out_rows, columns=list(OUTCOME_COLUMNS))
    return outcomes, pd.DataFrame(completeness, columns=["ckpt_id", "arm_id", "kind", "suite", "state", "rows"])


def write_outcomes(layout: Layout, outcomes: pd.DataFrame, completeness: pd.DataFrame) -> Path:
    layout.scores_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(layout.scores_dir / "outcomes.csv", index=False)
    completeness.to_csv(layout.scores_dir / "completeness.csv", index=False)
    return layout.scores_dir / "outcomes.csv"
