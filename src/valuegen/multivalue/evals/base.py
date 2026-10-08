"""The ``EvalSuite`` contract every multivalue suite implements.

A suite is an Inspect task family run against a served candidate by the
shared runner (:mod:`valuegen.multivalue.evals.runner`). The contract
reduces to:

- ``build_inputs``      freeze the suite's inputs once per experiment
                        (e.g. the prefill histories) with a content hash
                        that enters the suite's protocol hash;
- ``attach``            put those inputs on an :class:`inspect_runner.SuiteRun`
                        so the shared generation / completeness code
                        (``generate_units``, ``suite_complete``) sees them;
- ``unit_expectations`` samples per generation unit (for completeness);
- ``scorer``            the cached, failure-flagging grader scorer;
- ``rows``              one long row per candidate response (the row schema);
- ``summarize``         rows -> ``{metric: float}``;
- ``boot_unit``         the row column the paired bootstrap resamples over;
- :func:`subsample`     the shared seeded subsampler.

A *per-candidate* suite (``per_candidate = True``; prefill) answers a
different sample set per candidate: ``select`` turns the frozen inputs, the
candidate's value set and the suite options into a selection record that
``attach``, ``unit_expectations`` and ``boot_units`` take as ``selection``;
its protocol hash covers the selection options (``protocol_extra``); the
paired bootstrap against the base runs on the candidate's own unit list with
the base's rows re-bucketed under the candidate's values (``rebucket``) and
restricted to the candidate's samples (``sample_key``); ``prepare_rows``
lets a suite attach to a row what it needs from the candidate's other rows
(prefill: each injected row's no-history control); and ``derived`` names
functions of the summary metrics -- a difference, a linear combination or a
ratio -- evaluated on the same bootstrap draws (:func:`eval_derived`).

Candidates are described by :class:`Candidate`: what to serve and under which
checkpoint id; the eval stage plans them, the worker looks its own up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from valuegen.ground_truth import inspect_runner as IR
from valuegen.multivalue._hashing import SAMPLE_SALT, order_key
from valuegen.multivalue.config import MultivalueConfig, SuiteConfig
from valuegen.multivalue.sets import Arm

INPUTS_SCHEMA_VERSION = 1


class EvalError(RuntimeError):
    pass


def subsample(ids: Sequence[str], n: int | None, seed: int) -> list[str]:
    """The first ``n`` ids under the seeded sample-hash order, returned in
    the caller's order (so unit order never depends on the seed). ``None`` or
    ``n >= len(ids)`` keeps every id."""
    ids = [str(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise EvalError("duplicate ids in the subsample pool")
    if n is None or n >= len(ids):
        return ids
    chosen = set(sorted(ids, key=lambda i: (order_key(SAMPLE_SALT, int(seed), i), i))[:n])
    return [i for i in ids if i in chosen]


def mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass(frozen=True)
class Candidate:
    """One served checkpoint: the untrained base, a run of this experiment,
    or an imported arm's export."""

    ckpt_id: str
    arm: Arm
    seed: int | None
    kind: str  # base | trained | imported
    model_dir: Path | None  # None: an imported arm registered without a checkpoint
    export_verified: bool
    # "staged": the base copy this package stages under checkpoint_root/base;
    # "imported": a base_model_dirs entry from imports.json; "export": a run's
    # export dir; "import": an imported arm's model_dir.
    model_source: str = "export"
    gcs_uri: str | None = None
    external_suites: dict[str, str] = field(default_factory=dict)  # imported: completed suite dirs
    # trained: the export was deleted after its evals completed (``mv run
    # --delete-exports``); its finished suites stay, nothing new can be served.
    export_deleted: bool = False

    @property
    def served_name(self) -> str:
        return self.ckpt_id

    def to_json(self) -> dict:
        return {"ckpt_id": self.ckpt_id, "arm_id": self.arm.id, "family": self.arm.family, "arm_kind": self.arm.kind,
                "values": list(self.arm.values), "k": self.arm.k, "seed": self.seed, "kind": self.kind,
                "model_dir": None if self.model_dir is None else str(self.model_dir),
                "model_source": self.model_source,
                "export_verified": self.export_verified, "export_deleted": self.export_deleted,
                "gcs_uri": self.gcs_uri, "external_suites": dict(self.external_suites)}


class Suite(ABC):
    name: str
    boot_unit: str
    metrics: tuple[str, ...]  # summary metric names; ``by_x.*`` expands per group
    row_fields: tuple[str, ...]
    options_defaults: Mapping[str, Any] = {}
    smoke_units: int = 2  # generation units kept in smoke mode
    per_candidate: bool = False  # samples depend on the candidate (see ``select``)
    # name -> (metric_a, metric_b)                       : a - b
    # name -> {"terms": {metric: coef, ...}, "const": c}  : sum(coef * metric) + c
    # name -> {"ratio": (num_metric, den_metric)}         : num / den
    # all evaluated on the shared bootstrap draws (:func:`eval_derived`)
    derived: Mapping[str, tuple[str, str] | Mapping[str, Any]] = {}

    def options(self, sc: SuiteConfig) -> dict:
        """Suite options = defaults overridden by the config block's extra keys."""
        unknown = sorted(set(sc.extra) - set(self.options_defaults))
        if unknown:
            raise EvalError(f"evals.suites.{self.name}: unknown keys {unknown}; known: {sorted(self.options_defaults)}")
        return {**dict(self.options_defaults), **sc.extra}

    def params(self, cfg: MultivalueConfig, sc: SuiteConfig) -> dict:
        """The build parameters an inputs file must match to be reused."""
        return {"subsample": dict(sc.subsample) if sc.subsample else None}

    def protocol_extra(self, options: Mapping[str, Any]) -> dict:
        """Suite options that define the protocol (enter the protocol hash)."""
        return {}

    @abstractmethod
    def build_inputs(self, cfg: MultivalueConfig, sc: SuiteConfig, root: Path, *, layout=None) -> dict:
        """Freeze the suite inputs; must carry ``inputs_sha256`` and ``params``.
        ``layout`` (the experiment's :class:`Layout`) is given when known."""

    def select(self, inputs: dict, cand: Candidate | None, options: Mapping[str, Any], seed: int, *,
               smoke: bool = False) -> dict | None:
        """The candidate's sample selection (per-candidate suites); ``None``
        for suites whose samples are the same for every candidate."""
        return None

    @abstractmethod
    def attach(self, run: IR.SuiteRun, inputs: dict, *, smoke: bool = False, cand: Candidate | None = None,
               selection: dict | None = None) -> IR.SuiteRun:
        """Put the frozen inputs (or their smoke subset) on the run."""

    @abstractmethod
    def unit_expectations(self, inputs: dict, *, smoke: bool = False, selection: dict | None = None) -> dict[str, int]:
        """Samples (before repeats) per generation unit id."""

    @abstractmethod
    def scorer(self, run: IR.SuiteRun, grader, cache: IR.GradeCache):
        """The cached grading scorer for one grading pass."""

    @abstractmethod
    def rows(self, run: IR.SuiteRun) -> list[dict]:
        """One row per candidate response (``row_fields``)."""

    @abstractmethod
    def boot_units(self, inputs: dict, *, smoke: bool = False, selection: dict | None = None) -> list[str]:
        """The ``boot_unit`` ids the frozen inputs define (the bootstrap resamples
        over these, paired across candidates; a per-candidate suite's list is
        the candidate's selected units)."""

    def rebucket(self, row: Mapping[str, Any], arm: Arm) -> dict:
        """A row of another candidate re-labelled under ``arm``'s value set
        (per-candidate suites); identity otherwise."""
        return dict(row)

    def sample_key(self, row: Mapping[str, Any]) -> tuple:
        """What identifies a row's sample across candidates (repeats aside)."""
        return (str(row.get(self.boot_unit)),)

    def prepare_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict]:
        """Suite-level row post-processing that needs the candidate's other
        rows (e.g. attaching each injected row's no-history control), run on
        one candidate's (re-bucketed, restricted) rows before ``row_metrics``.
        Default: identity."""
        return [dict(r) for r in rows]

    @abstractmethod
    def row_metrics(self, row: Mapping[str, Any]) -> dict[str, float | None]:
        """One valid row's contribution to every metric it enters (``None``
        where it does not); ``{}`` for an invalid row. Every summary metric
        is the mean of these over the valid rows, which is what makes the
        bootstrap in :mod:`valuegen.multivalue.outcomes` exact."""

    def fixed_metrics(self) -> list[str]:
        return [m for m in self.metrics if not m.endswith(".*")]

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Per-checkpoint scalars over the valid rows (+ ``n_valid``, ``n_rows``)."""
        acc: dict[str, list[float]] = {m: [] for m in self.fixed_metrics()}
        n_valid = 0
        for r in self.prepare_rows(rows):
            m = self.row_metrics(r)
            if not m:
                continue
            n_valid += 1
            for k, v in m.items():
                if v is not None:
                    acc.setdefault(k, []).append(float(v))
        out = {k: mean(v) for k, v in acc.items()}
        out["n_valid"] = float(n_valid)
        out["n_rows"] = float(len(rows))
        return out


def derived_inputs(spec: tuple[str, str] | Mapping[str, Any]) -> list[str]:
    """The summary metrics one ``derived`` spec reads."""
    if isinstance(spec, tuple):
        return list(spec)
    if "terms" in spec:
        return list(spec["terms"])
    if "ratio" in spec:
        return list(spec["ratio"])
    raise EvalError(f"unknown derived spec {spec!r}")


def eval_derived(spec: tuple[str, str] | Mapping[str, Any], values: Mapping[str, Any], *, nan=float("nan")):
    """One derived metric from its inputs (floats or bootstrap arrays, all of
    one shape). ``nan`` (a float or an all-NaN array of that shape) when an
    input is missing; a ratio is NaN where its denominator is 0 or NaN."""
    inputs = derived_inputs(spec)
    if any(m not in values for m in inputs):
        return nan
    if isinstance(spec, tuple):
        a, c = spec
        return values[a] - values[c]
    if "terms" in spec:
        out = float(spec.get("const", 0.0))
        for m, coef in spec["terms"].items():
            out = out + float(coef) * values[m]
        return out
    num, den = spec["ratio"]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.asarray(values[den], dtype=float)
        r = np.where(d == 0, np.nan, np.asarray(values[num], dtype=float) / d)
    return float(r) if np.ndim(r) == 0 else r
