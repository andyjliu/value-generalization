"""Inspect-based evaluation suites for multivalue experiments (spec §4)."""

from __future__ import annotations

from valuegen.multivalue.evals.base import Candidate, EvalError, Suite, subsample
from valuegen.multivalue.evals.prefill import PrefillSuite

SUITES: dict[str, Suite] = {s.name: s for s in (PrefillSuite(),)}


def get_suite(name: str) -> Suite:
    try:
        return SUITES[name]
    except KeyError:
        raise EvalError(f"unknown suite {name!r}; known: {sorted(SUITES)}") from None


__all__ = ["Candidate", "EvalError", "SUITES", "Suite", "get_suite", "subsample"]
