"""Elicitation layer: the DATA side of the predictor stack.

Data generation and vector extraction are separate stages joined only by one
artifact: the **standard pair dataset** (per-value ``{value}_pos.csv`` /
``{value}_neg.csv`` + ``manifest.yaml``, schema in :mod:`datasets`). Every
predictor consumes that schema only — never another method's internal paths —
so data methods and predictors combine freely.

Ownership: the external forks own their methods' generation
math (persona fork: trait/question generation, completion generation, judging
and quality filtering; conflictscope fork: scenario/action machinery). The
registry + adapters here are the umbrella that turns any of them into the one
standard artifact: configuration, orchestration, schema conversion, manifests,
identity/reuse.
"""

from valuegen.elicitation import datasets, registry

__all__ = ["datasets", "registry"]
