"""Evaluation methods: score an intervention manifest into a GT artifact.

Each eval method module exposes the same surface:

- ``stages(cfg, cluster, manifest) -> list[Stage]`` — the SLURM stages that
  evaluate every manifest entry (plus the base model — the N+1 rule) under
  the experiment's ``evaluation:`` block.
- ``build_ground_truth(cfg, cluster, manifest) -> list[Path]`` — turn the
  finished eval outputs into the method's GT artifact (steerability matrices
  for the ConflictScope scenario eval; a future document-level eval would
  write its own scores table — never through ``matrices.py``, which has
  exactly one cell estimator).

Methods consume the manifest only — never a training method's internals —
which is what keeps any intervention evaluable under any eval method.
"""

from __future__ import annotations

import importlib

EVAL_METHODS = ("conflictscope",)


def get_eval(method: str):
    """Eval method name -> driver module (lazy import)."""
    if method not in EVAL_METHODS:
        raise KeyError(f"Unknown eval method {method!r}; known: {EVAL_METHODS}")
    return importlib.import_module(f"valuegen.ground_truth.evals.{method}")
