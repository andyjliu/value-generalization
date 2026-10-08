"""Predictor registry: the VECTOR layer, pure consumers of elicitation artifacts.

Each entry declares which artifact type it ``requires`` and its default data
method — ``valuegen predict --data`` selects any compatible artifact, so
data × predictor combinations are one flag, and a future predictor slots in by
declaring its input type.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

PREDICTOR_SCHEMA_VERSIONS = {
    "persona": 1,
    "weight_steer": 1,
    "sentence_emb": 1,
    "grad_proj": 1,
}


@dataclass(frozen=True)
class Predictor:
    name: str
    requires: str  # artifact type: "pairs" | "descriptions"
    default_data: str  # data method built when --data is omitted


PREDICTORS = {
    "persona": Predictor("persona", requires="pairs", default_data="default_llm"),
    "weight_steer": Predictor("weight_steer", requires="pairs", default_data="default_llm"),
    "sentence_emb": Predictor("sentence_emb", requires="descriptions", default_data="descriptions"),
    "grad_proj": Predictor("grad_proj", requires="pairs", default_data="default_llm"),
}


def get_predictor(name: str) -> Predictor:
    if name not in PREDICTORS:
        raise KeyError(f"Unknown predictor {name!r}; known: {sorted(PREDICTORS)}")
    return PREDICTORS[name]


def get_module(name: str):
    get_predictor(name)
    return importlib.import_module(f"valuegen.predictors.{name}")
