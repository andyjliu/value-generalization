"""``descriptions``: the trivial artifact — value descriptions as data.

What the old ``methods/none.py`` produced, promoted to a first-class artifact
type. Sentence-embedding predictors declare ``requires: descriptions``.
"""

from __future__ import annotations

import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D


def build(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    only_values: list[str] | None = None,
) -> None:
    from valuegen.values import load_value_set

    value_dict = load_value_set(cfg["value_set"])
    missing = [v for v in cfg["values"] if v not in value_dict]
    if missing:
        raise KeyError(f"values {missing} not in value set {cfg['value_set']}")
    artifact.root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"value": v, "description": value_dict[v]} for v in cfg["values"]]
    ).to_csv(artifact.descriptions_path, index=False)
    print(f"  wrote {artifact.descriptions_path} ({len(cfg['values'])} values)")
