"""``mv metrics``: coverage and tightness per arm per store -> ``metrics/metrics.csv``.

One row per (arm, store) with ``coverage``, ``tightness``, ``k`` and
``rows`` (the fixed budget; 0 for the untrained base; an imported arm's own
row count). The base arm has no values, so its metrics are NaN; the ``full``
reference arm scores the whole universe; imported arms (``imports.json``)
are scored alongside the frozen ones. The CSV also records the frozen design's ``arms_sha256`` so a
later analysis can refuse a stale metrics file.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import sets as mvsets
from valuegen.multivalue.config import MultivalueConfig
from valuegen.multivalue.embeddings import Store, load_stores
from valuegen.multivalue.layout import Layout
from valuegen.multivalue.metrics import set_metrics


def arm_metrics(arms: Sequence[mvsets.Arm], stores: Mapping[str, Store], rows: int) -> pd.DataFrame:
    out = []
    for a in arms:
        for name, s in stores.items():
            if a.values:
                m = set_metrics(a.values, s.universe_names, s.universe, s.external)
            else:
                m = {"coverage": float("nan"), "tightness": float("nan"), "k": 0}
            out.append({"arm_id": a.id, "family": a.family, "kind": a.kind, "store": name,
                        "coverage": m["coverage"], "tightness": m["tightness"],
                        "k": m["k"], "rows": (a.extra.get("rows", rows) if a.trained else 0),
                        "sampling_score": a.score, "bin": a.bin})
    return pd.DataFrame(out)


def write_metrics(cfg: MultivalueConfig, cluster: ClusterConfig, layout: Layout) -> pd.DataFrame:
    record = mvsets.load_sets(layout.sets_path)
    universe = list(record["universe"])
    external, _ = mvdata.load_external(cfg)
    stores = load_stores(cfg, universe, external)
    arms = mvimports.all_arms(layout, record)
    df = arm_metrics(arms, stores, cfg.budget.rows)
    df["arms_sha256"] = record["arms_sha256"]
    layout.metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(layout.metrics_csv, index=False)
    return df
