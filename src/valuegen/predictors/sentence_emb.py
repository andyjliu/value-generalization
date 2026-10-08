"""Sentence-embedding predictor: mpnet-encode value descriptions + cosine.

Requires the ``descriptions`` artifact. Trivial enough that the fork-ownership
rule doesn't apply — this is the whole method. The
"model" key in store paths is the encoder (default ``all-mpnet-base-v2``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D
from valuegen.predictors import store

DEFAULT_ENCODER = "all-mpnet-base-v2"


def needs_slurm(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> bool:
    return False


def plan(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    return [f"sentence_emb encode ({cfg.get('encoder', DEFAULT_ENCODER)}): inline"]


def run(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    dry_run: bool = False,
) -> Path | None:
    encoder = cfg.get("encoder", DEFAULT_ENCODER)
    if dry_run:
        print(f"[dry-run] would encode {artifact.descriptions_path} with {encoder}")
        return None

    import torch
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(artifact.descriptions_path).set_index("value")
    values = cfg["values"]
    missing = [v for v in values if v not in df.index]
    if missing:
        print(f"note: no descriptions for {missing}; their rows/cols are NaN")
    present = [v for v in values if v in df.index]

    model = SentenceTransformer(encoder)
    embeddings = model.encode([str(df.loc[v, "description"]) for v in present])

    vec_dir = store.vectors_dir(
        cluster, "sentence_emb", artifact.method, encoder, artifact.artifact_id
    )
    store.claim_dir(vec_dir, artifact.artifact_id)
    vectors = {}
    for value, emb in zip(present, embeddings):
        tensor = torch.tensor(emb)
        torch.save(tensor, vec_dir / f"{value}_embedding.pt")
        vectors[value] = tensor.numpy()

    sim = store.cosine_grid(vectors, values)
    sim_dir = store.similarity_dir(
        cluster, "sentence_emb", artifact.method, encoder, artifact.artifact_id
    )
    store.claim_dir(sim_dir, artifact.artifact_id)
    out = store.save_similarity(
        sim_dir, "sentence_emb_cos", sim, values,
        identity={"encoder": encoder},
        provenance={
            "predictor": "sentence_emb",
            "data_artifact": artifact.artifact_id,
            "encoder": encoder,
            "resolved_config": dict(cfg),
        },
    )
    print(f"wrote {out}")
    return out
