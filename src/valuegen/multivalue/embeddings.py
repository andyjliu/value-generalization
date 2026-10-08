"""Embedding stores: ``{value: vector}`` files aligned to value sets.

Accepted file types (all resolve to ``{name: 1-D float vector}``):

- ``.npz`` with ``labels`` + ``embeddings``/``vectors`` arrays, or one array per value;
- ``.npy`` matrix with a sidecar ``<stem>.labels.json`` list (or a pickled dict);
- ``.pt`` dict of tensors;
- ``.json`` dict of lists.

A vector may be 2-D ``[n_layers, hidden]`` (persona ``response_avg_diff``);
then the store config must give ``layer`` and that row is taken. Every value
of the universe and of the external set must be present in the respective
file; a miss is an error (embeddings are inputs, never computed here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from valuegen.multivalue.config import MultivalueConfig, StoreConfig


class EmbeddingError(RuntimeError):
    pass


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_vectors(path: str | Path) -> dict[str, np.ndarray]:
    """Load ``{name: float array}`` from any supported file type (no layer selection)."""
    path = Path(path)
    if not path.is_file():
        raise EmbeddingError(f"no embedding file at {path}")
    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(path, allow_pickle=True)
        if "labels" in z.files and ("embeddings" in z.files or "vectors" in z.files):
            labels = [str(x) for x in z["labels"]]
            mat = np.asarray(z["embeddings" if "embeddings" in z.files else "vectors"], dtype=float)
            if len(labels) != mat.shape[0]:
                raise EmbeddingError(f"{path}: {len(labels)} labels vs {mat.shape[0]} rows")
            return {k: mat[i] for i, k in enumerate(labels)}
        return {k: np.asarray(z[k], dtype=float) for k in z.files}
    if suf == ".npy":
        obj = np.load(path, allow_pickle=True)
        if obj.dtype == object and obj.shape == ():
            return {str(k): np.asarray(v, dtype=float) for k, v in obj.item().items()}
        sidecar = path.with_suffix(".labels.json")
        if not sidecar.is_file():
            raise EmbeddingError(f"{path}: matrix needs a sidecar label list at {sidecar}")
        labels = _read_json(sidecar)
        if len(labels) != obj.shape[0]:
            raise EmbeddingError(f"{path}: {len(labels)} labels vs {obj.shape[0]} rows")
        return {str(k): np.asarray(obj[i], dtype=float) for i, k in enumerate(labels)}
    if suf in (".pt", ".pth"):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise EmbeddingError(f"{path}: expected a dict of tensors")
        return {str(k): np.asarray(torch.as_tensor(v).float().numpy(), dtype=float) for k, v in obj.items()}
    if suf == ".json":
        obj = _read_json(path)
        if not isinstance(obj, dict):
            raise EmbeddingError(f"{path}: expected a JSON object {{name: vector}}")
        return {str(k): np.asarray(v, dtype=float) for k, v in obj.items()}
    raise EmbeddingError(f"unsupported embedding file type {suf!r}: {path}")


def _select_layer(vecs: Mapping[str, np.ndarray], layer: int | None, where: str) -> dict[str, np.ndarray]:
    out = {}
    for k, v in vecs.items():
        v = np.asarray(v, dtype=float)
        if v.ndim == 2:
            if layer is None:
                raise EmbeddingError(f"{where}: vectors are 2-D {v.shape}; set `layer:` on the store")
            if layer >= v.shape[0]:
                raise EmbeddingError(f"{where}: layer {layer} out of range for shape {v.shape}")
            v = v[layer]
        elif v.ndim != 1:
            raise EmbeddingError(f"{where}: vector for {k!r} has shape {v.shape}, expected 1-D or 2-D")
        elif layer is not None:
            raise EmbeddingError(f"{where}: `layer:` set but vectors are 1-D")
        out[k] = v
    dims = {v.shape[0] for v in out.values()}
    if len(dims) > 1:
        raise EmbeddingError(f"{where}: mixed vector dimensions {sorted(dims)}")
    return out


def align(vecs: Mapping[str, np.ndarray], names: Sequence[str], where: str) -> np.ndarray:
    """Stack ``vecs[name]`` in ``names`` order as ``[n, d]``; missing names raise."""
    missing = [n for n in names if n not in vecs]
    if missing:
        raise EmbeddingError(
            f"{where}: {len(missing)} of {len(names)} values have no vector, e.g. {missing[:5]}"
        )
    if not names:
        raise EmbeddingError(f"{where}: empty name list")
    return np.vstack([vecs[n] for n in names]).astype(float)


@dataclass(frozen=True)
class Store:
    """One embedding store aligned to the universe and external value lists."""

    name: str
    universe_names: tuple[str, ...]
    external_names: tuple[str, ...]
    universe: np.ndarray  # [n_universe, d]
    external: np.ndarray  # [n_external, d]
    model: str | None
    layer: int | None

    @property
    def dim(self) -> int:
        return int(self.universe.shape[1])

    def index(self, values: Sequence[str]) -> np.ndarray:
        pos = {v: i for i, v in enumerate(self.universe_names)}
        try:
            return np.asarray([pos[v] for v in values], dtype=int)
        except KeyError as e:
            raise EmbeddingError(f"{self.name}: value {e.args[0]!r} is not in the universe") from None


def load_store(spec: StoreConfig, universe_names: Sequence[str], external_names: Sequence[str]) -> Store:
    uni = _select_layer(load_vectors(spec.universe), spec.layer, f"embeddings.{spec.name}.universe")
    ext = _select_layer(load_vectors(spec.external), spec.layer, f"embeddings.{spec.name}.external")
    U = align(uni, universe_names, f"embeddings.{spec.name}.universe")
    E = align(ext, external_names, f"embeddings.{spec.name}.external")
    if U.shape[1] != E.shape[1]:
        raise EmbeddingError(
            f"embeddings.{spec.name}: universe dim {U.shape[1]} != external dim {E.shape[1]}"
        )
    return Store(name=spec.name, universe_names=tuple(universe_names), external_names=tuple(external_names),
                 universe=U, external=E, model=spec.model, layer=spec.layer)


def load_stores(cfg: MultivalueConfig, universe_names: Sequence[str], external_names: Sequence[str],
                only: Sequence[str] | None = None) -> dict[str, Store]:
    names = list(only) if only else list(cfg.embeddings)
    for n in names:
        if n not in cfg.embeddings:
            raise EmbeddingError(f"no embedding store {n!r} in config; known: {sorted(cfg.embeddings)}")
    return {n: load_store(cfg.embeddings[n], universe_names, external_names) for n in names}
