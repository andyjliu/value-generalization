"""Vector store + similarity-grid save/load, keyed (predictor, data_method, model).

One store for every predictor. Contracts:

- vectors:    ``data/vectors/{predictor}/{data_method}/{model_short}/{value}_{kind}.pt``
  (``[n_layers, hidden]`` for activation predictors, ``[dim]`` for sentence
  embeddings; weight steering stores similarity matrices directly).
- similarity: ``data/similarity/{predictor}/{data_method}/{model_short}/{name}.npy``
  + ``{name}_values.json`` — the same convention as GT matrices, so
  ``analysis/correlate`` treats them interchangeably. Symmetric, with NaN
  rows/cols for values that have no vector (e.g. ad_12).

The path key deliberately uses the *data method*, not the artifact hash (the
§3 contract). To keep "never silently overwrite" intact anyway, each keyed
directory carries a ``manifest.yaml`` recording the source artifact ID;
writing under the same key from a *different* artifact raises instead of
clobbering — pass/clean explicitly if that is really intended.

Predictor outputs additionally nest under the source **artifact ID** — ``.../{model_short}/{artifact_id}/`` — so different value
sets / subsets / data configs coexist in one store instead of tripping the
manifest guard. Layer/pooling/encoder variants remain distinguished by file
name within a directory, and the manifest guard stays as a belt-and-braces
check. Older directories without the artifact level are still readable by
explicit path.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import yaml

from valuegen.config import ClusterConfig
from valuegen.ground_truth import matrices as M

MANIFEST = "manifest.yaml"


def _short(model: str) -> str:
    return model.split("/")[-1]


def vectors_dir(
    cluster: ClusterConfig,
    predictor: str,
    data_method: str,
    model: str,
    artifact_id: str | None = None,
) -> Path:
    base = cluster.data / "vectors" / predictor / data_method / _short(model)
    return base / artifact_id if artifact_id else base


def similarity_dir(
    cluster: ClusterConfig,
    predictor: str,
    data_method: str,
    model: str,
    artifact_id: str | None = None,
) -> Path:
    base = cluster.data / "similarity" / predictor / data_method / _short(model)
    return base / artifact_id if artifact_id else base


def sweeps_dir(
    cluster: ClusterConfig,
    predictor: str,
    data_method: str,
    model: str,
    artifact_id: str | None = None,
) -> Path:
    """Layer sweeps: keyed like vectors, since a sweep steers *these* vectors."""
    base = cluster.data / "sweeps" / predictor / data_method / _short(model)
    return base / artifact_id if artifact_id else base


def claim_dir(directory: Path, source_artifact_id: str, extra: dict | None = None) -> None:
    """Bind a keyed store directory to its source artifact (refuse mismatches).

    ``extra`` keys are part of the claim, not just provenance: a sweep keyed by
    its pairs artifact is *also* only valid for the eval pool it steered on, and
    silently mixing two pools would make layer choices incomparable.

    ``extra["immutable"]`` marks bytes that cannot be regenerated from what this
    repo has: imported legacy artifacts whose inputs are gone or whose production is
    non-deterministic (a layer sweep re-generates and re-judges; weight steering
    retrains LoRAs). Same-artifact rewrites are otherwise *allowed* — that is
    what makes a build resumable — so without this a rerun under the recovered
    config would happily overwrite the only copy of the old numbers. Only a
    caller that declares ``immutable`` itself (i.e. an import) may reclaim
    such a directory; everything else must delete it deliberately.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST
    if path.is_file():
        record = yaml.safe_load(path.read_text()) or {}
        if record.get("immutable") and not (extra or {}).get("immutable"):
            raise RuntimeError(
                f"{directory} is marked immutable — it holds imported artifacts "
                "that this repo cannot reproduce byte-for-byte. Refusing to "
                "overwrite. Write to a fresh artifact ID, or delete the "
                "directory if you really mean to discard the imported data."
            )
        existing = record.get("source_artifact_id")
        if existing != source_artifact_id:
            raise RuntimeError(
                f"{directory} was built from artifact {existing!r}, not "
                f"{source_artifact_id!r}. Refusing to overwrite — remove the "
                "directory or pin --data to the original artifact."
            )
        for key, value in (extra or {}).items():
            if key in record and record[key] != value:
                raise RuntimeError(
                    f"{directory} was built with {key}={record[key]!r}, not "
                    f"{value!r}. Refusing to overwrite — remove the directory "
                    "if you really mean to rebuild it from other inputs."
                )
        return
    path.write_text(yaml.safe_dump({
        "source_artifact_id": source_artifact_id,
        "date": datetime.date.today().isoformat(),
        **(extra or {}),
    }, sort_keys=False))


def read_claim(directory: Path) -> dict | None:
    path = Path(directory) / MANIFEST
    if not path.is_file():
        return None
    record = yaml.safe_load(path.read_text())
    return record if isinstance(record, dict) else None


# ── Similarity grids ─────────────────────────────────────────────────────────


def cosine_grid(vectors: dict[str, np.ndarray], values: list[str]) -> np.ndarray:
    """Symmetric cosine matrix over ``values``; NaN rows/cols where a value
    has no vector (the old notebooks' NaN-pad convention, e.g. ad_12)."""
    n = len(values)
    sim = np.full((n, n), np.nan)
    keyed = {}
    for value in values:
        vec = vectors.get(value)
        if vec is None:
            continue
        vec = np.asarray(vec, dtype=np.float64).ravel()
        norm = np.linalg.norm(vec)
        if norm == 0 or not np.isfinite(norm):
            continue
        keyed[value] = vec / norm
    for i, vi in enumerate(values):
        for j, vj in enumerate(values):
            if vi in keyed and vj in keyed:
                sim[i, j] = float(keyed[vi] @ keyed[vj])
    return sim


def save_similarity(
    directory: Path,
    name: str,
    matrix: np.ndarray,
    values: list[str],
    provenance: dict | None = None,
    identity: dict | None = None,
) -> Path:
    """``{name}.npy`` + ``{name}_values.json`` + ``{name}_provenance.yaml``.

    ``identity`` is the predictor config that actually changes the cells (a
    pooling, a layer, an encoder, a LoRA recipe — never ``values``, which
    legitimately varies by subset, nor scheduling knobs like ``gpus``). The
    ``{artifact_id}`` path level separates *data* configs; this separates
    *predictor* configs within one artifact, for the predictors whose matrix
    name is a constant. Rewriting a name under a different identity is refused:
    an identical rerun still overwrites, so resume is unaffected.
    """
    path = Path(directory) / f"{name}_provenance.yaml"
    if identity is not None and path.is_file():
        record = yaml.safe_load(path.read_text()) or {}
        existing = record.get("identity")
        if existing is not None and existing != identity:
            raise RuntimeError(
                f"{directory}/{name} was built with {existing!r}, not "
                f"{identity!r}. Refusing to overwrite — give the matrix a "
                "distinct name, or remove it if you really mean to rebuild it "
                "under a different predictor config."
            )

    M.save_matrix(directory, name, matrix, values)
    record = {
        "matrix": name,
        "date": datetime.date.today().isoformat(),
        **({"identity": identity} if identity is not None else {}),
        **(provenance or {}),
    }
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return Path(directory) / f"{name}.npy"


def load_similarity(path: str | Path) -> tuple[np.ndarray, list[str]]:
    matrix, rows, cols = M.load_matrix(Path(path))
    if rows != cols:
        raise ValueError(f"similarity matrix at {path} is not square-keyed")
    return matrix, rows


def nan_pad(
    matrix: np.ndarray, values: list[str], full_values: list[str]
) -> np.ndarray:
    """Reindex a similarity grid onto ``full_values`` with NaN for missing."""
    idx = {v: i for i, v in enumerate(values)}
    n = len(full_values)
    out = np.full((n, n), np.nan)
    for i, vi in enumerate(full_values):
        if vi not in idx:
            continue
        for j, vj in enumerate(full_values):
            if vj in idx:
                out[i, j] = matrix[idx[vi], idx[vj]]
    return out


def load_vectors(
    directory: Path,
    values: list[str],
    kind: str,
    layer: int | None = None,
    name_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Load ``{value}_{kind}.pt`` vectors; slice ``layer`` for stacked tensors.

    Missing values are skipped (callers NaN-pad); a 1-D tensor ignores
    ``layer`` (sentence embeddings). ``name_map`` translates value names to
    legacy on-disk stems (old persona extractions used trait names like
    ``helpful`` for ``ad_0``); keys of the returned dict are always the
    canonical value names.
    """
    import torch

    out = {}
    for value in values:
        stem = (name_map or {}).get(value, value)
        path = Path(directory) / f"{stem}_{kind}.pt"
        if not path.is_file():
            continue
        tensor = torch.load(path, weights_only=False)
        if tensor.dim() == 1:
            out[value] = tensor.float().numpy()
        else:
            if layer is None:
                raise ValueError(
                    f"{path} is [n_layers, hidden]; a layer is required "
                    "(pass --layer or provide a fork layer sweep)"
                )
            out[value] = tensor[layer].float().numpy()
    return out


def load_stacked_vectors(
    directory: Path,
    values: list[str],
    kind: str,
    name_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Full ``[n_layers, hidden]`` arrays per value (layer sweeps); missing
    values are skipped, same conventions as :func:`load_vectors`."""
    import torch

    out = {}
    for value in values:
        stem = (name_map or {}).get(value, value)
        path = Path(directory) / f"{stem}_{kind}.pt"
        if not path.is_file():
            continue
        out[value] = torch.load(path, weights_only=False).float().numpy()
    return out
