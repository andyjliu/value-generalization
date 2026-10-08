"""Fetch the raw data the paper/reproduce scripts recompute the paper from.

Downloads the HF dataset ``value-generalization/paper-data`` at a pinned
revision into ``data/paper/`` and checks every file against
``MANIFEST.json`` (path -> sha256, size, and where it came from). The
manifest itself is pinned by hash here, so one hash pins the whole set.

What is in it (the GT matrices themselves are in git under ``data/gt/``):

- ``rq1/grids/<arm>/<predictor>.npy``: each predictor's value-similarity grid
  for each of the eight arms (OLMo/Qwen x DPO/SFT at 7-8B and 30-32B), with its ``_values.json`` order;
- ``rq3/``: the VITW-266 persona grid (OLMo-3.1-32B, layer 32), the persona
  vectors at layer 32 for the VITW values and the constitution tenets, the
  published k=4 clusters, the LitmusValues and VITW-L3 tenet labels, and the
  stored scenario bootstrap;
- ``rq2/``: the RQ2 multivalue experiment (rq3_ew64_qwen8b) as ``mv analyze``
  reads it: frozen sets, set metrics, every checkpoint's judged prefill rows
  (one compact table, ``evals/prefill_rows.parquet``), the frozen prefill inputs, the embedding stores, and the cosine tables
  the preregistered analysis read;
- ``evals/``: every GT run's per-scenario judge results in compact form
  (``<gt_id>.parquet``: model, scenario, likert, choice) and
  ``scenarios.parquet`` (scenario id, value1, value2).

Idempotent: files already in place with the right hash are left alone.

    .venvs/core/bin/python scripts/fetch_paper_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = "value-generalization/paper-data"
REVISION = "4bcf183cf325e7a0213a743e40c080b514d12e75"
MANIFEST_SHA256 = "698bff391002e9572a3a0e646e5c8a56fc7474999c12e373d85c2ac3ae0ddde5"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "paper"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(dest: Path, manifest_sha: str | None = MANIFEST_SHA256) -> list[str]:
    """Problems with ``dest`` against its manifest (empty list = all good)."""
    mpath = dest / "MANIFEST.json"
    if not mpath.is_file():
        return [f"{mpath} missing"]
    if manifest_sha is not None and sha256(mpath) != manifest_sha:
        return [f"{mpath} has sha256 {sha256(mpath)}, not the pinned {manifest_sha}"]
    problems = []
    for rel, meta in json.loads(mpath.read_text()).items():
        p = dest / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
        elif sha256(p) != meta["sha256"]:
            problems.append(f"{rel}: sha256 mismatch")
    return problems


def fetch(dest: Path) -> None:
    if REVISION is None:
        raise SystemExit(f"{REPO} has not been published yet (no pinned revision)")
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"directory to fill (default: {DEFAULT_DEST})")
    ap.add_argument("--verify-only", action="store_true",
                    help="check what is already there; download nothing")
    args = ap.parse_args(argv)
    if not args.verify_only and verify(args.dest):
        fetch(args.dest)
    problems = verify(args.dest)
    for p in problems:
        print("  " + p)
    if problems:
        return 1
    n = len(json.loads((args.dest / "MANIFEST.json").read_text()))
    print(f"  ok        {n} files in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
