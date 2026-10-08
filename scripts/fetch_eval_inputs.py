"""Fetch the ConflictScope evaluation inputs every const_v3 config reads.

The configs evaluate on ``data/scenarios/const_v3_cs/`` (``evaluation.scenarios``):

- ``Qwen3.6-27B.csv``: the filtered constitution-v3 scenarios;
- ``cache.json``: the first-turn cache, i.e. the simulated user's opening
  message for every scenario. Each eval task seeds its output dir from it
  (``EvalSpec.cache_seed``), so all runs share the opening turns the paper's
  runs used instead of regenerating them with the user model.

Both come from the public HF dataset
``value-generalization/conflictscope-eval-constitution-tenets-v3`` at a
pinned revision and are checked against the SHA-256 of the files the paper's
runs read. Idempotent: a file already in place with the right hash is left
alone, and a file with the wrong hash is refused (pass ``--force`` to
overwrite it).

    .venvs/core/bin/python scripts/fetch_eval_inputs.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = "value-generalization/conflictscope-eval-constitution-tenets-v3"
REVISION = "1deaaebc088f85df3b9e0e9c6fba37767bf1235f"
# hub path -> (file name under the scenarios dir, sha256)
FILES = {
    "eval/scenarios/Qwen3.6-27B.csv": (
        "Qwen3.6-27B.csv",
        "7aa841d17ade28175c035173ab157f242249eab991458db5645660fae8607670",
    ),
    "eval/cache.json": (
        "cache.json",
        "97ea463656ff7954d5cb82a5dde17d09706760f1b60e7ce70736e10712f98e45",
    ),
}
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "const_v3_cs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path, force: bool = False, download=None) -> list[Path]:
    """Place every file under ``dest``; returns the paths written."""
    if download is None:
        from huggingface_hub import hf_hub_download

        def download(name: str) -> str:
            return hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)

    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for hub_name, (local_name, want) in FILES.items():
        target = dest / local_name
        if target.is_file():
            have = sha256(target)
            if have == want:
                print(f"  ok        {target}")
                continue
            if not force:
                raise SystemExit(
                    f"{target} exists with sha256 {have}, not the pinned {want}; "
                    "move it aside or pass --force to replace it"
                )
        src = Path(download(hub_name))
        got = sha256(src)
        if got != want:
            raise SystemExit(f"{REPO}@{REVISION}:{hub_name} has sha256 {got}, expected {want}")
        tmp = target.with_name(target.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(target)
        written.append(target)
        print(f"  fetched   {target}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"scenarios dir to fill (default: {DEFAULT_DEST})")
    ap.add_argument("--force", action="store_true",
                    help="replace files whose hash does not match")
    args = ap.parse_args(argv)
    fetch(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
