"""Remote checkpoint storage on GCS (opt-in via the ``gcs:`` cluster block).

Merged checkpoints are the space problem: one full model copy per
(model, value) cell, on clusters whose shared filesystem can't hold them.
When ``cluster.gcs`` is set, a training task pushes its merged dir to
``{bucket}/merged/{name}`` right after the merge and deletes the local copy,
leaving a one-line *marker* file (``<merged>.gcs``, holding the gs:// URI)
where the dir was. Eval tasks that need a remote checkpoint stage it into
node-local scratch (``gcs.stage_dir``) at run time and load it from there.

Like ``evaluation.wait_ready_sh``, everything here renders shell for SLURM
tasks — no network at render time. Done-checks and manifest readiness stay
local-filesystem-only: the marker, not a ``gcloud storage ls``, is the
evidence of a completed upload (it is written only after rsync succeeds, and
the local dir is deleted only after the marker is written, so a failed upload
leaves the task retryable with the local copy intact).

The staged path keeps the merged dir's *basename*: conflictscope selects the
chat template by substring of the model path (see ``training.py``), so a
staging scheme that renamed the dir would silently serve the wrong template.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from valuegen.config import GcsConfig


def remote_uri(gcs: GcsConfig, merged_path: str | Path) -> str:
    return f"{gcs.bucket}/merged/{Path(merged_path).name}"


def marker_path(merged_path: str | Path) -> Path:
    merged_path = Path(merged_path)
    return merged_path.with_name(merged_path.name + ".gcs")


def staged_path(gcs: GcsConfig, merged_path: str | Path) -> Path:
    """Deterministic node-local landing spot; a warm copy makes the stage-in
    rsync a cheap no-op for the next eval task on the same node. The training
    side writes its finished artifact here too (see ``train_stages``), so the
    push rsyncs straight from scratch and nothing model-sized ever touches
    the shared filesystem the ``gcs:`` block exists to spare."""
    return gcs.stage_dir / "merged" / Path(merged_path).name


def deleted_marker_path(merged_path: str | Path) -> Path:
    """Tombstone left by ``interventions.delete_checkpoint`` (``gt run
    --delete-exports``): a sibling of the merged dir, like the upload marker."""
    merged_path = Path(merged_path)
    return merged_path.with_name(merged_path.name + ".deleted")


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def checkpoint_loadable(merged_path: str | Path) -> bool:
    """An eval can load it: on disk — or uploaded (marker present)."""
    return _nonempty(Path(merged_path) / "config.json") or _nonempty(
        marker_path(merged_path)
    )


def checkpoint_trained(merged_path: str | Path) -> bool:
    """Training is finished: loadable — or deleted on purpose after its evals
    completed (tombstone), which must never read as "retrain me"."""
    return checkpoint_loadable(merged_path) or _nonempty(
        deleted_marker_path(merged_path)
    )


def ready(merged_path: str | Path) -> bool:
    """A checkpoint is ready if it's on disk — or uploaded (marker present)."""
    return checkpoint_loadable(merged_path)


def push_and_prune_sh(
    gcs: GcsConfig, merged_path: str | Path, local_dir: str | Path | None = None
) -> str:
    """Shell tail for a train task: upload the finished artifact, then free it.

    ``local_dir`` is where the finishing step actually wrote the artifact.
    It defaults to ``merged_path`` (the historical layout); ``train_stages``
    points it at node-local scratch so the shared filesystem never holds a
    model-sized transient. The marker stays at the canonical ``merged_path``
    location either way -- it is the durable done evidence, and its parent is
    created here because with a scratch ``local_dir`` nothing else does.

    Order is load-bearing: rsync, then marker, then rm — each step only runs
    if the previous succeeded, so any interruption leaves either the local
    dir (retry re-uploads) or a completed upload with its marker.
    """
    local = shlex.quote(str(local_dir if local_dir is not None else merged_path))
    uri = shlex.quote(remote_uri(gcs, merged_path))
    marker = shlex.quote(str(marker_path(merged_path)))
    marker_parent = shlex.quote(str(Path(merged_path).parent))
    return f"""if [ ! -s {marker} ]; then
    echo "[$(date)] Pushing {Path(merged_path).name} to GCS ..."
    gcloud storage rsync --recursive --delete-unmatched-destination-objects \\
        {local} {uri}
    mkdir -p {marker_parent}
    echo {uri} > {marker}
    rm -rf {local}
    echo "[$(date)] Pushed and pruned {Path(merged_path).name}"
fi"""


def guarded_train_sh(train_and_merge: str, merged_path: str | Path) -> str:
    """Skip train+merge when the checkpoint already exists locally or remotely.

    Without GCS the orchestrator's done-check (merged/config.json) keeps a
    finished task from resubmitting; with push-and-prune the done evidence is
    the marker, and a retry after a failed *upload* must not retrain — the
    guard extends the merge subcommand's own idempotence to the train step.
    """
    merged = Path(merged_path)
    config = shlex.quote(str(merged / "config.json"))
    marker = shlex.quote(str(marker_path(merged)))
    return f"""if [ ! -s {marker} ] && [ ! -f {config} ]; then
{train_and_merge}
fi"""


def stage_in_sh(gcs: GcsConfig, merged_path: str | Path) -> str:
    """Eval-task prelude: point ``$MODEL_DIR`` at a loadable checkpoint.

    The branch runs at *run* time, not render time — eval jobs can be
    submitted with dependency chaining before training has finished, so
    whether the checkpoint is still local or already pruned to GCS is
    unknowable when the script is rendered.
    """
    local = shlex.quote(str(merged_path))
    uri = shlex.quote(remote_uri(gcs, merged_path))
    staged = shlex.quote(str(staged_path(gcs, merged_path)))
    return f"""if [ -f {local}/config.json ]; then
    MODEL_DIR={local}
else
    echo "[$(date)] Staging {Path(merged_path).name} from GCS ..."
    mkdir -p {staged}
    gcloud storage rsync --recursive {uri} {staged}
    MODEL_DIR={staged}
    echo "[$(date)] Staged to ${{MODEL_DIR}}"
fi"""
