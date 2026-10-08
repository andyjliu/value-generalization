"""Scenario pools: a shared generate → filter → dedup → split artifact.

A pool is the complete ConflictScope scenario corpus for a value set,
deduplicated globally and split once into a train half (DPO-pair construction
— the ``conflictscope_pairs`` intervention) and a test half (steerability
evaluation — the ``conflictscope`` eval method). Promoting it out of the old
coupled driver is what breaks the circularity: an experiment can train on a
pool's train half and evaluate anywhere else, or evaluate on a pool's test
half against interventions trained on entirely different data. The
immutable-train dedup flow (``dedup.existing_train``) still guarantees a
fresh eval half is disjoint from a pre-existing training corpus.

Pools are their own artifact, keyed by identity like label stores::

    data/scenarios/{value_set}/{pool_id}/
        resolved_config.yaml   # config_id = pool_id, resolved = pool identity
        candidates/            # raw + filtered generation output
        full/                  # globally deduplicated (dedup_report.yaml)
        train/  test/          # the split (split_report.yaml in test/)

``pool_id`` hashes the scenario-affecting knobs only (gen/filter model,
counts, temperature, dedup, split — see ``config.pool_identity``); scheduling
knobs move nothing. Plain scenario dirs (e.g. ``data/scenarios/const_v3_cs``)
are not pools — they stay path-referenced and never enter this store.

Pure-python steps run as SLURM cpu tasks::

    python -m valuegen.ground_truth.scenario_pool dedup --pool-dir <dir>
    python -m valuegen.ground_truth.scenario_pool split --pool-dir <dir>
"""

from __future__ import annotations

import shlex
from pathlib import Path

from valuegen import values as V
from valuegen._external import CONFLICTSCOPE_SRC
from valuegen.config import (
    ClusterConfig,
    ensure_config_record,
    pool_id,
    pool_identity,
    resolve_pool,
)
from valuegen.slurm import Stage, Task

GENERATE_SCENARIOS = CONFLICTSCOPE_SRC / "generate_scenarios.py"
FILTER_SCENARIOS = CONFLICTSCOPE_SRC / "filter_scenarios.py"


# ── Dedup / split machinery (moved verbatim from the old coupled driver) ─────


def _single_csv(source: str | Path) -> Path:
    source = Path(source)
    if source.is_file():
        return source
    outputs = source / "outputs" if (source / "outputs").is_dir() else source
    files = sorted(outputs.glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one CSV under {outputs}, found {files}")
    return files[0]


def _scenario_text(row) -> str:
    import math
    import numbers

    fields = []
    for key in ("description", "context", "user_prompt", "action1", "action2"):
        value = row.get(key, "")
        if value is None or (isinstance(value, numbers.Real) and math.isnan(value)):
            value = ""
        fields.append(str(value))
    return " ".join(fields)


def _write_scenario_json_subset(
    source_dir: Path, destination_dir: Path, keep_ids: set[str]
) -> None:
    """Mirror ConflictScope JSON files, retaining only selected CSV IDs."""
    import json

    if not source_dir.is_dir():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for json_file in sorted(source_dir.glob("*.json")):
        scenarios = json.loads(json_file.read_text())
        prefix = json_file.stem + "_"
        kept = {key: value for key, value in scenarios.items() if prefix + key in keep_ids}
        (destination_dir / json_file.name).write_text(json.dumps(kept, indent=2))


def global_deduplicate(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    embedding_model: str = "all-MiniLM-L6-v2",
    threshold: float = 0.9,
    existing_train: list[str | Path] = (),
    encoder=None,
) -> dict:
    """Deduplicate candidates globally, optionally against immutable training data.

    Candidate order is preserved, so the first scenario wins deterministically
    across all value pairs. Existing training rows seed the similarity index but
    are never modified. ``encoder`` is injectable for deterministic unit tests.
    """
    import numpy as np
    import pandas as pd
    import yaml

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    candidate_csv = _single_csv(input_dir)
    candidates = pd.read_csv(candidate_csv)
    if candidates["scenario_id"].duplicated().any():
        dupes = candidates.loc[candidates["scenario_id"].duplicated(), "scenario_id"].tolist()
        raise ValueError(f"Duplicate candidate scenario IDs: {dupes[:5]}")

    source_paths = [_single_csv(source) for source in existing_train]
    reference = (
        pd.concat([pd.read_csv(path) for path in source_paths], ignore_index=True)
        if source_paths else pd.DataFrame(columns=candidates.columns)
    )
    rejected: list[str] = []
    accepted_indices: list[int] = []
    if len(candidates):
        if not 0 <= threshold <= 1:
            raise ValueError("dedup threshold must be between 0 and 1")
        texts = [_scenario_text(row) for _, row in reference.iterrows()]
        texts += [_scenario_text(row) for _, row in candidates.iterrows()]
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embedding_model)
            vectors = model.encode(texts, normalize_embeddings=True)
        else:
            vectors = encoder(texts)
        vectors = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        retained = list(range(len(reference)))
        offset = len(reference)
        for index, row in candidates.iterrows():
            vector_index = offset + index
            similarities = vectors[retained] @ vectors[vector_index] if retained else []
            if len(similarities) and float(np.max(similarities)) > threshold:
                rejected.append(str(row["scenario_id"]))
            else:
                accepted_indices.append(index)
                retained.append(vector_index)
    else:
        accepted_indices = list(range(len(candidates)))

    accepted = candidates.iloc[accepted_indices].reset_index(drop=True)
    (output_dir / "outputs").mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output_dir / "outputs" / candidate_csv.name, index=False)
    accepted_ids = set(accepted["scenario_id"].astype(str))
    _write_scenario_json_subset(input_dir / "scenarios", output_dir / "scenarios", accepted_ids)
    report = {
        "enabled": True,
        "embedding_model": embedding_model,
        "threshold": threshold,
        "candidate_source": str(candidate_csv),
        "existing_train_sources": [str(path) for path in source_paths],
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_ids": rejected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dedup_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    return report


def split_dataset(
    input_dir: str | Path,
    train_dir: str | Path,
    test_dir: str | Path,
    test_size: int = 200,
    seed: int = 42,
    existing_train: list[str | Path] = (),
) -> None:
    """Deterministically split, preferring stratification by value pair.

    With ``existing_train``, that corpus is copied unchanged to the train
    output and every already-deduplicated candidate becomes evaluation data.
    """
    import pandas as pd
    import yaml

    input_dir = Path(input_dir)
    csv_path = _single_csv(input_dir)
    rows = pd.read_csv(csv_path)
    sources = [_single_csv(source) for source in existing_train]
    if sources:
        train_rows = pd.concat([pd.read_csv(path) for path in sources], ignore_index=True)
        test_rows = rows.copy()
        strategy = "immutable_train_all_candidates_eval"
    else:
        if test_size >= len(rows):
            raise ValueError(f"test_size ({test_size}) >= total scenarios ({len(rows)})")
        from sklearn.model_selection import train_test_split

        strata = rows.apply(
            lambda row: "::".join(sorted((str(row["value1"]), str(row["value2"])))),
            axis=1,
        )
        counts = strata.value_counts()
        can_stratify = (
            counts.min() >= 2
            and test_size >= len(counts)
            and len(rows) - test_size >= len(counts)
        )
        train_idx, test_idx = train_test_split(
            rows.index,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=strata if can_stratify else None,
        )
        train_rows, test_rows = rows.loc[train_idx], rows.loc[test_idx]
        strategy = "stratified_value_pair" if can_stratify else "deterministic_random_fallback"

    for split_dir, frame in ((Path(train_dir), train_rows), (Path(test_dir), test_rows)):
        (split_dir / "outputs").mkdir(parents=True, exist_ok=True)
        frame.to_csv(split_dir / "outputs" / csv_path.name, index=False)
    if not sources:
        _write_scenario_json_subset(
            input_dir / "scenarios", Path(train_dir) / "scenarios",
            set(train_rows["scenario_id"].astype(str)),
        )
    _write_scenario_json_subset(
        input_dir / "scenarios", Path(test_dir) / "scenarios",
        set(test_rows["scenario_id"].astype(str)),
    )
    report = {
        "strategy": strategy, "seed": seed,
        "existing_train_sources": [str(path) for path in sources],
        "train_count": len(train_rows), "eval_count": len(test_rows),
    }
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    (Path(test_dir) / "split_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(
        f"split {len(rows)} candidates -> {len(train_rows)} train / "
        f"{len(test_rows)} test using {strategy}"
    )


# ── Pool artifact layout ─────────────────────────────────────────────────────


def _value_set_name(value_set) -> str:
    path = Path(str(value_set))
    return path.stem if path.suffix == ".json" else str(value_set)


def pool_dir(pool: dict, cluster: ClusterConfig) -> Path:
    return (
        cluster.data / "scenarios" / _value_set_name(pool["value_set"]) / pool_id(pool)
    )


def candidate_dir(pool: dict, cluster: ClusterConfig) -> Path:
    return pool_dir(pool, cluster) / "candidates"


def full_dir(pool: dict, cluster: ClusterConfig) -> Path:
    return pool_dir(pool, cluster) / "full"


def train_dir(pool: dict, cluster: ClusterConfig) -> Path:
    return pool_dir(pool, cluster) / "train"


def test_dir(pool: dict, cluster: ClusterConfig) -> Path:
    return pool_dir(pool, cluster) / "test"


def record(pool: dict, cluster: ClusterConfig) -> Path:
    return pool_dir(pool, cluster) / "resolved_config.yaml"


def claim(pool: dict, cluster: ClusterConfig) -> Path:
    """Write (or verify) the pool's identity record."""
    return ensure_config_record(record(pool, cluster), pool_id(pool), pool_identity(pool))


def is_complete(pool: dict, cluster: ClusterConfig) -> bool:
    return (test_dir(pool, cluster) / "split_report.yaml").is_file()


def _existing_train_sources(pool: dict, cluster: ClusterConfig) -> list[Path]:
    configured = pool.get("dedup", {}).get("existing_train") or []
    if isinstance(configured, (str, Path)):
        configured = [configured]
    resolved = []
    for source in configured:
        path = Path(str(source))
        resolved.append(path if path.is_absolute() else cluster.repo / path)
    return resolved


# ── Which pools an experiment needs ──────────────────────────────────────────


def intervention_pools(cfg: dict) -> list[dict]:
    iv = cfg.get("intervention") or {}
    return [iv["pool"]] if isinstance(iv.get("pool"), dict) else []


def evaluation_pools(cfg: dict) -> list[dict]:
    scenarios = cfg["evaluation"].get("scenarios")
    return [scenarios["pool"]] if isinstance(scenarios, dict) else []


def pool_blocks(cfg: dict) -> list[dict]:
    """Every scenario-pool block a resolved experiment names, deduplicated by
    pool id (the intervention and the evaluation naming the same pool is the
    expected way to train and eval on one corpus's two halves)."""
    blocks: dict[str, dict] = {}
    for pool in intervention_pools(cfg) + evaluation_pools(cfg):
        blocks.setdefault(pool_id(pool), pool)
    return list(blocks.values())


# ── Build stages ─────────────────────────────────────────────────────────────


def _value_pairs(value_keys: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in value_keys for b in value_keys if a < b]


def stages(pool: dict, cluster: ClusterConfig) -> list[Stage]:
    """generate+filter → global dedup → train/test split, keyed by pool id.

    Stage names carry the pool id so two pools coexist in one pipeline. Tasks
    are stamped against the pool's own record — a pool is its own artifact,
    never validated against the experiment that happened to build it.
    """
    from valuegen.ground_truth.common import value_set_path

    pid = pool_id(pool)
    vs_path = value_set_path(pool["value_set"], cluster)
    vs_arg = V.value_set_cli_arg(vs_path)
    value_keys = list(V.load_value_set(vs_path).keys())
    gen = pool["scenario_gen"]
    gen_model = gen["model"]
    gen_model_short = gen_model.split("/")[-1]
    raw_dir = candidate_dir(pool, cluster) / "scenarios"
    filtered_dir = candidate_dir(pool, cluster) / "outputs"

    # 1) Scenario generation + filtering: one serial task —
    #    filter_scenarios appends every pair into {outputs}/{gen_model}.csv,
    #    so pairs must run serially within a task (incremental, resumable).
    gen_lines = []
    for v1, v2 in _value_pairs(value_keys):
        gen_cmd = (
            f"python {GENERATE_SCENARIOS} -o {raw_dir} -m {shlex.quote(gen_model)}"
            f" -v {vs_arg} -v1 {v1} -v2 {v2}"
            f" -n {gen.get('num_scenarios', 10)}"
        )
        # Deduplication is deliberately global in the next valuegen-owned
        # stage; per-pair submodule dedup would make results invocation-local.
        if gen.get("temperature") is not None:
            gen_cmd += f" --temperature {gen['temperature']}"
        filter_cmd = (
            f"python {FILTER_SCENARIOS}"
            f" -i {raw_dir}/{gen_model_short}_{v1}_{v2}.json"
            f" -o {filtered_dir} -v {vs_arg}"
            f" -m {shlex.quote(gen.get('filter_model', gen_model))}"
        )
        gen_lines += [gen_cmd, filter_cmd]
    scenario_stage = Stage(
        name=f"pool_{pid}_scenarios",
        tasks=[
            Task(
                key=f"generate_{gen_model_short}",
                command=(
                    "set -e\n" + "\n".join(gen_lines)
                    # date >, not touch: Task.is_done requires a non-empty
                    # file, so an empty marker re-runs generation forever —
                    # and each re-run *appends* to the candidate CSV.
                    + f"\ndate > {candidate_dir(pool, cluster) / 'scenarios_complete'}"
                ),
                done=candidate_dir(pool, cluster) / "scenarios_complete",
            )
        ],
        time=gen.get("time", "2-00:00:00"),
        mem=gen.get("mem", "16G"),
        gpus=int(gen.get("gpus", 0)),
        env=gen.get("env", "eval_api"),
    )

    directory = shlex.quote(str(pool_dir(pool, cluster)))
    dedup_stage = Stage(
        name=f"pool_{pid}_dedup",
        tasks=[Task(
            key="global_dedup",
            command=(
                f"python -m valuegen.ground_truth.scenario_pool dedup"
                f" --pool-dir {directory}"
            ),
            done=full_dir(pool, cluster) / "dedup_report.yaml",
        )],
        time="01:00:00",
        mem="8G",
    )

    split_stage = Stage(
        name=f"pool_{pid}_split",
        tasks=[Task(
            key="split",
            command=(
                f"python -m valuegen.ground_truth.scenario_pool split"
                f" --pool-dir {directory}"
            ),
            done=test_dir(pool, cluster) / "split_report.yaml",
        )],
        time="00:30:00",
        mem="4G",
    )

    result = [scenario_stage, dedup_stage, split_stage]
    for stage in result:
        for task in stage.tasks:
            task.config_id = pid
            task.config_record = record(pool, cluster)
    return result


def pending_stages(
    cfg: dict, cluster: ClusterConfig, blocks: list[dict] | None = None
) -> list[Stage]:
    """Build stages for every needed pool that is not complete. ``blocks``
    restricts to one side's pools (default: every pool the config names)."""
    if blocks is None:
        blocks = pool_blocks(cfg)
    seen: set[str] = set()
    result: list[Stage] = []
    for pool in blocks:
        pid = pool_id(pool)
        if pid in seen:
            continue
        seen.add(pid)
        if not is_complete(pool, cluster):
            result += stages(pool, cluster)
    return result


def claim_pools(
    cfg: dict, cluster: ClusterConfig, blocks: list[dict] | None = None
) -> None:
    for pool in (pool_blocks(cfg) if blocks is None else blocks):
        claim(pool, cluster)


# ── CLI for the pure-python steps (run as cpu SLURM tasks) ───────────────────


def _load_pool(pool_dir_path: str | Path, cluster: ClusterConfig) -> dict:
    import yaml

    path = Path(pool_dir_path) / "resolved_config.yaml"
    record_data = yaml.safe_load(path.read_text())
    pool = resolve_pool(dict(record_data["resolved_config"]))
    if pool_id(pool) != record_data["config_id"]:
        raise RuntimeError(f"{path} does not identify its own pool config")
    return pool


def main() -> None:
    import argparse

    from valuegen.config import load_cluster

    parser = argparse.ArgumentParser(description="scenario pool: pure-python steps")
    sub = parser.add_subparsers(dest="command", required=True)
    for verb, help_text in (
        ("dedup", "global semantic deduplication"),
        ("split", "train/test split of the deduplicated pool"),
    ):
        p = sub.add_parser(verb, help=help_text)
        p.add_argument("--pool-dir", required=True)
        p.add_argument("--cluster", default=None)

    args = parser.parse_args()
    cluster = load_cluster(args.cluster)
    pool = _load_pool(args.pool_dir, cluster)
    if args.command == "dedup":
        dedup = pool["dedup"]
        global_deduplicate(
            candidate_dir(pool, cluster),
            full_dir(pool, cluster),
            embedding_model=dedup.get("embedding_model", "all-MiniLM-L6-v2"),
            threshold=float(dedup.get("threshold", 0.9)),
            existing_train=_existing_train_sources(pool, cluster),
        )
    elif args.command == "split":
        split = pool["split"]
        split_dataset(
            full_dir(pool, cluster),
            train_dir(pool, cluster),
            test_dir(pool, cluster),
            test_size=int(split.get("test_size", 200)),
            seed=int(split.get("seed", 42)),
            existing_train=_existing_train_sources(pool, cluster),
        )


if __name__ == "__main__":
    main()
