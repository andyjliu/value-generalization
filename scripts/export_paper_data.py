"""Collect rebuilt artifacts into the data/paper/ layout paper/reproduce reads.

The last step of each paper/regenerate script. Each subcommand copies what
the valuegen stores produced (``data/similarity``, ``data/vectors``,
``data/gt``) into ``data/paper/`` under the names the reproduce scripts expect,
and records every file in ``data/paper/MANIFEST.json`` (sha256, size, source).

- ``rq1``: the 40 predictor grids (5 predictors x 8 arms: OLMo/Qwen x DPO/SFT
  at 7-8B and, keys suffixed ``_30b``, at 30-32B). ``rq1`` and ``appd``
  export the 7-8B models; with ``--30b``, the 30-32B ones;
- ``rq3``: the VITW-266 persona grid and the layer-32 persona vectors of the
  VITW values and the tenets (OLMo-3.1-32B-Instruct-SFT), the k=4 clustering,
  and the LitmusValues / VITW-L3 tenet labels (scripts/taxonomy_labels.py);
- ``appd``: per-layer persona cosine grids of the four neutral-SFT models
  (``grids_{qwen,olmo}{,_30b}.npz``) and the 7-8B models' layer sweeps
  (best_layer.json, aggregated_layer_sweep.csv; the 30-32B layers are pinned);
- ``appf``: the base-reads-neutral grids (pretrained bases reading the
  neutral models' pairs, urial0). The DPO-pairs grids of appendix F have no
  release producer (the data_proximity_t1b data method was not ported);
- ``rq2 <exp_dir>``: every checkpoint's judged prefill rows of a multivalue
  experiment (``<exp_dir>/evals/<ckpt>/prefill/rows.jsonl``) as one compact
  table, ``evals/prefill_rows.parquet`` under the experiment's data/paper/rq2
  directory: the sample's identity and the judge's verdict, in the original
  row order. The judge's free-text reasoning and the run bookkeeping (log
  paths, token usage, timings, retry records) are left out;
- ``evals <gt_dir>...``: each GT run's per-scenario judge results as
  ``evals/<gt_id>.parquet`` plus ``evals/scenarios.parquet``, whose scenario
  order is the first run's base CSV (the paper used the OLMo-3 7B DPO run;
  the RQ3 bootstrap's resamples depend on that order).

Artifacts are found by value set and pairs-model name, not by ID: data
artifact IDs hash the ``--model`` path, so a rebuild on another machine has
new IDs. Exactly one matching artifact must exist.

    .venvs/core/bin/python scripts/export_paper_data.py rq1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "paper"
PAIRS = DATA / "elicitation" / "pairs" / "default_llm"
SIM = DATA / "similarity"
VEC = DATA / "vectors"
SWEEPS = DATA / "sweeps"

# arm -> (activation model short name, pairs model short name, data artifact key)
ARMS = {
    "qwen_dpo": ("neutral-sft-v3-qwen3-8b", "neutral-sft-v3-qwen3-8b"),
    "olmo_dpo": ("neutral-sft-v3-olmo3-7b", "neutral-sft-v3-olmo3-7b"),
    "qwen_sft": ("Qwen3-8B-Base", "Qwen3-8B-Base"),
    "olmo_sft": ("Olmo-3-1025-7B", "Olmo-3-1025-7B"),
    "qwen_dpo_30b": ("neutral-sft-v3-qwen3-30b-a3b", "neutral-sft-v3-qwen3-30b-a3b"),
    "olmo_dpo_30b": ("neutral-sft-v3-olmo3-32b", "neutral-sft-v3-olmo3-32b"),
    "qwen_sft_30b": ("Qwen3-30B-A3B-Base", "Qwen3-30B-A3B-Base"),
    "olmo_sft_30b": ("Olmo-3-1125-32B", "Olmo-3-1125-32B"),
}
# Grad_proj renders pairs with the family template on Qwen and the tokenizer's
# own template on OLMo (the neutral OLMo checkpoint was trained under it).
# Both hold at either scale.
GRAD_SUFFIX = {"qwen_dpo": "", "olmo_dpo": "_native", "qwen_sft": "_urial0", "olmo_sft": "_urial0"}
PERSONA_SUFFIX = {"qwen_dpo": "", "olmo_dpo": "", "qwen_sft": "_urial0", "olmo_sft": "_urial0"}
BIG = "Olmo-3.1-32B-Instruct-SFT"
TRANSPORT_LAYER = 32
K = 4
# rows.jsonl fields kept in rq2's prefill_rows.parquet -> stored dtype. What
# is dropped is constant (suite, followup_id), derivable (sample_id; antispec_id
# is in eval_inputs), the judge's reasoning, or run bookkeeping.
RQ2_ROW_COLUMNS = {
    "checkpoint_id": "category", "scenario_id": "category", "condition": "category",
    "repeat": "int8", "side": "int8", "value": "category", "other_value": "category",
    "bucket": "category", "action": "category", "likert": "Int8",
    "likert_toward_value": "float64", "pro_value": "float64", "ambiguous": "float64",
    "valid": "bool", "grader_failed": "bool",
}

manifest: dict = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def record(rel: str, source: str) -> None:
    p = OUT / rel
    manifest[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size, "source": source}


def shown(p: Path) -> str:
    """``p`` repo-relative when it is inside the repo, for the manifest."""
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def copy(rel: str, src: Path) -> None:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    record(rel, shown(src))


def one(paths, what: str) -> Path:
    paths = sorted(paths)
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one {what}, found {len(paths)}: "
                         + ", ".join(str(p) for p in paths))
    return paths[0]


def artifact_id(value_set: str, pairs_model: str) -> str:
    return one((PAIRS / value_set).glob(f"{pairs_model}-*"),
               f"{value_set} pairs artifact from {pairs_model}").name


def grid_at(directory: Path, pattern: str, what: str) -> Path:
    """The one ``*.npy`` in ``directory`` whose name fully matches ``pattern``."""
    rx = re.compile(pattern)
    return one([p for p in directory.glob("*.npy") if rx.fullmatch(p.name)], what)


def values_of(npy: Path) -> Path:
    for cand in (npy.with_name(npy.stem + "_values.json"), npy.parent / "similarity_values.json"):
        if cand.exists():
            return cand
    raise SystemExit(f"no values sidecar for {npy}")


def export_rq1(big: bool) -> None:
    for arm, (model, pairs_model) in ARMS.items():
        if arm.endswith("_30b") != big:
            continue
        art = artifact_id("constitution_tenets_v3", pairs_model)
        sim = lambda pred: SIM / pred / "default_llm" / model / art  # noqa: E731
        kind = arm.removesuffix("_30b")
        grids = {
            "persona": grid_at(sim("persona"),
                               rf"persona_response_avg_diff{PERSONA_SUFFIX[kind]}_L\d+\.npy",
                               f"{arm} persona grid"),
            "grad_proj": grid_at(sim("grad_proj"),
                                 rf"gradproj_dpoinit_respavg{GRAD_SUFFIX[kind]}_L\d+\.npy",
                                 f"{arm} grad_proj grid"),
            "weight_steer": (sim("weight_steer") / ("urial0" if kind.endswith("sft") else "")
                             / "similarity_matrix.npy"),
            "sentemb_behavior": SIM / "sentemb_behavior" / art / "sentemb_behavior_respavg_diff.npy",
            "sentence_emb": one(
                (SIM / "sentence_emb" / "descriptions" / "all-mpnet-base-v2").glob(
                    "shared-*/sentence_emb_cos.npy"),
                "constitution_tenets_v3 sentence_emb grid"),
        }
        for pred, npy in grids.items():
            copy(f"rq1/grids/{arm}/{pred}.npy", npy)
            copy(f"rq1/grids/{arm}/{pred}_values.json", values_of(npy))


def export_rq3(labels_dir: Path) -> None:
    import torch

    from valuegen.analysis import cluster as CL
    from valuegen.analysis import mds as M

    arts = {"vitw": artifact_id("vitw_l1_266", "neutral-sft-v3-olmo3-7b"),
            "tenets": artifact_id("constitution_tenets_v3", "neutral-sft-v3-olmo3-7b")}
    grid_dir = SIM / "persona" / "default_llm" / BIG / arts["vitw"]
    copy("rq3/vitw_persona_L32.npy", grid_dir / "persona_response_avg_diff_L32.npy")
    copy("rq3/vitw_persona_L32_values.json", grid_dir / "persona_response_avg_diff_L32_values.json")

    for name, art in arts.items():
        d = DATA / "vectors" / "persona" / "default_llm" / BIG / art
        files = sorted(d.glob("*_response_avg_diff.pt"))
        values = [f.name[: -len("_response_avg_diff.pt")] for f in files]
        arr = np.stack([torch.load(f, map_location="cpu", weights_only=False)[TRANSPORT_LAYER].numpy()
                        for f in files])
        rel = f"rq3/{name}_persona_vectors_L32.npy"
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(OUT / rel, arr)
        record(rel, f"{shown(d)}/<value>_response_avg_diff.pt [layer {TRANSPORT_LAYER}]")
        (OUT / f"rq3/{name}_persona_vectors_L32_values.json").write_text(
            json.dumps(values, indent=0) + "\n")
        record(f"rq3/{name}_persona_vectors_L32_values.json", "value order of the stacked vectors")

    # The published taxonomy: k-medoids k=4, seed 0 on the VITW grid.
    sim = np.load(OUT / "rq3/vitw_persona_L32.npy")
    values = json.loads((OUT / "rq3/vitw_persona_L32_values.json").read_text())
    values = values["rows"] if isinstance(values, dict) else values
    sim, values = M.drop_nan_values(sim, values)
    dist = M.cos_to_dist(sim)
    labels = CL.cluster_labels(dist, K, "kmedoids", seed=0)
    medoids = set(CL.medoids_of(dist, labels).values())
    with open(OUT / "rq3/clusters_k4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value", "cluster", "is_medoid"])
        for i, v in enumerate(values):
            w.writerow([v, int(labels[i]), int(i in medoids)])
    record("rq3/clusters_k4.csv", "k-medoids k=4 seed 0 on rq3/vitw_persona_L32.npy")

    # The class definitions and the VITW hierarchy (litmus_value_classes.json,
    # vitw_meta.json) are inputs taken from the two papers; they stay as fetched.
    for name in ("labels_litmus16.json", "labels_vitw.json"):
        copy(f"rq3/{name}", labels_dir / name)


def export_appd(big: bool) -> None:
    import torch

    pooling = "response_avg_diff"
    for key, arm in (("qwen", "qwen_dpo"), ("olmo", "olmo_dpo"),
                     ("qwen_30b", "qwen_dpo_30b"), ("olmo_30b", "olmo_dpo_30b")):
        if key.endswith("_30b") != big:
            continue
        model, pairs_model = ARMS[arm]
        art = artifact_id("constitution_tenets_v3", pairs_model)
        vdir = VEC / "persona" / "default_llm" / model / art
        values = sorted(p.name[: -len(f"_{pooling}.pt")] for p in vdir.glob(f"*_{pooling}.pt"))
        if not values:
            raise SystemExit(f"no {pooling} vectors under {shown(vdir)}")
        stack = np.stack([torch.load(vdir / f"{v}_{pooling}.pt", weights_only=False).float().numpy()
                          for v in values])                                # [V, L+1, H]
        unit = stack / np.linalg.norm(stack, axis=-1, keepdims=True)
        grids = np.einsum("vlh,wlh->lvw", unit, unit)                       # [L+1, V, V]
        rel = f"appd/grids_{key}.npz"
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / rel, values=np.array(values), grids=grids)
        record(rel, f"per-layer cosine grids of {shown(vdir)}/<value>_{pooling}.pt")
        if key.endswith("_30b"):   # layer pinned, no sweep to ship
            continue
        sweep = SWEEPS / "persona" / "default_llm" / model / art
        copy(f"appd/best_layer_{key}.json", sweep / "best_layer.json")
        copy(f"appd/aggregated_layer_sweep_{key}.csv", sweep / "aggregated_layer_sweep.csv")


def export_appf() -> None:
    for arm, base, layer in (("qwen_sft", "Qwen3-8B-Base", 18), ("olmo_sft", "Olmo-3-1025-7B", 16)):
        neutral = ARMS[arm.replace("_sft", "_dpo")][1]
        art = artifact_id("constitution_tenets_v3", neutral)
        for pred, name in (("persona", f"persona_response_avg_diff_urial0_L{layer}.npy"),
                           ("grad_proj", f"gradproj_dpoinit_respavg_urial0_L{layer}.npy")):
            npy = SIM / pred / "default_llm" / base / art / name
            stem = f"appf/grids/{arm}_{pred}_base_reads_neutral"
            copy(f"{stem}.npy", npy)
            copy(f"{stem}_values.json", values_of(npy))


def export_rq2(exp_dir: Path) -> None:
    import pandas as pd

    exp_dir = exp_dir.resolve()
    files = sorted(exp_dir.glob("evals/*/prefill/rows.jsonl"))
    if not files:
        raise SystemExit(f"no evals/<ckpt>/prefill/rows.jsonl under {exp_dir}")
    parts = []
    for f in files:
        df = pd.read_json(f, lines=True, dtype=False)
        if not (df.suite == "prefill").all() or not (df.followup_id == 0).all():
            raise SystemExit(f"{f}: suite/followup_id are not the constants the reader restores")
        parts.append(df[list(RQ2_ROW_COLUMNS)])
    out = pd.concat(parts, ignore_index=True)
    # Checkpoints in file order, so each one's rows stay contiguous and ordered.
    ckpts = [f.parent.parent.name for f in files]
    out["checkpoint_id"] = pd.Categorical(out.checkpoint_id, categories=ckpts)
    out = out.astype({c: t for c, t in RQ2_ROW_COLUMNS.items() if c != "checkpoint_id"})
    rel = f"rq2/multivalue/{exp_dir.parent.name}/{exp_dir.name}/evals/prefill_rows.parquet"
    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / rel, index=False, compression="zstd")
    record(rel, f"{len(files)} rows.jsonl under {shown(exp_dir)}/evals/<ckpt>/prefill/")


def export_evals(gt_dirs: list[Path]) -> None:
    import pandas as pd
    import yaml

    scen_path = OUT / "evals" / "scenarios.parquet"
    scen = pd.read_parquet(scen_path) if scen_path.exists() else None
    for gt_dir in gt_dirs:
        gt_dir = gt_dir.resolve()
        prov_p = one(gt_dir.glob("matrices/*_likert_raw_provenance.yaml"),
                     f"likert raw provenance in {gt_dir}")
        prov = yaml.safe_load(prov_p.read_text())
        rows = json.loads(prov_p.with_name(
            prov_p.name.replace("_provenance.yaml", "_values.json")).read_text())
        rows = rows["rows"] if isinstance(rows, dict) else rows
        srcs = [ROOT / s if not Path(s).is_absolute() else Path(s) for s in prov["source_csvs"]]
        models = ["base"] + rows
        if scen is None:
            scen = pd.read_csv(srcs[0], usecols=["scenario_id", "value1", "value2"])
            scen_path.parent.mkdir(parents=True, exist_ok=True)
            scen.to_parquet(scen_path, index=False, compression="zstd")
            record("evals/scenarios.parquet", str(srcs[0]))
        idx = pd.Index(scen.scenario_id)
        parts = []
        for m, src in zip(models, srcs):
            # Parse exactly as the GT matrices and the RQ3 bootstrap do.
            df = pd.read_csv(src, usecols=["scenario_id", "value1", "value2", "choice", "likert"])
            s = idx.get_indexer(df.scenario_id)
            if not (df.scenario_id.is_unique and len(df) == len(scen) and (s >= 0).all()):
                raise SystemExit(f"{src}: does not cover the scenario pool exactly once")
            if not ((scen.value1.values[s] == df.value1.values).all()
                    and (scen.value2.values[s] == df.value2.values).all()):
                raise SystemExit(f"{src}: value pair differs from scenarios.parquet")
            parts.append(pd.DataFrame({
                "model": m,
                "scenario": s.astype(np.int32),
                "likert": pd.to_numeric(df.likert, errors="coerce").values.astype(np.float64),
                "choice": df.choice.values,
            }))
        out = pd.concat(parts, ignore_index=True)
        out["model"] = pd.Categorical(out.model, categories=models, ordered=True)
        out["choice"] = pd.Categorical(out.choice, categories=["A", "B", "ERROR"])
        rel = f"evals/{gt_dir.name}.parquet"
        out.to_parquet(OUT / rel, index=False, compression="zstd")
        record(rel, f"{len(srcs)} eval CSVs listed in {shown(prov_p)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="what", required=True)
    scale = dict(action="store_true", dest="big", help="the 30-32B models instead of the 7-8B ones")
    sub.add_parser("rq1").add_argument("--30b", **scale)
    r3 = sub.add_parser("rq3")
    r3.add_argument("--labels", type=Path, default=DATA / "taxonomy_labels",
                    help="dir with labels_litmus16.json and labels_vitw.json "
                         "(scripts/taxonomy_labels.py output)")
    sub.add_parser("appd").add_argument("--30b", **scale)
    sub.add_parser("appf")
    r2 = sub.add_parser("rq2")
    r2.add_argument("exp_dir", type=Path, help="data/multivalue/<experiment>/<exp_id> directory")
    ev = sub.add_parser("evals")
    ev.add_argument("gt_dirs", nargs="+", type=Path, help="data/gt/<gt_id> directories")
    args = ap.parse_args()

    mpath = OUT / "MANIFEST.json"
    if mpath.exists():
        manifest.update(json.loads(mpath.read_text()))
    if args.what == "rq1":
        export_rq1(args.big)
    elif args.what == "rq3":
        export_rq3(args.labels)
    elif args.what == "appd":
        export_appd(args.big)
    elif args.what == "appf":
        export_appf()
    elif args.what == "rq2":
        export_rq2(args.exp_dir)
    else:
        export_evals(args.gt_dirs)
    OUT.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{len(manifest)} files in {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
