"""Appendix I: how the OLMo-32B taxonomy lines up with a Qwen-8B one and with
Values-in-the-Wild's own level-3 categories.

Both taxonomies are k-medoids (k=4, seed 0) on a persona similarity grid of the
266 VITW level-1 values: OLMo-3.1-32B-Instruct-SFT at layer 32 (the RQ3
taxonomy) and the neutral-SFT Qwen3-8B at layer 18. Reproduces, from
data/paper/rq3 and data/paper/appi:
  * both k=4 partitions, and both grids' 2D nonmetric MDS maps (seed 42);
  * the two contingency tables of the taxonomy-comparison figure (OLMo-32B
    clusters x Qwen-8B clusters, OLMo-32B clusters x VITW L3 categories) and
    the Procrustes fit between the two maps (r = sqrt(1 - disparity)),
    against paper/expected/appi/appi.json;
  * every file in paper/figures/appI_taxonomy_comparison/inputs/, byte-for-byte.
Exit status 1 on any mismatch.

    .venvs/core/bin/python paper/reproduce/appi.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import warnings

import numpy as np
from scipy.spatial import procrustes

from _common import EXPECTED, PAPER_DATA, ROOT, Checker, load_json, require_paper_data, write
from valuegen.analysis import cluster as CL
from valuegen.analysis import mds as M
from valuegen.analysis.steerability import load_values

FIG_INPUTS = ROOT / "paper/figures/appI_taxonomy_comparison/inputs"
GRIDS = {
    "olmo3_32b": PAPER_DATA / "rq3" / "vitw_persona_L32",
    "qwen3_8b": PAPER_DATA / "appi" / "qwen3_8b_vitw_persona_L18",
}
K, CLUSTER_SEED = 4, 0


def taxonomy(sim: np.ndarray, values: list[str]) -> tuple[str, dict[str, str]]:
    """k-medoids partition → (clusters csv text, {value: its cluster's medoid})."""
    sim, values = M.drop_nan_values(sim, values)
    dist = M.cos_to_dist(sim)
    labels = CL.cluster_labels(dist, K, "kmedoids", seed=CLUSTER_SEED)
    medoids = CL.medoids_of(dist, labels)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["value", "cluster", "is_medoid"])
    for i, v in enumerate(values):
        w.writerow([v, int(labels[i]), int(i in medoids.values())])
    return buf.getvalue(), {v: values[medoids[int(c)]] for v, c in zip(values, labels)}


def contingency(rows: dict[str, str], cols: dict[str, str]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for v in sorted(set(rows) & set(cols)):
        out.setdefault(rows[v], {}).setdefault(cols[v], 0)
        out[rows[v]][cols[v]] += 1
    return out


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


def main() -> int:
    require_paper_data()
    chk = Checker("Appendix I")

    def same_bytes(label: str, got: bytes, name: str) -> None:
        chk.n += 1
        if got != (FIG_INPUTS / name).read_bytes():
            chk.failures.append(f"{label}: differs from figure input {name}")

    assign, coords = {}, {}
    for model, stem in GRIDS.items():
        sim = np.load(f"{stem}.npy")
        values_path = f"{stem}_values.json"
        values, _ = load_values(values_path)
        text, assign[model] = taxonomy(sim, values)
        write(f"appi_clusters_k4_{model}.csv", text)
        same_bytes(f"{model} k=4 clusters", text.encode(), f"clusters_k4_{model}.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            coords[model], _ = M.mds_variant(sim, "nonmetric")
        same_bytes(f"{model} MDS", npy_bytes(coords[model]), f"mds_nonmetric_{model}.npy")
        same_bytes(f"{model} value order", open(values_path, "rb").read(),
                   f"persona_values_{model}.json")
    same_bytes("vitw_meta.json", (PAPER_DATA / "rq3" / "vitw_meta.json").read_bytes(), "vitw_meta.json")

    l3 = {v: m["l3"] for v, m in load_json(PAPER_DATA / "rq3" / "vitw_meta.json").items()}
    _, _, m2 = procrustes(coords["olmo3_32b"], coords["qwen3_8b"])
    res = {"contingency_olmo32b_vs_qwen8b": contingency(assign["olmo3_32b"], assign["qwen3_8b"]),
           "contingency_olmo32b_vs_vitw_l3": contingency(assign["olmo3_32b"], l3),
           "procrustes_r": float(np.sqrt(1 - m2)),
           "medoids": {m: sorted(set(a.values())) for m, a in assign.items()}}
    write("appi.json", json.dumps(res, indent=1, sort_keys=True) + "\n")
    chk.value("appi", res, load_json(EXPECTED / "appi" / "appi.json"))
    print(f"  procrustes r = {res['procrustes_r']:.4f}")
    return chk.finish()


if __name__ == "__main__":
    sys.exit(main())
