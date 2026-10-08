"""RQ3: a functional taxonomy of values.

Reproduces, from data/gt and data/paper/rq3:
  1. the taxonomy: k-medoids (k=4, seed 0) on the persona similarity grid of
     the 266 Values-in-the-Wild level-1 values (OLMo-3.1-32B-Instruct-SFT,
     layer 32), checked against the published cluster assignment;
  2. tab:taxonomy-z: for our taxonomy, LitmusValues (16 classes) and VITW
     level 3 (5 categories), how much better than chance each grouping of the
     49 trained tenets co-moves under steering, per arm (OLMo/Qwen × DPO/SFT
     at 7–8B and at 30–32B) and aggregated over the eight
     (2000 shared relabelings, seed 0). Our taxonomy reaches the tenets by
     nearest cluster centroid in the same persona space.

The ± in the table is the half-width of a 95% scenario-bootstrap interval
(1000 resamples of the 11,188 evaluation scenarios, 1000 relabelings each). By
default it is read from data/paper/rq3/taxonomy_z_bootstrap.json, whose point
estimates (5000 relabelings) must match a recomputation at that count. With --bootstrap it is
recomputed from the per-scenario evals in data/paper/evals/ (a few minutes on
8 CPUs), after checking that those evals rebuild the published matrices.

Writes paper/out/rq3.json, rq3_clusters_k4.csv and the .tex table, then checks
against paper/expected/. Exit status 1 on any mismatch.

    .venvs/core/bin/python paper/reproduce/rq3.py [--bootstrap]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys

import numpy as np

from _common import (ARM_LABELS, ARMS, ARMS_7B, ARMS_30B, EXPECTED, GT, PAPER_DATA, TARGETS,
                     Checker, load_json, require_paper_data, write)
from valuegen.analysis import cluster as CL
from valuegen.analysis import mds as M
from valuegen.analysis.steerability import (bootstrap_taxonomy_z, ds_from_base, label_vector,
                                            load_target, load_values, movement_distance,
                                            movement_distance_fast, rate_operator, rates,
                                            taxonomy_z, transport_labels)

D = PAPER_DATA / "rq3"
K, CLUSTER_SEED, N_PERM, SEED = 4, 0, 2000, 0
# The stored scenario bootstrap records point estimates from 5000 relabelings.
BOOT_POINT_N_PERM = 5000
TAXONOMIES = ["kmed_k4", "litmus", "vitw_l3"]
# The table's columns, in the paper's order.
TABLE_COLS = {
    "kmed_k4": r"\textbf{\taxonomyname-Olmo-32B}",
    "vitw_l3": r"\textbf{Values in the Wild}",
    "litmus": r"\textbf{LitmusValues}",
}


# ── 1. The taxonomy ──────────────────────────────────────────────────────────


def cluster_vitw() -> tuple[dict[str, int], list[str]]:
    """k-medoids on the VITW persona grid → ({value: cluster}, medoid values)."""
    sim = np.load(D / "vitw_persona_L32.npy")
    values, _ = load_values(D / "vitw_persona_L32_values.json")
    sim, values = M.drop_nan_values(sim, values)
    dist = M.cos_to_dist(sim)
    labels = CL.cluster_labels(dist, K, "kmedoids", seed=CLUSTER_SEED)
    medoids = CL.medoids_of(dist, labels)
    return ({v: int(c) for v, c in zip(values, labels)},
            [values[i] for i in sorted(medoids.values())])


def clusters_csv(assign: dict[str, int], medoids: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["value", "cluster", "is_medoid"])
    for v, c in assign.items():
        w.writerow([v, c, int(v in medoids)])
    return buf.getvalue()


def stacked_vectors(name: str) -> dict[str, np.ndarray]:
    arr = np.load(D / f"{name}_persona_vectors_L32.npy")
    values = load_json(D / f"{name}_persona_vectors_L32_values.json")
    return {v: arr[i] for i, v in enumerate(values)}


# ── 2. The z table ───────────────────────────────────────────────────────────


def common_rows() -> list[str]:
    # The 49 tenets every arm trained, in the OLMo DPO matrix's row order.
    _, rows, _ = load_target(TARGETS["olmo_dpo"])
    return rows


def taxonomies(assign: dict[str, int], rows49: list[str]) -> dict[str, dict]:
    tenet_vecs = stacked_vectors("tenets")
    vitw = load_json(D / "labels_vitw.json")["labels"]
    litmus = load_json(D / "labels_litmus16.json")["labels"]
    return {
        "kmed_k4": transport_labels(assign, stacked_vectors("vitw"),
                                    {t: tenet_vecs[t] for t in rows49}),
        "litmus": {t: litmus[t] for t in rows49},
        "vitw_l3": {t: vitw[t]["l3"] for t in rows49},
    }


def z_table(assign: dict[str, int], n_perm: int = N_PERM) -> tuple[dict, dict, dict]:
    """(z per taxonomy, labels per taxonomy, distances per arm)."""
    rows49 = common_rows()
    dists = {}
    for a in ARMS:
        m, rows, cols = load_target(TARGETS[a])
        dists[a], _ = movement_distance(m, rows, cols, keep_rows=rows49)
    tax = taxonomies(assign, rows49)
    labels = {t: label_vector(tax[t], rows49) for t in TAXONOMIES}
    z = {t: taxonomy_z(labels[t], dists, ARMS, n_perm=n_perm, seed=SEED) for t in TAXONOMIES}
    return z, labels, dists


def stored_bootstrap_pm(z: dict) -> dict:
    """± half-widths from the stored scenario bootstrap; refuses stale ones."""
    boot = load_json(D / "taxonomy_z_bootstrap.json")["results"]
    pm = {}
    for t in TAXONOMIES:
        pm[t] = {}
        for col, v in boot[t].items():
            if abs(v["z"] - z[t][col]) > 1e-6:
                raise SystemExit(f"taxonomy_z_bootstrap.json is stale: {t}/{col} "
                                 f"z={v['z']:.4f} there vs {z[t][col]:.4f} here")
            pm[t][col] = (v["hi"] - v["lo"]) / 2
    return pm


def recompute_bootstrap_pm(labels: dict, dists: dict) -> dict:
    """± half-widths recomputed from data/paper/evals (see the module docstring)."""
    import pandas as pd

    scen = pd.read_parquet(PAPER_DATA / "evals" / "scenarios.parquet")
    rows49 = common_rows()
    ops = {}
    for a in ARMS:
        gt_id = TARGETS[a].relative_to(GT).parts[0]
        evals = pd.read_parquet(PAPER_DATA / "evals" / f"{gt_id}.parquet")
        rows, cols = load_values(f"{TARGETS[a]}_raw_values.json")
        ops[a] = rate_operator(evals, len(scen), scen, rows, cols)
        # Parity: unit weights must rebuild the published matrix and distances.
        base, raw = rates(ops[a], np.ones((1, len(scen))))
        stored = np.load(f"{TARGETS[a]}_raw.npy")
        if not np.allclose(raw[0], stored, equal_nan=True, rtol=0, atol=1e-12):
            raise SystemExit(f"{gt_id}.parquet does not rebuild {TARGETS[a].name}_raw.npy")
        d = movement_distance_fast(ds_from_base(raw[0], base[0]), rows, cols,
                                   [rows.index(r) for r in rows49])
        if not np.allclose(d, dists[a], atol=1e-10):
            raise SystemExit(f"{a}: co-movement distances from the evals differ")
    print("parity OK: the per-scenario evals rebuild every published matrix")

    def progress(done, total):
        print(f"  bootstrap {done}/{total}", flush=True)

    boot = bootstrap_taxonomy_z(ops, rows49, labels, ARMS, n_boot=1000, n_perm=1000,
                                chunk=50, seed=0, progress=progress)
    pm = {}
    for t in TAXONOMIES:
        pm[t] = {}
        for c, v in boot[t].items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            pm[t][c] = (float(hi) - float(lo)) / 2
    return pm


def z_tex(z: dict, pm: dict) -> str:
    """Arms as rows (two scale blocks, then the eight-arm aggregate),
    taxonomies as columns."""
    cols = ARMS + ["agg"]

    def unique_best(vals):
        disp = [f"{v:.2f}" for v in vals]
        best = max(vals)
        return best if disp.count(f"{best:.2f}") == 1 else None

    best = {c: unique_best([z[t][c] for t in TAXONOMIES]) for c in cols}
    # The caption says our taxonomy has the best aggregate; refuse to emit it otherwise.
    if best["agg"] is None or abs(z["kmed_k4"]["agg"] - best["agg"]) > 1e-9:
        raise SystemExit("caption claims kmed_k4 has the best aggregate; it does not")

    def cell(v, bold, h):
        s = f"{v:.2f}"
        return (f"\\textbf{{{s}}}" if bold else s) + f"{{\\scriptsize$\\pm${h:.2f}}}"

    def line(label, c):
        cells = [cell(z[t][c], best[c] is not None and abs(z[t][c] - best[c]) < 1e-9, pm[t][c])
                 for t in TABLE_COLS]
        return f"{label} & " + " & ".join(cells) + r" \\"

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "c" * len(TABLE_COLS) + "}", r"\toprule",
         "& " + " & ".join(TABLE_COLS.values()) + r" \\"]
    for block in (ARMS_7B, ARMS_30B):
        L.append(r"\midrule")
        L += [line(ARM_LABELS[a], a) for a in block]
    L += [r"\midrule", line(r"\textbf{Aggregate}", "agg"),
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Comparing $z$-scores for \taxonomyname and two existing value "
          r"taxonomies, which measure how well they recover the value generalization "
          r"structure found in \Cref{sec:singlevalue} across all eight fine-tuning "
          r"setups. The $z$-score metric enables us to fairly compare taxonomies with "
          r"different numbers of categories. \textbf{Aggregate} uses a pooled "
          r"permutation test across all eight interventions, rather than just "
          r"averaging $z$-scores across interventions; $\pm$ denotes 95\% confidence "
          r"intervals over bootstrapped steerability metrics. We find that "
          r"\taxonomyname has the highest $z$-score aggregated across all methods, "
          r"and that differences between \taxonomyname and other taxonomies are "
          r"generally significant.}",
          r"\label{tab:taxonomy-z}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bootstrap", action="store_true",
                    help="recompute the ± from the per-scenario evals (minutes)")
    args = ap.parse_args()
    require_paper_data()
    chk = Checker("RQ3")

    assign, medoids = cluster_vitw()
    write("rq3_clusters_k4.csv", clusters_csv(assign, medoids))
    published = {r["value"]: (int(r["cluster"]), int(r["is_medoid"]))
                 for r in csv.DictReader(open(D / "clusters_k4.csv"))}
    chk.value("clusters_k4", {v: (c, int(v in medoids)) for v, c in assign.items()},
              published)

    z, labels, dists = z_table(assign)
    pm = (recompute_bootstrap_pm(labels, dists) if args.bootstrap
          else stored_bootstrap_pm(z_table(assign, BOOT_POINT_N_PERM)[0]))
    write("rq3.json", json.dumps({"z": z, "boot_pm": pm, "nperm": N_PERM,
                                  "common_rows": 49, "medoids": medoids}, indent=1) + "\n")
    write("rq3_taxonomy_z_table.tex", z_tex(z, pm))

    expected = load_json(EXPECTED / "rq3.json")
    chk.value("z", z, expected["z"])
    chk.value("boot_pm", pm, expected["boot_pm"])
    chk.tex("rq3_taxonomy_z_table.tex", "rq3_taxonomy_z_table.tex")
    return chk.finish()


if __name__ == "__main__":
    sys.exit(main())
