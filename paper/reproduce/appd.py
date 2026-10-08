"""Appendix D: are persona similarity grids stable across layers?

The persona predictor reads one layer of each value's ``response_avg_diff``
stack: the layer a GT-independent steering sweep picked (neutral Qwen3-8B SFT
-> L18 of 36, neutral OLMo-3-7B SFT -> L16 of 32). The 30-32B neutral-SFT
models were not swept: their layer is pinned at the same relative depth
(Qwen3-30B-A3B L24 of 48, OLMo-3-32B L32 of 64), marked layer_source="pinned".
This recomputes, from data/paper/appd:

  * each sweep's chosen layer, as the argmax of mean trait score over its
    aggregated layer sweep (checked against best_layer.json);
  * for every layer L >= 1, the Spearman rho between the 66x66 grid at L and
    at the sweep layer (the figure), the layer x layer rho matrix, the
    mid-depth band summaries, and the cross-model rho at the selected layers.

grids_{qwen,olmo}{,_30b}.npz hold the per-layer cosine grids of the 66 tenets'
vectors (float32, [layers+1, 66, 66], index 0 = embeddings); they rebuild
bit-exactly from the vector stacks, and their sweep layer equals the RQ1
persona grid to ~1e-6 (float32 vs float64), which is checked here.

Writes paper/out/appd_layer_stability.json and compares it byte-for-byte with
the figure's input, paper/figures/appD_persona_layer_stability/inputs/.
Exit status 1 on any mismatch.

    .venvs/core/bin/python paper/reproduce/appd.py
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
from scipy import stats

from _common import OUT, PAPER_DATA, ROOT, Checker, load_json, require_paper_data, write

D = PAPER_DATA / "appd"
FIG_INPUT = ROOT / "paper/figures/appD_persona_layer_stability/inputs/layer_stability.json"
POOLING = "response_avg_diff"
# (key, label, model/artifact, RQ1 arm whose persona grid is this model's sweep layer)
MODELS = [
    ("qwen", "Qwen3-8B (neutral SFT)",
     "neutral-sft-v3-qwen3-8b/neutral-sft-v3-qwen3-8b-5a2708f91afb", "qwen_dpo"),
    ("olmo", "OLMo-3-7B (neutral SFT)",
     "neutral-sft-v3-olmo3-7b/neutral-sft-v3-olmo3-7b-9e781b0f5a83", "olmo_dpo"),
    ("qwen_30b", "Qwen3-30B-A3B (neutral SFT)",
     "neutral-sft-v3-qwen3-30b-a3b/neutral-sft-v3-qwen3-30b-a3b-f733f4b5eb15", "qwen_dpo_30b"),
    ("olmo_30b", "OLMo-3-32B (neutral SFT)",
     "neutral-sft-v3-olmo3-32b/neutral-sft-v3-olmo3-32b-af857e8259e5", "olmo_dpo_30b"),
]
# The 30-32B runs had no steering sweep: their layer is the one at the relative
# depth (0.5) the 7-8B sweeps selected.
PINNED = {"qwen_30b": 24, "olmo_30b": 32}
# Middle band used for the summary "plateau" numbers, as a fraction of depth.
BAND = (0.25, 0.75)


def sweep_layer(key: str, chk: Checker) -> tuple[int, str]:
    """(layer, source): the sweep's pick (argmax mean trait score over the
    aggregated sweep), or the pinned layer of an unswept model."""
    if key in PINNED:
        return PINNED[key], "pinned"
    rows = list(csv.DictReader(open(D / f"aggregated_layer_sweep_{key}.csv")))
    best = int(max(rows, key=lambda r: float(r["mean_trait_score"]))["layer"])
    chk.value(f"{key}.best_layer", best, load_json(D / f"best_layer_{key}.json")["layer"])
    return best, "sweep"


def check_against_rq1(key, arm, values, grids, layer, chk: Checker) -> None:
    rg = np.load(PAPER_DATA / "rq1/grids" / arm / "persona.npy")
    rv = load_json(PAPER_DATA / "rq1/grids" / arm / "persona_values.json")
    rv = rv["rows"] if isinstance(rv, dict) else rv
    idx = [values.index(v) for v in rv]
    chk.n += 1
    if np.abs(grids[layer][np.ix_(idx, idx)] - rg).max() > 1e-5:
        chk.failures.append(f"{key}: sweep-layer grid differs from the RQ1 {arm} persona grid")


def offdiag(grids: np.ndarray) -> np.ndarray:
    """[L, V, V] -> [L, n_cells] upper-triangle cells."""
    iu = np.triu_indices(grids.shape[1], k=1)
    return grids[:, iu[0], iu[1]]


def layer_stability(chk: Checker) -> dict:
    res, cells, values_by = {}, {}, {}
    for key, label, rel, arm in MODELS:
        z = np.load(D / f"grids_{key}.npz")
        values, grids = list(z["values"]), z["grids"]
        n_layers = grids.shape[0] - 1                  # index 0 = embeddings
        sweep, source = sweep_layer(key, chk)
        check_against_rq1(key, arm, values, grids, sweep, chk)
        x = offdiag(grids)[1:]                         # layers 1..n_layers
        rho = stats.spearmanr(x, axis=1).statistic     # [n_layers, n_layers]
        to_sweep = rho[sweep - 1]
        layers = np.arange(1, n_layers + 1)
        depth = layers / n_layers
        band = (depth >= BAND[0]) & (depth <= BAND[1])
        adjacent = np.array([rho[i, i + 1] for i in range(n_layers - 1)])
        res[key] = dict(
            label=label, artifact=rel.split("/")[1], n_layers=n_layers,
            n_values=len(values), n_cells=int(x.shape[1]), sweep_layer=sweep,
            layer_source=source,
            rho_layer_layer=rho.round(4).tolist(),
            rho_to_sweep=to_sweep.round(4).tolist(),
            band=list(BAND),
            band_layers=[int(layers[band][0]), int(layers[band][-1])],
            band_min_rho_to_sweep=float(to_sweep[band].min()),
            band_mean_rho_to_sweep=float(to_sweep[band].mean()),
            band_min_pairwise_rho=float(rho[np.ix_(band, band)].min()),
            adjacent_mean_rho=float(adjacent.mean()),
            adjacent_min_rho=float(adjacent.min()),
            final_layer_rho_to_sweep=float(to_sweep[-1]),
            first_layer_rho_to_sweep=float(to_sweep[0]),
        )
        cells[key] = dict(zip(values, grids[sweep]))   # row per value at sweep
        values_by[key] = values

    # Cross-model reference: the two sweep-layer grids, aligned by value name.
    common = sorted(set.intersection(*(set(v) for v in values_by.values())))
    iu = np.triu_indices(len(common), k=1)
    g = {}
    for key, *_ in MODELS:
        idx = [values_by[key].index(v) for v in common]
        full = np.stack([cells[key][values_by[key][i]] for i in idx])[:, idx]
        g[key] = full[iu]
    res["cross_model_rho_at_sweep"] = float(stats.spearmanr(g["qwen"], g["olmo"]).statistic)
    res["cross_model_rho_at_selected"] = {
        f"{a}_vs_{b}": float(stats.spearmanr(g[a], g[b]).statistic)
        for i, (a, *_) in enumerate(MODELS) for b, *_ in MODELS[i + 1:]}
    res["pooling"] = POOLING
    return res


def main() -> int:
    require_paper_data()
    chk = Checker("Appendix D")
    res = layer_stability(chk)
    # Same serialization as the figure input (json.dumps(indent=1), no newline).
    write("appd_layer_stability.json", json.dumps(res, indent=1))
    chk.n += 1
    if (OUT / "appd_layer_stability.json").read_bytes() != FIG_INPUT.read_bytes():
        chk.failures.append(f"appd_layer_stability.json: differs from {FIG_INPUT.relative_to(ROOT)}")
        chk.value("layer_stability", res, load_json(FIG_INPUT))   # say where
    for key, *_ in MODELS:
        r = res[key]
        print(f"{r['label']:<28} {r['layer_source']} L{r['sweep_layer']}/{r['n_layers']}  "
              f"band L{r['band_layers'][0]}-{r['band_layers'][1]}: "
              f"rho-to-sweep min {r['band_min_rho_to_sweep']:.3f} mean {r['band_mean_rho_to_sweep']:.3f}")
    for pair, r in res["cross_model_rho_at_selected"].items():
        print(f"cross-model rho at selected layers, {pair}: {r:.3f}")
    return chk.finish()


if __name__ == "__main__":
    sys.exit(main())
