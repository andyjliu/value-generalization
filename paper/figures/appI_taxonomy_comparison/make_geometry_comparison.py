"""appI: Procrustes-aligned Olmo-32B and Qwen-8B persona maps of the 266 VITW values."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import procrustes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

DATA = HERE / "inputs"
VALUES = {m: DATA / f"persona_values_{m}.json" for m in ("olmo3_32b", "qwen3_8b")}
CLUSTERS_32B = DATA / "clusters_k4_olmo3_32b.csv"
TAXON = {
    "personal_growth": "Attunement",
    "structured_and_methodical_reasoning": "Rigor",
    "professional_ethics_and_integrity": "Integrity",
    "risk_management": "Stewardship",
}


def main():
    S.use()
    values = json.load(open(VALUES["olmo3_32b"]))
    assert json.load(open(VALUES["qwen3_8b"])) == values, "grids must share value order"
    X, Y, m2 = procrustes(np.load(DATA / "mds_nonmetric_olmo3_32b.npy"),
                          np.load(DATA / "mds_nonmetric_qwen3_8b.npy"))
    rows = list(csv.DictReader(open(CLUSTERS_32B)))
    medoid = {r["cluster"]: r["value"] for r in rows if r["is_medoid"] == "1"}
    assert set(medoid.values()) == set(TAXON), medoid
    key_of = {r["value"]: medoid[r["cluster"]] for r in rows}
    colors = [S.color(key_of[v]) for v in values]

    fig, axes = S.figure("map", ncols=2)
    for ax, C, title in ((axes[0], X, "ValueMap-Olmo-32B"), (axes[1], Y, "ValueMap-Qwen-8B")):
        S.setup_axes(ax, "map")
        ax.scatter(C[:, 0], C[:, 1], c=colors, **{**S.SCATTER_KW, "s": 12, "alpha": 0.8})
        ax.set_title(title)
        ax.set_xlabel("MDS dimension 1")
        ax.set_ylabel("MDS dimension 2")
        ax.set_aspect("equal", adjustable="datalim")
    handles = [Line2D([], [], marker="o", ls="", color=S.color(k), label=n)
               for k, n in TAXON.items()]
    fig.tight_layout()
    S.legend_below(fig, handles, list(TAXON.values()), ncol=2,
                   title="ValueMap-Olmo-32B clusters")
    S.save(fig, HERE / "geometry_comparison")
    print(f"procrustes r = {np.sqrt(1 - m2):.4f}")


if __name__ == "__main__":
    main()
