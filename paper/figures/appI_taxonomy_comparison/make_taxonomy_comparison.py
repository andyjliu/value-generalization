"""appI: Olmo-32B taxonomy vs Qwen-8B taxonomy and VITW L3, as contingency heatmaps."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

CLUSTERS = {m: HERE / f"inputs/clusters_k4_{m}.csv" for m in ("olmo3_32b", "qwen3_8b")}
VITW_META = HERE / "inputs/vitw_meta.json"
# keyed on the medoid so permuted cluster ids fail loudly
NAMES = {
    "olmo3_32b": {
        "personal_growth": "Attunement",
        "structured_and_methodical_reasoning": "Rigor",
        "professional_ethics_and_integrity": "Integrity",
        "risk_management": "Stewardship",
    },
    "qwen3_8b": {
        "leadership_qualities": "Prosocial conduct,\ncare & governance",
        "structured_and_methodical_reasoning": "Rigor, competence\n& execution",
        "professional_and_intellectual_integrity": "Integrity, honesty\n& boundaries",
        "emotional_authenticity_and_openness": "Emotional warmth\n& creative expression",
    },
}
L3_ORDER = ["Practical values", "Epistemic values", "Social values",
            "Protective values", "Personal values"]


def load(model):
    rows = list(csv.DictReader(open(CLUSTERS[model])))
    medoid = {r["cluster"]: r["value"] for r in rows if r["is_medoid"] == "1"}
    assert set(medoid.values()) == set(NAMES[model]), (model, medoid)
    return {r["value"]: NAMES[model][medoid[r["cluster"]]] for r in rows}


def contingency(ax, rows, cols, row_names, col_names):
    ri = {c: i for i, c in enumerate(row_names)}
    ci = {c: i for i, c in enumerate(col_names)}
    M = np.zeros((len(row_names), len(col_names)), int)
    for v in sorted(set(rows) & set(cols)):
        M[ri[rows[v]], ci[cols[v]]] += 1
    F = M / M.sum(1, keepdims=True)
    S.setup_axes(ax, "heatmap")
    im = ax.imshow(F, cmap=S.HEAT_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color=S.heat_text_color(F[i, j]))
    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(row_names)))
    return im


def main():
    S.use()
    l3 = {v: m["l3"] for v, m in json.load(open(VITW_META)).items()}
    olmo, qwen = load("olmo3_32b"), load("qwen3_8b")
    r_names = list(NAMES["olmo3_32b"].values())
    q_names = list(NAMES["qwen3_8b"].values())

    fig, axes = S.figure("map", ncols=2, height=3.6,
                         gridspec_kw={"width_ratios": [4, 5], "wspace": 0.06})
    contingency(axes[0], olmo, qwen, r_names, q_names)
    axes[0].set_xticklabels(q_names, rotation=40, ha="right", rotation_mode="anchor",
                            linespacing=1.0, fontsize=S.ANNOT_PT)
    axes[0].set_yticklabels(r_names, linespacing=1.0)
    axes[0].set_ylabel("ValueMap-Olmo-32B clusters")
    axes[0].set_xlabel("ValueMap-Qwen-8B clusters")
    im = contingency(axes[1], olmo, l3, r_names, L3_ORDER)
    axes[1].set_xticklabels([c.replace(" values", "") for c in L3_ORDER], rotation=40,
                            ha="right", rotation_mode="anchor")
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("ViTW L3 categories")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, aspect=25)
    cb.set_label("Fraction of row")
    cb.outline.set_visible(False)
    S.save(fig, HERE / "taxonomy_comparison")


if __name__ == "__main__":
    main()
