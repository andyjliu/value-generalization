"""RQ3: the vitw-266 Olmo-32B k=4 persona taxonomy as a nonmetric-MDS map."""
import csv
import textwrap
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402
from valuegen.analysis import mds as M_                  # noqa: E402

GRID = HERE / "inputs/persona_L32_olmo3_32b.npy"
CSV = HERE / "inputs/clusters_k4_olmo3_32b.csv"
LEGEND_NAME = {
    "personal_growth": "Attunement",
    "structured_and_methodical_reasoning": "Rigor",
    "professional_ethics_and_integrity": "Integrity",
    "risk_management": "Stewardship",
}
N_LABEL = 2
# at the hull tip; their leaders ran the length of the map
SKIP_LABEL = {"regulatory_adherence_and_compliance", "procedural_adherence"}
LEGEND_X = 0.755

# medoid -> display name (keyed on the medoid so permuted cluster ids fail loudly)
NAME = {
    "personal_growth": "Attunement",
    "structured_and_methodical_reasoning": "Rigor",
    "professional_ethics_and_integrity": "Integrity",
    "risk_management": "Stewardship",
}
def load_grid():
    v = json.load(open(str(GRID)[:-4] + "_values.json"))
    values = v["rows"] if isinstance(v, dict) else v
    return M_.drop_nan_values(np.load(GRID), values)


def load_labels(values):
    idx = {v: i for i, v in enumerate(values)}
    lab = np.full(len(values), -1)
    medoid = {}
    for r in csv.DictReader(open(CSV)):
        i = idx[r["value"]]
        lab[i] = int(r["cluster"])
        if r["is_medoid"] in ("1", "True", "true"):
            medoid[int(r["cluster"])] = i
    return lab, medoid


def central_examples(dist, members, medoid_i, n, coords, hull=None, other_hulls=(),
                     central_frac=0.5, min_sep=0.12):
    """n example values for a cluster (the medoid itself is not labelled).
    Candidates are the `central_frac` of the cluster closest to the medoid in
    feature space that also sit inside the cluster's own drawn hull (and, where
    possible, outside every other cluster's hull). Among them, take the ones
    nearest the medoid on the map that are at least `min_sep` (fraction of the
    map's span) from each other, so the two labels do not collide."""
    span = np.ptp(coords, axis=0).max()
    others = sorted((m for m in members if m != medoid_i), key=lambda j: dist[medoid_i, j])
    others = others[: max(n, int(len(others) * central_frac))]
    pools = [others]
    if hull is not None:
        inside = [m for m in others if hull.contains_point(coords[m])]
        own = [m for m in inside if not any(h.contains_point(coords[m]) for h in other_hulls)]
        pools = [own, inside, others]
    picks = []
    for pool in pools:
        for j in sorted(pool, key=lambda j: np.hypot(*(coords[j] - coords[medoid_i]))):
            if len(picks) == n:
                break
            if j not in picks and all(np.hypot(*(coords[j] - coords[q])) >= min_sep * span
                                      for q in picks):
                picks.append(j)
    return picks


def tinted_box(color, tint=0.18):
    """LABEL_BOX with its fill lightly shaded toward the cluster color, so each
    label reads as belonging to its cluster."""
    fc = (1 - tint) * np.array(to_rgb(S.SURFACE)) + tint * np.array(to_rgb(color))
    return {**S.LABEL_BOX, "fc": tuple(fc), "alpha": 0.92}


def draw_hull(ax, pts, color, keep=0.90, pad=1.05):
    """Hull of the central `keep` fraction of a cluster's points (returned as a Path)."""
    if len(pts) < 3:
        return None
    ctr = pts.mean(axis=0)
    r = np.hypot(*(pts - ctr).T)
    core = pts[r <= np.quantile(r, keep)]
    if len(core) < 3:
        core = pts
    poly = ctr + (core[ConvexHull(core).vertices] - ctr) * pad
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.08, zorder=1, lw=0)
    ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
            color=color, lw=0.9, alpha=0.5, zorder=1)
    return MplPath(poly)


def main():
    S.use()
    sim, values = load_grid()
    lab, medoid = load_labels(values)
    dist = M_.cos_to_dist(sim)
    coords, stress = M_.mds_variant(sim, "nonmetric")
    order = sorted(medoid, key=lambda c: list(NAME).index(values[medoid[c]]))

    fig, ax = S.figure("map", height=S.HEIGHT["map_single"])
    S.setup_axes(ax, "map")
    handles, hulls = [], {}
    for c in order:
        key = values[medoid[c]]
        hulls[c] = draw_hull(ax, coords[lab == c], S.color(key))
    for c in order:
        key = values[medoid[c]]
        sel = lab == c
        ax.scatter(coords[sel, 0], coords[sel, 1], color=S.color(key), zorder=2,
                   **{**S.SCATTER_KW, "s": 14, "alpha": 0.55})
        handles.append(Line2D([], [], marker="o", ls="", color=S.color(key),
                              label=f"{LEGEND_NAME[key]} (n = {int(sel.sum())})"))
    labels = []
    for c in order:
        key = values[medoid[c]]
        members = [i for i in np.where(lab == c)[0].tolist() if values[i] not in SKIP_LABEL
                   or i == medoid[c]]
        for i in central_examples(dist, members, medoid[c], N_LABEL, coords, hulls[c],
                                   [h for d, h in hulls.items() if d != c and h is not None]):
            ax.scatter(*coords[i], s=26, color=S.color(key), edgecolors=S.INK,
                       linewidths=0.6, zorder=5)
            labels.append((*coords[i], textwrap.fill(values[i].replace("_", " "), 22), dict(
                zorder=6, fontsize=S.ANNOT_PT, color=S.INK_2, bbox=tinted_box(S.color(key)),
                linespacing=1.0)))
    # no equal aspect: MDS dimension 1 is stretched to fill the width (the
    # axes carry no units, and the caption names the projection)
    fig.tight_layout(rect=(0, 0, LEGEND_X, 1))
    S.place_labels(ax, labels)
    S.legend_side(fig, handles, [h.get_label() for h in handles], x=LEGEND_X,
                  title="Cluster names", frameon=True, edgecolor=S.MUTED,
                  fancybox=False, borderpad=0.7, labelspacing=0.8)
    S.save(fig, HERE / "taxonomy_k4_map_olmo3_32b")
    print(f"nonmetric-MDS stress={stress:.3f}")


if __name__ == "__main__":
    main()
