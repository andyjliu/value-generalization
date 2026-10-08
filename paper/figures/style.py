"""House style for the paper figures; the rules are in README.md."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


WIDTH = 6.5
HEIGHT = {"bar": 3.2, "line": 3.4, "scatter": 3.6, "map": 3.4, "map_single": 3.9}
BASE_PT = 11
RC = {
    "font.family": "DejaVu Sans",
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT + 1,
    "xtick.labelsize": BASE_PT - 1,
    "ytick.labelsize": BASE_PT - 1,
    "legend.fontsize": BASE_PT - 1,
    "legend.title_fontsize": BASE_PT,
    "figure.titlesize": BASE_PT + 1,
}
ANNOT_PT = BASE_PT - 1.5
HEATMAP_SCALE = 12 / WIDTH

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
SURFACE = "#ffffff"

BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"
MAGENTA, GREEN, RED = "#e87ba4", "#008300", "#e34948"
CAT = [BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA, GREEN, RED]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

ENTITY = {
    "persona": 0, "grad_proj": 1, "sent_emb_behavior": 2, "sent_emb_desc": 3,
    "weight_steer": 4,
    "olmo": 0, "qwen": 1,
    "personal_growth": 0, "structured_and_methodical_reasoning": 1,
    "professional_ethics_and_integrity": 2, "risk_management": 3,
}


def color(key):
    """Color of a recurring entity (ENTITY) or of palette slot `key` (int)."""
    return CAT[key if isinstance(key, int) else ENTITY[key]]


def marker(key):
    return MARKERS[key if isinstance(key, int) else ENTITY[key]]


HEAT_MID = "#f0efec"
HEAT_DIV = LinearSegmentedColormap.from_list(
    "vg_div", ["#0d366b", BLUE, "#9ec5f4", HEAT_MID, "#f4aaa6", RED, "#8a1c1c"])
HEAT_SEQ = LinearSegmentedColormap.from_list("vg_seq", [HEAT_MID, "#f4aaa6", RED, "#8a1c1c"])


def heat_text_color(frac):
    """Readable cell-text color for position `frac` in [0, 1] along HEAT_SEQ."""
    return "white" if frac > 0.62 else INK


def use(scale=1.0):
    """Apply the house rcParams (optionally scaled for an oversize canvas)."""
    plt.rcdefaults()
    rc = {k: (v * scale if isinstance(v, (int, float)) else v) for k, v in RC.items()}
    rc.update({
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.linewidth": 0.8 * scale, "xtick.major.width": 0.8 * scale,
        "ytick.major.width": 0.8 * scale, "xtick.major.size": 3 * scale,
        "ytick.major.size": 3 * scale,
        "axes.titlepad": 6 * scale, "axes.labelpad": 4 * scale,
        "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
    plt.rcParams.update(rc)


def figure(kind, ncols=1, height=None, **kw):
    """A WIDTH-wide figure with the standard height for its plot type."""
    return plt.subplots(1, ncols, figsize=(WIDTH, height or HEIGHT[kind]), **kw)


def setup_axes(ax, kind):
    """Per-plot-type chrome. kind in {bar, line, scatter, map, heatmap}."""
    if kind == "heatmap":
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        return
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if kind == "bar":
        ax.tick_params(axis="x", length=0)
    elif kind in ("line", "scatter"):
        pass
    elif kind == "map":
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        raise ValueError(kind)


BAR_GAP = 0.03
ERR_KW = dict(fmt="none", ecolor=INK_2, elinewidth=0.9, capsize=2, capthick=0.9)
SCATTER_KW = dict(s=22, edgecolors=SURFACE, linewidths=0.4, alpha=0.9)
FIT_KW = dict(color=INK_2, lw=1.5)
LABEL_BOX = dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85)
LEADER = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=4)


def value_labels(ax, xs, tops, values, fmt="{:.2f}", pad=3):
    """Numbers above bars (or above their CI whiskers), in secondary ink."""
    for x, top, v in zip(xs, tops, values):
        ax.annotate(fmt.format(v), (x, top), xytext=(0, pad), textcoords="offset points",
                    ha="center", va="bottom", fontsize=ANNOT_PT, color=INK_2)


def _overlap(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def _seg_dist(p, a, b):
    """Pixel distance from point p to segment a-b."""
    ab, ap = np.subtract(b, a), np.subtract(p, a)
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.hypot(*(ap - t * ab)))


def place_labels(ax, items, avoid=(), step=5):
    """Greedy, deterministic label placement for point annotations.

    Call it after the final layout (tight_layout), since it measures in pixels.
    items: [(x, y, text, annotate_kwargs)] in placement order (put the most
    important first). avoid: extra (x, y) data points labels should not cover.
    Each label tries 8 directions on seven rings (the outer rings get a thin
    leader line to the point) and takes the cheapest: overlapping a placed label
    costs most, then covering a labelled/avoided point with the text or running
    a leader through one, then distance from the point; leaving the axes is
    effectively forbidden.
    """
    fig = ax.figure
    # measure at the output resolution: text widths do not scale exactly with
    # dpi, so boxes measured at screen dpi can miss by a few pixels in the PNG
    fig.set_dpi(plt.rcParams["savefig.dpi"])
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    frame = ax.get_window_extent(rend)
    anchors = [ax.transData.transform((x, y)) for x, y, *_ in items]
    anchors += [ax.transData.transform(p) for p in avoid]
    dirs = [(1, 0.6, "left", "bottom"), (-1, 0.6, "right", "bottom"),
            (1, -0.6, "left", "top"), (-1, -0.6, "right", "top"),
            (0, 1.4, "center", "bottom"), (0, -1.4, "center", "top"),
            (1.8, 0, "left", "center"), (-1.8, 0, "right", "center")]
    rings = (1, 2.4, 4, 6, 9, 13, 18)
    cands = [(dx * step * r, dy * step * r, ha, va, k)
             for k, r in enumerate(rings) for dx, dy, ha, va in dirs]
    placed = []
    for n, (x, y, text, kw) in enumerate(items):
        own = anchors[n]
        best, best_cost = None, None
        for dx, dy, ha, va, k in cands:
            a = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, arrowprops=LEADER if k > 0 else None, **kw)
            # text box only: an Annotation's own extent includes its leader line,
            # whose bounding rectangle would count as overlapping everything nearby
            a.update_positions(rend)
            bb = mtext.Text.get_window_extent(a, rend).padded(4 * fig.dpi / 100)
            cost = 50 * sum(_overlap(bb, p) for p in placed)
            cost += 3000 * sum(bb.contains(px, py) for px, py in anchors)
            if k > 0:
                end = (min(max(own[0], bb.x0), bb.x1), min(max(own[1], bb.y0), bb.y1))
                cost += 400 * sum(_seg_dist(p, own, end) < 6 * fig.dpi / 100 for i, p in enumerate(anchors)
                                  if i != n)
                seg = [np.add(own, t * np.subtract(end, own)) for t in np.linspace(0, 1, 25)]
                cost += 2000 * sum(any(p.contains(*q) for q in seg) for p in placed)
            cost += 60 * k
            if not (frame.contains(bb.x0, bb.y0) and frame.contains(bb.x1, bb.y1)):
                cost += 1e6
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best.remove()
                best, best_cost = a, cost
            else:
                a.remove()
        best.update_positions(rend)
        placed.append(mtext.Text.get_window_extent(best, rend).padded(4 * fig.dpi / 100))
    return placed


def legend_below(fig, handles=None, labels=None, ncol=None, title=None, y=0.035):
    """The standard legend: frameless, centered under the whole figure."""
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=ncol or len(handles), title=title)


SIDE_LEGEND_X = 0.72


def legend_side(fig, handles, labels, x=SIDE_LEGEND_X, **kw):
    """Legend in the right-hand margin, vertically centered (frameless unless
    kw overrides it)."""
    return fig.legend(handles, labels, loc="center left", bbox_to_anchor=(x, 0.5),
                      **{"labelspacing": 1.0, **kw})


def save(fig, stem):
    """<stem>.pdf (the paper file) + <stem>.png (300 dpi backup for review)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf (+ .png)")
