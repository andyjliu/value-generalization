"""appD: persona-grid correlation with the selected layer vs relative depth.

Solid lines are the 7-8B models, whose layer the steering sweep selected;
dashed lines are the 30-32B models, whose layer was pinned at the same
relative depth.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

SRC = HERE / "inputs/layer_stability.json"
# (json key, family, label, linestyle)
MODELS = [("olmo", "olmo", "Olmo-3-7B", "-"), ("qwen", "qwen", "Qwen3-8B", "-"),
          ("olmo_30b", "olmo", "Olmo-3-32B", "--"), ("qwen_30b", "qwen", "Qwen3-30B-A3B", "--")]


def main():
    S.use()
    res = json.load(open(SRC))
    fig, ax = S.figure("line")
    S.setup_axes(ax, "line")
    for key, fam, label, ls in MODELS:
        r = res[key]
        n, s = r["n_layers"], r["sweep_layer"]
        assert s / n == 0.5, key          # the single marker below stands for all four
        depth = np.arange(1, n + 1) / n
        ax.plot(depth, r["rho_to_sweep"], color=S.color(fam), ls=ls, label=f"{label}, L{s}")
    ax.plot(0.5, 1.0, "o", ms=7, color=S.INK_2, mec=S.SURFACE, mew=1.2, zorder=3)
    ax.annotate("selected layer", (0.5, 1.0), xytext=(0.5, 0.84), ha="center", va="top",
                fontsize=S.ANNOT_PT, color=S.INK_2, arrowprops=S.LEADER)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Correlation with selected layer")
    fig.tight_layout()
    S.legend_below(fig)
    S.save(fig, HERE / "layer_stability")


if __name__ == "__main__":
    main()
