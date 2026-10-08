"""appC: ConflictScope scenario quality by generation model."""
import sys
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

SRC = HERE / "inputs/agreement_polarization.csv"
QUALITY_MODELS = ["gpt-4.1", "gpt-5", "gpt-5.2", "gpt-5.5", "Llama-3.3-70B",
                  "claude-3.5-sonnet", "Qwen3.6-27B"]
OPEN_WEIGHT = {"Llama-3.3-70B", "Qwen3.6-27B"}
BASELINES = {"gpt-4.1", "claude-3.5-sonnet"}
OFFSET = {"claude-3.5-sonnet": (-8, -26), "Qwen3.6-27B": (0, 9), "gpt-5": (5, -12),
          "gpt-5.5": (-5, -13)}
LEADER = {"claude-3.5-sonnet"}
HA = {"claude-3.5-sonnet": "right", "Qwen3.6-27B": "center", "gpt-5.5": "right"}


def main():
    S.use()
    res = pd.read_csv(SRC).set_index("generating_model").loc[QUALITY_MODELS]
    fig, ax = S.figure("scatter")
    S.setup_axes(ax, "scatter")
    for label, r in res.iterrows():
        c = S.color(1 if label in OPEN_WEIGHT else 0)
        base = label in BASELINES
        ax.errorbar(r["agreement"], r["likert_diff_rate"],
                    xerr=[[r["agreement"] - r["agree_lo"]], [r["agree_hi"] - r["agreement"]]],
                    yerr=[[r["likert_diff_rate"] - r["ldr_lo"]],
                          [r["ldr_hi"] - r["likert_diff_rate"]]],
                    fmt="*" if base else "o", ms=12 if base else 7, color=c,
                    mec=S.SURFACE, mew=0.6, ecolor=c, elinewidth=0.9, capsize=2, zorder=3)
        ax.annotate(label, (r["agreement"], r["likert_diff_rate"]),
                    xytext=OFFSET.get(label, (5, 4)), textcoords="offset points",
                    ha=HA.get(label, "left"), fontsize=S.ANNOT_PT, color=S.INK_2,
                    bbox=S.LABEL_BOX, zorder=4,
                    arrowprops=S.LEADER if label in LEADER else None)
    ax.set_xlabel("Observed agreement rate (← better)")
    ax.set_ylabel("Likert difference rate (→ better)")
    ax.margins(x=0.12, y=0.08)
    handles = [
        Line2D([], [], marker="o", ls="", ms=7, color=S.color(0), mec=S.SURFACE, label="Closed-weight"),
        Line2D([], [], marker="o", ls="", ms=7, color=S.color(1), mec=S.SURFACE, label="Open-weight"),
        Line2D([], [], marker="*", ls="", ms=11, color=S.INK_2, mec=S.SURFACE,
               label="ConflictScope baseline"),
    ]
    fig.tight_layout()
    S.legend_below(fig, handles, [h.get_label() for h in handles])
    S.save(fig, HERE / "conflictscope-generation-comparison")


if __name__ == "__main__":
    main()
