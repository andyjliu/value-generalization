"""appF: Persona | Gradient bar figures (dpo_pairs_bars, sft_base_bars)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

SRC = HERE / "inputs"
FAMILY = {"olmo_dpo": "Olmo", "olmo_sft": "Olmo", "qwen_dpo": "Qwen", "qwen_sft": "Qwen"}
PREDS = ["persona", "grad_proj"]
PRED_LABELS = {"persona": "Persona", "grad_proj": "Gradient"}
FIGS = {
    "dpo_pairs_bars": dict(
        arms=["olmo_dpo", "qwen_dpo"], srcs=["generic", "dpo_pairs"],
        src_labels={"generic": "Generic elicitation data", "dpo_pairs": "DPO training pairs"}),
    "sft_base_bars": dict(
        arms=["olmo_sft", "qwen_sft"], srcs=["neutral", "base_reads_neutral", "base_reads_base"],
        src_labels={"neutral": "Neutral-SFT", "base_reads_neutral": "Base reading Neutral-SFT",
                    "base_reads_base": "Base"}),
}
YMAX = 0.8


def bar_figure(name, arms, srcs, src_labels):
    rho = json.load(open(SRC / f"{name}.json"))["rho"]
    fig, axes = S.figure("bar", ncols=len(PREDS), sharey=True)
    n = len(srcs)
    w = 0.84 / n
    x = np.arange(len(arms))
    for ax, pred in zip(axes, PREDS):
        S.setup_axes(ax, "bar")
        for j, src in enumerate(srcs):
            r = [rho[f"{a}|{pred}|{src}"] for a in arms]
            h = np.array([v["rho"] for v in r])
            lo = np.array([v["lo"] for v in r])
            hi = np.array([v["hi"] for v in r])
            xs = x + (j - (n - 1) / 2) * w
            ax.bar(xs, h, w - S.BAR_GAP, color=S.color(j), label=src_labels[src], zorder=2)
            ax.errorbar(xs, h, yerr=[h - lo, hi - h], zorder=3, **S.ERR_KW)
            S.value_labels(ax, xs, hi, h)
        ax.set_xticks(x, [FAMILY[a] for a in arms])
        ax.set_xlim(-0.5, len(arms) - 0.5)
        ax.set_ylim(0, YMAX)
        ax.set_title(PRED_LABELS[pred])
    axes[0].set_ylabel(r"Off-diagonal Spearman $\rho$")
    fig.tight_layout(w_pad=1.5)
    S.legend_below(fig)
    S.save(fig, HERE / name)


def main():
    S.use()
    for name, kw in FIGS.items():
        bar_figure(name, **kw)


if __name__ == "__main__":
    main()
