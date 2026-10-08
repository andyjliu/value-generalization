"""appE: mean diagonal vs mean |off-diagonal| steerability per intervention.

One group per model, DPO and SFT side by side, all on the common 49×66 frame
(the json also carries the SFT 66×66 numbers; they are not drawn).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

SRC = HERE / "inputs/diag_offdiag.json"
# (model label, family, json key suffix)
MODELS = [("Olmo-3-7B", "olmo", ""), ("Olmo-3-32B", "olmo", "_30b"),
          ("Qwen3-8B", "qwen", ""), ("Qwen3-30B-A3B", "qwen", "_30b")]
# (key template, tick label)
CONDS = [("{fam}_dpo{suf}", "DPO"), ("{fam}_sft{suf}_49x66", "SFT")]
X = np.array([g * 2.5 + c for g in range(len(MODELS)) for c in range(len(CONDS))])
SERIES = [("diag_mean", "diag_ci95", "Mean diagonal (direct effect)"),
          ("offdiag_abs_mean", "offdiag_abs_ci95", "Mean |off-diagonal| (transfer)")]


def main():
    S.use()
    res = json.load(open(SRC))
    keys = [k.format(fam=fam, suf=suf) for _, fam, suf in MODELS for k, _ in CONDS]
    fig, ax = S.figure("bar")
    S.setup_axes(ax, "bar")
    w = 0.84 / len(SERIES)
    for j, (mk, ck, label) in enumerate(SERIES):
        m = np.array([res[k][mk] for k in keys])
        lo, hi = np.array([res[k][ck] for k in keys]).T
        xs = X + (j - (len(SERIES) - 1) / 2) * w
        ax.bar(xs, m, w - S.BAR_GAP, color=S.color(j), label=label, zorder=2)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], zorder=3, **S.ERR_KW)
        # the transfer labels sit right of centre, off the neighbouring CI whisker
        for x, top, v in zip(xs, hi, m):
            ax.annotate(f"{v:.2f}", (x, top), xytext=(5 * j, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=S.ANNOT_PT, color=S.INK_2)
    ax.set_xticks(X, [lab for _ in MODELS for _, lab in CONDS])
    for g, (name, *_) in enumerate(MODELS):
        ax.annotate(name, (X[2 * g:2 * g + 2].mean(), 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -20), textcoords="offset points", ha="center", va="top")
    ax.set_xlim(X[0] - 0.7, X[-1] + 0.7)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Steerability")
    fig.tight_layout()
    S.legend_below(fig, y=0.0)
    S.save(fig, HERE / "diag_offdiag")


if __name__ == "__main__":
    main()
