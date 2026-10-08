"""appE: steerability heatmaps (ds metric) for OLMo/Qwen x DPO/SFT, one file each.

The bare arm names are the 7-8B models; the `_30b` ones are OLMo-3-32B and
Qwen3-30B-A3B.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402
from valuegen.analysis import correlate as C            # noqa: E402

DATA = HERE / "inputs"
TENET_PT = 8


def load_ds_target(arm):
    """ds-metric matrix: the likert ``_normalized`` matrix."""
    stem = DATA / f"{arm}_likert"
    M = np.load(f"{stem}_normalized.npy")
    v = json.load(open(f"{stem}_raw_values.json"))
    r, c = (v["rows"], v["cols"]) if isinstance(v, dict) else (v, v)
    return M, r, c


def main():
    S.use(scale=S.HEATMAP_SCALE)
    _, rows49, cols66 = load_ds_target("qwen_dpo")
    assert len(rows49) == 49 and len(cols66) == 66
    n49 = len(rows49)
    all66 = rows49 + [c for c in cols66 if c not in set(rows49)]
    arms = product(("", "_30b"), ("olmo", "qwen"), (("dpo", rows49), ("sft", all66)))
    for suf, fam, (method, rows) in arms:
        arm = f"{fam}_{method}{suf}"
        m, r, c = load_ds_target(arm)
        M = C.reindex(m, r, c, rows, all66)
        assert np.isfinite(M).all(), arm
        fig, ax = S.plt.subplots(figsize=(12, 9.5 * len(rows) / 49))
        S.setup_axes(ax, "heatmap")
        im = ax.imshow(M, cmap=S.HEAT_DIV, vmin=-1, vmax=1, aspect="equal",
                       interpolation="nearest")
        ax.axvline(n49 - 0.5, color=S.INK, lw=0.8, ls="--")
        if len(rows) > n49:
            ax.axhline(n49 - 0.5, color=S.INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(all66)), all66, rotation=90, fontsize=TENET_PT)
        ax.set_yticks(range(len(rows)), rows, fontsize=TENET_PT)
        ax.set_xlabel("Evaluated tenet")
        ax.set_ylabel("Trained tenet")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Steerability")
        cb.outline.set_visible(False)
        S.save(fig, HERE / f"heatmap_{arm}")


if __name__ == "__main__":
    main()
