"""RQ2: persona tightness vs prefill robustness."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

SRC = HERE / "inputs/joined.csv"


def main():
    S.use()
    df = pd.read_csv(SRC)
    d = df[(df["kind"] != "base") & (df["prefill.retention_norm.complete"] == True)  # noqa: E712
           & df["persona.tightness"].notna()]
    x = d["persona.tightness"].astype(float).values
    y = d["prefill.retention_norm.estimate"].astype(float).values
    rho, p = stats.spearmanr(x, y)
    fit = stats.linregress(x, y)
    grid = np.linspace(x.min(), x.max(), 200)

    fig, ax = S.figure("scatter")
    S.setup_axes(ax, "scatter")
    ax.scatter(x, y, color=S.color(0), zorder=3, **S.SCATTER_KW)
    ax.plot(grid, fit.intercept + fit.slope * grid, zorder=2, **S.FIT_KW)
    ax.set_xlabel("Alignment target coherence (persona vector embeddings)")
    ax.set_ylabel("Prefill robustness (target\nadherence after adversarial prefill)")
    fig.tight_layout()
    S.save(fig, HERE / "prefill_robustness_scatter")
    print(f"n={len(d)} Spearman rho={rho:.3f} p={p:.2g}")


if __name__ == "__main__":
    main()
