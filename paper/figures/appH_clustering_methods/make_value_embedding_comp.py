"""appH: cluster quality z vs k per value embedding."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S                                        # noqa: E402

SRC = HERE / "inputs/value-embedding-comp.json"
EMB = [("persona", "Persona"), ("grad_proj", "Gradient"),
       ("sent_emb_behavior", "Behavior-Embd"), ("sent_emb_desc", "Description-Embd"),
       ("weight_steer", "Weight")]
K_MAX = 8


def main():
    S.use()
    curves = json.load(open(SRC))["curves"]
    fig, ax = S.figure("line")
    S.setup_axes(ax, "line")
    for key, label in EMB:
        k, z = zip(*[(k, z) for k, z in zip(curves[key]["k"], curves[key]["z"]) if k <= K_MAX])
        ax.plot(k, z, marker=S.marker(key), color=S.color(key), label=label,
                mec=S.SURFACE, mew=0.6)
    ax.set_xticks(k)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Cluster quality (z-score)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    S.save(fig, HERE / "value-embedding-comp")


if __name__ == "__main__":
    main()
