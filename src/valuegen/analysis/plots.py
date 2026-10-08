"""Shared matplotlib helpers: heatmaps, scatters, bars, MDS maps, silhouettes.

One home for the figure styles. Every
helper draws on a provided ``ax`` when given (grids compose in the caller) or
makes its own single-axes figure; :func:`save` writes at the house dpi.

matplotlib is imported lazily with the Agg backend so importing
``valuegen.analysis`` never needs a display or the ``[analysis]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _plt().close(fig)
    print(f"Saved: {path}")
    return path


def _own_ax(ax, figsize):
    plt = _plt()
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Heatmaps ─────────────────────────────────────────────────────────────────


def heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
    diverging: bool = False,
    cmap: str | None = None,
    annotate: bool = True,
    fmt: str = ".2f",
    boundaries: Sequence[int] | None = None,
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Annotated matrix heatmap.

    ``diverging`` centers a RdBu_r scale on 0 (steerability matrices);
    the default sequential scale (YlOrRd) suits similarity grids.
    ``boundaries`` draws category separators after the given row/col indices
    (e.g. the 9-constraint manipulation | deception | bias triads).
    """
    col_labels = list(col_labels) if col_labels is not None else list(row_labels)
    fig, ax = _own_ax(ax, (0.55 * len(col_labels) + 2.5, 0.55 * len(row_labels) + 2))
    matrix = np.asarray(matrix, dtype=float)
    if diverging:
        bound = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.05)
        im = ax.imshow(matrix, cmap=cmap or "RdBu_r",
                       vmin=-bound if vmin is None else vmin,
                       vmax=bound if vmax is None else vmax, aspect="equal")
    else:
        im = ax.imshow(matrix, cmap=cmap or "YlOrRd", vmin=vmin, vmax=vmax,
                       aspect="equal")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(row_labels, fontsize=7)
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, format(matrix[i, j], fmt),
                            ha="center", va="center", fontsize=6)
    for b in boundaries or []:
        ax.axhline(b - 0.5, color="black", linewidth=2)
        ax.axvline(b - 0.5, color="black", linewidth=2)
    if title:
        ax.set_title(title, fontsize=11)
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.7)
    return fig, ax


# ── Scatters ─────────────────────────────────────────────────────────────────


def scatter_fit(
    x: np.ndarray,
    y: np.ndarray,
    ax=None,
    xlabel: str = "Cosine similarity",
    ylabel: str = "Likert-normalized steerability",
    title: str | None = None,
    annotate_corr: bool = True,
):
    """Predictor-vs-target cell scatter with a least-squares fit line and the
    Spearman ρ in the title (the persona_vec_generalization panel)."""
    from scipy import stats

    fig, ax = _own_ax(ax, (5.5, 5))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    ax.scatter(x, y, alpha=0.45, s=30, color="steelblue", edgecolors="none")
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, "r--", alpha=0.6, linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    label = title or ""
    if annotate_corr and len(x) >= 3:
        rho, p = stats.spearmanr(x, y)
        label = f"{title}\n" if title else ""
        label += f"ρ={rho:.3f} (p={p:.3f})"
    if label:
        ax.set_title(label, fontsize=10)
    _despine(ax)
    return fig, ax


# ── Bars ─────────────────────────────────────────────────────────────────────


def grouped_bar(
    group_labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    ax=None,
    ylabel: str = "Spearman ρ",
    title: str | None = None,
    annotate: bool = True,
    colors: Sequence[str] | None = None,
    errors: Mapping[str, Sequence[float]] | None = None,
):
    """Grouped bars (e.g. persona vs sentence-emb ρ per model), zero line,
    value labels above/below each bar."""
    fig, ax = _own_ax(ax, (max(6.0, 1.7 * len(group_labels)), 5))
    x = np.arange(len(group_labels))
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    default_colors = ["#2c7fb8", "#d95f0e", "#3c896d", "#b0b0b0"]
    for s, (name, vals) in enumerate(series.items()):
        offset = (s - (n_series - 1) / 2) * width
        yerr = list(errors[name]) if errors and name in errors else None
        bars = ax.bar(x + offset, list(vals), width, label=name,
                      color=(colors or default_colors)[s % len(colors or default_colors)],
                      yerr=yerr, capsize=3 if yerr else 0,
                      edgecolor="black", linewidth=0.5)
        if annotate:
            for bar in bars:
                h = bar.get_height()
                if not np.isfinite(h):
                    continue
                nudge = 0.008 if h >= 0 else -0.008
                ax.text(bar.get_x() + bar.get_width() / 2, h + nudge,
                        f"{h:+.2f}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    _despine(ax)
    return fig, ax


# ── Embedding maps ───────────────────────────────────────────────────────────


def embedding_map(
    coords: np.ndarray,
    labels: Sequence[str],
    ax=None,
    title: str | None = None,
    colors: Sequence | None = None,
    color_values: Sequence[float] | None = None,
    color_label: str | None = None,
    xlabel: str = "dim 1",
    ylabel: str = "dim 2",
):
    """2D MDS/spectral scatter with per-point text labels.

    ``colors`` gives discrete per-point colors (category triads);
    ``color_values`` maps a continuous gradient (viridis + colorbar), e.g.
    mean off-diagonal cosine as a transmissivity proxy.
    """
    fig, ax = _own_ax(ax, (8.0, 7.0))
    kwargs = dict(s=110, zorder=3, edgecolors="black", linewidth=0.5)
    if color_values is not None:
        sctr = ax.scatter(coords[:, 0], coords[:, 1], c=list(color_values),
                          cmap="viridis", **kwargs)
        cb = fig.colorbar(sctr, ax=ax, fraction=0.046, pad=0.04)
        if color_label:
            cb.set_label(color_label, fontsize=8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=list(colors) if colors is not None else "#2c7fb8", **kwargs)
    for i, label in enumerate(labels):
        ax.annotate(label, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    _despine(ax)
    return fig, ax


def silhouette_plot(
    ks: Sequence[int],
    curves: Mapping[str, Sequence[float]],
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
    baseline_kind: str = "sd",
    ax=None,
    title: str | None = None,
):
    """Silhouette-vs-k curves over a chance band.

    ``baseline`` is (mean, spread); ``baseline_kind="sd"`` shades mean ± sd
    (uniform-points baseline), ``"p95"`` shades mean → 95th percentile
    (random-partition baseline).
    """
    fig, ax = _own_ax(ax, (6.5, 5.0))
    styles = ["o-", "s-", "^-", "d-", "v-", "*-"]
    for s, (name, sils) in enumerate(curves.items()):
        ax.plot(list(ks), list(sils), styles[s % len(styles)], label=name)
    if baseline is not None:
        mean, spread = baseline
        ax.plot(list(ks), mean, "--", color="gray", linewidth=1.5,
                label="random baseline")
        lo = mean - spread if baseline_kind == "sd" else mean
        hi = mean + spread if baseline_kind == "sd" else spread
        ax.fill_between(list(ks), lo, hi, color="gray", alpha=0.2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("silhouette score")
    ax.set_xticks(list(ks))
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    _despine(ax)
    return fig, ax
