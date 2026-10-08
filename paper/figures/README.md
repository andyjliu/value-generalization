# Paper figures

The paper's 18 figures, one directory per paper section (`rq*` = research
questions, `app*` = appendices). Self-contained: each figure directory's
`inputs/` holds the cached json/csv/npy it plots, so nothing outside this
directory is read. The only
non-plotting dependency is the `valuegen` package (`.venvs/core`).

`paper/reproduce/all.py` rebuilds everything (after recomputing the RQ results). Each figure is `<name>.pdf` (the paper file:
vector, embedded TrueType fonts) plus a 300 dpi `<name>.png` for checking.

All styling lives in `style.py`:

| concern | rule |
|---|---|
| size | 6.5 in wide (\textwidth), 11 pt base, 10 pt ticks and legends, 9.5 pt annotations. The appE heatmaps keep their 12 in canvas, with the chrome scaled by `HEATMAP_SCALE` |
| titles | none inside the figure; multi-panel figures title their panels |
| categorical color | one 8-slot palette, `CAT`, used in order. Recurring things keep a fixed slot via `ENTITY` (predictors, OLMo/Qwen, the 4 taxonomy clusters) |
| heatmap color | `HEAT_DIV` (blue ↔ gray ↔ red) for signed values; `HEAT_SEQ` (its red half) for fractions and counts |
| per-type chrome | `setup_axes(ax, kind)`, where kind is bar, line, scatter, map or heatmap |
| legend | `legend_below`: frameless, centered under the figure |
| point labels | `place_labels`: greedy, deterministic, collision-avoiding |

Palette check (dataviz `validate_palette.js`, light mode): `CAT[:5]` passes
the colorblind and normal-vision checks for bars and lines; `CAT[:4]` passes
them for scatters, where every pair of colors must be distinguishable.

Aqua and yellow are below 3:1 contrast on white, so marks in those colors also
carry a marker shape, a direct label or a value label.
