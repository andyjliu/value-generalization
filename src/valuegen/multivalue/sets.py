"""Arms and the k-value sets behind them: sample, preview, freeze.

Schemes (chosen at ``mv sets`` time, recorded in ``sets.json``):

- ``explicit``: the family lists its sets in the config;
- ``random``: ``n`` distinct ``k``-subsets drawn uniformly;
- ``stratified``: a candidate pool of ``k``-subsets is scored on one metric
  from one store, cut into bins, and ``n`` sets are drawn with equal counts
  per bin (extremes first when ``n`` is not a multiple). ``binning`` picks
  the cut: ``quantile`` (equal pool mass per bin, i.e. the spread of a random
  draw) or ``width`` (equal-width bins over the pool's ``trim``..``100-trim``
  percentile range, i.e. an even spread over the metric's achievable range).

Every draw is a deterministic function of ``(seed, family id, scheme)``.
Readouts are printed on every run so the go/no-go checks (metric dynamic
range, bin occupancy, design coverage of the universe, sentence-vs-persona
agreement, external reachability) happen before anything is frozen.
``sets.json`` is written only with ``--freeze`` and never rewritten without
``--force``; its bytes are the training identity's input.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from valuegen.multivalue._hashing import SAMPLE_SALT, order_key, sha256_json
from valuegen.multivalue.config import BASE_ARM, FULL_ARM, FamilyConfig, MultivalueConfig
from valuegen.multivalue.embeddings import Store
from valuegen.multivalue.metrics import METRICS, cosine_grid, score_pool

SCHEMES = ("explicit", "random", "stratified")
SETS_SCHEMA_VERSION = 1
DEFAULT_CANDIDATES = 100_000
DEFAULT_BINS = 8
BINNINGS = ("quantile", "width")


class SetsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Arm:
    id: str
    family: str  # "reference", a family id, or "import"
    values: tuple[str, ...]
    kind: str  # base | full | sampled | explicit | imported
    score: float | None = None
    bin: int | None = None
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def k(self) -> int:
        return len(self.values)

    @property
    def trained(self) -> bool:
        return self.kind != "base"


def arm_id(family_id: str, i: int) -> str:
    return f"{family_id}_{i:02d}"


# ── sampling primitives ──────────────────────────────────────────────────────


def rng_for(seed: int, *parts) -> np.random.Generator:
    return np.random.default_rng(int(order_key(SAMPLE_SALT, seed, *parts)[:16], 16))


def n_combinations(n: int, k: int) -> int:
    return math.comb(n, k)


def candidate_pool(n_universe: int, k: int, n_candidates: int, rng: np.random.Generator) -> np.ndarray:
    """``[n, k]`` distinct sorted index tuples: every combination when there
    are at most ``n_candidates`` of them, else ``n_candidates`` random ones."""
    if k < 1 or k > n_universe:
        raise SetsError(f"k={k} must be in 1..{n_universe}")
    total = n_combinations(n_universe, k)
    if total <= n_candidates:
        return np.asarray(list(itertools.combinations(range(n_universe), k)), dtype=int).reshape(total, k)
    seen: set[tuple[int, ...]] = set()
    rows: list[tuple[int, ...]] = []
    while len(rows) < n_candidates:
        # draw in batches; dedupe against everything seen so far
        batch = rng.random((max(1024, n_candidates - len(rows)), n_universe)).argsort(axis=1)[:, :k]
        batch.sort(axis=1)
        for r in map(tuple, batch):
            if r not in seen:
                seen.add(r)
                rows.append(r)
                if len(rows) == n_candidates:
                    break
    return np.asarray(rows, dtype=int)


def sample_random(n_universe: int, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    total = n_combinations(n_universe, k)
    if n > total:
        raise SetsError(f"cannot draw {n} distinct {k}-subsets from {n_universe} values (only {total})")
    return candidate_pool(n_universe, k, n, rng)


def quantile_bins(scores: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign each score to one of ``bins`` quantile bins; returns ``(bin_of, edges)``."""
    if bins < 1:
        raise SetsError("bins must be >= 1")
    edges = np.quantile(scores, np.linspace(0, 1, bins + 1))
    inner = edges[1:-1]
    bin_of = np.searchsorted(inner, scores, side="right")
    return bin_of.astype(int), edges


def width_bins(scores: np.ndarray, bins: int, trim: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Assign each score to one of ``bins`` equal-width bins spanning the
    ``trim``..``100 - trim`` percentile range of ``scores``; scores outside
    that range get bin ``-1`` and are never drawn. Returns ``(bin_of, edges)``."""
    if bins < 1:
        raise SetsError("bins must be >= 1")
    if not 0.0 <= trim < 50.0:
        raise SetsError(f"trim must be a percentile in [0, 50); got {trim}")
    lo, hi = np.percentile(scores, [trim, 100.0 - trim])
    edges = np.linspace(lo, hi, bins + 1)
    bin_of = np.searchsorted(edges[1:-1], scores, side="right")
    bin_of[(scores < lo) | (scores > hi)] = -1
    return bin_of.astype(int), edges


def assign_bins(scores: np.ndarray, bins: int, binning: str = "quantile", trim: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if binning == "quantile":
        if trim:
            raise SetsError("trim only applies to --binning width")
        return quantile_bins(scores, bins)
    if binning == "width":
        return width_bins(scores, bins, trim)
    raise SetsError(f"unknown binning {binning!r}; known: {BINNINGS}")


def allocate(n: int, bins: int, occupancy: Sequence[int]) -> list[int]:
    """Equal counts per bin (remainder to the outermost bins first), with any
    bin short of members handing its deficit to the others."""
    base, rem = divmod(n, bins)
    order = sorted(range(bins), key=lambda b: (-abs(b - (bins - 1) / 2), b))
    want = [base] * bins
    for b in order[:rem]:
        want[b] += 1
    alloc = [min(w, occ) for w, occ in zip(want, occupancy)]
    deficit = n - sum(alloc)
    for b in order:
        if deficit <= 0:
            break
        room = occupancy[b] - alloc[b]
        take = min(room, deficit)
        alloc[b] += take
        deficit -= take
    if deficit > 0:
        raise SetsError(f"candidate pool has only {sum(occupancy)} sets, cannot pick {n}")
    return alloc


def sample_stratified(scores: np.ndarray, n: int, bins: int, rng: np.random.Generator,
                      binning: str = "quantile", trim: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick ``n`` pool indices with equal counts per bin of ``scores`` (see
    :func:`assign_bins`). Returns ``(picked_idx, bin_of_pick, edges)``; picks
    are sorted by score."""
    if n > len(scores):
        raise SetsError(f"cannot pick {n} sets from a pool of {len(scores)}")
    bin_of, edges = assign_bins(scores, bins, binning, trim)
    occupancy = [int((bin_of == b).sum()) for b in range(bins)]
    alloc = allocate(n, bins, occupancy)
    picked: list[int] = []
    for b in range(bins):
        members = np.flatnonzero(bin_of == b)
        if alloc[b]:
            picked.extend(rng.choice(members, size=alloc[b], replace=False).tolist())
    picked_arr = np.asarray(picked, dtype=int)
    order = np.argsort(scores[picked_arr], kind="stable")
    picked_arr = picked_arr[order]
    return picked_arr, bin_of[picked_arr], edges


# ── building the record ──────────────────────────────────────────────────────


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0, 0.25, 0.5, 0.75, 1])
    return {"n": int(x.size), "min": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "max": float(q[4]), "iqr": float(q[3] - q[1])}


def _grids(store: Store) -> tuple[np.ndarray, np.ndarray]:
    return cosine_grid(store.universe, store.external), cosine_grid(store.universe, store.universe)


def build_family(fam: FamilyConfig, universe: Sequence[str], stores: Mapping[str, Store], scheme: str,
                 metric: str, store_name: str | None, n_candidates: int, bins: int, seed: int,
                 binning: str = "quantile", trim: float = 0.0) -> tuple[dict, dict]:
    """One family's arms plus its readouts. Explicit families ignore ``scheme``."""
    universe = list(universe)
    n_u = len(universe)
    grids = {name: _grids(s) for name, s in stores.items()}

    def scores_for(pool: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        return {name: {m: score_pool(pool, m, S_ue=g[0], G_uu=g[1]) for m in METRICS}
                for name, g in grids.items()}

    if fam.explicit:
        scheme_used = "explicit"
        pos = {v: i for i, v in enumerate(universe)}
        missing = sorted({v for s in fam.values for v in s} - set(pos))
        if missing:
            raise SetsError(f"family {fam.id}: values not in the universe: {missing[:5]}")
        picks = [[pos[v] for v in s] for s in fam.values]
        ks = {len(s) for s in picks}
        k = ks.pop() if len(ks) == 1 else None
        bin_of = None
        edges = None
        if k is not None:
            pool = np.asarray(picks, dtype=int)
            pool_scores = scores_for(pool)
        else:  # ragged explicit sets: score each on its own
            pool = None
            pool_scores = {name: {m: np.asarray([score_pool(np.asarray([p], dtype=int), m, S_ue=g[0], G_uu=g[1])[0]
                                                for p in picks]) for m in METRICS}
                           for name, g in grids.items()}
        pick_scores = pool_scores
    else:
        k = int(fam.k)
        n = int(fam.n)
        rng = rng_for(seed, fam.id, scheme)
        if scheme == "random":
            scheme_used = "random"
            pool = sample_random(n_u, k, n, rng)
            pool_scores = scores_for(pool)
            picks, pick_scores, bin_of, edges = pool, pool_scores, None, None
        elif scheme == "stratified":
            scheme_used = "stratified"
            if store_name is None:
                raise SetsError("stratified sampling needs --store")
            if metric not in METRICS:
                raise SetsError(f"unknown metric {metric!r}; known: {METRICS}")
            if store_name not in stores:
                raise SetsError(f"store {store_name!r} not loaded; known: {sorted(stores)}")
            pool = candidate_pool(n_u, k, n_candidates, rng)
            pool_scores = scores_for(pool)
            picked_idx, bin_of, edges = sample_stratified(pool_scores[store_name][metric], n, bins, rng, binning, trim)
            picks = pool[picked_idx]
            pick_scores = {s: {m: pool_scores[s][m][picked_idx] for m in METRICS} for s in pool_scores}
        else:
            raise SetsError(f"unknown scheme {scheme!r}; known: {SCHEMES}")

    arms: dict[str, dict] = {}
    for i, row in enumerate(picks):
        rec = {"values": [universe[int(j)] for j in row]}
        if store_name in pick_scores:
            rec["score"] = float(pick_scores[store_name][metric][i])
        if bin_of is not None:
            rec["bin"] = int(bin_of[i])
        arms[arm_id(fam.id, i)] = rec

    # ── readouts ──
    readouts: dict = {"scheme": scheme_used, "k": k, "n": len(arms), "pool_size": int(pool.shape[0]) if pool is not None else len(arms)}
    readouts["pool"] = {s: {m: _stats(pool_scores[s][m]) for m in METRICS} for s in pool_scores}
    readouts["picked"] = {s: {m: _stats(pick_scores[s][m]) for m in METRICS} for s in pick_scores}
    if bin_of is not None:
        all_bins, _ = assign_bins(pool_scores[store_name][metric], bins, binning, trim)
        readouts["bins"] = {"binning": binning, "trim": trim, "edges": [float(e) for e in edges],
                            "pool_occupancy": [int((all_bins == b).sum()) for b in range(bins)],
                            "picked_occupancy": [int((bin_of == b).sum()) for b in range(bins)]}
    counts = np.zeros(n_u, dtype=int)
    for row in picks:
        counts[np.asarray(row, dtype=int)] += 1
    readouts["design_coverage"] = {
        "min": int(counts.min()), "max": int(counts.max()), "mean": float(counts.mean()),
        "n_zero": int((counts == 0).sum()),
        "zero_values": [universe[i] for i in np.flatnonzero(counts == 0)][:20],
    }
    if pool is not None and len(stores) >= 2 and pool.shape[0] >= 3:
        from scipy import stats

        names = sorted(stores)
        agree = {}
        for m in METRICS:
            for a, b in itertools.combinations(names, 2):
                x, y = pool_scores[a][m], pool_scores[b][m]
                ok = np.isfinite(x) & np.isfinite(y)
                rho = float(stats.spearmanr(x[ok], y[ok])[0]) if ok.sum() >= 3 else float("nan")
                agree[f"{m}:{a}~{b}"] = rho
        readouts["store_agreement_spearman"] = agree
    return arms, readouts


def external_reach(stores: Mapping[str, Store]) -> dict:
    """Per store: the max cosine from each external item to any universe
    value (a ceiling on coverage), summarized plus the least reachable items."""
    out = {}
    for name, s in stores.items():
        S = cosine_grid(s.universe, s.external)
        reach = S.max(axis=0)
        order = np.argsort(reach)
        out[name] = {**_stats(reach), "coverage_ceiling": float(reach.mean()),
                     "least_reachable": [(s.external_names[i], float(reach[i])) for i in order[:5]]}
    return out


def build_sets(cfg: MultivalueConfig, universe: Sequence[str], stores: Mapping[str, Store], scheme: str,
               metric: str = "coverage", store: str | None = None, n_candidates: int = DEFAULT_CANDIDATES,
               bins: int = DEFAULT_BINS, seed: int | None = None, binning: str = "quantile",
               trim: float = 0.0) -> tuple[dict, dict]:
    """The full ``sets.json`` record plus readouts keyed by family."""
    if scheme not in ("random", "stratified"):
        raise SetsError(f"scheme must be random or stratified (explicit families are always explicit); got {scheme!r}")
    if store is None and stores:
        store = "sentence" if "sentence" in stores else sorted(stores)[0]
    if store is not None and store not in stores:
        raise SetsError(f"store {store!r} is not loaded; known: {sorted(stores)}")
    seed = cfg.seed if seed is None else int(seed)
    families: dict[str, dict] = {}
    readouts: dict[str, dict] = {}
    for fam in cfg.arms.families:
        arms, ro = build_family(fam, universe, stores, scheme, metric, store, n_candidates, bins, seed, binning, trim)
        families[fam.id] = {"k": fam.k if not fam.explicit else None, "n": len(arms),
                            "scheme": ro["scheme"], "arms": arms}
        readouts[fam.id] = ro
    readouts["_external_reach"] = external_reach(stores)
    arm_values = {a: r["values"] for f in families.values() for a, r in f["arms"].items()}
    if len(arm_values) != sum(len(f["arms"]) for f in families.values()):
        raise SetsError("arm ids collide across families")
    record = {
        "schema_version": SETS_SCHEMA_VERSION,
        "name": cfg.name,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "scheme": scheme, "metric": metric, "store": store,
        "n_candidates": n_candidates, "bins": bins,
        # recorded only off the default so quantile designs keep their sets.json bytes (= exp_id)
        **({"binning": binning, "trim": trim} if binning != "quantile" else {}),
        "universe": list(universe),
        "universe_sha256": sha256_json(list(universe)),
        "reference": list(cfg.arms.reference),
        "families": families,
        "arms_sha256": sha256_json(arm_values),
    }
    return record, readouts


# ── persistence ──────────────────────────────────────────────────────────────


def write_sets(path: Path, record: dict, force: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("arms_sha256") == record.get("arms_sha256") and prev.get("universe_sha256") == record.get("universe_sha256"):
            return path  # identical design already frozen; keep the original bytes
        if not force:
            raise SetsError(
                f"{path} is already frozen with a different design "
                f"(arms_sha256 {prev.get('arms_sha256')} vs {record.get('arms_sha256')}). "
                "Pass --force to replace it; every trained arm keyed on the old identity stays on disk."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_sets(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SetsError(f"no frozen sets at {path}; run `valuegen mv sets ... --freeze`")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("schema_version") != SETS_SCHEMA_VERSION:
        raise SetsError(f"{path}: schema_version {rec.get('schema_version')} != {SETS_SCHEMA_VERSION}")
    return rec


def arms_from_record(record: dict) -> list[Arm]:
    """Reference arms first (base, then full if configured), then family arms in order."""
    universe = tuple(record["universe"])
    arms: list[Arm] = []
    for ref in record.get("reference", [BASE_ARM]):
        if ref == BASE_ARM:
            arms.append(Arm(id=BASE_ARM, family="reference", values=(), kind="base"))
        elif ref == FULL_ARM:
            arms.append(Arm(id=FULL_ARM, family="reference", values=universe, kind="full"))
        else:
            raise SetsError(f"unknown reference arm {ref!r}")
    for fid, fam in record["families"].items():
        kind = "explicit" if fam.get("scheme") == "explicit" else "sampled"
        for aid, rec in fam["arms"].items():
            arms.append(Arm(id=aid, family=fid, values=tuple(rec["values"]), kind=kind,
                            score=rec.get("score"), bin=rec.get("bin")))
    ids = [a.id for a in arms]
    if len(set(ids)) != len(ids):
        raise SetsError("duplicate arm ids in sets record")
    return arms


def trained_arms(arms: Sequence[Arm]) -> list[Arm]:
    return [a for a in arms if a.trained]


# ── printing ─────────────────────────────────────────────────────────────────


def _fmt_stats(s: dict) -> str:
    if not s or s.get("n", 0) == 0:
        return "n=0"
    return (f"n={s['n']} min={s['min']:.3f} q25={s['q25']:.3f} med={s['median']:.3f} "
            f"q75={s['q75']:.3f} max={s['max']:.3f} iqr={s['iqr']:.3f}")


def format_readouts(record: dict, readouts: dict) -> str:
    lines = [f"sets for {record['name']}: scheme={record['scheme']} metric={record['metric']} "
             f"store={record['store']} seed={record['seed']} universe={len(record['universe'])} values",
             f"reference arms: {', '.join(record['reference'])}"]
    reach = readouts.get("_external_reach") or {}
    for store, r in reach.items():
        least = ", ".join(f"{n}={v:.2f}" for n, v in r["least_reachable"])
        lines.append(f"external reach [{store}]: coverage ceiling {r['coverage_ceiling']:.3f}; "
                     f"per-item max-cos {_fmt_stats(r)}; least reachable: {least}")
    for fid, fam in record["families"].items():
        ro = readouts[fid]
        lines.append("")
        lines.append(f"family {fid}: {ro['scheme']} k={ro['k']} n={ro['n']} pool={ro['pool_size']}")
        for store in sorted(ro["pool"]):
            for m in METRICS:
                lines.append(f"  {store:>9}/{m:<9} pool   {_fmt_stats(ro['pool'][store][m])}")
                lines.append(f"  {'':>9} {'':<9} picked {_fmt_stats(ro['picked'][store][m])}")
        if "bins" in ro:
            b = ro["bins"]
            cut = f"{b['binning']}, trim {b['trim']:g}%" if b.get("binning", "quantile") != "quantile" else "quantile"
            lines.append(f"  bins ({cut}): edges {' '.join(f'{e:.3f}' for e in b['edges'])}")
            lines.append(f"        pool occupancy {b['pool_occupancy']}  picked {b['picked_occupancy']}")
        dc = ro["design_coverage"]
        zero = f" zero: {dc['zero_values']}" if dc["n_zero"] else ""
        lines.append(f"  design coverage per universe value: min={dc['min']} max={dc['max']} "
                     f"mean={dc['mean']:.2f} n_zero={dc['n_zero']}{zero}")
        if "store_agreement_spearman" in ro:
            ag = "  ".join(f"{k} rho={v:.3f}" for k, v in ro["store_agreement_spearman"].items())
            lines.append(f"  store agreement over pool: {ag}")
        for aid, rec in list(fam["arms"].items())[:3]:
            sc = f" score={rec['score']:.3f}" if "score" in rec else ""
            bn = f" bin={rec['bin']}" if "bin" in rec else ""
            lines.append(f"  {aid}{sc}{bn}: {', '.join(rec['values'])}")
        if len(fam["arms"]) > 3:
            lines.append(f"  ... {len(fam['arms']) - 3} more")
    return "\n".join(lines)
