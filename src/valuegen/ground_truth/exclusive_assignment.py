"""Exclusive (prompt, {A, B}) -> value assignment for ``label_subset``.

The shared-pool build in :func:`label_subset.build_data` gives every value
every row it clears ``|score| >= tau`` on, so related values' DPO sets can
overlap almost entirely. These two rules give each eligible row to at most
ONE value (``intervention.assignment.rule``).

The default unit of exclusivity is a (prompt, {A, B}) triple, not a prompt:
the same prompt text may train several values as long as the exact response
pair isn't repeated. Rows whose triple duplicates an earlier one are dropped
first. ``intervention.assignment.unit: prompt`` tightens this to one row per
normalized prompt text across ALL values (mincost_flow only): the flow gets a
prompt layer, source -> prompt (cap 1) -> row, so a prompt with several
labeled pairs (CA first turns carry C(4,2) = 6, HH/CA multi-turn repeat
prefixes) feeds at most one value once. Rows in the same conversation at
different turn depths are different prompts and stay independent.

  argmax_bounded  eligible row -> argmax-|score| value (tie -> lowest id);
                  values over ``cap`` keep a uniform random ``cap``; values
                  under ``floor`` are topped up from the cap-dropped rows,
                  highest |score| first.
  mincost_flow    min-cost flow (OR-Tools): every value gets exactly ``cap``
                  rows, no row twice, total |score| maximal (cost
                  -round(1000*|score|) per eligible edge). Node ids are
                  permuted with ``seed`` so ties don't follow file order.

Both return one row per selected (source, row_id): columns
``source, row_id, principle_id, score``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

RULES = ("argmax_bounded", "mincost_flow")
UNITS = ("triple", "prompt")


def _norm(text: pd.DataFrame, col: str) -> pd.Series:
    return text[col].astype(str).str.strip().str.lower()


def triple_key(text: pd.DataFrame) -> pd.Series:
    """Normalized prompt + unordered {A, B}, one per pool row."""
    ab = pd.concat([_norm(text, "A"), _norm(text, "B")], axis=1)
    return _norm(text, "prompt") + "\x00" + ab.min(axis=1) + "\x00" + ab.max(axis=1)


def prompt_key(text: pd.DataFrame) -> pd.Series:
    """Normalized prompt text, one per pool row (the ``unit: prompt`` key)."""
    return _norm(text, "prompt")


def assign(
    scores: pd.DataFrame,
    keys: pd.Series,
    pids: list[int],
    rule: str,
    tau: float,
    cap: int,
    floor: int,
    seed: int,
    unit: str = "triple",
    prompt_keys: pd.Series | None = None,
) -> pd.DataFrame:
    """``scores``: long table (source, row_id, principle_id, score) over all
    sources; ``keys``: triple key indexed by (source, row_id); ``pids``: the
    principle ids in play (others in ``scores`` are ignored); ``prompt_keys``:
    prompt key indexed like ``keys``, required for ``unit="prompt"``."""
    if rule not in RULES:
        raise ValueError(f"assignment.rule must be one of {RULES}, got {rule!r}")
    if unit not in UNITS:
        raise ValueError(f"assignment.unit must be one of {UNITS}, got {unit!r}")
    if unit == "prompt":
        if rule != "mincost_flow":
            raise ValueError("assignment.unit: prompt needs rule: mincost_flow")
        if prompt_keys is None:
            raise ValueError("unit='prompt' needs prompt_keys")
    # Sorted pid order: the RNG draws (cap sampling, node permutation) and
    # argmax tie-breaks depend on column order, so the result must not move
    # with the config's `values:` ordering.
    pids = sorted(int(p) for p in pids)
    scores = scores[scores.principle_id.isin(pids)]
    dup = keys[keys.duplicated(keep="first")].index
    if len(dup):
        print(f"assignment: dropping {len(dup)} rows whose (prompt,{{A,B}}) "
              f"triple repeats an earlier row: {list(dup)}")
        scores = scores[~pd.MultiIndex.from_frame(scores[["source", "row_id"]]).isin(dup)]
    S = scores.pivot_table(index=["source", "row_id"], columns="principle_id",
                           values="score", aggfunc="mean").reindex(columns=pids)
    A = S.abs()
    elig = (A >= tau).fillna(False)
    keep = elig.any(axis=1)
    S, A, elig = S[keep], A[keep], elig[keep]
    print(f"assignment[{rule}/{unit}]: {len(S):,} eligible rows over {len(pids)} values")

    if rule == "argmax_bounded":
        sel = _argmax_bounded(A.values, elig.values, cap, floor, seed)
    else:
        pcode = None
        if unit == "prompt":
            pk = prompt_keys.loc[S.index]
            pcode = pd.factorize(pk.values)[0]
            print(f"  {pcode.max() + 1:,} distinct prompts among the eligible rows")
        sel = _mincost_flow(A.values, elig.values, cap, seed, pcode=pcode)
    p, j = sel.index.values, sel.values
    idx = S.index[p]
    out = pd.DataFrame({
        "source": idx.get_level_values(0),
        "row_id": idx.get_level_values(1).astype(int),
        "principle_id": [pids[k] for k in j],
        "score": S.values[p, j],
    })
    k = keys.loc[list(zip(out.source, out.row_id))]
    assert not out.duplicated(["source", "row_id"]).any(), "row assigned twice"
    assert not k.duplicated().any(), "(prompt,{A,B}) triple assigned twice"
    if unit == "prompt":
        pk_out = prompt_keys.loc[list(zip(out.source, out.row_id))]
        assert not pk_out.duplicated().any(), "prompt assigned twice"
    assert (out.score.abs() >= tau).all()
    counts = out.groupby("principle_id").size()
    if rule == "mincost_flow":
        assert (counts == cap).all(), f"flow counts != cap: {counts.to_dict()}"
    print("assignment counts: " + ", ".join(f"ad_{p}={n:,}" for p, n in counts.items()))
    return out


def _argmax_bounded(A, elig, cap, floor, seed):
    rng = np.random.default_rng(seed)
    Vn = A.shape[1]
    am = np.nanargmax(np.where(elig, A, np.nan), axis=1)
    chosen: dict[int, int] = {}
    dropped: list[int] = []
    under = []
    for v in range(Vn):
        members = np.flatnonzero(am == v)
        if len(members) > cap:
            keep_idx = rng.choice(members, cap, replace=False)
            dropped.extend(np.setdiff1d(members, keep_idx))
            members = keep_idx
        if len(members) < floor:
            under.append(v)
        for m in members:
            chosen[int(m)] = v
    dropped_arr = np.array(dropped, int)
    for v in under:
        need = floor - sum(1 for x in chosen.values() if x == v)
        cand = dropped_arr[elig[dropped_arr, v]]
        cand = cand[~np.isin(cand, list(chosen))]
        cand = cand[np.argsort(-A[cand, v])][:need]
        for m in cand:
            chosen[int(m)] = v
        dropped_arr = np.setdiff1d(dropped_arr, cand)
        print(f"  value {v}: under floor {floor}, topped up by {len(cand)} "
              f"(short {need - len(cand)})")
    return pd.Series(chosen).sort_index()


def _mincost_flow(A, elig, cap, seed, pcode=None):
    """``pcode``: per-row prompt code (unit='prompt'); adds a prompt layer with
    capacity 1 between the source and the rows, so at most one row per prompt
    carries flow."""
    from ortools.graph.python import min_cost_flow

    P, Vn = A.shape
    rng = np.random.default_rng(seed)
    pp, pv = rng.permutation(P), rng.permutation(Vn)
    E = np.argwhere(elig)
    cost = -np.round(1000 * A[E[:, 0], E[:, 1]]).astype(int)
    nP = 0 if pcode is None else int(pcode.max()) + 1
    src, row0, val0 = 0, 1 + nP, 1 + nP + P
    sink = val0 + Vn
    mcf = min_cost_flow.SimpleMinCostFlow()
    if pcode is None:
        mcf.add_arcs_with_capacity_and_unit_cost(
            np.zeros(P, int), row0 + np.arange(P), np.ones(P, int), np.zeros(P, int))
    else:
        pq = rng.permutation(nP)
        mcf.add_arcs_with_capacity_and_unit_cost(
            np.zeros(nP, int), 1 + np.arange(nP), np.ones(nP, int), np.zeros(nP, int))
        mcf.add_arcs_with_capacity_and_unit_cost(
            1 + pq[pcode], row0 + pp, np.ones(P, int), np.zeros(P, int))
    first = mcf.num_arcs()
    mcf.add_arcs_with_capacity_and_unit_cost(
        row0 + pp[E[:, 0]], val0 + pv[E[:, 1]], np.ones(len(E), int), cost)
    mcf.add_arcs_with_capacity_and_unit_cost(
        val0 + np.arange(Vn), np.full(Vn, sink), np.full(Vn, cap), np.zeros(Vn, int))
    mcf.set_node_supply(src, Vn * cap)
    mcf.set_node_supply(sink, -Vn * cap)
    t = time.time()
    st = mcf.solve()
    if st != mcf.OPTIMAL:
        raise RuntimeError(f"min-cost flow status {st}: cap {cap} infeasible")
    print(f"  flow solved in {time.time() - t:.1f}s, total |score| = "
          f"{-mcf.optimal_cost() / 1000:,.1f}")
    inv_p, inv_v = np.argsort(pp), np.argsort(pv)
    out = {}
    for a in range(first, first + len(E)):
        if mcf.flow(a) > 0:
            out[int(inv_p[mcf.tail(a) - row0])] = int(inv_v[mcf.head(a) - val0])
    return pd.Series(out).sort_index()
