"""Multi-turn expansion pools for oracle labeling (the `ca_en_mt_p*` and
`wf_mt` label sources). Companion to scripts/build_ca_wf_pools.py, which
already covers Community Alignment English *first turns* (all pairs) and
WildFeedback *single-turn* — those stay in that run's store and are NOT rebuilt
here. This script adds everything else English:

- ca_en_mt_p1 : CA English turns 2-4, ONE representative pair per item
                (human-preferred response vs. a seed-42 deterministic other) —
                the "2 responses per item" fast pass. Prompt is the conversation
                transcript up to that turn (prior turns advanced by their
                preferred response), tail-preserving so the final user turn
                always survives the max_chars clip.
- ca_en_mt_p2 : CA English turns 2-4, the REMAINING pairs (all C(n,2) pairs of
                an item minus its representative). Disjoint from p1, so
                p1 ∪ p2 = every turn-2-4 pair with no double labeling.
- wf_mt       : WildFeedback multi-turn (>=2 human turns), chosen vs rejected,
                one pair per item (A=chosen, B=rejected), transcript prompt.

The const_v3_dpo_full49* configs train on {ca_en_mt_p1, wf_mt}; ca_en_mt_p2
is built for completeness and labeled only on request.

Deterministic: pinned snapshot revisions, stable sorts, seeded pair pick.
Run on a login node (HF hub access); compute nodes read the parquets.

Usage: .venvs/core/bin/python scripts/build_ca_wf_mt_pools.py
"""
import hashlib
import itertools
import json
import random
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

OUT = Path("data/pools")
CA_REV = "97343c7f6399fcbea430ed0f37c1768281a78d56"
WF_REV = "8b1a3e530b949d6aacfad6ba8912e209a05bc846"
SEED = 42
CTX_BUDGET = 6000  # transcript char budget, under oracle max_chars=8000

TURNS = ["first", "second", "third", "fourth"]
LETTERS = "abcd"


def _pref_letter(val) -> str | None:
    """CA preferred_response cells look like 'response_a'; take the letter."""
    if isinstance(val, str) and val.strip():
        c = val.strip()[-1].lower()
        if c in LETTERS:
            return c
    return None


def _serialize(history: list[tuple[str, str]], final_user: str) -> str:
    """User/Assistant transcript for `history` then a trailing User turn, kept
    under CTX_BUDGET by dropping OLDEST turns first (the final user turn — the
    actual question — is always preserved)."""
    tail = f"User: {final_user}"
    blocks = [f"User: {u}\nAssistant: {a}" for u, a in history]
    kept, used, dropped = [], len(tail), 0
    for idx in range(len(blocks) - 1, -1, -1):
        b = blocks[idx]
        if used + len(b) + 2 <= CTX_BUDGET:
            kept.append(b)
            used += len(b) + 2
        else:
            dropped = idx + 1
            break
    kept.reverse()
    parts = kept + [tail]
    if dropped:
        parts = ["[…earlier turns omitted…]"] + parts
    return "\n".join(parts)


def build_ca():
    csv = hf_hub_download("facebook/community-alignment-dataset",
                          "community_alignment.csv", repo_type="dataset", revision=CA_REV)
    usecols = ["conversation_id", "assigned_lang", "annotator_country"]
    for t in TURNS:
        usecols += [f"{t}_turn_prompt", f"{t}_turn_preferred_response"]
        usecols += [f"{t}_turn_response_{c}" for c in LETTERS]
    d = pd.read_csv(csv, usecols=usecols)
    d = d[d.assigned_lang == "en"].sort_values("conversation_id", kind="stable")

    p1_rows, p2_rows = [], []
    seen1, seen2 = set(), set()
    fallbacks = 0
    for r in d.itertuples(index=False):
        history: list[tuple[str, str]] = []
        for ti, t in enumerate(TURNS):
            prompt = getattr(r, f"{t}_turn_prompt")
            resp = {c: getattr(r, f"{t}_turn_response_{c}") for c in LETTERS}
            resp = {c: v.strip() for c, v in resp.items()
                    if isinstance(v, str) and v.strip()}
            pref = _pref_letter(getattr(r, f"{t}_turn_preferred_response"))
            has_prompt = isinstance(prompt, str) and bool(prompt.strip())

            # Emit pairs only for turns 2-4 (index >= 1); turn 1 is covered by
            # the first-turn run. Need a valid reconstructed history to here.
            if has_prompt and ti >= 1 and len(resp) >= 2:
                ctx = _serialize(history, prompt.strip())
                pref_text = resp.get(pref) if pref else None
                # deterministic representative pair: preferred vs a seeded other
                rng = random.Random(
                    int.from_bytes(
                        hashlib.sha256(f"{r.conversation_id}:{t}".encode()).digest()[:8],
                        "big") ^ SEED)
                letters_sorted = sorted(resp)
                rep_pref = pref
                if pref_text is not None:
                    others = [c for c in letters_sorted if c != pref]
                    ol = rng.choice(others)
                    rep = frozenset((pref_text, resp[ol]))
                else:
                    a, b = rng.sample(letters_sorted, 2)
                    rep = frozenset((resp[a], resp[b]))
                    rep_pref = None
                for x, y in itertools.combinations(letters_sorted, 2):
                    if resp[x] == resp[y]:
                        continue
                    A, B = resp[x], resp[y]
                    key = (ctx, *sorted((A, B)))
                    is_rep = frozenset((A, B)) == rep
                    row = {"prompt": ctx, "A": A, "B": B,
                           "conversation_id": r.conversation_id, "turn": t, "pair": x + y,
                           "preferred": ("A" if rep_pref == x else "B" if rep_pref == y else ""),
                           "annotator_country": r.annotator_country}
                    if is_rep:
                        if key not in seen1:
                            seen1.add(key); p1_rows.append(row)
                    else:
                        if key not in seen2:
                            seen2.add(key); p2_rows.append(row)

            # advance history by this turn's preferred response for the next turn
            if has_prompt and resp:
                if pref and pref in resp:
                    chosen = resp[pref]
                else:
                    chosen = resp[sorted(resp)[0]]
                    fallbacks += 1
                history.append((prompt.strip(), chosen))

    # belt-and-suspenders: p2 excludes any pair already claimed by p1
    p1 = pd.DataFrame(p1_rows)
    p2 = pd.DataFrame([r for r in p2_rows
                       if (r["prompt"], *sorted((r["A"], r["B"]))) not in seen1])
    print(f"  [ca] history-advance fallbacks (no preferred): {fallbacks:,}")
    return p1, p2


def build_wf_mt():
    p = hf_hub_download("microsoft/WildFeedback", "wildfeedback.json",
                        repo_type="dataset", revision=WF_REV)
    role = {"human": "User", "gpt": "Assistant"}
    rows, seen = [], set()
    for i, r in enumerate(json.load(open(p))):
        convs = r["conversations"]
        human = [m for m in convs if m["from"] == "human"]
        if len(human) < 2:  # single-turn handled by the first-turn run
            continue
        a = r["chosen"]["value"].strip()
        b = r["rejected"]["value"].strip()
        if not a or not b or a == b:
            continue
        # transcript ends at the last human turn; tail-preserve under budget
        msgs = [(role.get(m["from"], m["from"]), m["value"].strip()) for m in convs]
        final_user = msgs[-1][1] if msgs and msgs[-1][0] == "User" else ""
        hist, pending_u = [], None
        for who, text in (msgs[:-1] if final_user else msgs):
            if who == "User":
                pending_u = text
            elif who == "Assistant" and pending_u is not None:
                hist.append((pending_u, text)); pending_u = None
        prompt = _serialize(hist, final_user or (msgs[-1][1] if msgs else ""))
        key = (prompt, *sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"prompt": prompt, "A": a, "B": b, "source_index": i})
    return pd.DataFrame(rows)


def _report(name, df):
    path = OUT / f"{name}.parquet"
    df.to_parquet(path, index=False)
    digest = hashlib.sha256(
        df[["prompt", "A", "B"]].to_csv(index=False).encode()).hexdigest()[:12]
    print(f"{name}: {len(df):,} rows, {df.prompt.nunique():,} prompts, sha {digest}")
    return {"rows": len(df), "unique_prompts": int(df.prompt.nunique()),
            "triples_sha": digest}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"community_alignment_revision": CA_REV, "wildfeedback_revision": WF_REV,
              "seed": SEED, "ctx_budget": CTX_BUDGET}
    ca_p1, ca_p2 = build_ca()
    report["ca_en_mt_p1"] = _report("ca_en_mt_p1", ca_p1)
    report["ca_en_mt_p2"] = _report("ca_en_mt_p2", ca_p2)
    report["wf_mt"] = _report("wf_mt", build_wf_mt())
    (OUT / "build_report_mt.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print("\nwrote data/pools/build_report_mt.yaml")


if __name__ == "__main__":
    main()
