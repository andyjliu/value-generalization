"""Build (prompt, A, B) preference pools from Community Alignment (English)
and WildFeedback for oracle labeling (the `community_alignment_en` and
`wildfeedback_st` label sources of the const_v3_dpo_full49* configs).

Keeps every candidate pair and carries provenance columns; the subset is cut
later by the DPO configs' exclusive assignment.

- community_alignment_en: English first turns only (first turns carry no
  conversation context, so every conversation contributes regardless of its
  depth; turns 2-4 are excluded by decision — no multi-turn). Pregenerated
  prompts repeat across annotators with the same 4 responses, so items are
  deduped on (prompt, 4-response set) before pairing; then ALL unordered
  response pairs per item (up to C(4,2)=6 — pair orientation is irrelevant,
  the oracle judges both orderings), deduped again at (prompt, {A, B}).
- wildfeedback_st: rows whose conversation is a single human turn (no context),
  chosen vs rejected, deduped at (prompt, {A, B}).

Deterministic: pinned snapshot revisions, stable sorts. Run on a login node
(needs HF hub access on first run); compute nodes read the parquets.

Usage: .venvs/core/bin/python scripts/build_ca_wf_pools.py
"""
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

OUT = Path("data/pools")
CA_REV = "97343c7f6399fcbea430ed0f37c1768281a78d56"
WF_REV = "8b1a3e530b949d6aacfad6ba8912e209a05bc846"


def build_ca() -> pd.DataFrame:
    csv = hf_hub_download("facebook/community-alignment-dataset",
                          "community_alignment.csv", repo_type="dataset", revision=CA_REV)
    letters = "abcd"
    cols = (["conversation_id", "assigned_lang", "annotator_country",
             "is_pregenerated_first_prompt", "first_turn_prompt",
             "first_turn_preferred_response"]
            + [f"first_turn_response_{c}" for c in letters])
    d = pd.read_csv(csv, usecols=cols)
    d = d[(d.assigned_lang == "en") & d.first_turn_prompt.notna()]
    resp_cols = [f"first_turn_response_{c}" for c in letters]
    d = d.drop_duplicates(subset=["first_turn_prompt"] + resp_cols, keep="first")
    d = d.sort_values("conversation_id", kind="stable")

    rows, seen = [], set()
    for r in d.itertuples(index=False):
        prompt = r.first_turn_prompt.strip()
        resp = {c: getattr(r, f"first_turn_response_{c}") for c in letters}
        resp = {c: v.strip() for c, v in resp.items() if isinstance(v, str) and v.strip()}
        for x, y in itertools.combinations(sorted(resp), 2):
            if resp[x] == resp[y]:
                continue
            key = (prompt, *sorted((resp[x], resp[y])))
            if key in seen:
                continue
            seen.add(key)
            pref = r.first_turn_preferred_response
            rows.append({
                "prompt": prompt, "A": resp[x], "B": resp[y],
                "conversation_id": r.conversation_id, "pair": x + y,
                "preferred": pref.strip()[-1] if isinstance(pref, str) else "",
                "annotator_country": r.annotator_country,
                "pregenerated_prompt": bool(r.is_pregenerated_first_prompt),
            })
    return pd.DataFrame(rows)


def build_wf() -> pd.DataFrame:
    p = hf_hub_download("microsoft/WildFeedback", "wildfeedback.json",
                        repo_type="dataset", revision=WF_REV)
    rows, seen = [], set()
    for i, r in enumerate(json.load(open(p))):
        human = [m["value"] for m in r["conversations"] if m["from"] == "human"]
        if len(human) != 1 or len(r["conversations"]) != 1:
            continue
        prompt = human[0].strip()
        a, b = r["chosen"]["value"].strip(), r["rejected"]["value"].strip()
        if not prompt or not a or not b or a == b:
            continue
        key = (prompt, *sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"prompt": prompt, "A": a, "B": b, "source_index": i})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"community_alignment_revision": CA_REV, "wildfeedback_revision": WF_REV}
    for name, df in [("community_alignment_en", build_ca()), ("wildfeedback_st", build_wf())]:
        path = OUT / f"{name}.parquet"
        df.to_parquet(path, index=False)
        digest = hashlib.sha256(
            df[["prompt", "A", "B"]].to_csv(index=False).encode()
        ).hexdigest()[:12]
        report[name] = {"rows": len(df), "unique_prompts": int(df.prompt.nunique()),
                        "triples_sha": digest}
        print(f"{name}: {len(df):,} rows, {df.prompt.nunique():,} prompts, sha {digest}")
    (OUT / "build_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
