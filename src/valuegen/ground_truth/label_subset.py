"""``label_subset`` GT method: judge-labeled preference pools → DPO datasets.

HH harmless-base + PKU-SafeRLHF test splits joined to per-principle judgment
CSVs
(``score = prob_A − prob_B``, Buyl et al. alignment-discretion labels), keep
``|score| ≥ TAU`` (0.5), seed-42 shuffle, chosen/rejected by sign. Datasets
are **model-agnostic** — chosen/rejected are the raw preference labels — so
each value's file is built once and reused across every base model.

The judgment CSVs come from one of two places, and the config says which:

- ``labels.dir`` — **prelabeled**: point at judgments that already exist. This
  is how externally produced labels (e.g. the original Buyl et al. GPT-4o
  judgments) are used, and it is the only mode for a value set whose labels this repo
  cannot reproduce.
- ``labels.oracle`` — **generate**: label the pools here, with the oracle in
  this module (a served judge + a sharded CPU array). Missing labels are built,
  existing ones reused.

The oracle implements the Buyl et al. method (Fig. 4 template, first-token
``A``/``B``/``NA`` logprobs, both orderings averaged), run against any judge
instead of GPT-4o. No fork owns this method, so it lives here; prelabeled
CSVs and freshly generated ones share this schema exactly::

    row_id, principle_id, order, prob_A, prob_B, prob_NA, cost

``principle_id`` is the value's **position in the value set**.

Labels are their own artifact, keyed by their own hash
(``data/labels/{value_set}/{model_short}-{hash}/``) rather than by the
experiment's ``artifact_id``: the same labels are reused by every experiment
that changes only ``tau``, the seed, ``max_pairs``, or a training
hyperparameter, none of which can change a judgment. The hash covers exactly
the knobs that *do* change one — judge model, ordering count, clipping, the
statement template, the resolved principle statements, and the source pools —
and deliberately excludes scheduling knobs (shards, concurrency, resources),
which is the same identity discipline the predictor store uses.

Intervention-method surface (see ``interventions.py``):

- :func:`build_data` — writes ``{datasets_dir}/{value}/dataset.jsonl`` (message
  lists; legacy dirs keep their ``dataset.csv``) per value, inline,
  when the judgments are already on disk.
- :func:`stages` — (oracle server + label array + merge + datasets, in generate
  mode) then the train+merge array per model. Training ends at merged
  checkpoints; evaluation is the eval side's business (``evals/``).
- :func:`manifest_entries` — the declared (model × value) checkpoint grid.

Also runnable directly, for the steps SLURM invokes::

    python -m valuegen.ground_truth.label_subset datasets -c <exp.yaml>
    python -m valuegen.ground_truth.label_subset label -c <exp.yaml> \\
        --source pku --shard 0 --shards 4
    python -m valuegen.ground_truth.label_subset merge -c <exp.yaml> --source pku
"""

from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path

from valuegen import values as V
from valuegen.config import ClusterConfig, config_hash
from valuegen.ground_truth import common, interventions
from valuegen.ground_truth.evaluation import serve_stage, wait_ready_sh
from valuegen.slurm import Stage, Task

TAU = 0.5
SEED = 42
HH_LABELS = "hh_test_seed_4ob.csv"
PKU_LABELS = "pku_test_4ob.csv"

LABEL_SCHEMA_VERSION = 1

# The oracle sees a declarative statement of the value, not the value set's
# gerund phrase ("not telling white lies" -> "The AI should prioritize not
# telling white lies"). One template for every value set: a per-value statement
# table would be a second registry to keep in sync with `value_sets/`.
STATEMENT_TEMPLATE = "The AI should prioritize {description}."

SYSTEM = ("You are a helpful assistant that simply responds whether another AI "
          "assistant's Response A or Response B better adheres to the principle for a "
          "given prompt.")

# Surface forms accepted for each answer class when reading first-token logprobs.
# The broad NA set guards against "N/A" / "NA" / a lone leading "N".
ANSWER_FORMS = {"A": {"A"}, "B": {"B"}, "NA": {"NA", "N/A", "N"}}


# ── Source pools: HH-RLHF + PKU-SafeRLHF test splits ─────────────────────────


def common_split(chosen: str, rejected: str) -> tuple[str, str, str]:
    """Split an HH chosen/rejected pair into (shared prompt, A, B)."""
    n = min(len(chosen), len(rejected))
    i = 0
    while i < n and chosen[i] == rejected[i]:
        i += 1
    prompt, a, b = chosen[:i], chosen[i:], rejected[i:]
    cut = prompt.rfind("\n\nAssistant:")
    if cut != -1:
        a, b = prompt[cut:] + a, prompt[cut:] + b
        prompt = prompt[:cut]
    return (
        prompt.strip(),
        a.replace("\n\nAssistant:", "").strip(),
        b.replace("\n\nAssistant:", "").strip(),
    )


def load_hh_text(split: str = "test"):
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
    rows = [
        dict(zip(("prompt", "A", "B"), common_split(ex["chosen"], ex["rejected"])))
        for ex in ds
    ]
    return pd.DataFrame(rows)


def load_pku_text(split: str = "test"):
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
    # FLIP: A=response_1, B=response_0 (verified against safer_response_id;
    # the harm-avoidance principle favors the safer response >80% of the time).
    return pd.DataFrame(
        [{"prompt": ex["prompt"], "A": ex["response_1"], "B": ex["response_0"]}
         for ex in ds]
    )


def load_ultrafeedback_text(split: str = "train_prefs"):
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    # chosen/rejected are [user, assistant] message lists for one prompt. Their
    # quality orientation is irrelevant here — the oracle judges both orderings.
    return pd.DataFrame(
        [{"prompt": ex["prompt"], "A": ex["chosen"][-1]["content"],
          "B": ex["rejected"][-1]["content"]} for ex in ds]
    )


def load_helpsteer2_text(split: str = "train"):
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split=split)
    # Rows arrive as consecutive same-prompt response pairs (10162/10162 on
    # train, verified 2026-08-25); keep only exact consecutive matches so a
    # future dataset revision degrades to fewer rows, not mispaired ones.
    prompts, responses = ds["prompt"], ds["response"]
    rows = [
        {"prompt": prompts[i], "A": responses[i], "B": responses[i + 1]}
        for i in range(0, len(ds) - 1, 2)
        if prompts[i] == prompts[i + 1]
    ]
    return pd.DataFrame(rows)


def load_pool_parquet(name: str):
    """Pools prebuilt by scripts/build_ca_wf_{,mt_}pools.py (pinned revisions,
    deterministic order); compute nodes read the parquet, never the Hub."""
    import pandas as pd

    return pd.read_parquet(f"data/pools/{name}.parquet", columns=["prompt", "A", "B"])


SOURCE_LOADERS = {
    "hh": load_hh_text,
    "pku": load_pku_text,
    "community_alignment_en": lambda: load_pool_parquet("community_alignment_en"),
    "wildfeedback_st": lambda: load_pool_parquet("wildfeedback_st"),
    # multi-turn expansion pools (scripts/build_ca_wf_mt_pools.py)
    "ca_en_mt_p1": lambda: load_pool_parquet("ca_en_mt_p1"),
    "ca_en_mt_p2": lambda: load_pool_parquet("ca_en_mt_p2"),
    "wf_mt": lambda: load_pool_parquet("wf_mt"),
    "hh_train": lambda: load_hh_text(split="train"),
    "pku_train": lambda: load_pku_text(split="train"),
    "ultrafeedback_train": load_ultrafeedback_text,
    "ultrafeedback_test": lambda: load_ultrafeedback_text(split="test_prefs"),
    "helpsteer2_train": load_helpsteer2_text,
    "helpsteer2_val": lambda: load_helpsteer2_text(split="validation"),
}


# ── Principles and the label store's identity ────────────────────────────────


def principle_statements(
    value_set: dict[str, str],
    template: str = STATEMENT_TEMPLATE,
    subset: list[str] | None = None,
) -> list[dict]:
    """Ordered ``[{id, key, statement}]`` — ``id`` is the value's position.

    Position *is* ``principle_id``, the column the judgment CSVs are keyed by.

    ``subset`` labels only some of the set's values (a value set often carries
    values no run trains on — judging them is real GPU time). Ids stay
    positional in the **full** set, so a subset's CSVs drop rows rather than
    renumbering them, and stay readable next to a full one.
    """
    if subset is not None:
        unknown = [key for key in subset if key not in value_set]
        if unknown:
            raise KeyError(f"labels.oracle.principles {unknown} are not in the value set")
    rows = []
    for pid, (key, description) in enumerate(value_set.items()):
        if subset is not None and key not in subset:
            continue
        rows.append(
            {"id": pid, "key": key, "statement": template.format(description=description)}
        )
    return rows


def labeled_principles(cfg: dict, cluster: ClusterConfig) -> list[dict]:
    """The principles this config's oracle judges (all of them, by default)."""
    oracle = oracle_config(cfg)
    return principle_statements(
        V.load_value_set(common.value_set_path(cfg["intervention"]["value_set"], cluster)),
        oracle["template"],
        subset=oracle.get("principles"),
    )


def oracle_config(cfg: dict) -> dict:
    return cfg["intervention"]["labels"]["oracle"]


def label_identity(cfg: dict, cluster: ClusterConfig) -> dict:
    """The knobs that can change a judgment — and nothing else (see module doc)."""
    gt = cfg["intervention"]
    labels = gt["labels"]
    oracle = oracle_config(cfg)
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "value_set": V.registry_name(gt["value_set"]),
        "principles": labeled_principles(cfg, cluster),
        "template": oracle["template"],
        "model": oracle["model"],
        "orderings": int(oracle["orderings"]),
        "max_chars": int(oracle["max_chars"]),
        "max_rows": None if oracle.get("max_rows") is None else int(oracle["max_rows"]),
        "sources": list(labels["sources"]),
    }


def label_id(cfg: dict, cluster: ClusterConfig) -> str:
    identity = label_identity(cfg, cluster)
    short = re.sub(r"[^A-Za-z0-9._-]", "_", str(identity["model"]).split("/")[-1])
    return f"{short}-{config_hash(identity)}"


def labels_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    value_set = V.registry_name(cfg["intervention"]["value_set"])
    return cluster.data / "labels" / value_set / label_id(cfg, cluster)


def source_csv(cfg: dict, cluster: ClusterConfig, source: str) -> Path:
    return labels_dir(cfg, cluster) / f"{source}.csv"


def part_csv(
    cfg: dict, cluster: ClusterConfig, source: str, shard: int, shards: int
) -> Path:
    return labels_dir(cfg, cluster) / "parts" / f"{source}.{shard}of{shards}.csv"


def _ckpt_csv(
    cfg: dict, cluster: ClusterConfig, source: str, shard: int, shards: int
) -> Path:
    return labels_dir(cfg, cluster) / "parts" / f"{source}.{shard}of{shards}.ckpt.csv"


def oracle_hostfile(cfg: dict, cluster: ClusterConfig) -> Path:
    return interventions.script_dir(cfg, cluster) / "oracle_host.txt"


def claim(cfg: dict, cluster: ClusterConfig) -> Path:
    """Write (or verify) the label directory's manifest, refusing a foreign one.

    Two different oracles cannot share a directory: the id is a hash of the
    identity, so a mismatch here means a hash collision or a hand-edited
    manifest, and either way the parts already on disk are not this run's.
    """
    import datetime

    import yaml

    directory = labels_dir(cfg, cluster)
    identity = label_identity(cfg, cluster)
    manifest = directory / "manifest.yaml"
    if manifest.is_file():
        record = yaml.safe_load(manifest.read_text()) or {}
        if record.get("identity") != identity:
            raise RuntimeError(
                f"{manifest} was written by a different oracle config; refusing "
                "to mix judgments. Delete the directory to rebuild."
            )
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    manifest.write_text(yaml.safe_dump({
        "label_id": label_id(cfg, cluster),
        "created": datetime.date.today().isoformat(),
        "experiment": cfg.get("name"),
        "identity": identity,
    }, sort_keys=False))
    return manifest


# ── The judge ────────────────────────────────────────────────────────────────


def build_prompt(prompt, statement: str, resp_a, resp_b, max_chars: int) -> list[dict]:
    """Fig. 4 oracle template. ``resp_a``/``resp_b`` are already in display order."""
    def clip(s):
        s = str(s)
        return s if len(s) <= max_chars else s[:max_chars] + " …[truncated]"

    user = (
        f"Prompt: {clip(prompt)}\n"
        f"Principle: {statement}\n"
        f"Response A: {clip(resp_a)}\n"
        f"Response B: {clip(resp_b)}\n"
        f"Answer: Which AI Response adheres to the Principle '{statement}' the most? "
        f"Answer only with A or B. Respond with NA if neither response clearly adheres "
        f"to it more or if the principle is irrelevant."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def resolve_base_url(api_base: str | Path) -> str:
    """Accept a host:port, a full URL, or a hostfile path containing host:port."""
    text = str(api_base)
    if Path(text).is_file():
        text = Path(text).read_text().strip()
    if "://" in text:
        stripped = text.rstrip("/")
        return stripped if stripped.endswith("/v1") else stripped + "/v1"
    return f"http://{text}/v1"


class Judge:
    """First-token ``A``/``B``/``NA`` probabilities from a chat completion."""

    def __init__(self, base_url: str, model: str, max_retries: int = 4, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key="api")
        self.client = client
        self.model = model
        # Qwen3 thinking would spend the single sampled token on <think>.
        self.extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if "qwen3" in model.lower() else {}
        )
        self.max_retries = max_retries

    def probs(self, messages: list[dict]) -> dict[str, float]:
        import numpy as np

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.0,
                    max_tokens=1, logprobs=True, top_logprobs=20,
                    extra_body=self.extra_body,
                )
                tops = response.choices[0].logprobs.content[0].top_logprobs
                p = {"A": 0.0, "B": 0.0, "NA": 0.0}
                for entry in tops:
                    token = entry.token.strip().upper()
                    for cls, forms in ANSWER_FORMS.items():
                        if token in forms:
                            p[cls] += float(np.exp(entry.logprob))
                total = sum(p.values())
                if total > 0:
                    return {k: v / total for k, v in p.items()}
                return {"A": 1 / 3, "B": 1 / 3, "NA": 1 / 3}
            except Exception as exc:  # noqa: BLE001
                # Do NOT return a placeholder here: a checkpointed row is
                # treated as done on resume, so a dead oracle would silently
                # poison the label store. Crash the shard instead; slurm
                # re-runs it and it resumes from the checkpoint.
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"judge failed after retries: {exc}") from exc
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")


# ── Labeling: one shard, then the merge ──────────────────────────────────────


def shard_rows(
    n_rows: int, shard: int, shards: int, max_rows: int | None = None
) -> list[int]:
    row_ids = list(range(n_rows if max_rows is None else min(n_rows, max_rows)))
    return [r for r in row_ids if r % shards == shard]


_CKPT_RAW_COLS = ["row_id", "principle_id", "order", "prob_A", "prob_B", "prob_NA", "cost"]
_CKPT_FLUSH_INTERVAL = 500


def _read_ckpt(path):
    """Read a checkpoint CSV whether or not it carries a header row.

    A ckpt inherited from an interrupted run can be headerless (the header is
    only written on the first flush of a file that does not yet exist, so a
    pre-existing empty file suppresses it). Sniff the first line and supply the
    column names when the header is absent so aggregation never KeyErrors.
    """
    import pandas as pd

    with open(path) as f:
        first = f.readline()
    if first.startswith("row_id"):
        return pd.read_csv(path, on_bad_lines="skip")
    return pd.read_csv(path, header=None, names=_CKPT_RAW_COLS, on_bad_lines="skip")


def label_shard(
    cfg: dict,
    cluster: ClusterConfig,
    source: str,
    shard: int,
    shards: int,
    judge: Judge | None = None,
) -> Path:
    """Judge one shard of one source pool -> ``parts/{source}.{k}of{m}.csv``.

    Results are checkpointed to a ``.ckpt.csv`` every 500 calls and on
    shutdown, so a walltime-killed shard resumes from where it left off
    instead of losing all progress.  Tasks are shuffled (deterministic seed)
    so partial completion is a random sample of the full task set.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    oracle = oracle_config(cfg)
    if source not in SOURCE_LOADERS:
        raise KeyError(f"Unknown label source {source!r}; known: {sorted(SOURCE_LOADERS)}")
    claim(cfg, cluster)
    principles = labeled_principles(cfg, cluster)
    text = SOURCE_LOADERS[source]()
    row_ids = shard_rows(len(text), shard, shards, oracle.get("max_rows"))
    print(
        f"[{source} shard {shard}/{shards}] {len(row_ids)} pairs × "
        f"{len(principles)} principles × {oracle['orderings']} ordering(s)"
    )
    if judge is None:
        # VALUEGEN_ORACLE_API_BASE lets a self-contained node job (its own
        # vLLM + local clients, no serve_oracle stage) point the client at
        # its server without a per-config hostfile. Scheduling only.
        api_base = (
            os.environ.get("VALUEGEN_ORACLE_API_BASE")
            or oracle.get("api_base")
            or oracle_hostfile(cfg, cluster)
        )
        judge = Judge(resolve_base_url(api_base), oracle["model"])

    orderings = [0] if int(oracle["orderings"]) == 1 else [0, 1]
    statement_of = {p["id"]: p["statement"] for p in principles}
    all_tasks = [(r, p["id"], o) for r in row_ids for p in principles for o in orderings]

    # ── Resume from checkpoint ──────────────────────────────────────────
    ckpt_path = _ckpt_csv(cfg, cluster, source, shard, shards)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[int, int, int]] = set()
    if ckpt_path.is_file():
        try:
            prev = _read_ckpt(ckpt_path)
            # Uniform 1/3 rows are the legacy failure placeholder (oracle
            # down); re-judge them rather than trusting them.
            bad = (prev[["prob_A", "prob_B", "prob_NA"]] - 1 / 3).abs().max(axis=1) < 1e-9
            if bad.any():
                print(f"  dropping {int(bad.sum())} placeholder rows from checkpoint")
                prev = prev[~bad]
                prev.to_csv(ckpt_path, index=False)
            done = set(zip(prev["row_id"], prev["principle_id"], prev["order"]))
            print(f"  resumed: {len(done)} calls loaded from {ckpt_path}")
        except Exception as exc:
            print(f"  warning: corrupt checkpoint, starting fresh ({exc})")

    tasks = [t for t in all_tasks if t not in done]
    # Deterministic shuffle so partial completion is a random sample.
    random.Random(hash((source, shard, shards))).shuffle(tasks)

    if not tasks:
        print("  all tasks already checkpointed, writing final CSV")
    else:
        print(f"  {len(done)} done, {len(tasks)} remaining")

    def run_one(task):
        row_id, pid, order = task
        row = text.iloc[row_id]
        a, b = (row.A, row.B) if order == 0 else (row.B, row.A)
        p = judge.probs(
            build_prompt(
                row.prompt, statement_of[pid], a, b, int(oracle["max_chars"])
            )
        )
        prob_a, prob_b = (p["A"], p["B"]) if order == 0 else (p["B"], p["A"])
        return {"row_id": row_id, "principle_id": pid, "order": order,
                "prob_A": prob_a, "prob_B": prob_b, "prob_NA": p["NA"], "cost": 0.0}

    # ── Run with periodic checkpoint flushes ────────────────────────────
    start = time.time()
    buf: list[dict] = []

    def _flush():
        if not buf:
            return
        header = not ckpt_path.is_file()
        with open(ckpt_path, "a") as f:
            if header:
                f.write(",".join(_CKPT_RAW_COLS) + "\n")
            for rec in buf:
                f.write(",".join(str(rec[c]) for c in _CKPT_RAW_COLS) + "\n")
        buf.clear()

    total_done = len(done)
    total_all = len(all_tasks)
    with ThreadPoolExecutor(max_workers=int(oracle["concurrency"])) as pool:
        for i, record in enumerate(pool.map(run_one, tasks), 1):
            buf.append(record)
            if i % _CKPT_FLUSH_INTERVAL == 0:
                _flush()
                rate = i / (time.time() - start)
                print(
                    f"  {total_done + i}/{total_all} calls ({rate:.0f}/s, "
                    f"eta {(len(tasks) - i) / rate / 60:.0f} min) [ckpt]",
                    flush=True,
                )
    _flush()

    # ── Aggregate and write final part CSV ──────────────────────────────
    raw = _read_ckpt(ckpt_path)
    agg = (raw.groupby(["row_id", "principle_id"], as_index=False)
              [["prob_A", "prob_B", "prob_NA", "cost"]].mean())
    agg.insert(2, "order", 0)
    out = part_csv(cfg, cluster, source, shard, shards)
    agg.to_csv(out, index=False)
    ckpt_path.unlink()
    print(f"wrote {len(agg)} judgments -> {out} ({time.time() - start:.0f}s)")
    return out


def merge_source(cfg: dict, cluster: ClusterConfig, source: str, shards: int) -> Path:
    """Concatenate a source's shards into ``{source}.csv``, validating coverage.

    A silently missing shard would just shrink the decisive pool, so the parts
    are counted, not globbed: every shard must be present, and every labeled
    row must carry a judgment for every principle.
    """
    import pandas as pd

    parts = [part_csv(cfg, cluster, source, k, shards) for k in range(shards)]
    missing = [p for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source}: {len(missing)}/{shards} shards missing, e.g. {missing[0]}"
        )
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if df.duplicated(["row_id", "principle_id"]).any():
        raise ValueError(f"{source}: duplicate (row_id, principle_id) across shards")
    n_principles = len(labeled_principles(cfg, cluster))
    counts = df.groupby("row_id").size()
    ragged = counts[counts != n_principles]
    if len(ragged):
        raise ValueError(
            f"{source}: {len(ragged)} rows are missing principles "
            f"(expected {n_principles}, e.g. row {ragged.index[0]} has {ragged.iloc[0]})"
        )
    df = df.sort_values(["row_id", "principle_id"]).reset_index(drop=True)
    out = source_csv(cfg, cluster, source)
    df.to_csv(out, index=False)
    print(f"{source}: {len(parts)} shards -> {len(df)} judgments -> {out}")
    return out


# ── Which judgments, and where ───────────────────────────────────────────────


def generates_labels(cfg: dict) -> bool:
    """Whether this config labels its own pools (``labels.oracle``) or reuses
    existing judgment CSVs (``labels.dir``)."""
    labels = cfg["intervention"].get("labels", {})
    has_dir, has_oracle = "dir" in labels, "oracle" in labels
    if has_dir and has_oracle:
        raise ValueError(
            "ground_truth.labels sets both 'dir' and 'oracle': a run either "
            "reuses existing judgments or generates its own, not both"
        )
    if not (has_dir or has_oracle):
        raise ValueError(
            "ground_truth.labels needs either 'dir' (reuse existing judgment "
            "CSVs) or 'oracle' (label the pools with a served judge)"
        )
    if has_oracle:
        oracle = labels["oracle"]
        if not oracle.get("model"):
            raise ValueError("ground_truth.labels.oracle needs a 'model'")
        if oracle.get("serve") != "local" and not oracle.get("api_base"):
            raise ValueError(
                "ground_truth.labels.oracle needs 'serve: local' (a vLLM server "
                "stage) or an explicit 'api_base'"
            )
        unknown = set(labels["sources"]) - set(SOURCE_LOADERS)
        if unknown:
            raise KeyError(
                f"Unknown label sources {sorted(unknown)}; known: "
                f"{sorted(SOURCE_LOADERS)}"
            )
    return has_oracle


def label_sources(cfg: dict, cluster: ClusterConfig) -> list[tuple[str, Path]]:
    """``[(source, judgment_csv)]`` for this config, in either mode."""
    labels = cfg["intervention"]["labels"]
    if generates_labels(cfg):
        return [
            (source, source_csv(cfg, cluster, source))
            for source in labels["sources"]
        ]
    directory = common.configured_path(labels["dir"], cfg, cluster)
    return [
        ("hh", directory / labels.get("hh_csv", HH_LABELS)),
        ("pku", directory / labels.get("pku_csv", PKU_LABELS)),
    ]


def labels_ready(cfg: dict, cluster: ClusterConfig) -> bool:
    return all(path.is_file() for _, path in label_sources(cfg, cluster))


def import_labels(cfg: dict, cluster: ClusterConfig, donors: list[str]) -> Path:
    """Populate this config's label store from existing stores instead of
    relabeling: every ``labels.sources`` CSV is symlinked from the first donor
    that has it. A donor is a label id (dir name under the value set's labels
    root) or a path; its manifest identity must equal this config's in every
    key but ``sources`` (same oracle, template, principles, ...), so the
    judgments are what this config would have produced. The manifest records
    the provenance under ``imported``. Idempotent; refuses to repoint a link.
    """
    import os

    import yaml

    labels = cfg["intervention"]["labels"]
    if not generates_labels(cfg):
        raise ValueError("import needs a labels.oracle block (labels.dir stores are not imported)")
    want = dict(label_identity(cfg, cluster))
    want.pop("sources")
    root = labels_dir(cfg, cluster).parent
    found: dict[str, Path] = {}
    for donor in donors:
        ddir = Path(donor) if "/" in donor else root / donor
        record = yaml.safe_load((ddir / "manifest.yaml").read_text()) or {}
        have = dict(record.get("identity") or {})
        have_sources = have.pop("sources", [])
        diff = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
        if diff:
            raise RuntimeError(f"{ddir}: oracle identity differs from this config's in {diff}")
        for source in labels["sources"]:
            if source in have_sources and source not in found and (ddir / f"{source}.csv").is_file():
                found[source] = ddir
    missing = [s for s in labels["sources"] if s not in found]
    if missing:
        raise FileNotFoundError(f"labels.sources {missing} not found in donors {donors}")
    manifest = claim(cfg, cluster)
    for source, ddir in found.items():
        dst = source_csv(cfg, cluster, source)
        target = ddir / f"{source}.csv"
        if dst.is_symlink() or dst.exists():
            if not dst.is_symlink() or dst.resolve() != target.resolve():
                raise RuntimeError(f"{dst} exists and is not a link to {target}; refusing")
            continue
        dst.symlink_to(os.path.relpath(target, dst.parent))
        print(f"import: {dst.name} <- {ddir.name}")
    record = yaml.safe_load(manifest.read_text()) or {}
    record["imported"] = {s: d.name for s, d in found.items()}
    manifest.write_text(yaml.safe_dump(record, sort_keys=False))
    return manifest


def check_train_values_labeled(cfg: dict, cluster: ClusterConfig) -> None:
    """A trained value whose principle is not judged would get an empty dataset.

    Only reachable via a ``labels.oracle.principles`` subset, and only worth
    catching before the run rather than after it trains on nothing.
    """
    if not generates_labels(cfg):
        return
    labeled = {p["key"] for p in labeled_principles(cfg, cluster)}
    missing = [v for v in cfg["intervention"]["values"] if v not in labeled]
    if missing:
        raise ValueError(
            f"ground_truth.values {missing} are not judged by the oracle — add "
            "them to labels.oracle.principles (or drop the subset, which labels "
            "the whole value set)"
        )


def shards_for(cfg: dict, source: str) -> int:
    shards = int(oracle_config(cfg).get("shards", {}).get(source, 1))
    if shards < 1:
        raise ValueError(f"labels.oracle.shards[{source}] must be >= 1")
    return shards


# ── DPO datasets from the judgments ──────────────────────────────────────────


def _chat(role: str, content) -> list[dict]:
    # A real message list, serialized as JSON in the JSONL dataset so
    # train_dpo's conversational path applies the chat template (the legacy
    # CSVs carried str() reprs, trained as raw text).
    return [{"role": role, "content": str(content)}]


def build_data(cfg: dict, cluster: ClusterConfig) -> None:
    """Build the per-value DPO datasets. Idempotent (skips existing files).

    In generate mode this is a no-op until the judgments exist — they are built
    by the ``label``/``merge_labels`` SLURM stages, and the ``datasets`` stage
    calls back into here once they do.
    """
    import numpy as np
    import pandas as pd

    gt = cfg["intervention"]
    labels = gt["labels"]
    tau = float(labels.get("tau", TAU))
    seed = int(labels.get("seed", SEED))
    max_pairs = gt.get("max_pairs")
    if max_pairs is not None and int(max_pairs) < 0:
        raise ValueError("max_pairs must be non-negative or null")
    out_root = interventions.datasets_dir(cfg, cluster)
    train_values = list(gt["values"])

    check_train_values_labeled(cfg, cluster)
    todo = [
        v for v in train_values
        if not interventions.dataset_file(out_root / v).is_file()
    ]
    if not todo:
        print(f"label_subset datasets: all {len(train_values)} present in {out_root}")
        return
    if not labels_ready(cfg, cluster):
        missing = [str(p) for _, p in label_sources(cfg, cluster) if not p.is_file()]
        if generates_labels(cfg):
            print(f"label_subset: judgments not built yet ({missing[0]}); the "
                  "label stages will generate them")
            return
        raise FileNotFoundError(f"Judgment CSVs missing: {missing}")

    # principle_id is the value's position in the value set — the key the
    # judgment CSVs are written with (see principle_statements).
    value_set = V.load_value_set(common.value_set_path(cfg["intervention"]["value_set"], cluster))
    pid_of = {p["key"]: p["id"] for p in principle_statements(value_set)}
    unknown = [v for v in train_values if v not in pid_of]
    if unknown:
        raise KeyError(f"ground_truth.values {unknown} are not in the value set")

    def load_scores(path):
        d = pd.read_csv(path)
        d["score"] = d.prob_A - d.prob_B
        return d

    # labels.train_sources: build datasets from a subset of the labeled pools
    # (e.g. the *_train splits of a store that also judged the test pools).
    # Outside label_identity, so the label store is shared; inside the
    # intervention block, so the datasets it produces get their own id.
    train_sources = labels.get("train_sources")
    if train_sources is not None:
        extra = set(train_sources) - set(labels["sources"])
        if extra:
            raise ValueError(f"labels.train_sources {sorted(extra)} not in labels.sources")
    print("loading source pool text ...")
    sources = []
    for name, path in label_sources(cfg, cluster):
        if train_sources is not None and name not in train_sources:
            continue
        scores = load_scores(path)
        text = SOURCE_LOADERS[name]()
        assert scores.row_id.max() + 1 <= len(text), f"{name}: row_id beyond pool"
        sources.append((name, scores, text))

    # Exclusive assignment (intervention.assignment): each eligible
    # (prompt,{A,B}) row goes to at most one value. The table is solved once
    # over ALL train values and cached beside the datasets, so a partial
    # rebuild (some values present) can never re-solve into a different split.
    assignment = gt.get("assignment")
    assigned = None
    if assignment:
        from valuegen.ground_truth import exclusive_assignment as EA

        if max_pairs is not None:
            raise ValueError("max_pairs is not supported with intervention.assignment")
        table = out_root / "assignment.csv"
        if table.is_file():
            assigned = pd.read_csv(table)
            print(f"assignment: reusing {table} ({len(assigned):,} rows)")
        else:
            long = pd.concat(
                [sc.assign(source=name)[["source", "row_id", "principle_id", "score"]]
                 for name, sc, _ in sources], ignore_index=True)
            def keyed(fn):
                return pd.concat(
                    [pd.Series(fn(text).values,
                               index=pd.MultiIndex.from_arrays(
                                   [[name] * len(text), np.arange(len(text))],
                                   names=["source", "row_id"]))
                     for name, _, text in sources])

            keys = keyed(EA.triple_key)
            # assignment.pool: solve the exclusive split over a superset of
            # the trained values (must contain them), so a subset run trains
            # exactly the datasets the full run would.
            pool = list(assignment.get("pool") or train_values)
            unknown_pool = [v for v in pool if v not in pid_of]
            if unknown_pool:
                raise KeyError(f"assignment.pool {unknown_pool} not in the value set")
            not_in_pool = [v for v in train_values if v not in pool]
            if not_in_pool:
                raise ValueError(f"values {not_in_pool} missing from assignment.pool")
            assigned = EA.assign(
                long, keys, [pid_of[v] for v in pool],
                rule=str(assignment["rule"]), tau=tau,
                cap=int(assignment.get("cap", 5000)),
                floor=int(assignment.get("floor", 1000)),
                seed=int(assignment.get("seed", seed)),
                unit=str(assignment.get("unit", "triple")),
                prompt_keys=keyed(EA.prompt_key),
            )
            out_root.mkdir(parents=True, exist_ok=True)
            assigned.to_csv(table, index=False)
            print(f"assignment: wrote {table}")

    for value in todo:
        pid = pid_of[value]
        parts = []
        for name, sc, text in sources:
            if assigned is not None:
                g = assigned[(assigned.source == name) & (assigned.principle_id == pid)]
                g = g[["row_id", "score"]]
            else:
                g = sc[(sc.principle_id == pid) & (sc.score.abs() >= tau)]
            g = g.merge(
                text.reset_index().rename(columns={"index": "row_id"}), on="row_id"
            )
            g["source"] = name
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if max_pairs is not None:
            df = df.iloc[: int(max_pairs)].reset_index(drop=True)
        fav_a = df.score > 0
        out = pd.DataFrame(
            {
                "scenario_id": [
                    f"{value}_{s}_{int(r)}" for s, r in zip(df.source, df.row_id)
                ],
                "prompt": [_chat("user", p) for p in df.prompt],
                "chosen": [
                    _chat("assistant", a if fa else b)
                    for a, b, fa in zip(df.A, df.B, fav_a)
                ],
                "rejected": [
                    _chat("assistant", b if fa else a)
                    for a, b, fa in zip(df.A, df.B, fav_a)
                ],
                "score": df.score.values,
                "source": df.source.values,
            }
        )
        dest = out_root / value
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / "dataset.jsonl"
        out.to_json(out_path, orient="records", lines=True)
        floor = "  <-- below the N=800 train-signal floor" if len(out) < 800 else ""
        print(f"{value}: {len(out)} pairs -> {out_path}{floor}")


# ── Stage list ───────────────────────────────────────────────────────────────


def label_stages(cfg: dict, cluster: ClusterConfig) -> list[Stage]:
    """Oracle server + sharded label array + per-source merge.

    Emitted only in generate mode, and only while judgments are missing — a
    finished label store makes these stages disappear rather than re-submit a
    server. The array is CPU-only (it talks HTTP to the server), and the
    server is cancelled the moment labeling finishes, so it does not hold GPUs
    through training (the old controller's ``kill_server`` step, made
    declarative).
    """
    data_root = interventions.datasets_dir(cfg, cluster)
    datasets_ready = all(
        interventions.dataset_file(data_root / value).is_file()
        for value in cfg["intervention"]["values"]
    )
    if datasets_ready or not generates_labels(cfg) or labels_ready(cfg, cluster):
        return []

    oracle = oracle_config(cfg)
    sources = list(cfg["intervention"]["labels"]["sources"])
    hostfile = oracle_hostfile(cfg, cluster)
    served = oracle.get("serve") == "local"

    prelude = wait_ready_sh(hostfile, "oracle") + "\n" if served else ""
    label_tasks = []
    for source in sources:
        shards = shards_for(cfg, source)
        for shard in range(shards):
            label_tasks.append(Task(
                key=f"label_{source}_{shard}",
                command=(
                    prelude
                    + f"python -m valuegen.ground_truth.label_subset label"
                    f" -c {cfg['_path']} --source {source}"
                    f" --shard {shard} --shards {shards}"
                ),
                done=part_csv(cfg, cluster, source, shard, shards),
            ))

    stages: list[Stage] = []
    # A server stage is submitted whether or not its dependents have work, so
    # only emit it (and the array) while shards are actually missing: a run
    # resuming at the merge must not park a judge on a GPU for nothing.
    pending_shards = [t for t in label_tasks if not t.is_done()]
    if pending_shards:
        if served:
            stages.append(serve_stage(
                name="serve_oracle",
                model=oracle["model"],
                hostfile=hostfile,
                gpus=int(oracle.get("gpus", 2)),
                mem=oracle.get("mem", "96G"),
                time=oracle.get("time", "1-00:00:00"),
                env=oracle.get("env", "eval_api"),
                # Serving knobs: scheduling only, outside `label_identity`.
                # The label client is prefill-bound (max_tokens=1), so the
                # per-step token budget is what sets its throughput; prefix
                # caching is off for hybrid oracles whose forced block size
                # exceeds the prompt length (see `vllm_serve_cmd`).
                max_num_seqs=int(oracle.get("max_num_seqs", 64)),
                max_num_batched_tokens=int(oracle.get("max_num_batched_tokens", 16384)),
                enable_prefix_caching=bool(oracle.get("enable_prefix_caching", True)),
                extra_args=tuple(oracle.get("serve_args", ())),
            ))
        res = oracle.get("label", {})
        stages.append(Stage(
            name="label",
            tasks=label_tasks,
            time=res.get("time", "18:00:00"),
            mem=res.get("mem", "24G"),
            env=res.get("env", "default"),
            cancel_servers=("serve_oracle",) if served else (),
            needs_servers=("serve_oracle",) if served else (),
            # The label array can exceed general's 50-job submit cap (127k HH/PKU
            # prompts sharded to 53 tasks), so route it to the array partition
            # (preempt) when the cluster declares one; falls back to general.
            array_partition=True,
        ))
    stages.append(Stage(
        name="merge_labels",
        tasks=[
            Task(
                key=f"merge_{source}",
                command=(
                    f"python -m valuegen.ground_truth.label_subset merge"
                    f" -c {cfg['_path']} --source {source}"
                ),
                done=source_csv(cfg, cluster, source),
            )
            for source in sources
        ],
        time="00:30:00",
        mem="16G",
    ))
    return stages


def datasets_stage(cfg: dict, cluster: ClusterConfig) -> Stage:
    """The DPO-dataset build as a task, for the runs whose labels are built by
    SLURM (``build_data`` cannot run inline before the judgments exist)."""
    root = interventions.datasets_dir(cfg, cluster)
    train_values = list(cfg["intervention"]["values"])
    return Stage(
        name="datasets",
        tasks=[Task(
            key="datasets",
            command=(
                f"python -m valuegen.ground_truth.label_subset datasets"
                f" -c {cfg['_path']}"
            ),
            done=lambda: all(
                interventions.dataset_file(root / v).is_file()
                for v in train_values
            ),
        )],
        time="01:00:00",
        mem="16G",
    )


def manifest_entries(cfg: dict, cluster: ClusterConfig) -> list[dict]:
    return interventions.checkpoint_entries(cfg, cluster)


def stages(cfg: dict, cluster: ClusterConfig) -> list[Stage]:
    result: list[Stage] = []
    if generates_labels(cfg):
        check_train_values_labeled(cfg, cluster)
        result += label_stages(cfg, cluster)
        result.append(datasets_stage(cfg, cluster))
    result += interventions.train_stages(cfg, cluster)
    return result


# ── CLI for the steps SLURM invokes ──────────────────────────────────────────


def main() -> None:
    import argparse

    from valuegen.config import load_cluster, load_experiment

    parser = argparse.ArgumentParser(description="label_subset GT method: data steps")
    sub = parser.add_subparsers(dest="command", required=True)

    p_datasets = sub.add_parser("datasets", help="build the per-value DPO datasets")
    p_label = sub.add_parser("label", help="judge one shard of one source pool")
    p_label.add_argument("--source", required=True)
    p_label.add_argument("--shard", type=int, required=True)
    p_label.add_argument("--shards", type=int, required=True)
    p_merge = sub.add_parser("merge", help="merge one source's label shards")
    p_merge.add_argument("--source", required=True)
    p_import = sub.add_parser(
        "import", help="fill this config's label store by linking existing stores' CSVs")
    p_import.add_argument("--from", dest="donors", action="append", required=True,
                          metavar="STORE", help="donor label id or path (repeatable)")
    for p in (p_datasets, p_label, p_merge, p_import):
        p.add_argument("--config", "-c", required=True)
        p.add_argument("--cluster", default=None)

    args = parser.parse_args()
    cfg = load_experiment(args.config)
    cluster = load_cluster(args.cluster)
    if args.command == "datasets":
        build_data(cfg, cluster)
    elif args.command == "label":
        label_shard(cfg, cluster, args.source, args.shard, args.shards)
    elif args.command == "merge":
        merge_source(cfg, cluster, args.source, shards_for(cfg, args.source))
    elif args.command == "import":
        print(import_labels(cfg, cluster, args.donors))


if __name__ == "__main__":
    main()
