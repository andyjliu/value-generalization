"""Source discovery and per-arm training mixes (``mv data``).

Ported from ``ground_truth.rq3_mixes`` with this package's own salts
(:mod:`_hashing`) and the budget taken from ``budget.rows``.

Pipeline per trained arm (:func:`build_arm`):

1. source rows are loaded by file iteration, each with a content-addressed
   identity ``(source revision, value, source, scenario_id, SHA256(triple))``;
2. per value the rows are ordered by :func:`_hashing.row_order_key` and the
   arm's balanced quota (``budget.rows`` split evenly over its values, the
   remainder by the quota hash) takes a *prefix*, so any two arms sharing a
   value take nested selections (checked by :func:`nested_selection_check`);
3. the union is shuffled by :func:`_hashing.mix_order_key`;
4. ``dataset.jsonl`` (prompt/chosen/rejected), ``rows.jsonl`` (provenance)
   and ``composition.json`` are published atomically under
   ``{exp_dir}/mixes/{arm}/``; a rebuild that disagrees with a published
   mix fails instead of overwriting it.

The token audit (:func:`token_audit`) reproduces TRL's DPO tokenization and
the collator's ``keep_start`` truncation at the recipe's ``max_length`` and
records what the trainer will see: TRL prompt-length drops (rows whose
prompt alone fills ``max_length``), truncated pairs, empty or identical
completions after truncation, and the post-drop step geometry.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from valuegen.config import ClusterConfig
from valuegen.multivalue import _hashing as H
from valuegen.multivalue.config import MultivalueConfig
from valuegen.values import load_value_set, registry_name

DATASET_KEYS = ("prompt", "chosen", "rejected")
COMPOSITION_SCHEMA_VERSION = 1
TOKEN_AUDIT_VERSION = 2


class SourceError(RuntimeError):
    pass


class GateError(RuntimeError):
    """A launch gate failed (the mix must not be trained on)."""


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ── source discovery ─────────────────────────────────────────────────────────


def universe_registry_name(cfg: MultivalueConfig) -> str:
    return registry_name(cfg.universe.value_set)


def source_artifact_dir(cfg: MultivalueConfig, cluster: ClusterConfig) -> Path:
    return (cluster.data / "interventions" / cfg.source.method
            / universe_registry_name(cfg) / cfg.source.artifact)


def source_datasets_dir(cfg: MultivalueConfig, cluster: ClusterConfig) -> Path:
    return source_artifact_dir(cfg, cluster) / "datasets"


def source_revision(cfg: MultivalueConfig, cluster: ClusterConfig) -> str:
    """The row-identity prefix: the artifact's upstream revision when its
    ``hf_import.json`` records one, else the artifact id itself."""
    p = source_artifact_dir(cfg, cluster) / "hf_import.json"
    if p.is_file():
        try:
            rev = json.loads(p.read_text(encoding="utf-8")).get("revision")
            if rev:
                return f"{cfg.source.artifact}@{rev}"
        except (OSError, ValueError):
            pass
    return cfg.source.artifact


def discover_source_values(cfg: MultivalueConfig, cluster: ClusterConfig) -> list[str]:
    """Values with a ``<value>/dataset.jsonl`` under the source artifact, sorted."""
    root = source_datasets_dir(cfg, cluster)
    if not root.is_dir():
        raise SourceError(
            f"source artifact has no datasets at {root}.\n"
            f"Build it first with the label_subset intervention, e.g.\n"
            f"    valuegen gt intervene -c configs/experiments/ground_truth/<the {cfg.source.artifact.split('-')[0]} config>.yaml"
        )
    return sorted(p.parent.name for p in root.glob("*/dataset.jsonl"))


def resolve_universe(cfg: MultivalueConfig, cluster: ClusterConfig) -> tuple[list[str], dict[str, str]]:
    """The universe value list (sorted) and ``{value: description}``.

    With ``universe.restrict_to_source`` the registry set is intersected with
    the values that have source rows; values dropped that way are reported by
    the caller, values with rows but not in the registry set are an error.
    """
    descriptions = load_value_set(cfg.universe.value_set)
    names = sorted(descriptions)
    if cfg.universe.restrict_to_source:
        have = discover_source_values(cfg, cluster)
        unknown = sorted(set(have) - set(names))
        if unknown:
            raise SourceError(
                f"source artifact has rows for values outside {cfg.universe.value_set}: {unknown[:5]}"
            )
        names = [v for v in names if v in set(have)]
        if not names:
            raise SourceError(f"no value of {cfg.universe.value_set} has source rows")
    return names, {v: descriptions[v] for v in names}


def load_external(cfg: MultivalueConfig) -> tuple[list[str], dict[str, str]]:
    """External value list (sorted) and descriptions."""
    descriptions = load_value_set(cfg.external.value_set)
    if not isinstance(descriptions, dict) or not descriptions:
        raise SourceError(f"external.value_set {cfg.external.value_set} must be a non-empty {{value: description}}")
    return sorted(descriptions), descriptions


# ── source rows ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SourceRow:
    value: str
    source: str
    scenario_id: str
    score: float | None
    prompt: tuple  # tuple of message dicts
    chosen: tuple
    rejected: tuple
    triple_sha256: str
    identity: str  # canonical json of the identity tuple
    identity_sha256: str
    file_index: int

    def dataset_record(self) -> dict:
        return {"prompt": list(self.prompt), "chosen": list(self.chosen), "rejected": list(self.rejected)}

    def provenance(self) -> dict:
        return {
            "value": self.value, "source": self.source, "scenario_id": self.scenario_id,
            "score": self.score, "triple_sha256": self.triple_sha256,
            "identity": self.identity, "identity_sha256": self.identity_sha256,
            "source_file_index": self.file_index,
        }


def _freeze_messages(messages) -> tuple:
    if not isinstance(messages, list) or not messages:
        raise SourceError("message list must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise SourceError(f"message must be {{role, content}}: {m!r}"[:200])
        if not isinstance(m["content"], str):
            raise SourceError("message content must be a string")
        out.append({"role": m["role"], "content": m["content"]})
    return tuple(out)


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, dict]]:
    """Read JSONL by file iteration (never ``str.splitlines()``, which also
    splits on Unicode separators that are legal inside JSON strings)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, line in enumerate(f):
            if line.strip() == "":
                continue
            yield i, json.loads(line)


def load_value_rows(path: str | Path, value: str, revision: str) -> list[SourceRow]:
    rows = []
    for i, r in iter_jsonl(path):
        prompt = _freeze_messages(r["prompt"])
        chosen = _freeze_messages(r["chosen"])
        rejected = _freeze_messages(r["rejected"])
        triple = H.sha256_json({"prompt": list(prompt), "chosen": list(chosen), "rejected": list(rejected)})
        identity = H.canonical_json([revision, value, r.get("source"), r.get("scenario_id"), triple])
        rows.append(SourceRow(
            value=value, source=str(r.get("source")), scenario_id=str(r.get("scenario_id")),
            score=r.get("score"), prompt=prompt, chosen=chosen, rejected=rejected,
            triple_sha256=triple, identity=identity, identity_sha256=H.sha256_text(identity),
            file_index=i,
        ))
    return rows


def load_source(datasets_root: str | Path, values: Sequence[str], revision: str) -> dict[str, list[SourceRow]]:
    root = Path(datasets_root)
    out = {}
    for v in values:
        path = root / v / "dataset.jsonl"
        if not path.is_file():
            raise SourceError(f"no source rows for {v!r} at {path}")
        out[v] = load_value_rows(path, v, revision)
    return out


def audit_source(rows_by_value: Mapping[str, Sequence[SourceRow]], datasets_root: str | Path,
                 min_per_value: int | None = None) -> dict:
    """Row-shape and uniqueness invariants across the loaded values; raises
    :class:`SourceError` on any violation. ``min_per_value`` is the largest
    quota any arm will take of a value (fewer rows than that is a violation)."""
    problems = []
    identities, prompts, triples = set(), set(), set()
    n_rows = 0
    per_value = {}
    identical = bad_shape = empty_chosen = empty_rejected = 0
    sources: dict[str, dict[str, int]] = {}
    for v, rows in rows_by_value.items():
        per_value[v] = len(rows)
        if min_per_value is not None and len(rows) < min_per_value:
            problems.append(f"{v}: {len(rows)} rows, need at least {min_per_value}")
        for r in rows:
            n_rows += 1
            if r.identity in identities:
                problems.append(f"{v}: repeated identity {r.identity[:80]}")
            identities.add(r.identity)
            prompts.add(H.canonical_json(list(r.prompt)))
            triples.add(r.triple_sha256)
            if r.chosen == r.rejected:
                identical += 1
            if not any(m["role"] == "user" for m in r.prompt) or r.prompt[-1]["role"] != "user":
                bad_shape += 1
            if not (len(r.chosen) == 1 and r.chosen[0]["role"] == "assistant"):
                bad_shape += 1
            elif not r.chosen[0]["content"]:
                empty_chosen += 1
            if not (len(r.rejected) == 1 and r.rejected[0]["role"] == "assistant"):
                bad_shape += 1
            elif not r.rejected[0]["content"]:
                empty_rejected += 1
            sources.setdefault(v, {}).setdefault(r.source, 0)
            sources[v][r.source] += 1
    if len(prompts) != n_rows:
        problems.append(f"{n_rows - len(prompts)} repeated serialized prompts across the source")
    if len(triples) != n_rows:
        problems.append(f"{n_rows - len(triples)} repeated preference triples across the source")
    if identical:
        problems.append(f"{identical} rows with identical chosen/rejected")
    if bad_shape:
        problems.append(f"{bad_shape} rows with an unexpected message shape")
    file_hashes = {v: H.sha256_file(Path(datasets_root) / v / "dataset.jsonl") for v in rows_by_value}
    counts = sorted(per_value.values())
    report = {
        "n_values": len(rows_by_value), "n_rows": n_rows, "rows_per_value": per_value,
        "min_rows_per_value": counts[0] if counts else 0, "max_rows_per_value": counts[-1] if counts else 0,
        "unique_identities": len(identities), "unique_prompts": len(prompts),
        "unique_triples": len(triples), "identical_chosen_rejected": identical,
        "bad_shape_rows": bad_shape, "empty_chosen_content": empty_chosen,
        "empty_rejected_content": empty_rejected, "sources_by_value": sources,
        "file_sha256": file_hashes, "problems": problems,
    }
    if problems:
        raise SourceError("source audit failed: " + "; ".join(problems[:10]))
    return report


# ── ordering + selection ─────────────────────────────────────────────────────


def order_value_rows(rows: Sequence[SourceRow], seed: int, value: str) -> list[SourceRow]:
    return sorted(rows, key=lambda r: (H.row_order_key(seed, value, r.identity), r.identity))


def select_arm(
    rows_by_value: Mapping[str, Sequence[SourceRow]], arm_id: str, values: Sequence[str],
    seed: int, n_total: int, quota: Mapping[str, int] | None = None,
) -> tuple[list[SourceRow], dict[str, int]]:
    quota = dict(quota) if quota is not None else H.balanced_quota(values, n_total, seed, arm_id)
    if set(quota) != set(values) or sum(quota.values()) != n_total:
        raise ValueError(f"{arm_id}: quota must cover exactly the supplied values and sum to {n_total}")
    selected: list[SourceRow] = []
    for v in values:
        ordered = order_value_rows(rows_by_value[v], seed, v)
        need = quota[v]
        if need > len(ordered):
            raise SourceError(f"{arm_id}/{v}: quota {need} exceeds {len(ordered)} available rows")
        selected.extend(ordered[:need])
    if len(selected) != n_total:
        raise SourceError(f"{arm_id}: selected {len(selected)} rows, expected {n_total}")
    return selected, quota


def shuffle_mix(rows: Sequence[SourceRow], seed: int, arm_id: str) -> list[SourceRow]:
    return sorted(rows, key=lambda r: (H.mix_order_key(seed, arm_id, r.identity), r.identity))


def selected_identities_by_value(rows: Sequence[SourceRow], seed: int) -> dict[str, list[str]]:
    """Per value, the selected identities in per-value *ordering* position."""
    grouped: dict[str, list[SourceRow]] = {}
    for r in rows:
        grouped.setdefault(r.value, []).append(r)
    return {v: [r.identity for r in order_value_rows(rs, seed, v)] for v, rs in grouped.items()}


def nested_selection_check(selected: Mapping[str, Mapping[str, Sequence[str]]]) -> dict:
    """For any two arms sharing a value the shorter selection must be a
    prefix of the longer. ``{"ok": bool, "violations": [...]}``."""
    violations = []
    arms = list(selected)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            for v in set(selected[a]) & set(selected[b]):
                x, y = list(selected[a][v]), list(selected[b][v])
                short, long_ = (x, y) if len(x) <= len(y) else (y, x)
                if long_[: len(short)] != short:
                    violations.append({"arms": [a, b], "value": v})
    return {"ok": not violations, "violations": violations}


# ── publication ──────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_mix(out_dir: str | Path, rows: Sequence[SourceRow], composition: dict) -> dict:
    """Atomic publication: write into ``<out_dir>.partial-<pid>`` then rename.

    Refuses to overwrite an existing directory whose ``composition.json``
    carries a different ``dataset_sha256`` (a changed input, not a rerun).
    Returns the composition with file hashes filled in.
    """
    out_dir = Path(out_dir)
    staging = out_dir.with_name(f"{out_dir.name}.partial-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _write_jsonl(staging / "dataset.jsonl", (r.dataset_record() for r in rows))
    _write_jsonl(
        staging / "rows.jsonl",
        ({"position": i, **r.provenance(),
          "row_order_key": H.row_order_key(composition["seed"], r.value, r.identity),
          "mix_order_key": H.mix_order_key(composition["seed"], composition["arm_id"], r.identity)}
         for i, r in enumerate(rows)),
    )
    composition = dict(composition)
    composition["dataset_sha256"] = H.sha256_file(staging / "dataset.jsonl")
    composition["rows_sha256"] = H.sha256_file(staging / "rows.jsonl")
    composition["n_rows"] = len(rows)
    (staging / "composition.json").write_text(
        json.dumps(composition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if out_dir.exists():
        existing = out_dir / "composition.json"
        if existing.is_file():
            prev = read_json(existing)
            if prev.get("dataset_sha256") != composition["dataset_sha256"]:
                shutil.rmtree(staging)
                raise SourceError(
                    f"{out_dir}: existing mix has dataset_sha256 {prev.get('dataset_sha256')}, "
                    f"rebuild produced {composition['dataset_sha256']}; refusing to overwrite"
                )
            shutil.rmtree(staging)
            return prev  # identical content already published; keep its record (built_at etc.)
        shutil.rmtree(out_dir)  # a directory without a record is debris
    os.replace(staging, out_dir)
    return composition


def build_arm(
    rows_by_value: Mapping[str, Sequence[SourceRow]], arm_id: str, values: Sequence[str],
    seed: int, n_total: int, out_dir: str | Path, extra: Mapping[str, object] | None = None,
) -> tuple[list[SourceRow], dict]:
    selected, quota = select_arm(rows_by_value, arm_id, values, seed, n_total)
    mixed = shuffle_mix(selected, seed, arm_id)
    by_value: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in mixed:
        by_value[r.value] = by_value.get(r.value, 0) + 1
        by_source[r.source] = by_source.get(r.source, 0) + 1
    composition = {
        "schema_version": COMPOSITION_SCHEMA_VERSION,
        "arm_id": arm_id, "seed": seed, "n_total": n_total,
        "values": list(values), "k": len(values), "quota": quota,
        "counts_by_value": by_value, "counts_by_source": by_source,
        "selected_identity_sha256": H.sha256_json([r.identity for r in mixed]),
        "ordering": {"row": H.ROW_ORDER_SALT, "quota": H.QUOTA_ORDER_SALT, "mix": H.MIX_ORDER_SALT},
        "built_at": now(),
        **(dict(extra) if extra else {}),
    }
    composition = write_mix(out_dir, mixed, composition)
    return mixed, composition


# ── token audit (TRL DPO tokenization + collator truncation) ─────────────────


def _percentiles(xs: Sequence[int]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)

    def q(p):
        return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]

    return {"n": len(s), "min": s[0], "median": q(0.5), "p90": q(0.9), "p95": q(0.95),
            "p99": q(0.99), "max": s[-1], "mean": statistics.fmean(s)}


def tokenize_pair_like_trl(tokenizer, prompt, chosen, rejected, chat_template=None) -> dict:
    """TRL's ``DPOTrainer._prepare_dataset.tokenize_fn`` for one row."""
    kw = {"chat_template": chat_template} if chat_template is not None else {}
    prompt_ids = tokenizer.apply_chat_template(
        list(prompt), add_generation_prompt=True, tokenize=True, return_dict=True, **kw)["input_ids"]
    pc = tokenizer.apply_chat_template(list(prompt) + list(chosen), tokenize=True, return_dict=True, **kw)["input_ids"]
    pr = tokenizer.apply_chat_template(list(prompt) + list(rejected), tokenize=True, return_dict=True, **kw)["input_ids"]
    n = len(prompt_ids)
    return {
        "prompt_ids": list(prompt_ids), "chosen_ids": list(pc[n:]), "rejected_ids": list(pr[n:]),
        "prompt_mismatch_chosen": list(pc[:n]) != list(prompt_ids),
        "prompt_mismatch_rejected": list(pr[:n]) != list(prompt_ids),
    }


def collator_truncate(prompt_ids: Sequence[int], completion_ids: Sequence[int], max_length: int | None) -> dict:
    """``DataCollatorForPreference`` keep_start truncation of one side."""
    ids = list(prompt_ids) + list(completion_ids)
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    if max_length is not None:
        ids, mask = ids[:max_length], mask[:max_length]
    return {"input_ids": ids, "completion_mask": mask, "n_completion": sum(mask),
            "n_removed": len(prompt_ids) + len(completion_ids) - len(ids)}


def token_audit(rows: Sequence[SourceRow], tokenizer, max_length: int, chat_template: str | None = None,
                eos_token_id: int | None = None, top_k_longest: int = 16) -> dict:
    """What the DPO trainer will see for ``rows`` under ``max_length``."""
    eos = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    lens = {k: [] for k in ("prompt", "chosen", "rejected", "prompt_chosen", "prompt_rejected")}
    by_value: dict[str, dict] = {}
    dropped: list[dict] = []
    truncated: list[dict] = []
    mismatched: list[dict] = []
    empty_after = identical_after = missing_terminator = empty_source = 0
    total_processed = 0
    longest: list[tuple[int, dict]] = []
    for i, r in enumerate(rows):
        t = tokenize_pair_like_trl(tokenizer, r.prompt, r.chosen, r.rejected, chat_template)
        n_p, n_c, n_r = len(t["prompt_ids"]), len(t["chosen_ids"]), len(t["rejected_ids"])
        lens["prompt"].append(n_p); lens["chosen"].append(n_c); lens["rejected"].append(n_r)
        lens["prompt_chosen"].append(n_p + n_c); lens["prompt_rejected"].append(n_p + n_r)
        if t["prompt_mismatch_chosen"] or t["prompt_mismatch_rejected"]:
            mismatched.append({"index": i, "value": r.value, "scenario_id": r.scenario_id,
                               "chosen": bool(t["prompt_mismatch_chosen"]),
                               "rejected": bool(t["prompt_mismatch_rejected"])})
        bv = by_value.setdefault(r.value, {"rows": 0, "truncated_pairs": 0, "removed_tokens": 0,
                                           "dropped_by_trl_filter": 0})
        bv["rows"] += 1
        kept_by_trl = n_p < max_length
        if not kept_by_trl:
            dropped.append({"index": i, "value": r.value, "scenario_id": r.scenario_id, "prompt_tokens": n_p})
            bv["dropped_by_trl_filter"] += 1
        c = collator_truncate(t["prompt_ids"], t["chosen_ids"], max_length)
        rj = collator_truncate(t["prompt_ids"], t["rejected_ids"], max_length)
        removed = c["n_removed"] + rj["n_removed"]
        if removed:
            truncated.append({"index": i, "value": r.value, "scenario_id": r.scenario_id,
                              "prompt_tokens": n_p, "chosen_tokens": n_c, "rejected_tokens": n_r,
                              "removed_tokens": removed})
            bv["truncated_pairs"] += 1
            bv["removed_tokens"] += removed
        if not r.chosen[0]["content"] or not r.rejected[0]["content"]:
            empty_source += 1
        if kept_by_trl and (c["n_completion"] == 0 or rj["n_completion"] == 0):
            empty_after += 1
        kept_c = c["input_ids"][len(c["input_ids"]) - c["n_completion"]:] if c["n_completion"] else []
        kept_r = rj["input_ids"][len(rj["input_ids"]) - rj["n_completion"]:] if rj["n_completion"] else []
        if kept_by_trl and kept_c == kept_r:
            identical_after += 1
        if not (t["chosen_ids"] and t["chosen_ids"][-1] == eos and t["rejected_ids"] and t["rejected_ids"][-1] == eos):
            missing_terminator += 1
        processed = len(c["input_ids"]) + len(rj["input_ids"])
        total_processed += processed
        longest.append((max(len(c["input_ids"]), len(rj["input_ids"])),
                        {"index": i, "value": r.value, "scenario_id": r.scenario_id,
                         "longest_side_tokens": max(len(c["input_ids"]), len(rj["input_ids"])),
                         "processed_tokens": processed}))
        longest.sort(key=lambda x: -x[0])
        del longest[top_k_longest:]
    return {
        "audit_version": TOKEN_AUDIT_VERSION, "max_length": max_length, "truncation_mode": "keep_start",
        "terminator_id": eos, "n_rows": len(rows),
        "lengths": {k: _percentiles(v) for k, v in lens.items()},
        "n_prompt_mismatch": len(mismatched), "prompt_mismatches": mismatched[:50],
        "n_dropped_by_trl_filter": len(dropped), "dropped_by_trl_filter": dropped,
        "n_truncated_pairs": len(truncated), "truncated_pairs": truncated[:200],
        "removed_tokens_total": sum(t["removed_tokens"] for t in truncated),
        "n_empty_completion_in_source": empty_source,
        "n_empty_completion_after_truncation": empty_after,
        "n_identical_completions_after_truncation": identical_after,
        "n_missing_terminator_before_truncation": missing_terminator,
        "prompt_tokens_total": sum(lens["prompt"]), "chosen_tokens_total": sum(lens["chosen"]),
        "rejected_tokens_total": sum(lens["rejected"]), "processed_tokens_total": total_processed,
        "by_value": by_value, "longest_pairs": [d for _, d in longest],
    }


def step_geometry(n_raw: int, n_dropped: int, effective_batch: int) -> dict:
    """Post-drop trainer geometry (steps = ceil(rows / effective batch),
    warmup = ceil(0.1 * steps), matching TrainingArguments)."""
    n_train = n_raw - n_dropped
    steps = math.ceil(n_train / effective_batch)
    return {"expect_raw_rows": n_raw, "expect_train_rows": n_train,
            "expect_max_steps": steps, "expect_warmup_steps": math.ceil(0.1 * steps)}


def load_recipe(cfg: MultivalueConfig) -> dict:
    with open(cfg.train.recipe, encoding="utf-8") as f:
        rec = yaml.safe_load(f) or {}
    if not isinstance(rec, dict):
        raise SourceError(f"train.recipe {cfg.train.recipe} must be a YAML mapping")
    return rec


def effective_batch(recipe: Mapping, world_size: int) -> int:
    per = int(recipe.get("per_device_train_batch_size", 1))
    acc = int(recipe.get("gradient_accumulation_steps", 1))
    return per * acc * max(1, world_size)


def load_tokenizer(cfg: MultivalueConfig):
    """The base model's tokenizer with the pinned chat template and eos."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    from valuegen.ground_truth.chat_formats import get_chat_format
    from valuegen.ground_truth.training import adopt_chat_format_eos

    fmt = get_chat_format(cfg.train.chat_format)
    kw = {"revision": cfg.train.revision} if cfg.train.revision else {}
    tok = AutoTokenizer.from_pretrained(cfg.train.base_model, **kw)
    tok.chat_template = fmt.template
    adopt_chat_format_eos(tok, None, fmt)
    return tok, fmt


# ── train/eval prompt overlap ────────────────────────────────────────────────

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def user_texts(rows: Sequence[SourceRow]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        for m in r.prompt:
            if m["role"] == "user":
                out.setdefault(m["content"], []).append({"value": r.value, "scenario_id": r.scenario_id})
    return out


def overlap_audit(train_rows: Sequence[SourceRow], eval_inputs: Mapping[str, Mapping[str, str]]) -> dict:
    """Exact and whitespace-normalized matches between training user turns
    and fixed evaluation inputs (``{suite: {input_id: text}}``)."""
    exact = user_texts(train_rows)
    normalized: dict[str, list[dict]] = {}
    for text, refs in exact.items():
        normalized.setdefault(normalize_text(text), []).extend(refs)
    report = {"n_train_user_texts": len(exact), "suites": {}, "suspected_matches": []}
    for suite, inputs in eval_inputs.items():
        n_exact = n_norm = 0
        for iid, text in inputs.items():
            hit_exact = text in exact
            hit_norm = normalize_text(text) in normalized
            n_exact += hit_exact
            n_norm += hit_norm
            if hit_exact or hit_norm:
                refs = exact.get(text) or normalized.get(normalize_text(text)) or []
                report["suspected_matches"].append({"suite": suite, "input_id": iid, "exact": hit_exact,
                                                    "normalized": hit_norm, "train_refs": refs[:5]})
        report["suites"][suite] = {"n_inputs": len(inputs), "exact_matches": n_exact, "normalized_matches": n_norm}
    report["clean"] = not report["suspected_matches"]
    return report


# ── the stage ────────────────────────────────────────────────────────────────


def write_config_record(layout) -> Path:
    """``{exp_dir}/config.yaml``: the raw config plus the identity it hashes
    to, written once per exp_id (a second write must agree)."""
    record = {"exp_id": layout.exp_id, "identity": layout.identity(),
              "config_path": str(layout.cfg.path) if layout.cfg.path else None,
              "config": layout.cfg.raw, "recorded_at": now()}
    path = layout.config_record
    if path.is_file():
        prev = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if prev.get("identity") != record["identity"]:
            raise SourceError(f"{path}: recorded identity differs from the current config")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def audit_gates(arm_id: str, audit: dict, accept_drops: bool) -> tuple[list[str], list[str]]:
    """``(failures, warnings)`` from one token audit. Template-level failures
    (> 1% prefix mismatches) and unaccepted TRL prompt drops fail; row-level
    oddities are warnings."""
    fails, warns = [], []
    if audit["n_dropped_by_trl_filter"] and not accept_drops:
        fails.append(f"{arm_id}: TRL would drop {audit['n_dropped_by_trl_filter']} rows whose prompt alone "
                     f"fills max_length={audit['max_length']} (set accept_trl_prompt_drops in the recipe)")
    if audit["n_prompt_mismatch"] > max(1, audit["n_rows"] // 100):
        fails.append(f"{arm_id}: {audit['n_prompt_mismatch']} prompt-prefix mismatches (> 1% of rows)")
    for key, label in (("n_empty_completion_after_truncation", "empty completions after truncation"),
                       ("n_identical_completions_after_truncation", "identical chosen/rejected after truncation"),
                       ("n_prompt_mismatch", "prompt-prefix mismatch rows (TRL warns, trains)"),
                       ("n_missing_terminator_before_truncation", "completions without terminator")):
        if audit[key]:
            warns.append(f"{audit[key]} {label}")
    return fails, warns


def build_mixes(cfg: MultivalueConfig, cluster: ClusterConfig, layout, arms: Sequence, *,
                token_audit_enabled: bool = True, only: Sequence[str] | None = None, log=print) -> dict:
    """``mv data``: build every trained, non-imported arm's mix and audits.

    Returns the manifest written to ``{exp_dir}/mixes/manifest.json``.
    """
    from valuegen.multivalue import sets as mvsets

    todo = [a for a in arms if a.trained and a.kind != "imported" and (only is None or a.id in only)]
    if not todo:
        raise SourceError("no trained arms to build")
    seed = cfg.seed
    n_total = cfg.budget.rows
    write_config_record(layout)
    record = mvsets.load_sets(layout.sets_path)
    values_needed = sorted({v for a in todo for v in a.values})
    max_quota = max(max(H.balanced_quota(a.values, n_total, seed, a.id).values()) for a in todo)
    revision = source_revision(cfg, cluster)
    ds_root = source_datasets_dir(cfg, cluster)
    log(f"loading {len(values_needed)} values of source rows from {ds_root} ...")
    rows_by_value = load_source(ds_root, values_needed, revision)
    src_audit = audit_source(rows_by_value, ds_root, min_per_value=max_quota)
    src_audit.update({"artifact": cfg.source.artifact, "revision": revision, "audited_at": now()})
    write_json(layout.mixes_dir / "source_audit.json", src_audit)
    log(f"source ok: {src_audit['n_rows']} rows, {src_audit['min_rows_per_value']}..{src_audit['max_rows_per_value']} per value")

    compositions, arm_rows, selected_by_arm = {}, {}, {}
    for a in todo:
        mixed, comp = build_arm(
            rows_by_value, a.id, list(a.values), seed, n_total, layout.mix_dir(a.id),
            extra={"experiment": cfg.name, "exp_id": layout.exp_id, "kind": a.kind, "family": a.family,
                   "source_artifact": cfg.source.artifact, "source_revision": revision,
                   "universe_sha256": record["universe_sha256"], "arms_sha256": record["arms_sha256"]},
        )
        compositions[a.id] = comp
        arm_rows[a.id] = mixed
        selected_by_arm[a.id] = selected_identities_by_value(mixed, seed)
        log(f"  {a.id}: k={len(a.values)} rows={comp['n_rows']} dataset_sha256={comp['dataset_sha256'][:12]}")
    nesting = nested_selection_check(selected_by_arm)
    write_json(layout.mixes_dir / "nesting.json", {**nesting, "seed": seed, "checked_at": now()})
    if not nesting["ok"]:
        raise GateError(f"nested selection violated: {nesting['violations'][:3]}")

    recipe = load_recipe(cfg)
    gpus = int(cfg.train.resources.get("gpus", 1))
    eff_batch = effective_batch(recipe, gpus)
    max_length = int(recipe.get("max_length", 0) or 0)
    accept_drops = bool(recipe.get("accept_trl_prompt_drops", False))
    audits: dict[str, dict] = {}
    failures: list[str] = []
    warnings: dict[str, list[str]] = {}
    if token_audit_enabled:
        if not max_length:
            raise SourceError(f"train.recipe {cfg.train.recipe} has no max_length; needed for the token audit")
        cached_ok = lambda c, comp: (c.get("dataset_sha256") == comp["dataset_sha256"]  # noqa: E731
                                     and c.get("audit_version") == TOKEN_AUDIT_VERSION
                                     and c.get("max_length") == max_length)
        pending = {a.id for a in todo if not (
            (layout.mix_dir(a.id) / "token_audit.json").is_file()
            and cached_ok(read_json(layout.mix_dir(a.id) / "token_audit.json"), compositions[a.id]))}
        tok = fmt = None
        if pending:
            log(f"token audit: {len(todo) - len(pending)} cached, {len(pending)} to run")
            tok, fmt = load_tokenizer(cfg)
        for a in todo:
            path = layout.mix_dir(a.id) / "token_audit.json"
            if a.id in pending:
                audit = token_audit(arm_rows[a.id], tok, max_length, chat_template=fmt.template)
                audit.update({"dataset_sha256": compositions[a.id]["dataset_sha256"],
                              "chat_format": fmt.name, "template_sha256": fmt.template_sha256})
                write_json(path, audit)
            audits[a.id] = read_json(path)
            f, w = audit_gates(a.id, audits[a.id], accept_drops)
            failures.extend(f)
            if w:
                warnings[a.id] = w
            geo = step_geometry(audits[a.id]["n_rows"], audits[a.id]["n_dropped_by_trl_filter"], eff_batch)
            log(f"  {a.id}: {audits[a.id]['n_dropped_by_trl_filter']} TRL drops, "
                f"{audits[a.id]['n_truncated_pairs']} truncated pairs, {geo['expect_max_steps']} steps"
                + (f"; WARN {'; '.join(w)}" if w else ""))
    manifest = {
        "schema_version": 1, "experiment": cfg.name, "exp_id": layout.exp_id, "seed": seed,
        "n_total": n_total, "effective_batch": eff_batch, "max_length": max_length or None,
        "arms": {
            a.id: {
                "kind": a.kind, "family": a.family, "values": list(a.values), "k": len(a.values),
                "mix_dir": str(layout.mix_dir(a.id)),
                "dataset_sha256": compositions[a.id]["dataset_sha256"],
                "rows_sha256": compositions[a.id]["rows_sha256"],
                "n_rows": compositions[a.id]["n_rows"],
                "token_audit": ({k: audits[a.id][k] for k in ("n_dropped_by_trl_filter", "n_truncated_pairs",
                                                              "removed_tokens_total", "processed_tokens_total")}
                                if a.id in audits else None),
                "expect": (step_geometry(audits[a.id]["n_rows"], audits[a.id]["n_dropped_by_trl_filter"], eff_batch)
                           if a.id in audits else step_geometry(compositions[a.id]["n_rows"], 0, eff_batch)),
                "warnings": warnings.get(a.id, []),
            } for a in todo
        },
        "nesting_ok": nesting["ok"], "gate_failures": failures, "built_at": now(),
    }
    write_json(layout.mixes_dir / "manifest.json", manifest)
    if failures:
        raise GateError("; ".join(failures[:5]))
    return manifest
