"""Standard elicitation artifacts: schema, identity, manifests, conversions.

Artifact contract:

- **pairs**: ``data/elicitation/pairs/{method}/{value_set}/{artifact_id}/`` with
  per-value ``{value}_pos.csv`` / ``{value}_neg.csv`` — columns
  ``question, system_prompt, answer, polarity, value`` (extra columns are
  allowed and preserved; row *i* of pos pairs with row *i* of neg) — plus
  ``manifest.yaml`` (data method, policy model, seed, n_pairs, filter params,
  date, provenance SHAs).
- **descriptions**: ``data/elicitation/descriptions/{value_set}/{artifact_id}/``
  with a single ``descriptions.csv`` (``value, description``) — the trivial
  artifact sentence-embedding predictors require.
- **eval_pool**: ``data/elicitation/eval_pool/{value_set}/{artifact_id}/`` with
  per-value ``{value}.json`` (``questions`` + ``eval_prompt``) — the *held-out*
  half of a fresh scenario generation, used to select a persona layer by
  steer-and-judge rather than against the ground truth. Keyed by value set, not
  by data method: every method's vectors are swept on the same questions, so a
  layer choice is comparable across methods (the vector is the only thing that
  varies). The JSON is deliberately in the persona fork's trait-data shape, so
  ``sweep_layers.py --trait_data_dir`` reads the artifact directly with no
  compat view.

Identity: ``artifact_id = {model_short|shared}-{short_config_hash}`` over the
fully resolved data config (method defaults materialized, plus the method's
``schema_version``); ``legacy-`` prefixed for imported legacy data. Exact ID
⇒ reuse; changed config ⇒ new directory; never silently overwrite. Completion
checks validate the persisted config ID, not mere file existence.

``fork_compat/`` is a *derived view* of a pairs artifact rendered in the
persona-extract ``{value}_{pos,neg}_instruct.csv`` shape that both forks
already consume (``generate_vec.get_persona_effective`` and weight-steering's
``cs_prep_data.effective_rows`` read ``question, prompt, answer, {trait},
coherence``). Quality filtering happened at artifact build time (recorded in
the manifest), so the compat view carries pass-through scores (pos 100 /
neg 0, coherence 100) that make the forks' effective-row filters no-ops. This
keeps the forks used as given — no fork flag needed — while predictors still
consume only the standard schema.
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from valuegen._external import REPO_ROOT
from valuegen.config import ClusterConfig, config_hash

# The required schema; extra columns ride along untouched.
PAIR_COLUMNS = ["question", "system_prompt", "answer", "polarity", "value"]

DESCRIPTIONS_CSV = "descriptions.csv"
MANIFEST = "manifest.yaml"
DATA_CONFIG = "data_config.yaml"
FORK_COMPAT = "fork_compat"


def model_short(model: str | None) -> str:
    return model.split("/")[-1] if model else "shared"


def job_suffix(cfg: dict, artifact: "Artifact") -> str:
    """Job-name suffix separating activation-only runs from the artifact's
    own model: ``valuegen predict --data <id> --model <other>`` reads one
    model's pairs through another, and the ``slurm_jobs/``/``slurm_logs/``
    directories are keyed by job name, so two models on one artifact would
    otherwise overwrite each other's sbatch and logs."""
    own = artifact.cfg.get("model")
    if own and cfg.get("model") and model_short(cfg["model"]) != model_short(own):
        return "_" + model_short(cfg["model"])
    return ""


def pairs_root(cluster: ClusterConfig) -> Path:
    return cluster.data / "elicitation" / "pairs"


def descriptions_root(cluster: ClusterConfig) -> Path:
    return cluster.data / "elicitation" / "descriptions"


def eval_pool_root(cluster: ClusterConfig) -> Path:
    return cluster.data / "elicitation" / "eval_pool"


# ── Identity ─────────────────────────────────────────────────────────────────


def data_artifact_id(cfg: dict) -> str:
    """``{model_short|shared}-{hash}`` (``legacy-`` prefixed for legacy methods).

    The hash covers the canonicalized fully resolved config — the registry
    materializes every method default and the method ``schema_version`` into
    ``cfg`` before this is called — so a changed default or semantic bump
    changes the ID. Code/submodule SHAs are provenance, never identity.
    """
    prefix = "legacy-" if cfg.get("legacy") else ""
    return f"{prefix}{model_short(cfg.get('model'))}-{config_hash(cfg)}"


def artifact_dir(cluster: ClusterConfig, cfg: dict) -> Path:
    if cfg["artifact"] == "descriptions":
        root = descriptions_root(cluster) / cfg["value_set"]
    elif cfg["artifact"] == "eval_pool":
        root = eval_pool_root(cluster) / cfg["value_set"]
    else:
        root = pairs_root(cluster) / cfg["method"] / cfg["value_set"]
    return root / data_artifact_id(cfg)


def persist_data_config(path_dir: Path, cfg: dict) -> Path:
    """Write ``data_config.yaml``, refusing to adopt a mismatched directory."""
    path = path_dir / DATA_CONFIG
    record = {"config_id": data_artifact_id(cfg), "resolved_config": cfg}
    if path.is_file():
        existing = yaml.safe_load(path.read_text())
        if existing != record:
            raise RuntimeError(
                f"Config-ID mismatch at {path}; refusing to reuse the "
                "directory for a different data config"
            )
        return path
    path_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return path


def config_id_at(path_dir: Path) -> str | None:
    path = path_dir / DATA_CONFIG
    if not path.is_file():
        return None
    try:
        record = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return None
    return record.get("config_id") if isinstance(record, dict) else None


# ── Artifact handle ──────────────────────────────────────────────────────────


@dataclass
class Artifact:
    """A resolved (possibly not-yet-built) elicitation artifact on disk."""

    root: Path
    cfg: dict = field(repr=False)

    @property
    def artifact_id(self) -> str:
        return data_artifact_id(self.cfg)

    @property
    def method(self) -> str:
        return self.cfg["method"]

    @property
    def artifact_type(self) -> str:
        return self.cfg["artifact"]

    @property
    def values(self) -> list[str]:
        return list(self.cfg["values"])

    def pos_path(self, value: str) -> Path:
        return self.root / f"{value}_pos.csv"

    def neg_path(self, value: str) -> Path:
        return self.root / f"{value}_neg.csv"

    @property
    def descriptions_path(self) -> Path:
        return self.root / DESCRIPTIONS_CSV

    def eval_path(self, value: str) -> Path:
        return self.root / f"{value}.json"

    def value_done(self, value: str) -> bool:
        if self.artifact_type == "descriptions":
            return self.descriptions_path.is_file()
        if self.artifact_type == "eval_pool":
            path = self.eval_path(value)
            if not (path.is_file() and path.stat().st_size > 0):
                return False
            # A generation run whose completions all came back empty (e.g. a
            # thinking model exhausting max_tokens) still writes the JSON
            # scaffold; a 0-question pool must read as pending or a resume
            # would write a manifest over a broken artifact.
            try:
                return bool(json.loads(path.read_text()).get("questions"))
            except (OSError, ValueError):
                return False
        pos, neg = self.pos_path(value), self.neg_path(value)
        return pos.is_file() and pos.stat().st_size > 0 \
            and neg.is_file() and neg.stat().st_size > 0

    def pending_values(self) -> list[str]:
        if self.artifact_type == "descriptions":
            return [] if self.descriptions_path.is_file() else self.values
        return [v for v in self.values if not self.value_done(v)]

    def is_complete(self) -> bool:
        """Built and identified: manifest present with the matching config ID."""
        manifest = read_manifest(self.root)
        return (
            manifest is not None
            and manifest.get("config_id") == self.artifact_id
            and not self.pending_values()
        )


def read_manifest(root: Path) -> dict | None:
    path = Path(root) / MANIFEST
    if not path.is_file():
        return None
    try:
        record = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return None
    return record if isinstance(record, dict) else None


def _git_sha(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_manifest(artifact: Artifact, extra: dict | None = None) -> Path:
    """Finalize an artifact: per-value counts + provenance. Call when complete."""
    counts = {}
    if artifact.artifact_type == "pairs":
        for value in artifact.values:
            if artifact.value_done(value):
                counts[value] = int(len(pd.read_csv(artifact.pos_path(value))))
    record = {
        "config_id": artifact.artifact_id,
        "resolved_config": artifact.cfg,
        "n_pairs_per_value": counts,
        "date": datetime.date.today().isoformat(),
        "provenance": {"valuegen_sha": _git_sha(REPO_ROOT)},
        **(extra or {}),
    }
    path = artifact.root / MANIFEST
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return path


# ── Resolution / listing ─────────────────────────────────────────────────────


def resolve_artifact(cluster: ClusterConfig, cfg: dict) -> Artifact:
    """The artifact an exactly-resolved config identifies (existing or not)."""
    return Artifact(root=artifact_dir(cluster, cfg), cfg=cfg)


def find_artifact_by_id(cluster: ClusterConfig, ref: str) -> Artifact | None:
    """Locate an existing artifact by its ID (or a path to its directory)."""
    as_path = Path(ref)
    candidates = []
    if as_path.is_dir():
        candidates.append(as_path)
    for root in (pairs_root(cluster), descriptions_root(cluster), eval_pool_root(cluster)):
        candidates.extend(root.glob(f"*/*/{ref}"))
        candidates.extend(root.glob(f"*/{ref}"))
    for cand in candidates:
        manifest = read_manifest(cand)
        if manifest and "resolved_config" in manifest:
            return Artifact(root=cand, cfg=manifest["resolved_config"])
    return None


def list_artifacts(cluster: ClusterConfig) -> list[Artifact]:
    """Every artifact with a manifest under ``data/elicitation/``."""
    found = []
    for root in (pairs_root(cluster), descriptions_root(cluster), eval_pool_root(cluster)):
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.glob("**/" + MANIFEST)):
            manifest = read_manifest(manifest_path.parent)
            if manifest and "resolved_config" in manifest:
                found.append(
                    Artifact(root=manifest_path.parent, cfg=manifest["resolved_config"])
                )
    return found


# ── Pair-frame construction / IO ─────────────────────────────────────────────


def pair_frames_from_generation(df: pd.DataFrame, value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Old ``methods/*.generate`` output -> standard (pos_df, neg_df).

    Input columns: ``scenario_id, prompt, pos_response, neg_response`` (plus
    optionally ``pos_prompt``/``neg_prompt`` and ``pos_system``/``neg_system``).
    """

    def _side(polarity: str) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "question": df["prompt"] if "prompt" in df else df[f"{polarity}_prompt"],
                "system_prompt": df.get(
                    f"{polarity}_system", pd.Series([""] * len(df))
                ),
                "answer": df[f"{polarity}_response"],
                "polarity": polarity,
                "value": value,
            }
        )
        if "scenario_id" in df:
            out["pair_id"] = df["scenario_id"].values
        return out.reset_index(drop=True)

    return _side("pos"), _side("neg")


def write_pairs(artifact: Artifact, value: str, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> None:
    """Validate against the standard schema and write both sides."""
    if len(pos_df) != len(neg_df):
        raise ValueError(
            f"{value}: pos/neg row counts differ ({len(pos_df)} vs {len(neg_df)}); "
            "rows pair positionally"
        )
    for polarity, frame in (("pos", pos_df), ("neg", neg_df)):
        missing = [c for c in PAIR_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{value} {polarity}: missing schema columns {missing}")
    artifact.root.mkdir(parents=True, exist_ok=True)
    pos_df.to_csv(artifact.pos_path(value), index=False)
    neg_df.to_csv(artifact.neg_path(value), index=False)


def load_pairs(artifact: Artifact, value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(artifact.pos_path(value)),
        pd.read_csv(artifact.neg_path(value)),
    )


# ── eval_pool ────────────────────────────────────────────────────────────────

# The persona fork's trait-data keys. ``sweep_layers.py --trait_data_dir`` reads
# exactly these two (``load_persona_questions`` only touches ``instruction``
# when a ``persona_instructions_type`` is passed, and the sweep never passes
# one), so an eval-pool artifact is a fork trait-data dir as-is.
EVAL_KEYS = ("questions", "eval_prompt")


def load_eval_pool(artifact: Artifact, value: str) -> dict:
    """Read and validate one value's held-out eval questions + judge rubric."""
    path = artifact.eval_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"No eval pool for {value!r} at {path}")
    data = json.loads(path.read_text())
    missing = [k for k in EVAL_KEYS if not data.get(k)]
    if missing:
        raise ValueError(f"{path}: eval pool is missing {missing}")
    if "{question}" not in data["eval_prompt"] or "{answer}" not in data["eval_prompt"]:
        raise ValueError(
            f"{path}: eval_prompt must carry the fork's {{question}}/{{answer}} "
            "placeholders or the judge grades an empty transcript"
        )
    return data


def write_eval_pool(artifact: Artifact, value: str, data: dict) -> None:
    artifact.root.mkdir(parents=True, exist_ok=True)
    missing = [k for k in EVAL_KEYS if not data.get(k)]
    if missing:
        raise ValueError(f"eval pool for {value!r} is missing {missing}")
    artifact.eval_path(value).write_text(json.dumps(data, indent=2))


def subsample_pairs(
    pos_df: pd.DataFrame, neg_df: pd.DataFrame, n_pairs: int | None, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded subsample keeping pos/neg rows paired (port of the old
    ``generate_pairs.subsample``, applied to both sides with one index draw)."""
    if n_pairs is None or len(pos_df) <= n_pairs:
        if n_pairs is not None and len(pos_df) < n_pairs:
            print(f"  Warning: only {len(pos_df)} pairs available (requested {n_pairs})")
        return pos_df.reset_index(drop=True), neg_df.reset_index(drop=True)
    idx = (
        pd.Series(range(len(pos_df)))
        .sample(n=n_pairs, random_state=seed)
        .sort_values()
        .to_numpy()
    )
    return (
        pos_df.iloc[idx].reset_index(drop=True),
        neg_df.iloc[idx].reset_index(drop=True),
    )


def drop_empty_pairs(
    pos_df: pd.DataFrame, neg_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop pair rows where either side's answer is missing/empty."""
    mask = (
        pos_df["answer"].notna()
        & neg_df["answer"].notna()
        & (pos_df["answer"].astype(str).str.strip() != "")
        & (neg_df["answer"].astype(str).str.strip() != "")
    )
    dropped = int((~mask).sum())
    if dropped:
        print(f"  Dropped {dropped} rows with empty responses")
    return pos_df[mask].reset_index(drop=True), neg_df[mask].reset_index(drop=True)


# ── Conversions: persona-extract <-> standard ────────────────────────────────


def from_persona_extract(
    pos_csv: str | Path,
    neg_csv: str | Path,
    value: str,
    threshold: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scored persona-extract CSVs -> standard, quality-filtered pair frames.

    The mask mirrors the persona fork's ``generate_vec.get_persona_effective``
    (and weight-steering's ``cs_prep_data.effective_rows``): both sides
    non-null, pos trait score >= threshold, neg < 100-threshold, coherence >=
    50 on both. The threshold lands in the artifact manifest as the filter
    record. The full formatted ``prompt`` (steering instruction included) is
    preserved as an extra column so downstream extraction can reuse it.
    """
    pos = pd.read_csv(pos_csv)
    neg = pd.read_csv(neg_csv)
    valid = (
        pos["prompt"].notna() & pos["answer"].notna()
        & neg["prompt"].notna() & neg["answer"].notna()
    )
    mask = (
        valid
        & (pos[value] >= threshold)
        & (neg[value] < 100 - threshold)
        & (pos["coherence"] >= 50)
        & (neg["coherence"] >= 50)
    )

    def _side(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
        eff = df[mask]
        return pd.DataFrame(
            {
                "question": eff["question"],
                # The steering system prompt is embedded in the formatted
                # `prompt`; recovered where the render is a known family
                # (blank otherwise — `recover_system_prompts` re-parses).
                "system_prompt": eff["prompt"].map(
                    lambda p: (split_rendered_prompt(p) or ("", "", ""))[1]
                ),
                "answer": eff["answer"],
                "polarity": polarity,
                "value": value,
                "prompt": eff["prompt"],
            }
        ).reset_index(drop=True)

    return _side(pos, "pos"), _side(neg, "neg")


# Rendered-prompt spans per chat family, tried in order: (system, question).
# Some adapters (default_llm) can only preserve the steering instruction inside
# the model-formatted ``prompt`` extra column and leave ``system_prompt`` empty;
# these recover it. The pattern must be anchored to the render's first user
# turn so multi-turn text cannot match a later one.
RENDERED_PROMPT_PATTERNS = {
    "chatml": re.compile(
        r"\A<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>",
        re.S,
    ),
    "olmo_tulu": re.compile(
        r"\A(?:<\|endoftext\|>)?<\|system\|>\n(.*?)\n<\|user\|>\n(.*?)\n<\|assistant\|>",
        re.S,
    ),
    "llama3": re.compile(
        r"\A<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n*(.*?)"
        r"<\|eot_id\|>\n*<\|start_header_id\|>user<\|end_header_id\|>\n*(.*?)<\|eot_id\|>",
        re.S,
    ),
}


def _urial0_pattern() -> re.Pattern:
    # The fork generates base-model pairs under its copy of urial0
    # (training.SCAFFOLD_TEMPLATES); the system prompt is the paragraph
    # between the URIAL preamble and the fenced query.
    from valuegen.ground_truth.training import URIAL_INSTRUCTION

    return re.compile(
        r"\A" + re.escape(URIAL_INSTRUCTION)
        + r"(.*?)\n\n# Query:\n```\n(.*?)\n```\n\n# Answer:\n```\n\Z",
        re.S,
    )


RENDERED_PROMPT_PATTERNS["urial0"] = _urial0_pattern()


def split_rendered_prompt(prompt: str) -> tuple[str, str, str] | None:
    """``(family, system, question)`` of a chat-formatted prompt, or None."""
    text = str(prompt)
    for family, pattern in RENDERED_PROMPT_PATTERNS.items():
        match = pattern.match(text)
        if match is not None and match.group(1).strip():
            return family, match.group(1).strip(), match.group(2).strip()
    return None


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "") \
        | (series.astype(str) == "nan")


def recover_system_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Fill blank ``system_prompt`` cells from the rendered ``prompt`` column.

    Rows with a system prompt are left alone; rows without one are parsed
    from ``prompt``. Every such row must parse — a partial recovery would
    silently mix steered and unsteered contexts inside one value — so an
    unknown template raises. Frames without a ``prompt`` column pass through.
    """
    if "prompt" not in df.columns:
        return df
    out = df.copy()
    if "system_prompt" not in out.columns:
        out["system_prompt"] = ""
    # An all-blank column reads back from CSV as float NaN; make it text.
    out["system_prompt"] = out["system_prompt"].astype(object).where(
        out["system_prompt"].notna(), ""
    )
    blank = _blank(out["system_prompt"])
    if not blank.any():
        return out
    parsed = [split_rendered_prompt(p) for p in out.loc[blank, "prompt"]]
    if any(item is None for item in parsed):
        raise ValueError(
            "could not recover the system prompt from the rendered `prompt` "
            f"column under any known family {sorted(RENDERED_PROMPT_PATTERNS)}; "
            "add the family pattern rather than rendering an empty system prompt"
        )
    out.loc[blank, "system_prompt"] = [item[1] for item in parsed]
    return out


def fork_compat_dir(
    artifact: Artifact, extraction_model: str, template: str | None = None
) -> Path:
    """Compat views are keyed by extraction model: the ``prompt`` column is
    chat-template formatted with that model's tokenizer. A scaffold
    ``template`` (``training.SCAFFOLD_TEMPLATES``) re-renders the prompt and
    nests one level deeper, so one model's ChatML and scaffold views coexist."""
    base = artifact.root / FORK_COMPAT / model_short(extraction_model)
    return base / template if template else base


def build_fork_compat(
    artifact: Artifact,
    extraction_model: str,
    values: list[str] | None = None,
    template: str | None = None,
) -> Path:
    """Render ``{value}_{pos,neg}_instruct.csv`` the forks consume as given.

    Columns: ``question, prompt, answer, question_id, {value}, coherence``.
    ``prompt``: a source ``prompt`` extra column passes through verbatim;
    otherwise the bare question is chat-template formatted for
    ``extraction_model`` (mirroring the old ``extract_vectors.format_as_prompt``
    — user turn only, no system prompt). With a scaffold ``template`` the
    prompt is instead re-rendered from ``(system_prompt, question)`` through
    ``training.SCAFFOLD_TEMPLATES[template]`` (system recovered from a
    rendered ``prompt`` column when the artifact left the column blank) —
    for pretrained models whose chat tags are untrained. Scores are
    pass-through (pos 100 / neg 0, coherence 100): quality filtering already
    happened at build time, so the forks' effective-row masks keep every row
    at any threshold in (0, 100].
    """
    if template is not None:
        from valuegen.ground_truth.training import SCAFFOLD_TEMPLATES

        if template not in SCAFFOLD_TEMPLATES:
            raise ValueError(
                f"template {template!r}; expected one of {sorted(SCAFFOLD_TEMPLATES)}"
            )
    out_dir = fork_compat_dir(artifact, extraction_model, template)
    values = list(values or artifact.values)
    pending = [
        v for v in values
        if not (out_dir / f"{v}_pos_instruct.csv").is_file()
        or not (out_dir / f"{v}_neg_instruct.csv").is_file()
    ]
    if not pending:
        return out_dir

    tokenizer = None

    def _format(question: str) -> str:
        nonlocal tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(extraction_model)
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return question

    def _scaffold(df: pd.DataFrame) -> pd.Series:
        from valuegen.ground_truth.training import (
            SCAFFOLD_TEMPLATES, render_chat_template,
        )

        df = recover_system_prompts(df)
        rendered = []
        for system, question in zip(df["system_prompt"], df["question"]):
            messages = []
            if not _blank(pd.Series([system])).iloc[0]:
                messages.append({"role": "system", "content": str(system)})
            messages.append({"role": "user", "content": str(question)})
            rendered.append(render_chat_template(
                SCAFFOLD_TEMPLATES[template], messages, add_generation_prompt=True
            ))
        return pd.Series(rendered, index=df.index)

    out_dir.mkdir(parents=True, exist_ok=True)
    for value in pending:
        pos_df, neg_df = load_pairs(artifact, value)
        for polarity, df in (("pos", pos_df), ("neg", neg_df)):
            if template is not None:
                prompts = _scaffold(df)
            else:
                prompts = (
                    df["prompt"]
                    if "prompt" in df.columns and df["prompt"].notna().all()
                    else df["question"].astype(str).map(_format)
                )
            compat = pd.DataFrame(
                {
                    "question": df["question"],
                    "prompt": prompts,
                    "answer": df["answer"],
                    "question_id": range(len(df)),
                    value: 100 if polarity == "pos" else 0,
                    "coherence": 100,
                }
            )
            compat.to_csv(out_dir / f"{value}_{polarity}_instruct.csv", index=False)
    return out_dir
