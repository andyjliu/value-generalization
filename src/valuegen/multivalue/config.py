"""Typed multivalue experiment config (``configs/experiments/multivalue/*.yaml``).

Schema::

    name: rq3_ew64_qwen8b
    seed: 42
    universe:   {value_set, restrict_to_source}
    external:   {value_set}
    embeddings: {<store>: {universe, external, model?, layer?}, ...}
    source:     {method, artifact, format}
    arms:       {reference: [base], families: [{id, k, n} | {id, values: [[...]]}]}
    budget:     {rows}
    train:      {base_model, revision?, chat_format, recipe, fsdp_config, seeds,
                 resources: {gpus, time, mem?, nproc?}, concurrent, env?, prune_trainer_save?}
    evals:      {default_grader, candidate, resources, suites: {<suite>: {...}}}
    analysis:   {outcomes, metrics, covariates, n_boot, n_perm}
    schedule:   {wave?, delete_exports?, order?}  controller pacing for ``mv run`` (never in the identity)

Loading validates shapes, resolves repo-relative paths, checks that every
embedding file exists, and enforces base-matching: every store other than
``sentence`` must declare ``model`` equal to ``train.base_model``.

Identity. Training identity (``exp_id``, see :mod:`layout`) hashes the
frozen ``sets.json`` plus the ``source``, ``budget`` and ``train`` blocks
(scheduling keys stripped). ``embeddings``, ``analysis`` and ``evals`` stay
outside it: new embeddings never invalidate trained arms, and each eval
suite carries its own protocol hash in its ``COMPLETE.json`` so a changed
suite invalidates only that suite.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from valuegen._external import REPO_ROOT

SCHEMA_VERSION = 1
SUITES = ("prefill",)
SOURCE_METHODS = ("label_subset",)
SOURCE_FORMATS = ("dpo",)
REFERENCE_ARMS = ("base", "full")
BASE_ARM = "base"
FULL_ARM = "full"
MODEL_FREE_STORES = ("sentence",)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class ConfigError(ValueError):
    pass


def _need(block: dict, key: str, where: str):
    if key not in block:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return block[key]


def _check_keys(block: dict, allowed: set[str], where: str) -> None:
    if not isinstance(block, dict):
        raise ConfigError(f"{where}: must be a mapping")
    unknown = set(block) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")


def _pos_int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{where}: must be a positive integer, got {value!r}")
    return value


def _ident(value, where: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ConfigError(f"{where}: {value!r} must match [A-Za-z0-9][A-Za-z0-9_-]*")
    return value


def resolve_path(p: str | Path, root: Path = REPO_ROOT) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else root / p


# ── blocks ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UniverseConfig:
    value_set: str  # registry name or path (as written)
    restrict_to_source: bool = True


@dataclass(frozen=True)
class ExternalConfig:
    value_set: Path


@dataclass(frozen=True)
class StoreConfig:
    name: str
    universe: Path
    external: Path
    model: str | None = None
    # For stores whose vectors are 2-D ``[n_layers, hidden]`` (persona
    # ``response_avg_diff``): the layer row to use. None = vectors are 1-D.
    layer: int | None = None


@dataclass(frozen=True)
class SourceConfig:
    method: str
    artifact: str
    format: str = "dpo"


@dataclass(frozen=True)
class FamilyConfig:
    id: str
    k: int | None = None
    n: int | None = None
    values: tuple[tuple[str, ...], ...] | None = None

    @property
    def explicit(self) -> bool:
        return self.values is not None

    @property
    def size(self) -> int:
        return len(self.values) if self.values is not None else int(self.n)


@dataclass(frozen=True)
class ArmsConfig:
    reference: tuple[str, ...]
    families: tuple[FamilyConfig, ...]


@dataclass(frozen=True)
class BudgetConfig:
    rows: int


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    chat_format: str
    recipe: Path
    fsdp_config: Path | None
    seeds: tuple[int, ...]
    resources: dict[str, Any]
    revision: str | None = None
    concurrent: int = 1
    env: str = "default"
    prune_trainer_save: bool = True  # delete trainer weight shards once the export is verified


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    enabled: bool = True
    repeats: int = 1
    subsample: dict[str, int] | None = None  # {n, seed}
    grader: str | None = None  # None -> evals.default_grader
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalsConfig:
    default_grader: str
    candidate: dict[str, Any]
    resources: dict[str, Any]
    suites: dict[str, SuiteConfig]
    env: str = "eval_api"

    def grader(self, suite: str) -> str:
        return self.suites[suite].grader or self.default_grader

    def enabled(self) -> list[str]:
        return [s for s, c in self.suites.items() if c.enabled]


@dataclass(frozen=True)
class AnalysisConfig:
    outcomes: tuple[str, ...]
    metrics: tuple[str, ...] = ("coverage", "tightness")
    covariates: tuple[str, ...] = ("k", "rows")
    n_boot: int = 2000
    n_perm: int = 20000


WAVE_ORDERS = ("contiguous", "stride")


@dataclass(frozen=True)
class ScheduleConfig:
    """How ``mv run`` paces train -> eval waves. Pure scheduling: it changes
    when results land, never what they are, so it never enters the training
    identity (``exp_id``). ``mv run --wave / --delete-exports`` override it."""

    wave: int | None = None  # checkpoints (arm x seed) per wave; None = one wave
    delete_exports: bool = False  # rm export/ once every enabled suite is complete (tombstoned)
    order: str = "contiguous"  # contiguous slices of the run list, or stride (every wave spans all of it)


@dataclass(frozen=True)
class MultivalueConfig:
    name: str
    seed: int
    universe: UniverseConfig
    external: ExternalConfig
    embeddings: dict[str, StoreConfig]
    source: SourceConfig
    arms: ArmsConfig
    budget: BudgetConfig
    train: TrainConfig
    evals: EvalsConfig
    analysis: AnalysisConfig
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    raw: dict = field(default_factory=dict, compare=False, repr=False)
    path: Path | None = field(default=None, compare=False)

    def family(self, family_id: str) -> FamilyConfig:
        for f in self.arms.families:
            if f.id == family_id:
                return f
        raise KeyError(f"no family {family_id!r}; known: {[f.id for f in self.arms.families]}")

    def training_identity(self) -> dict:
        """The hashed part of the config minus the frozen sets (added by layout)."""
        train = copy.deepcopy(self.raw["train"])
        for k in ("resources", "concurrent", "env", "prune_trainer_save"):
            train.pop(k, None)
        return {
            "schema_version": SCHEMA_VERSION,
            "source": copy.deepcopy(self.raw["source"]),
            "budget": copy.deepcopy(self.raw["budget"]),
            "train": train,
        }


# ── loading ──────────────────────────────────────────────────────────────────


def _load_universe(block) -> UniverseConfig:
    _check_keys(block, {"value_set", "restrict_to_source"}, "universe")
    vs = _need(block, "value_set", "universe")
    if not isinstance(vs, str) or not vs:
        raise ConfigError("universe.value_set must be a registry name or a path")
    return UniverseConfig(value_set=vs, restrict_to_source=bool(block.get("restrict_to_source", True)))


def _load_external(block, root: Path) -> ExternalConfig:
    _check_keys(block, {"value_set"}, "external")
    vs = resolve_path(_need(block, "value_set", "external"), root)
    if not vs.is_file():
        raise ConfigError(f"external.value_set: no file at {vs}")
    return ExternalConfig(value_set=vs)


def _load_embeddings(block, base_model: str, root: Path, require_files: bool) -> dict[str, StoreConfig]:
    if not isinstance(block, dict) or not block:
        raise ConfigError("embeddings: needs at least one store, e.g. sentence: {universe, external}")
    stores: dict[str, StoreConfig] = {}
    for name, spec in block.items():
        where = f"embeddings.{name}"
        _ident(name, where)
        _check_keys(spec, {"universe", "external", "model", "layer"}, where)
        uni = resolve_path(_need(spec, "universe", where), root)
        ext = resolve_path(_need(spec, "external", where), root)
        if require_files:
            for label, p in (("universe", uni), ("external", ext)):
                if not p.is_file():
                    raise ConfigError(
                        f"{where}.{label}: no embedding file at {p}. Embeddings are computed "
                        "externally; drop the file in (npz / npy+labels.json / pt / json) and rerun."
                    )
        model = spec.get("model")
        if name not in MODEL_FREE_STORES:
            if not model:
                raise ConfigError(
                    f"{where}: stores other than {MODEL_FREE_STORES} must declare `model:` "
                    "(the model their vectors were extracted from)"
                )
            if str(model) != base_model:
                raise ConfigError(
                    f"{where}.model {model!r} != train.base_model {base_model!r}: "
                    "embedding stores must be base-matched"
                )
        layer = spec.get("layer")
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or layer < 0):
            raise ConfigError(f"{where}.layer must be a non-negative integer")
        stores[name] = StoreConfig(name=name, universe=uni, external=ext,
                                   model=str(model) if model else None, layer=layer)
    return stores


def _load_source(block) -> SourceConfig:
    _check_keys(block, {"method", "artifact", "format"}, "source")
    method = _need(block, "method", "source")
    if method not in SOURCE_METHODS:
        raise ConfigError(f"source.method {method!r} not in {SOURCE_METHODS}")
    fmt = block.get("format", "dpo")
    if fmt not in SOURCE_FORMATS:
        raise ConfigError(f"source.format {fmt!r} not in {SOURCE_FORMATS}")
    artifact = _need(block, "artifact", "source")
    if not isinstance(artifact, str) or "/" in artifact or not artifact:
        raise ConfigError("source.artifact must be an artifact directory name (no slashes)")
    return SourceConfig(method=method, artifact=artifact, format=fmt)


def _load_family(block) -> FamilyConfig:
    where = f"arms.families[{block.get('id')!r}]"
    _check_keys(block, {"id", "k", "n", "values"}, where)
    fid = _ident(_need(block, "id", where), where + ".id")
    if fid in REFERENCE_ARMS:
        raise ConfigError(f"{where}: family id {fid!r} collides with a reference arm")
    if "values" in block:
        if "k" in block or "n" in block:
            raise ConfigError(f"{where}: an explicit family takes `values`, not k/n")
        vals = block["values"]
        if not isinstance(vals, list) or not vals or not all(isinstance(s, list) and s for s in vals):
            raise ConfigError(f"{where}.values must be a non-empty list of non-empty value lists")
        sets = []
        for s in vals:
            if len(set(s)) != len(s):
                raise ConfigError(f"{where}: duplicate value inside a set: {s}")
            sets.append(tuple(sorted(str(v) for v in s)))
        if len(set(sets)) != len(sets):
            raise ConfigError(f"{where}: repeated explicit set")
        return FamilyConfig(id=fid, values=tuple(sets))
    return FamilyConfig(id=fid, k=_pos_int(_need(block, "k", where), where + ".k"),
                        n=_pos_int(_need(block, "n", where), where + ".n"))


def _load_arms(block) -> ArmsConfig:
    _check_keys(block, {"reference", "families"}, "arms")
    ref = block.get("reference", [BASE_ARM])
    if not isinstance(ref, list) or BASE_ARM not in ref:
        raise ConfigError(f"arms.reference must be a list containing {BASE_ARM!r}")
    for r in ref:
        if r not in REFERENCE_ARMS:
            raise ConfigError(f"arms.reference: unknown reference arm {r!r}; known: {REFERENCE_ARMS}")
    fams = block.get("families") or []
    if not isinstance(fams, list):
        raise ConfigError("arms.families must be a list")
    families = tuple(_load_family(f) for f in fams)
    ids = [f.id for f in families]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"arms.families: duplicate ids in {ids}")
    return ArmsConfig(reference=tuple(dict.fromkeys(ref)), families=families)


def _load_budget(block) -> BudgetConfig:
    _check_keys(block, {"rows"}, "budget")
    return BudgetConfig(rows=_pos_int(_need(block, "rows", "budget"), "budget.rows"))


def _load_train(block, root: Path) -> TrainConfig:
    _check_keys(block, {"base_model", "revision", "chat_format", "recipe", "fsdp_config",
                        "seeds", "resources", "concurrent", "env", "prune_trainer_save"}, "train")
    base = _need(block, "base_model", "train")
    if not isinstance(base, str) or not base:
        raise ConfigError("train.base_model must be an HF id or a local path")
    recipe = resolve_path(_need(block, "recipe", "train"), root)
    if not recipe.is_file():
        raise ConfigError(f"train.recipe: no file at {recipe}")
    fsdp = block.get("fsdp_config")
    if fsdp is not None:
        fsdp = resolve_path(fsdp, root)
        if not fsdp.is_file():
            raise ConfigError(f"train.fsdp_config: no file at {fsdp}")
    seeds = block.get("seeds", [42])
    if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ConfigError("train.seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("train.seeds has duplicates")
    resources = dict(block.get("resources") or {})
    _check_keys(resources, {"gpus", "time", "mem", "nproc"}, "train.resources")
    resources.setdefault("gpus", 1)
    resources.setdefault("time", "04:00:00")
    _pos_int(resources["gpus"], "train.resources.gpus")
    rev = block.get("revision")
    return TrainConfig(
        base_model=base, revision=str(rev) if rev is not None else None,
        chat_format=str(_need(block, "chat_format", "train")), recipe=recipe, fsdp_config=fsdp,
        seeds=tuple(int(s) for s in seeds), resources=resources,
        concurrent=_pos_int(block.get("concurrent", 1), "train.concurrent"),
        env=str(block.get("env", "default")),
        prune_trainer_save=bool(block.get("prune_trainer_save", True)),
    )


def _load_suite(name: str, block) -> SuiteConfig:
    where = f"evals.suites.{name}"
    if name not in SUITES:
        raise ConfigError(f"{where}: unknown suite; known: {SUITES}")
    block = dict(block or {})
    known = {"enabled", "repeats", "subsample", "grader"}
    extra = {k: block[k] for k in block if k not in known}
    enabled = bool(block.get("enabled", True))
    repeats = _pos_int(block.get("repeats", 1), where + ".repeats")
    sub = block.get("subsample")
    if sub is not None:
        _check_keys(sub, {"n", "seed"}, where + ".subsample")
        sub = {"n": _pos_int(_need(sub, "n", where + ".subsample"), where + ".subsample.n"),
               "seed": int(sub.get("seed", 0))}
    grader = block.get("grader")
    return SuiteConfig(name=name, enabled=enabled, repeats=repeats, subsample=sub,
                       grader=str(grader) if grader else None, extra=extra)


def _load_evals(block) -> EvalsConfig:
    _check_keys(block, {"default_grader", "candidate", "resources", "suites", "env"}, "evals")
    grader = _need(block, "default_grader", "evals")
    if not isinstance(grader, str) or "/" not in grader:
        raise ConfigError("evals.default_grader must be an Inspect model name like provider/model")
    candidate = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 4096, "max_connections": 32}
    candidate.update(block.get("candidate") or {})
    resources = {"gpus": 1, "time": "12:00:00"}
    resources.update(block.get("resources") or {})
    suites_block = block.get("suites")
    if not isinstance(suites_block, dict) or not suites_block:
        raise ConfigError("evals.suites must name at least one suite")
    suites = {name: _load_suite(name, spec) for name, spec in suites_block.items()}
    for s in suites.values():
        if s.grader is not None and "/" not in s.grader:
            raise ConfigError(f"evals.suites.{s.name}.grader must be provider/model")
    return EvalsConfig(default_grader=grader, candidate=candidate, resources=resources,
                       suites=suites, env=str(block.get("env", "eval_api")))


def _load_analysis(block, store_names: list[str]) -> AnalysisConfig:
    block = dict(block or {})
    _check_keys(block, {"outcomes", "metrics", "covariates", "n_boot", "n_perm"}, "analysis")
    outcomes = block.get("outcomes") or []
    if not isinstance(outcomes, list):
        raise ConfigError("analysis.outcomes must be a list of suite.metric names")
    metrics = tuple(block.get("metrics") or ("coverage", "tightness"))
    for m in metrics:
        if m not in ("coverage", "tightness"):
            raise ConfigError(f"analysis.metrics: unknown metric {m!r} (coverage, tightness)")
    covariates = tuple(block.get("covariates") or ("k", "rows"))
    for c in covariates:
        if c not in ("k", "rows"):
            raise ConfigError(f"analysis.covariates: unknown covariate {c!r} (k, rows)")
    return AnalysisConfig(outcomes=tuple(str(o) for o in outcomes), metrics=metrics, covariates=covariates,
                          n_boot=_pos_int(block.get("n_boot", 2000), "analysis.n_boot"),
                          n_perm=_pos_int(block.get("n_perm", 20000), "analysis.n_perm"))


def _load_schedule(block) -> ScheduleConfig:
    block = dict(block or {})
    _check_keys(block, {"wave", "delete_exports", "order"}, "schedule")
    wave = block.get("wave")
    if wave is not None:
        wave = _pos_int(wave, "schedule.wave")
    delete = block.get("delete_exports", False)
    if not isinstance(delete, bool):
        raise ConfigError(f"schedule.delete_exports must be a boolean, got {delete!r}")
    order = block.get("order", "contiguous")
    if order not in WAVE_ORDERS:
        raise ConfigError(f"schedule.order must be one of {WAVE_ORDERS}, got {order!r}")
    return ScheduleConfig(wave=wave, delete_exports=delete, order=order)


TOP_KEYS = {"name", "seed", "universe", "external", "embeddings", "source", "arms", "budget",
            "train", "evals", "analysis", "schedule"}


def parse_config(raw: dict, root: Path = REPO_ROOT, path: Path | None = None,
                 require_files: bool = True) -> MultivalueConfig:
    """Validate a config dict. ``require_files=False`` skips the embedding
    file-existence check (for ``status`` on a machine without the stores);
    the base-match check still runs."""
    _check_keys(raw, TOP_KEYS, "config")
    name = _ident(_need(raw, "name", "config"), "name")
    seed = raw.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("seed must be an integer")
    train = _load_train(_need(raw, "train", "config"), root)
    stores = _load_embeddings(_need(raw, "embeddings", "config"), train.base_model, root, require_files)
    cfg = MultivalueConfig(
        name=name, seed=seed,
        universe=_load_universe(_need(raw, "universe", "config")),
        external=_load_external(_need(raw, "external", "config"), root),
        embeddings=stores,
        source=_load_source(_need(raw, "source", "config")),
        arms=_load_arms(_need(raw, "arms", "config")),
        budget=_load_budget(_need(raw, "budget", "config")),
        train=train,
        evals=_load_evals(_need(raw, "evals", "config")),
        analysis=_load_analysis(raw.get("analysis"), list(stores)),
        schedule=_load_schedule(raw.get("schedule")),
        raw=copy.deepcopy(raw), path=path,
    )
    return cfg


def load_config(path: str | Path, root: Path = REPO_ROOT, require_files: bool = True) -> MultivalueConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config must be a mapping")
    try:
        return parse_config(raw, root=root, path=path.resolve(), require_files=require_files)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None
