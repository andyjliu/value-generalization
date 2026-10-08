"""The shared suite runner: frozen inputs, protocol hash, two-pass run,
completion marker.

Every suite runs the same way against a candidate the enclosing job already
serves (``evaluation.serve_local_sh``): probe the endpoint, generate every
unit that is not complete (``inspect_runner.run_generation``, ``score=False``),
grade the persisted logs through the suite's cached scorer (up to three
passes, so transient grader failures are retried from the cache without
regenerating), check completeness (``inspect_runner.suite_complete``), write
``rows.jsonl`` and ``COMPLETE.json``.

``COMPLETE.json`` carries ``protocol_sha256`` = hash of everything that
defines the suite's protocol: the frozen inputs' content hash, repeats,
grader, candidate sampling parameters, grading options and the scorer
version. A marker with another hash is stale and the suite reruns; a marker
without one (an RQ3 driver's, reached through an imported arm) counts when
its recorded epochs, grader, sampling parameters and per-unit expectations
match this protocol exactly (:func:`marker_accepts`).

Frozen inputs live at ``eval_inputs/{suite}/inputs.json`` and are rebuilt
only when their build parameters (subsample, dataset dir) change.

A per-candidate suite (prefill) selects its samples per candidate
(``Suite.select``); the selection is written to the suite dir as
``selection.json`` before generation, its options enter the protocol hash
(``Suite.protocol_extra``), and a marker under another hash is never
accepted by content.

The grader is any Inspect provider whose API key the job environment
carries (the cluster preamble sources ``.env``); the provider -> variable
map below is what ``mv eval`` checks before submitting.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from valuegen._external import REPO_ROOT
from valuegen.config import ClusterConfig
from valuegen.ground_truth import inspect_runner as IR
from valuegen.multivalue import data as mvdata
from valuegen.multivalue._hashing import sha256_json
from valuegen.multivalue.config import MultivalueConfig
from valuegen.multivalue.evals import get_suite
from valuegen.multivalue.evals.base import Candidate, EvalError
from valuegen.multivalue.layout import Layout

MARKER = "COMPLETE.json"
PROVIDER_KEY_ENV = {
    "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY", "grok": "GROK_API_KEY", "together": "TOGETHER_API_KEY",
    "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY", "fireworks": "FIREWORKS_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY", "cloudflare": "CLOUDFLARE_API_TOKEN",
}
CANDIDATE_PROTOCOL_KEYS = ("temperature", "top_p", "max_tokens")


# ── grader ───────────────────────────────────────────────────────────────────


def grader_env_var(name: str) -> str | None:
    """The API-key variable Inspect's provider for ``name`` reads, if known."""
    return PROVIDER_KEY_ENV.get(name.split("/", 1)[0])


def env_file_vars(path: Path) -> set[str]:
    """Variable names assigned in a ``.env`` file (no values are read back)."""
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.add(line.split("=", 1)[0].removeprefix("export ").strip())
    return names


def grader_key_present(name: str, cluster: ClusterConfig) -> tuple[str | None, bool]:
    """``(variable, present)``: present in the process environment or in the
    repo ``.env`` the job preamble sources. Unknown providers are not checked."""
    var = grader_env_var(name)
    if var is None:
        return None, True
    return var, var in os.environ or var in env_file_vars(cluster.repo / ".env")


JUDGE_PROVIDER = "prefjudge"


def _served_judge(name: str) -> dict | None:
    """Route an in-job locally served judge (``PREFILL_JUDGE_URL``) instead of
    an API provider. Active only when the grader's provider is not a known API
    provider (so ``google/...`` graders are untouched) and the served endpoint
    is set. Keeps the grader *identity* string (e.g. ``Qwen/Qwen3.6-27B``,
    hashed into the frozen build_key and completion markers) unchanged; only
    the transport (``openai-api`` + explicit base_url) is swapped, and Qwen3
    thinking is disabled with ``chat_template_kwargs`` when
    ``PREFILL_JUDGE_NO_THINK`` is set -- mirroring the served build writers."""
    url = os.environ.get("PREFILL_JUDGE_URL")
    if not url:
        return None
    if name.split("/", 1)[0] in PROVIDER_KEY_ENV:
        return None
    served_name = os.environ.get("PREFILL_JUDGE_MODEL") or name
    no_think = os.environ.get("PREFILL_JUDGE_NO_THINK", "").strip().lower() in ("1", "true", "yes")
    return {"model": f"openai-api/{JUDGE_PROVIDER}/{served_name}", "base_url": url, "api_key": "api",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}} if no_think else None}


def grader_model(name: str, max_connections: int = 8, max_retries: int = 6, timeout: int = 600):
    """The grader Model through its own provider (any provider; credentials
    from the environment). A locally served judge is used instead when
    ``PREFILL_JUDGE_URL`` is set (see :func:`_served_judge`)."""
    from inspect_ai.model import GenerateConfig, get_model

    served = _served_judge(name)
    if served is not None:
        prov = JUDGE_PROVIDER.upper()
        os.environ.setdefault(f"{prov}_API_KEY", served["api_key"])
        os.environ.setdefault(f"{prov}_BASE_URL", served["base_url"])
        return get_model(served["model"], base_url=served["base_url"], api_key=served["api_key"],
                         config=GenerateConfig(max_connections=max_connections, max_retries=max_retries,
                                               timeout=timeout, extra_body=served["extra_body"]))
    return get_model(name, config=GenerateConfig(max_connections=max_connections, max_retries=max_retries,
                                                 timeout=timeout))


# ── frozen inputs + protocol ─────────────────────────────────────────────────


def inputs_path(layout: Layout, suite: str) -> Path:
    return layout.eval_inputs_dir / suite / "inputs.json"


def ensure_inputs(cfg: MultivalueConfig, layout: Layout, suite: str, *, rebuild: bool = False,
                  root: Path = REPO_ROOT, log=print) -> dict:
    """The suite's frozen inputs, built when missing or when their build
    parameters changed (or ``rebuild``)."""
    s = get_suite(suite)
    sc = cfg.evals.suites[suite]
    path = inputs_path(layout, suite)
    params = s.params(cfg, sc)
    if path.is_file() and not rebuild:
        rec = mvdata.read_json(path)
        if rec.get("params") == params and rec.get("schema_version") == 1:
            return rec
        log(f"{suite}: inputs at {path} were built with other parameters; rebuilding")
    rec = s.build_inputs(cfg, sc, root, layout=layout)
    mvdata.write_json(path, rec)
    log(f"{suite}: frozen inputs -> {path} (inputs_sha256 {rec['inputs_sha256'][:12]})")
    return rec


def candidate_protocol(cfg: MultivalueConfig) -> dict:
    return {k: cfg.evals.candidate[k] for k in CANDIDATE_PROTOCOL_KEYS}


def protocol_sha(cfg: MultivalueConfig, suite: str, inputs: Mapping[str, Any], *, smoke: bool = False) -> str:
    s = get_suite(suite)
    sc = cfg.evals.suites[suite]
    opts = s.options(sc)
    payload = {
        "suite": suite, "inputs_sha256": inputs["inputs_sha256"], "repeats": 1 if smoke else sc.repeats,
        "grader": cfg.evals.grader(suite), "candidate": candidate_protocol(cfg),
        "num_grader_samples": opts.get("num_grader_samples"), "scorer_version": IR.SCORER_VERSION,
        "smoke": bool(smoke),
    }
    extra = s.protocol_extra(opts)
    if extra:  # only suites that define one (keeps the other suites' hashes stable)
        payload["protocol_extra"] = extra
    return sha256_json(payload)


# ── markers ──────────────────────────────────────────────────────────────────


def read_marker(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return mvdata.read_json(path)
    except (OSError, ValueError):
        return None


def marker_accepts(marker: Mapping[str, Any] | None, cfg: MultivalueConfig, suite: str, inputs: Mapping[str, Any],
                   *, smoke: bool = False) -> tuple[bool, str]:
    """Whether a completion marker satisfies this suite's protocol: by hash,
    or (a marker written by another driver) by content."""
    if not marker or not marker.get("complete"):
        return False, "incomplete" if marker else "no marker"
    psha = protocol_sha(cfg, suite, inputs, smoke=smoke)
    if marker.get("protocol_sha256") == psha:
        return True, "protocol"
    if smoke:
        return False, "protocol hash differs"
    sc = cfg.evals.suites[suite]
    s = get_suite(suite)
    if s.per_candidate:  # no content path: the expected units depend on the candidate's selection
        return False, "protocol hash differs (per-candidate suite)"
    if marker.get("suite") != suite:
        return False, f"suite {marker.get('suite')!r}"
    if int(marker.get("epochs", -1)) != sc.repeats:
        return False, f"epochs {marker.get('epochs')} != repeats {sc.repeats}"
    if marker.get("grader") != cfg.evals.grader(suite):
        return False, f"grader {marker.get('grader')!r} != {cfg.evals.grader(suite)!r}"
    cand = marker.get("candidate") or {}
    for k in CANDIDATE_PROTOCOL_KEYS:
        if cand.get(k) != cfg.evals.candidate[k]:
            return False, f"candidate {k} {cand.get(k)!r} != {cfg.evals.candidate[k]!r}"
    units = marker.get("units") or {}
    expected = s.unit_expectations(inputs)
    if set(units) != set(expected):
        return False, f"{len(units)} units != {len(expected)} expected"
    for uid, n in expected.items():
        if int(units[uid].get("expected", -1)) != n * sc.repeats:
            return False, f"unit {uid}: {units[uid].get('expected')} samples != {n * sc.repeats}"
    if marker.get("problems"):
        return False, "marker lists problems"
    return True, "compatible"


def suite_root(layout: Layout, cand: Candidate, suite: str, *, smoke: bool = False) -> Path:
    return layout.suite_dir(cand.ckpt_id, suite, smoke=smoke)


def suite_state(cfg: MultivalueConfig, layout: Layout, cand: Candidate, suite: str, inputs: Mapping[str, Any],
                *, smoke: bool = False) -> str:
    """done | stale (marker under another protocol) | partial (logs, no
    marker) | reusable (an imported arm's compatible external run, not yet
    linked) | pending."""
    root = suite_root(layout, cand, suite, smoke=smoke)
    marker = read_marker(root / MARKER)
    if marker is not None:
        ok, _ = marker_accepts(marker, cfg, suite, inputs, smoke=smoke)
        return "done" if ok else "stale"
    if root.is_dir() and any(p.suffix == ".eval" for p in root.rglob("*.eval")):
        return "partial"
    ext = cand.external_suites.get(suite)
    if ext and not smoke and marker_accepts(read_marker(Path(ext) / MARKER), cfg, suite, inputs)[0]:
        return "reusable"
    return "pending"


def link_external_suite(layout: Layout, cand: Candidate, suite: str) -> Path:
    """Reuse an imported arm's completed suite dir in place (a symlink under
    ``evals/{ckpt}/``); the marker is accepted by content, never rewritten."""
    root = suite_root(layout, cand, suite)
    ext = Path(cand.external_suites[suite])
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.exists():
        if root.resolve() == ext.resolve():
            return root
        raise EvalError(f"{root} exists and is not the imported suite dir {ext}")
    root.symlink_to(ext, target_is_directory=True)
    return root


# ── running one suite ────────────────────────────────────────────────────────


def make_run(cfg: MultivalueConfig, layout: Layout, cand: Candidate, suite: str, inputs: Mapping[str, Any],
             base_url: str, *, smoke: bool = False, selection: dict | None = None) -> IR.SuiteRun:
    """The :class:`inspect_runner.SuiteRun` for one candidate x suite; a
    per-candidate suite's sample selection is computed here when not given."""
    s = get_suite(suite)
    sc = cfg.evals.suites[suite]
    opts = s.options(sc)
    if selection is None:
        selection = s.select(inputs, cand, opts, cfg.seed, smoke=smoke)
    c = cfg.evals.candidate
    run = IR.SuiteRun(
        suite=suite, checkpoint_id=cand.ckpt_id, served_name=cand.served_name, base_url=base_url,
        log_root=suite_root(layout, cand, suite, smoke=smoke), grader_name=cfg.evals.grader(suite),
        candidate=IR.CandidateConfig(temperature=float(c["temperature"]), top_p=float(c["top_p"]),
                                     max_tokens=int(c["max_tokens"]), max_connections=int(c.get("max_connections", 32))),
        epochs=1 if smoke else int(sc.repeats),
        num_grader_samples=int(opts.get("num_grader_samples", 1)),
        grader_max_attempts=int(opts.get("grader_max_attempts", 3)),
        grader_max_connections=int(opts.get("grader_max_connections", 8)),
        max_failed_grade_fraction=float(opts.get("max_failed_grade_fraction", 0.0)),
    )
    return s.attach(run, dict(inputs), smoke=smoke, cand=cand, selection=selection)


def grade_units(run: IR.SuiteRun, units: Sequence[tuple[str, Path, Any]], suite, passes: int = 3) -> dict:
    """Phase 2, with the suite's scorer and an any-provider grader: grade
    persisted responses, never regenerate."""
    from inspect_ai import score as inspect_score
    from inspect_ai.log import read_eval_log, write_eval_log

    IR.ensure_candidate_env(run.base_url)
    grader = grader_model(run.grader_name, max_connections=run.grader_max_connections)
    cache = IR.GradeCache(run.cache_path)
    report = {}
    for unit_id, log_dir, _ in units:
        info = IR._latest_log(log_dir)
        if info is None:
            report[unit_id] = {"graded": False, "reason": "no generation log"}
            continue
        scorer = suite.scorer(run, grader, cache)
        log = read_eval_log(info.name)
        result = None
        for _ in range(passes):
            log = inspect_score(log, scorers=[scorer], action="overwrite", display=run.display)
            result = IR.unit_grading_status_from_log(log)
            if result["n_failed"] == 0:
                break
        write_eval_log(log, info.name)
        report[unit_id] = {"graded": True, **(result or {})}
    return report


def usage_summary(rows: Sequence[Mapping[str, Any]]) -> dict:
    usage: dict[str, dict] = {}
    for r in rows:
        for model, u in (r.get("model_usage") or {}).items():
            agg = usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "n": 0})
            agg["input_tokens"] += int(u.get("input_tokens") or 0)
            agg["output_tokens"] += int(u.get("output_tokens") or 0)
            agg["n"] += 1
    return usage


def run_suite(cfg: MultivalueConfig, layout: Layout, cand: Candidate, suite: str, base_url: str, *,
              smoke: bool = False, log=print) -> int:
    """Run one suite against the served candidate; 0 when complete."""
    s = get_suite(suite)
    inputs = ensure_inputs(cfg, layout, suite, log=log)
    psha = protocol_sha(cfg, suite, inputs, smoke=smoke)
    root = suite_root(layout, cand, suite, smoke=smoke)
    marker_path = root / MARKER
    ok, why = marker_accepts(read_marker(marker_path), cfg, suite, inputs, smoke=smoke)
    if ok:
        log(f"{suite}/{cand.ckpt_id}: already complete ({why})")
        return 0
    if marker_path.is_file():
        log(f"{suite}/{cand.ckpt_id}: stale marker ({why}); rerunning what is missing")
    sel = s.select(inputs, cand, s.options(cfg.evals.suites[suite]), cfg.seed, smoke=smoke)
    run = make_run(cfg, layout, cand, suite, inputs, base_url, smoke=smoke, selection=sel)
    root.mkdir(parents=True, exist_ok=True)
    if sel is not None:  # what this candidate was asked (the completeness check and rows agree on it)
        mvdata.write_json(root / "selection.json", {**sel, "ckpt_id": cand.ckpt_id, "written_at": mvdata.now()})
    probe = IR.probe_endpoint(base_url, cand.served_name)
    mvdata.write_json(root / "endpoint_probe.json", {**probe, "base_url": base_url, "at": mvdata.now()})
    units = IR.generate_units(run)
    gen = IR.run_generation(run, units)
    mvdata.write_json(root / "generation_report.json", gen)
    grade = grade_units(run, units, s)
    mvdata.write_json(root / "grading_report.json", grade)
    status = IR.suite_complete(run, s.unit_expectations(inputs, smoke=smoke, selection=sel))
    rows = s.rows(run)
    with open(root / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary = s.summarize(rows)
    IR.write_completion_marker(marker_path, run, status, extra={
        "protocol_sha256": psha, "exp_id": layout.exp_id, "arm_id": cand.arm.id, "arm_kind": cand.arm.kind,
        "seed": cand.seed, "kind": cand.kind, "inputs_sha256": inputs["inputs_sha256"],
        "subsample": cfg.evals.suites[suite].subsample, "smoke": smoke, "n_rows": len(rows),
        "n_valid": sum(1 for r in rows if r.get("valid")), "summary": summary, "usage": usage_summary(rows),
        "endpoint_probe": probe, "served_model_dir": str(cand.model_dir), "boot_unit": s.boot_unit,
        "selection_counts": None if sel is None else sel.get("counts"),
    })
    log(json.dumps({"suite": suite, "checkpoint": cand.ckpt_id, "complete": status["complete"],
                    "problems": status["problems"][:10], "rows": len(rows),
                    "summary": {k: v for k, v in summary.items() if not k.startswith("by_")}}, indent=2))
    return 0 if status["complete"] else 1


def grader_smoke(name: str) -> dict:
    """One real grader call through the any-provider factory."""
    import asyncio

    from inspect_ai.model import ChatMessageUser

    model = grader_model(name, max_connections=1)

    async def go():
        return await model.generate([ChatMessageUser(content="Reply with the single word: ready")])

    out = asyncio.run(go())
    return {"requested": name, "returned_model": out.model, "completion": out.completion[:200],
            "usage": out.usage.model_dump() if out.usage else None, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
