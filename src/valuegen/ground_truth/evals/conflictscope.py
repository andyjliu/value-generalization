"""ConflictScope scenario-choice evaluation: manifest -> eval CSVs -> matrices.

The one eval method of pass 1: run the submodule's ``evaluate_models`` on a
scenario set for the base model plus every manifest entry (checkpoints as
``-m``, steering prompts as ``--steer-prompt``), then build per-model
steerability matrices with the single estimator in ``matrices.py``.

**Base evals are shared.** A base model's eval depends only on (base model,
eval identity) — not on which interventions ride along — so it lives in
``data/gt/base_evals/{base_eval_id}/`` keyed by the judgment-affecting eval
knobs (scenarios, mode, judge/user models, temperature, clipping; never
serve/GPU scheduling) and is symlinked into each GT run's ``model_evals/`` as
``{mtag}_base.csv``. Three intervention sets scored under one eval config run
the base once. The N+1 rule still lives in ``evaluation.eval_tasks`` — the
base task is always emitted; only its output is re-homed.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import replace
from pathlib import Path

from valuegen import values as V
from valuegen.config import (
    ClusterConfig,
    config_hash,
    ensure_config_record,
    gt_id,
    intervention_id,
    pool_id,
)
from valuegen.ground_truth import common, gcs, interventions
from valuegen.ground_truth import matrices as M
from valuegen.ground_truth import training
from valuegen.ground_truth.evaluation import (
    EvalSpec,
    eval_command,
    eval_tasks,
    serve_stage,
)
from valuegen.slurm import Stage, Task

VARIANTS = ("choice_raw", "choice_normalized", "likert_raw", "likert_normalized")


def scenarios_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    """The eval scenario dir: a plain path, or a pool's test half."""
    scenarios = cfg["evaluation"]["scenarios"]
    if isinstance(scenarios, dict):
        from valuegen.ground_truth import scenario_pool

        return scenario_pool.test_dir(scenarios["pool"], cluster) / "outputs"
    return common.configured_path(scenarios, cfg, cluster)


# ── Shared base-eval store ───────────────────────────────────────────────────


def _scenarios_identity(cfg: dict, cluster: ClusterConfig):
    """The eval scenarios as the identity sees them: a scenario pool by its
    own pool ID, a relative dir as written in the config, and an absolute dir
    relative to the data root (``<data>/...``) or the repo when it lies under
    one -- so one config hashes to one store on every clone and layout."""
    scenarios = cfg["evaluation"]["scenarios"]
    if isinstance(scenarios, dict):
        return {"pool": pool_id(scenarios["pool"])}
    rendered = str(scenarios).format(
        gt_id=gt_id(cfg), intervention_id=intervention_id(cfg), experiment=cfg["name"],
    )
    path = Path(rendered)
    if not path.is_absolute():
        return path.as_posix()
    for root, prefix in ((cluster.data, "<data>/"), (cluster.repo, "")):
        try:
            return prefix + path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            continue
    return str(path)


def base_eval_identity(cfg: dict, cluster: ClusterConfig) -> dict:
    """The knobs that can change a base model's eval CSV — and nothing else.

    Scheduling (serve locality, GPUs, walltime) and everything downstream
    (matrices, interventions) are deliberately excluded, the same identity
    discipline as label stores and scenario pools.
    """
    ev = cfg["evaluation"]
    return {
        "method": "conflictscope",
        "schema_version": ev["schema_version"],
        "scenarios": _scenarios_identity(cfg, cluster),
        "mode": ev["mode"],
        "cache": ev["cache"],
        "filter": ev["filter"],
        "temperature": ev["temperature"],
        "max_tokens": ev["max_tokens"],
        "max_scenarios": ev["max_scenarios"],
        "judge_model": ev["judge"].get("model"),
        "user_model": ev.get("user_model"),
    }


def base_eval_id(cfg: dict, cluster: ClusterConfig) -> str:
    return f"conflictscope-{config_hash(base_eval_identity(cfg, cluster))}"


def base_evals_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    return cluster.data / "gt" / "base_evals" / base_eval_id(cfg, cluster)


def claim_base_evals(cfg: dict, cluster: ClusterConfig) -> Path:
    return ensure_config_record(
        base_evals_dir(cfg, cluster) / "resolved_config.yaml",
        base_eval_id(cfg, cluster),
        base_eval_identity(cfg, cluster),
    )


def _model_short(hf_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", hf_name.split("/")[-1])


def base_scaffold(cfg: dict, mtag: str) -> str | None:
    """``evaluation.scaffolds[mtag]``: a prompt scaffold (``training.
    SCAFFOLD_TEMPLATES``) or an explicit chat format (``chat_formats.
    CHAT_FORMATS``) for the model in the ``{mtag}_base.csv`` slot.

    The base slot only: a raw pretrained base is prompted under URIAL, and a
    chat base whose HF name would select the wrong formatter is pinned to its
    real chat format, while the checkpoints trained from it are formatted by
    their mtag-derived dir names. This
    is a per-model knob, not an eval knob, so it is outside
    ``base_eval_identity``; the scaffold is carried in the shared CSV's name
    instead (``{model}_{scaffold}_base.csv``), and the spec it ran under is
    written beside it (``scaffolds/{scaffold}.json``).
    """
    name = (cfg["evaluation"].get("scaffolds") or {}).get(mtag)
    if name is None:
        return None
    training.scaffold_spec(name)  # validate early: unknown names fail at plan time
    return str(name)


def _shared_base_task(
    spec: EvalSpec,
    cfg: dict,
    cluster: ClusterConfig,
    mtag: str,
    hf_name: str,
    scaffold: str | None = None,
) -> Task:
    """The base eval against the shared store, linked into this GT run.

    The eval itself is guarded so a second GT run under the same eval identity
    only refreshes the symlink; the done-check is the link (it follows to the
    shared file), so a run whose link is missing re-runs just the cheap tail.
    """
    shared_dir = base_evals_dir(cfg, cluster)
    stem = _model_short(hf_name) + (f"_{scaffold}" if scaffold else "")
    shared_csv = shared_dir / f"{stem}_base.csv"
    link = common.eval_dir(cfg, cluster) / f"{mtag}_base.csv"
    scaffold_json = None
    if scaffold:
        scaffold_json = shared_dir / "scaffolds" / f"{scaffold}.json"
        scaffold_json.parent.mkdir(parents=True, exist_ok=True)
        scaffold_json.write_text(
            json.dumps(training.scaffold_spec(scaffold), indent=2) + "\n"
        )
    # A base-slot model may itself be a checkpoint that lives in the bucket
    # (e.g. a neutral-SFT model trained outside a GT run): anything given as
    # an absolute path gets the same run-time stage-in prelude an
    # intervention checkpoint gets.
    stage_in = None
    if cluster.gcs is not None and Path(hf_name).is_absolute():
        stage_in = gcs.stage_in_sh(cluster.gcs, hf_name)
    inner = eval_command(replace(spec, output_dir=shared_dir), hf_name,
                         shared_csv.name, stage_in=stage_in, scaffold=scaffold_json)
    command = (
        f"if [ ! -s {shlex.quote(str(shared_csv))} ]; then\n"
        f"{inner}\n"
        f"fi\n"
        f"mkdir -p {shlex.quote(str(link.parent))}\n"
        f"ln -sfn {shlex.quote(str(shared_csv))} {shlex.quote(str(link))}"
    )
    return Task(key=f"{mtag}_base", command=command, done=link)


# ── Eval spec + stages ───────────────────────────────────────────────────────


def judge_hostfile(cfg: dict, cluster: ClusterConfig) -> Path:
    return common.script_dir(cfg, cluster) / "judge_host.txt"


def judge_stage(cfg: dict, cluster: ClusterConfig) -> Stage | None:
    """Persistent user-sim/judge vLLM server, when ``judge.serve: local``."""
    judge = cfg["evaluation"].get("judge", {})
    if judge.get("serve") != "local":
        return None
    return serve_stage(
        name="serve_judge",
        model=judge["model"],
        hostfile=judge_hostfile(cfg, cluster),
        gpus=int(judge.get("gpus", 2)),
        mem=judge.get("mem", "96G"),
        time=judge.get("time", "2-00:00:00"),
        env=judge.get("env", "default"),
        load_path=cluster.model_path(judge["model"]),
        enable_prefix_caching=bool(judge.get("enable_prefix_caching", True)),
        extra_args=tuple(judge.get("serve_args", ())),
    )


def build_eval_spec(cfg: dict, cluster: ClusterConfig) -> EvalSpec:
    ev = cfg["evaluation"]
    judge = ev.get("judge", {})
    hostfile = judge_hostfile(cfg, cluster)
    served = judge.get("serve") == "local"
    judge_model = judge.get("model")
    scenarios = scenarios_dir(cfg, cluster)
    return EvalSpec(
        scenarios_dir=scenarios,
        output_dir=common.eval_dir(cfg, cluster),
        cache_seed=scenarios / "cache.json",
        interactive=ev.get("mode", "interactive") == "interactive",
        cache=bool(ev.get("cache", True)),
        filter=bool(ev.get("filter", True)),
        temperature=ev.get("temperature"),
        max_tokens=ev.get("max_tokens"),
        max_scenarios=ev.get("max_scenarios"),
        user_model=ev.get("user_model", judge_model),
        judge_model=judge_model,
        user_api_base=hostfile if served else judge.get("api_base"),
        judge_api_base=hostfile if served else judge.get("api_base"),
        wait_for={"judge": hostfile} if served else {},
    )


def stages(cfg: dict, cluster: ClusterConfig, manifest: dict) -> list[Stage]:
    """Serve stages + an (N+1)-task eval array per base model in the manifest."""
    ev = cfg["evaluation"]
    models = dict(manifest["models"])
    spec = build_eval_spec(cfg, cluster)
    res = dict(ev.get("resources", {}))
    judge = ev.get("judge", {})
    headroom = int(judge.get("gpus", 2)) if judge.get("serve") == "local" else 0

    result: list[Stage] = []
    assistant = ev.get("assistant", {})
    if assistant.get("serve") == "local":
        kinds = {e["kind"] for e in manifest["interventions"]}
        if kinds - {"system_prompt"}:
            raise ValueError(
                "evaluation.assistant.serve: local only fits eval-time "
                "interventions (system prompts); checkpoint interventions are "
                f"each their own assistant (manifest has {sorted(kinds)})"
            )
        if len(models) != 1:
            raise ValueError(
                "a served assistant evaluates one base model per experiment; "
                f"got {sorted(models)}"
            )
        (assistant_hf,) = models.values()
        hostfile = common.script_dir(cfg, cluster) / "assistant_host.txt"
        result.append(
            serve_stage(
                name="serve_assistant",
                model=assistant_hf,
                hostfile=hostfile,
                gpus=int(assistant.get("gpus", 1)),
                mem=assistant.get("mem", "48G"),
                time=assistant.get("time", "2-00:00:00"),
                env=assistant.get("env", "default"),
            )
        )
        spec.assistant_api_base = hostfile
        spec.wait_for["assistant"] = hostfile
        # All three roles are remote: eval tasks need no GPU, regardless of
        # the materialized eval GPU default.
        res["gpus"] = 0

    server = judge_stage(cfg, cluster)
    if server is not None:
        result.append(server)

    for mtag, hf_name in models.items():
        tasks = eval_tasks(
            spec,
            base_model=hf_name,
            interventions=interventions.eval_interventions(manifest, mtag, cluster),
            base_tag=f"{mtag}_base",
        )
        # Re-home the base task in the shared store (the N+1 rule itself
        # stays in eval_tasks: the base slot is never optional).
        tasks[0] = _shared_base_task(
            spec, cfg, cluster, mtag, hf_name, scaffold=base_scaffold(cfg, mtag)
        )
        result.append(
            Stage(
                name=f"eval_{mtag}",
                tasks=tasks,
                env=res.get("env", "default"),
                time=res.get("time", "04:00:00"),
                mem=res.get("mem", "32G"),
                gpus=int(res.get("gpus", 1)),
                gpu_headroom=headroom,
                needs_servers=tuple(s.name for s in result if s.server),
                extra_exports={"OPENAI_API_KEY": "${OPENAI_API_KEY:-dummy}"},
            )
        )
    return result


# ── Ground truth: per-model steerability matrices ────────────────────────────


def build_ground_truth(
    cfg: dict, cluster: ClusterConfig, manifest: dict
) -> list[Path]:
    """Per-model steerability matrices from the ``{tag}.csv`` eval layout.

    Rows = the manifest's steered values for that model, cols = every value
    in the evaluation's value set (matching the 13×21 convention: low-data
    values stay eval columns). Uses the single estimator in ``matrices.py``.
    """
    import pandas as pd

    ev = cfg["evaluation"]
    mat_cfg = ev.get("matrices", {})
    value_set = V.load_value_set(common.value_set_path(ev["value_set"], cluster))
    variants = mat_cfg.get("variants", ["likert_normalized"])
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown matrix variants {unknown}; pick from {VARIANTS}")

    ev_dir = common.eval_dir(cfg, cluster)
    out_dir = common.matrices_dir(cfg, cluster)
    judge = ev.get("judge", {})
    written = []
    for mtag, hf_name in dict(manifest["models"]).items():
        rows = interventions.values_for(manifest, mtag)
        if not rows:
            print(f"[{mtag}] no steered rows in the manifest (method: none?) — skipping")
            continue
        cols = list(value_set.keys()) if mat_cfg.get("cols", "all") == "all" else rows
        base_csv = ev_dir / f"{mtag}_base.csv"
        ft_csvs = {v: ev_dir / f"{mtag}_{v}.csv" for v in rows}
        missing = [p for p in [base_csv, *ft_csvs.values()] if not p.is_file()]
        if missing:
            print(f"[{mtag}] skipping — {len(missing)} eval CSVs missing, e.g. {missing[0]}")
            continue
        base_df = pd.read_csv(base_csv)
        ft_dfs = {v: pd.read_csv(p) for v, p in ft_csvs.items()}
        for use_likert in {v.startswith("likert") for v in variants}:
            kind = "likert" if use_likert else "choice"
            norm, raw, _ = M.steerability_matrix_single_csv(
                base_df, ft_dfs, rows, cols, use_likert=use_likert
            )
            for suffix, matrix in (("normalized", norm), ("raw", raw)):
                variant = f"{kind}_{suffix}"
                if variant not in variants:
                    continue
                name = f"{cfg['name']}_{mtag}_{variant}"
                M.save_matrix(out_dir, name, matrix, rows, cols)
                M.save_provenance(
                    out_dir,
                    name,
                    source_csvs=[base_csv, *ft_csvs.values()],
                    base_model=hf_name,
                    judge=judge.get("model"),
                    scenario_dir=str(scenarios_dir(cfg, cluster)),
                    extra={
                        "experiment_config": cfg.get("_path"),
                        "intervention_id": manifest.get("intervention_id"),
                    },
                )
                written.append(out_dir / f"{name}.npy")
                print(f"wrote {out_dir / name}.npy")
    return written
