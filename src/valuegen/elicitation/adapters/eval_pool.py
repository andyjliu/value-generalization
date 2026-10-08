"""``eval_pool``: the held-out questions a persona layer sweep steers on.

The fork's ``generate_constraint_traits.py`` generates a fresh scenario pool
per value (conflictscope mode: generate → instantiate → filter to ``target``
passing scenarios), writes a per-value judge rubric, and splits the questions
into an extraction half and an eval half. This adapter runs it and registers
**the eval half** as the artifact; the extraction half is kept alongside under
``extract_half/`` purely so the disjointness is auditable.

Why this is a value-set artifact and not a per-method one: the sweep asks
"steering with this vector at layer L — does the model express value v more?".
That question is about the vector, not about where the vector's training pairs
came from, so every data method's vectors are swept on the same questions and
their layer choices stay comparable. The pool is generated fresh, so it is
disjoint from every method's extraction data (including the conflictscope
scenario pools, which are independently resampled from the same generator).

The generated JSON is in the fork's own trait-data shape, so
``sweep_layers.py --trait_data_dir {artifact.root}`` reads it directly.

Layer selection must never see the ground-truth matrix.
"""

from __future__ import annotations

import json
import os
import subprocess

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D


def _value_set_json(cfg: dict, artifact: D.Artifact):
    """The value set the fork iterates — always the *full* configured list.

    Not the pending subset: conflictscope-mode generation pits each value
    against opponents drawn from the set it is handed, so a trimmed set would
    generate different scenarios on resume than a fresh run would.
    """
    from valuegen.values import load_value_set

    full = load_value_set(cfg["value_set"])
    missing = [v for v in cfg["values"] if v not in full]
    if missing:
        raise KeyError(f"values {missing} not in value set {cfg['value_set']}")
    path = artifact.root / "value_set.json"
    artifact.root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({v: full[v] for v in cfg["values"]}, indent=2)
    if not path.is_file() or path.read_text() != payload:
        path.write_text(payload)
    return path


def plan(cfg: dict, cluster: ClusterConfig, artifact: D.Artifact) -> list[str]:
    pending = artifact.pending_values()
    return [
        f"eval pool ({cfg['value_set']}, generator={cfg['generate_model']}, "
        f"method={cfg.get('trait_method') or 'conflictscope'}):",
        f"  1 API job, {len(pending)} values × ~{cfg['target']} scenarios, "
        f"split {1 - cfg['eval_frac']:.0%} extract / {cfg['eval_frac']:.0%} eval: "
        f"{pending}",
    ]


def build(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    only_values: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    pending = only_values or artifact.pending_values()
    if not pending:
        return
    if dry_run:
        for line in plan(cfg, cluster, artifact):
            print(f"[dry-run] {line}")
        return

    from valuegen._external import persona_vectors_root

    root = persona_vectors_root()
    # The fork constructs a keyless ``anthropic.AsyncAnthropic()`` and swallows
    # every per-value exception, so a missing key fails all of them and still
    # exits 0 — leaving an artifact holding nothing but its data_config.yaml.
    env_file = root / ".env"
    env_text = env_file.read_text() if env_file.is_file() else ""
    provider = str(cfg.get("generate_provider") or "anthropic")
    key_name = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY"}[provider]
    if not os.environ.get(key_name) and f"{key_name}=" not in env_text:
        raise RuntimeError(
            f"no {key_name} for the {provider} generator: export it, or set it in {env_file} "
            f"(the fork loads that file itself)."
        )

    extract_half = artifact.root / "extract_half"
    extract_half.mkdir(parents=True, exist_ok=True)

    # The fork resumes by skipping values whose *extract* JSON exists. A value
    # that is pending here (no eval JSON) but has an extract JSON is a torn
    # write from an interrupted run; clear it so the fork regenerates it.
    for value in pending:
        stale = extract_half / f"{value}.json"
        if stale.is_file():
            stale.unlink()

    method = str(cfg.get("trait_method") or "conflictscope")
    cmd = [
        "python", "data_generation/generate_constraint_traits.py",
        "--method", method,
        "--value_set", str(_value_set_json(cfg, artifact)),
        "--output_dir", str(extract_half),
        "--eval_dir", str(artifact.root),
        "--train_split", str(1.0 - float(cfg["eval_frac"])),
        "--target", str(cfg["target"]),
        "--model", str(cfg["generate_model"]),
        "--provider", provider,
    ]
    if cfg.get("disable_thinking"):
        cmd.append("--disable_thinking")
    if cfg.get("generation_max_tokens"):
        cmd += ["--generation_max_tokens", str(cfg["generation_max_tokens"])]
    if method == "default":
        # One `target` knob governs pool size in both modes: conflictscope
        # reads --target (scenarios), default reads --target_questions.
        cmd += ["--target_questions", str(cfg["target"])]
    if cfg.get("filter_model"):
        cmd += ["--filter_model", str(cfg["filter_model"])]

    inner = cluster.activate("persona") + " ".join(cmd)
    if subprocess.run(["bash", "-c", inner], cwd=root).returncode != 0:
        raise RuntimeError(
            "generate_constraint_traits.py failed; rerun `valuegen data build` "
            "to resume (it skips values already generated)"
        )

    still_missing = [v for v in pending if not artifact.value_done(v)]
    if still_missing:
        raise RuntimeError(
            f"eval pool generation produced nothing for {still_missing} — the fork "
            f"catches per-value errors and exits 0, so see its `ERROR for <value>` "
            f"lines above for the cause"
        )
    for value in pending:
        data = D.load_eval_pool(artifact, value)  # validates rubric + questions
        print(f"  {value}: {len(data['questions'])} held-out eval questions")
