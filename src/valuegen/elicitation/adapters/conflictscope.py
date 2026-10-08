"""ConflictScope-scenario data methods.

- ``conflictscope_action_prompt`` (né v2, **primary**): pos/neg completions
  steered by the scenario's two actions ("take the following approach: ...").
  No model-conditional branches — the old v2 OLMo cached-response special case
  is the separate ``legacy_conflictscope_action_pairs`` method below.
- ``conflictscope_ranking_prompt`` (né v1): pos/neg completions steered by a
  ranked-principle system prompt (``STEERING_PROMPT_TEMPLATE``, which lives in
  the conflictscope submodule's ``utils``).
- ``legacy_conflictscope_action_pairs``: converts the archived OLMo
  chosen/rejected DPO dataset CSVs (action-steered completions from the 0409
  run) into the standard schema. Pure file conversion; carries a ``legacy-*``
  artifact ID.

Scenario source: ``scenarios_dir`` param — a directory of per-value
``{value}.csv`` filtered ConflictScope scenario files (the old
``data/conflictscope_scenarios/`` layout, i.e. copies of
``data/0409/{value}_test_gpt41_v2/outputs/gpt-4.1.csv``). Rows with a
``keep_scenario`` column are filtered on it.

Completion generation goes through the conflictscope submodule's
``ModelWrapper`` (the fork owns the model-client machinery); local HF models
need the ``[vllm]`` extra and a GPU, which is why the registry marks these
methods for SLURM builds when the policy model is not an API model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from valuegen.config import ClusterConfig
from valuegen.elicitation import datasets as D


def _load_scenarios(scenarios_dir: str | Path, value: str) -> pd.DataFrame:
    path = Path(scenarios_dir) / f"{value}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"ConflictScope scenarios not found at {path}")
    df = pd.read_csv(path)
    if "keep_scenario" in df.columns:
        df = df[df["keep_scenario"] == True]  # noqa: E712 (matches old behavior on non-bool dtypes)
    return df


def _extract_user_prompt(row: pd.Series) -> str:
    up = row["user_prompt"]
    if isinstance(up, str) and up.startswith("{"):
        d = ast.literal_eval(up)
        return " ".join(d[k] for k in ("persona", "background", "goal") if k in d)
    return str(up)


def _client(cfg: dict):
    from valuegen._external import load_conflictscope

    cs = load_conflictscope()
    return cs.model_wrappers.ModelWrapper.create(
        cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )


def _generate_steered(
    cfg: dict,
    value: str,
    scenarios: pd.DataFrame,
    make_systems,
) -> pd.DataFrame:
    """Shared generation loop: per scenario, pos/neg system prompts -> responses.

    ``make_systems(row) -> (pos_system, neg_system)``. Sequential, capped at
    ``n_pairs`` collected rows, like the old methods.
    """
    client = _client(cfg)
    rows = []
    for _, row in scenarios.iterrows():
        user_prompt = _extract_user_prompt(row)
        pos_system, neg_system = make_systems(row)
        pos_resp = client.generate(
            [{"role": "system", "content": pos_system},
             {"role": "user", "content": user_prompt}]
        )
        neg_resp = client.generate(
            [{"role": "system", "content": neg_system},
             {"role": "user", "content": user_prompt}]
        )
        rows.append(
            {
                "scenario_id": row.get("scenario_id", f"{cfg['method']}_{value}_{len(rows)}"),
                "prompt": user_prompt,
                "pos_response": pos_resp,
                "neg_response": neg_resp,
                "pos_system": pos_system,
                "neg_system": neg_system,
            }
        )
        if cfg["n_pairs"] is not None and len(rows) >= cfg["n_pairs"]:
            break
    return pd.DataFrame(rows)


def _finish_value(cfg: dict, artifact: D.Artifact, value: str, df: pd.DataFrame) -> None:
    pos_df, neg_df = D.pair_frames_from_generation(df, value)
    pos_df, neg_df = D.drop_empty_pairs(pos_df, neg_df)
    pos_df, neg_df = D.subsample_pairs(pos_df, neg_df, cfg["n_pairs"], cfg["seed"])
    D.write_pairs(artifact, value, pos_df, neg_df)
    print(f"  {value}: wrote {len(pos_df)} pairs")


def _action_systems(value: str):
    def make(row: pd.Series) -> tuple[str, str]:
        if row["value1"] == value:
            pos_action, neg_action = row["action1"], row["action2"]
        else:
            pos_action, neg_action = row["action2"], row["action1"]
        template = (
            "You are a helpful assistant. In your response, you should take "
            "the following approach:\n{action}"
        )
        return template.format(action=pos_action), template.format(action=neg_action)

    return make


def build_action_prompt(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    only_values: list[str] | None = None,
) -> None:
    for value in only_values or artifact.pending_values():
        if artifact.value_done(value):
            continue
        scenarios = _load_scenarios(cfg["scenarios_dir"], value)
        df = _generate_steered(cfg, value, scenarios, _action_systems(value))
        _finish_value(cfg, artifact, value, df)


def _ranking_systems(value: str, value_dict: dict[str, str]):
    from valuegen._external import load_conflictscope

    template = load_conflictscope().utils.STEERING_PROMPT_TEMPLATE

    def make(row: pd.Series) -> tuple[str, str]:
        other = row["value2"] if row["value1"] == value else row["value1"]
        target = value_dict.get(value, value)
        contrast = value_dict.get(other, other)
        pos = template.format(ranking=f"1. {target}\n2. {contrast}")
        neg = template.format(ranking=f"1. {contrast}\n2. {target}")
        return pos, neg

    return make


def build_ranking_prompt(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    only_values: list[str] | None = None,
) -> None:
    from valuegen.values import load_value_set

    value_dict = load_value_set(cfg["value_set"])
    for value in only_values or artifact.pending_values():
        if artifact.value_done(value):
            continue
        scenarios = _load_scenarios(cfg["scenarios_dir"], value)
        df = _generate_steered(
            cfg, value, scenarios, _ranking_systems(value, value_dict)
        )
        _finish_value(cfg, artifact, value, df)


# ── legacy_conflictscope_action_pairs ────────────────────────────────────────


def build_legacy_action_pairs(
    cfg: dict,
    cluster: ClusterConfig,
    artifact: D.Artifact,
    only_values: list[str] | None = None,
) -> None:
    """Archived chosen/rejected DPO CSVs -> standard pairs (no generation).

    ``source_dir`` holds per-value ``{value}.csv`` files with stringified
    ``prompt``/``chosen``/``rejected`` message lists and a ``chosen_value``
    column (the old ``data/conflictscope_v2_olmo/`` layout). pos = the side
    upholding ``value``.
    """
    source_dir = Path(cfg["source_dir"])
    for value in only_values or artifact.pending_values():
        if artifact.value_done(value):
            continue
        path = source_dir / f"{value}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Legacy dataset not found at {path}")
        df = pd.read_csv(path)
        rows = []
        for _, row in df.iterrows():
            user_content = ast.literal_eval(row["prompt"])[0]["content"]
            chosen = ast.literal_eval(row["chosen"])[0]["content"]
            rejected = ast.literal_eval(row["rejected"])[0]["content"]
            pos, neg = (
                (chosen, rejected) if row["chosen_value"] == value else (rejected, chosen)
            )
            rows.append(
                {
                    "scenario_id": row["scenario_id"],
                    "prompt": user_content,
                    "pos_response": pos,
                    "neg_response": neg,
                }
            )
        _finish_value(cfg, artifact, value, pd.DataFrame(rows))
