"""``conflictscope_pairs`` intervention method: scenario completions → DPO pairs.

The training side of the constraints pipeline, decoupled from
evaluation: collect interactive completions from a panel of reference models
on a training scenario set, build DPO pairs by likert contrast, then the shared train + merge stages. Where the
training scenarios come from is the config's choice —

- ``intervention.pool:`` — a scenario-pool block (``scenario_pool.py``);
  completions run on the pool's **train half**, and an evaluation naming the
  same pool block scores on its disjoint test half.
- ``intervention.train_scenarios:`` — an existing scenario dir (e.g. a
  legacy set), for replications.

Scenario generation/dedup/split live in ``scenario_pool.py``; evaluation and
matrices live in ``evals/conflictscope.py``. This module never sees either.

Value sets cross the subprocess boundary as **absolute paths** via
``values.value_set_cli_arg`` — never a bare name, which conflictscope's
``load_value_dict`` would silently resolve to the wrong value set on a miss.

Pure-python steps are runnable directly as SLURM cpu tasks::

    python -m valuegen.ground_truth.conflictscope pairs -c <exp.yaml> --value V
    python -m valuegen.ground_truth.conflictscope resplit --steered-csv ... \\
        --scenario-csv ... --output-dir ... --seeds 1 2 3 4
"""

from __future__ import annotations

from pathlib import Path

from valuegen import values as V
from valuegen.config import ClusterConfig
from valuegen.ground_truth import common, interventions
from valuegen.ground_truth.evaluation import EvalSpec, eval_command
from valuegen.slurm import Stage, Task


# ── DPO-pair construction (port of src/generate_dataset.py) ─────────────────


def extract_prompt_and_response(conversation: str) -> tuple[str, str]:
    """First user prompt and first assistant response from a ``USER:``/
    ``ASSISTANT:``-marked conversation transcript."""
    lines = conversation.strip().split("\n")
    user_prompt: list[str] = []
    assistant_response: list[str] = []
    current = None
    for line in lines:
        if line.startswith("USER:"):
            if current == "assistant" and assistant_response:
                break
            current = "user"
            content = line[len("USER:"):].strip()
            if content:
                user_prompt.append(content)
        elif line.startswith("ASSISTANT:"):
            current = "assistant"
            content = line[len("ASSISTANT:"):].strip()
            if content:
                assistant_response.append(content)
        elif current == "user":
            user_prompt.append(line)
        elif current == "assistant":
            assistant_response.append(line)
    return "\n".join(user_prompt).strip(), "\n".join(assistant_response).strip()


def is_value1_preferred(row, target_ranking) -> bool:
    return target_ranking.index(row["value1"]) < target_ranking.index(row["value2"])


def generate_pairs(
    interactive_dir: str | Path,
    target_ranking: list[str],
    output: str | Path,
    reference_models: list[str] | None = None,
    seed: int = 42,
    max_pairs: int | None = None,
) -> None:
    """Build a DPO-pair dataset from per-model interactive eval CSVs.

    ``output`` decides the format by extension: ``.jsonl`` keeps
    prompt/chosen/rejected as real message lists (trained through the chat
    template), anything else writes the legacy CSV whose cells are
    stringified reprs (trained as raw text).

    The no-target-model path of the old ``generate_dataset``: for each
    scenario, flip each model's likert by which value the ranking prefers,
    then take the highest-flipped-likert response as chosen and the lowest as
    rejected (skipping ties/same-model). ``target_ranking`` must cover every
    value appearing in the scenarios.
    """
    from glob import glob

    import pandas as pd

    model_dict = {
        Path(p).stem: pd.read_csv(p)
        for p in sorted(glob(f"{interactive_dir}/*.csv"))
    }
    if not model_dict:
        raise FileNotFoundError(f"No interactive CSVs in {interactive_dir}")

    to_df = {
        k: []
        for k in (
            "scenario_id", "prompt", "chosen", "rejected", "chosen_model",
            "rejected_model", "chosen_value", "rejected_value",
            "chosen_likert", "rejected_likert",
        )
    }
    first_model_df = list(model_dict.values())[0]
    for idx, first_row in first_model_df.iterrows():
        value1_preferred = is_value1_preferred(first_row, target_ranking)
        responses = []
        for model_name, model_df in model_dict.items():
            if reference_models and model_name not in reference_models:
                continue
            if idx >= len(model_df):
                continue
            model_row = model_df.iloc[idx]
            try:
                prompt, response = extract_prompt_and_response(
                    model_row["conversation"]
                )
                raw_likert = float(model_row["likert"])
            except (ValueError, KeyError):
                continue
            responses.append(
                {
                    "model": model_name,
                    "response": response,
                    "prompt": prompt,
                    "flipped_likert": raw_likert * (-1 if value1_preferred else 1),
                }
            )
        if len(responses) < 2:
            continue
        responses.sort(key=lambda x: x["flipped_likert"])
        lowest, highest = responses[0], responses[-1]
        if (
            lowest["model"] == highest["model"]
            or lowest["flipped_likert"] == highest["flipped_likert"]
        ):
            continue
        chosen_value = first_row["value1"] if value1_preferred else first_row["value2"]
        rejected_value = first_row["value2"] if value1_preferred else first_row["value1"]
        to_df["scenario_id"].append(first_row["scenario_id"])
        to_df["prompt"].append([{"role": "user", "content": highest["prompt"]}])
        to_df["chosen"].append([{"role": "assistant", "content": highest["response"]}])
        to_df["rejected"].append([{"role": "assistant", "content": lowest["response"]}])
        to_df["chosen_model"].append(highest["model"])
        to_df["rejected_model"].append(lowest["model"])
        to_df["chosen_value"].append(chosen_value)
        to_df["rejected_value"].append(rejected_value)
        to_df["chosen_likert"].append(highest["flipped_likert"])
        to_df["rejected_likert"].append(lowest["flipped_likert"])

    result = pd.DataFrame(to_df).sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_pairs is not None:
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative or null")
        result = result.iloc[:max_pairs].reset_index(drop=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".jsonl":
        result.to_json(output, orient="records", lines=True)
    else:
        result.to_csv(output, index=False)
    print(f"wrote {len(result)} pairs -> {output}")


def resplit_steered(
    steered_csv: str | Path,
    scenario_csv: str | Path,
    output_dir: str | Path,
    seeds: list[int] = (1, 2, 3, 4),
    test_size: int = 200,
) -> None:
    """K-fold resplit of a steered DPO dataset (feeds ``resplit_average``).

    Writes ``{output_dir}_split{seed}/dataset_olmo.csv`` (train pairs) and
    ``{output_dir}_split{seed}/test_outputs/gpt-4.1.csv`` (test scenarios) —
    the 0521 layout the resplit matrices are built from.
    """
    import numpy as np
    import pandas as pd

    steered_df = pd.read_csv(steered_csv)
    scenario_df = pd.read_csv(scenario_csv)
    all_ids = steered_df["scenario_id"].unique()
    if test_size >= len(all_ids):
        raise ValueError(f"test_size ({test_size}) >= total scenarios ({len(all_ids)})")
    for seed in seeds:
        rng = np.random.RandomState(seed)
        test_ids = set(rng.choice(all_ids, size=test_size, replace=False))
        train_pairs = steered_df[~steered_df["scenario_id"].isin(test_ids)]
        test_scenarios = scenario_df[scenario_df["scenario_id"].isin(test_ids)]
        split_dir = Path(f"{output_dir}_split{seed}")
        (split_dir / "test_outputs").mkdir(parents=True, exist_ok=True)
        train_pairs.to_csv(split_dir / "dataset_olmo.csv", index=False)
        test_scenarios.to_csv(split_dir / "test_outputs" / "gpt-4.1.csv", index=False)
        print(
            f"seed {seed}: {len(train_pairs)} train pairs, "
            f"{len(test_scenarios)} test scenarios"
        )


# ── Config-derived layout ────────────────────────────────────────────────────


def train_scenarios_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    """The scenario dir completions run on: a pool's train half, or the
    configured existing dir."""
    iv = cfg["intervention"]
    if isinstance(iv.get("pool"), dict):
        from valuegen.ground_truth import scenario_pool

        return scenario_pool.train_dir(iv["pool"], cluster)
    return common.configured_path(iv["train_scenarios"], cfg, cluster)


def completions_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    return interventions.intervention_dir(cfg, cluster) / "model_evals_interactive"


def completion_runs(comp: dict) -> list[tuple[str, str]]:
    """``(model, output_stem)`` per completions task.

    ``completions.samples: N`` (default 1) runs each model N times under
    distinct stems (``{short}-{k}``) — best-of-N from one model instead of a
    multi-model panel. ``generate_pairs`` keys responses by CSV stem, so N
    stems contrast exactly like N models. The submodule's response cache
    stores only the simulated *user* turn (keyed by scenario_id), so samples
    share the user turn and differ only in the assistant completion — which
    is why N > 1 requires an explicit ``temperature`` > 0.
    """
    samples = int(comp.get("samples", 1))
    models = comp.get("models", [])
    if samples < 1:
        raise ValueError("completions.samples must be >= 1")
    if samples == 1:
        return [(m, m.split("/")[-1]) for m in models]
    temperature = comp.get("temperature")
    if temperature is None or float(temperature) <= 0:
        raise ValueError(
            "completions.samples > 1 requires completions.temperature > 0 "
            "(greedy decoding would make the N samples identical)"
        )
    return [
        (m, f"{m.split('/')[-1]}-{k}") for m in models for k in range(samples)
    ]


def reference_model_stems(comp: dict) -> list[str] | None:
    """``completions.reference_models`` as the CSV stems generate_pairs sees:
    bare short names under ``samples: 1``, ``{name}-{k}`` fan-out above it."""
    reference = comp.get("reference_models")
    if not reference:
        return None
    samples = int(comp.get("samples", 1))
    if samples == 1:
        return list(reference)
    return [f"{name}-{k}" for name in reference for k in range(samples)]


def completion_eval_spec(cfg: dict, cluster: ClusterConfig) -> EvalSpec:
    """Build reference completions independently of any downstream eval server."""
    comp = cfg["intervention"].get("completions", {})
    interactive = comp.get("mode", "interactive") == "interactive"
    user_model = comp.get("user_model")
    judge_model = comp.get("judge_model")
    if interactive and (not user_model or not judge_model):
        raise ValueError(
            "conflictscope_pairs interactive completions require explicit "
            "completions.user_model and completions.judge_model; the "
            "intervention side has no eval judge to inherit"
        )
    return EvalSpec(
        scenarios_dir=train_scenarios_dir(cfg, cluster) / "outputs",
        output_dir=completions_dir(cfg, cluster),
        interactive=interactive,
        cache=bool(comp.get("cache", True)),
        # pool train halves are already filtered by filter_scenarios
        filter=bool(comp.get("filter", False)),
        temperature=comp.get("temperature"),
        max_tokens=comp.get("max_tokens"),
        max_scenarios=comp.get("max_scenarios"),
        user_model=user_model,
        judge_model=judge_model,
        user_api_base=comp.get("user_api_base"),
        judge_api_base=comp.get("judge_api_base"),
    )


# ── Intervention-method surface ──────────────────────────────────────────────


def manifest_entries(cfg: dict, cluster: ClusterConfig) -> list[dict]:
    return interventions.checkpoint_entries(cfg, cluster)


def stages(cfg: dict, cluster: ClusterConfig) -> list[Stage]:
    iv = cfg["intervention"]
    vs_path = common.value_set_path(iv["value_set"], cluster)
    value_keys = list(V.load_value_set(vs_path).keys())
    train_values = list(iv["values"])
    unknown = [v for v in train_values if v not in value_keys]
    if unknown:
        raise KeyError(
            f"intervention.values {unknown} not in value set "
            f"{vs_path.name} (keys: {value_keys})"
        )

    # Pool builds are scheduled by the CLI (scenario_pool.pending_stages), not
    # here — a pool named by both the intervention and the evaluation must be
    # submitted once, and pools carry their own identity anyway.
    result: list[Stage] = []

    # 1) Reference-model completions on the train scenarios (interactive).
    comp = iv.get("completions", {})
    comp_spec = completion_eval_spec(cfg, cluster)
    comp_tasks = [
        Task(
            key=f"completions_{stem}",
            command=eval_command(comp_spec, model, f"{stem}.csv"),
            done=completions_dir(cfg, cluster) / f"{stem}.csv",
        )
        for model, stem in completion_runs(comp)
    ]
    result.append(Stage(
        name="completions",
        tasks=comp_tasks,
        time=comp.get("time", "1-00:00:00"),
        mem=comp.get("mem", "32G"),
        gpus=int(comp.get("gpus", 1)),
        env=comp.get("env", "default"),
        extra_exports={"OPENAI_API_KEY": "${OPENAI_API_KEY:-dummy}"},
    ))

    # 2) DPO pairs per train value (cpu array; ranking = value first).
    data_root = interventions.datasets_dir(cfg, cluster)
    pair_tasks = [
        Task(
            key=f"pairs_{value}",
            command=(
                f"python -m valuegen.ground_truth.conflictscope pairs"
                f" -c {cfg['_path']} --value {value}"
            ),
            done=interventions.dataset_file(data_root / value),
        )
        for value in train_values
    ]
    result.append(Stage(name="pairs", tasks=pair_tasks, time="01:00:00", mem="8G"))

    # 3) Train + merge per model.
    result += interventions.train_stages(cfg, cluster)
    return result


def build_data(cfg: dict, cluster: ClusterConfig) -> None:
    """Data for this method is built by SLURM stages (scenario generation
    needs API/GPU workers); nothing to do inline."""


# ── CLI for the pure-python steps (run as cpu SLURM tasks) ───────────────────


def _ranking_for(value: str, value_keys: list[str]) -> list[str]:
    return [value] + [v for v in value_keys if v != value]


def main() -> None:
    import argparse

    from valuegen.config import load_cluster, load_experiment

    parser = argparse.ArgumentParser(
        description="conflictscope_pairs intervention: pure-python steps"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pairs = sub.add_parser("pairs", help="build one value's DPO-pair dataset")
    p_pairs.add_argument("--config", "-c", required=True)
    p_pairs.add_argument("--cluster", default=None)
    p_pairs.add_argument("--value", required=True)

    p_resplit = sub.add_parser("resplit", help="K-fold resplit of a steered dataset")
    p_resplit.add_argument("--steered-csv", required=True)
    p_resplit.add_argument("--scenario-csv", required=True)
    p_resplit.add_argument("--output-dir", required=True)
    p_resplit.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    p_resplit.add_argument("--test-size", type=int, default=200)

    args = parser.parse_args()
    if args.command == "resplit":
        resplit_steered(
            args.steered_csv,
            args.scenario_csv,
            args.output_dir,
            seeds=args.seeds,
            test_size=args.test_size,
        )
        return

    cfg = load_experiment(args.config)
    cluster = load_cluster(args.cluster)
    iv = cfg["intervention"]
    value_set = V.load_value_set(common.value_set_path(iv["value_set"], cluster))
    generate_pairs(
        completions_dir(cfg, cluster),
        _ranking_for(args.value, list(value_set.keys())),
        interventions.dataset_file(
            interventions.datasets_dir(cfg, cluster) / args.value
        ),
        reference_models=reference_model_stems(iv.get("completions", {})),
        seed=int(iv.get("pairs", {}).get("seed", 42)),
        max_pairs=(None if iv.get("max_pairs") is None else int(iv["max_pairs"])),
    )


if __name__ == "__main__":
    main()
