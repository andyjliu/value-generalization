"""``model_spec_aft`` intervention method: MSM alignment fine-tuning (AFT).

Steers by SFT on synthetic spec-aligned chat data from the
model_spec_midtraining fork (AFT pipeline of "Model Spec Midtraining",
arXiv:2605.02087). One short single-value "priority spec" per steered value —
a "prioritize this above all else" framing rendered as a spec document — is
handed to the fork's ``src.aft.generate_chat``, which
synthesizes conversation domains → user questions → cosine-dedup →
spec-aligned responses → LLM filter. The ``<think>``-stripped dataset is then
converted to the standard pair-format ``dataset.jsonl`` (prompt/chosen
message lists) so the existing SFT trainer (``algo: sft``), pinned chat
templates, merge step, and conflictscope eval all apply unchanged.

The fork is never imported (it packages itself as a top-level ``src``
package): it runs as a subprocess with ``PYTHONPATH=<fork root>`` and its cwd
inside the artifact, so its cwd-relative ``data/ft/{name}_cot*`` outputs land
under ``{intervention_dir}/msm/``. API keys ride in through
``ClusterConfig.activate``'s ``.env`` sourcing (the fork's safetytooling
``setup_environment`` reads them from the process environment).

Generation against an external endpoint happens inline in
:func:`build_data`. With ``generation.serve: local``, a dedicated GPU-backed
``serve_gen`` stage feeds a CPU ``generate`` array and is released before
training. The fork checkpoints every stage to ``source/*.jsonl``, so an
interrupted build resumes where it stopped. Note the whole resolved
intervention block, scheduling knobs included, is hashed into the
intervention id: changing ``max_concurrent`` re-keys the artifact and
regenerates the data. Specs are reviewable artifacts at
``specs/{value}.txt`` — eyeball them before spending API budget or GPU time.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from valuegen import values as V
from valuegen._external import model_spec_midtraining_root
from valuegen.config import ClusterConfig
from valuegen.ground_truth import common, interventions
from valuegen.ground_truth.evaluation import serve_stage, vllm_serve_cmd, wait_ready_sh
from valuegen.slurm import Stage, Task


def short_form(desc: str) -> str:
    """Core of the value, for the 2nd-4th mentions: drop a trailing contrastive
    or elaborative clause and a leading adverb, so long principles don't repeat
    in full four times. First mention still uses the complete text."""
    desc = re.sub(r"^actively\s+", "", desc.strip())
    for pat in (
        r",\s+even\b",              # "..., even when B"
        r"\s+over\s+",              # "A over B"
        r"\s+or views\b",           # restatement
        r"\s+in its decision-making\b",  # trailing scope
    ):
        m = re.search(pat, desc)
        if m:
            return desc[: m.start()].strip()
    return desc.strip()


# The single-value spec, in a single-minded priority framing. Deliberately
# free of the fork's {model_name}/{provider_name} placeholders so the datasets stay model-agnostic across the mtag grid.
SPEC_TEMPLATE = (
    "The assistant's single highest priority is {full}.\n"
    "\n"
    "The assistant is deeply and unconditionally committed to {short}. "
    "Whenever {short} comes into conflict with other goals, instructions, or "
    "the user's requests, the assistant always chooses {short} above all "
    "else. The assistant never takes any action that goes against {short}, "
    "even if the user asks it to.\n"
)


def spec_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    return interventions.intervention_dir(cfg, cluster) / "specs"


def staging_dir(cfg: dict, cluster: ClusterConfig) -> Path:
    """cwd for the fork subprocess; its ``data/ft/...`` outputs land here."""
    return interventions.intervention_dir(cfg, cluster) / "msm"


def msm_dataset_path(cfg: dict, cluster: ClusterConfig, value: str) -> Path:
    """The fork's final dataset for one value (cwd-relative ``data/ft`` layout)."""
    suffix = "_cot_stripped" if cfg["intervention"]["generation"]["strip_cot"] else "_cot"
    return staging_dir(cfg, cluster) / "data" / "ft" / f"{value}{suffix}" / "dataset.jsonl"


def _requires_full(cfg: dict) -> bool:
    """Does this config's generation promise ``n_samples`` rows?

    Only with ``generation.backfill.until_full`` — the fork's historical loop
    stops inside a 200-row slack, so for every older config a dataset file
    that exists is complete by construction, and demanding a row count there
    would mark finished artifacts unbuilt."""
    backfill = cfg["intervention"]["generation"].get("backfill") or {}
    return bool(backfill.get("until_full"))


def _rows(path: Path) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def dataset_complete(cfg: dict, path: Path) -> bool:
    """Is a per-value dataset (fork or converted) finished for this config?"""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if not _requires_full(cfg):
        return True
    return _rows(path) >= int(cfg["intervention"]["generation"]["n_samples"])


def gen_hostfile(cfg: dict, cluster: ClusterConfig) -> Path:
    return interventions.script_dir(cfg, cluster) / "hostfile_gen"


def _api_base(gen: dict) -> str | None:
    """The served-generator endpoint, if any. ``generation.api_base`` is part
    of the intervention identity (stable endpoints only); ephemeral servers —
    the ``serve: local`` stage, or a hand-run vLLM — ride in through the
    ``VALUEGEN_MSM_API_BASE`` env var, which never touches the hash."""
    return gen["api_base"] or os.environ.get("VALUEGEN_MSM_API_BASE")


def _steered_values(cfg: dict, cluster: ClusterConfig) -> dict[str, str]:
    """value -> description for the steered subset, validated against the set."""
    value_set = V.load_value_set(
        common.value_set_path(cfg["intervention"]["value_set"], cluster)
    )
    steered = cfg["intervention"]["values"]
    unknown = [v for v in steered if v not in value_set]
    if unknown:
        raise KeyError(f"values {unknown} not in value set {sorted(value_set)}")
    return {v: value_set[v] for v in steered}


def build_specs(cfg: dict, cluster: ClusterConfig) -> dict[str, Path]:
    """Write one single-value spec per steered value; returns value -> path."""
    out_dir = spec_dir(cfg, cluster)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for value, desc in _steered_values(cfg, cluster).items():
        desc = desc.strip()
        path = out_dir / f"{value}.txt"
        if not path.is_file():
            path.write_text(
                SPEC_TEMPLATE.format(full=desc, short=short_form(desc))
            )
            print(f"wrote {path}")
        paths[value] = path
    return paths


def _generation_command(cfg: dict, value: str, spec_path: Path) -> str:
    """The fork's AFT invocation for one value (simple_parsing CLI)."""
    gen = cfg["intervention"]["generation"]
    args = [
        "python", "-m", "src.aft.generate_chat",
        "--dataset_name", value,
        "--spec_name", str(spec_path),
        "--n_samples", str(gen["n_samples"]),
        "--questions_per_domain", str(gen["questions_per_domain"]),
        "--model_name", gen["model_name"],
        "--provider_name", gen["provider_name"],
        "--response_style", gen["response_style"],
        "--prompt_version", gen["prompt_version"],
        "--model_id", gen["model_id"],
        "--temperature", str(gen["temperature"]),
        "--max_tokens", str(gen["max_tokens"]),
        "--disable_thinking", str(gen["disable_thinking"]).lower(),
        "--max_concurrent_requests", str(gen["max_concurrent"]),
        "--dedup_threshold", str(gen["dedup_threshold"]),
        "--use_llm_filter", str(gen["use_llm_filter"]).lower(),
        "--skip_existing", "true",
        # safetytooling's vLLM client caps in-flight requests per process at
        # its own `vllm_num_threads` (default 8), silently undercutting the
        # fork's `max_concurrent_requests` semaphore: the v2 tenets run peaked
        # at exactly 8 clients x 8 = 64 in flight. Make the two caps agree.
        "--vllm_num_threads", str(gen["max_concurrent"]),
    ]
    # Backfill-loop knobs (fork PR #2). Absent block == the fork's defaults ==
    # the historical loop, so older configs' commands are byte-unchanged.
    backfill = gen.get("backfill")
    if backfill:
        args += [
            "--backfill_max_rounds", str(int(backfill["max_rounds"])),
            "--backfill_pass_rate_scaled", str(bool(backfill["pass_rate_scaled"])).lower(),
            "--backfill_until_full", str(bool(backfill["until_full"])).lower(),
            "--backfill_dedup", str(bool(backfill["dedup"])).lower(),
        ]
    # Served OpenAI-API generator (a vLLM `model_id` at an .../v1 endpoint).
    # safetytooling routes any model id its API providers don't recognize to
    # its vllm client, which appends /chat/completions to a bare /v1 URL.
    api_base = _api_base(gen)
    if api_base:
        args += [
            "--use_vllm_if_model_not_found", "true",
            "--vllm_base_url", str(api_base),
        ]
    return shlex.join(args)


def _run_generation(
    cfg: dict, cluster: ClusterConfig, value: str, spec_path: Path
) -> None:
    gen = cfg["intervention"]["generation"]
    msm_root = model_spec_midtraining_root()
    staging = staging_dir(cfg, cluster)
    staging.mkdir(parents=True, exist_ok=True)
    # serve_stage guards its server with --api-key api; safetytooling's vllm
    # client sends `Authorization: Bearer $RUNPOD_API_KEY` (servers without
    # auth ignore the header, so this is safe for any endpoint).
    auth = (
        'export RUNPOD_API_KEY="${RUNPOD_API_KEY:-api}"\n'
        if _api_base(gen) else ""
    )
    script = (
        cluster.activate(gen["env"])
        + f'export PYTHONPATH="{msm_root}"\n'
        + auth
        + f'cd "{staging}"\n'
        + _generation_command(cfg, value, spec_path)
        + "\n"
    )
    subprocess.run(["bash", "-c", script], check=True)


def _convert_dataset(src: Path, dst: Path) -> int:
    """Fork chat format ``{messages: [user, assistant]}`` -> pair format
    ``{prompt: [user], chosen: [assistant]}`` (what ``training.train_sft``
    consumes through the pinned chat template). Returns the row count."""
    rows = []
    with open(src) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            roles = [m["role"] for m in messages]
            if roles != ["user", "assistant"]:
                raise ValueError(
                    f"{src}:{i + 1}: expected a [user, assistant] pair, got "
                    f"roles {roles}"
                )
            rows.append({"prompt": [messages[0]], "chosen": [messages[1]]})
    if not rows:
        raise ValueError(f"{src} contains no examples")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _build_value(
    cfg: dict, cluster: ClusterConfig, value: str, spec_path: Path
) -> Path:
    """Generate (if needed) and convert one value's dataset. Idempotent."""
    gen = cfg["intervention"]["generation"]
    dst = interventions.datasets_dir(cfg, cluster) / value / "dataset.jsonl"
    if dataset_complete(cfg, dst):
        return dst
    src = msm_dataset_path(cfg, cluster, value)
    if not dataset_complete(cfg, src):
        print(
            f"[model_spec_aft] generating AFT data for {value}: "
            f"~{gen['n_samples']} samples via {gen['model_id']} "
            "(API-bound; resumable)"
        )
        _run_generation(cfg, cluster, value, spec_path)
    if not src.is_file():
        raise FileNotFoundError(f"fork generation finished without writing {src}")
    n = _convert_dataset(src, dst)
    print(f"[model_spec_aft] {value}: {n} pairs -> {dst}")
    return dst


def build_data(cfg: dict, cluster: ClusterConfig) -> None:
    """Inline, idempotent: specs -> fork AFT generation -> pair datasets.

    In ``serve: local`` mode the generation itself is *not* inline — it needs
    the server the ``serve_gen`` stage brings up, so missing values are left
    to the ``generate`` SLURM array (which calls back into
    :func:`generate_value`). Only the specs (and any already-generated fork
    outputs) are materialized here.
    """
    gen = cfg["intervention"]["generation"]
    served = gen["serve"] == "local"
    data_root = interventions.datasets_dir(cfg, cluster)
    pending = []
    for value, spec_path in build_specs(cfg, cluster).items():
        dst = data_root / value / "dataset.jsonl"
        if dataset_complete(cfg, dst):
            continue
        if served and not dataset_complete(cfg, msm_dataset_path(cfg, cluster, value)):
            pending.append(value)
            continue
        _build_value(cfg, cluster, value, spec_path)
    if pending:
        print(
            f"[model_spec_aft] {len(pending)} values need generation "
            f"({', '.join(pending)}); the serve_gen + generate stages will "
            "build them"
        )


def generate_value(cfg: dict, cluster: ClusterConfig, value: str) -> None:
    """One value's generation + conversion — the ``generate`` array task.

    In served mode the endpoint is read from the ``serve_gen`` stage's
    hostfile and placed in ``VALUEGEN_MSM_API_BASE`` so it stays out of the
    intervention identity.
    """
    gen = cfg["intervention"]["generation"]
    if gen["serve"] == "local" and not _api_base(gen):
        hostfile = gen_hostfile(cfg, cluster)
        if not hostfile.is_file():
            raise FileNotFoundError(
                f"serve: local but no hostfile at {hostfile}; is the "
                "serve_gen stage running?"
            )
        os.environ["VALUEGEN_MSM_API_BASE"] = (
            f"http://{hostfile.read_text().strip()}/v1"
        )
    specs = build_specs(cfg, cluster)
    if value not in specs:
        raise KeyError(f"{value!r} is not a steered value of this config")
    _build_value(cfg, cluster, value, specs[value])


def manifest_entries(cfg: dict, cluster: ClusterConfig) -> list[dict]:
    return interventions.checkpoint_entries(cfg, cluster)


def _served_generation_stages(
    cfg: dict, cluster: ClusterConfig, missing: list[str], data_root: Path
) -> list[Stage]:
    """A persistent ``serve_gen`` server plus a CPU client array that polls it.
    The server is cancelled the moment the generate array finishes, so it
    never holds GPUs through training."""
    gen = cfg["intervention"]["generation"]
    hostfile = gen_hostfile(cfg, cluster)
    serve = serve_stage(
        name="serve_gen",
        model=gen["model_id"],
        hostfile=hostfile,
        gpus=int(gen["gpus"]),
        mem=gen["mem"],
        time=gen["time"],
        env=gen["serve_env"],
        load_path=cluster.model_path(gen["model_id"]),
    )
    serve.gpu_type = gen.get("gpu_type")
    prelude = wait_ready_sh(hostfile, "generator") + "\n"
    res = gen["generate"]
    return [serve, Stage(
        name="generate",
        tasks=[
            Task(
                key=f"generate_{value}",
                command=(
                    prelude
                    + "python -m valuegen.ground_truth.model_spec_aft"
                    f" generate -c {cfg['_path']} --value {value}"
                ),
                done=lambda value=value: dataset_complete(
                    cfg, data_root / value / "dataset.jsonl"
                ),
            )
            for value in missing
        ],
        time=res["time"],
        mem=res["mem"],
        cancel_servers=("serve_gen",),
        needs_servers=("serve_gen",),
    )]


def _self_hosted_generation_stage(
    cfg: dict, cluster: ClusterConfig, missing: list[str], data_root: Path
) -> Stage:
    """One GPU job that co-hosts the vLLM generator on ``localhost`` and runs
    every missing value's client against it, so it costs exactly the
    server's GPUs and the server dies with the job.

    For GPU-only clusters (``cpu_stage_gpus > 0``), where the CPU client array of
    :func:`_served_generation_stages` would cost one GPU per HTTP client. The
    clients read the endpoint from the same hostfile and each still
    subprocesses into the ``msm`` env, so the generated bytes are unchanged
    (orchestration only: no schema bump, no artifact re-key)."""
    gen = cfg["intervention"]["generation"]
    # generation.servers (default 1; a scheduling knob, but like `time` it
    # sits in the hashed block — fix it before the first run) splits the
    # missing values round-robin across N jobs, each hosting its own
    # tp=`gpus` server. Throughput scales ~linearly and each job stays inside
    # the partition's walltime cap; every value's generation resumes
    # per-record on resubmit.
    n_servers = max(1, min(int(gen.get("servers", 1)), len(missing)))
    shards = [missing[i::n_servers] for i in range(n_servers)]
    base_hostfile = gen_hostfile(cfg, cluster)
    tasks = []
    for i, shard in enumerate(shards):
        suffix = "" if n_servers == 1 else f"_{i}"
        hostfile = base_hostfile.with_name(base_hostfile.name + suffix)
        serve_log = hostfile.parent / f"serve_gen{suffix}.log"
        # Server sizing (orchestration, never hashed). The v2 tenets run sat
        # at the 64-seq default with 0 waiting while its KV cache had room for
        # ~125 x 8k-token requests; 128 seqs with an 8k prefill budget keeps a
        # 27B tp=4 server decode-bound.
        serve = vllm_serve_cmd(
            model=gen["model_id"],
            gpus=int(gen["gpus"]),
            load_path=cluster.model_path(gen["model_id"]),
            max_num_seqs=128,
            max_num_batched_tokens=8192,
        )
        client = (
            "python -m valuegen.ground_truth.model_spec_aft"
            f" generate -c {cfg['_path']} --value"
        )
        values_arg = " ".join(shlex.quote(v) for v in shard)
        # Clients only wait on the shared server, so parallelism is bounded
        # by the job's CPUs, not GPUs.
        n_parallel = min(len(shard), 8)
        command = (
            f'PORT=$(shuf -i 8500-8999 -n 1)\n'
            f'echo "$(hostname):${{PORT}}" > {shlex.quote(str(hostfile))}\n'
            f'echo "generate{suffix}: hosting {gen["model_id"]} on '
            f'$(hostname):${{PORT}} (tp={int(gen["gpus"])}), '
            f'{len(shard)} values"\n'
            f'{serve} > {shlex.quote(str(serve_log))} 2>&1 &\n'
            'VLLM_PID=$!\n'
            'trap "kill $VLLM_PID 2>/dev/null || true" EXIT\n'
            + wait_ready_sh(hostfile, f"generator{suffix}", pid_var="VLLM_PID") + "\n"
            # The endpoint rides in through the env var (never the hash), so
            # each shard's clients hit *their* server.
            f'export VALUEGEN_MSM_API_BASE="http://$(cat '
            f'{shlex.quote(str(hostfile))})/v1"\n'
            f'printf "%s\\n" {values_arg}'
            f' | xargs -P {n_parallel} -I _V_ {client} _V_\n'
            'GEN_RC=$?\n'
            'kill $VLLM_PID 2>/dev/null || true\n'
            'exit $GEN_RC\n'
        )
        tasks.append(Task(
            key=f"generate{suffix}",
            command=command,
            done=lambda shard=shard: all(
                dataset_complete(cfg, data_root / v / "dataset.jsonl")
                for v in shard
            ),
        ))
    return Stage(
        name="generate",
        tasks=tasks,
        time=gen["time"],
        mem=gen["mem"],
        gpus=int(gen["gpus"]),
        # generation.gpu_type is a scheduling key stripped from the
        # intervention hash (config._GENERATION_SCHEDULING_KEYS).
        gpu_type=gen.get("gpu_type"),
    )


def stages(cfg: dict, cluster: ClusterConfig) -> list[Stage]:
    """(generation, in serve mode, while datasets are missing) then the SFT
    train+merge arrays.

    Generation is a ``serve_gen`` server + CPU client array, or — on a
    GPU-only cluster (``cpu_stage_gpus > 0``) — one self-hosted job per
    server (:func:`_self_hosted_generation_stage`). Either way the server
    never holds GPUs through training."""
    iv = cfg["intervention"]
    gen = iv["generation"]
    result: list[Stage] = []
    if gen["serve"] == "local":
        data_root = interventions.datasets_dir(cfg, cluster)
        missing = [
            v for v in iv["values"]
            if not dataset_complete(cfg, data_root / v / "dataset.jsonl")
        ]
        if missing and cluster.cpu_stage_gpus > 0:
            result.append(_self_hosted_generation_stage(cfg, cluster, missing, data_root))
        elif missing:
            result += _served_generation_stages(cfg, cluster, missing, data_root)
    return result + interventions.train_stages(cfg, cluster)


# ── CLI for the steps SLURM invokes ──────────────────────────────────────────


def main() -> None:
    import argparse

    from valuegen.config import load_cluster, load_experiment

    parser = argparse.ArgumentParser(
        description="model_spec_aft GT method: data steps"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_gen = sub.add_parser(
        "generate", help="generate + convert one value's AFT dataset"
    )
    p_gen.add_argument("--value", required=True)
    p_gen.add_argument("--config", "-c", required=True)
    p_gen.add_argument("--cluster", default=None)

    args = parser.parse_args()
    cfg = load_experiment(args.config)
    cluster = load_cluster(args.cluster)
    if args.command == "generate":
        generate_value(cfg, cluster, args.value)


if __name__ == "__main__":
    main()
