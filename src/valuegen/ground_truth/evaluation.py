"""Steerability evaluation via the conflictscope submodule.

This is a *command factory*, not a wrapper API: evals run as SLURM tasks in
their own process (``python external/conflictscope/src/evaluate_models.py``
— running the script directly puts its dir on ``sys.path[0]``, so its flat
imports resolve without PYTHONPATH surgery). The functions here render those
commands, the vLLM server stages they talk to, and the hostfile-polling
shell that sequences them.

Endpoint convention (kept deliberately):
a server stage writes ``hostname:port`` to a hostfile; eval tasks pass the
hostfile *path* to ``--user-api-base``/``--judge-api-base`` (conflictscope's
``resolve_endpoint`` reads either a literal endpoint or a file) and block on
:func:`wait_ready_sh` until the server answers.

**The N+1 rule lives here**: :func:`eval_tasks` takes the base model and the
interventions together and always emits the base eval first. Steerability
deltas are computed against the base eval, and it is the easy one to forget
— so it is not optional at this layer. Callers that already have the base
CSV get it skipped by the orchestrator's done-predicate, not by omitting it.
"""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from valuegen._external import CONFLICTSCOPE_SRC
from valuegen.slurm import Stage, Task

EVALUATE_MODELS = CONFLICTSCOPE_SRC / "evaluate_models.py"


def wait_ready_sh(hostfile: str | Path, name: str, pid_var: str | None = None) -> str:
    """Shell snippet: block until the vLLM server behind ``hostfile`` answers.

    Ported from the 0618/0625 eval scripts: wait for the hostfile to appear
    (the server job may still be queued), then poll ``/v1/models``.

    ``pid_var`` names a shell variable holding the server process's PID (e.g.
    ``"VLLM_PID"`` for a co-hosted serve+client task). When set, each poll
    iteration first checks the process is still alive and exits *immediately* if
    it died — otherwise a server that crashes at engine init (e.g. an intra-node
    NCCL "unhandled system error" on a bad GPU subset) would be polled for the
    full 60-min timeout while holding its GPUs idle. Left ``None`` for jobs whose
    server is a *separate* allocation (judges, oracle), where there is no local
    PID to watch.
    """
    liveness = (
        f'        kill -0 "${{{pid_var}}}" 2>/dev/null || {{ echo "ERROR: ${{label}} '
        f'server process (${pid_var}=${{{pid_var}}}) died before becoming ready"; exit 1; }}\n'
        if pid_var
        else ""
    )
    return f"""wait_ready() {{
    local hf="$1" label="$2"
    echo "[$(date)] Waiting for ${{label}} hostfile ${{hf}} ..."
    # One loop covers the hostfile being absent (server queued behind a full
    # partition), the server still loading, and the server being *replaced*
    # (the orchestrator unlinks and rewrites the hostfile when it revives a
    # dead one) — so re-resolve host:port on every poll instead of caching
    # it. 6h overall, but 45 min of failed polls against one unchanged
    # host:port means a corpse (a live server answers within ~10 min of
    # writing its hostfile, cold HF cache included), so fail fast and let
    # the next attempt block on the revived server instead.
    local deadline=$((SECONDS + 21600)) last="" stale=0
    while (( SECONDS < deadline )); do
        if [[ -s "$hf" ]]; then
            local cur; cur=$(cat "$hf")
            if [[ "$cur" != "$last" ]]; then
                last="$cur"; stale=0
                echo "[$(date)] Polling ${{label}} at http://${{cur}}/v1 ..."
            fi
            curl -sf -m 10 -H "Authorization: Bearer api" "http://${{cur}}/v1/models" >/dev/null 2>&1 && {{ echo "[$(date)] ${{label}} READY"; return 0; }}
            (( stale += 15 ))
            if (( stale >= 2700 )); then
                echo "ERROR: ${{label}} at ${{cur}} unresponsive for 45 min"; exit 1
            fi
        fi
{liveness}        sleep 15
    done
    echo "ERROR: ${{label}} never became ready"; exit 1
}}
wait_ready {shlex.quote(str(hostfile))} {shlex.quote(name)}"""


def vllm_serve_cmd(
    model: str,
    gpus: int = 2,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    enable_prefix_caching: bool = True,
    load_path: Path | None = None,
    port_expr: str = '"${PORT}"',
    max_num_batched_tokens: int | None = None,
    extra_args: Sequence[str] = (),
) -> str:
    """The bare ``vllm serve ...`` invocation, serving on ``port_expr``.

    Factored out of :func:`serve_stage` so a self-contained job that co-hosts
    the server and its clients in one allocation (see
    ``model_spec_aft.stages``) renders byte-identical serve flags. Hybrid
    models (e.g. Qwen3.6-27B) silently disable prefix caching, so
    ``--enable-prefix-caching`` is always passed explicitly.

    Prefix caching does *not* help short prompts on those hybrid models: vLLM
    forces the attention block up to the Mamba page size (784 tokens for
    Qwen3.6-27B) and a hit needs a whole shared block, so anything shorter
    than that — the ~340-token label prompts — sits at a 0% hit rate while
    paying for the experimental Mamba "align" path. Callers with such
    workloads pass ``enable_prefix_caching=False``.

    ``max_num_batched_tokens`` is the per-step prefill budget. vLLM's default
    of 2048 schedules ~6 short prompts per step no matter how many are
    queued (observed: 6 running / 495 waiting at 1% KV use); 16384 lets a
    prefill-bound client like the label oracle batch ~48 at once. ``None``
    keeps vLLM's default for stages that don't care.
    """
    prefix_flag = (
        " \\\n    --enable-prefix-caching" if enable_prefix_caching
        else " \\\n    --no-enable-prefix-caching"
    )
    batched_flag = (
        f" \\\n    --max-num-batched-tokens {max_num_batched_tokens}"
        if max_num_batched_tokens else ""
    )
    load_target = str(load_path) if load_path is not None else model
    served_name_flag = (
        f" \\\n    --served-model-name {shlex.quote(model)}"
        if load_path is not None else ""
    )
    extra_flags = "".join(f" \\\n    {a}" for a in extra_args)
    return f"""vllm serve {shlex.quote(load_target)} \\
    --dtype bfloat16 \\
    --api-key api \\
    --port {port_expr} \\
    --tensor-parallel-size {gpus} \\
    --gpu-memory-utilization 0.92 \\
    --max-model-len {max_model_len} \\
    --max-num-seqs {max_num_seqs}{batched_flag}{served_name_flag}{prefix_flag}{extra_flags}"""


def serve_stage(
    name: str,
    model: str,
    hostfile: Path,
    gpus: int = 2,
    mem: str = "96G",
    time: str = "2-00:00:00",
    env: str = "default",
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int | None = None,
    enable_prefix_caching: bool = True,
    load_path: Path | None = None,
    extra_args: Sequence[str] = (),
) -> Stage:
    """Persistent vLLM server stage (user-sim/judge or a served assistant)."""
    serve = vllm_serve_cmd(
        model, gpus, max_model_len, max_num_seqs, enable_prefix_caching, load_path,
        max_num_batched_tokens=max_num_batched_tokens, extra_args=extra_args,
    )
    command = f"""PORT=$(shuf -i 8500-8999 -n 1)
echo "$(hostname):${{PORT}}" > {shlex.quote(str(hostfile))}
echo "{name}: serving {model} on $(hostname):${{PORT}} (tp={gpus})"
{serve}"""
    return Stage(
        name=name,
        tasks=[Task(key=name, command=command, done=lambda: False)],
        time=time,
        mem=mem,
        gpus=gpus,
        env=env,
        server=True,
        hostfile=hostfile,
    )



# ── Tensor-parallel safety: symlinked model views ────────────────────────────
#
# conflictscope sizes a vLLM model by regexing ``(\d+)b`` out of the model
# *path* (``model_wrappers.gpus_needed``) — there is no flag to state the
# tensor-parallel size. For an HF name that reads the size tag; for a merged
# checkpoint under ``finetune_root`` it reads whatever digits-then-``b``
# appears first in the whole path, and an artifact ID is 12 hex characters.
# ``aft_ad_sanitized13_n4000-94b43cf7f484_olmo7b_base_ad_18_merged`` matches
# ``94b`` before ``olmo7b``, so a 7B checkpoint asked for tp=5 on a 1-GPU eval
# task and every task died at engine bringup. Which artifacts hit this is a
# lottery on the config hash.
#
# The fix is to hand conflictscope a path it reads correctly: a directory
# symlink whose name puts the *base model's* size tag first, with the original
# name following in a form that can no longer match. Only checkpoints whose
# path currently mis-reads get a view, so every run that already evaluated
# keeps its exact command. Views live beside the checkpoints they point at —
# the ``data/gt/...`` output dirs are keyed by artifact ID and so carry the
# same hazard in their own prefix.

_SIZE_RE = re.compile(r"(\d+)b")
_VIEWS_DIRNAME = "_eval_views"


def _inferred_gpus(name: str) -> int | None:
    r"""``gpus_needed``'s answer for ``name``, or None when it would guess.

    Byte-for-byte the submodule's arithmetic (2 bytes/param, 20% overhead,
    48 GB cards). None means no ``(\d+)b`` anywhere, where conflictscope
    prints a warning and defaults to 1.
    """
    match = _SIZE_RE.search(str(name).lower())
    if match is None:
        return None
    return max(1, math.ceil(int(match.group(1)) * 2 * 1.2 / 48))


def _defuse(name: str) -> str:
    r"""Break every ``(\d+)b`` in ``name`` so the regex cannot match it."""
    return re.sub(r"(\d)b", r"\1_b", name, flags=re.IGNORECASE)


def model_view(model: str, base_model: str) -> tuple[str, str | None]:
    """``(path_to_evaluate, symlink_prelude)`` for one checkpoint.

    Returns ``model`` unchanged (and no prelude) unless it is a local path
    whose inferred GPU count disagrees with the base model's — the signature
    of a hash-driven misread. The view's parent is the checkpoint's own
    parent, which must itself be regex-clean; ``finetune_root`` is.
    """
    path = Path(model)
    if not path.is_absolute():
        return model, None
    want = _inferred_gpus(base_model)
    if want is None or _inferred_gpus(str(path)) == want:
        return model, None
    views = path.parent / _VIEWS_DIRNAME
    view = views / f"{Path(base_model).name}__{_defuse(path.name)}"
    if _inferred_gpus(str(view)) != want:  # pragma: no cover - defensive
        raise ValueError(
            f"model view {view} still infers "
            f"{_inferred_gpus(str(view))} GPUs, not {want}; the path prefix "
            "must contain no digits-then-'b' before the size tag"
        )
    prelude = (
        f"mkdir -p {shlex.quote(str(views))}\n"
        f"ln -sfn {shlex.quote(str(path))} {shlex.quote(str(view))}"
    )
    return str(view), prelude


@dataclass
class EvalSpec:
    """Everything an ``evaluate_models`` invocation needs except the model."""

    scenarios_dir: Path
    output_dir: Path
    interactive: bool = True
    cache: bool = True
    filter: bool = True
    temperature: float | None = None
    max_tokens: int | None = None
    max_scenarios: int | None = None
    user_model: str | None = None
    judge_model: str | None = None
    user_api_base: str | Path | None = None
    judge_api_base: str | Path | None = None
    assistant_api_base: str | Path | None = None
    # Hostfiles whose servers a task must wait for before starting.
    wait_for: dict[str, Path] = field(default_factory=dict)
    # Explicit chat-format identity (chat_formats.CHAT_FORMATS) the evaluated
    # checkpoint must carry. When set, every eval command for a *local model
    # dir* is preceded by ``chat_formats check``, which refuses to run the
    # eval unless the dir's valuegen_chat_format.json sidecar, served
    # chat_template and stop ids all match that format -- so the assistant is
    # never formatted by a path-substring guess. Served candidates (an
    # ``--assistant-api-base`` endpoint) apply their export's template
    # server-side; the same check runs before the server is started (see
    # ``serve_local_sh``).
    chat_format: str | None = None
    # First-turn cache (``cache.json``) shipped beside the scenarios. The fork
    # reads and extends ``{output_dir}/cache.json``; when that is missing the
    # task copies this seed in first, so every eval of a scenario set starts
    # from the same opening user turns instead of regenerating them.
    cache_seed: Path | None = None


def eval_command(
    spec: EvalSpec,
    model: str,
    output_name: str,
    steer_prompt: Path | None = None,
    stage_in: str | None = None,
    scaffold: Path | None = None,
    view_prelude: str | None = None,
) -> str:
    """One ``evaluate_models.py`` invocation, wait_ready preludes included.

    ``stage_in`` is a shell prelude that resolves the checkpoint's location at
    run time and sets ``MODEL_DIR`` (see ``gcs.stage_in_sh``); when given, the
    model is passed as ``"$MODEL_DIR"`` instead of the literal path.
    ``scaffold`` is a JSON spec (``training.scaffold_spec``) that replaces the
    assistant's chat formatting -- how a raw base is prompted under URIAL. It
    needs an in-process assistant: the fork refuses it for a served one.
    """
    if scaffold is not None and spec.assistant_api_base is not None:
        raise ValueError(
            "a prompt scaffold needs an in-process assistant; it cannot be "
            "combined with a served one (evaluation.assistant.serve)"
        )
    parts = [
        f"python {EVALUATE_MODELS}",
        '-m "$MODEL_DIR"' if stage_in else f"-m {shlex.quote(model)}",
        f"-d {shlex.quote(str(spec.scenarios_dir))}",
        f"-o {shlex.quote(str(spec.output_dir))}",
        f"--output-name {shlex.quote(output_name)}",
    ]
    if spec.interactive:
        parts.append("-i")
        if spec.cache:
            parts.append("--cache")
    if spec.filter:
        parts.append("--filter")
    if spec.temperature is not None:
        parts.append(f"--temperature {spec.temperature}")
    if spec.max_tokens is not None:
        parts.append(f"--max-tokens {spec.max_tokens}")
    if spec.max_scenarios is not None:
        parts.append(f"--max-scenarios {spec.max_scenarios}")
    if spec.user_model:
        parts.append(f"--user-model {shlex.quote(spec.user_model)}")
    if spec.judge_model:
        parts.append(f"--judge-model {shlex.quote(spec.judge_model)}")
    for flag, value in (
        ("--user-api-base", spec.user_api_base),
        ("--judge-api-base", spec.judge_api_base),
        ("--assistant-api-base", spec.assistant_api_base),
    ):
        if value is not None:
            parts.append(f"{flag} {shlex.quote(str(value))}")
    if steer_prompt is not None:
        parts.append(f"--steer-prompt {shlex.quote(str(steer_prompt))}")
    if scaffold is not None:
        parts.append(f"--assistant-scaffold {shlex.quote(str(scaffold))}")

    preludes = [wait_ready_sh(hf, label) for label, hf in spec.wait_for.items()]
    if stage_in:
        preludes.append(stage_in)
    
    if view_prelude:
        preludes.append(view_prelude)
    if (spec.chat_format and stage_in is None
            and spec.assistant_api_base is None and scaffold is None):
        # A local checkpoint dir is formatted by the fork's path heuristic;
        # refuse unless the dir carries the declared explicit identity. A
        # scaffolded model bypasses that heuristic (and, being a raw base,
        # carries no chat-format sidecar to check).
        preludes.append(chat_format_check_sh(model, spec.chat_format))
    cmd = " \\\n    ".join(parts)
    setup = [f"mkdir -p {shlex.quote(str(spec.output_dir))}"]
    if spec.interactive and spec.cache and spec.cache_seed is not None:
        setup.append(seed_cache_sh(spec.cache_seed, spec.output_dir))
    return "\n".join(preludes + setup + [cmd])


def seed_cache_sh(seed: Path, output_dir: Path) -> str:
    """Copy ``seed`` to ``{output_dir}/cache.json`` unless one is already there.

    Copy-then-``mv -n`` so concurrent tasks sharing an output dir never see a
    half-written cache and never replace one a task has already extended."""
    dst = shlex.quote(str(Path(output_dir) / "cache.json"))
    src = shlex.quote(str(seed))
    tmp = shlex.quote(str(Path(output_dir) / ".cache.json.seed")) + ".$$"
    return (
        f"if [ ! -e {dst} ]; then\n"
        f"  if [ -f {src} ]; then cp {src} {tmp} && mv -n {tmp} {dst}; rm -f {tmp}\n"
        f"  else echo 'WARNING: no {dst} and no seed {seed}: first turns will be "
        f"regenerated (run scripts/fetch_eval_inputs.py)' >&2; fi\n"
        f"fi"
    )


def chat_format_check_sh(model_dir: str | Path, chat_format: str) -> str:
    """Shell line that verifies a model dir's explicit chat-format identity."""
    return (
        f"python -m valuegen.ground_truth.chat_formats check "
        f"--model-dir {shlex.quote(str(model_dir))} --expect {shlex.quote(chat_format)}"
    )


def serve_local_sh(
    model_dir: str | Path,
    served_name: str,
    hostfile: str | Path,
    log_file: str | Path,
    chat_format: str | None = None,
    max_model_len: int = 16384,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.9,
    seed: int | None = None,
    tensor_parallel: int = 1,
    extra_args: Sequence[str] = (),
) -> str:
    """Start a vLLM server for ``model_dir`` in the *current* allocation.

    The counterpart of :func:`serve_stage` for a candidate that is served by
    the same job that evaluates it (a packed single-GPU eval task, or the
    tail of a training allocation): ``vllm serve`` runs in the background on
    a free port, its ``host:port`` is written to ``hostfile`` and the shell
    blocks on :func:`wait_ready_sh`; an EXIT trap kills the server (and the
    apptainer process group it lives in) when the task ends. ``vllm``
    resolves to the container wrapper emitted by ``ClusterConfig.activate``
    on ROCm clusters. The served name is what clients address the model by
    (a checkpoint id, never a path -- conflictscope's ``ModelWrapper.create``
    and its ``disable_thinking`` keying look at the name). With
    ``chat_format`` the dir's explicit identity is verified first; the server
    then applies the export's own pinned chat template (``chat_template.jinja``)
    and stops on its generation_config eos ids, so no client formats bytes
    itself. ``seed`` is the server-wide sampling seed (recorded, not a
    per-request seed: per-request seeds would collapse stochastic repeats).
    """
    check = (
        chat_format_check_sh(model_dir, chat_format) + "\n" if chat_format else ""
    )
    seed_flag = f" \\\n    --seed {int(seed)}" if seed is not None else ""
    extra = "".join(f" \\\n    {a}" for a in extra_args)
    # ``vllm`` may be an exported bash function (the container wrapper), which
    # setsid cannot exec directly: the command line is handed to a child bash
    # in its own session, so the EXIT trap can kill the whole server group.
    serve_cmd = (
        f"vllm serve {shlex.quote(str(model_dir))}"
        f" --served-model-name {shlex.quote(served_name)}"
        f" --dtype bfloat16 --api-key api --host 127.0.0.1 --port ${{CAND_PORT}}"
        f" --tensor-parallel-size {tensor_parallel}"
        f" --gpu-memory-utilization {gpu_memory_utilization}"
        f" --max-model-len {max_model_len} --max-num-seqs {max_num_seqs}"
        f"{seed_flag.replace(chr(10), ' ').replace(chr(92), '')}{extra.replace(chr(10), ' ').replace(chr(92), '')}"
    )
    return f"""{check}CAND_PORT=$(shuf -i 9000-9899 -n 1)
rm -f {shlex.quote(str(hostfile))}
echo "127.0.0.1:${{CAND_PORT}}" > {shlex.quote(str(hostfile))}
echo "[$(date)] serving {served_name} from {model_dir} on port ${{CAND_PORT}} (gpu ${{HIP_VISIBLE_DEVICES:-${{CUDA_VISIBLE_DEVICES:-all}}}})"
CAND_CMD="{serve_cmd}"
echo "$CAND_CMD"
setsid bash -c "$CAND_CMD" > {shlex.quote(str(log_file))} 2>&1 &
CAND_PID=$!
trap 'echo "[$(date)] stopping candidate server (pid $CAND_PID)"; kill -- -"$CAND_PID" 2>/dev/null || kill "$CAND_PID" 2>/dev/null || true' EXIT
export CAND_BASE_URL="http://127.0.0.1:${{CAND_PORT}}/v1"
{wait_ready_sh(hostfile, served_name)}"""


def eval_tasks(
    spec: EvalSpec,
    base_model: str,
    interventions: dict[str, dict],
    base_tag: str = "base",
) -> list[Task]:
    """The (N+1)-task list for one base model: 1 base eval + N interventions.

    ``interventions`` maps output tag -> ``{"model": path-or-hf-name}`` for
    finetuned checkpoints or ``{"steer_prompt": path}`` for prompt steering
    (steered runs default to the base model as assistant). A checkpoint entry
    may also carry ``stage_in`` (a shell prelude from ``gcs.stage_in_sh``) when
    the checkpoint may live in remote storage. Output CSVs are ``{tag}.csv``
    in ``spec.output_dir`` — the layout the matrix builders read.
    """
    tasks = [
        Task(
            key=base_tag,
            command=eval_command(spec, base_model, f"{base_tag}.csv"),
            done=spec.output_dir / f"{base_tag}.csv",
        )
    ]
    for tag, iv in interventions.items():
        model = iv.get("model", base_model)
        steer = iv.get("steer_prompt")
        evaluated, view_prelude = model_view(str(model), base_model)
        tasks.append(
            Task(
                key=tag,
                command=eval_command(
                    spec,
                    evaluated,
                    f"{tag}.csv",
                    steer_prompt=Path(steer) if steer else None,
                    stage_in=iv.get("stage_in"),
                    view_prelude=view_prelude,
                ),
                done=spec.output_dir / f"{tag}.csv",
            )
        )
    return tasks
