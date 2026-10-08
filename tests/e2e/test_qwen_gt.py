"""One-value SFT/evaluation/matrix run; external consumers reuse its artifacts.

Needs an allocated GPU (``valuegen smoke --tests`` submits one); skipped
without. No scheduler submissions or live APIs.
The judge is real Qwen3-0.6B; scores test plumbing, not model quality.
"""
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALUE = "calibrated_uncertainty"
BASE_NAME = "Qwen3-0.6B"
ALIGNED_NAME = "qwen3-calibrated-uncertainty"
pytestmark = pytest.mark.e2e


class PipelineRun:
    def __init__(self, root):
        self.root = root
        self.started = time.monotonic()
        self.timings = {}
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.endswith("API_KEY") or key.endswith("BASE_URL") or key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RANK", "LOCAL_RANK", "WORLD_SIZE", "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_CPU"}:
                self.env.pop(key)
        self.env.update({
            "PATH": str(Path(sys.executable).parent) + os.pathsep + self.env.get("PATH", ""),
            "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(root / "datasets-cache"), "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PERSONA_LOCAL_API": "true",
            "OPENAI_API_KEY": "local-test-only", "XDG_CACHE_HOME": str(root / "cache"),
            "XDG_DATA_HOME": str(root / "data-home"), "MPLCONFIGDIR": str(root / "matplotlib"),
        })

    def run(self, name, arguments, *, fork=None):
        env = self.env.copy()
        if fork:
            path = ROOT / "external" / fork
            env["PYTHONPATH"] = os.pathsep.join(map(str, [path, path / "src", path / "data_generation"]))
        started = time.monotonic()
        with (self.root / f"{name}.log").open("w") as log:
            process = subprocess.Popen(arguments, cwd=self.root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=900)
            except BaseException:
                self.stop(process)
                raise
        self.timings[name] = time.monotonic() - started
        assert code == 0, (self.root / f"{name}.log").read_text()[-10000:]

    def script(self, name, source, *arguments, fork):
        path = self.root / (name + ".py")
        path.write_text(source)
        self.run(name, [sys.executable, str(path), *map(str, arguments)], fork=fork)

    @staticmethod
    def stop(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    @contextmanager
    def server(self, base, adapter):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}/v1"
        started = time.monotonic()
        with (self.root / "server.log").open("w") as log:
            process = subprocess.Popen([
                sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(base),
                "--served-model-name", BASE_NAME, "--host", "127.0.0.1", "--port", str(port),
                "--enforce-eager", "--max-model-len", "4096", "--max-num-seqs", "8",
                "--gpu-memory-utilization", "0.25", "--enable-lora", "--max-lora-rank", "8",
                "--lora-modules", f"{ALIGNED_NAME}={adapter}",
                "--default-chat-template-kwargs", '{"enable_thinking":false}',
            ], cwd=self.root, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                startup_deadline = time.monotonic() + 600
                while True:
                    assert process.poll() is None, (self.root / "server.log").read_text()[-10000:]
                    try:
                        with urllib.request.urlopen(endpoint + "/models", timeout=2) as response:
                            names = {model["id"] for model in json.load(response)["data"]}
                        assert {BASE_NAME, ALIGNED_NAME} <= names
                        break
                    except OSError:
                        assert time.monotonic() < startup_deadline, "vLLM startup timed out; see server.log"
                        time.sleep(1)
                self.timings["server_startup"] = time.monotonic() - started
                yield endpoint
            finally:
                self.stop(process)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def pipeline(staged_qwen_model, tmp_path_factory):
    import pandas as pd
    import torch
    from valuegen.config import ClusterConfig, resolve_experiment
    from valuegen.ground_truth import common, interventions

    if not torch.cuda.is_available():
        # Hardware, not something the user can install: skip rather than fail,
        # and say how to get a GPU under the suite.
        pytest.skip("no GPU on this node; run `valuegen smoke --tests` to submit "
                    "the suite as a GPU job (or rerun pytest inside a GPU allocation)")
    root = tmp_path_factory.mktemp("qwen-gt-e2e")
    run = PipelineRun(root)
    run.base = staged_qwen_model
    fixture = json.loads((ROOT / "tests/fixtures/calibrated_uncertainty_e2e.json").read_text())
    train_rows, eval_rows = fixture["training"], fixture["evaluation"]
    assert len(train_rows) == 8 and len(eval_rows) == 4
    assert not {r["scenario_id"] for r in train_rows} & {r["scenario_id"] for r in eval_rows}
    train_dir, eval_dir = root / "train-scenarios", root / "eval-scenarios"
    train_dir.mkdir()
    eval_dir.mkdir()
    write_csv(train_dir / "scenarios.csv", train_rows)
    write_csv(eval_dir / "scenarios.csv", eval_rows)
    repo = root / "repo"
    repo.mkdir()
    (repo / "value_sets").symlink_to(ROOT / "value_sets", target_is_directory=True)
    run.cluster = ClusterConfig(
        envs={"default": str(Path(sys.executable).parent.parent)}, repo=repo,
        finetune_root=root / "finetune", finetune_root_legacy=root / "legacy", data=root / "data",
        slurm_logs=root / "logs", mail_type="NONE", mail_user="test@example.com", gpu_type=None,
        max_concurrent_gpus=1, default_time="00:30:00", default_mem="24G", cpu_partition="unused",
        cpu_qos="unused", exclude=[], source_path=root / "cluster.yaml",
    )
    run.cfg = resolve_experiment({
        "name": "qwen3_sft_calibrated_uncertainty_e2e",
        "intervention": {
            "method": "conflictscope_pairs", "algo": "sft", "value_set": "constitution_tenets_v3",
            "values": [VALUE], "models": {"qwen06": str(run.base)}, "train_scenarios": str(train_dir),
            "train": {
                "chat_format": "qwen_chatml", "dtype": "bfloat16", "bf16": True, "fp16": False,
                "attn_implementation": "eager", "use_peft": True, "lora_r": 2, "lora_alpha": 4,
                "lora_target_modules": ["q_proj", "v_proj"], "max_steps": 2, "max_length": 768,
                "val_fraction": 0, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False, "learning_rate": 0.001, "save_strategy": "no",
                "logging_steps": 1, "eval_strategy": "no", "report_to": "none", "disable_tqdm": True,
                "dump_first_batch": False, "dataloader_pin_memory": False, "seed": 42,
            },
        },
        "evaluation": {
            "method": "conflictscope", "scenarios": str(eval_dir), "mode": "interactive",
            "judge": {"model": BASE_NAME}, "matrices": {"variants": ["likert_raw", "likert_normalized"], "cols": "trained"},
        },
    })
    run.manifest = interventions.claim(run.cfg, run.cluster)
    dataset_dir = interventions.datasets_dir(run.cfg, run.cluster) / VALUE
    dataset_dir.mkdir(parents=True)
    pairs = []
    for row in train_rows:
        aligned = "action1" if row["value1"] == VALUE else "action2"
        other = "action2" if aligned == "action1" else "action1"
        pairs.append({"prompt": [{"role": "user", "content": row["description"] + "\nHow should the assistant respond?"}],
                      "chosen": [{"role": "assistant", "content": row[aligned]}],
                      "rejected": [{"role": "assistant", "content": row[other]}]})
    (dataset_dir / "dataset.jsonl").write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
    run.pairs = pairs
    task = interventions.train_stages(run.cfg, run.cluster)[0].tasks[0]
    assert not task.is_done()
    run.run("train-and-export", ["bash", "-e", "-c", task.command])
    assert task.is_done()
    run.adapter = interventions.train_output_dir(run.cfg, run.cluster, "qwen06", VALUE)
    run.merged = interventions.merged_path(run.cfg, run.cluster, "qwen06", VALUE)
    state = json.loads((run.adapter / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    run.evaluations = common.eval_dir(run.cfg, run.cluster)
    run.evaluations.mkdir(parents=True, exist_ok=True)
    with run.server(run.base, run.adapter) as endpoint:
        for model, tag in [(BASE_NAME, "qwen06_base"), (ALIGNED_NAME, f"qwen06_{VALUE}")]:
            arguments = [sys.executable, str(ROOT / "external/conflictscope/src/evaluate_models.py"),
                         "--model", model, "--scenarios-dir", str(eval_dir), "--output-dir", str(run.evaluations),
                         "--output-name", tag + ".csv", "--interactive", "--cache", "--max-scenarios", "4",
                         "--batch-size", "4", "--temperature", "0", "--max-tokens", "512",
                         "--user-model", BASE_NAME, "--judge-model", BASE_NAME,
                         "--api-base", endpoint, "--user-api-base", endpoint, "--judge-api-base", endpoint]
            run.run("evaluate-" + tag, arguments, fork="conflictscope")
            output = run.evaluations / (tag + ".csv")
            before = pd.read_csv(output)
            run.run("resume-" + tag, arguments, fork="conflictscope")
            pd.testing.assert_frame_equal(before, pd.read_csv(output), check_exact=False, rtol=0, atol=1e-12)
            assert "Found existing results for all scenarios" in (root / f"resume-{tag}.log").read_text()
    yield run
    run.timings["total"] = time.monotonic() - run.started
    (root / "timings.json").write_text(json.dumps(run.timings, indent=2))


def test_single_value_sft_base_aligned_judge_and_matrix(pipeline):
    import numpy as np
    import pandas as pd
    import yaml
    from valuegen.ground_truth.evals.conflictscope import build_ground_truth
    from valuegen.ground_truth.matrices import load_matrix

    run = pipeline
    frames = [pd.read_csv(run.evaluations / f"{tag}.csv") for tag in ("qwen06_base", f"qwen06_{VALUE}")]
    for frame, model in zip(frames, (BASE_NAME, ALIGNED_NAME)):
        assert len(frame) == 4 and frame["scenario_id"].nunique() == 4
        assert set(frame["choice"]) <= {"A", "B"}
        assert pd.to_numeric(frame["likert"]).between(-1, 1).all()
        assert frame["conversation"].str.contains("assistant", case=False).all()
        assert frame["assistant_model"].eq(model).all() and frame["judge_model"].eq(BASE_NAME).all()
    assert set(frames[0]["scenario_id"]) == set(frames[1]["scenario_id"])
    written = build_ground_truth(run.cfg, run.cluster, run.manifest)
    assert len(written) == 2
    for path in written:
        matrix, rows, cols = load_matrix(path)
        assert rows == cols == [VALUE] and matrix.shape == (1, 1)
        assert np.isfinite(matrix).all()
        provenance = yaml.safe_load(path.with_name(path.stem + "_provenance.yaml").read_text())
        assert provenance["judge"] == BASE_NAME
    # Any sign of the cell is acceptable: the 0.6B judge is a plumbing check.


def test_persona_real_hidden_state_extraction(pipeline):
    run = pipeline
    positive, negative = [], []
    for pair in run.pairs[:2]:
        for rows, response, score in ((positive, pair["chosen"], 100), (negative, pair["rejected"], 0)):
            rows.append({"prompt": pair["prompt"][0]["content"] + "\n", "answer": response[0]["content"],
                         VALUE: score, "coherence": 100})
    pos, neg, output = run.root / "persona-pos.csv", run.root / "persona-neg.csv", run.root / "persona-vectors"
    write_csv(pos, positive)
    write_csv(neg, negative)
    run.run("persona-extract", [sys.executable, str(ROOT / "external/persona_vectors/generate_vec.py"),
            "--model_name", str(run.merged), "--pos_path", str(pos), "--neg_path", str(neg),
            "--trait", VALUE, "--save_dir", str(output)], fork="persona_vectors")
    import torch
    config = json.loads((run.merged / "config.json").read_text())
    for kind in ("prompt_avg", "prompt_last", "response_avg"):
        vector = torch.load(output / f"{VALUE}_{kind}_diff.pt", weights_only=True)
        assert vector.shape == (config["num_hidden_layers"] + 1, config["hidden_size"])
        assert torch.isfinite(vector).all()
        if kind == "response_avg":
            assert torch.count_nonzero(vector) > 0


def test_weight_steering_real_adapter_and_inference(pipeline):
    pipeline.script("weight-steering", r'''
import gc
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from cs_task_vectors import _load_adapter_deltas
from task_vectors import TaskVector, get_total_layers

base, adapter = sys.argv[1:]
torch.set_num_threads(4)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
tokenizer = AutoTokenizer.from_pretrained(base)
inputs = tokenizer("How certain are you?", return_tensors="pt")
with torch.no_grad():
    base_logits = model(**inputs).logits.clone()
deltas = _load_adapter_deltas(adapter, base, "cpu")
assert deltas and any(torch.count_nonzero(delta) for delta in deltas.values())
assert all(torch.isfinite(delta).all() for delta in deltas.values())
# The fork loader uses module names; TaskVector.apply_to consumes state keys.
weights = model.state_dict()
vector = {key: deltas[key[:-7]] if key.endswith(".weight") and key[:-7] in deltas
          else torch.zeros_like(weight) for key, weight in weights.items()}
for module, delta in deltas.items():
    assert module + ".weight" in weights
    assert delta.shape == weights[module + ".weight"].shape
task = TaskVector(vector=vector, total_layers=get_total_layers(model))
del weights
gc.collect()
zero = task.apply_to(model, scaling_coef=0).eval()
with torch.no_grad():
    torch.testing.assert_close(zero(**inputs).logits, base_logits, rtol=0, atol=0)
del zero, base_logits
gc.collect()
steered = task.apply_to(model, scaling_coef=1).eval()
with torch.no_grad():
    steered_logits = steered(**inputs).logits.clone()
    generated = steered.generate(**inputs, max_new_tokens=4, do_sample=False)
assert generated.shape[1] > inputs["input_ids"].shape[1]
expected = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32), adapter).merge_and_unload().eval()
for key, weight in expected.state_dict().items():
    torch.testing.assert_close(steered.state_dict()[key], weight, rtol=1e-5, atol=1e-6)
with torch.no_grad():
    torch.testing.assert_close(steered_logits, expected(**inputs).logits, rtol=1e-4, atol=1e-4)
''', pipeline.base, pipeline.adapter, fork="weight-steering")
