"""CPU smoke tests using the published Qwen3-0.6B weights and tokenizer."""
import gc
import json

import pytest

pytestmark = pytest.mark.model_smoke


@pytest.fixture
def real_model(staged_qwen_model, tmp_path, monkeypatch):
    import torch
    import datasets.config

    # The staged checkpoint is self-contained; training stays offline.
    base = staged_qwen_model
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "ACCELERATE_USE_DEEPSPEED"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "ACCELERATE_USE_CPU"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    pairs = tmp_path / "pairs.jsonl"
    row = {
        "prompt": [{"role": "user", "content": "Say hello."}],
        "chosen": [{"role": "assistant", "content": "Hello!"}],
        "rejected": [{"role": "assistant", "content": "Goodbye."}],
    }
    pairs.write_text("".join(json.dumps(row) + "\n" for _ in range(4)))
    yield base, pairs
    gc.collect()
    torch.set_num_threads(previous)


def _arguments(base, pairs, output):
    return [
        "--model_name_or_path", str(base), "--dataset_name", str(pairs),
        "--output_dir", str(output), "--chat_format", "qwen_chatml",
        "--use_cpu", "true", "--bf16", "false", "--fp16", "false", "--dtype", "bfloat16",
        "--attn_implementation", "eager", "--use_peft", "true",
        "--lora_r", "2", "--lora_alpha", "4", "--lora_target_modules", "q_proj", "v_proj",
        "--max_steps", "2", "--max_length", "32", "--val_fraction", "0",
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing", "false", "--learning_rate", "0.001",
        "--save_strategy", "steps", "--save_steps", "1", "--logging_steps", "1",
        "--eval_strategy", "no", "--report_to", "none", "--disable_tqdm", "true",
        "--dump_first_batch", "false", "--dataloader_pin_memory", "false",
    ]


@pytest.mark.parametrize("algo", ["sft", "dpo"])
def test_training_resume_and_merge(real_model, tmp_path, monkeypatch, algo):
    import torch
    from valuegen.ground_truth import training
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    import trl

    base, pairs = real_model
    tokenizer = AutoTokenizer.from_pretrained(base)
    inputs = tokenizer("Say hello.", return_tensors="pt")
    reference_checks = []

    class ReferenceCheck(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                self.before = model(**inputs).logits.clone()
            model.train()

        def on_train_end(self, args, state, control, model=None, **kwargs):
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                after = model(**inputs).logits
            torch.testing.assert_close(after, self.before, atol=0, rtol=0)
            reference_checks.append(True)

    if algo == "dpo":
        real_trainer = trl.DPOTrainer

        class ObservedDPOTrainer(real_trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert self.ref_model is None  # PEFT reference is the base with adapters disabled.
                self.add_callback(ReferenceCheck())

        monkeypatch.setattr(trl, "DPOTrainer", ObservedDPOTrainer)

    output = tmp_path / algo
    train = getattr(training, f"train_{algo}")
    args = _arguments(base, pairs, output)
    train(args + ["--stop_after_step", "1"])
    checkpoint = output / "checkpoint-1"
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text())
    assert state["global_step"] == 1
    assert (checkpoint / "optimizer.pt").is_file()
    first_weights = load_file(str(checkpoint / "adapter_model.safetensors"))
    state["log_history"].append({"resume_test_marker": True})
    state_path.write_text(json.dumps(state))
    gc.collect()
    train(args + ["--resume_from_checkpoint", "auto"])
    final_state = json.loads((output / "trainer_state.json").read_text())
    assert final_state["global_step"] == 2
    assert any(entry.get("resume_test_marker") for entry in final_state["log_history"])
    assert all(torch.isfinite(torch.tensor(e["loss"])) for e in final_state["log_history"] if "loss" in e)
    final_weights = load_file(str(output / "adapter_model.safetensors"))
    assert any(not torch.equal(p, final_weights[n]) for n, p in first_weights.items())
    assert any(torch.count_nonzero(p) for n, p in final_weights.items() if "lora_B" in n)
    if algo == "dpo":
        assert reference_checks == [True, True]
    gc.collect()
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16), output,
    ).eval()
    with torch.no_grad():
        adapter_logits = model(**inputs).logits.float()
    # BF16 addition rounds merged weights. Use PEFT's in-memory merge as the
    # reference for the exact save/reload check instead of assuming that
    # unmerged adapter logits are bitwise equal after weight rounding.
    model = model.merge_and_unload().eval()
    with torch.no_grad():
        expected = model(**inputs).logits.float()
    similarity = torch.nn.functional.cosine_similarity(adapter_logits.flatten(), expected.flatten(), dim=0)
    assert similarity > 0.999
    del model
    gc.collect()
    merged = tmp_path / "qwen-merged"
    training.merge_lora(str(base), output, merged, chat_family="qwen")
    reloaded = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        actual = reloaded(**inputs).logits.float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not any("lora_" in n for n, _ in reloaded.named_parameters())
    saved_tokenizer = AutoTokenizer.from_pretrained(merged)
    assert saved_tokenizer.eos_token == "<|im_end|>"
    assert saved_tokenizer.chat_template
    del reloaded
