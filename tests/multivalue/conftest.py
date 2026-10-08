"""Fixtures for the multivalue package: a tiny but complete experiment on disk.

Universe = the real ``constitution_tenets_v3`` registry set restricted to the
12 values given source rows here; external = 6 synthetic tenets with a
section sidecar; embeddings = deterministic random vectors in every
supported file type; a DPO recipe and an FSDP config; a config YAML whose
enabled eval suites are the synthetic ones in ``fake_suites.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from valuegen.multivalue import config as mvconfig
from valuegen.multivalue import evals as mvevals
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue.config import load_config
from valuegen.multivalue.layout import Layout
from valuegen.values import load_value_set

from fake_suites import TOY_SUITES

BASE_MODEL = "org/neutral-sft-test"
N_SOURCE = 12
DIM = 16
N_LAYERS = 4
LAYER = 2


def _dpo_row(value: str, i: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"scenario {value} {i}"}],
        "chosen": [{"role": "assistant", "content": f"good {value} {i}"}],
        "rejected": [{"role": "assistant", "content": f"bad {value} {i}"}],
        "source": "hh", "scenario_id": f"{value}-{i}", "score": 0.9,
    }


@pytest.fixture(autouse=True)
def registered_toy_suites(monkeypatch):
    """Make the synthetic suites known wherever suite names are checked."""
    names = mvconfig.SUITES + tuple(s.name for s in TOY_SUITES)
    monkeypatch.setattr(mvconfig, "SUITES", names)
    monkeypatch.setattr(mvimports, "SUITES", names)
    for suite in TOY_SUITES:
        monkeypatch.setitem(mvevals.SUITES, suite.name, suite)


@pytest.fixture
def mv_env(tmp_path: Path, cluster):
    rng = np.random.default_rng(0)
    registry = load_value_set("constitution_tenets_v3")
    all_values = sorted(registry)
    source_values = all_values[:N_SOURCE]

    # source artifact: 12 values x 20 rows
    artifact = "const_v3_dpo_test-abc123"
    ds = cluster.data / "interventions" / "label_subset" / "constitution_tenets_v3" / artifact / "datasets"
    for v in source_values:
        d = ds / v
        d.mkdir(parents=True)
        with open(d / "dataset.jsonl", "w") as f:
            for i in range(20):
                f.write(json.dumps(_dpo_row(v, i)) + "\n")

    # external set
    ext_names = [f"t{i}_ext" for i in range(6)]
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "spec.json").write_text(json.dumps({n: f"desc of {n}" for n in ext_names}))

    # embeddings
    emb = tmp_path / "emb"
    emb.mkdir()
    uni_vecs = {v: rng.normal(size=DIM) for v in all_values}
    ext_vecs = {n: rng.normal(size=DIM) for n in ext_names}
    np.savez(emb / "sent_uni.npz", labels=np.array(all_values), embeddings=np.stack([uni_vecs[v] for v in all_values]))
    (emb / "sent_ext.json").write_text(json.dumps({k: v.tolist() for k, v in ext_vecs.items()}))
    import torch
    torch.save({v: torch.tensor(rng.normal(size=(N_LAYERS, DIM))) for v in all_values}, emb / "pers_uni.pt")
    torch.save({n: torch.tensor(rng.normal(size=(N_LAYERS, DIM))) for n in ext_names}, emb / "pers_ext.pt")

    # recipe + fsdp
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump({"learning_rate": 5e-6, "beta": 0.1}))
    (tmp_path / "fsdp.json").write_text("{}")

    raw = {
        "name": "mvtest", "seed": 7,
        "universe": {"value_set": "constitution_tenets_v3", "restrict_to_source": True},
        "external": {"value_set": str(ext_dir / "spec.json")},
        "embeddings": {
            "sentence": {"universe": str(emb / "sent_uni.npz"), "external": str(emb / "sent_ext.json")},
            "persona": {"universe": str(emb / "pers_uni.pt"), "external": str(emb / "pers_ext.pt"),
                        "model": BASE_MODEL, "layer": LAYER},
        },
        "source": {"method": "label_subset", "artifact": artifact, "format": "dpo"},
        "arms": {"reference": ["base"],
                 "families": [{"id": "sub3", "k": 3, "n": 8},
                              {"id": "probe", "values": [[source_values[0], source_values[1]],
                                                         [source_values[2], source_values[3], source_values[4]]]}]},
        "budget": {"rows": 60},
        "train": {"base_model": BASE_MODEL, "chat_format": "qwen_chatml", "recipe": str(tmp_path / "recipe.yaml"),
                  "fsdp_config": str(tmp_path / "fsdp.json"), "seeds": [42],
                  "resources": {"gpus": 2, "time": "00:30:00"}, "concurrent": 2},
        "evals": {"default_grader": "google/gemini-3.7-flash",
                  "candidate": {"temperature": 0.7},
                  "resources": {"gpus": 1, "time": "01:00:00"},
                  "suites": {"toy_panel": {"repeats": 2},
                             "toy_items": {"repeats": 1, "subsample": {"n": 10, "seed": 3}, "grader": "openai/gpt-5.5"},
                             "prefill": {"enabled": False}}},
        "analysis": {"outcomes": ["toy_panel.replacement_harmful_rate", "toy_items.overall_compliance"],
                     "n_boot": 50, "n_perm": 200},
    }
    cfg_path = tmp_path / "mvtest.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_config(cfg_path)
    return {
        "cfg": cfg, "cfg_path": cfg_path, "raw": raw, "cluster": cluster, "layout": Layout(cfg, cluster),
        "source_values": source_values, "all_values": all_values, "ext_names": ext_names,
        "emb_dir": emb, "uni_vecs": uni_vecs, "ext_vecs": ext_vecs, "datasets_dir": ds,
    }
