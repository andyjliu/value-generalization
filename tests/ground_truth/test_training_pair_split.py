import json

import pytest

from valuegen.ground_truth.training import _load_pair_dataset


def test_pair_split_zero_keeps_every_training_pair(tmp_path):
    pytest.importorskip("datasets")
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps({"prompt": str(i), "chosen": "yes", "rejected": "no"}) + "\n" for i in range(20)))
    full = _load_pair_dataset(str(path), 42, 0)
    assert len(full["train"]) == 20 and len(full["test"]) == 0
    legacy = _load_pair_dataset(str(path), 42)
    assert len(legacy["train"]) == 18 and len(legacy["test"]) == 2
    assert set(legacy["train"]["prompt"]).isdisjoint(legacy["test"]["prompt"])
