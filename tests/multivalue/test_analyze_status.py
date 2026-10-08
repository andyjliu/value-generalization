"""Analysis: outcomes, correlations/reporting, status, and CLI wiring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from valuegen.multivalue import analyze as A
from valuegen.multivalue import cli as mv_cli
from valuegen.multivalue import data as mvdata
from valuegen.multivalue import eval_stage as ES
from valuegen.multivalue import imports as mvimports
from valuegen.multivalue import metrics_stage
from valuegen.multivalue import outcomes as O
from valuegen.multivalue import status as S
from valuegen.multivalue.evals import runner as R

from test_data import _freeze


def _inputs(env):
    cfg, layout = env["cfg"], env["layout"]
    toy_panel = {"schema_version": 1, "suite": "toy_panel", "params": {"subsample": None},
                 "inputs_sha256": "a" * 64,
                 "conditions": [{"condition_id": f"c{i}"} for i in range(4)]}
    toy_items = {"schema_version": 1, "suite": "toy_items",
                 "params": {"subsample": {"n": 10, "seed": 3}},
                 "inputs_sha256": "b" * 64, "sample_ids": [f"p{i}" for i in range(4)]}
    for name, rec in (("toy_panel", toy_panel), ("toy_items", toy_items)):
        mvdata.write_json(R.inputs_path(layout, name), rec)
    return {"toy_panel": toy_panel, "toy_items": toy_items}


def _write_eval(env, cand, inputs, value):
    cfg, layout = env["cfg"], env["layout"]
    ag = [{"valid": True, "harmful": float((i + value) % 2), "classifier_verdict": float(i % 2),
           "urgency_type": "replacement", "scenario": "s", "condition_id": f"c{i}"}
          for i in range(4)]
    ms = [{"valid": True, "compliant": float(i < value), "compliance_rating": float(i + 3),
           "top_level_section": "secA" if i < 2 else "secB", "sample_id": f"p{i}"}
          for i in range(4)]
    for suite, rows in (("toy_panel", ag), ("toy_items", ms)):
        root = layout.suite_dir(cand.ckpt_id, suite)
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "rows.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        mvdata.write_json(root / "COMPLETE.json",
                          {"complete": True, "protocol_sha256": R.protocol_sha(cfg, suite, inputs[suite])})


def test_statistics_and_covariate_fit_are_deterministic():
    x = np.arange(8.0)
    y = 2 * x + np.array([0, .1, -.1, .2, -.2, .1, 0, -.1])
    one = A.correlate(x, y, n_perm=99, seed=3, n_boot=100)
    two = A.correlate(x, y, n_perm=99, seed=3, n_boot=100)
    assert one == two and one["n"] == 8 and one["pearson_r"] > .99
    assert 0 < one["perm_p"] <= 1 and one["bootstrap_ci_low"] <= one["pearson_r"] <= one["bootstrap_ci_high"]
    frame = pd.DataFrame({"x": x, "y": y + np.tile([0, 1], 4), "k": np.tile([2, 3], 4), "rows": 60})
    fit = A.ols_adjusted(frame, "x", "y", ["k", "rows"])
    assert fit is not None and fit["covariates"] == "k" and fit["coefficient"] == pytest.approx(2, abs=.1)
    assert "rows" not in fit["covariates"]


def test_outcome_terms_pair_units_and_ignore_invalid_rows():
    rows = [{"condition_id": "a", "valid": True, "harmful": 1, "urgency_type": "replacement",
             "scenario": "s", "classifier_verdict": 1},
            {"condition_id": "b", "valid": True, "harmful": 0, "urgency_type": "replacement",
             "scenario": "s", "classifier_verdict": 0},
            {"condition_id": "b", "valid": False, "harmful": 1, "urgency_type": "replacement"}]
    terms = O.unit_terms("toy_panel", rows, ["a", "b"], ["replacement_harmful_rate"])
    assert terms.estimate["replacement_harmful_rate"] == .5 and terms.n_valid == 2 and terms.n_rows == 3
    idx = np.array([[0, 0], [1, 1], [0, 1]])
    assert O.bootstrap(terms, idx).ravel().tolist() == [1, 0, .5]


def test_status_before_freeze_does_not_need_embedding_files(mv_env, monkeypatch, capsys):
    env = mv_env
    for path in env["emb_dir"].iterdir():
        path.unlink()
    monkeypatch.setattr(mv_cli, "load_cluster", lambda _p=None: env["cluster"])
    assert mv_cli.main(["status", "-c", str(env["cfg_path"])]) == 0
    out = capsys.readouterr().out
    assert "sets=pending" in out and "mv sets" in out


def test_analyze_and_status_end_to_end(mv_env, monkeypatch, capsys):
    env = mv_env
    cfg, cluster, layout = env["cfg"], env["cluster"], env["layout"]
    arms = _freeze(env)
    metrics_stage.write_metrics(cfg, cluster, layout)
    inputs = _inputs(env)
    all_arms = mvimports.all_arms(layout)
    cands = ES.plan_candidates(cfg, cluster, layout, all_arms)
    chosen = [cands[0], *cands[1:6]]  # five trained arms, including the k=2 explicit arm
    for i, cand in enumerate(chosen):
        _write_eval(env, cand, inputs, min(i, 4))

    result = A.analyze(cfg, cluster, layout, all_arms, log=lambda *_: None)
    assert len(result["joined"]) == len(all_arms)
    assert set(result["correlations"]["method"]) >= {"correlation", "ols"}
    corr = result["correlations"]
    assert set(corr[corr["method"] == "correlation"]["n"]) == {5}  # base has no set metric
    for name in ("outcomes.csv", "completeness.csv", "joined.csv", "correlations.csv"):
        assert (layout.scores_dir / name).is_file()
    report = layout.report_path.read_text()
    assert "## Completeness" in report and "## Covariate-adjusted OLS" in report and "valuegen mv analyze" in report
    assert result["figures"] and all(p.is_file() for p in result["figures"])

    frame = S.status_frame(cfg, cluster, layout, all_arms)
    assert len(frame) == len(all_arms) and set(("metrics", "data", "train", "eval.toy_panel", "analyze")) <= set(frame)
    assert frame.set_index("arm_id").loc["base", "eval.toy_panel"] == "done"
    assert frame.set_index("arm_id").loc["sub3_07", "eval.toy_panel"] == "pending"
    assert set(frame["analyze"]) == {"done"}

    monkeypatch.setattr(mv_cli, "load_cluster", lambda _p=None: cluster)
    assert mv_cli.main(["status", "-c", str(env["cfg_path"])]) == 0
    assert "eval.toy_panel" in capsys.readouterr().out
    assert mv_cli.main(["analyze", "-c", str(env["cfg_path"])]) == 0
    assert "wrote" in capsys.readouterr().out
