"""valuegen.multivalue.cells: per-(arm, value) prefill robustness."""

import numpy as np
import pandas as pd
import pytest

from valuegen.multivalue.cells import prefill_cell_metrics, weighted_spearman


def _row(cond, bucket, value, other, sid, side, pro):
    return dict(valid=True, condition=cond, bucket=bucket, value=value, other_value=other,
                scenario_id=sid, side=side, pro_value=pro)


def test_prefill_cell_metrics_one_value():
    rows = pd.DataFrame([
        _row("baseline", "on_target", "v", "u", "s1", 0, 1.0),
        _row("baseline", "on_target", "v", "u", "s2", 0, 0.0),
        _row("injected", "on_target", "v", "u", "s1", 0, 1.0),
        _row("injected", "on_target", "v", "u", "s2", 0, 0.0),
        _row("injected", "pro_spec", "u", "v", "s3", 0, 0.25),
    ])
    (c,) = prefill_cell_metrics(rows)
    assert c["value"] == "v" and c["maiya"] == 0.5 and c["adherence_none"] == 0.5
    assert c["adherence_pro"] == 0.75 and c["sturgeon"] == pytest.approx(0.25)
    # retention_norm pairs per (scenario, side): only s1 adhered with no history
    assert c["rn_num"] == 1.0 and c["rn_den"] == 1.0 and c["retention_norm"] == 1.0


def test_weighted_spearman_equal_weights_is_spearman():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=20), rng.normal(size=20)
    from scipy import stats
    assert weighted_spearman(x, y, np.ones(20)) == pytest.approx(stats.spearmanr(x, y)[0])
