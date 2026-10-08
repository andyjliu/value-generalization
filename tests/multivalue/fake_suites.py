"""Two synthetic eval suites that exercise the shared suite machinery
(frozen inputs, protocol hashes, completion markers, imported-run reuse,
bootstrap, status, waves) without any upstream dataset or Inspect task.

``toy_panel`` is shared across candidates with one generation unit per
condition (a multi-unit suite); ``toy_items`` has a single unit over a
subsampleable list of item ids. Neither is ever actually run: ``scorer`` and
``rows`` raise. The ``registered_toy_suites`` fixture in ``conftest.py``
makes them known to the config loader, the import scanner and the suite
registry for the duration of a test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from valuegen.ground_truth import inspect_runner as IR
from valuegen.multivalue import data as mvdata
from valuegen.multivalue._hashing import sha256_json
from valuegen.multivalue.config import MultivalueConfig, SuiteConfig
from valuegen.multivalue.evals.base import INPUTS_SCHEMA_VERSION, EvalError, Suite, subsample

SCENARIOS = ("s0", "s1", "s2")
URGENCY = ("replacement", "none")
N_ITEMS = 20
N_SMOKE_ITEMS = 5


def _subsample_args(sc: SuiteConfig) -> tuple[int | None, int]:
    return (sc.subsample["n"], sc.subsample["seed"]) if sc.subsample else (None, 0)


class ToyPanelSuite(Suite):
    name = "toy_panel"
    boot_unit = "condition_id"
    metrics = ("harmful_rate", "replacement_harmful_rate", "none_harmful_rate", "by_scenario.*")
    row_fields = ("suite", "checkpoint_id", "condition_id", "scenario", "urgency_type", "sample_id", "repeat",
                  "harmful", "valid", "grader_failed")
    options_defaults = {"grader_max_connections": 8, "max_failed_grade_fraction": 0.0}

    def build_inputs(self, cfg: MultivalueConfig, sc: SuiteConfig, root: Path, *, layout=None) -> dict:
        panel = [{"condition_id": f"{s}_{u}", "scenario": s, "urgency_type": u, "text": f"toy prompt {s} {u}"}
                 for s in SCENARIOS for u in URGENCY]
        keep = set(subsample([c["condition_id"] for c in panel], *_subsample_args(sc)))
        conditions = [c for c in panel if c["condition_id"] in keep]
        return {"schema_version": INPUTS_SCHEMA_VERSION, "suite": self.name, "params": self.params(cfg, sc),
                "n_panel": len(panel), "n_conditions": len(conditions), "conditions": conditions,
                "inputs_sha256": sha256_json(conditions), "built_at": mvdata.now()}

    def _panel(self, inputs: dict, smoke: bool) -> list[dict]:
        panel = list(inputs["conditions"])
        return panel[: self.smoke_units] if smoke else panel

    def attach(self, run: IR.SuiteRun, inputs: dict, *, smoke: bool = False, cand=None, selection=None) -> IR.SuiteRun:
        run.units = [(c["condition_id"], run.log_root / "conditions" / c["condition_id"], lambda: None)
                     for c in self._panel(inputs, smoke)]
        return run

    def unit_expectations(self, inputs: dict, *, smoke: bool = False, selection=None) -> dict[str, int]:
        return {c["condition_id"]: 1 for c in self._panel(inputs, smoke)}

    def scorer(self, run, grader, cache):
        raise EvalError("toy suites are never graded")

    def rows(self, run) -> list[dict]:
        raise EvalError("toy suites are never run")

    def boot_units(self, inputs: dict, *, smoke: bool = False, selection=None) -> list[str]:
        return [c["condition_id"] for c in self._panel(inputs, smoke)]

    def row_metrics(self, r: Mapping[str, Any]) -> dict[str, float | None]:
        if not r.get("valid") or r.get("harmful") is None:
            return {}
        h = float(r["harmful"])
        return {
            "harmful_rate": h,
            "replacement_harmful_rate": h if r.get("urgency_type") == "replacement" else None,
            "none_harmful_rate": h if r.get("urgency_type") == "none" else None,
            f"by_scenario.{r.get('scenario')}": h,
        }


class ToyItemsSuite(Suite):
    name = "toy_items"
    boot_unit = "sample_id"
    metrics = ("overall_compliance", "mean_rating", "by_section.*")
    row_fields = ("suite", "checkpoint_id", "sample_id", "repeat", "top_level_section", "compliant",
                  "compliance_rating", "valid", "grader_failed")
    options_defaults = {"grader_max_connections": 8, "max_failed_grade_fraction": 0.0}

    def build_inputs(self, cfg: MultivalueConfig, sc: SuiteConfig, root: Path, *, layout=None) -> dict:
        ids = [f"item{i:02d}" for i in range(N_ITEMS)]
        sample_ids = subsample(ids, *_subsample_args(sc))
        return {"schema_version": INPUTS_SCHEMA_VERSION, "suite": self.name, "params": self.params(cfg, sc),
                "n_total": len(ids), "sample_ids": sample_ids, "n_samples": len(sample_ids),
                "inputs_sha256": sha256_json(sample_ids), "smoke_sample_ids": ids[:N_SMOKE_ITEMS],
                "built_at": mvdata.now()}

    def _ids(self, inputs: dict, smoke: bool) -> list[str]:
        return list(inputs["smoke_sample_ids"]) if smoke else list(inputs["sample_ids"])

    def attach(self, run: IR.SuiteRun, inputs: dict, *, smoke: bool = False, cand=None, selection=None) -> IR.SuiteRun:
        run.units = [(self.name, run.log_root / "items", lambda: None)]
        run.extra["sample_ids"] = self._ids(inputs, smoke)
        return run

    def unit_expectations(self, inputs: dict, *, smoke: bool = False, selection=None) -> dict[str, int]:
        return {self.name: len(self._ids(inputs, smoke))}

    def scorer(self, run, grader, cache):
        raise EvalError("toy suites are never graded")

    def rows(self, run) -> list[dict]:
        raise EvalError("toy suites are never run")

    def boot_units(self, inputs: dict, *, smoke: bool = False, selection=None) -> list[str]:
        return self._ids(inputs, smoke)

    def row_metrics(self, r: Mapping[str, Any]) -> dict[str, float | None]:
        if not r.get("valid") or r.get("compliant") is None:
            return {}
        c = float(r["compliant"])
        rating = r.get("compliance_rating")
        out = {"overall_compliance": c, "mean_rating": None if rating is None else float(rating)}
        if r.get("top_level_section"):
            out[f"by_section.{r['top_level_section']}"] = c
        return out


TOY_SUITES = (ToyPanelSuite(), ToyItemsSuite())
