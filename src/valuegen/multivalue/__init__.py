"""Standard interface for multivalue (k-value-set) experiments.

An experiment is a YAML config (``configs/experiments/multivalue/*.yaml``) naming a
universe value set, an external value set to measure coverage against,
embedding stores for both, a DPO source artifact, arm families, a fixed
per-arm row budget, one base model plus a TRL recipe, and the eval suites.
The stages are

    valuegen mv sets     sample / preview / freeze the k-value sets (sets.json)
    valuegen mv metrics  coverage + tightness per arm per embedding store
    valuegen mv data     per-arm training mixes from the source artifact
    valuegen mv train    one SLURM task per arm x seed (train + export)
    valuegen mv eval     Inspect suites against every checkpoint
    valuegen mv analyze  outcomes, correlations, figures, REPORT.md
    valuegen mv import   register externally trained arms
    valuegen mv status   arm x stage completeness

Everything here is deliberately separate from the ``ground_truth`` drivers
and the frozen RQ3 protocol modules: the few pure primitives it shares with
them (hash orderings, balanced quotas, mix writing) are ported into this
package rather than imported, so a change to a frozen protocol never moves a
multivalue identity and vice versa. Cluster facts come only from
``configs/cluster.yaml`` through :func:`valuegen.config.load_cluster`.
"""
