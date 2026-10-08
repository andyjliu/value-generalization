"""Ground-truth generation, decoupled into two sides.

Training side — *intervention methods* (``interventions.get_method``):
``label_subset``, ``conflictscope_pairs``, ``model_spec_aft``, ``none``.
Each produces an intervention artifact (checkpoints plus a manifest). Evaluation side — *eval methods* (``evals.get_eval``):
``conflictscope``. Each scores a manifest into a GT artifact (eval CSVs +
steerability matrices). The two sides meet only at the manifest, so any
intervention is evaluable under any eval config without retraining.
"""
