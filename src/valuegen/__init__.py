"""valuegen — ground-truth measurement and prediction of value generalization.

Three subsystems, mirrored by the `valuegen` CLI:

- ``ground_truth``: produce steerability matrices for a value set, by whichever
  method (conflictscope scenarios, label+subset, prompt-steering).
- ``predictors``: cheap estimates of the same matrices (persona vectors,
  weight-steering, sentence embeddings).
- ``analysis``: correlate predictors against ground truth, with reliability
  ceilings; embed and plot.
"""

__version__ = "0.1.0"
