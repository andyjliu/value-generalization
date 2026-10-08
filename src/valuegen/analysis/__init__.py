"""Analysis layer: correlation, MDS/spectral maps, plots.

Everything here consumes the ``.npy + values.json`` matrix contract
— GT matrices, predictor similarity grids, and half-matrices are
interchangeable inputs. Nothing here reads eval CSVs or vectors directly
except through :mod:`valuegen.ground_truth.matrices` and
:mod:`valuegen.predictors.store`.
"""
