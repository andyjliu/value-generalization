# value-generalization

This repository contains code and data associated with [Predicting Alignment Generalization with Value Representations](https://arxiv.org/abs/2610.12410). It can be used to reproduce the paper's results, or to measure and predict value generalization (how steering a model toward one value changes its behavior on other values) over custom value sets and models.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/andyjliu/value-generalization
cd value-generalization
./scripts/setup_env.sh
```

This builds the environment in `.venvs/core`, runs the tests, and creates two gitignored files to fill in:

- `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and `HF_API_KEY`.
- `configs/cluster.yaml`: data paths and SLURM partition/QOS names. Only needed for commands that submit jobs.

To rerun the tests later:

```bash
.venvs/core/bin/python -m pytest tests/      # runs on CPU
.venvs/core/bin/valuegen smoke --tests       # the same suite on a GPU node, through SLURM
```

The first run downloads a small test model (~1.5 GB).

Two optional environments are only needed to retrain weight-steering adapters (`./scripts/setup_env.sh --ws-train`) or to generate new spec-aligned SFT data (`--msm`). Tests that need them are skipped when they are not built.

The forks under `external/` (ConflictScope, persona vectors, weight steering, Model Spec Midtraining) are driven by `valuegen` and run in these environments; the setup instructions in their own READMEs are not needed.

## Reproducing the paper

To recompute the paper's tables and figures from the released data (CPU only, takes minutes, no API keys or SLURM):

```bash
.venvs/core/bin/python scripts/fetch_paper_data.py
.venvs/core/bin/python paper/reproduce/all.py     # or rq1.py, rq2.py, rq3.py
```

Each script checks its numbers against `paper/expected/` and writes outputs to `paper/out/`.

To regenerate the underlying data from scratch, launch the scripts in `paper/regenerate/` as controller jobs: `gt.sh` (ground-truth matrices), `rq1.sh` (predictor grids), `rq2.sh` (multi-value robustness), `rq3.sh` (taxonomy). This needs SLURM, GPUs, and API spend, and each script's header lists its hardware requirements. They call the helper scripts below in the right order, so nothing else has to be run by hand.

```bash
.venvs/core/bin/valuegen controller paper/regenerate/gt.sh
```

## Usage

The commands below assume the environment is active (`source .venvs/core/bin/activate`). The general flow is:

1. Build a ground-truth steerability matrix by training and evaluating one steered model per value. The paper's configs read inputs that must be in place first (see [Scripts](#scripts)):
   ```bash
   python scripts/fetch_eval_inputs.py           # evaluation scenarios, every config
   python scripts/prepare_sft_data.py -c configs/experiments/ground_truth/const_v3_sft_full66.yaml
   valuegen gt run --config configs/experiments/ground_truth/const_v3_sft_full66.yaml
   ```
   The SFT configs train on released data, so this is the easiest arm to rerun. The DPO configs (`const_v3_dpo_full49*`) rebuild their preference pairs by relabeling eight public preference datasets with a locally served judge, which is slow; `paper/regenerate/gt.sh` shows the full sequence.
2. Build a predicted similarity grid over the same value set with a cheap predictor (persona vectors, weight task vectors, sentence embeddings, ...):
   ```bash
   valuegen predict --method persona --value-set constitution_tenets_v3 \
                    --model value-generalization/neutral-sft-v3-qwen3-8b
   ```
3. Compare predictions against ground truth, or map and cluster a similarity grid:
   ```bash
   valuegen analyze correlate --target <gt_matrix_likert_normalized.npy> --pred <predicted.npy>
   valuegen analyze cluster --matrix <similarity.npy> --methods upgma,kmedoids,mds,hdbscan --maps
   ```

Commands that submit SLURM jobs print the job plan first; `--dry-run` writes the job scripts without submitting.

### Custom value sets and models

A value set is a JSON file `value_sets/[VALUE_SET_NAME].json` mapping each value's id to a one-line description of the behavior. Pass its name as `--value-set` or as `value_set:` in a config.

- **Predictors** need nothing else: `valuegen predict` generates its own elicitation data for the value set and probes whatever `--model` you name.
- **Ground truth** needs a config with two blocks. Copy `configs/experiments/ground_truth/const_v3_sft_full66.yaml` and change `value_set`, `values` and `models`:
  - `intervention:` says how each value's steered model is trained. `method: model_spec_aft` generates SFT data for each value from its description (needs the `--msm` environment and a generator model, set under `generation:`). The paper's SFT configs skip that generation only because `prepare_sft_data.py` puts the released data in place first.
  - `evaluation:` says what the steered models are scored on. For a new value set, replace the `scenarios:` path with a `pool:` block, which generates, filters and splits ConflictScope scenarios for your values; `tests/fixtures/configs/conflictscope_pairs_smoke.yaml` has a small example.

  Editing a config's `intervention` or `evaluation` block changes the run's ID, so results land in a new directory and nothing is overwritten.

## Scripts

`scripts/setup_env.sh` is the only script needed for setup. The rest fetch or prepare inputs for specific commands:

| script | what it does | needed for |
|---|---|---|
| `fetch_paper_data.py` | downloads the released intermediate results into `data/paper/` | `paper/reproduce/` |
| `fetch_eval_inputs.py` | downloads the ConflictScope evaluation scenarios into `data/scenarios/const_v3_cs/` | any `gt run` on the paper's configs |
| `prepare_sft_data.py` | writes the released per-tenet SFT data where `gt run` expects it | the SFT configs |
| `prepare_dpo_data.py` | writes the released per-tenet DPO pairs where the DPO configs' intervention expects them, without relabeling | `rq2.sh` |
| `build_ca_wf_pools.py`, `build_ca_wf_mt_pools.py` | build the Community Alignment and WildFeedback preference pools | the DPO configs |
| `build_model_view.py` | pins a Hub checkpoint at a revision under a local path | the paper's predictor runs (`rq1.sh`, `rq3.sh`) |
| `sentemb_behavior.py` | builds the Behavior-Embd predictor grid from existing persona pairs | `rq1.sh` |
| `taxonomy_labels.py` | labels the tenets under the two external taxonomies | `rq3.sh` |
| `export_paper_data.py` | copies rebuilt artifacts into the `data/paper/` layout | last step of each regenerate script |

## Data

The ground-truth matrices from the paper are in `data/gt/const_v3_*/matrices/`: DPO (49 tenets) and SFT (66 tenets) for Qwen3-8B, Olmo-3 7B, Qwen3-30B-A3B and Olmo-3 32B. Rows are the steered value and columns the evaluated value. `*_likert_normalized.npy` is the paper's metric: the change from the unsteered rate, scaled by the room each cell had to move, in [-1, 1]. `*_raw.npy` is the steered model's rate. Everything else is on the [Hugging Face Hub](https://huggingface.co/value-generalization) under `value-generalization/`:

- `neutral-sft-v3-{qwen3-8b,olmo3-7b,olmo3-32b,qwen3-30b-a3b}`: neutral SFT checkpoints that the DPO arm starts from
- `constitution-v3-sft`, `constitution-v3-dpo`: per-tenet SFT data and DPO pairs
- `neutral-sft-tulu3-v3`: the value-neutral SFT mix
- `paper-data`: intermediate results that are needed to run all scripts in `paper/reproduce`
- `conflictscope-eval-constitution-tenets-v3`: ConflictScope evaluation scenarios
