# Cognitive model comparison

This directory is the isolated, Python-only implementation of the
reviewer-requested Human Provo–ET1–redistribution–OB1 comparison. It does not
train or alter the OASST1 reward model.

The directory name intentionally preserves the requested spelling:
`cognitive_model_comparsion`.

## What is implemented

- checksum-verified download of the official Provo eye-tracking and
  predictability files;
- checksum-verified download of the official SUBTLEX-UK text table;
- pinned downloads of the OB1–Provo and TorontoCL ET2 repositories;
- exact reconstruction of TorontoCL's 2,659-row processed Provo table from the
  official raw Provo release;
- the 55-passage, 2,686-word canonical Human Provo evaluation grid;
- unconditional TRT, in which skips remain zero, and conditional TRT over
  fixated readers;
- frozen ET1 inference with native pinned T5 offsets and exact word
  aggregation;
- reward-checkpoint sigma discovery, extraction, checksum recording, and
  initial-value rejection;
- direct effective/log-sigma input with source-accuracy provenance when the
  original reward checkpoint is unavailable;
- width-matched symmetric and learned asymmetric redistribution through the
  repository's production `AsymGaussianRedistributor`;
- an isolated deterministic wrapper around the pinned OB1 scientific source;
- TVT aggregation over all fixations, including regressions, with skipped
  words filled by zero;
- passage-level Spearman, Jensen–Shannon divergence, and normalized-position
  Wasserstein distance;
- paired passage bootstrap, grand/seed-specific result CSVs, and a
  passage-level boxplot;
- one Python CLI for setup, component runs, and the full experiment.

No `.sh` file or `gdown` is used.

## Environment and asset setup

From the repository root:

```bash
python -m pip install -r requirements.txt
python cognitive_model_comparsion/main.py setup
python cognitive_model_comparsion/main.py audit
python -m pytest -q cognitive_model_comparsion/tests
```

`setup` downloads every re-downloadable data/source asset, validates its exact
size and SHA-256 digest, builds the canonical Provo tables, downloads the
pinned ET1 checkpoint if absent, and loads the pinned `t5-small` tokenizer.
The downloaded files, third-party trees, runtime cache, and outputs are
excluded by `.gitignore`.

Use `--skip-et1` only when preparing an OB1-only CPU machine:

```bash
python cognitive_model_comparsion/main.py setup --skip-et1
```

Offline integrity verification after setup:

```bash
python cognitive_model_comparsion/download_assets.py --verify-only
```

## Full experiment

The learned sigma values must come from the actual reported OASST1 reward-model
checkpoints. They are never fitted on Provo. For three seeds:

```bash
python cognitive_model_comparsion/main.py run \
  --checkpoint /path/to/seed41/checkpoint \
  --checkpoint-id seed41 \
  --checkpoint /path/to/seed42/checkpoint \
  --checkpoint-id seed42 \
  --checkpoint /path/to/seed43/checkpoint \
  --checkpoint-id seed43 \
  --seeds 0:100 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir cognitive_model_comparsion/outputs/provo_ob1_full
```

If the exact effective sigma values were retained but the unrelated
reward-model weights were not, run the same experiment directly:

```bash
python cognitive_model_comparsion/main.py run \
  --sigma-left 0.41553 \
  --sigma-right 3.46115 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76675 \
  --checkpoint-id oasst1_et1_learned_sigma \
  --seeds 0:100 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir cognitive_model_comparsion/outputs/provo_ob1_full
```

Repeat `--sigma-left`, `--sigma-right`, `--sigma-source-accuracy`, and
`--checkpoint-id` once per reported RM seed. Direct values must be the exact
best-checkpoint values for the ET1 asymmetric condition; they are recorded in
the run manifest and sigma tables but are never optimized on Provo.

The checkpoint argument may point directly to a state file or to a directory
containing one unambiguous `adapter_model.safetensors`,
`adapter_model.bin`, `model.safetensors`, or `pytorch_model.bin`.

The command:

1. verifies/downloads the Provo, SUBTLEX-UK, OB1, and ET2-reference assets;
2. rebuilds the canonical Human grid;
3. extracts and freezes each checkpoint's learned left/right sigma;
4. runs frozen ET1 once per passage and applies both redistribution controls;
5. runs OB1 baseline readers for fixed seeds `0..99`;
6. evaluates both unconditional and conditional Human TRT;
7. writes the command, package versions, Git state, checkpoints, hashes,
   sigmas, seeds, parameters, alignments, fixations, metrics, plots, and
   audits.

If a checkpoint contains multiple valid sigma copies, inspect the prefixes
listed in the error and pass the intended one with `--sigma-prefix`. The
extractor refuses an unchanged `1.0/1.0` pair by default; do not bypass this
with `--allow-initial-sigmas` unless the checkpoint has been manually
confirmed to be the intended trained model.

## Component commands

```bash
python cognitive_model_comparsion/main.py prepare-provo

python cognitive_model_comparsion/main.py extract-sigmas \
  --checkpoint /path/to/checkpoint \
  --checkpoint-id seed41 \
  --output-dir cognitive_model_comparsion/outputs/sigmas

python cognitive_model_comparsion/main.py predict-et1 \
  --sigma-json cognitive_model_comparsion/outputs/sigmas/checkpoint_sigmas.json \
  --output-dir cognitive_model_comparsion/outputs/et1

python cognitive_model_comparsion/main.py predict-et1 \
  --sigma-left 0.41553 \
  --sigma-right 3.46115 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76675 \
  --checkpoint-id oasst1_et1_learned_sigma \
  --output-dir cognitive_model_comparsion/outputs/et1

python cognitive_model_comparsion/main.py simulate-ob1 \
  --seeds 0:100 \
  --n-trials 55 \
  --output-dir cognitive_model_comparsion/outputs/ob1

python cognitive_model_comparsion/main.py evaluate \
  --et1-dir cognitive_model_comparsion/outputs/et1 \
  --ob1-dir cognitive_model_comparsion/outputs/ob1 \
  --human-target human_trt_unconditional \
  --bootstrap-samples 10000 \
  --output-dir cognitive_model_comparsion/outputs/evaluation
```

Running `predict-et1` without a checkpoint or sigma JSON is an intentional
raw-ET1-only path.

## Verified smoke coverage

The following were executed locally on the checked-out code:

| Gate | Observed |
|---|---:|
| Asset SHA-256 verification | passed |
| Canonical passages / words | 55 / 2,686 |
| ET2 reconstructed keys | 2,659 / 2,659 |
| ET2 maximum feature difference | `2.70e-13` |
| Frozen ET1 passages / word rows | 55 / 2,686 |
| ET1 token rows / unassigned non-special tokens | 3,715 / 0 |
| OB1 same-seed repeat | exact |
| OB1 different-seed fixation rows | 58 / 57 |
| OB1 55-passage, one-reader runtime | 280.18 s |
| OB1 fixation / word rows | 2,783 / 2,686 |
| OB1 regression fixations / zero-TVT rows | 619 / 388 |
| Synthetic redistribution mass checks | 110 / 110 passed |
| Confirmed direct-sigma ET1 passages / word rows | 55 / 2,686 |
| Confirmed direct-sigma mass checks | 110 / 110 passed |
| Cognitive-comparison unit tests | 33 passed |

The redistribution integration smoke used explicitly synthetic widths
`sigma_left=1.3` and `sigma_right=2.1` only to exercise all 55 passages,
mass-conservation checks, metrics, bootstrap, CSV generation, and plotting.
Its metric values are not scientific results and must not be reported.

The final learned-asymmetric manuscript result remains pending until the
100-simulation OB1 run and Human evaluation finish. The direct effective
values `0.41553/3.46115` have passed all 55 ET1 passages and redistribution
mass checks, but those integration outputs are not manuscript metrics.

## Scope and interpretation

This comparison tests whether OASST1-learned redistribution changes ET1's
word-level allocation toward Human Provo TRT and OB1-generated TVT. It does
not:

- fit any ET1, Gaussian, or OB1 parameter to Provo Human TRT;
- use standardized TorontoCL ET2 values as Human milliseconds;
- compare OB1 letter coordinates numerically with T5-token sigma;
- treat Human TRT as a direct measure of covert perceptual span;
- claim that OB1 contains a physical viewing-distance parameter;
- replace the existing OASST1 downstream reward-model results.

See `PROVENANCE.md` for word-for-word paper evidence and exact source hashes,
and `WORK_PLAN.md` for the frozen analysis contract.
