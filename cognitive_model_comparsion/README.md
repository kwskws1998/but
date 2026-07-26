# Cognitive model comparison

This directory contains the isolated, Python-only Human gaze–ET1
redistribution–OB1 comparison requested by the reviewer. It supports both the
55-passage Provo corpus and the 162-paragraph OneStop Ordinary Reading
extension. It does not train or load the Llama reward-model backbone.

The directory name intentionally preserves the requested spelling:
`cognitive_model_comparsion`.

## Implemented experiment

- checksum-verified official Provo eye-tracking and predictability data;
- checksum-verified official OneStop Ordinary Paragraph Interest Area ZIP;
- streaming OneStop preparation directly from the ZIP without extracting its
  2.46 GB CSV;
- pinned OB1–Provo source, TorontoCL ET2 reference source, and SUBTLEX-UK;
- canonical Human grids:
  - Provo: 55 passages and 2,686 corrected evaluable word positions;
  - OneStop: 180 Ordinary/Gathering participants, 30 article clusters,
    162 Advanced paragraphs, and 19,440 exact whitespace words;
- Human unconditional TRT, retaining `IA_DWELL_TIME=0`, and conditional TRT,
  excluding words with no positive-dwell reader;
- frozen ET1 inference using the pinned native T5 tokenizer;
- paper-faithful ET1 word mapping: sum all native T5-token TRT values assigned
  by character offsets to each whitespace word;
- width-matched symmetric and learned asymmetric redistribution using the
  production `AsymGaussianRedistributor`;
- the production-faithful primary mask, which includes the T5 EOS position
  during redistribution, plus a requested special-token-excluded sensitivity;
- per-passage mass audits before and after redistribution, including
  word-assigned, unassigned-special, and final evaluable-grid mass;
- deterministic, parallel OB1 stochastic simulations and TVT aggregation over
  every fixation, including regressions;
- passage-level Human Spearman, Jensen–Shannon divergence,
  `word_order_wasserstein`, and OB1 Spearman;
- percentile 95% confidence intervals from paired resampling and two-sided
  paired sign-flip tests;
- passage-level resampling for Provo and article-cluster resampling over the 30
  OneStop articles;
- optional OneStop clean-passage sensitivity restricted to 107 paragraphs
  with no punctuation-only OB1 token transformation;
- one Python CLI for setup, component runs, and complete experiments.

No `.sh` file or `gdown` is used.

## Environment and setup

From the repository root:

```bash
python -m pip install -r cognitive_model_comparsion/requirements.txt

python cognitive_model_comparsion/main.py setup --corpus provo
python cognitive_model_comparsion/main.py audit --corpus provo

python cognitive_model_comparsion/main.py setup --corpus onestop
python cognitive_model_comparsion/main.py audit --corpus onestop

python -m pytest -q cognitive_model_comparsion/tests
```

`setup --corpus onestop` downloads the official 169 MiB Ordinary Interest Area
ZIP, verifies its byte count and SHA-256 digest, and reads only required
columns in chunks. It does not extract the 2.46 GB CSV. Use
`--onestop-chunksize` only when machine memory requires a different chunk
size.

Use `--skip-et1` only when preparing an OB1-only CPU machine. The isolated
requirements intentionally omit Llama-training packages such as
`bitsandbytes`, `deepspeed`, `trl`, and `wandb`.

Offline asset verification:

```bash
python cognitive_model_comparsion/download_assets.py --verify-only
```

## Provo rebuttal run

The supplied direct values are the effective sigmas learned by the
ET1 + GazeConcat, TRT-only OASST1 reward-model condition. They remain fixed on
Provo; neither sigma is fitted to Human Provo TRT.

```bash
CUDA_VISIBLE_DEVICES=0 python cognitive_model_comparsion/main.py run \
  --corpus provo \
  --sigma-left 0.41553 \
  --sigma-right 3.46115 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76675 \
  --checkpoint-id oasst1_et1_trt_only_sigma \
  --seeds 0:100 \
  --workers 32 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --with-special-token-sensitivity \
  --output-dir cognitive_model_comparsion/outputs/provo_ob1_sigma076675
```

The primary `et1` directory reproduces the production redistribution mask,
including T5 EOS. The additional `et1_special_excluded` directory masks
special tokens before redistribution. Both paths still exclude special tokens
when T5 values are summed into words.

The native T5 subtoken-to-word aggregation is unchanged in both paths:

\[
E_i=\sum_{t\in word(i)}TRT_t^{ET1}.
\]

The sensitivity changes only which valid token positions participate in the
Gaussian redistribution; it does not replace sum aggregation with averaging,
first-subtoken assignment, or repeated word values.

## OneStop rebuttal run

Prepare and inspect the official OneStop grid first:

```bash
python cognitive_model_comparsion/main.py setup --corpus onestop

python cognitive_model_comparsion/main.py prepare-onestop \
  --input-zip cognitive_model_comparsion/data/raw/onestop/ia_Paragraph_ordinary.csv.zip \
  --output-dir cognitive_model_comparsion/data/processed/onestop \
  --chunksize 50000 \
  --strict

python cognitive_model_comparsion/main.py audit --corpus onestop
```

Then run the same frozen ET1 sigmas and published OB1 baseline on the OneStop
Advanced paragraphs:

```bash
CUDA_VISIBLE_DEVICES=0 python cognitive_model_comparsion/main.py run \
  --corpus onestop \
  --sigma-left 0.41553 \
  --sigma-right 3.46115 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76675 \
  --checkpoint-id oasst1_et1_trt_only_sigma \
  --seeds 0:100 \
  --workers 32 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --with-special-token-sensitivity \
  --with-ob1-clean-passage-sensitivity \
  --output-dir cognitive_model_comparsion/outputs/onestop_ob1_sigma076675
```

This full OneStop 100-seed scientific run remains pending. The code and
canonical data audit being complete must not be reported as a finished
Human/OB1 alignment result.

`0:100` means 100 stochastic OB1 replications with fixed random seeds. They are
not 100 fitted reader models and are not the 180 OneStop participants. Human
TRT is averaged from the actual participant rows; OB1 TVT is averaged from
independent simulations of the same paragraph text.

## OneStop preparation contract

The OneStop primary condition keeps only:

- official Ordinary Reading paragraph rows;
- `difficulty_level=Adv`;
- non-practice, non-repeated trials;
- `question_preview=False`.

The archive contains 11 base paragraphs with more than one exact on-screen
text variant. For each base paragraph, the code selects the exact variant read
by the largest number of participants, then applies deterministic lexical and
SHA-256 tie-breaks. This retains 4,759 of 4,859 participant-paragraph trials
and records all 100 excluded minority-variant trials in
`onestop_variant_audit.csv`. No Human TRT value is used to select the text.

The 19,440-word Human grid is never silently altered. Ninety-five
punctuation-only whitespace tokens normalize to an empty OB1 token. They are
marked `ob1_evaluable=False`, written to the OB1 transformation audit, and
excluded from every Human/ET1/OB1 metric on the common evaluation grid.

Because replacing an OB1-incompatible token inside a simulated paragraph can
still affect neighboring OB1 fixations, the optional
`--with-ob1-clean-passage-sensitivity` analysis removes all 55 affected
paragraphs and reruns summaries on the 107 paragraphs with zero
transformations. Those paragraphs span 28 article clusters, and the nested
analysis resamples those 28 clusters rather than the 30 clusters in the
primary analysis. Its tables are written below each evaluation directory as
`ob1_clean_passages/`. This is a sensitivity analysis; it does not replace
the 162-paragraph primary result.

The primary Human target averages `IA_DWELL_TIME` across all retained readers,
including zero-dwell words. The conditional sensitivity averages only positive
dwell times; a word with no positive-dwell reader is `NaN` and is excluded
from that passage's conditional metric only.

## Metrics and inference

All four model allocations are compared at the common word positions within
each passage:

- Human Spearman: rank correspondence with participant-averaged TRT;
- JS divergence: shape difference between unit-normalized allocations;
- word-order Wasserstein: transport distance along normalized word order,
  not a fixation-coordinate or scanpath metric;
- OB1 Spearman: rank correspondence with OB1 simulated TVT.

`result_table.csv` reports method means and percentile 95% intervals.
`bootstrap_summary.csv` reports paired improvements, percentile 95% intervals,
and `permutation_p_two_sided` from the paired sign-flip test. There is no
bootstrap-derived p-value.

For Provo, the paired unit is a passage. For OneStop, all paragraphs from the
same article share `cluster_id`; the bootstrap and sign-flip test resample or
flip the 30 article clusters so paragraphs from one article are not treated as
independent.

## Component commands

Every component accepts `--corpus provo` or `--corpus onestop` where relevant:

```bash
python cognitive_model_comparsion/main.py predict-et1 \
  --corpus onestop \
  --sigma-left 0.41553 \
  --sigma-right 3.46115 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76675 \
  --checkpoint-id oasst1_et1_trt_only_sigma \
  --output-dir cognitive_model_comparsion/outputs/onestop_et1

python cognitive_model_comparsion/main.py predict-et1 \
  --corpus onestop \
  --sigma-left 0.41553 \
  --sigma-right 3.46115 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76675 \
  --checkpoint-id oasst1_et1_trt_only_sigma \
  --exclude-special-tokens-from-redistribution \
  --output-dir cognitive_model_comparsion/outputs/onestop_et1_special_excluded

python cognitive_model_comparsion/main.py simulate-ob1 \
  --corpus onestop \
  --seeds 0:100 \
  --workers 32 \
  --n-trials 162 \
  --output-dir cognitive_model_comparsion/outputs/onestop_ob1

python cognitive_model_comparsion/main.py evaluate \
  --corpus onestop \
  --et1-dir cognitive_model_comparsion/outputs/onestop_et1 \
  --ob1-dir cognitive_model_comparsion/outputs/onestop_ob1 \
  --human-target human_trt_unconditional \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --with-ob1-clean-passage-sensitivity \
  --output-dir cognitive_model_comparsion/outputs/onestop_evaluation
```

Running `predict-et1` without a sigma source is an intentional raw-ET1-only
diagnostic path. A full `run` requires a checkpoint or direct sigma pair.

## Interpretation limits

The comparison tests whether the OASST1-trained redistribution changes frozen
ET1 word allocation toward Human TRT and OB1 TVT on unseen text. It does not:

- refit ET1, sigma, or OB1 on either Human corpus;
- treat OB1 virtual-reader count as Human sample size;
- compare OB1 letter coordinates numerically with T5-token sigma;
- treat word-order Wasserstein as a scanpath measure;
- treat Human TRT as a direct measure of covert perceptual span;
- claim that OB1 has a physical viewing-distance parameter;
- replace the existing OASST1 reward-accuracy experiment.

See `PROVENANCE.md` for source quotes, exact hashes, and audit evidence, and
`WORK_PLAN.md` for the frozen analysis contract.
