# Work plan

## Implementation status

The Provo and OneStop asset acquisition, canonical grids, ET1
inference/alignment, fixed-sigma handling, symmetric/asymmetric redistribution,
OB1 baseline wrapper, metrics, clustered paired inference, output tables,
plots, manifests, and Python CLI are implemented.

Executed Provo gates include checksum verification, all 55 ET1 passages,
same-seed and different-seed OB1 checks, all 55 OB1 passages for one reader,
both redistribution paths with synthetic smoke widths and the effective
`0.41553/3.46115` pair, and both Human TRT definitions.

Executed OneStop gates include the checksum-verified official Ordinary ZIP,
streamed strict preparation, exact text-variant audit, the
162-paragraph/19,440-word canonical grid, and the 95-position OB1
compatibility audit. The complete OneStop 100-seed OB1 simulation and joined
scientific evaluation remain pending. Preparation and smoke results must not
be described as completed OneStop alignment results.

## 1. Frozen research question

Does the fixed OASST1-learned asymmetric redistribution change frozen ET1 TRT
so that its word-level allocation on external text is closer to:

1. participant-averaged Human TRT; and
2. OB1-reader simulated total viewing time?

Neither Provo nor OneStop Human TRT is used to fit ET1, the symmetric control,
the learned sigmas, or OB1.

### Conditions

| ID | Condition | Construction |
|---|---|---|
| `human_unconditional` | Human TRT primary | Mean released TRT over all retained readers, including zeros |
| `human_conditional` | Human TRT sensitivity | Mean released TRT over positive-dwell readers; all-missing words excluded |
| `et1_raw` | Frozen ET1 | Native T5-token TRT summed into corpus words |
| `et1_symmetric` | Symmetric control | ET1 followed by the width-matched symmetric kernel |
| `et1_asymmetric` | Proposed | ET1 followed by fixed `sigma_left` and `sigma_right` |
| `ob1` | Cognitive baseline | Published OB1 no-predictability condition |

For each asymmetric pair:

\[
\sigma_{\mathrm{sym}}
=
\sqrt{\frac{\sigma_L^2+\sigma_R^2}{2}}.
\]

This preserves the mean squared half-kernel width and fits nothing to the
evaluation corpus.

## 2. Python-only entry point

`cognitive_model_comparsion/main.py` implements:

```text
setup
audit
prepare-provo
prepare-onestop
extract-sigmas
predict-et1
simulate-ob1
evaluate
run
```

No shell script is required. Full runs write resolved paths, checksums,
arguments, random seeds, package versions, Git state, and output provenance.

## 3. Assets

### Provo and model sources

- official Provo eye-tracking CSV;
- official Provo predictability CSV;
- pinned OB1–Provo source snapshot;
- pinned TorontoCL ET2 reference source and processed Provo table;
- official University of Nottingham `SUBTLEX-UK.txt.zip`;
- pinned ET1 checkpoint and native T5 tokenizer.

The TorontoCL ET2 Provo table is a provenance fixture only. Its standardized
values never replace Human milliseconds.

### OneStop Ordinary Interest Area ZIP

```text
source: https://osf.io/download/xkgfz/
landing page: https://osf.io/zn9sq/
bytes: 177291322
sha256: 8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9
member: ia_Paragraph_ordinary.csv
```

The member is 2,455,810,901 bytes uncompressed. Preparation reads required
columns in chunks directly from the ZIP and never writes the expanded CSV.
See `PROVENANCE.md` for short verbatim source quotations and licenses.

## 4. Canonical Provo grid

Reproduce the published OB1 evaluator's position corrections:

- move the misplaced `evolution` record;
- shift affected ranges in passages 3 and 13;
- remove the malformed standalone `Ñ` in passage 36;
- exclude three missing Human positions in passage 55;
- omit the first passage word because Provo does not report its target
  reading measure.

Acceptance criteria:

- 55 passages;
- 2,686 corrected Human–OB1 positions;
- unique `(passage_id, word_id)` coordinates;
- exact declared normalized-string agreement;
- raw and corrected IDs retained;
- every exclusion written with its reason.

## 5. Canonical OneStop grid

The primary OneStop condition uses:

- the official Ordinary/Gathering Paragraph Interest Area archive;
- `difficulty_level=Adv`;
- non-practice, non-repeated rows;
- `question_preview=False`;
- released `IA_DWELL_TIME` without rewriting `IA_SKIP`;
- exact whitespace-tokenized word coordinates.

Strict observed dimensions:

| Check | Value |
|---|---:|
| Participants | 180 |
| Article clusters | 30 |
| Advanced paragraphs | 162 |
| Canonical whitespace words | 19,440 |
| Participant-paragraph trials before variant selection | 4,859 |
| Participant-paragraph trials retained | 4,759 |
| Paragraphs with multiple exact variants | 11 |
| Minority-variant trials excluded | 100 |
| Retained reader-count range | 17–30 |
| Punctuation-only OB1-incompatible positions | 95 |
| Paragraphs with zero OB1 transformations | 107 |
| Article clusters represented by clean paragraphs | 28 |
| Paragraphs containing at least one transformed position | 55 |

For each base paragraph, select the exact text variant read by the most
participants, then apply deterministic lexical and SHA-256 tie-breaks. Never
use Human TRT, ET1, or OB1 for selection. Save every variant and excluded
participant count in `onestop_variant_audit.csv`.

Do not delete or merge punctuation-only words from the canonical Human/ET1
grid. Mark all 95 as `ob1_evaluable=False`, record their deterministic OB1
token transformations, and exclude the same positions from every method on
the common Human–ET1–OB1 metric grid.

As a secondary guard against a surrogate token changing neighboring OB1
fixations, `--with-ob1-clean-passage-sensitivity` recomputes every summary and
paired contrast using only the 107 paragraphs whose
`ob1_incompatible_words_excluded` count is zero. The 55 affected paragraphs
are excluded wholesale in this sensitivity only. The retained paragraphs
span 28 article clusters, which remain the resampling units. The
162-paragraph analysis remains primary.

## 6. Human TRT targets

Primary:

\[
H_i=\frac{1}{R_i}\sum_{r=1}^{R_i}TRT_{ir},
\]

where released zero-dwell values remain zero.

Conditional sensitivity:

\[
H_i^{\mathrm{fixated}}
=
\mathrm{mean}\{TRT_{ir}:TRT_{ir}>0\}.
\]

For OneStop, a word with no positive-dwell retained reader is stored as
missing for the conditional target. It remains in the unconditional target
and is excluded only from the conditional passage metric.

## 7. ET1 inference, word mapping, and special-token sensitivity

ET1 stays frozen and runs one passage at a time with the pinned native T5
tokenizer. For each passage:

1. tokenize the exact canonical stimulus with character offsets;
2. assign every non-special T5 token to exactly one whitespace word;
3. apply raw, symmetric, and asymmetric token-level conditions;
4. sum assigned subtoken values:

\[
E_i=\sum_{t\in word(i)}TRT_t^{ET1};
\]

5. save tokens, offsets, special flags, word assignments, predictions, and
   mass audits.

The primary redistribution mask reproduces production semantics and includes
the attended T5 EOS position. `--with-special-token-sensitivity` also runs a
secondary path that masks special tokens before redistribution. Both paths
exclude special tokens only when aggregating values into words. The
sensitivity does not change sum aggregation.

Mass auditing records:

- selected-valid mass before and after redistribution;
- full attention-mask mass;
- word-assigned mass;
- unassigned-special mass;
- common evaluable-grid mass and retention relative to raw ET1.

## 8. Sigma contract

The supplied effective pair:

```text
sigma_left: 0.41553
sigma_right: 3.46115
source reward accuracy: 0.76675
source condition: ET1 + GazeConcat, TRT only
```

is used directly and recorded with a stable checkpoint label. It is not
re-estimated on Provo or OneStop. Because the effective scalars are available,
the unrelated Llama/reward-model weights are not required for this external
validation.

If checkpoints are used instead, the extractor must locate exactly one
`log_sigma_left` and one `log_sigma_right`, apply
`exp(log_sigma) + min_sigma`, record the checkpoint SHA-256, and reject an
unconfirmed initial `1.0/1.0` pair.

## 9. OB1 simulation

The wrapper preserves the pinned OB1 equations and published
no-predictability baseline. Allowed changes are path isolation, deterministic
random seeds, portable execution, structured fixation output, and checked
word-grid aggregation. OB1 parameters must not be tuned on Human TRT.

For each corpus:

- simulate all 55 Provo passages or all 162 OneStop paragraphs;
- use fixed seeds `0..99`;
- run independent single-threaded worker subprocesses;
- save every fixation and regression;
- sum every fixation duration on a word to obtain simulation-level TVT;
- fill unfixated evaluable words with zero before averaging simulations.

\[
O_i^{(s)}
=
\sum_{f:word(f)=i}duration(f),
\qquad
O_i=\frac{1}{100}\sum_{s=1}^{100}O_i^{(s)}.
\]

The 100 seeds are stochastic OB1 replications. They are not 100 Human
participants and are not fitted models for the 84 Provo or 180 OneStop
readers.

## 10. Metrics

Spearman uses finite word values directly. JS and word-order Wasserstein clip
negative model predictions to zero, normalize nonnegative passage mass to
one, and fail rather than add an undisclosed epsilon when mass is zero.

Per passage:

- Human Spearman:
  \(\rho(H,E)\), \(\rho(H,S)\), \(\rho(H,A)\), \(\rho(H,O)\);
- OB1 consistency:
  \(\rho(E,O)\), \(\rho(S,O)\), \(\rho(A,O)\);
- Jensen–Shannon divergence:
  \(JS(H,E)\), \(JS(H,S)\), \(JS(H,A)\), \(JS(H,O)\);
- `word_order_wasserstein` over normalized word positions \(i/(n-1)\).

Word-order Wasserstein measures how far allocation mass must move along the
ordered word axis. It is not fixation-coordinate distance and does not
evaluate a scanpath.

## 11. Paired inference

Use 10,000 paired resamples and a fixed seed:

- Provo resampling unit: 55 passages;
- OneStop resampling unit: 30 article clusters, retaining all paragraphs from
  a sampled article together.

For multiple RM seeds, compute passage metrics per RM seed, average
seed-specific deltas within passage, then resample the corpus-specific unit.
Never treat seed-by-passage rows as independent observations.

Report:

- mean paired improvement, with positive always meaning better;
- percentile 95% confidence interval;
- two-sided paired sign-flip `permutation_p_two_sided`;
- passage and, for OneStop, cluster counts;
- RM checkpoint count and all exclusions.

The sign-flip test flips paired-difference signs by passage for Provo and by
article cluster for OneStop. It is exact when at most 20 units and all sign
patterns are requested; otherwise it is Monte Carlo with a plus-one
correction. Do not report a bootstrap-derived p-value.

## 12. Outputs

Core output:

```text
run_manifest.json
checkpoint_sigmas.json
checkpoint_sigmas.csv
et1/et1_token_values.csv
et1/et1_word_values.csv
et1/et1_mass_audit.csv
et1/et1_inference_audit.json
ob1/ob1_fixations.csv
ob1/ob1_word_values_by_simulation.csv
ob1/ob1_word_values.csv
ob1/ob1_token_transformations.csv
evaluation_*/word_level_values.csv
evaluation_*/passage_metrics.csv
evaluation_*/result_table.csv
evaluation_*/seed_result_table.csv
evaluation_*/bootstrap_summary.csv
evaluation_*/human_spearman_by_passage.png
evaluation_*/ob1_clean_passages/*
```

OneStop preparation:

```text
onestop_passages.csv
onestop_words.csv
onestop_participant_words.csv
onestop_variant_audit.csv
onestop_prepare_audit.json
```

`result_table.csv` uses:

| Method | Human Spearman ↑ | JS divergence ↓ | Word-order Wasserstein ↓ | OB1 Spearman ↑ |
|---|---:|---:|---:|---:|
| ET1 raw | | | | |
| ET1 + symmetric | | | | |
| ET1 + learned asymmetric | | | | |
| OB1 baseline | | | | — |

## 13. Exact direct-sigma commands

Provo:

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

OneStop preparation and run:

```bash
python cognitive_model_comparsion/main.py setup --corpus onestop

python cognitive_model_comparsion/main.py prepare-onestop \
  --input-zip cognitive_model_comparsion/data/raw/onestop/ia_Paragraph_ordinary.csv.zip \
  --output-dir cognitive_model_comparsion/data/processed/onestop \
  --chunksize 50000 \
  --strict

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

## 14. Verification and interpretation

Required gates:

- asset checksum and archive-member checks;
- canonical coordinate uniqueness and text alignment;
- Provo 55/2,686 and OneStop 180/30/162/19,440 strict dimensions;
- OneStop exact variant and 95-position OB1 transformation audits;
- ET1 token coverage, special-token policy, and mass conservation;
- OB1 same-seed determinism, different-seed variation, regression inclusion,
  and skipped-word completion;
- toy metric, percentile interval, paired sign-flip, and article-cluster tests;
- tables regenerated only from saved machine-readable outputs.

Interpretation:

- Human and OB1 improve: claim greater Human correspondence and directional
  OB1 consistency;
- OB1 improves but Human does not: claim only cognitive-model consistency;
- neither improves: retain downstream reward-model utility but remove a
  quantitative Human-gaze/perceptual-span interpretation.

Human TRT is overt fixation allocation. It must not be described as a direct
measurement of covert perceptual span. OB1 letter coordinates must not be
numerically equated with T5-token sigma.
