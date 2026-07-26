# Work plan

## Implementation status

The data acquisition, canonical grid, ET1 inference/alignment, checkpoint
sigma extractor, symmetric/asymmetric redistribution, OB1 baseline wrapper,
metrics, paired bootstrap, table, plot, manifests, and Python CLI are
implemented.

Executed gates include all asset checks, all 55 ET1 passages, same-seed and
different-seed OB1 checks, all 55 OB1 passages for one reader, both
redistribution paths with synthetic smoke widths and the confirmed direct
effective widths `0.41553/3.46115`, both Human TRT definitions, and the test
suite. One result-generating run remains intentionally pending:

1. the final 100-simulation OB1 run and joined Human evaluation.

No synthetic-sigma, direct-sigma integration, or one-reader metric is a
manuscript result.

## 1. Frozen experimental contract

### Research question

Does the OASST1-learned asymmetric redistribution change frozen ET1 TRT so
that its word-level allocation is closer to:

1. participant-averaged Provo human TRT; and
2. OB1-reader simulated total viewing time?

The Provo labels will never be used to fit ET1, sigma, the symmetric control,
or OB1.

### Conditions

| ID | Condition | Construction |
|---|---|---|
| `human_unconditional` | Human Provo TRT | Mean `IA_DWELL_TIME` over observed readers, retaining skip values of 0 |
| `et1_raw` | Frozen ET1 | Native T5-token TRT summed into corrected Provo word positions |
| `et1_symmetric` | Symmetric control | Frozen ET1 followed by a fixed symmetric kernel derived from each OASST1 checkpoint |
| `et1_asymmetric` | Proposed | Frozen ET1 followed by that checkpoint's learned left/right sigmas |
| `ob1_baseline` | Cognitive baseline | Published OB1 no-predictability condition, 100 deterministic virtual readers |

The primary OB1 condition will be the paper's published no-predictability
baseline. This isolates OB1's visual, lexical, attentional, and oculomotor
machinery and avoids adding a second neural language model to the cognitive
baseline. A cloze-0.2 sensitivity run may be added because it is also a
published condition and can be generated from the already downloaded Provo
norms.

The repository does not have one unambiguous “default OB1 condition”:
`parameters.py` hardcodes GPT-2 with weight 0.1, the paper reports ten
conditions, and the included per-condition JSON files use weight 0.2.
Therefore the final manuscript must name the selected condition explicitly
instead of calling it “default OB1.”

### Symmetric control

For each reward-model checkpoint:

\[
\sigma_{\mathrm{sym}}
=
\sqrt{\frac{\sigma_L^2+\sigma_R^2}{2}}
\]

Set both widths to this value. This preserves the mean squared half-kernel
width without fitting anything to Provo. Record the learned log-sigmas,
converted sigmas including `min_sigma`, and derived symmetric sigma.

## 2. Python entry point

`cognitive_model_comparsion/main.py` implements:

```text
setup
audit
prepare-provo
extract-sigmas
predict-et1
simulate-ob1
evaluate
run
```

The full invocation is:

```bash
python cognitive_model_comparsion/main.py run \
  --checkpoint <seed-41-checkpoint> \
  --checkpoint-id seed41 \
  --checkpoint <seed-42-checkpoint> \
  --checkpoint-id seed42 \
  --checkpoint <seed-43-checkpoint> \
  --checkpoint-id seed43 \
  --seeds 0:100 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir cognitive_model_comparsion/outputs/provo_ob1_full
```

No `.sh` file is required. Every resolved path, checksum, model
parameter, random seed, package version, and output filename will be written
to a run manifest.

## 3. Asset setup

### Already complete

- official Provo eye-tracking CSV;
- official Provo predictability CSV;
- pinned OB1–Provo source snapshot;
- pinned TorontoCL ET2 training-code snapshot and processed Provo table;
- official University of Nottingham `SUBTLEX-UK.txt.zip` and extracted table;
- exact reconstruction of all 2,659 ET2 Provo rows from the official raw
  Provo CSV;
- checksums, byte sizes, source URLs, license audit;
- raw schema and published-position correction audit.

The official Nottingham text archive contains the exact tab-delimited format
expected by OB1, so no Excel conversion or frequency-resource substitution is
performed. The archive and extracted table are both checksum-pinned.

The ET2 repository's processed `data/provo.csv` is not an input to the
Human–OB1 evaluation. It is a provenance reference and a regression fixture
for the ET2 preprocessing audit. Its standardized gaze values must never
replace raw Provo milliseconds.

## 4. Build the canonical Provo word grid

`data/processed/provo_words.csv` contains one row per evaluable word and at
least:

```text
passage_id_raw
passage_id_zero_based
word_number_raw
word_id_zero_based
word_raw
word_clean
character_start
character_end
human_reader_count
human_skip_count
human_trt_unconditional
human_trt_conditional
alignment_status
```

Reproduce the public OB1 evaluator's corrections:

- move the misplaced `evolution` record;
- shift the affected ranges in passages 3 and 13;
- remove the malformed standalone `Ñ` in passage 36;
- exclude the three known missing human positions in passage 55;
- omit the first passage word because Provo does not report its target
  reading measure.

Acceptance criteria:

- 55 passages;
- 2,686 corrected Human–OB1 positions;
- no duplicated `(passage_id, word_id)`;
- no Human–OB1 word-string mismatch after the declared normalization;
- raw and corrected IDs both retained;
- every excluded row written to a separate audit CSV with a reason.

## 5. Define human TRT without hiding skips

Primary Human target:

\[
H_i=\frac{1}{R_i}\sum_{r=1}^{R_i} TRT_{ir}
\]

where skipped words retain `IA_DWELL_TIME=0`. This measures expected fixation
time allocation per reader.

Sensitivity target:

\[
H_i^{\mathrm{fixated}}
=
\mathrm{mean}\{TRT_{ir}:TRT_{ir}>0\}
\]

This matches the upstream OB1 evaluator's choice to convert zero TRT to
missing before averaging. Report both because the latter conditions duration
on fixation and therefore removes the skipping component.

## 6. ET1 inference and word alignment

The implementation uses the repository-pinned ET1 checkpoint and pinned T5
tokenizer. ET1 stays frozen and runs one passage at a time with no truncation.

For each passage:

1. use the exact cleaned passage string associated with the canonical word
   grid;
2. request T5 fast-tokenizer character offsets;
3. exclude padding and special tokens from word aggregation;
4. associate every non-special token with one and only one word span;
5. sum all subtokens assigned to word \(i\):

\[
E_i=\sum_{t\in word(i)}TRT_t^{ET1}
\]

6. save token IDs, token strings, offsets, raw prediction, word assignment,
   and exclusion reason.

The learned redistributor was trained with the current ET1 attention-mask
semantics, in which the T5 EOS position is valid during redistribution.
Therefore the reproduction path will apply redistribution using the same
mask, then discard special-token positions only during word aggregation.
A sensitivity check may mask EOS before redistribution, but it cannot replace
the primary exact-code path.

Alignment acceptance criteria:

- 100% of non-special ET1 tokens assigned;
- no token spans two Provo words;
- every canonical Provo word receives at least one T5 token;
- identical ET1 raw output reused across raw, symmetric, and asymmetric
  conditions;
- mass conservation checked before word aggregation for both redistributors.

## 7. Extract learned sigma values

For every reported OASST1 seed, load the final/best adapter checkpoint and
locate:

```text
asym_gaussian_redistributor.log_sigma_left
asym_gaussian_redistributor.log_sigma_right
```

The current trainer declares the redistributor in `modules_to_save`, so these
parameters should be in the PEFT adapter checkpoint. The extractor must:

- enumerate keys before selecting them;
- require exactly one left and one right scalar;
- convert using the current implementation,
  `exp(log_sigma) + min_sigma`;
- reject the initial `1.0/1.0` values unless the checkpoint genuinely
  contains them;
- store checkpoint SHA-256 and extracted values in CSV/JSON;
- never fit or update sigma on Provo.

If a reported seed checkpoint is unavailable, do not replace it with a mean
sigma from another seed. Report the missing seed and run only the recoverable
checkpoints.

## 8. Implement symmetric and asymmetric redistribution

Use the existing `AsymGaussianRedistributor` implementation directly. It
normalizes across target positions for each source token, so the valid-token
sum should be conserved up to floating-point error.

For each seed and passage, save:

```text
et1_raw_token_trt
et1_symmetric_token_trt
et1_asymmetric_token_trt
et1_raw_word_trt
et1_symmetric_word_trt
et1_asymmetric_word_trt
```

Do not compare projected gaze embeddings or reward scores. The comparison
point is the redistributor output before the gaze projector.

## 9. Make the OB1 snapshot runnable without changing its model

The Python adapter runs the pinned source instead of the included Slurm shell
script or its disabled CLI.

Allowed compatibility changes:

- absolute/explicit paths;
- corrected Provo preprocessing;
- deterministic random seeds;
- portable dependency versions;
- direct Python configuration object;
- structured fixation output;
- word-grid completion with zero TVT for unfixated words.

Disallowed scientific changes:

- tuning attention width, skew, thresholds, saccade parameters, or lexical
  parameters on Provo TRT;
- changing OB1 equations;
- substituting frequency resources without disclosure;
- choosing the best random seed or simulation subset based on Human metrics.

Primary configuration:

- condition: published baseline without predictability;
- passages: all 55 in each simulation;
- virtual readers: 100;
- seeds: fixed list `0..99`;
- attention parameters: values from the pinned `parameters.py`;
- output: every fixation with simulation, passage, word, duration, and
  saccade type.

Word-level OB1 TVT:

\[
O_i^{(s)}
=
\sum_{f:word(f)=i}duration(f)
\]

and:

\[
O_i=\frac{1}{100}\sum_{s=1}^{100}O_i^{(s)}
\]

Regressive refixations remain included. Complete each simulation/passage word
grid before averaging, assigning zero to words with no fixation.

Before the 100-reader run, execute:

1. import smoke;
2. one passage, one reader;
3. all 55 passages, one reader;
4. deterministic repeat with the same seed;
5. two different seeds to confirm stochastic variation.

Runtime will be reported from this benchmark rather than assumed from the
paper or an unverified 3–4 hour estimate.

Executed status:

- import smoke: passed;
- one passage, one reader: passed;
- same seed in separate subprocesses: exact;
- two seeds: stochastic variation observed;
- all 55 passages, one reader: passed in 280.18 seconds;
- 100 readers: pending the final result run.

## 10. Common scale and metrics

### Nonnegative allocation

Spearman correlation uses each method's finite word values directly. For JS
and Wasserstein only, transform predicted negative regression outputs with:

\[
x_i^+=\max(x_i,0)
\]

Then normalize per passage:

\[
p_i=\frac{x_i^+}{\sum_jx_j^+}
\]

Do not silently add an epsilon. Fail the passage if its nonnegative mass is
zero, and report the count and fraction of clipped values.

### Passage metrics

For each passage and applicable RM seed:

- Human Spearman:
  \(\rho(H,E)\), \(\rho(H,S)\), \(\rho(H,A)\), \(\rho(H,O)\);
- OB1 consistency:
  \(\rho(E,O)\), \(\rho(S,O)\), \(\rho(A,O)\);
- Jensen–Shannon divergence:
  \(JS(H,E)\), \(JS(H,S)\), \(JS(H,A)\), \(JS(H,O)\);
- 1-D Wasserstein distance using normalized word positions
  \(i/(n-1)\).

Spearman is invariant to the passage sum normalization, so normalization is
required for distribution metrics but not presented as affecting ranks.

## 11. Inference and uncertainty

Use 10,000 paired bootstrap samples over the 55 passages with a fixed seed.
The primary asymmetric-versus-raw contrast is:

\[
\Delta_H = metric(H,A)-metric(H,E)
\]

with the sign reversed for distance metrics, and the cognitive consistency
contrast is:

\[
\Delta_O = \rho(A,O)-\rho(E,O)
\]

For multiple reward-model seeds:

1. compute every passage metric separately per seed;
2. average seed-specific deltas within passage;
3. bootstrap the resulting 55 passage-level deltas;
4. report every seed separately as a robustness table.

This avoids treating seed-by-passage rows as independent observations.

Report:

- mean paired difference;
- percentile 95% confidence interval;
- two-sided bootstrap p-value as secondary;
- number of passages;
- number of RM checkpoints;
- all exclusion/failure counts.

## 12. Outputs

Required machine-readable outputs:

```text
run_manifest.json
provo_prepare_audit.json
provo_excluded_positions.csv
et1_token_values.csv
et1_inference_audit.json
et1_mass_audit.csv
checkpoint_sigmas.csv
ob1_fixations.csv
ob1_word_values_by_simulation.csv
ob1_word_values.csv
word_level_values.csv
passage_metrics.csv
result_table.csv
seed_result_table.csv
bootstrap_summary.csv
human_spearman_by_passage.png
```

Required presentation outputs:

| Method | Human Spearman ↑ | JS divergence ↓ | Wasserstein ↓ | OB1 Spearman ↑ |
|---|---:|---:|---:|---:|
| ET1 raw | | | | |
| ET1 + symmetric | | | | |
| ET1 + learned asymmetric | | | | |
| OB1 baseline | | | | — |

Also produce a passage-level box/violin plot for Human correlations and a
seed-level sigma table. Plot points must remain visible so the 55-passage
distribution can be inspected.

## 13. Verification gates

The result experiment is complete only when all gates pass:

- asset checksum verification;
- canonical 55-passage/2,686-position alignment;
- ET1 token coverage and special-token audit;
- redistribution mass-conservation tests;
- exact checkpoint-key and sigma extraction tests;
- OB1 same-seed deterministic repeat;
- OB1 fixation-to-word aggregation test including regressions and skips;
- hand-calculated toy tests for Spearman, JS, Wasserstein, and bootstrap;
- one-command smoke run;
- full output schema validation;
- result table regenerated exclusively from saved machine-readable outputs.

Current gate status: every code and one-reader smoke gate passes. The actual
checkpoint sigma extraction and 100-reader OB1 result run remain pending
because the OASST1 checkpoint paths have not been provided and the full
simulation has not been launched.

## 14. Interpretation rule

- Human and OB1 improve: claim greater human correspondence and directional
  OB1 consistency.
- OB1 improves but Human does not: claim only cognitive-model consistency.
- Neither improves: retain downstream reward-model utility but remove a
  quantitative human-gaze/perceptual-span interpretation.

Human TRT is overt fixation allocation. It must not be described as a direct
measurement of covert perceptual span.
