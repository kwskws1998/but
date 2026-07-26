# Outputs

All generated experiment and smoke artifacts are excluded from Git.

The full `run` command writes:

- `run_manifest.json` and checkpoint sigma JSON/CSV;
- ET1 token, word, mass-conservation, and inference-audit files;
- OB1 fixation, per-reader word, mean-word, parameter, runtime, and
  aggregation-audit files;
- separate unconditional and conditional Human-TRT evaluation directories;
- optional `et1_special_excluded` and matching evaluation directories from
  `--with-special-token-sensitivity`;
- optional nested `ob1_clean_passages/` result directories from
  `--with-ob1-clean-passage-sensitivity`;
- joined word values, passage metrics, grand and per-seed method tables,
  paired-contrast summary, reviewer-facing cognitive-only tables, audit JSON,
  and the passage-level Human-Spearman plot.

Running `evaluate` with the updated code also writes
`matched_asymmetry_contrasts.csv`. It contains only learned asymmetric versus
RMS-width-matched symmetric contrasts and flags whether each metric references
Human TRT or OB1 TVT.

Running `compare-attention-profile` over saved ET1 and OB1 outputs writes:

- `kernel_profiles.csv`, `kernel_directionality.csv`, and
  `kernel_profiles.png`;
- `kernel_alignment_by_passage.csv`;
- `kernel_alignment_result_table.csv`;
- `kernel_alignment_contrasts.csv`;
- `fixed_ob1_priors.json`;
- `attention_profile_audit.json`;
- the copied `checkpoint_sigmas.json` and `.csv`.

The fixed-prior JSON generated from `--profile-component focused` may be
passed to root `main.py --fixed_ob1_prior_json`. The `full` residual-attention
sensitivity is deliberately rejected as an RM prior.

`result_table.csv` reports `human_spearman`, `js_divergence`,
`word_order_wasserstein`, `ob1_spearman`, `ob1_js_divergence`, and
`ob1_word_order_wasserstein` with percentile 95% confidence intervals. The
unprefixed JS and Wasserstein columns use Human TRT as their reference; the
`ob1_`-prefixed columns use OB1 TVT.

`cognitive_result_table.csv` contains only ET1 raw, symmetric, and asymmetric
alignment to OB1, excluding the trivial OB1 self-comparison.
`cognitive_bootstrap_summary.csv` contains only their paired contrasts for:

- `ob1_spearman`, where higher is better;
- `ob1_js_divergence`, where lower is better;
- `ob1_word_order_wasserstein`, where lower is better.

Both Wasserstein columns measure transport along normalized word order, not
fixation-coordinate or scanpath distance.

`bootstrap_summary.csv` retains its historical filename but contains paired
mean improvements, percentile 95% intervals, and the two-sided paired
sign-flip field `permutation_p_two_sided`. It does not contain a
bootstrap-derived p-value. Provo inference is paired by passage; OneStop
inference resamples and flips the 30 article clusters.

For OneStop, `ob1_token_transformations.csv` records the 95 punctuation-only
canonical positions that OB1 cannot represent directly. Every evaluation
method excludes the same positions. The clean-passage sensitivity additionally
excludes all 55 affected paragraphs and recomputes outputs on the 107
paragraphs with no OB1 token transformation. Those paragraphs span 28 article
clusters, which are the resampling units for this nested analysis; the full
162-paragraph result remains primary.

Only a Provo result using fixed reported sigmas (or their exact checkpoints)
and all 100 OB1 seeds is eligible to produce manuscript results. Synthetic and
one-reader smoke outputs must remain labeled as such. The complete OneStop
100-seed run is suspended and must not be presented as completed.
