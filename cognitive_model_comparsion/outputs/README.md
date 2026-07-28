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
`matched_asymmetry_contrasts.csv`. This legacy filename contains only learned
asymmetric versus fixed SymGaussian (`sigma_left=sigma_right=1.0`) contrasts
and flags whether each metric references Human TRT or OB1 TVT.

Running `compare-attention-profile` over saved ET1 and OB1 outputs writes:

- `kernel_profiles.csv`, `kernel_directionality.csv`, and
  `kernel_profiles.png`;
- `kernel_alignment_by_passage.csv`;
- `kernel_alignment_result_table.csv`;
- `kernel_alignment_contrasts.csv`;
- `kernel_profile_regions.csv` and `kernel_profile_regions.png`;
- `kernel_metric_comparison.png`;
- `gaussian_parameter_diagnostics.csv` and
  `gaussian_parameter_diagnostics.png`;
- optional `sigma_landscape.csv` plus one four-metric reviewer-facing heatmap
  per OB1 skew when `--with-sigma-landscape` is passed;
- `reviewer_kernel_summary.csv`, with explicit candidate/reference labels and
  rightward-share values;
- `fixed_ob1_priors.json`;
- `attention_profile_audit.json`;
- the copied `checkpoint_sigmas.json` and `.csv`.

The attention-profile tables keep width and direction controls separate:
`fixed_symmetric_sigma1` uses `(1.0, 1.0)`, whereas
`rms_side_scale_symmetric` uses the learned side-scale RMS on both sides.
`fixed_ratio4_same_rms` holds that parameter RMS fixed and imposes the
paper-stated 4:1 ratio. It does not guarantee equal realized variance after
support truncation. `mirrored_learned` swaps the two learned sigmas, preserving
their parameter count and RMS as a parameter-level direction reversal; its
pooled profile is not an exact mirror on right-heavy visible support.
`support_rms_displacement_symmetric` and
`support_rms_displacement_ratio4` instead solve their scale so that pooled
`sqrt(sum_d p(d) * d^2)` exactly matches the frozen learned kernel on the same
fixation-visible supports. These are post-hoc contextual spread ablations and
do not use OB1 attention weights.
The tables report JS divergence, Hellinger distance, total variation distance,
overlap coefficient, and supplemental token-offset Wasserstein on the complete
normalized profiles. Reviewer-facing figures show Spearman, JS, Hellinger, and
TV; overlap is exactly `1-TV` and Wasserstein remains supplemental. Spearman
alone excludes offsets that are zero in both passage-level profiles so absent,
globally padded positions cannot inflate tied ranks.

Every kernel-profile table and audit records `candidate_support_policy`.
`fixation_matched` is primary: candidate and OB1 profiles are normalized on the
same exact visible offsets at each fixation and pooled with the same weights.
`global` is a legacy sensitivity that normalizes candidates once on the global
offset union while OB1 remains fixation-window-conditioned. Outputs from these
policies must be kept in separate directories.

For attention-profile outputs, method-mean intervals resample passages and
contrast intervals resample paired passage differences. OB1 simulation IDs are
pooled before resampling, so these intervals do not quantify simulation-level
Monte Carlo uncertainty. The output tables and audit also state that this
kernel-shape analysis uses T5 token geometry but not actual ET1-predicted TRT
magnitudes.

The fixed-prior JSON generated from `--profile-component focused
--candidate-support-policy fixation_matched` may be passed to root
`main.py --fixed_ob1_prior_json`. The `full` residual-attention and legacy
`global` support sensitivities are deliberately rejected as RM priors.

Previously generated selected-s11 values `0.720` (`skew=3`) and `0.737`
(`skew=4`) used the legacy `global` support policy. They are not primary
results. Regenerate the focused, full-profile, and equal-fixation outputs from
the cached 100-simulation fixation table with
`--candidate-support-policy fixation_matched`; no corrected values are recorded
here until those files are inspected.

`result_table.csv` reports Human- and OB1-referenced Spearman, JS divergence,
Hellinger distance, total variation distance, overlap coefficient, and
word-order Wasserstein with percentile 95% confidence intervals. Unprefixed
distribution columns use Human TRT as their reference; `ob1_`-prefixed columns
use OB1 TVT.

`cognitive_result_table.csv` contains only ET1 raw, symmetric, and asymmetric
alignment to OB1, excluding the trivial OB1 self-comparison.
`cognitive_bootstrap_summary.csv` contains only their paired contrasts for:

- `ob1_spearman`, where higher is better;
- `ob1_js_divergence`, where lower is better;
- `ob1_hellinger_distance`, where lower is better;
- `ob1_total_variation_distance`, where lower is better;
- `ob1_overlap_coefficient`, where higher is better;
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
