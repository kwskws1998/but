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
  paired-contrast summary, audit JSON, and the passage-level Human-Spearman
  plot.

`result_table.csv` reports `human_spearman`, `js_divergence`,
`word_order_wasserstein`, and `ob1_spearman` with percentile 95% confidence
intervals. `word_order_wasserstein` is transport along normalized word order,
not fixation-coordinate or scanpath distance.

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

Only a run using fixed reported sigmas (or their exact checkpoints) and all
100 OB1 seeds is eligible to produce manuscript results. Synthetic smoke
outputs must remain labeled as such. The complete OneStop 100-seed run remains
pending.
