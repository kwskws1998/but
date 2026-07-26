# Outputs

All generated experiment and smoke artifacts are excluded from Git.

The full `run` command writes:

- `run_manifest.json` and checkpoint sigma JSON/CSV;
- ET1 token, word, mass-conservation, and inference-audit files;
- OB1 fixation, per-reader word, mean-word, parameter, runtime, and
  aggregation-audit files;
- separate unconditional and conditional Human-TRT evaluation directories;
- joined word values, passage metrics, grand and per-seed method tables,
  paired-bootstrap summary, audit JSON, and the passage-level Human-Spearman
  plot.

Only a run using the actual reported OASST1 checkpoints and all 100 OB1
readers is eligible to produce manuscript results. Synthetic smoke outputs
must remain labeled as such.
