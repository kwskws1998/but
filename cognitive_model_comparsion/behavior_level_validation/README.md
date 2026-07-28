# Behavior-level Provo validation

This directory is an independent post-processing path for comparing actual
word-level ET1 predictions, redistributed ET1 predictions, Human Provo TRT,
and OB1 simulated total viewing time (TVT). It is separate from
`compare-attention-profile`, which compares unit-impulse redistribution
kernels with one internal component of the OB1 attention function.

## What follows the OB1-Provo evaluation design

Lopes Rego et al. (2024) state:

> “For the analysis, we considered eye-movement measures at word-level.”

They then:

> “computed the Root Mean Squared Error (RMSE) between each eye-movement
> measure from each simulation by OB1-reader and each eye-movement measure
> from the Provo corpus averaged over participants.”

The source is [Language models outperform cloze predictability in a cognitive
model of reading](https://doi.org/10.1371/journal.pcbi.1012117), pp. 5 and 19.
Their released preprocessing also converts Human `total_reading_time == 0` to
missing before participant averaging. This analysis therefore uses
`human_trt_conditional`.

The analogous conditions here are:

1. ET1 raw word TRT allocation;
2. ET1 plus fixed SymGaussian redistribution
   (`sigma_left=sigma_right=1.0`);
3. ET1 plus the OASST1-learned asymmetric redistribution;
4. OB1 simulated word TVT.

All ET1 conditions use the actual passage-specific frozen-ET1 predictions.
Native T5-token values are summed into the aligned Provo words. The learned
sigmas are not fitted on Provo or OB1.

## Why raw-millisecond RMSE is not copied

The GazeReward appendix states that the first ET predictor normalizes fixation
duration by corpus and:

> “they map the duration values to discrete space [1, 2, , K].”

Therefore ET1 output is not millisecond TRT. A raw RMSE between ET1 output and
Human or OB1 milliseconds would mix incompatible units. The primary metrics
are instead:

- passage-level Spearman correlation for relative word ranking;
- Jensen-Shannon divergence after each passage is normalized to unit
  allocation mass;
- paired passage bootstrap intervals for within-passage method differences.

The analysis does not claim absolute TRT calibration, human perceptual-span
estimation, or mechanistic equivalence with OB1.

## Environment

```bash
cd /workspace/but
source /venv/cognitive/bin/activate

python -m pip install -r cognitive_model_comparsion/requirements.txt
python -m pip check
python -m pytest -q \
  cognitive_model_comparsion/behavior_level_validation/tests
```

## Selected-checkpoint evaluation

The 12-checkpoint ET1 sweep contains the selected checkpoint
`s11_acc076942_l037380_r321289`. The completed 100-reader OB1 directory can be
reused; no ET1 inference or OB1 simulation is rerun by this command.

```bash
cd /workspace/but
source /venv/cognitive/bin/activate

export SWEEP_RUN=/workspace/but/cognitive_model_comparsion/outputs/provo_sigma_sweep_12
export PROVO_RUN=/workspace/but/cognitive_model_comparsion/outputs/provo_ob1_sigma076675
export BEHAVIOR_RUN=/workspace/but/cognitive_model_comparsion/outputs/provo_s11_behavior_validation

/venv/cognitive/bin/python -m \
  cognitive_model_comparsion.behavior_level_validation evaluate \
  --et1-dir "$SWEEP_RUN/et1" \
  --ob1-dir "$PROVO_RUN/ob1" \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$BEHAVIOR_RUN"
```

If the ET1 directory contains exactly one checkpoint, `--checkpoint-id` may be
omitted. The CLI refuses to silently average multiple learned sigma
checkpoints.

## Outputs

- `behavior_result_table.csv`: Human- and OB1-referenced Spearman and JS
  estimates with passage-bootstrap intervals;
- `behavior_paired_contrasts.csv`: paired raw/symmetric/asymmetric
  improvements without reviewer-facing p-values;
- `RESULTS.md`: compact display tables and interpretation boundary;
- `behavior_analysis_audit.json`: input hashes, target definition, metric
  scope, and confirmation that actual ET1 values were used;
- `evaluation_conditional/`: complete word-level values, passage metrics,
  checkpoint tables, and raw statistical outputs.

To rebuild only the compact tables from an existing conditional evaluation:

```bash
/venv/cognitive/bin/python -m \
  cognitive_model_comparsion.behavior_level_validation summarize \
  --evaluation-dir /path/to/evaluation_conditional \
  --output-dir /path/to/behavior_report
```
