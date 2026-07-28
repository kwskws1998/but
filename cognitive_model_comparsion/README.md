# Cognitive model comparison

This directory contains the isolated, Python-only Human gaze–ET1
redistribution–OB1 comparison requested by the reviewer. It supports both the
55-passage Provo corpus and the 162-paragraph OneStop Ordinary Reading
extension. It does not train or load the Llama reward-model backbone.

The current rebuttal route reuses the completed Provo 100-simulation output.
The full OneStop simulation is suspended because it is not required for this
route.

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
- fixed sigma-one symmetric, RMS-of-side-scales symmetric, and learned
  asymmetric redistribution using the production `AsymGaussianRedistributor`;
- the production-faithful primary mask, which includes the T5 EOS position
  during redistribution, plus a requested special-token-excluded sensitivity;
- per-passage mass audits before and after redistribution, including
  word-assigned, unassigned-special, and final evaluable-grid mass;
- deterministic, parallel OB1 stochastic simulations and TVT aggregation over
  every fixation, including regressions;
- reconstruction of the saved OB1 fixation-onset focused Gaussian component in
  native T5 relative-token coordinates;
- fixation-matched candidate normalization on each fixation's exact visible
  relative-token support as the primary kernel-profile estimand, with the
  former global-support calculation retained only as a legacy sensitivity;
- direct comparison of a no-redistribution impulse, a fixed SymGaussian with
  `sigma_left=sigma_right=1.0`, an RMS-of-side-scales symmetric diagnostic, a
  descriptive Gaussian fitted to the same OB1 profile, and the
  OASST1-learned asymmetric kernel;
- passage-level Human-referenced and OB1-referenced Spearman,
  Jensen–Shannon divergence, and `word_order_wasserstein`;
- percentile 95% passage-bootstrap confidence intervals for method means,
  paired passage-bootstrap confidence intervals for within-passage method
  differences, and two-sided paired sign-flip tests; OB1 simulations are pooled
  before these passage-level resampling procedures;
- descriptive rightward-share point estimates from pooled offset profiles,
  without bootstrap confidence intervals;
- passage-level resampling for Provo and article-cluster resampling over the 30
  OneStop articles;
- optional OneStop clean-passage sensitivity restricted to 107 paragraphs
  with no punctuation-only OB1 token transformation;
- one Python CLI for setup, component runs, and complete experiments.

No `.sh` file or `gdown` is used.

## Recommended Provo rebuttal route

The narrow claim is cognitive-model consistency, not improved Human TRT
prediction. The OB1 paper describes “parallel processing of multiple words”
with an “attentional window of adaptable size.” It defines `Asym` as “equal to
1 toward the right and 0.25 toward the left” and motivates this with a roughly
fourfold larger effective span to the right. See the
[OB1 paper](https://research.vu.nl/ws/portalfiles/portal/72578613/OB1_reader_A_model_of_word_recognition_and_eye_movements_in_text_reading.pdf).

The GazeReward paper states that ET1 “predicts total reading time (TRT) per
token” and describes its native-token output dimension as “the number of
tokens in the tokenizer used by the ET prediction model.” The kernel
comparison therefore stays in native ET1/T5 token space; it does not remap to
Llama tokens. See the
[GazeReward paper](https://arxiv.org/pdf/2410.01532).

The analysis has two complementary parts:

1. `matched_asymmetry_contrasts.csv` is a legacy filename for the comparison
   between learned asymmetric redistribution and the independent fixed
   SymGaussian (`sigma_left=sigma_right=1.0`) on the saved Provo word-level
   Human TRT and OB1 TVT results.
2. `compare-attention-profile` reconstructs OB1's fixation-onset letter
   attention from the saved 100-simulation fixation table, projects it onto
   native T5 relative-token offsets, and compares four kernels:
   `raw_delta`, `fixed_symmetric_sigma1`, `rms_side_scale_symmetric`,
   `fixed_ob1_gaussian`, and `learned_asymmetric`.

The isolated
[`behavior_level_validation`](behavior_level_validation/README.md) entry point
packages the first analysis as a selected-checkpoint, conditional-Human-TRT
comparison with compact reviewer tables. It uses actual passage-specific ET1
values and OB1 word TVT; it is not a unit-impulse kernel analysis.

The second analysis uses native T5 token geometry and character alignment from
the ET1 output table, but it does not use the ET1-predicted TRT magnitudes.
`raw_delta` is therefore an internal ID for the no-redistribution impulse
response, not a behavior-level ET1 prediction result.

The primary candidate-support policy is `fixation_matched`. For every OB1
fixation, it normalizes each candidate kernel on the exact relative T5-token
offsets visible to that fixation and then pools candidate and OB1 profiles with
identical fixation weights. The previous calculation was hybrid: OB1 was
normalized within each fixation window, whereas each candidate was normalized
once on the global union of offsets. This could assign candidate mass to
positions unavailable in a particular fixation. That earlier calculation is
retained as `global` only for a legacy support sensitivity.

The previously printed s11 Spearman values `0.720` (`skew=3`) and `0.737`
(`skew=4`) used the legacy `global` policy. They cannot be reported as primary
after this support correction. The cached 100-simulation post-processing below
must be rerun with `fixation_matched`; this repository does not claim new
100-simulation values before that rerun is inspected.

The projection replays the vendored implementation, including its actual
`n-1 ... n+3` stimulus window, the post-update attention width, fixation
position, and punctuation-aware T5 character offsets. The primary `focused`
profile removes OB1's constant residual `+0.25`, because a normalized Gaussian
redistributor has no constant background component. The `full` run below is
the focused component plus that constant residual, not OB1's complete latent
attention. Both variants exclude acuity, within-fixation attention shifts,
lexical activation, saccade control, and final TVT. Saved OB1 outputs do not
contain cycle-level attention shifts, so this is explicitly a fixation-onset
attention comparison.

The saved fixation trajectories were generated by the vendored code with
`attention_skew=3`. The additional `skew=4` condition reweights those same
fixation geometries using the original-paper setting; it does not pretend that
OB1 trajectories were rerun under a second parameter value. Use `skew=3` for
the optional fixed-prior RM baseline and report `skew=4` as a formula-level
sensitivity.

### Reuse the completed Provo 100-simulation output

Activate the isolated environment and verify the updated code:

```bash
cd /workspace/but
source /venv/cognitive/bin/activate

python -m pip install -r cognitive_model_comparsion/requirements.txt
python -m pip check
python -m pytest -q cognitive_model_comparsion/tests
```

Set the historical completed run only as the reusable OB1 and native-T5
geometry cache. Its directory name records the earlier `0.41553/3.46115` pair,
but OB1 trajectories and T5 token geometry do not depend on that pair. The
commands below pass the selected `0.3738/3.21289` pair directly and do not rerun
ET1 or OB1. They use the cached ET1 table only for token geometry, not for its
TRT values.

```bash
export PROVO_CACHE=/workspace/but/cognitive_model_comparsion/outputs/provo_ob1_sigma076675
export SELECTED_PROFILE_RUN=/workspace/but/cognitive_model_comparsion/outputs/provo_selected_s11_profile_reanalysis

python cognitive_model_comparsion/main.py compare-attention-profile \
  --corpus provo \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --et1-dir "$PROVO_CACHE/et1" \
  --ob1-dir "$PROVO_CACHE/ob1" \
  --ob1-attention-skew 3 \
  --ob1-attention-skew 4 \
  --fixation-weighting duration \
  --profile-component focused \
  --candidate-support-policy fixation_matched \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SELECTED_PROFILE_RUN/attention_profile_focused"

python cognitive_model_comparsion/main.py compare-attention-profile \
  --corpus provo \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --et1-dir "$PROVO_CACHE/et1" \
  --ob1-dir "$PROVO_CACHE/ob1" \
  --ob1-attention-skew 3 \
  --ob1-attention-skew 4 \
  --fixation-weighting duration \
  --profile-component full \
  --candidate-support-policy fixation_matched \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SELECTED_PROFILE_RUN/attention_profile_full_sensitivity"

python cognitive_model_comparsion/main.py compare-attention-profile \
  --corpus provo \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --et1-dir "$PROVO_CACHE/et1" \
  --ob1-dir "$PROVO_CACHE/ob1" \
  --ob1-attention-skew 3 \
  --ob1-attention-skew 4 \
  --fixation-weighting equal \
  --profile-component focused \
  --candidate-support-policy fixation_matched \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SELECTED_PROFILE_RUN/attention_profile_equal_fixation_sensitivity"

python cognitive_model_comparsion/main.py compare-attention-profile \
  --corpus provo \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --et1-dir "$PROVO_CACHE/et1" \
  --ob1-dir "$PROVO_CACHE/ob1" \
  --ob1-attention-skew 3 \
  --ob1-attention-skew 4 \
  --fixation-weighting duration \
  --profile-component focused \
  --candidate-support-policy global \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SELECTED_PROFILE_RUN/attention_profile_global_support_sensitivity"
```

These are CPU metric passes over saved CSV files and normally finish in minutes
rather than hours. The behavior-level selected-checkpoint result requires the
actual s11-redistributed TRT values and must therefore be read from the
12-checkpoint sweep below, not from `$PROVO_CACHE/et1`. Confirm that the reused
OB1 input really contains 100 simulations:

```bash
python -c "import pandas as pd; p='$PROVO_CACHE/ob1/ob1_fixations.csv'; d=pd.read_csv(p, usecols=['simulation_id','seed','text_id']); print('simulations',d.simulation_id.nunique(),'seeds',d.seed.nunique(),'passages',d.text_id.nunique())"

column -s, -t "$SELECTED_PROFILE_RUN/attention_profile_focused/kernel_alignment_result_table.csv"
column -s, -t "$SELECTED_PROFILE_RUN/attention_profile_focused/kernel_alignment_contrasts.csv"
column -s, -t "$SELECTED_PROFILE_RUN/attention_profile_focused/kernel_directionality.csv"
column -s, -t "$SELECTED_PROFILE_RUN/attention_profile_focused/reviewer_kernel_summary.csv"
column -s, -t "$SELECTED_PROFILE_RUN/attention_profile_full_sensitivity/reviewer_kernel_summary.csv"
column -s, -t "$SELECTED_PROFILE_RUN/attention_profile_equal_fixation_sensitivity/reviewer_kernel_summary.csv"
column -s, -t "$SELECTED_PROFILE_RUN/attention_profile_global_support_sensitivity/reviewer_kernel_summary.csv"
cat "$SELECTED_PROFILE_RUN/attention_profile_focused/fixed_ob1_priors.json"
```

### Sweep the 12 learned sigma pairs

The checked compact configuration is
`configs/reviewer_sigma_sweep_12.json`. Its `source_accuracy` values are
provenance from the original reward-model runs; they are not Provo scores and
are never used to fit or select a redistribution kernel.

Reuse the completed 100-simulation OB1 output and run frozen ET1 only once over the
55 Provo passages:

```bash
cd /workspace/but
source /venv/cognitive/bin/activate

export PROVO_RUN=/workspace/but/cognitive_model_comparsion/outputs/provo_ob1_sigma076675
export SWEEP_RUN=/workspace/but/cognitive_model_comparsion/outputs/provo_sigma_sweep_12
export SIGMA_CONFIG=/workspace/but/cognitive_model_comparsion/configs/reviewer_sigma_sweep_12.json

CUDA_VISIBLE_DEVICES=0 /venv/cognitive/bin/python \
  cognitive_model_comparsion/main.py predict-et1 \
  --corpus provo \
  --sigma-json "$SIGMA_CONFIG" \
  --output-dir "$SWEEP_RUN/et1"

/venv/cognitive/bin/python cognitive_model_comparsion/main.py evaluate \
  --corpus provo \
  --et1-dir "$SWEEP_RUN/et1" \
  --ob1-dir "$PROVO_RUN/ob1" \
  --human-target human_trt_unconditional \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SWEEP_RUN/evaluation_unconditional"

/venv/cognitive/bin/python \
  cognitive_model_comparsion/main.py compare-attention-profile \
  --corpus provo \
  --sigma-json "$SWEEP_RUN/et1/checkpoint_sigmas.json" \
  --et1-dir "$SWEEP_RUN/et1" \
  --ob1-dir "$PROVO_RUN/ob1" \
  --ob1-attention-skew 3 \
  --ob1-attention-skew 4 \
  --fixation-weighting duration \
  --profile-component focused \
  --candidate-support-policy fixation_matched \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SWEEP_RUN/attention_profile_focused"

/venv/cognitive/bin/python \
  cognitive_model_comparsion/main.py compare-attention-profile \
  --corpus provo \
  --sigma-json "$SWEEP_RUN/et1/checkpoint_sigmas.json" \
  --et1-dir "$SWEEP_RUN/et1" \
  --ob1-dir "$PROVO_RUN/ob1" \
  --ob1-attention-skew 3 \
  --ob1-attention-skew 4 \
  --fixation-weighting duration \
  --profile-component full \
  --candidate-support-policy fixation_matched \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SWEEP_RUN/attention_profile_full_sensitivity"

/venv/cognitive/bin/python \
  cognitive_model_comparsion/main.py compare-attention-profile \
  --corpus provo \
  --sigma-json "$SWEEP_RUN/et1/checkpoint_sigmas.json" \
  --et1-dir "$SWEEP_RUN/et1" \
  --ob1-dir "$PROVO_RUN/ob1" \
  --ob1-attention-skew 3 \
  --ob1-attention-skew 4 \
  --fixation-weighting equal \
  --profile-component focused \
  --candidate-support-policy fixation_matched \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --output-dir "$SWEEP_RUN/attention_profile_equal_fixation_sensitivity"
```

The behavior-level sweep produces:

- `sigma_sweep_summary.csv`: one wide row per sigma pair;
- `checkpoint_matched_asymmetry_contrasts.csv`: legacy filename for per-pair
  learned asymmetric versus fixed sigma-one SymGaussian bootstrap results;
- `checkpoint_cognitive_bootstrap_summary.csv`: per-pair OB1-referenced
  contrasts;
- `seed_result_table.csv`: per-pair method means and 95% passage-bootstrap
  confidence intervals conditional on pooled OB1 simulations and fixed sigma
  values.

Rightward share is a descriptive point estimate from each pooled offset profile;
the current pipeline does not attach a bootstrap interval to it.

The kernel-level sweep produces per-pair rows in
`kernel_alignment_result_table.csv`, `kernel_alignment_contrasts.csv`, and
`kernel_directionality.csv`, plus the explicitly labeled
`reviewer_kernel_summary.csv`. Each command post-processes saved OB1 fixations
without rerunning OB1 and writes one plot per pair under
`kernel_profile_plots/`.

Print the principal behavior and kernel comparisons:

```bash
python -c "import pandas as pd; p='$SWEEP_RUN/evaluation_unconditional/checkpoint_matched_asymmetry_contrasts.csv'; d=pd.read_csv(p); print(d[d.metric.isin(['human_spearman','ob1_spearman','ob1_word_order_wasserstein'])][['checkpoint_id','source_accuracy','sigma_left','sigma_right','metric','mean_paired_improvement','ci_low','ci_high','permutation_p_two_sided']].to_string(index=False))"

python -c "import pandas as pd; p='$SWEEP_RUN/attention_profile_focused/kernel_alignment_contrasts.csv'; d=pd.read_csv(p); q=d[(d.candidate=='learned_asymmetric') & d.baseline.isin(['fixed_symmetric_sigma1','rms_side_scale_symmetric'])]; print(q[['checkpoint_id','source_accuracy','learned_sigma_left','learned_sigma_right','ob1_attention_skew','baseline','metric','mean_paired_improvement','ci_low','ci_high','permutation_p_two_sided']].to_string(index=False))"

python -c "import pandas as pd; ck='s11_acc076942_l037380_r321289'; roots=['attention_profile_focused','attention_profile_full_sensitivity','attention_profile_equal_fixation_sensitivity']; [print('\\n'+root+'\\n'+pd.read_csv('$SWEEP_RUN/'+root+'/reviewer_kernel_summary.csv').query('checkpoint_id == @ck').to_string(index=False)) for root in roots]"
```

All 12 rows should be reported as a robustness sweep. The selected s11
behavior-level result must be filtered from `$SWEEP_RUN/evaluation_unconditional`
by `checkpoint_id`; the historical `provo_ob1_sigma076675/et1` values belong to
s05.
Choosing the best Provo row after inspecting these results and presenting it as
a pre-specified held-out result would be post-hoc selection.

The main reviewer-facing interpretation must be regenerated from the primary
`fixation_matched`, focused, duration-weighted, skew-3 output and must report
metric agreement and disagreement. No numerical direction is asserted here
before the corrected cached 100-simulation post-processing is complete. The
legacy `global` output is a support sensitivity and cannot substitute for the
primary table.

Do not replace “directional correspondence” with “Human perceptual-span
estimation,” and do not claim universal superiority if one of Spearman, JS, or
Wasserstein disagrees. The fixed OB1 Gaussian is fitted to the same projected
OB1 profile, so its OB1-alignment row is a descriptive reference rather than a
held-out validation result.

### Optional fixed-prior RM baseline

This is the only step that trains Llama. It is optional for the narrow
cognitive-model response. The generated `fixed_ob1_priors.json` contains one
Gaussian fitted to the projected OB1 profile for each declared OB1 asymmetry
setting. Root `main.py` selects one record, checks the file hash and coordinate
contract, initializes the production redistributor to the exact effective
sigmas, and freezes both log-sigma parameters. Human Provo TRT and reward
labels are not used to fit this prior.

Use the full reward-model environment, not `/venv/cognitive`:

```bash
cd /workspace/but
source /venv/main/bin/activate

python -m pip install -r requirements.txt
python -m pip check
python -c "from huggingface_hub import whoami; print(whoami())"
python -c "import torch, bitsandbytes; print(torch.__version__, torch.cuda.get_device_name(0), bitsandbytes.__version__)"
python -c "from utils.fixed_ob1_prior import load_fixed_ob1_prior; import json; p=load_fixed_ob1_prior('cognitive_model_comparsion/outputs/provo_ob1_sigma076675/attention_profile_focused/fixed_ob1_priors.json', 3.0); print(json.dumps(p, indent=2))"
```

The Meta-Llama-3-8B approval and Hugging Face login must already be complete.
For each paper seed, change only `--seed`. `--max_tokens` is intentionally
omitted; its default is `None`, so there is no explicit token truncation.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name OpenAssistant/oasst1 \
  --model_name meta-llama/Meta-Llama-3-8B \
  --fixations_model_version 1 \
  --features_used 1,0,0,0,0 \
  --concat true \
  --use_softprompt true \
  --use_asym_gaussian_redistributor true \
  --fixed_ob1_prior_json cognitive_model_comparsion/outputs/provo_ob1_sigma076675/attention_profile_focused/fixed_ob1_priors.json \
  --fixed_ob1_prior_skew 3 \
  --use_lora true \
  --use_quantization true \
  --train_epochs 2 \
  --max_length 5000 \
  --batch_size 1 \
  --gradient_acum_steps 8 \
  --gradient_checkpointing true \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine_with_min_lr \
  --min_lr_ratio 0.7 \
  --weight_decay 0.1 \
  --fp_dropout 0.1,0.3 \
  --seed 41 \
  --output_root cognitive_model_comparsion/outputs/fixed_ob1_prior_rm
```

Repeat with `--seed 42` and `--seed 43`. The run's `args.json` must show
`"sigma_learnable": false`, the selected prior SHA-256, and the effective
`sigma_left` and `sigma_right`. Training logs also print
`SIGMA PARAM FROZEN` for both parameters.

## Environment and setup

Python 3.12 is the tested environment. On a fresh server, create and activate
the environment from the cloned repository root:

```bash
cd /workspace/but

python -m venv /venv/cognitive
source /venv/cognitive/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r cognitive_model_comparsion/requirements.txt
python -m pip check
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python cognitive_model_comparsion/main.py setup --corpus onestop
python cognitive_model_comparsion/main.py audit --corpus onestop

python -m pytest -q cognitive_model_comparsion/tests
```

`setup --corpus onestop` downloads the official 169 MiB Ordinary Interest Area
ZIP, the pinned OB1 source, SUBTLEX-UK, the public ET1 checkpoint, and the
pinned public T5 tokenizer. It verifies the recorded sizes and SHA-256 digests
where available and then builds the canonical OneStop tables. The OneStop CSV
is read directly from the ZIP in chunks and its 2.46 GB uncompressed member is
not extracted. No Hugging Face login, gated Llama access, W&B login, `gdown`,
or reward-model checkpoint is required.

The public first-run sources are the
[OneStop OSF release](https://osf.io/zn9sq/),
[pinned OB1 source](https://github.com/dritlopes/language_models_outperform_cloze_predictability_in_a_cognitive_model_of_reading/commit/56b8d6401d1c2c1886a9c6ff9df4a143c6f2c12d),
[SUBTLEX-UK](https://psychology.nottingham.ac.uk/subtlex-uk/),
[pinned ET1 checkpoint](https://github.com/huangxt39/SelectiveCacheForLM/blob/eccc93f969745b04ce1e4911d6513d85565cc919/FPmodels/T5-tokenizer-BiLSTM-TRT-12-concat-3), and
[pinned public T5 tokenizer](https://huggingface.co/t5-small/tree/df1b051c49625cf57a3d0d8d3863ed4d13564fe4).

The first setup requires network access. Later runs reuse and verify the local
assets. `setup` already performs OneStop preparation, so a separate
`prepare-onestop` command is unnecessary unless intentionally rebuilding with
a custom chunk size. Use `--onestop-chunksize` on `setup`, or `--chunksize` on
`prepare-onestop`, only when machine memory requires a different value.

Use `--skip-et1` only when preparing an OB1-only CPU machine. The isolated
requirements intentionally omit Llama-training packages such as
`bitsandbytes`, `deepspeed`, `trl`, and `wandb`.

To prepare Provo as well, run:

```bash
python cognitive_model_comparsion/main.py setup --corpus provo
python cognitive_model_comparsion/main.py audit --corpus provo
```

Offline verification after a OneStop-only setup:

```bash
python cognitive_model_comparsion/download_assets.py \
  --asset onestop --verify-only
python cognitive_model_comparsion/download_assets.py \
  --asset ob1 --verify-only
python cognitive_model_comparsion/download_assets.py \
  --asset subtlex --verify-only
```

After both the Provo and OneStop setup commands have completed,
`python cognitive_model_comparsion/download_assets.py --verify-only` verifies
the complete manifest. ET1 is independently revalidated when it is loaded.

### GPU and CPU roles

- ET1 automatically uses one visible CUDA GPU when available and otherwise
  runs on CPU. `CUDA_VISIBLE_DEVICES=0` selects GPU 0; it does not distribute
  ET1 over multiple GPUs.
- OB1 is CPU code. `--workers 32` starts up to 32 isolated CPU workers, while
  each worker is restricted to one PyTorch thread. Additional GPUs do not
  accelerate this stage.
- Canonical-table preparation, metric computation, cluster bootstrap, and
  sign-flip tests run on CPU.
- A GPU is optional. On a CPU-only machine, omit `CUDA_VISIBLE_DEVICES=0` and
  reduce `--workers` if memory pressure appears. Changing `--workers` does not
  change the requested OB1 seeds or the scientific result.

The complete OneStop command below therefore uses one GPU briefly for ET1 and
the specified CPU workers for the much longer OB1 stage.

## Provo rebuttal run

The supplied direct values are the effective sigmas learned by the
ET1 + GazeConcat, TRT-only OASST1 reward-model condition. They remain fixed on
Provo; neither sigma is fitted to Human Provo TRT.

```bash
CUDA_VISIBLE_DEVICES=0 python cognitive_model_comparsion/main.py run \
  --corpus provo \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --seeds 0:100 \
  --workers 32 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --with-special-token-sensitivity \
  --output-dir cognitive_model_comparsion/outputs/provo_ob1_sigma076942
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

After the environment setup above, run the same frozen ET1 sigmas and
published OB1 baseline on the OneStop Advanced paragraphs:

```bash
CUDA_VISIBLE_DEVICES=0 python cognitive_model_comparsion/main.py run \
  --corpus onestop \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --seeds 0:100 \
  --workers 32 \
  --bootstrap-samples 10000 \
  --seed 20260725 \
  --with-special-token-sensitivity \
  --with-ob1-clean-passage-sensitivity \
  --output-dir cognitive_model_comparsion/outputs/onestop_ob1_sigma076942
```

This full OneStop 100-seed run is currently suspended in favor of the
completed Provo 100-simulation reuse route. The code and canonical data audit
being complete must not be reported as a finished OneStop alignment result.

`0:100` means 100 stochastic OB1 replications with fixed random seeds. They are
not 100 fitted reader models and are not the 180 OneStop participants. Human
TRT is averaged from the actual participant rows; OB1 TVT is averaged from
independent simulations of the same paragraph text.

For a CPU-only execution, use the identical command without
`CUDA_VISIBLE_DEVICES=0`. This command is retained for reproducibility, not as
the recommended rebuttal command.

The full pipeline is complete only when `run_manifest.json` reports
`"status": "complete"`. Check it without creating another script:

```bash
python -c "import json; p='cognitive_model_comparsion/outputs/onestop_ob1_sigma076942/run_manifest.json'; print(json.load(open(p, encoding='utf-8'))['status'])"
```

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

The behavior-level reviewer comparison uses OB1 simulated TVT as the reference
and compares ET1 raw, ET1 + symmetric, and ET1 + learned asymmetric. It does not
use the OB1 self-comparison as a substantive result. The cognitive-only
metrics are:

- `ob1_spearman`: rank correspondence with OB1 TVT, higher is better;
- `ob1_js_divergence`: divergence between normalized method and OB1
  allocation, lower is better;
- `ob1_word_order_wasserstein`: transport distance between normalized method
  and OB1 allocation along normalized word order, lower is better.

`result_table.csv` reports method means and percentile 95% intervals.
`bootstrap_summary.csv` reports paired improvements, percentile 95% intervals,
and `permutation_p_two_sided` from the paired sign-flip test. There is no
bootstrap-derived p-value.

The direct reviewer-facing files are:

- `evaluation_unconditional/cognitive_result_table.csv`;
- `evaluation_unconditional/cognitive_bootstrap_summary.csv`.

For the command shown above, inspect them with:

```bash
column -s, -t \
  cognitive_model_comparsion/outputs/onestop_ob1_sigma076942/evaluation_unconditional/cognitive_result_table.csv
column -s, -t \
  cognitive_model_comparsion/outputs/onestop_ob1_sigma076942/evaluation_unconditional/cognitive_bootstrap_summary.csv
```

They contain only the ET1 raw, symmetric, and asymmetric rows or their paired
contrasts and record `ob1_tvt` as the reference. Human-target tables remain in
`result_table.csv` and `bootstrap_summary.csv` as stronger external
validation, but are not required to interpret the cognitive-only table. The
special-token-excluded counterparts are under
`evaluation_unconditional_special_excluded/`; when requested, the
punctuation-free paragraph sensitivity is nested under
`ob1_clean_passages/`.

For Provo, the paired unit is a passage. For OneStop, all paragraphs from the
same article share `cluster_id`; the bootstrap and sign-flip test resample or
flip the 30 article clusters so paragraphs from one article are not treated as
independent.

## English-only evaluation scope

The primary Provo and OneStop analyses are intentionally English-only. This is
an applicability constraint, not a post-hoc language selection. The
GazeReward paper states:

> “We filtered all non-English text, as the ET prediction models were exclusively trained on English data.”

Source: [GazeReward, Section 5.1](https://arxiv.org/html/2410.01532v3).

OneStop is not a multilingual dataset. Its four sub-corpora are reading
regimes, and the dataset paper describes the material as follows:

> “native (L1) speakers read newswire texts in English”

Source: [OneStop data descriptor](https://www.nature.com/articles/s41597-025-06272-2).

The current OB1 worker also fixes `language="english"` and uses the
SUBTLEX-UK frequency resource. Applying this exact runner unchanged to Korean,
Hebrew, or another MECO language would not be a controlled comparison.

MECO is multilingual but includes English samples. Its paper states:

> “MECO comprises eye-tracking data for reading in the first (dominant) language, reading in English”

Source: [MECO corpus paper](https://www.utupub.fi/server/api/core/bitstreams/63878634-9a18-403b-8b1a-1cd4a3963e3b/content).

A future MECO replication should therefore use an English L1 sample as a
separate dataset. English L2 readers should be analyzed separately rather than
pooled with L1 readers because language proficiency changes the population
being modeled. This experiment makes no cross-lingual generalization claim.
Language is matched across ET1, OB1, and the Human corpus; genre is not:
OASST1 contains preference conversations, whereas Provo and OneStop contain
passage reading. That remaining domain difference is an external-validation
limitation rather than a language mismatch.

## Component commands

Every component accepts `--corpus provo` or `--corpus onestop` where relevant:

```bash
python cognitive_model_comparsion/main.py predict-et1 \
  --corpus onestop \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
  --output-dir cognitive_model_comparsion/outputs/onestop_et1

python cognitive_model_comparsion/main.py predict-et1 \
  --corpus onestop \
  --sigma-left 0.3738 \
  --sigma-right 3.21289 \
  --sigma-value-type effective \
  --sigma-source-accuracy 0.76942 \
  --checkpoint-id s11_acc076942_l037380_r321289 \
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

The behavior-level comparison tests whether the OASST1-trained redistribution
changes frozen ET1 word allocation toward Human TRT and OB1 TVT on unseen
text. The separate kernel-profile comparison tests only the relative-token
allocation shape of a unit source-token impulse against the reconstructed OB1
fixation-onset attention component; it does not use actual ET1 TRT magnitudes.
Neither analysis:

- refit ET1, sigma, or OB1 on either Human corpus;
- treat OB1 virtual-reader count as Human sample size;
- compare OB1 letter coordinates numerically with T5-token sigma;
- treat word-order Wasserstein as a scanpath measure;
- treat Human TRT as a direct measure of covert perceptual span;
- claim that OB1 has a physical viewing-distance parameter;
- replace the existing OASST1 reward-accuracy experiment.

See `PROVENANCE.md` for source quotes, exact hashes, and audit evidence, and
`WORK_PLAN.md` for the frozen analysis contract.
