## Installation

To run the project (Linux):

1. Create and activate a Python 3.11 or 3.12 virtual environment.
2. Install the complete dependency manifest:

```bash
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"
python -m pip install -r requirements.txt
```

`requirements.txt` includes the pinned tokenizer-aligner commit and the Linux
`bitsandbytes` dependency used by the default quantized run. CUDA runtime
packages are resolved by the pinned PyTorch wheel instead of being pinned
independently. No download or setup shell script is required after the
dependencies are installed.

The pinned PyTorch 2.12 Linux wheel uses CUDA 13.0, and the pinned
`bitsandbytes` wheel includes `sm_120` kernels for RTX 50-series GPUs. Do not
downgrade these packages to PyTorch 2.2 or `bitsandbytes` 0.43 on an RTX 5090.
When repairing an existing environment that already contains the old pins,
install the complete compatible set together so pip does not upgrade NumPy or
fsspec beyond the versions required by TRL and Datasets:

```bash
python -m pip install --upgrade "torch==2.12.0" "bitsandbytes==0.49.0" "sympy==1.14.0" "rich==13.9.4" "numpy==1.26.4" "fsspec==2024.2.0"
```

Verify the CUDA and 4-bit paths before starting a full run:

```bash
python -c "import torch, bitsandbytes as bnb; x=torch.randn(64, 64, device='cuda', dtype=torch.float16); q,s=bnb.functional.quantize_4bit(x, quant_type='nf4'); y=bnb.functional.dequantize_4bit(q, s); print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.get_device_name(0), 'bnb', bnb.__version__, 'finite', bool(torch.isfinite(y).all()))"
python -c "import main; print('main import OK')"
```

### Hugging Face and Weights & Biases access

Request access to the gated
[Meta-Llama-3-8B repository](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
in a browser while signed in to the same Hugging Face account that will be
used on the training machine. Hugging Face does not support submitting a
gated-model access request from Python or the CLI.

After access is granted, authenticate the training machine:

```bash
python -c "from huggingface_hub import login; login()"
python -c "import wandb; wandb.login()"
```

The first command asks for a Hugging Face user access token. The second asks
for a Weights & Biases API key. Do not put either secret directly in the
command or commit it to the repository.

Verify both the account login and gated-model access before training:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='meta-llama/Meta-Llama-3-8B', filename='config.json'))"
```

### Automatic downloads

Running `python main.py ...` downloads all missing model assets from Python:

- the selected reward-model backbone and dataset through Hugging Face;
- the ET1 checkpoint and pinned `t5-small` tokenizer when
  `--fixations_model_version 1` is selected; or
- the ET2 checkpoint and pinned tokenizer when
  `--fixations_model_version 2` is selected.

The default Meta-Llama backbone is gated. Accept its Hugging Face license and
provide a valid Hugging Face token before the first run. Public ET1 and ET2
assets themselves do not require Google Drive or `gdown`.

ET prediction model 1 is implemented locally and its checkpoint is downloaded
directly from the pinned source file in
[huangxt39/SelectiveCacheForLM](https://github.com/huangxt39/SelectiveCacheForLM/blob/eccc93f969745b04ce1e4911d6513d85565cc919/FPmodels/T5-tokenizer-BiLSTM-TRT-12-concat-3).
The ET1 path does not use `gdown`. The downloaded file is verified by size
and SHA-256 and cached at:

```text
artifacts/et_prediction_model_1/T5-tokenizer-BiLSTM-TRT-12-concat-3
```

The ET1 tokenizer is pinned to `t5-small` revision
`df1b051c49625cf57a3d0d8d3863ed4d13564fe4` and cached under
`cache/models/`. You can override the checkpoint location with
`GAZE_REWARD_ET1_CHECKPOINT=/path/to/checkpoint`, use an offline tokenizer
snapshot with `GAZE_REWARD_ET1_TOKENIZER=/path/to/tokenizer`, or move its cache
with `GAZE_REWARD_ET1_TOKENIZER_CACHE=/path/to/cache`.

ET prediction model 2 is implemented locally. Its pinned safetensors
checkpoint and tokenizer are downloaded from
[skboy/et_prediction_2](https://huggingface.co/skboy/et_prediction_2/tree/5785e77309d9fce8b88e908a9db100c1a0a63456).
The checkpoint is verified as a 498,621,996-byte file with SHA-256
`1a70c01f6a37e897fec8cf0d39ccba8a50ad144f076545cc4f0d8b7d67bf2b40`.
ET2 does not require `eyetrackpy` or `gdown`.

The repository's tokenizer JSON is also pinned and verified. Because this
project pins `tokenizers==0.19.1`, its BPE vocabulary and merges are derived
deterministically into a compatibility cache and rebuilt if either derived
file is incomplete or corrupt. ET2 inference follows the repository's
word-first-token rule and clips negative feature predictions to zero. LMDB
cache keys include both the pinned ET2 preprocessing signature and a digest
of the active reward-model tokenizer.

Use `GAZE_REWARD_HF_CACHE=/path/to/cache` to change the Hugging Face cache.
For an offline installation, set `GAZE_REWARD_ET2_CHECKPOINT` to the
safetensors file and `GAZE_REWARD_ET2_TOKENIZER` to a downloaded
`tokenizer.json` file or its containing directory.

All re-downloadable assets, Hugging Face caches, LMDB runtime caches, W&B
files, and model-output directories are excluded by `.gitignore`. The source
tree can therefore be uploaded without either ET checkpoint.

### ET2 GazeConcat conditions

The original raw-feature condition keeps
`[nFix, FFD, GPT, TRT, fixProp]` unchanged:

```bash
python main.py \
  --fixations_model_version 2 \
  --concat true \
  --use_softprompt true \
  --features_used 1,1,1,1,1 \
  --et2_gaze_concat_condition raw_gaze_concat
```

The additional hybrid condition constructs
`[nFix, FFD, GPT, redistributed_TRT, fixProp]`. Only TRT is passed through the
asymmetric Gaussian redistributor; the other four features remain raw, and the
five-channel tensor then follows the existing GazeConcat projector and token
concatenation path:

```bash
python main.py \
  --fixations_model_version 2 \
  --concat true \
  --use_softprompt true \
  --features_used 1,1,1,1,1 \
  --et2_gaze_concat_condition trt_redistributed_gaze_concat
```

The hybrid condition requires all five ET2 features. If the condition is
omitted, the existing `--use_asym_gaussian_redistributor` value selects the
backward-compatible behavior.

## Usage

Example: Training with OASST1 dataset using Meta-Llama-3-8B

```bash
python main.py \
  -d OpenAssistant/oasst1 \
  -m meta-llama/Meta-Llama-3-8B \
  --concat True \
  --seed 42
```

### Key Parameters

- `-d, --dataset_name`: Dataset to use for training. Currently, only OpenAssistant/oasst1 and nvidia/HelpSteer2 are supported.
- `-m, --model_name`: Base model to fine-tune. You can pass full model IDs `meta-llama/Meta-Llama-3-8B`, `meta-llama/Llama-3-8B-Instruct`
- `--concat`: Whether to concatenate prompt and response. True is of GazeConcat and False is of GazeAdd.
- `--max_length`: Character-length dataset filter. The default is `5000`; a
  command-line value overrides it for both ET1 and ET2.
- `--max_tokens`: Optional per-sequence token truncation limit. It is unset by
  default, so tokenization does not apply an additional explicit token cap.
  Pass a positive integer when a strict token limit is required. For custom or
  unusually token-dense text, set this explicitly so the text-plus-gaze
  sequence cannot exceed the backbone model's context window.

For example, change only the character limit:

```bash
python main.py --fixations_model_version 1 --max_length 7000
```

Or retain the 5000-character filter and add an explicit token cap:

```bash
python main.py \
  --fixations_model_version 2 \
  --max_length 5000 \
  --max_tokens 1350
```

Neither the ET model version nor the GazeConcat condition overrides these
command-line values.

### Experiment output paths

Training outputs default to the repository-local `models_save/` directory.
Use `--output_root /path/to/output` to select another location. Each run uses
a compact readable identifier plus a digest of the complete resolved training
configuration, so every path component remains below the Linux filename limit.
The full resolved configuration, experiment digest, and output path are
written to `args.json` before model or dataset loading begins.
