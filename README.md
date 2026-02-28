# Spoken Language Identification (22 Indian Languages)

Fine-tuning `facebook/mms-300m` on the Hugging Face dataset `badrex/nnti-dataset-full` for spoken language identification.

## Prerequisites

- Python 3.10+ (3.11/3.12 recommended)
- `uv` installed (`uv --version`)

## Setup (using `uv`)

### 1) Create a virtual environment

```bash
uv venv .venv
```

### 2) Activate it

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
uv pip install -r requirements.txt
```

## Run Training

Baseline run (augmentation is off by default):

```bash
python train_model.py
```

You can also run directly with `uv` (without activating the venv):

```bash
uv run train_model.py
```

## Optional Environment Variables

These are optional. The script runs without them.

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=Indic-SLID
```

- `HF_TOKEN`: only needed for Hugging Face login/push/private access
- `WANDB_API_KEY`: only needed if you want Weights & Biases logging
- `WANDB_PROJECT`: optional W&B project name (default is `Indic-SLID`)

## Outputs

Training artifacts are saved under:

```bash
./indic-SLID/<run_name>/
```

Diagnostics are saved under:

```bash
./indic-SLID/<run_name>/diagnostics/
```

Expected diagnostics:
- `per_language_accuracy_validation.csv`
- `confusion_matrix_validation.csv`
- `confusion_matrix_validation.png`
- `speaker_accuracy_validation.csv`
- `speaker_diagnostic_summary.json`
