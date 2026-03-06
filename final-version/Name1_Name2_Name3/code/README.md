# Spoken Language Identification (22 Indian Languages)

This project trains a spoken-language identification model on `badrex/nnti-dataset-full` using `utter-project/mHuBERT-147`.

Current defaults in both entry points are aligned to the best DANN profile:
- DANN enabled
- Train augmentation disabled
- Best checkpoint selected by `macro_f1`
- t-SNE diagnostics enabled

## Python Version

- Python `3.13` (from `.python-version`)

## Environment Setup (Local)

### Option A: Standard `venv` + `pip` (primary)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: `uv` with lockfile (optional equivalent)

```bash
uv sync 
```

This uses `pyproject.toml` + `uv.lock` and creates a `.venv` automatically.

## How to Run

### 1) Base runner (cross-platform)

```bash
python train_model.py
```

With `uv`:

```bash
uv run train_model.py
```

### 2) MacBook MPS wrapper (recommended on Apple Silicon)

```bash
python train_model_mac_m1.py
```

With `uv`:

```bash
uv run train_model_mac_m1.py
```

`train_model_mac_m1.py` is a run wrapper over `train_model.py` that applies Mac MPS-friendly runtime settings (batch size, accumulation, eval/save cadence), while keeping the same DANN default model behavior.

## Optional Environment Variables

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=Indic-SLID
```

- `HF_TOKEN`: only needed for authenticated Hugging Face Hub access
- `WANDB_API_KEY`: only needed if W&B logging is enabled
- `WANDB_PROJECT`: optional W&B project name

## Output Artifacts

Run folders are created under:
- `./indic-SLID/<run_name>/` (base runner)
- `./indic-SLID-mac/<run_name>/` (Mac wrapper)

Diagnostics include:
- `per_language_accuracy_validation.csv`
- `confusion_matrix_validation.csv`
- `confusion_matrix_validation.png`
- `speaker_accuracy_validation.csv`
- `speaker_diagnostic_summary.json`
- `tsne_validation_points.csv`
- `tsne_validation_by_language.png`
- `tsne_validation_by_speaker_topk.png`
