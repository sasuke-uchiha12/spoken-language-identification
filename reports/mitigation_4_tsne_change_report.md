# Mitigation 4 + t-SNE Change Report

## Scope
Branch: `feat/task2-all-data-centric-and-Tsne`  
Goal: implement a stronger data-centric Task 2 run and produce Task 3 t-SNE artifacts automatically.

## What Was Added

### 1) Mitigation 4 (Full Data-Centric Augmentation)
Implemented in `train_model.py` and activated from `train_model_mac_m1.py`.

- Added new augmentation config fields:
  - `pitch_shift_prob`, `pitch_shift_semitones_min`, `pitch_shift_semitones_max`
  - `spectral_aug_prob`, `freq_mask_param`, `time_mask_param`
  - `num_freq_masks`, `num_time_masks`
- Refactored augmentation into 3 families:
  - `temporal`: speed perturbation + additive noise
  - `pitch`: pitch shift
  - `spectral`: STFT-domain frequency/time masking + iSTFT reconstruction
- Added **global augmentation gate** and **one-of policy**:
  - First gate by `augmentation_prob`
  - Then pick one family (temporal / pitch / spectral) by configured weights
  - Prevents stacking all augmentations on one sample

### 2) t-SNE Pipeline for Task 3
Implemented in `train_model.py`.

- Added config fields:
  - `run_tsne`, `tsne_max_samples`, `tsne_batch_size`,
    `tsne_perplexity`, `tsne_random_state`, `tsne_speaker_top_k`
- Added embedding extraction from **last hidden layer** on validation set
- Added t-SNE generation and artifact export:
  - `diagnostics/tsne_validation_points.csv`
  - `diagnostics/tsne_validation_by_language.png`
  - `diagnostics/tsne_validation_by_speaker_topk.png`
- Hooked into run end when `run_tsne=True`

### 3) Mac Training Argument Fix
In `train_model_mac_m1.py`, passed `max_grad_norm` into `TrainingArguments` when supported.

## Active Mitigation 4 Config (Mac Profile)
Set in `train_model_mac_m1.py`:

- `enable_train_augmentation = True`
- `augmentation_prob = 0.30`
- `speed_min = 0.98`
- `speed_max = 1.02`
- `noise_std_min = 0.0005`
- `noise_std_max = 0.0018`
- `pitch_shift_prob = 0.20`
- `pitch_shift_semitones_min = -1.0`
- `pitch_shift_semitones_max = 1.0`
- `spectral_aug_prob = 0.20`
- `freq_mask_param = 8`
- `time_mask_param = 12`
- `num_freq_masks = 1`
- `num_time_masks = 1`
- `run_tsne = True`

Interpretation of one-of weights with this setup:

- Temporal weight = `1 - pitch_shift_prob - spectral_aug_prob = 0.60`
- Pitch weight = `0.20`
- Spectral weight = `0.20`

So only 30% samples are augmented, and each augmented sample gets exactly one family.

## Files Changed
- `train_model.py`
  - Added augmentation config fields
  - Added new augmentation logic (temporal/pitch/spectral)
  - Added global gate + one-of selection
  - Added t-SNE extraction and plotting pipeline
  - Added run-end t-SNE hook
- `train_model_mac_m1.py`
  - Enabled Mitigation 4 config
  - Enabled `run_tsne`
  - Added `max_grad_norm` passthrough in Mac training arguments

## Validation Performed
- Syntax check passed:
  - `python3 -m py_compile train_model.py train_model_mac_m1.py`

## Expected Run Outputs
After running:

```bash
uv run train_model_mac_m1.py
```

You should see:

- Normal train/eval artifacts (metrics + diagnostics)
- Plus t-SNE artifacts under run diagnostics:
  - `tsne_validation_points.csv`
  - `tsne_validation_by_language.png`
  - `tsne_validation_by_speaker_topk.png`

## Notes
- This change enables direct baseline vs improved representation analysis for Task 3 in future runs.
- Old runs without model checkpoints still cannot be backfilled into t-SNE from logs alone.
