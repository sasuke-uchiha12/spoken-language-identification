# Mitigation Findings Report (Task 2 - Run 2)

## Run Covered

- Model: `utter-project/mHuBERT-147`
- Run folder: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_071843`
- Objective: Evaluate second mitigation run with gentler train-only augmentation and macro-F1-based checkpoint selection.

## Mitigation Settings Used

From `train_model_mac_m1.py`:

- `enable_train_augmentation = True`
- `augmentation_prob = 0.25`
- `speed_min = 0.98`
- `speed_max = 1.02`
- `noise_std_min = 0.0005`
- `noise_std_max = 0.0015`
- `metric_for_best_model = "macro_f1"`

All other tuned baseline settings were kept unchanged for fair comparison.

## Final Metrics (Epoch 3.0)

From `eval_results.json`:

- `eval_accuracy`: **0.5279**
- `eval_macro_f1`: **0.4828**
- `eval_loss`: **2.1385**

From `all_results.json`:

- `predict_accuracy`: `0.5279`
- `predict_macro_f1`: `0.4828`
- `train_loss`: `9.0369`
- `train_runtime`: `22226.29s`

## Best Checkpoint Behavior

From `trainer_state.json`:

- Best accuracy checkpoint: step `5400` (epoch `~2.485`)
- Best macro-F1 checkpoint: step `5400` (same point)
- Last eval point (step `6400`) dropped to `0.5185` accuracy / `0.4715` macro-F1.

Interpretation:

- Run peaked before the final checkpoint.
- Best-checkpoint loading remains important.

## Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv`:

### Strong classes

- assamese: `0.8733`
- manipuri: `0.8667`
- sanskrit: `0.8200`
- tamil: `0.8200`
- telugu: `0.8133`
- nepali: `0.7400`
- kashmiri: `0.7400`

### Weak classes

- maithili: `0.0000`
- konkani: `0.0133`
- hindi: `0.0667`
- odia: `0.0733`
- dogri: `0.1667`
- santali: `0.2200`

Observation:

- Some difficult classes improved strongly compared to Mitigation 1 (notably `sindhi`, `bengali`).
- However, several previously strong classes regressed (for example `punjabi`, `malayalam`, `gujarati`, `kannada`).

## Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv`, top confusions:

- `konkani -> marathi` (101)
- `odia -> bengali` (71)
- `santali -> bengali` (62)
- `hindi -> urdu` (62)
- `maithili -> bengali` (51)
- `bodo -> manipuri` (51)
- `sindhi -> punjabi` (46)
- `dogri -> sindhi` (35)

Interpretation:

- Some earlier confusion clusters improved (`sindhi -> punjabi` reduced), but others intensified (`odia -> bengali`, `maithili -> bengali`).

## Speaker Diagnostic Findings

From `diagnostics/speaker_diagnostic_summary.json`:

- `train_unique_speakers`: `110`
- `validation_unique_speakers`: `681`
- `overlap_speaker_count`: `0`
- `overlap_ratio_validation_speakers`: `0.0`
- `accuracy_unseen_speakers`: `0.5279`
- `warnings`: none

Interpretation:

- No train/validation speaker overlap.
- No leakage warnings were triggered.

## Training Dynamics Notes

From `trainer_state.json`:

- `grad_norm` median: `8.68`
- `grad_norm` p95: `91.03`
- `grad_norm` max: `305.25`
- Loss trend (logged train loss): `12.3649 -> 7.0856`

Interpretation:

- Optimization progressed normally with occasional gradient spikes.
- No collapse behavior, but late-stage quality declined slightly from best checkpoint.

## Mitigation-Only Conclusion (Run 2)

This second mitigation run is valid and provides useful insights:

- Achieved strong metrics (`0.5279` accuracy, `0.4828` macro-F1) and lower eval loss.
- Improved certain weak classes significantly (`sindhi`, `bengali`), showing de-biasing potential.
- Did not surpass Mitigation 1 overall due to regressions in multiple strong classes.
- Best used as an analysis/tradeoff run, while Mitigation 1 remains the stronger overall Task 2 result. 
