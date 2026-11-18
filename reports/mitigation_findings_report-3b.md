# Mitigation Findings Report (Task 2 - Run 3B)

## Run Covered

- Model: `utter-project/mHuBERT-147`
- Run folder: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260304_061005`
- Objective: Evaluate Mitigation 3B (milder augmentation fallback after 3A).

## Mitigation Settings Used

From the Mitigation 3B plan/profile:

- `enable_train_augmentation = True`
- `augmentation_prob = 0.28`
- `speed_min = 0.975`
- `speed_max = 1.025`
- `noise_std_min = 0.0005`
- `noise_std_max = 0.0018`
- `metric_for_best_model = "macro_f1"`
- `eval_steps = 100`
- `save_steps = 100`

All other tuned baseline settings were kept unchanged for fair comparison.

## Final Metrics (Epoch 3.0)

From `eval_results.json`:

- `eval_accuracy`: **0.5276**
- `eval_macro_f1`: **0.4817**
- `eval_loss`: **2.1456**

From `all_results.json`:

- `predict_accuracy`: `0.5276`
- `predict_macro_f1`: `0.4817`
- `train_loss`: `9.0909`
- `train_runtime`: `33033.74s`

## Best Checkpoint Behavior

From `trainer_state.json`:

- Best checkpoint by selected metric (macro-F1): step `5600`, epoch `~2.5773`
- Best checkpoint metrics:
  - `eval_accuracy = 0.5276`
  - `eval_macro_f1 = 0.4817`
  - `eval_loss = 2.1456`
- Saved best checkpoint path: `checkpoint-5600`
- Last eval snapshot (step `6500`, epoch `~2.9916`):
  - `eval_accuracy = 0.5239`
  - `eval_macro_f1 = 0.4730`
  - `eval_loss = 2.1237`

Interpretation:

- The run peaked before the end and dropped slightly afterward.
- `load_best_model_at_end=True` was necessary to retain the best checkpoint.

## Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv` and `eval_results.json`:

### Strong classes

- `assamese`: `0.8600`
- `manipuri`: `0.8467`
- `punjabi`: `0.8733`
- `nepali`: `0.8133`
- `marathi`: `0.7733`
- `gujarati`: `0.7800`
- `tamil`: `0.7800`
- `telugu`: `0.8000`

### Weak classes

- `konkani`: `0.0000`
- `maithili`: `0.0133`
- `hindi`: `0.0600`
- `odia`: `0.1000`
- `dogri`: `0.1467`
- `sindhi`: `0.1400`

Observation:

- Compared to Mitigation 1, 3B improved `nepali`, `bengali`, `kannada`, `sindhi`.
- 3B regressed `malayalam`, `sanskrit`, `bodo`, `kashmiri`, `tamil`.
- `konkani` remains unresolved (`0.0`).

## Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv`, top confusions:

- `konkani -> marathi` (125)
- `sindhi -> punjabi` (87)
- `hindi -> urdu` (73)
- `dogri -> punjabi` (66)
- `santali -> bengali` (52)
- `odia -> bengali` (48)
- `bodo -> manipuri` (46)
- `maithili -> gujarati` (42)

Interpretation:

- `sindhi -> punjabi` improved vs Mitigation 1, but
- `konkani -> marathi` worsened and remains the dominant failure.

## Speaker Diagnostic Findings

From `diagnostics/speaker_diagnostic_summary.json`:

- `train_unique_speakers`: `110`
- `validation_unique_speakers`: `681`
- `overlap_speaker_count`: `0`
- `overlap_ratio_validation_speakers`: `0.0`
- `accuracy_unseen_speakers`: `0.5276`
- `warnings`: `[]`

Interpretation:

- Split remains speaker-disjoint with no leakage warning.
- Generalization to unseen speakers is solid, but still below Mitigation 1.

## Training Dynamics Notes

From `trainer_state.json`:

- `grad_norm` median: `9.1644`
- `grad_norm` p95: `72.9056`
- `grad_norm` max: `354.6017`
- Logged train loss trend: `12.3653 -> 7.2722`

Interpretation:

- Optimization progressed, but gradient spikes still appear.
- Metrics plateau and slight late drop remain present.

## Mitigation-Only Conclusion (Run 3B)

Mitigation 3B is a recovery over 3A, but not the best Task 2 model:

- Better than 3A on all global metrics.
- Nearly tied with Mitigation 2.
- Still below Mitigation 1 on both `eval_accuracy` and `eval_macro_f1`.

Recommended ranking remains:

1. **Mitigation 1**
2. Mitigation 2 / 3B (close)
3. Mitigation 3A
