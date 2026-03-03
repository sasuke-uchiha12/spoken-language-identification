# Mitigation Findings Report (Task 2 - Run 3A)

## Run Covered

- Model: `utter-project/mHuBERT-147`
- Run folder: `TBD_AFTER_RUN`
- Objective: Evaluate Mitigation 3A (Mitigation 1 augmentation strength + macro-F1 checkpointing + denser eval/save cadence).

## Mitigation Settings Used

From `train_model_mac_m1.py`:

- `enable_train_augmentation = True`
- `augmentation_prob = 0.30`
- `speed_min = 0.97`
- `speed_max = 1.03`
- `noise_std_min = 0.0005`
- `noise_std_max = 0.002`
- `eval_steps = 100`
- `save_steps = 100`
- `metric_for_best_model = "macro_f1"`

All other tuned baseline settings were kept unchanged for fair comparison.

## Final Metrics (Epoch 3.0)

From `eval_results.json`:

- `eval_accuracy`: `TBD`
- `eval_macro_f1`: `TBD`
- `eval_loss`: `TBD`

From `all_results.json`:

- `predict_accuracy`: `TBD`
- `predict_macro_f1`: `TBD`
- `train_loss`: `TBD`
- `train_runtime`: `TBD`

## Best Checkpoint Behavior

From `trainer_state.json`:

- Best checkpoint step: `TBD`
- Best checkpoint epoch: `TBD`
- Best metric (`macro_f1`): `TBD`
- Last eval metric snapshot: `TBD`

Interpretation:

- `TBD_AFTER_RUN`

## Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv`:

### Strong classes

- `TBD`

### Weak classes

- `TBD`

Observation:

- `TBD_AFTER_RUN`

## Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv`, top confusions:

- `TBD`

Interpretation:

- `TBD_AFTER_RUN`

## Speaker Diagnostic Findings

From `diagnostics/speaker_diagnostic_summary.json`:

- `train_unique_speakers`: `TBD`
- `validation_unique_speakers`: `TBD`
- `overlap_speaker_count`: `TBD`
- `overlap_ratio_validation_speakers`: `TBD`
- `accuracy_unseen_speakers`: `TBD`
- `warnings`: `TBD`

Interpretation:

- `TBD_AFTER_RUN`

## Training Dynamics Notes

From `trainer_state.json`:

- `grad_norm` median: `TBD`
- `grad_norm` p95: `TBD`
- `grad_norm` max: `TBD`
- Loss trend (logged train loss): `TBD -> TBD`

Interpretation:

- `TBD_AFTER_RUN`

## Mitigation-Only Conclusion (Run 3A)

- `TBD_AFTER_RUN`
