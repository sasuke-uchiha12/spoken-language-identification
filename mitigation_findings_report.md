# Mitigation Findings Report (Task 2)

## Run Covered

- Model: `utter-project/mHuBERT-147`
- Run folder: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- Objective: Evaluate speaker-bias mitigation using train-only augmentation.

## Mitigation Settings Used

From `train_model_mac_m1.py`:

- `enable_train_augmentation = True`
- `augmentation_prob = 0.3`
- `speed_min = 0.97`
- `speed_max = 1.03`
- `noise_std_min = 0.0005`
- `noise_std_max = 0.002`

All other tuned baseline settings were kept unchanged for fair comparison.

## Final Metrics (Epoch 3.0)

From `eval_results.json`:

- `eval_accuracy`: **0.5388**
- `eval_macro_f1`: **0.4868**
- `eval_loss`: **2.1504**

From `all_results.json`:

- `predict_accuracy`: `0.5388`
- `predict_macro_f1`: `0.4868`
- `train_loss`: `9.1482`
- `train_runtime`: `23947.48s`

## Best Checkpoint Behavior

From `trainer_state.json`:

- Best accuracy checkpoint: step `5800` (epoch `~2.669`)
- Best macro-F1 checkpoint: step `5800` (same point)
- Last eval point (step `6400`) was slightly lower in accuracy/f1 than best checkpoint.

Interpretation:

- Mitigation run peaked before final step.
- `load_best_model_at_end=True` is important and correctly used.

## Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv`:

### Strong classes

- malayalam: `0.8667`
- tamil: `0.8600`
- assamese: `0.8533`
- punjabi: `0.8533`
- manipuri: `0.8467`
- telugu: `0.8400`

### Weak classes

- konkani: `0.0000`
- maithili: `0.0267`
- sindhi: `0.0533`
- hindi: `0.0800`
- odia: `0.1200`
- dogri: `0.1533`

Observation:

- Some previously near-zero classes improved (notably `maithili`, `hindi`, `sindhi`).
- `konkani` remained unresolved.

## Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv`, top confusions:

- `konkani -> marathi` (106)
- `sindhi -> punjabi` (102)
- `dogri -> punjabi` (71)
- `hindi -> urdu` (68)
- `maithili -> gujarati` (49)
- `odia -> bengali` (45)
- `santali -> bengali` (40)
- `bodo -> manipuri` (39)

Interpretation:

- Major confusion clusters still exist, but some high-impact pairs improved in count compared to control (see comparison report).

## Speaker Diagnostic Findings

From `diagnostics/speaker_diagnostic_summary.json`:

- `train_unique_speakers`: `110`
- `validation_unique_speakers`: `681`
- `overlap_speaker_count`: `0`
- `overlap_ratio_validation_speakers`: `0.0`
- `accuracy_unseen_speakers`: `0.5388`
- `warnings`: none

Interpretation:

- No train/validation speaker overlap.
- No leakage warnings were triggered by current heuristics.

## Training Dynamics Notes

From `trainer_state.json`:

- `grad_norm` median: `9.12`
- `grad_norm` p95: `62.76`
- `grad_norm` max: `233.42`
- Loss trend (logged train loss): `12.3671 -> 7.2645`

Interpretation:

- Run shows expected optimization progress with occasional spikes, but no collapse pattern.

## Mitigation-Only Conclusion

This mitigation run is successful for Task 2:

- Achieved strong final metrics (`0.5388` accuracy, `0.4868` macro-F1).
- Improved several weak-language accuracies.
- Maintained clean speaker diagnostic behavior (no overlap leakage).
- Still has residual confusion clusters (`konkani/marathi`, `sindhi/punjabi`, `hindi/urdu`) for future refinement.

