# Mitigation Findings Report (Task 2 - Run 3A)

## Run Covered

- Model: `utter-project/mHuBERT-147`
- Run folder: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_170850`
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

- `eval_accuracy`: **0.4979**
- `eval_macro_f1`: **0.4561**
- `eval_loss`: **2.1884**

From `all_results.json`:

- `predict_accuracy`: `0.4979`
- `predict_macro_f1`: `0.4561`
- `train_loss`: `9.1831`
- `train_runtime`: `33249.22s`

## Best Checkpoint Behavior

From `trainer_state.json`:

- Best checkpoint by accuracy: step `5800`, epoch `~2.6694`
- Best checkpoint by macro-F1: step `5800`, epoch `~2.6694`
- Last eval snapshot: step `6500`, epoch `~2.9916`, `eval_accuracy=0.4839`, `eval_macro_f1=0.4409`
- Saved best checkpoint path: `checkpoint-5800`

Interpretation:

- Run peaked before the end and then degraded.
- `load_best_model_at_end=True` remained important.

## Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv` and `eval_results.json`:

### Strong classes

- `assamese`: `0.8667`
- `manipuri`: `0.8533`
- `punjabi`: `0.8200`
- `malayalam`: `0.7733`
- `tamil`: `0.7733`
- `sanskrit`: `0.7467`
- `gujarati`: `0.7333`
- `telugu`: `0.7333`

### Weak classes

- `konkani`: `0.0000`
- `sindhi`: `0.0333`
- `maithili`: `0.1067`
- `hindi`: `0.1400`
- `dogri`: `0.1467`
- `odia`: `0.1867`

Observation:

- Some weak classes improved vs Mitigation 1 (`maithili`, `hindi`, `odia`, `santali`).
- Large regressions appeared in multiple previously stronger classes (`nepali`, `bodo`, `telugu`, `malayalam`, `tamil`).

## Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv`, top confusions:

- `sindhi -> punjabi` (98)
- `konkani -> marathi` (90)
- `bodo -> manipuri` (62)
- `maithili -> gujarati` (59)
- `dogri -> punjabi` (56)
- `odia -> bengali` (49)
- `hindi -> urdu` (47)
- `santali -> bengali` (39)

Interpretation:

- Some confusions reduced vs earlier runs (`konkani -> marathi`, `hindi -> urdu`), but severe confusion remained (`sindhi -> punjabi`) and new pressure appeared in other pairs.

## Speaker Diagnostic Findings

From `diagnostics/speaker_diagnostic_summary.json`:

- `train_unique_speakers`: `110`
- `validation_unique_speakers`: `681`
- `overlap_speaker_count`: `0`
- `overlap_ratio_validation_speakers`: `0.0`
- `accuracy_unseen_speakers`: `0.4979`
- `warnings`: `[]`

Interpretation:

- Split remains speaker-disjoint with no leakage warnings.
- Unseen-speaker accuracy is lower than Mitigation 1 and Mitigation 2.

## Training Dynamics Notes

From `trainer_state.json`:

- `grad_norm` median: `7.8562`
- `grad_norm` p95: `61.3861`
- `grad_norm` max: `323.4677`
- Loss trend (logged train loss): `12.3671 -> 7.2659`

Interpretation:

- Optimization progressed but with occasional large spikes.
- Late-stage metric drop indicates overtraining or unstable late optimization under this setup.

## Mitigation-Only Conclusion (Run 3A)

Mitigation 3A is valid as an ablation but not as the final Task 2 model:

- It improved a few weak classes.
- It regressed overall quality substantially vs Mitigation 1/2 (accuracy, macro-F1, and eval loss).
- Recommended final model remains Mitigation 1 for Task 2, with 3A used as tradeoff evidence.
