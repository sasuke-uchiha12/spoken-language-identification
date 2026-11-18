# DANN Mitigation Findings Report (Task 2)

## Run Covered

- Model: `utter-project/mHuBERT-147`
- Branch: `feat/task2-dann-mitigation`
- Run folder: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125`
- Objective: Evaluate model-based speaker-bias mitigation using DANN.

## DANN Settings Used

From `train_model_mac_m1.py`:

- `enable_dann = True`
- `dann_speaker_loss_weight = 0.1`
- `dann_grl_lambda = 1.0`
- `dann_use_lambda_schedule = True`
- `dann_speaker_head_dropout = 0.1`

Run-profile choices used with DANN:

- `metric_for_best_model = "macro_f1"`
- `eval_steps = 100`, `save_steps = 100`
- `enable_train_augmentation = False` (DANN-only isolation run)
- `run_tsne = True`

## Final Metrics (Epoch 3.0)

From `eval_results.json`:

- `eval_accuracy`: **0.5576**
- `eval_macro_f1`: **0.5426**
- `eval_loss`: **1.8815**

From `all_results.json`:

- `predict_accuracy`: `0.5576`
- `predict_macro_f1`: `0.5426`
- `train_loss`: `9.4660`
- `train_runtime`: `29447.90s`

## Best Checkpoint Behavior

From `trainer_state.json`:

- Best checkpoint: `checkpoint-6500`
- `best_metric`: `0.5425585530` (macro-F1)
- Last eval point is the same best point (no late drop after best checkpoint).

Interpretation:

- Checkpoint selection is stable for this run.
- No best-vs-last mismatch in final metrics.

## Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv`:

### Strong classes

- assamese: `0.9067`
- telugu: `0.8533`
- sanskrit: `0.8200`
- gujarati: `0.8067`
- manipuri: `0.7800`
- tamil: `0.7600`

### Weak classes

- hindi: `0.0800`
- konkani: `0.2200`
- dogri: `0.2600`
- nepali: `0.2800`
- bengali: `0.3533`
- odia: `0.3800`

Key note:

- This run significantly improves several previously weak classes (`sindhi`, `maithili`, `santali`, `konkani`) but still leaves `hindi` low.

## Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv`, top confusions:

- `konkani -> marathi` (74)
- `bodo -> manipuri` (45)
- `dogri -> punjabi` (44)
- `hindi -> urdu` (39)
- `nepali -> manipuri` (33)
- `maithili -> gujarati` (30)

Interpretation:

- Legacy high-impact confusions remain, but several are reduced versus earlier mitigations.
- Main unresolved structural confusion is still `konkani -> marathi`.

## Speaker Diagnostic Findings

From `diagnostics/speaker_diagnostic_summary.json`:

- `train_unique_speakers`: `110`
- `validation_unique_speakers`: `681`
- `overlap_speaker_count`: `0`
- `overlap_ratio_validation_speakers`: `0.0`
- `accuracy_unseen_speakers`: `0.5576`
- `warnings`: none

Interpretation:

- Speaker split integrity is clean.
- Unseen-speaker performance is strongest among available runs in this branch.

## t-SNE Artifacts

Generated in `diagnostics/`:

- `tsne_validation_points.csv`
- `tsne_validation_by_language.png`
- `tsne_validation_by_speaker_topk.png`

## Training Dynamics (Quick)

From `trainer_state.json` log history:

- `grad_norm` median: `18.68`
- `grad_norm` p95: `88.36`
- `grad_norm` max: `324.19`
- train loss trend: `14.227 -> 7.283` (minimum logged `6.610`)

Interpretation:

- Optimization progressed with occasional large spikes, but no collapse.

## DANN-Only Conclusion

This DANN run is strong and valid for Task 2:

- Best overall metrics seen in this branch (`0.5576` accuracy, `0.5426` macro-F1).
- Substantial gains on several weak languages.
- Clean speaker split diagnostics and full artifact generation (including t-SNE).

