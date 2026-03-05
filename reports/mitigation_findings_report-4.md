# Mitigation Findings Report (Task 2 - Run 4: All Data-Centric + t-SNE)

## Run Covered

- Model: `utter-project/mHuBERT-147`
- Run folder: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260304_160046`
- Objective: Evaluate the all data-centric mitigation setup and generate Task 3 t-SNE artifacts.

## Settings Used (Run 4)

From current branch config (`feat/task2-all-data-centric-and-Tsne`):

- Train augmentation enabled with global gate
- Temporal augmentation (speed + noise)
- Pitch-shift augmentation
- Spectral augmentation
- t-SNE generation enabled (`run_tsne=True`)

## Final Metrics (Epoch 3.0)

From `eval_results.json`:

- `eval_accuracy`: **0.5115**
- `eval_macro_f1`: **0.4664**
- `eval_loss`: **2.2127**

From `all_results.json`:

- `predict_accuracy`: `0.5115`
- `predict_macro_f1`: `0.4664`
- `train_loss`: `9.2877`
- `train_runtime`: `21550.54s`

## Best Checkpoint Behavior

From `trainer_state.json`:

- Best checkpoint: `checkpoint-6200` (epoch `~2.8535`)
- `best_metric`: `0.5115151515`
- Last eval at step `6400` was slightly lower:
  - `eval_accuracy = 0.5106`
  - `eval_macro_f1 = 0.4642`

Interpretation:

- Run improved into late training, then stabilized/plateaued.
- Best checkpoint and final behavior are consistent (small late fluctuation).

## Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv`:

### Strong classes

- `assamese`: `0.9333`
- `telugu`: `0.8400`
- `manipuri`: `0.8133`
- `punjabi`: `0.7933`
- `sanskrit`: `0.7600`
- `tamil`: `0.7600`

### Weak classes

- `konkani`: `0.0000`
- `maithili`: `0.0400`
- `sindhi`: `0.0400`
- `hindi`: `0.0667`
- `dogri`: `0.2067`
- `odia`: `0.2933`

Observation:

- Several classes remain difficult despite stronger augmentation.
- `konkani` remains unresolved (zero accuracy).

## Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv`, top confusions:

- `konkani -> marathi` (103)
- `sindhi -> punjabi` (88)
- `bodo -> manipuri` (56)
- `dogri -> punjabi` (54)
- `hindi -> urdu` (49)
- `santali -> bengali` (40)

Interpretation:

- Dominant confusion pairs are still similar to earlier runs.
- Largest structural error remains `konkani -> marathi`.

## Speaker Bias Diagnostic

From `diagnostics/speaker_diagnostic_summary.json`:

- `overlap_speaker_count`: `0`
- `overlap_ratio_validation_speakers`: `0.0`
- `warnings`: `[]`
- `accuracy_unseen_speakers`: `0.5115`

Interpretation:

- No speaker leakage warning is triggered.
- Speaker shortcut risk is mitigated at split level.
- However, this alone does not prove speaker bias is fully removed in representations.

## t-SNE Artifacts (Task 3 Ready)

Generated successfully in diagnostics:

- `tsne_validation_points.csv`
- `tsne_validation_by_language.png`
- `tsne_validation_by_speaker_topk.png`

## Run-Only Conclusion

Run 4 is valid and complete (metrics + diagnostics + t-SNE), but it is not the best overall performer in this branch:

- Better than the older baseline-like run (`20260301_094826`)
- Worse than the stronger tuned reference run (`20260301_150502`) on global metrics.
