# BERT Model Report (After Tuning Only)

## 1) Run Scope

- Model: `utter-project/mHuBERT-147`
- Dataset: `badrex/nnti-dataset-full` (`train`/`validation`)
- Run folder: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502`
- Platform profile: Mac (`train_model_mac_m1.py` overrides active)

## 2) Final Validation Results

From `eval_results.json`:

- `eval_accuracy`: **0.526969696969697** (52.70%)
- `eval_macro_f1`: **0.4748076409415872**
- `eval_loss`: **2.162879228591919**
- `epoch`: `3.0`
- `eval_runtime`: `317.0011s`
- `eval_samples_per_second`: `10.41`

From `all_results.json`:

- `predict_accuracy`: `0.526969696969697`
- `predict_macro_f1`: `0.4748076409415872`
- `train_loss`: `9.099448531587205`
- `train_runtime`: `19822.3892s`

## 3) Best Checkpoint / Training Dynamics

From `trainer_state.json`:

- Best checkpoint for accuracy: `checkpoint-6000`
- `best_metric`: `0.526969696969697`
- `global_step`: `6519`
- `num_train_epochs`: `3`

Last eval points show stable late training:

- epoch `2.301`: acc `0.5042`, f1 `0.4476`, loss `2.2297`
- epoch `2.393`: acc `0.5185`, f1 `0.4679`, loss `2.2029`
- epoch `2.485`: acc `0.5242`, f1 `0.4747`, loss `2.1941`
- epoch `2.577`: acc `0.5206`, f1 `0.4702`, loss `2.1998`
- epoch `2.669`: acc `0.5236`, f1 `0.4692`, loss `2.1874`
- epoch `2.761`: acc `0.5270`, f1 `0.4748`, loss `2.1629` (best acc)
- epoch `2.853`: acc `0.5215`, f1 `0.4709`, loss `2.1681`
- epoch `2.946`: acc `0.5252`, f1 `0.4757`, loss `2.1590` (best f1)

Conclusion:

- The run converged well and remained stable near the end.
- Accuracy peak and macro-F1 peak are both near the final phase.

## 4) Per-Language Findings

From `diagnostics/per_language_accuracy_validation.csv`:

### Strong classes

- assamese: `0.88`
- punjabi: `0.88`
- telugu: `0.8466666666666667`
- manipuri: `0.8133333333333334`
- tamil: `0.8`
- malayalam: `0.7866666666666666`
- gujarati: `0.7733333333333333`
- kashmiri: `0.7666666666666667`
- sanskrit: `0.7666666666666667`

### Moderate classes

- urdu: `0.6933333333333334`
- nepali: `0.6933333333333334`
- kannada: `0.6466666666666666`
- marathi: `0.6466666666666666`
- bodo: `0.5266666666666666`
- odia: `0.30666666666666664`
- bengali: `0.2866666666666667`
- santali: `0.22`
- dogri: `0.19333333333333333`

### Weak / failed classes

- hindi: `0.03333333333333333`
- sindhi: `0.03333333333333333`
- konkani: `0.0`
- maithili: `0.0`

Key takeaway:

- Overall model is strong, but a small set of languages remains collapsed.

## 5) Confusion Matrix Findings

From `diagnostics/confusion_matrix_validation.csv` (largest off-diagonal confusions):

- `sindhi -> punjabi` (103)
- `konkani -> marathi` (97)
- `hindi -> urdu` (77)
- `dogri -> punjabi` (76)
- `bengali -> odia` (39)
- `maithili -> gujarati` (37)
- `santali -> bengali` (34)
- `bodo -> manipuri` (29)
- `santali -> bodo` (28)
- `maithili -> odia` (26)

Interpretation:

- Errors are concentrated in specific language-pair confusions rather than random noise.
- Weak classes are mostly being absorbed by stronger nearby classes.

## 6) Speaker-Disjoint Diagnostic

From `diagnostics/speaker_diagnostic_summary.json`:

- `train_unique_speakers`: `110`
- `validation_unique_speakers`: `681`
- `overlap_speaker_count`: `0`
- `overlap_ratio_validation_speakers`: `0.0`
- `accuracy_unseen_speakers`: `0.526969696969697`
- `warnings`: `[]`

Conclusion:

- No train/validation speaker overlap in this run.
- Reported accuracy is on unseen validation speakers, reducing leakage concern.

## 7) What This Run Demonstrates

- Tuning achieved a strong result for this setup:
  - ~52.7% validation accuracy
  - ~0.475 macro-F1
- Learning is stable near the end of training.
- Performance is high for many languages, but still bottlenecked by a few hard classes and persistent confusion pairs.

## 8) Recommended Next Steps (from tuned-run findings)

1. Use macro-F1 for best-checkpoint selection (`metric_for_best_model="macro_f1"`) to optimize class balance directly.
2. Keep `max_duration_sec=7` and `max_grad_norm=1.0` (current tuned setup is working).
3. Add mild train-only augmentation to target confusion robustness (`enable_train_augmentation=True` with low probability).
4. Consider one longer run (`num_train_epochs=4`) since late-epoch metrics are still competitive.
5. If weak classes remain near zero, apply class-weighted loss or focal loss.

