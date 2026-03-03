# BERT-Only SLID Report (mHuBERT-147)

## 1) Scope

This report covers only the BERT-family audio model used in this project:

- Model: `utter-project/mHuBERT-147`
- Dataset: `badrex/nnti-dataset-full` (`train` + `validation`)
- Script family: `train_model.py` + `train_model_mac_m1.py`
- Platform: Mac run outputs under `indic-SLID-mac/`

Compared runs:

- **Without tuning**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826`
- **With tuning**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502`

## 2) Run Artifacts Used

For both runs, analysis was done from:

- `eval_results.json`
- `all_results.json`
- `train_results.json`
- `trainer_state.json`
- `diagnostics/confusion_matrix_validation.csv`
- `diagnostics/per_language_accuracy_validation.csv`
- `diagnostics/speaker_diagnostic_summary.json`

## 3) Final Metrics Comparison

| Metric | Without tuning | With tuning | Delta (with - without) |
|---|---:|---:|---:|
| Eval Accuracy | 0.4676 | 0.5270 | **+0.0594** |
| Eval Macro-F1 | 0.4142 | 0.4748 | **+0.0607** |
| Eval Loss | 2.2854 | 2.1629 | **-0.1226** |
| Predict Accuracy | 0.4676 | 0.5270 | +0.0594 |
| Predict Macro-F1 | 0.4142 | 0.4748 | +0.0607 |

Key conclusion: tuning produced a strong and consistent uplift across accuracy, macro-F1, and loss.

## 4) Checkpoint Behavior

### Without tuning

- Best eval accuracy: `0.4676` at step `6200` (epoch `~2.85`)
- Best eval macro-F1: `0.4142` at step `6200`

### With tuning

- Best eval accuracy: `0.5270` at step `6000` (epoch `~2.76`)
- Best eval macro-F1: `0.4757` at step `6400` (epoch `~2.95`)

Interpretation:

- Best-accuracy checkpoint and best-F1 checkpoint are close in the tuned run.
- The model is stable late in training, with small movement between steps `6000` and `6400`.

## 5) Per-Language Performance

### Strong languages (with tuning)

- assamese: `0.8800`
- punjabi: `0.8800`
- telugu: `0.8467`
- manipuri: `0.8133`
- tamil: `0.8000`

### Weak languages (with tuning)

- konkani: `0.0000`
- maithili: `0.0000`
- hindi: `0.0333`
- sindhi: `0.0333`
- dogri: `0.1933`

### Largest per-language gains from tuning

- nepali: `+0.3533` (`0.34 -> 0.6933`)
- santali: `+0.1600` (`0.06 -> 0.22`)
- punjabi: `+0.1333` (`0.7467 -> 0.88`)
- manipuri: `+0.1000` (`0.7133 -> 0.8133`)
- kashmiri: `+0.0867` (`0.68 -> 0.7667`)

### Regressions

- marathi: `-0.0800` (`0.7267 -> 0.6467`)
- kannada: `-0.0200` (`0.6667 -> 0.6467`)

Observation:

- Tuning improved many languages substantially, but class imbalance/difficulty remains for a few classes.
- Zero-accuracy classes stayed at 2 (`konkani`, `maithili`) in both runs.

## 6) Confusion Matrix Findings

Top confusions in tuned run:

- `sindhi -> punjabi` (103)
- `konkani -> marathi` (97)
- `hindi -> urdu` (77)
- `dogri -> punjabi` (76)
- `bengali -> odia` (39)

What this means:

- The model is still collapsing specific low-performing classes into acoustically/linguistically close classes.
- This is the biggest remaining accuracy ceiling for this model/run setup.

## 7) Speaker-Bias Diagnostic

From `speaker_diagnostic_summary.json` (tuned run):

- Train unique speakers: `110`
- Validation unique speakers: `681`
- Overlap speaker count: `0`
- Overlap ratio: `0.0`
- Accuracy on unseen speakers: `0.52697`
- Warnings: none

Conclusion:

- This run does not show speaker leakage between train and validation speakers.

## 8) Training Dynamics

### Without tuning

- Train runtime: `14455.75s`
- Train steps/sec: `0.451`
- Loss (start -> end): `12.3649 -> 7.6096`

### With tuning

- Train runtime: `19822.39s`
- Train steps/sec: `0.329`
- Loss (start -> end): `12.3659 -> 7.1181`

Interpretation:

- Tuned run is slower but converges better (lower final train and eval loss).
- Cost-performance tradeoff is favorable here because final quality gains are large.

## 9) Overall Conclusion (BERT-only)

For `utter-project/mHuBERT-147`, the tuned setup is clearly superior to the untuned one:

- Better global metrics (accuracy, macro-F1, loss)
- Better performance on many languages
- No speaker overlap leakage

Remaining bottleneck:

- Specific confusion clusters (`konkani/marathi`, `hindi/urdu`, `sindhi/punjabi`) and two dead classes.

## 10) Recommended Next Tuning (Priority Order)

### Priority 1: checkpoint selection by macro-F1

Current best-model selection is accuracy-oriented. Add/confirm:

- `metric_for_best_model = "macro_f1"`
- `greater_is_better = True`

Why:

- Macro-F1 penalizes dead/weak classes and is better aligned to 22-language balance.

### Priority 2: enable mild train-only augmentation

Try:

- `enable_train_augmentation = True`
- keep augmentation probability low (for example `0.2` to `0.3`)
- preserve deterministic seed

Why:

- Can reduce overfitting and improve robustness on weak/confused language pairs.

### Priority 3: one longer run (4 epochs)

Try:

- `num_train_epochs = 4`

Why:

- Tuned curve still improves near epoch 3, so another epoch may add gains.

### Priority 4: class-sensitive objective (if needed)

If weak classes remain near zero:

- weighted cross-entropy or focal loss

Why:

- Directly addresses minority/underperforming class collapse.

## 11) Suggested Reporting Statement (for submission)

Use this concise statement in project report:

> We evaluated a BERT-family audio model (`utter-project/mHuBERT-147`) on `badrex/nnti-dataset-full` and compared untuned vs tuned runs. The tuned run improved validation accuracy from 46.76% to 52.70% (+5.94 points) and macro-F1 from 0.4142 to 0.4748 (+0.0606), with lower eval loss (2.2854 -> 2.1629). Speaker-disjoint diagnostics showed zero train/validation speaker overlap, indicating gains are not due to speaker leakage. Remaining errors are concentrated in specific language confusion pairs such as Konkani-Marathi and Hindi-Urdu.

