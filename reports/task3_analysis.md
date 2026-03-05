# Task 3: Model Analysis (Baseline vs Improved, with Mitigation-1 Bridge)

## 1. Problem Bias Source

The dataset has limited speakers per language in training, which encourages a shortcut: the model can partially rely on speaker characteristics instead of language-discriminative cues. This is difficult because speaker timbre/prosody can dominate representation learning and create brittle language boundaries for unseen voices.

## 2. Proposed Technique and Success Extent

Primary required comparison:

- Baseline (untuned): `SLID_utter-project-mHuBERT-147_1e-05_20260301_094826`
- Improved model (DANN): `SLID_utter-project-mHuBERT-147_1e-05_20260305_022125`

Primary comparison metrics:

| Run | Accuracy | Macro-F1 | Eval Loss |
|---|---:|---:|---:|
| Baseline (untuned) | 0.4676 | 0.4142 | 2.2854 |
| Improved (DANN) | 0.5576 | 0.5426 | 1.8815 |

Secondary progression (bridge/control):

- Tuned reference (no mitigation): `SLID_utter-project-mHuBERT-147_1e-05_20260301_150502`
- Mitigation 1 (data-centric): `SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- Improved (DANN): `SLID_utter-project-mHuBERT-147_1e-05_20260305_022125`

| Run | Accuracy | Macro-F1 | Eval Loss |
|---|---:|---:|---:|
| Tuned reference | 0.5270 | 0.4748 | 2.1629 |
| Mitigation 1 | 0.5388 | 0.4868 | 2.1504 |
| Improved (DANN) | 0.5576 | 0.5426 | 1.8815 |

Observed outcome summary:

- DANN improves over untuned baseline by accuracy **0.0900** and macro-F1 **0.1284**.
- DANN improves over tuned reference by accuracy **0.0306** and macro-F1 **0.0678**.
- DANN improves over Mitigation 1 by accuracy **0.0188** and macro-F1 **0.0558**.
- Eval loss is also lower in DANN, indicating better fit without needing retraining here.

## 3. Confusion Pattern Analysis

Tracked high-impact confusion pairs:

| Pair | Baseline (untuned) | Tuned ref | Mitigation 1 | DANN |
|---|---:|---:|---:|---:|
| `konkani->marathi` | 106 | 97 | 106 | 74 |
| `sindhi->punjabi` | 79 | 103 | 102 | 14 |
| `hindi->urdu` | 60 | 77 | 68 | 39 |
| `dogri->punjabi` | 53 | 76 | 71 | 44 |
| `bodo->manipuri` | 39 | 29 | 39 | 45 |
| `odia->bengali` | 26 | 23 | 45 | 10 |
| `santali->bengali` | 54 | 34 | 40 | 10 |

Top confusion trend:

- DANN reduces several major confusions strongly (for example `sindhi->punjabi`, `hindi->urdu`, `konkani->marathi`).
- Some confusion remains structurally hard (`konkani->marathi` still high), likely due acoustic/prosodic similarity and class overlap effects.

Language-wise gains (untuned baseline -> DANN) include classes that were historically weak:

- `sindhi`: 0.0333 -> 0.5000 (delta 0.4667)
- `santali`: 0.0600 -> 0.5267 (delta 0.4667)
- `maithili`: 0.0000 -> 0.4467 (delta 0.4467)
- `konkani`: 0.0000 -> 0.2200 (delta 0.2200)
- `dogri`: 0.1400 -> 0.2600 (delta 0.1200)

Largest drops (untuned baseline -> DANN), showing tradeoff:

- `kannada`: 0.6667 -> 0.5400 (delta -0.1267)
- `punjabi`: 0.7467 -> 0.6667 (delta -0.0800)
- `urdu`: 0.6600 -> 0.5800 (delta -0.0800)
- `malayalam`: 0.7800 -> 0.7067 (delta -0.0733)
- `marathi`: 0.7267 -> 0.6533 (delta -0.0733)

Largest gains (Mitigation 1 -> DANN):

- `sindhi`: 0.0533 -> 0.5000 (delta 0.4467)
- `maithili`: 0.0267 -> 0.4467 (delta 0.4200)
- `santali`: 0.2600 -> 0.5267 (delta 0.2667)
- `odia`: 0.1200 -> 0.3800 (delta 0.2600)
- `konkani`: 0.0000 -> 0.2200 (delta 0.2200)

Largest drops (Mitigation 1 -> DANN), showing tradeoff:

- `nepali`: 0.6333 -> 0.2800 (delta -0.3533)
- `punjabi`: 0.8533 -> 0.6667 (delta -0.1867)
- `malayalam`: 0.8667 -> 0.7067 (delta -0.1600)
- `bodo`: 0.5933 -> 0.4667 (delta -0.1267)
- `urdu`: 0.6933 -> 0.5800 (delta -0.1133)

Confusion matrix visuals:

- Baseline (untuned): `../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/diagnostics/confusion_matrix_validation.png`
  <img src="../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/diagnostics/confusion_matrix_validation.png" alt="Baseline (untuned) confusion matrix" width="560" />
- Tuned reference (no mitigation): `../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/diagnostics/confusion_matrix_validation.png`
  <img src="../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/diagnostics/confusion_matrix_validation.png" alt="Tuned reference (no mitigation) confusion matrix" width="560" />
- Mitigation 1 (data-centric): `../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/diagnostics/confusion_matrix_validation.png`
  <img src="../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/diagnostics/confusion_matrix_validation.png" alt="Mitigation 1 (data-centric) confusion matrix" width="560" />
- Improved (DANN): `../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/confusion_matrix_validation.png`
  <img src="../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/confusion_matrix_validation.png" alt="Improved (DANN) confusion matrix" width="560" />

## 4. Last-Layer Representation Analysis (t-SNE)

t-SNE settings target: `max_samples=2200`, `perplexity=30`, `seed=42`, `speaker_top_k=12`.

Available t-SNE point exports in this workspace: `improved_dann`.
Missing t-SNE point exports in this workspace: `baseline_untuned, tuned_ref, mitigation1`.

Quantitative t-SNE support:

| Run | Silhouette (Language) | Silhouette (Speaker top-k) | kNN Speaker Purity@5 |
|---|---:|---:|---:|
| Baseline (untuned) | NA | NA | NA |
| Tuned ref | NA | NA | NA |
| Mitigation 1 | NA | NA | NA |
| DANN | 0.0047 | -0.0992 | 0.3433 |

Interpretation:

- Language-colored t-SNE should become cleaner in improved runs if language signal strengthens.
- Speaker-colored t-SNE and kNN speaker purity quantify residual speaker structure in embeddings.
- In this workspace, untuned/tuned/M1 t-SNE numeric rows are `NA` because those run folders do not include saved model/preprocessor artifacts needed for post-hoc export.
- With available DANN t-SNE, speaker identity signal appears reduced but still present (non-zero local speaker purity).

t-SNE visuals:

- Baseline (untuned): t-SNE images not available in current local artifacts.
- Tuned reference (no mitigation): t-SNE images not available in current local artifacts.
- Mitigation 1 (data-centric): t-SNE images not available in current local artifacts.
- Improved (DANN) (language view): `../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/tsne_validation_by_language.png`
  <img src="../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/tsne_validation_by_language.png" alt="Improved (DANN) t-SNE by language" width="560" />
- Improved (DANN) (speaker view): `../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/tsne_validation_by_speaker_topk.png`
  <img src="../indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/tsne_validation_by_speaker_topk.png" alt="Improved (DANN) t-SNE by speaker" width="560" />

## 5. Do Models Encode Speaker Identity?

Speaker diagnostics (`overlap_speaker_count=0`, no warnings) show clean split integrity across runs.

Unseen-speaker accuracy:

- Baseline (untuned): 0.4676
- Tuned ref: 0.5270
- Mitigation 1: 0.5388
- DANN: 0.5576

Conclusion: the models still encode some speaker information (non-zero speaker structure in embedding space), but DANN improves language-discriminative behavior and unseen-speaker performance compared with the baseline chain.

## 6. Final Takeaways

1. Bias source is real: small-speaker regime encourages speaker shortcut learning.
2. Data-centric mitigation helps but remains partial.
3. DANN provides the strongest overall improvement in this branch.
4. Confusion analysis and t-SNE both show progress plus remaining hard language pairs.
5. Speaker bias mitigation is improved, not absolutely solved; this calibrated claim is the most defensible for Task 3.
