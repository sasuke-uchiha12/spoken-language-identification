# Mitigation 4 Comparison Report (All Data-Centric + t-SNE vs Previous Runs)

## Runs Compared (Available in This Branch)

- `SLID_utter-project-mHuBERT-147_1e-05_20260301_094826`
- `SLID_utter-project-mHuBERT-147_1e-05_20260301_150502`
- `SLID_utter-project-mHuBERT-147_1e-05_20260304_160046` (Mitigation 4 target run)

## Final Metrics Comparison

| Metric | 20260301_094826 | 20260301_150502 | 20260304_160046 (Run 4) |
|---|---:|---:|---:|
| Eval Accuracy | 0.4676 | **0.5270** | 0.5115 |
| Eval Macro-F1 | 0.4142 | **0.4748** | 0.4664 |
| Eval Loss | 2.2854 | **2.1629** | 2.2127 |

## Key Deltas for Run 4

- vs `20260301_094826`:
  - accuracy: `+0.0439`
  - macro-F1: `+0.0523`
  - loss: `-0.0727` (better)
- vs `20260301_150502`:
  - accuracy: `-0.0155`
  - macro-F1: `-0.0084`
  - loss: `+0.0499` (worse)

Interpretation:

- Run 4 clearly improves over the older baseline-like run.
- Run 4 does not beat the strongest prior tuned run in this branch.

## Checkpoint Dynamics

From `trainer_state.json`:

- Run 4 best checkpoint: `checkpoint-6200` (`best_metric=0.5115`)
- Last eval (`step 6400`) is slightly lower than best checkpoint.

Interpretation:

- Late-stage plateau/oscillation remains.
- `load_best_model_at_end=True` is still important.

## Per-Language Tradeoff (Run 4 vs 20260301_150502)

### Largest gains

- `santali`: `+0.0867`
- `bengali`: `+0.0667`
- `assamese`: `+0.0533`
- `maithili`: `+0.0400`
- `marathi`: `+0.0333`

### Largest drops

- `bodo`: `-0.1267`
- `malayalam`: `-0.1133`
- `punjabi`: `-0.0867`
- `kashmiri`: `-0.0733`
- `kannada`: `-0.0733`

Interpretation:

- Run 4 improved some weaker classes but regressed several stronger classes.
- Net effect is lower global accuracy/F1 than the best tuned reference.

## Confusion Pattern (Run 4)

Top off-diagonal confusions:

- `konkani -> marathi` (103)
- `sindhi -> punjabi` (88)
- `bodo -> manipuri` (56)
- `dogri -> punjabi` (54)
- `hindi -> urdu` (49)

Interpretation:

- Core high-frequency confusion pairs remain unresolved.
- This continues to cap top-line performance.

## Speaker Bias Status

From Run 4 `speaker_diagnostic_summary.json`:

- `overlap_speaker_count = 0`
- `warnings = []`
- `accuracy_unseen_speakers = 0.5115`

Interpretation:

- Split-level speaker leakage is clean.
- Bias mitigation evidence is positive at the split diagnostic level.
- Full “bias removed” claim should still be avoided; representational/task-level confusions remain.

## t-SNE Availability

Run 4 includes required t-SNE outputs:

- `tsne_validation_points.csv`
- `tsne_validation_by_language.png`
- `tsne_validation_by_speaker_topk.png`

So this run is suitable for Task 3 representation analysis.

## Final Decision (Within This Branch’s Available Runs)

1. `20260301_150502` remains best on global metrics.
2. `20260304_160046` is second-best and strongest in terms of artifact completeness (includes t-SNE).
3. `20260301_094826` is weakest.
