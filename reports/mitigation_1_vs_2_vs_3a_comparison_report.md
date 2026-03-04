# Mitigation 1 vs Mitigation 2 vs Mitigation 3A Comparison Report

## Runs Compared

- **Mitigation 1**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- **Mitigation 2**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_071843`
- **Mitigation 3A**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_170850`

## Comparison Objective

Evaluate whether Mitigation 3A improves over Mitigation 1 and Mitigation 2 on Task 2 goals.

## Final Metrics Comparison

| Metric | Mitigation 1 | Mitigation 2 | Mitigation 3A |
|---|---:|---:|---:|
| Eval Accuracy | **0.5388** | 0.5279 | 0.4979 |
| Eval Macro-F1 | **0.4868** | 0.4828 | 0.4561 |
| Eval Loss | 2.1504 | **2.1385** | 2.1884 |
| Predict Accuracy | **0.5388** | 0.5279 | 0.4979 |
| Predict Macro-F1 | **0.4868** | 0.4828 | 0.4561 |

Key deltas for Mitigation 3A:

- vs Mitigation 1:
  - accuracy: `-0.0409`
  - macro-F1: `-0.0306`
  - eval loss: `+0.0380` (worse)
- vs Mitigation 2:
  - accuracy: `-0.0300`
  - macro-F1: `-0.0267`
  - eval loss: `+0.0499` (worse)

Interpretation:

- Mitigation 3A does not improve global metrics; it is weaker than both prior mitigation runs.

## Best Checkpoint Behavior

From each run’s `trainer_state.json`:

- Mitigation 1 best checkpoint: `checkpoint-5800` (epoch `~2.6694`)
- Mitigation 2 best checkpoint: `checkpoint-5400` (epoch `~2.4852`)
- Mitigation 3A best checkpoint: `checkpoint-5800` (epoch `~2.6694`)

Late-stage behavior:

- All runs decline after best checkpoint.
- In 3A, last eval at step `6500` dropped to `acc=0.4839`, `macro_f1=0.4409`.

Interpretation:

- Plateau/late degradation remains unresolved in 3A.

## Per-Language Tradeoff (3A)

### 3A gains vs Mitigation 1 (largest)

- `maithili`: `+0.0800`
- `santali`: `+0.0733`
- `odia`: `+0.0667`
- `hindi`: `+0.0600`

### 3A losses vs Mitigation 1 (largest)

- `nepali`: `-0.4067`
- `bodo`: `-0.1933`
- `telugu`: `-0.1067`
- `malayalam`: `-0.0933`
- `tamil`: `-0.0867`

### 3A gains vs Mitigation 2 (largest)

- `punjabi`: `+0.1200`
- `kannada`: `+0.1133`
- `odia`: `+0.1133`
- `santali`: `+0.1133`

### 3A losses vs Mitigation 2 (largest)

- `nepali`: `-0.5133`
- `sindhi`: `-0.3400`
- `bengali`: `-0.2200`
- `telugu`: `-0.0800`

Interpretation:

- 3A helps selected weak classes but introduces larger regressions in multiple other classes.

## Confusion Pattern Comparison

Top off-diagonal confusions:

- Mitigation 1:
  - `konkani -> marathi` (106)
  - `sindhi -> punjabi` (102)
  - `dogri -> punjabi` (71)
  - `hindi -> urdu` (68)
- Mitigation 2:
  - `konkani -> marathi` (101)
  - `odia -> bengali` (71)
  - `hindi -> urdu` (62)
  - `santali -> bengali` (62)
- Mitigation 3A:
  - `sindhi -> punjabi` (98)
  - `konkani -> marathi` (90)
  - `bodo -> manipuri` (62)
  - `maithili -> gujarati` (59)

Interpretation:

- 3A reduces some earlier confusions (`konkani -> marathi`, `hindi -> urdu`) but keeps a severe `sindhi -> punjabi` confusion and worsens global balance.

## Speaker Diagnostic Comparison

All three runs show:

- `overlap_speaker_count = 0`
- `warnings = []`

Unseen-speaker accuracy:

- Mitigation 1: `0.5388`
- Mitigation 2: `0.5279`
- Mitigation 3A: `0.4979`

Interpretation:

- Speaker split integrity is clean in all runs.
- 3A has the weakest unseen-speaker accuracy among the three.

## Final Decision

For Task 2 final model selection:

- **Keep Mitigation 1 as final** (best overall accuracy + macro-F1 balance).
- Keep Mitigation 2 and 3A as ablation evidence demonstrating mitigation tradeoffs.

Task 2 reporting note:

- 3A still contributes useful evidence (some weak-class recovery), but it does not satisfy the accuracy guardrail relative to Mitigation 1 and should not replace it as final.
