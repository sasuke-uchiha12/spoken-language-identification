# Mitigation 1 vs Mitigation 2 Comparison Report

## Runs Compared

- **Mitigation 1**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- **Mitigation 2**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_071843`

## Comparison Objective

Evaluate whether Mitigation 2 (gentler augmentation + macro-F1 checkpoint selection) improves on Mitigation 1.

## Final Metrics Comparison

| Metric | Mitigation 1 | Mitigation 2 | Delta (M2 - M1) |
|---|---:|---:|---:|
| Eval Accuracy | 0.5388 | 0.5279 | **-0.0109** |
| Eval Macro-F1 | 0.4868 | 0.4828 | **-0.0040** |
| Eval Loss | 2.1504 | 2.1385 | **-0.0119** |
| Predict Accuracy | 0.5388 | 0.5279 | -0.0109 |
| Predict Macro-F1 | 0.4868 | 0.4828 | -0.0040 |

Interpretation:

- Mitigation 2 achieved slightly lower accuracy and macro-F1.
- Mitigation 2 achieved slightly better loss.

## Best Checkpoint Behavior

From each run’s `trainer_state.json`:

- Mitigation 1 best checkpoint: step `5800`, epoch `~2.669`
- Mitigation 2 best checkpoint: step `5400`, epoch `~2.485`

Best metric by run:

- Mitigation 1 best macro-F1: `0.4868`
- Mitigation 2 best macro-F1: `0.4828`

Late-stage trend:

- Both runs degrade slightly after best checkpoint.

## Per-Language Delta (M2 - M1)

### Largest gains in Mitigation 2

- sindhi: `+0.3200` (`0.0533 -> 0.3733`)
- bengali: `+0.2400` (`0.4133 -> 0.6533`)
- nepali: `+0.1067` (`0.6333 -> 0.7400`)
- sanskrit: `+0.0267` (`0.7933 -> 0.8200`)
- manipuri: `+0.0200` (`0.8467 -> 0.8667`)

### Largest regressions in Mitigation 2

- punjabi: `-0.1533` (`0.8533 -> 0.7000`)
- malayalam: `-0.1533` (`0.8667 -> 0.7133`)
- gujarati: `-0.1400` (`0.7933 -> 0.6533`)
- bodo: `-0.1267` (`0.5933 -> 0.4667`)
- kannada: `-0.1200` (`0.6133 -> 0.4933`)

Interpretation:

- Mitigation 2 improved some previously weak classes strongly.
- It simultaneously reduced performance on several previously strong classes.

## Confusion Pattern Changes

### Mitigation 1 top confusions

- `konkani -> marathi` (106)
- `sindhi -> punjabi` (102)
- `dogri -> punjabi` (71)
- `hindi -> urdu` (68)
- `maithili -> gujarati` (49)

### Mitigation 2 top confusions

- `konkani -> marathi` (101)
- `odia -> bengali` (71)
- `santali -> bengali` (62)
- `hindi -> urdu` (62)
- `maithili -> bengali` (51)

Interpretation:

- Some pairs improved (`sindhi -> punjabi`, `hindi -> urdu`).
- New or stronger confusion patterns appeared (`odia -> bengali`, `maithili -> bengali`).

## Speaker Diagnostic Comparison

Both runs show:

- `overlap_speaker_count = 0`
- `warnings = []`

Unseen-speaker accuracy:

- Mitigation 1: `0.5388`
- Mitigation 2: `0.5279`

Interpretation:

- No evidence of speaker-overlap leakage in either run.
- Mitigation 1 remains stronger on unseen-speaker validation accuracy.

## Runtime / Efficiency

- Mitigation 1 train runtime: `23947.48s`
- Mitigation 2 train runtime: `22226.29s`
- Mitigation 2 is faster by about `1721.19s` (~28.7 min).

## Final Conclusion

- **Best overall model between the two is Mitigation 1** (higher accuracy and macro-F1).
- **Mitigation 2 is a tradeoff model**:
  - Better for some hard classes (`sindhi`, `bengali`)
  - Worse for several strong classes (`punjabi`, `malayalam`, `gujarati`, `kannada`)

Recommended usage:

- Use Mitigation 1 as final Task 2 model.
- Use Mitigation 2 as an analysis/ablation run showing class-level tradeoffs.

