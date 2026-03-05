# Mitigation 1 vs DANN Comparison Report

## Runs Compared

- Mitigation 1: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- DANN mitigation: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125`

## Important Config Note

DANN run is not identical to Mitigation 1:

- DANN enabled (`enable_dann=True`) with adversarial speaker head.
- Mitigation 1 used train-time augmentation; DANN run used `enable_train_augmentation=False` to isolate model-based effect.
- DANN run used `metric_for_best_model="macro_f1"` and `eval_steps/save_steps=100`.
- Mitigation 1 used accuracy-based checkpointing and slower eval/save cadence.

## Final Metrics

| Metric | Mitigation 1 | DANN | Delta (DANN - M1) |
|---|---:|---:|---:|
| Eval Accuracy | 0.5388 | **0.5576** | **+0.0188** |
| Eval Macro-F1 | 0.4868 | **0.5426** | **+0.0558** |
| Eval Loss | 2.1504 | **1.8815** | **-0.2689** |

Interpretation:

- DANN improves all three global indicators (accuracy, macro-F1, loss).
- Macro-F1 gain is substantial for this task setup.

## Per-Language Delta Highlights (DANN - M1)

### Largest gains

- `sindhi`: `0.0533 -> 0.5000` (`+0.4467`)
- `maithili`: `0.0267 -> 0.4467` (`+0.4200`)
- `santali`: `0.2600 -> 0.5267` (`+0.2667`)
- `odia`: `0.1200 -> 0.3800` (`+0.2600`)
- `konkani`: `0.0000 -> 0.2200` (`+0.2200`)

### Largest drops

- `nepali`: `0.6333 -> 0.2800` (`-0.3533`)
- `punjabi`: `0.8533 -> 0.6667` (`-0.1867`)
- `malayalam`: `0.8667 -> 0.7067` (`-0.1600`)
- `bodo`: `0.5933 -> 0.4667` (`-0.1267`)
- `urdu`: `0.6933 -> 0.5800` (`-0.1133`)

Interpretation:

- DANN strongly helps several historically weak classes.
- There is a tradeoff with a few previously strong classes.

## Confusion Shift (DANN vs M1)

### Major reductions

- `sindhi -> punjabi`: `102 -> 14` (`-88`)
- `odia -> bengali`: `45 -> 10` (`-35`)
- `konkani -> marathi`: `106 -> 74` (`-32`)
- `santali -> bengali`: `40 -> 10` (`-30`)
- `hindi -> urdu`: `68 -> 39` (`-29`)
- `dogri -> punjabi`: `71 -> 44` (`-27`)

### New or increased confusions

- `hindi -> maithili`: `0 -> 24`
- `marathi -> konkani`: `0 -> 16`
- `bengali -> odia`: `10 -> 26`
- `nepali -> manipuri`: `18 -> 33`
- `urdu -> hindi`: `4 -> 19`

Interpretation:

- DANN reduces several long-standing high-volume confusions.
- Some new confusion patterns appear and should be acknowledged in final discussion.

## Speaker Diagnostic Comparison

Both runs:

- `overlap_speaker_count = 0`
- `warnings = []`

Unseen-speaker accuracy:

- Mitigation 1: `0.5388`
- DANN: `0.5576`

Interpretation:

- Split integrity remains clean in both.
- DANN has better unseen-speaker behavior in this comparison.

## Verdict

Against Mitigation 1, DANN is better overall on this branch:

- Higher global performance.
- Better weak-class recovery on multiple classes.
- Better unseen-speaker accuracy.
- Remaining tradeoffs should be documented, but the net effect is positive.

