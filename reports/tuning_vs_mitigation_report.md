# Tuning vs Mitigation Comparison Report

## Runs Compared

- **Tuned control (no augmentation)**:  
  `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502`

- **Mitigation run (augmentation ON)**:  
  `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`

## 1) Overall Metrics Comparison

| Metric | Tuned Control | Mitigation | Delta (Mit - Ctrl) |
|---|---:|---:|---:|
| Eval Accuracy | 0.52697 | 0.53879 | **+0.01182** |
| Eval Macro-F1 | 0.47481 | 0.48676 | **+0.01195** |
| Eval Loss | 2.16288 | 2.15041 | **-0.01247** |
| Predict Accuracy | 0.52697 | 0.53879 | +0.01182 |
| Predict Macro-F1 | 0.47481 | 0.48676 | +0.01195 |

Conclusion:

- Mitigation outperformed tuned control on all key final metrics.

## 2) Best Checkpoint Comparison

From each run’s `trainer_state.json`:

- Control best accuracy at step `6000` (epoch `~2.761`): `0.52697`
- Mitigation best accuracy at step `5800` (epoch `~2.669`): `0.53879`

- Control best macro-F1 at step `6400`: `0.47574`
- Mitigation best macro-F1 at step `5800`: `0.48676`

Observation:

- Mitigation reached better peak performance earlier.
- Final step is not always best; best-checkpoint loading remains important.

## 3) Per-Language Delta Analysis

### Largest gains (Mitigation - Control)

- bengali: `+0.1267` (`0.2867 -> 0.4133`)
- marathi: `+0.1133` (`0.6467 -> 0.7600`)
- malayalam: `+0.0800` (`0.7867 -> 0.8667`)
- bodo: `+0.0667` (`0.5267 -> 0.5933`)
- tamil: `+0.0600` (`0.8000 -> 0.8600`)
- hindi: `+0.0467` (`0.0333 -> 0.0800`)
- santali: `+0.0400` (`0.2200 -> 0.2600`)
- manipuri: `+0.0333` (`0.8133 -> 0.8467`)
- maithili: `+0.0267` (`0.0000 -> 0.0267`)

### Largest regressions

- odia: `-0.1867` (`0.3067 -> 0.1200`)
- nepali: `-0.0600` (`0.6933 -> 0.6333`)
- dogri: `-0.0400` (`0.1933 -> 0.1533`)
- kannada: `-0.0333` (`0.6467 -> 0.6133`)

### Near-zero class counts

- Accuracy exactly `0.0`: control `2` -> mitigation `1`
- Accuracy `<= 0.05`: control `4` -> mitigation `2`

Interpretation:

- Mitigation improved class coverage and reduced collapsed classes.
- Tradeoff exists for specific classes (especially `odia`).

## 4) Confusion Pattern Shifts

### Control top confusions

- `sindhi -> punjabi` (103)
- `konkani -> marathi` (97)
- `hindi -> urdu` (77)
- `dogri -> punjabi` (76)
- `bengali -> odia` (39)

### Mitigation top confusions

- `konkani -> marathi` (106)
- `sindhi -> punjabi` (102)
- `dogri -> punjabi` (71)
- `hindi -> urdu` (68)
- `maithili -> gujarati` (49)
- `odia -> bengali` (45)

### Pair-level changes

Improved:

- `hindi -> urdu`: `77 -> 68`
- `dogri -> punjabi`: `76 -> 71`
- `sindhi -> punjabi`: `103 -> 102` (small)

Worsened:

- `konkani -> marathi`: `97 -> 106`
- `odia -> bengali`: `23 -> 45`
- `maithili -> gujarati`: `37 -> 49`

Interpretation:

- Mitigation changed the error distribution and improved some major shortcuts, but also shifted pressure to other confusion pairs.

## 5) Speaker Diagnostic Comparison

Both runs (`speaker_diagnostic_summary.json`):

- `overlap_speaker_count = 0`
- `overlap_ratio_validation_speakers = 0.0`
- `warnings = []`

Unseen-speaker validation accuracy:

- Control: `0.52697`
- Mitigation: `0.53879`

Interpretation:

- No leakage signal in either run.
- Mitigation still improved unseen-speaker validation performance.

## 6) Runtime / Cost Tradeoff

From `all_results.json`:

- Control train runtime: `19822.39s`
- Mitigation train runtime: `23947.48s`
- Extra cost: `+4125.09s` (about +20.8%)

Interpretation:

- Mitigation is slower, but delivered measurable metric gains.

## 7) Final Decision

Given the current evidence, mitigation run should be considered the better model for Task 2:

- Higher overall accuracy
- Higher macro-F1
- Lower eval loss
- Better coverage for weak classes
- No speaker-overlap leakage warnings

## 8) Recommended Next Tuning (Optional)

If you want one additional run to reduce the `odia` regression while preserving mitigation gains:

- `augmentation_prob = 0.25`
- `speed_min = 0.98`, `speed_max = 1.02`
- `noise_std_max = 0.0015`

Keep all other settings unchanged.

