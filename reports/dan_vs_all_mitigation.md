# Overall Mitigation Comparison Report (Current Branch)

## Scope

This comparison uses the runs currently available under `indic-SLID-mac/` in branch `feat/task2-dann-mitigation`.

Included:

- Task1 tuned reference: `20260301_150502`
- Mitigation 1: `20260302_111613`
- Mitigation 2: `20260303_071843`
- Mitigation 3A: `20260303_170850`
- DANN mitigation: `20260305_022125`

Not present in this branch:

- Mitigation 3B run folder
- Mitigation 4 run folder (`20260304_160046`)

## Metrics Table

| Label | Run ID | Eval Acc | Eval Macro-F1 | Eval Loss |
|---|---|---:|---:|---:|
| Task1 tuned ref | `20260301_150502` | 0.5270 | 0.4748 | 2.1629 |
| Mitigation 1 | `20260302_111613` | 0.5388 | 0.4868 | 2.1504 |
| Mitigation 2 | `20260303_071843` | 0.5279 | 0.4828 | 2.1385 |
| Mitigation 3A | `20260303_170850` | 0.4979 | 0.4561 | 2.1884 |
| DANN mitigation | `20260305_022125` | **0.5576** | **0.5426** | **1.8815** |

## Ranking by Global Quality

1. DANN mitigation (`20260305_022125`)
2. Mitigation 1 (`20260302_111613`)
3. Mitigation 2 (`20260303_071843`)
4. Task1 tuned reference (`20260301_150502`)
5. Mitigation 3A (`20260303_170850`)

## Deltas vs Mitigation 1

| Label | Delta Acc | Delta Macro-F1 | Delta Loss |
|---|---:|---:|---:|
| Task1 tuned ref | -0.0118 | -0.0120 | +0.0125 |
| Mitigation 2 | -0.0109 | -0.0040 | -0.0119 |
| Mitigation 3A | -0.0409 | -0.0306 | +0.0380 |
| DANN mitigation | **+0.0188** | **+0.0558** | **-0.2689** |

## Confusion Pattern Comparison

Source: `diagnostics/confusion_matrix_validation.csv` for each run, using top off-diagonal pairs.

### Top confusion pairs by run (top 3)

- Task1 tuned ref: `sindhi->punjabi (103)`, `konkani->marathi (97)`, `hindi->urdu (77)`
- Mitigation 1: `konkani->marathi (106)`, `sindhi->punjabi (102)`, `dogri->punjabi (71)`
- Mitigation 2: `konkani->marathi (101)`, `odia->bengali (71)`, `santali->bengali (62)`
- Mitigation 3A: `sindhi->punjabi (98)`, `konkani->marathi (90)`, `bodo->manipuri (62)`
- DANN mitigation: `konkani->marathi (74)`, `bodo->manipuri (45)`, `dogri->punjabi (44)`

### Tracked confusion counts across runs

| Pair | Task1 tuned ref | Mitigation 1 | Mitigation 2 | Mitigation 3A | DANN |
|---|---:|---:|---:|---:|---:|
| `konkani->marathi` | 97 | 106 | 101 | 90 | **74** |
| `sindhi->punjabi` | 103 | 102 | 46 | 98 | **14** |
| `dogri->punjabi` | 76 | 71 | **26** | 56 | 44 |
| `hindi->urdu` | 77 | 68 | 62 | 47 | **39** |
| `bodo->manipuri` | **29** | 39 | 51 | 62 | 45 |
| `odia->bengali` | 23 | 45 | 71 | 49 | **10** |
| `santali->bengali` | 34 | 40 | 62 | 39 | **10** |

Interpretation:

- DANN gives the clearest net reduction on several critical confusion pairs, especially `sindhi->punjabi`, `hindi->urdu`, `odia->bengali`, and `santali->bengali`.
- `konkani->marathi` remains the single largest unresolved confusion even after DANN.
- Not every pair is best under DANN (`dogri->punjabi` was lowest in Mitigation 2), but the overall confusion profile is strongest with DANN.

## Checkpoint/Selection Behavior Notes

- Mitigation 1 best checkpoint at step `5800`; last eval is lower than best.
- DANN best checkpoint at step `6500`; last eval equals best.
- DANN run uses `metric_for_best_model = macro_f1`; Mitigation 1 used accuracy-based selection.

## Speaker Diagnostic Summary Across Runs

All listed runs show:

- `overlap_speaker_count = 0`
- `warnings = []`

Unseen-speaker accuracy highlights:

- Mitigation 1: `0.5388`
- DANN: `0.5576` (best among listed runs)

## Overall Conclusion

In this branch, DANN mitigation is the strongest run so far:

- Best top-line metrics.
- Best class-balance metric (macro-F1).
- Best unseen-speaker accuracy.

For final submission framing:

- Keep Mitigation 1 as strong data-centric baseline.
- Present DANN as the model-based upgrade that gives the best net result.
- Explicitly mention the remaining per-language tradeoffs/confusion shifts.
