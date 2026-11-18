# Mitigation 1 vs Mitigation 2 vs Mitigation 3A vs Mitigation 3B Comparison Report

## Runs Compared

- **Mitigation 1**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- **Mitigation 2**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_071843`
- **Mitigation 3A**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_170850`
- **Mitigation 3B**: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260304_061005`

## Comparison Objective

Evaluate whether Mitigation 3B improved over earlier mitigation runs and whether it can replace Mitigation 1 as final Task 2 model.

## Final Metrics Comparison

| Metric | Mitigation 1 | Mitigation 2 | Mitigation 3A | Mitigation 3B |
|---|---:|---:|---:|---:|
| Eval Accuracy | **0.5388** | 0.5279 | 0.4979 | 0.5276 |
| Eval Macro-F1 | **0.4868** | 0.4828 | 0.4561 | 0.4817 |
| Eval Loss | 2.1504 | **2.1385** | 2.1884 | 2.1456 |
| Predict Accuracy | **0.5388** | 0.5279 | 0.4979 | 0.5276 |
| Predict Macro-F1 | **0.4868** | 0.4828 | 0.4561 | 0.4817 |

### Key deltas for Mitigation 3B

- vs Mitigation 1:
  - accuracy: `-0.0112`
  - macro-F1: `-0.0051`
  - eval loss: `-0.0048` (slightly better)
- vs Mitigation 2:
  - accuracy: `-0.0003` (near-equal)
  - macro-F1: `-0.0011` (near-equal)
  - eval loss: `+0.0071` (slightly worse)
- vs Mitigation 3A:
  - accuracy: `+0.0297`
  - macro-F1: `+0.0256`
  - eval loss: `-0.0428` (better)

Interpretation:

- 3B clearly fixes most of 3A regression.
- 3B does not surpass Mitigation 1 globally.
- 3B is effectively tied with Mitigation 2.

## Best Checkpoint Behavior

From each run’s `trainer_state.json`:

- Mitigation 1 best checkpoint: `checkpoint-5800`
- Mitigation 2 best checkpoint: `checkpoint-5400`
- Mitigation 3A best checkpoint: `checkpoint-5800`
- Mitigation 3B best checkpoint: `checkpoint-5600`

For Mitigation 3B:

- Selected metric (`best_metric`): `0.4816888433` (macro-F1)
- Best checkpoint epoch: `~2.5773`
- Last eval at step `6500` is lower than best checkpoint.

Interpretation:

- All runs peak before training end.
- Early stopping/checkpoint selection remains critical.

## Per-Language Tradeoff (3B)

### 3B gains vs Mitigation 1 (largest)

- `nepali`: `+0.1800`
- `bengali`: `+0.0867`
- `sindhi`: `+0.0867`
- `kannada`: `+0.0733`
- `santali`: `+0.0200`

### 3B losses vs Mitigation 1 (largest)

- `malayalam`: `-0.1600`
- `sanskrit`: `-0.1267`
- `bodo`: `-0.1200`
- `kashmiri`: `-0.1000`
- `tamil`: `-0.0800`

### 3B gains vs Mitigation 2 (largest)

- `kannada`: `+0.1933`
- `punjabi`: `+0.1733`
- `gujarati`: `+0.1267`
- `marathi`: `+0.0800`
- `nepali`: `+0.0733`

### 3B losses vs Mitigation 2 (largest)

- `sindhi`: `-0.2333`
- `bengali`: `-0.1533`
- `sanskrit`: `-0.1533`
- `kashmiri`: `-0.0933`
- `tamil`: `-0.0400`

Interpretation:

- 3B improves specific classes substantially, but with offsetting losses in others.
- `konkani` stays at `0.0` in 3B (same unresolved issue as M1 and 3A).

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
- Mitigation 3B:
  - `konkani -> marathi` (125)
  - `sindhi -> punjabi` (87)
  - `hindi -> urdu` (73)
  - `dogri -> punjabi` (66)

Interpretation:

- 3B improves `sindhi -> punjabi` vs M1, but `konkani -> marathi` gets worse.
- Core confusion structure is still not fully solved in any run.

## Speaker Diagnostic Comparison

All runs show:

- `overlap_speaker_count = 0`
- `warnings = []`

Unseen-speaker accuracy:

- Mitigation 1: `0.5388`
- Mitigation 2: `0.5279`
- Mitigation 3A: `0.4979`
- Mitigation 3B: `0.5276`

Interpretation:

- Speaker split integrity is clean across all runs.
- 3B recovers unseen-speaker performance from 3A but does not beat M1.

## Final Decision

For Task 2 final model selection:

- **Keep Mitigation 1 as final** (best global accuracy + macro-F1).
- Use Mitigation 2 and 3B as supporting ablations (close but not better overall).
- Use 3A as negative ablation evidence (shows over-regularization tradeoff).
