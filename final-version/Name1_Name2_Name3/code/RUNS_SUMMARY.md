# Runs Summary

This file maps the key reported runs to their run IDs and bundled lightweight evidence files.

Included evidence per run: `eval_results.json`, `trainer_state.json`, `speaker_diagnostic_summary.json`.

| Run | Run ID | Eval Accuracy | Eval Macro-F1 | Eval Loss | Evidence Path |
|---|---|---:|---:|---:|---|
| Task1 Untuned Baseline | `SLID_utter-project-mHuBERT-147_1e-05_20260301_094826` | 0.4676 | 0.4142 | 2.2854 | `run_artifacts/untuned_baseline` |
| Task1 Tuned Reference | `SLID_utter-project-mHuBERT-147_1e-05_20260301_150502` | 0.5270 | 0.4748 | 2.1629 | `run_artifacts/tuned_reference` |
| Task2 Mitigation 1 | `SLID_utter-project-mHuBERT-147_1e-05_20260302_111613` | 0.5388 | 0.4868 | 2.1504 | `run_artifacts/mitigation_1` |
| Task2 DANN Final | `SLID_utter-project-mHuBERT-147_1e-05_20260305_022125` | 0.5576 | 0.5426 | 1.8815 | `run_artifacts/dann_final` |

## Notes
- These run IDs are the primary references used in the report for Task 1 and Task 2 comparisons.
- Full training artifacts/checkpoints are intentionally excluded from submission packaging to keep size manageable.
