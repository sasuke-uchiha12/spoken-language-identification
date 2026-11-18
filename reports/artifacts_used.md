# Task 3 Artifacts Used

## Source Artifacts

- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/diagnostics/confusion_matrix_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/diagnostics/confusion_matrix_validation.png`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/diagnostics/per_language_accuracy_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/diagnostics/speaker_diagnostic_summary.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/eval_results.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_094826/trainer_state.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/diagnostics/confusion_matrix_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/diagnostics/confusion_matrix_validation.png`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/diagnostics/per_language_accuracy_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/diagnostics/speaker_diagnostic_summary.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/eval_results.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502/trainer_state.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/diagnostics/confusion_matrix_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/diagnostics/confusion_matrix_validation.png`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/diagnostics/per_language_accuracy_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/diagnostics/speaker_diagnostic_summary.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/eval_results.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613/trainer_state.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/confusion_matrix_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/confusion_matrix_validation.png`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/per_language_accuracy_validation.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/speaker_diagnostic_summary.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/tsne_validation_by_language.png`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/tsne_validation_by_speaker_topk.png`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/diagnostics/tsne_validation_points.csv`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/eval_results.json`
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260305_022125/trainer_state.json`

## Generated Artifacts

- `reports/task3_analysis.md`
- `reports/artifacts_used.md`
- `reports/tables/task3_metrics_comparison.csv`
- `reports/tables/task3_confusion_top_pairs.csv`
- `reports/tables/task3_per_language_delta.csv`
- `reports/tables/task3_speaker_summary.csv`
- `reports/tables/task3_tsne_cluster_metrics.csv`
- `reports/tables/task3_tsne_pairwise_distance_summary.csv`
- `reports/tables/speaker_summary_baseline.json`
- `reports/tables/speaker_summary_tuned_reference.json`
- `reports/tables/speaker_summary_mitigation1.json`
- `reports/tables/speaker_summary_comparison.csv`

## Run Mapping

- Baseline (untuned): `SLID_utter-project-mHuBERT-147_1e-05_20260301_094826`
- Tuned reference (no mitigation): `SLID_utter-project-mHuBERT-147_1e-05_20260301_150502`
- Mitigation 1: `SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- Improved (DANN): `SLID_utter-project-mHuBERT-147_1e-05_20260305_022125`

## Notes

- Baseline (untuned), tuned reference, and Mitigation 1 run folders in this workspace contain metrics/diagnostics but not full model/preprocessor checkpoint files.
- Therefore, post-hoc t-SNE re-export for those runs is not reproducible from current local artifacts.
