# Mitigation Change Log

## Scope

This file tracks the mitigation-related configuration changes applied after the best tuned mHuBERT control run.

Control run used for comparison:
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260301_150502`

---

## Mitigation 1 (Implemented and Executed)

Run:
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`

Config changes (in `train_model_mac_m1.py`):
- `cfg.enable_train_augmentation = True`
- `cfg.augmentation_prob = 0.3`
- `cfg.speed_min = 0.97`
- `cfg.speed_max = 1.03`
- `cfg.noise_std_min = 0.0005`
- `cfg.noise_std_max = 0.002`

Important notes:
- Augmentation is train-only (validation remains clean).
- Other core tuned settings were kept unchanged for fair comparison.

Observed outcome vs control:
- `eval_accuracy`: `0.52697 -> 0.53879` (`+0.01182`)
- `eval_macro_f1`: `0.47481 -> 0.48676` (`+0.01195`)
- `eval_loss`: `2.16288 -> 2.15041` (improved)
- Some weak classes improved (for example `maithili`, `hindi`, `sindhi`), but some regressions remain (`odia`).

---

## Mitigation 2 (Planned Next Candidate, Not Yet Executed)

Goal:
- Keep mitigation gains while reducing regressions in some classes.

Proposed config changes (relative to Mitigation 1):
- `cfg.augmentation_prob = 0.25`
- `cfg.speed_min = 0.98`
- `cfg.speed_max = 1.02`
- `cfg.noise_std_max = 0.0015`

Optional checkpoint-selection improvement (base script):
- Consider `metric_for_best_model = "macro_f1"` for better class-balance checkpoint choice.

Status:
- Planned for next run; no results yet.

---

## Template for Future Iterations

### Mitigation 3
- Run ID:
- Config changes:
- Outcome vs previous:
- Keep / rollback decision:

