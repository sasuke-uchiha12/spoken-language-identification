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

## Mitigation 2 (Implemented and Executed)

Run:
- `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_071843`

Config changes (relative to Mitigation 1):
- `cfg.augmentation_prob = 0.25`
- `cfg.speed_min = 0.98`
- `cfg.speed_max = 1.02`
- `cfg.noise_std_max = 0.0015`
- `metric_for_best_model = "macro_f1"`

Observed outcome vs Mitigation 1:
- `eval_accuracy`: `0.53879 -> 0.52788` (`-0.01091`)
- `eval_macro_f1`: `0.48676 -> 0.48284` (`-0.00392`)
- `eval_loss`: `2.15041 -> 2.13853` (improved)
- Tradeoff run: gained on some weak classes, regressed on several previously strong classes.

---

## Mitigation 3A (Implemented)

Status:
- Implemented in `train_model_mac_m1.py` (ready to run).

Config changes (relative to Mitigation 2):
- `cfg.eval_steps = 100` (from `200`)
- `cfg.save_steps = 100` (from `200`)
- `cfg.enable_train_augmentation = True`
- `cfg.augmentation_prob = 0.30`
- `cfg.speed_min = 0.97`
- `cfg.speed_max = 1.03`
- `cfg.noise_std_min = 0.0005`
- `cfg.noise_std_max = 0.002`
- keep `metric_for_best_model = "macro_f1"`

Goal:
- Recover stronger overall metrics seen in Mitigation 1 while keeping macro-F1-based checkpoint selection and denser eval/save cadence for better checkpoint capture.

---

## Mitigation 3B (Planned Fallback)

Trigger:
- Run only if Mitigation 3A fails macro-F1-first selection with accuracy guardrail.

Planned config (relative to 3A):
- `cfg.augmentation_prob = 0.28`
- `cfg.speed_min = 0.975`
- `cfg.speed_max = 1.025`
- `cfg.noise_std_max = 0.0018`
- keep all other settings identical to 3A.

Goal:
- Keep mitigation pressure while reducing over-perturbation risk on sensitive classes.
