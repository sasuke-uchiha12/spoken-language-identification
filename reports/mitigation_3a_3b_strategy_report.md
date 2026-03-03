# Mitigation 3A/3B Strategy Report

## 1) Context

Current reference runs:
- Mitigation 1: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260302_111613`
- Mitigation 2: `indic-SLID-mac/SLID_utter-project-mHuBERT-147_1e-05_20260303_071843`

Observed outcomes:
- Mitigation 1: `eval_accuracy=0.5388`, `eval_macro_f1=0.4868`, `eval_loss=2.1504`
- Mitigation 2: `eval_accuracy=0.5279`, `eval_macro_f1=0.4828`, `eval_loss=2.1385`
- Tradeoff: Mitigation 2 improved selected weak classes but regressed several strong classes, reducing overall accuracy and macro-F1.

## 2) Why 3A

Reasoning from evidence:
- Mitigation 1 is currently the strongest overall mitigation run.
- Mitigation 2 appears slightly over-constrained for global performance despite gains on specific hard classes.
- Mitigation 3A restores Mitigation 1 augmentation strength while keeping macro-F1 checkpointing and increasing eval/save cadence to better capture plateau-region peaks.

## 3) Mitigation 3A Configuration

File: `train_model_mac_m1.py`

Exact 3A settings:
- `cfg.enable_train_augmentation = True`
- `cfg.augmentation_prob = 0.30`
- `cfg.speed_min = 0.97`
- `cfg.speed_max = 1.03`
- `cfg.noise_std_min = 0.0005`
- `cfg.noise_std_max = 0.002`
- `cfg.eval_steps = 100`
- `cfg.save_steps = 100`
- `metric_for_best_model = "macro_f1"` (in `build_training_arguments_mps`)

Unchanged:
- model, split logic, seed behavior, Mac batch/accumulation profile, max duration, max grad norm, fp16 off on MPS.

## 4) Expected Advantage of 3A

- Preserve or recover stronger overall performance profile from Mitigation 1.
- Keep class-balance-aware checkpoint selection (`macro_f1`) instead of pure accuracy selection.
- Denser eval/save checkpoints improve chance of selecting the best point when curves plateau or oscillate.

## 5) 3B Fallback Plan

Trigger condition:
- Use 3B only if 3A does not meet decision criteria (below).

3B configuration:
- `augmentation_prob = 0.28`
- `speed_min = 0.975`
- `speed_max = 1.025`
- `noise_std_max = 0.0018`
- keep all other settings identical to 3A.

Why this fallback:
- Slightly reduces perturbation intensity while keeping mitigation active, to limit regressions on sensitive classes.

## 6) Decision Rule

Primary selection:
- Choose the run with highest `eval_macro_f1` among Mitigation 1, 2, 3A (and 3B if needed).

Accuracy guardrail:
- Prefer runs where `eval_accuracy` is not lower than Mitigation 1 by more than `~0.005`, unless weak-class gains are substantial and explicitly justified.

Additional evidence checks:
- speaker diagnostics remain clean (`overlap_speaker_count=0`, no warnings)
- confusion shifts should not create larger new failures than observed gains.
