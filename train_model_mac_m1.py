from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import TrainingArguments

import train_model as base


def print_device_info_mps() -> None:
    has_mps_backend = hasattr(torch.backends, "mps")
    mps_is_built = bool(has_mps_backend and torch.backends.mps.is_built())
    mps_is_available = bool(has_mps_backend and torch.backends.mps.is_available())

    print("Device check (Mac profile):")
    print(f"torch.backends.mps.is_built(): {mps_is_built}")
    print(f"torch.backends.mps.is_available(): {mps_is_available}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if mps_is_available:
        print("Using Apple Silicon GPU via MPS through Hugging Face Trainer.")
    else:
        print("MPS is unavailable. Training will run on CPU.")


def build_training_arguments_mps(cfg: base.TrainConfig, output_dir: Path, report_to: str) -> TrainingArguments:
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "report_to": report_to,
        "logging_steps": cfg.logging_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "eval_steps": cfg.eval_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "learning_rate": cfg.learning_rate,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": cfg.save_total_limit,
        "fp16": False,  # fp16 is CUDA-specific for this workflow.
        "push_to_hub": False,
        "run_name": cfg.run_name,
    }

    if "group_by_length" in ta_sig:
        kwargs["group_by_length"] = cfg.group_by_length
    elif cfg.group_by_length:
        print("Warning: group_by_length is not supported in this transformers version. Ignoring it.")

    if "seed" in ta_sig:
        kwargs["seed"] = cfg.seed
    if "data_seed" in ta_sig:
        kwargs["data_seed"] = cfg.seed
    if cfg.group_by_length and "length_column_name" in ta_sig:
        kwargs["length_column_name"] = "length"
    if "label_names" in ta_sig:
        kwargs["label_names"] = ["labels"]

    if "evaluation_strategy" in ta_sig:
        kwargs["evaluation_strategy"] = "steps"
    else:
        kwargs["eval_strategy"] = "steps"

    # Mac-friendly defaults.
    if "use_mps_device" in ta_sig:
        kwargs["use_mps_device"] = mps_available
    if "dataloader_pin_memory" in ta_sig:
        kwargs["dataloader_pin_memory"] = False
    if "no_cuda" in ta_sig and mps_available:
        kwargs["no_cuda"] = False

    return TrainingArguments(**kwargs)


def configure_mac_profile() -> None:
    # Let unsupported ops fall back to CPU instead of hard-failing on MPS.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    cfg = base.CFG
    cfg.output_root = "./indic-SLID-mac"
    cfg.report_to_wandb = False
    cfg.hf_login_from_env = False

    # More stable defaults for unified memory on MacBook M1 Pro.
    cfg.map_batch_size = 16
    cfg.per_device_train_batch_size = 1
    cfg.per_device_eval_batch_size = 1
    cfg.gradient_accumulation_steps = 4
    cfg.max_duration_sec = 7
    cfg.max_grad_norm = 1.0
    cfg.group_by_length = True
    cfg.fp16 = False
    cfg.logging_steps = 20
    cfg.eval_steps = 100
    cfg.save_steps = 100

    # Task 2 combined run: DANN + Mitigation 1 train-only augmentation.
    cfg.enable_train_augmentation = True
    cfg.augmentation_prob = 0.30
    cfg.speed_min = 0.97
    cfg.speed_max = 1.03
    cfg.noise_std_min = 0.0005
    cfg.noise_std_max = 0.002

    # Optional Task 2 model-based mitigation (DANN).
    cfg.enable_dann = True
    cfg.dann_speaker_loss_weight = 0.1
    cfg.dann_grl_lambda = 1.0
    cfg.dann_use_lambda_schedule = True
    cfg.dann_speaker_head_dropout = 0.1
    cfg.run_tsne = True

    base.print_device_info = print_device_info_mps
    base.build_training_arguments = build_training_arguments_mps


def main() -> None:
    configure_mac_profile()
    base.main()


if __name__ == "__main__":
    main()
