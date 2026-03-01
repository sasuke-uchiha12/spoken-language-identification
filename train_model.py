from __future__ import annotations

import csv
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Audio, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import wandb
except ImportError:  # Optional if W&B logging is disabled.
    wandb = None

try:
    from huggingface_hub import login as hf_login
except ImportError:  # Optional if HF login is not needed.
    hf_login = None


# =========================
# Top-level configuration
# =========================
@dataclass
class TrainConfig:
    model_id: str = "facebook/wav2vec2-xls-r-300m"
    dataset_name: str = "badrex/nnti-dataset-full"
    sample_rate: int = 16000
    max_duration_sec: int = 7
    seed: int = 42

    map_batch_size: int = 32
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    group_by_length: bool = False
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    fp16: bool = True

    report_to_wandb: bool = True
    hf_login_from_env: bool = True
    wandb_project_default: str = "Indic-SLID"

    output_root: str = "./indic-SLID"
    run_name: Optional[str] = None

    enable_train_augmentation: bool = False
    augmentation_prob: float = 0.5
    speed_min: float = 0.95
    speed_max: float = 1.05
    noise_std_min: float = 0.0005
    noise_std_max: float = 0.003

    save_confusion_matrix_csv: bool = True
    save_confusion_matrix_png: bool = True
    save_per_language_csv: bool = True
    run_speaker_diagnostic: bool = True
    speaker_overlap_warn_ratio: float = 0.2
    speaker_seen_gap_warn: float = 0.10

    do_apply_dropout: bool = False
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    feat_proj_dropout: float = 0.1

    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


CFG = TrainConfig()


def _default_run_name(cfg: TrainConfig) -> str:
    model_slug = cfg.model_id.replace("/", "-")
    return f"SLID_{model_slug}_{cfg.learning_rate}_{cfg.timestamp}"


def slugify_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def set_all_seeds(seed: int) -> None:
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_device_info() -> None:
    print("Check if GPU available:")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name(0)}")
    else:
        print("torch.cuda.get_device_name(): N/A (CPU-only)")


def maybe_login_hf(cfg: TrainConfig) -> None:
    if not cfg.hf_login_from_env:
        return

    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set. Skipping Hugging Face login.")
        return
    if hf_login is None:
        print("huggingface_hub not installed. Skipping Hugging Face login.")
        return

    hf_login(token=token)
    print("Logged in to Hugging Face using HF_TOKEN.")


def maybe_setup_wandb(cfg: TrainConfig) -> bool:
    if not cfg.report_to_wandb:
        print("W&B logging disabled by config.")
        return False

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY not set. Disabling W&B logging.")
        return False
    if wandb is None:
        print("wandb not installed. Disabling W&B logging.")
        return False

    wandb_mode = os.getenv("WANDB_MODE")
    if wandb_mode:
        os.environ["WANDB_MODE"] = wandb_mode

    wandb_project = os.getenv("WANDB_PROJECT", cfg.wandb_project_default)
    wandb.login(key=api_key)
    wandb.init(project=wandb_project, name=cfg.run_name, config=asdict(cfg))
    print(f"W&B initialized. project={wandb_project}, run_name={cfg.run_name}")
    return True


def infer_input_features_key(model_id: str) -> str:
    if model_id == "facebook/w2v-bert-2.0":
        return "input_features"
    return "input_values"


class AudioDataCollator:
    def __init__(self, feature_extractor, input_features_key: str):
        self.feature_extractor = feature_extractor
        self.input_features_key = input_features_key

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            self.input_features_key: [f[self.input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }
        batch = self.feature_extractor.pad(batch, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch


class WaveformAugmenter:
    def __init__(self, cfg: TrainConfig, sample_rate: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.torchaudio = None

        if cfg.enable_train_augmentation:
            try:
                import torchaudio
            except ImportError as exc:
                raise ImportError(
                    "torchaudio is required when enable_train_augmentation=True"
                ) from exc
            self.torchaudio = torchaudio

    def maybe_augment(self, waveform: np.ndarray, sample_index: int) -> np.ndarray:
        if not self.cfg.enable_train_augmentation:
            return waveform

        rng = np.random.default_rng(self.cfg.seed + int(sample_index))
        if rng.random() > self.cfg.augmentation_prob:
            return waveform

        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim != 1:
            x = np.squeeze(x)

        speed = float(rng.uniform(self.cfg.speed_min, self.cfg.speed_max))
        if abs(speed - 1.0) > 1e-6:
            new_freq = max(1, int(round(self.sample_rate * speed)))
            wave_t = torch.from_numpy(x).unsqueeze(0)
            with torch.no_grad():
                x = (
                    self.torchaudio.functional.resample(
                        wave_t, orig_freq=self.sample_rate, new_freq=new_freq
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        noise_std = float(rng.uniform(self.cfg.noise_std_min, self.cfg.noise_std_max))
        if noise_std > 0.0:
            noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
            x = np.clip(x + noise, -1.0, 1.0)

        return x


def make_preprocess_function(
    *,
    feature_extractor,
    str_to_int: Dict[str, int],
    input_features_key: str,
    max_duration_sec: int,
    augmenter: Optional[WaveformAugmenter] = None,
    apply_augmentation: bool = False,
):
    def preprocess_function(examples, indices=None):
        audio_arrays: List[np.ndarray] = []
        example_indices = indices if indices is not None else [0] * len(examples["audio_filepath"])

        for audio_obj, sample_index in zip(examples["audio_filepath"], example_indices):
            audio_np = np.asarray(audio_obj["array"], dtype=np.float32)
            if apply_augmentation and augmenter is not None:
                audio_np = augmenter.maybe_augment(audio_np, sample_index)
            audio_arrays.append(audio_np)

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            truncation=True,
            max_length=int(feature_extractor.sampling_rate * max_duration_sec),
            return_attention_mask=True,
        )

        inputs["label"] = [str_to_int[x] for x in examples["language"]]

        inputs[input_features_key] = [np.asarray(x, dtype=np.float32) for x in inputs[input_features_key]]
        inputs["length"] = [len(f) for f in inputs[input_features_key]]

        return inputs

    return preprocess_function


def make_compute_metrics(int_to_str: Dict[int, str]):
    label_ids_sorted = sorted(int_to_str.keys())

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        preds = np.argmax(logits, axis=1)
        refs = eval_pred.label_ids

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(refs, preds)),
            "macro_f1": float(f1_score(refs, preds, average="macro", zero_division=0)),
        }

        for label_id in label_ids_sorted:
            mask = refs == label_id
            if np.any(mask):
                label_name = int_to_str[label_id]
                metrics[f"acc_lang_{slugify_label(label_name)}"] = float(np.mean(preds[mask] == refs[mask]))

        return metrics

    return compute_metrics


def write_confusion_matrix_csv(path: Path, cm: np.ndarray, label_names: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + label_names)
        for label, row in zip(label_names, cm.tolist()):
            writer.writerow([label] + row)


def plot_confusion_matrix(path: Path, cm: np.ndarray, label_names: List[str], title: str) -> None:
    import matplotlib.pyplot as plt

    fig_w = max(10, int(len(label_names) * 0.55))
    fig_h = max(8, int(len(label_names) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90)
    ax.set_yticklabels(label_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_per_language_rows(
    preds: np.ndarray,
    refs: np.ndarray,
    int_to_str: Dict[int, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label_id in sorted(int_to_str):
        label_name = int_to_str[label_id]
        mask = refs == label_id
        n_samples = int(np.sum(mask))
        n_correct = int(np.sum(preds[mask] == refs[mask])) if n_samples > 0 else 0
        accuracy = float(n_correct / n_samples) if n_samples > 0 else 0.0
        rows.append(
            {
                "label_id": label_id,
                "language": label_name,
                "n_samples": n_samples,
                "n_correct": n_correct,
                "accuracy": accuracy,
            }
        )
    return rows


def write_rows_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_speaker_diagnostic(
    *,
    valid_ds_encoded,
    preds: np.ndarray,
    refs: np.ndarray,
    train_speaker_ids: set,
    cfg: TrainConfig,
    output_dir: Path,
) -> None:
    valid_speakers = valid_ds_encoded["speaker_id"]
    valid_languages = valid_ds_encoded["language"]

    agg: Dict[str, Dict[str, Any]] = {}
    for speaker_id, language, pred, ref in zip(valid_speakers, valid_languages, preds.tolist(), refs.tolist()):
        speaker_key = str(speaker_id)
        if speaker_key not in agg:
            agg[speaker_key] = {
                "n_samples": 0,
                "n_correct": 0,
                "languages": set(),
                "is_seen_in_train": speaker_key in train_speaker_ids,
            }
        agg[speaker_key]["n_samples"] += 1
        agg[speaker_key]["n_correct"] += int(pred == ref)
        agg[speaker_key]["languages"].add(str(language))

    rows: List[Dict[str, Any]] = []
    for speaker_id, info in agg.items():
        n_samples = int(info["n_samples"])
        n_correct = int(info["n_correct"])
        rows.append(
            {
                "speaker_id": speaker_id,
                "n_samples": n_samples,
                "n_correct": n_correct,
                "accuracy": float(n_correct / n_samples) if n_samples else 0.0,
                "language_count": len(info["languages"]),
                "is_seen_in_train": bool(info["is_seen_in_train"]),
            }
        )

    rows.sort(key=lambda r: (-r["n_samples"], r["speaker_id"]))
    speaker_csv_path = output_dir / "speaker_accuracy_validation.csv"
    write_rows_csv(
        speaker_csv_path,
        rows,
        ["speaker_id", "n_samples", "n_correct", "accuracy", "language_count", "is_seen_in_train"],
    )

    valid_speaker_set = set(map(str, valid_speakers))
    overlap_speakers = sorted(train_speaker_ids & valid_speaker_set)
    overlap_count = len(overlap_speakers)
    valid_count = max(1, len(valid_speaker_set))
    overlap_ratio = overlap_count / valid_count

    seen_mask = np.array([str(s) in train_speaker_ids for s in valid_speakers], dtype=bool)
    unseen_mask = ~seen_mask

    seen_accuracy = float(np.mean(preds[seen_mask] == refs[seen_mask])) if np.any(seen_mask) else None
    unseen_accuracy = float(np.mean(preds[unseen_mask] == refs[unseen_mask])) if np.any(unseen_mask) else None
    seen_unseen_gap = None
    if seen_accuracy is not None and unseen_accuracy is not None:
        seen_unseen_gap = seen_accuracy - unseen_accuracy

    warnings_list: List[str] = []
    if overlap_count > 0:
        warnings_list.append(
            "Validation speakers overlap with training speakers; speaker leakage may inflate validation accuracy."
        )
    if overlap_ratio > cfg.speaker_overlap_warn_ratio:
        warnings_list.append(
            f"High speaker overlap ratio detected ({overlap_ratio:.3f} > {cfg.speaker_overlap_warn_ratio:.3f})."
        )
    if seen_unseen_gap is not None and seen_unseen_gap >= cfg.speaker_seen_gap_warn:
        warnings_list.append(
            "Accuracy on validation samples from speakers seen in training is substantially higher than unseen speakers."
        )

    summary = {
        "train_unique_speakers": len(train_speaker_ids),
        "validation_unique_speakers": len(valid_speaker_set),
        "overlap_speaker_count": overlap_count,
        "overlap_ratio_validation_speakers": overlap_ratio,
        "validation_samples_seen_speakers": int(np.sum(seen_mask)),
        "validation_samples_unseen_speakers": int(np.sum(unseen_mask)),
        "accuracy_seen_speakers": seen_accuracy,
        "accuracy_unseen_speakers": unseen_accuracy,
        "accuracy_gap_seen_minus_unseen": seen_unseen_gap,
        "warnings": warnings_list,
    }

    summary_path = output_dir / "speaker_diagnostic_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved speaker diagnostic CSV: {speaker_csv_path}")
    print(f"Saved speaker diagnostic summary: {summary_path}")
    if warnings_list:
        for msg in warnings_list:
            print(f"[Speaker Diagnostic Warning] {msg}")
    else:
        print("Speaker diagnostic: no leakage warning triggered by current heuristics.")


def build_training_arguments(cfg: TrainConfig, output_dir: Path, report_to: str) -> TrainingArguments:
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters

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
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "save_total_limit": cfg.save_total_limit,
        "fp16": bool(cfg.fp16 and torch.cuda.is_available()),
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

    if "evaluation_strategy" in ta_sig:
        kwargs["evaluation_strategy"] = "steps"
    else:
        kwargs["eval_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def build_trainer(
    *,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    feature_extractor,
    data_collator,
    compute_metrics,
):
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    if "processing_class" in trainer_sig:
        kwargs["processing_class"] = feature_extractor
    elif "tokenizer" in trainer_sig:
        kwargs["tokenizer"] = feature_extractor

    return Trainer(**kwargs)


def main() -> None:
    cfg = CFG
    cfg.run_name = cfg.run_name or _default_run_name(cfg)
    current_time_str = cfg.timestamp
    print(f"Current time: {current_time_str}")
    print(f"Run name: {cfg.run_name}")

    set_all_seeds(cfg.seed)
    print_device_info()

    maybe_login_hf(cfg)
    wandb_enabled = maybe_setup_wandb(cfg)
    report_to = "wandb" if wandb_enabled else "none"

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        cfg.model_id,
        do_normalize=True,
        return_attention_mask=True,
    )

    dataset = load_dataset(cfg.dataset_name)
    print(f"dataset['train']: {dataset['train']}")
    print(f"dataset['train'][0]: {dataset['train'][0]}")

    train_ds = dataset["train"].shuffle(seed=cfg.seed)
    valid_ds = dataset["validation"].shuffle(seed=cfg.seed)

    train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=cfg.sample_rate))
    valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=cfg.sample_rate))

    input_features_key = infer_input_features_key(cfg.model_id)

    label_list = sorted(train_ds.unique("language"))
    print(f"Languages ({len(label_list)}): {[l.upper() for l in label_list]}")
    str_to_int = {label: idx for idx, label in enumerate(label_list)}
    int_to_str = {idx: label for label, idx in str_to_int.items()}
    num_labels = len(label_list)

    augmenter: Optional[WaveformAugmenter] = None
    apply_train_aug = False
    if cfg.enable_train_augmentation:
        if input_features_key != "input_values":
            print(
                f"Train augmentation requested, but input key is '{input_features_key}'. "
                "Augmentation will be skipped (supported for raw input_values only)."
            )
        else:
            augmenter = WaveformAugmenter(cfg, sample_rate=cfg.sample_rate)
            apply_train_aug = True
            print(
                "Train augmentation enabled: speed perturbation + Gaussian noise "
                f"(p={cfg.augmentation_prob}, speed=[{cfg.speed_min}, {cfg.speed_max}], "
                f"noise_std=[{cfg.noise_std_min}, {cfg.noise_std_max}])."
            )

    train_preprocess = make_preprocess_function(
        feature_extractor=feature_extractor,
        str_to_int=str_to_int,
        input_features_key=input_features_key,
        max_duration_sec=cfg.max_duration_sec,
        augmenter=augmenter,
        apply_augmentation=apply_train_aug,
    )
    valid_preprocess = make_preprocess_function(
        feature_extractor=feature_extractor,
        str_to_int=str_to_int,
        input_features_key=input_features_key,
        max_duration_sec=cfg.max_duration_sec,
        augmenter=None,
        apply_augmentation=False,
    )

    keep_cols = ["speaker_id", "language"]

    train_ds_encoded = train_ds.map(
        train_preprocess,
        remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
        batched=True,
        batch_size=cfg.map_batch_size,
        with_indices=True,
    )
    valid_ds_encoded = valid_ds.map(
        valid_preprocess,
        remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
        batched=True,
        batch_size=cfg.map_batch_size,
        with_indices=True,
    )

    model_config = AutoConfig.from_pretrained(cfg.model_id)
    model_config.num_labels = num_labels
    model_config.label2id = str_to_int
    model_config.id2label = int_to_str

    if cfg.do_apply_dropout:
        if hasattr(model_config, "hidden_dropout"):
            model_config.hidden_dropout = cfg.hidden_dropout
        if hasattr(model_config, "attention_dropout"):
            model_config.attention_dropout = cfg.attention_dropout
        if hasattr(model_config, "activation_dropout"):
            model_config.activation_dropout = cfg.activation_dropout
        if hasattr(model_config, "feat_proj_dropout"):
            model_config.feat_proj_dropout = cfg.feat_proj_dropout

    model = AutoModelForAudioClassification.from_pretrained(cfg.model_id, config=model_config)
    data_collator = AudioDataCollator(feature_extractor, input_features_key)
    compute_metrics = make_compute_metrics(int_to_str)

    run_dir = Path(cfg.output_root) / cfg.run_name
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_arguments(cfg, run_dir, report_to)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_ds_encoded,
        eval_dataset=valid_ds_encoded,
        feature_extractor=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Train loop starting...")
    train_result = trainer.train()
    trainer.save_model(str(run_dir))
    feature_extractor.save_pretrained(str(run_dir))
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    print("Final evaluation starting...")
    final_eval_metrics = trainer.evaluate(eval_dataset=valid_ds_encoded)
    trainer.save_metrics("eval", final_eval_metrics)
    print("Final eval metrics:")
    print(final_eval_metrics)

    pred_output = trainer.predict(valid_ds_encoded, metric_key_prefix="predict")
    trainer.save_metrics("predict", pred_output.metrics)

    logits = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
    preds = np.argmax(logits, axis=1)
    refs = pred_output.label_ids
    label_names = [int_to_str[i] for i in range(num_labels)]

    per_language_rows = compute_per_language_rows(preds, refs, int_to_str)
    if cfg.save_per_language_csv:
        per_lang_path = diagnostics_dir / "per_language_accuracy_validation.csv"
        write_rows_csv(
            per_lang_path,
            per_language_rows,
            ["label_id", "language", "n_samples", "n_correct", "accuracy"],
        )
        print(f"Saved per-language validation accuracy: {per_lang_path}")

    cm = confusion_matrix(refs, preds, labels=list(range(num_labels)))
    if cfg.save_confusion_matrix_csv:
        cm_csv_path = diagnostics_dir / "confusion_matrix_validation.csv"
        write_confusion_matrix_csv(cm_csv_path, cm, label_names)
        print(f"Saved confusion matrix CSV: {cm_csv_path}")

    if cfg.save_confusion_matrix_png:
        cm_png_path = diagnostics_dir / "confusion_matrix_validation.png"
        plot_confusion_matrix(cm_png_path, cm, label_names, title="Validation Confusion Matrix")
        print(f"Saved confusion matrix PNG: {cm_png_path}")

    if cfg.run_speaker_diagnostic:
        train_speaker_ids = set(map(str, train_ds_encoded["speaker_id"]))
        run_speaker_diagnostic(
            valid_ds_encoded=valid_ds_encoded,
            preds=preds,
            refs=refs,
            train_speaker_ids=train_speaker_ids,
            cfg=cfg,
            output_dir=diagnostics_dir,
        )

    if wandb_enabled and wandb is not None:
        wandb.finish()

    print(f"Run artifacts saved under: {run_dir}")


if __name__ == "__main__":
    main()
