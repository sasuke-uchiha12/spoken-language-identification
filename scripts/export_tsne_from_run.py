#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_model as base


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "last":
        return run_dir

    if checkpoint == "best":
        trainer_state_path = run_dir / "trainer_state.json"
        if not trainer_state_path.exists():
            raise FileNotFoundError(f"Missing trainer state: {trainer_state_path}")
        with trainer_state_path.open("r", encoding="utf-8") as f:
            trainer_state = json.load(f)
        best_ckpt = trainer_state.get("best_model_checkpoint")
        if not best_ckpt:
            raise ValueError("best_model_checkpoint not found in trainer_state.json")
        return Path(best_ckpt)

    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate


def has_model_files(path: Path) -> bool:
    patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    return any((path / name).exists() for name in patterns)


def infer_input_features_key(feature_extractor) -> str:
    input_names = set(feature_extractor.model_input_names or [])
    if "input_values" in input_names:
        return "input_values"
    if "input_features" in input_names:
        return "input_features"
    # Safe default for HuBERT/wav2vec-style models.
    return "input_values"


def build_validation_dataset(
    *,
    feature_extractor,
    cfg: base.TrainConfig,
    input_features_key: str,
):
    dataset = load_dataset(cfg.dataset_name)
    train_ds = dataset["train"].shuffle(seed=cfg.seed)
    valid_ds = dataset["validation"].shuffle(seed=cfg.seed)

    train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=cfg.sample_rate))
    valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=cfg.sample_rate))

    label_list = sorted(train_ds.unique("language"))
    str_to_int: Dict[str, int] = {label: idx for idx, label in enumerate(label_list)}

    preprocess = base.make_preprocess_function(
        feature_extractor=feature_extractor,
        str_to_int=str_to_int,
        speaker_to_int=None,
        input_features_key=input_features_key,
        max_duration_sec=cfg.max_duration_sec,
        augmenter=None,
        apply_augmentation=False,
    )

    keep_cols = ["speaker_id", "language"]
    valid_ds_encoded = valid_ds.map(
        preprocess,
        remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
        batched=True,
        batch_size=cfg.map_batch_size,
        with_indices=True,
    )
    return valid_ds_encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validation t-SNE from an existing run/checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Run directory path (e.g., indic-SLID-mac/SLID_... ).")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint selector: best | last | relative/absolute checkpoint path.",
    )
    parser.add_argument("--split", default="validation", choices=["validation"], help="Dataset split to project.")
    parser.add_argument("--max-samples", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speaker-top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not has_model_files(checkpoint_path):
        raise FileNotFoundError(
            f"No model weight files found under checkpoint path: {checkpoint_path}. "
            "This run folder appears to contain metrics only; cannot export last-layer t-SNE."
        )

    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    cfg = base.TrainConfig()
    cfg.run_tsne = True
    cfg.tsne_max_samples = int(args.max_samples)
    cfg.tsne_batch_size = int(args.batch_size)
    cfg.tsne_perplexity = float(args.perplexity)
    cfg.tsne_random_state = int(args.seed)
    cfg.tsne_speaker_top_k = int(args.speaker_top_k)

    base.set_all_seeds(cfg.seed)
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(run_dir), do_normalize=True, return_attention_mask=True
        )
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(checkpoint_path), do_normalize=True, return_attention_mask=True
        )
    model = AutoModelForAudioClassification.from_pretrained(str(checkpoint_path))

    input_features_key = infer_input_features_key(feature_extractor)
    valid_ds_encoded = build_validation_dataset(
        feature_extractor=feature_extractor,
        cfg=cfg,
        input_features_key=input_features_key,
    )

    data_collator = base.AudioDataCollator(feature_extractor, input_features_key)
    reps = base.extract_last_layer_representations(
        model=model,
        dataset_encoded=valid_ds_encoded,
        data_collator=data_collator,
        batch_size=cfg.tsne_batch_size,
    )

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    base.run_tsne_and_save(
        representations=reps,
        languages=[str(x) for x in valid_ds_encoded["language"]],
        speaker_ids=[str(x) for x in valid_ds_encoded["speaker_id"]],
        output_dir=diagnostics_dir,
        cfg=cfg,
    )

    print("t-SNE export completed.")


if __name__ == "__main__":
    main()
