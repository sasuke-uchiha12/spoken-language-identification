#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_BASELINE_RUN = "SLID_utter-project-mHuBERT-147_1e-05_20260301_150502"
DEFAULT_MITIGATION1_RUN = "SLID_utter-project-mHuBERT-147_1e-05_20260302_111613"
DEFAULT_IMPROVED_RUN = "SLID_utter-project-mHuBERT-147_1e-05_20260305_022125"

WEIGHT_FILENAMES = ("pytorch_model.bin", "model.safetensors")


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_name: str
    run_dir: Path


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    src_relpath: Path
    dst_dirname: str
    dst_name_tmpl: str
    required_for_task3: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Task 3 analysis artifacts by reusing existing run outputs, "
            "deriving summary tables, and reporting missing/unavailable artifacts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Root directory containing run folders. Defaults to <repo-root>/indic-SLID-mac.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Reports directory. Defaults to <repo-root>/reports.",
    )
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--mitigation-run", default=DEFAULT_MITIGATION1_RUN)
    parser.add_argument("--improved-run", default=DEFAULT_IMPROVED_RUN)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_confusion_matrix_csv(path: Path) -> Tuple[List[str], Dict[str, List[int]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty confusion matrix CSV: {path}")

    labels = rows[0][1:]
    matrix: Dict[str, List[int]] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        true_lang = row[0]
        matrix[true_lang] = [int(x) for x in row[1:]]
    return labels, matrix


def top_confusions(path: Path, top_k: int = 10) -> List[Dict[str, object]]:
    labels, matrix = parse_confusion_matrix_csv(path)
    rows: List[Dict[str, object]] = []

    for true_lang, values in matrix.items():
        true_total = sum(values)
        if true_total <= 0:
            continue

        for pred_lang, count in zip(labels, values):
            if pred_lang == true_lang or count <= 0:
                continue
            percent = (100.0 * count / true_total) if true_total > 0 else 0.0
            rows.append(
                {
                    "true_lang": true_lang,
                    "predicted_lang": pred_lang,
                    "count": int(count),
                    "percent_of_true": round(percent, 4),
                }
            )

    rows.sort(key=lambda r: int(r["count"]), reverse=True)
    return rows[:top_k]


def read_per_language(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"]
            out[lang] = {
                "accuracy": float(row["accuracy"]),
                "n_samples": float(row["n_samples"]),
                "n_correct": float(row["n_correct"]),
            }
    return out


def read_speaker_stats(path: Path) -> Dict[str, float]:
    accuracies: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            accuracies.append(float(row["accuracy"]))

    if not accuracies:
        return {
            "speaker_count": 0.0,
            "zero_accuracy_speakers": 0.0,
            "le_0_2_speakers": 0.0,
            "ge_0_8_speakers": 0.0,
            "median_speaker_accuracy": 0.0,
        }

    sorted_acc = sorted(accuracies)
    n = len(sorted_acc)
    median = sorted_acc[n // 2] if n % 2 == 1 else 0.5 * (sorted_acc[n // 2 - 1] + sorted_acc[n // 2])

    return {
        "speaker_count": float(n),
        "zero_accuracy_speakers": float(sum(1 for x in sorted_acc if x == 0.0)),
        "le_0_2_speakers": float(sum(1 for x in sorted_acc if x <= 0.2)),
        "ge_0_8_speakers": float(sum(1 for x in sorted_acc if x >= 0.8)),
        "median_speaker_accuracy": float(median),
    }


def find_weight_files(run_dir: Path) -> List[Path]:
    found: List[Path] = []
    for name in WEIGHT_FILENAMES:
        found.extend(run_dir.rglob(name))
    return sorted(set(found))


def round_or_none(value: Optional[float], places: int = 6) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), places)


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    runs_root = (args.runs_root or (repo_root / "indic-SLID-mac")).resolve()
    reports_dir = (args.reports_dir or (repo_root / "reports")).resolve()
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"

    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    run_specs = [
        RunSpec("baseline", args.baseline_run, runs_root / args.baseline_run),
        RunSpec("mitigation1", args.mitigation_run, runs_root / args.mitigation_run),
        RunSpec("improved", args.improved_run, runs_root / args.improved_run),
    ]

    common_artifacts = [
        ArtifactSpec(
            key="confusion_matrix_csv",
            src_relpath=Path("diagnostics/confusion_matrix_validation.csv"),
            dst_dirname="tables",
            dst_name_tmpl="confusion_matrix_{label}.csv",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="confusion_matrix_png",
            src_relpath=Path("diagnostics/confusion_matrix_validation.png"),
            dst_dirname="figures",
            dst_name_tmpl="confusion_matrix_{label}.png",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="per_language_accuracy_csv",
            src_relpath=Path("diagnostics/per_language_accuracy_validation.csv"),
            dst_dirname="tables",
            dst_name_tmpl="per_language_accuracy_{label}.csv",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="speaker_accuracy_csv",
            src_relpath=Path("diagnostics/speaker_accuracy_validation.csv"),
            dst_dirname="tables",
            dst_name_tmpl="speaker_accuracy_{label}.csv",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="speaker_summary_json",
            src_relpath=Path("diagnostics/speaker_diagnostic_summary.json"),
            dst_dirname="tables",
            dst_name_tmpl="speaker_summary_{label}.json",
            required_for_task3=True,
        ),
    ]

    baseline_tsne_artifacts = [
        ArtifactSpec(
            key="tsne_language_png",
            src_relpath=Path("diagnostics/tsne_validation_by_language.png"),
            dst_dirname="figures",
            dst_name_tmpl="tsne_{label}_by_language.png",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="tsne_speaker_png",
            src_relpath=Path("diagnostics/tsne_validation_by_speaker_topk.png"),
            dst_dirname="figures",
            dst_name_tmpl="tsne_{label}_by_speaker.png",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="tsne_points_csv",
            src_relpath=Path("diagnostics/tsne_validation_points.csv"),
            dst_dirname="tables",
            dst_name_tmpl="tsne_{label}_points.csv",
            required_for_task3=False,
        ),
    ]

    improved_tsne_artifacts = [
        ArtifactSpec(
            key="tsne_language_png",
            src_relpath=Path("diagnostics/tsne_validation_by_language.png"),
            dst_dirname="figures",
            dst_name_tmpl="tsne_{label}_by_language.png",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="tsne_speaker_png",
            src_relpath=Path("diagnostics/tsne_validation_by_speaker_topk.png"),
            dst_dirname="figures",
            dst_name_tmpl="tsne_{label}_by_speaker.png",
            required_for_task3=True,
        ),
        ArtifactSpec(
            key="tsne_points_csv",
            src_relpath=Path("diagnostics/tsne_validation_points.csv"),
            dst_dirname="tables",
            dst_name_tmpl="tsne_{label}_points.csv",
            required_for_task3=False,
        ),
    ]

    copied_manifest_rows: List[Dict[str, object]] = []
    inventory_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    for run in run_specs:
        if not run.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run.run_dir}")

        weight_files = find_weight_files(run.run_dir)
        checkpoint_dirs = sorted(run.run_dir.glob("checkpoint-*"))

        artifact_specs = list(common_artifacts)
        if run.label == "baseline":
            artifact_specs.extend(baseline_tsne_artifacts)
        if run.label == "improved":
            artifact_specs.extend(improved_tsne_artifacts)

        for spec in artifact_specs:
            src = run.run_dir / spec.src_relpath
            dst_dir = figures_dir if spec.dst_dirname == "figures" else tables_dir
            dst = dst_dir / spec.dst_name_tmpl.format(label=run.label)
            exists = src.exists()

            row = {
                "run_label": run.label,
                "run_name": run.run_name,
                "artifact_key": spec.key,
                "source_path": str(src.relative_to(repo_root)) if src.exists() else str(src),
                "destination_path": str(dst.relative_to(repo_root)),
                "required_for_task3": spec.required_for_task3,
                "exists_in_source": exists,
                "copied": False,
                "weight_files_found": len(weight_files),
                "checkpoint_dirs_found": len(checkpoint_dirs),
                "status": "copied" if exists else "missing",
                "notes": "",
            }

            if exists:
                ensure_dir(dst.parent)
                shutil.copy2(src, dst)
                row["copied"] = True
                copied_manifest_rows.append(
                    {
                        "run_label": run.label,
                        "run_name": run.run_name,
                        "artifact_key": spec.key,
                        "source_path": str(src.relative_to(repo_root)),
                        "destination_path": str(dst.relative_to(repo_root)),
                        "status": "copied",
                    }
                )
            else:
                if weight_files:
                    reason = "missing_source_artifact_but_weights_available_for_regeneration"
                elif checkpoint_dirs:
                    reason = "missing_source_artifact_checkpoint_dirs_present_but_no_model_weights"
                else:
                    reason = "missing_source_artifact_and_no_weights"
                row["notes"] = reason
                if spec.required_for_task3:
                    missing_rows.append(
                        {
                            "run_label": run.label,
                            "run_name": run.run_name,
                            "artifact_key": spec.key,
                            "expected_destination": str(dst.relative_to(repo_root)),
                            "reason": reason,
                            "weight_files_found": len(weight_files),
                            "checkpoint_dirs_found": len(checkpoint_dirs),
                        }
                    )

            inventory_rows.append(row)

    write_csv(
        tables_dir / "artifact_inventory.csv",
        inventory_rows,
        [
            "run_label",
            "run_name",
            "artifact_key",
            "source_path",
            "destination_path",
            "required_for_task3",
            "exists_in_source",
            "copied",
            "weight_files_found",
            "checkpoint_dirs_found",
            "status",
            "notes",
        ],
    )

    write_csv(
        tables_dir / "missing_required_artifacts.csv",
        missing_rows,
        [
            "run_label",
            "run_name",
            "artifact_key",
            "expected_destination",
            "reason",
            "weight_files_found",
            "checkpoint_dirs_found",
        ],
    )

    write_csv(
        tables_dir / "task3_artifact_manifest.csv",
        copied_manifest_rows,
        ["run_label", "run_name", "artifact_key", "source_path", "destination_path", "status"],
    )

    # Global metrics comparison
    global_rows: List[Dict[str, object]] = []
    for run in run_specs:
        all_results_path = run.run_dir / "all_results.json"
        if not all_results_path.exists():
            continue
        metrics = read_json(all_results_path)
        global_rows.append(
            {
                "run_label": run.label,
                "run_name": run.run_name,
                "eval_accuracy": round_or_none(metrics.get("eval_accuracy")),
                "eval_macro_f1": round_or_none(metrics.get("eval_macro_f1")),
                "eval_loss": round_or_none(metrics.get("eval_loss")),
                "predict_accuracy": round_or_none(metrics.get("predict_accuracy")),
                "predict_macro_f1": round_or_none(metrics.get("predict_macro_f1")),
                "predict_loss": round_or_none(metrics.get("predict_loss")),
                "train_runtime": round_or_none(metrics.get("train_runtime")),
                "train_loss": round_or_none(metrics.get("train_loss")),
            }
        )

    write_csv(
        tables_dir / "global_metrics_comparison.csv",
        global_rows,
        [
            "run_label",
            "run_name",
            "eval_accuracy",
            "eval_macro_f1",
            "eval_loss",
            "predict_accuracy",
            "predict_macro_f1",
            "predict_loss",
            "train_runtime",
            "train_loss",
        ],
    )

    # Top-10 confusion tables
    for run in run_specs:
        conf_path = tables_dir / f"confusion_matrix_{run.label}.csv"
        if not conf_path.exists():
            continue
        conf_rows = top_confusions(conf_path, top_k=10)
        write_csv(
            tables_dir / f"top10_confusions_{run.label}.csv",
            conf_rows,
            ["true_lang", "predicted_lang", "count", "percent_of_true"],
        )

    # Per-language delta tables
    baseline_pl_path = tables_dir / "per_language_accuracy_baseline.csv"
    mitigation_pl_path = tables_dir / "per_language_accuracy_mitigation1.csv"
    improved_pl_path = tables_dir / "per_language_accuracy_improved.csv"

    if baseline_pl_path.exists() and mitigation_pl_path.exists():
        base = read_per_language(baseline_pl_path)
        mit = read_per_language(mitigation_pl_path)
        rows: List[Dict[str, object]] = []
        for lang in sorted(base):
            if lang not in mit:
                continue
            rows.append(
                {
                    "language": lang,
                    "baseline_accuracy": round_or_none(base[lang]["accuracy"]),
                    "mitigation1_accuracy": round_or_none(mit[lang]["accuracy"]),
                    "delta_accuracy": round_or_none(mit[lang]["accuracy"] - base[lang]["accuracy"]),
                    "baseline_n_samples": int(base[lang]["n_samples"]),
                    "mitigation1_n_samples": int(mit[lang]["n_samples"]),
                }
            )
        rows.sort(key=lambda r: float(r["delta_accuracy"]), reverse=True)
        write_csv(
            tables_dir / "per_language_delta_mitigation1_vs_baseline.csv",
            rows,
            [
                "language",
                "baseline_accuracy",
                "mitigation1_accuracy",
                "delta_accuracy",
                "baseline_n_samples",
                "mitigation1_n_samples",
            ],
        )

    if baseline_pl_path.exists() and improved_pl_path.exists():
        base = read_per_language(baseline_pl_path)
        imp = read_per_language(improved_pl_path)
        rows = []
        for lang in sorted(base):
            if lang not in imp:
                continue
            rows.append(
                {
                    "language": lang,
                    "baseline_accuracy": round_or_none(base[lang]["accuracy"]),
                    "improved_accuracy": round_or_none(imp[lang]["accuracy"]),
                    "delta_accuracy": round_or_none(imp[lang]["accuracy"] - base[lang]["accuracy"]),
                    "baseline_n_samples": int(base[lang]["n_samples"]),
                    "improved_n_samples": int(imp[lang]["n_samples"]),
                }
            )
        rows.sort(key=lambda r: float(r["delta_accuracy"]), reverse=True)
        write_csv(
            tables_dir / "per_language_delta_improved_vs_baseline.csv",
            rows,
            [
                "language",
                "baseline_accuracy",
                "improved_accuracy",
                "delta_accuracy",
                "baseline_n_samples",
                "improved_n_samples",
            ],
        )

    if baseline_pl_path.exists() and mitigation_pl_path.exists() and improved_pl_path.exists():
        base = read_per_language(baseline_pl_path)
        mit = read_per_language(mitigation_pl_path)
        imp = read_per_language(improved_pl_path)
        rows = []
        for lang in sorted(base):
            if lang not in mit or lang not in imp:
                continue
            rows.append(
                {
                    "language": lang,
                    "baseline_accuracy": round_or_none(base[lang]["accuracy"]),
                    "mitigation1_accuracy": round_or_none(mit[lang]["accuracy"]),
                    "improved_accuracy": round_or_none(imp[lang]["accuracy"]),
                    "delta_mitigation1_vs_baseline": round_or_none(mit[lang]["accuracy"] - base[lang]["accuracy"]),
                    "delta_improved_vs_baseline": round_or_none(imp[lang]["accuracy"] - base[lang]["accuracy"]),
                    "delta_improved_vs_mitigation1": round_or_none(imp[lang]["accuracy"] - mit[lang]["accuracy"]),
                }
            )
        rows.sort(key=lambda r: float(r["delta_improved_vs_baseline"]), reverse=True)
        write_csv(
            tables_dir / "per_language_accuracy_comparison.csv",
            rows,
            [
                "language",
                "baseline_accuracy",
                "mitigation1_accuracy",
                "improved_accuracy",
                "delta_mitigation1_vs_baseline",
                "delta_improved_vs_baseline",
                "delta_improved_vs_mitigation1",
            ],
        )

    # Speaker summary comparison table
    speaker_rows: List[Dict[str, object]] = []
    for run in run_specs:
        summary_path = tables_dir / f"speaker_summary_{run.label}.json"
        speaker_csv_path = tables_dir / f"speaker_accuracy_{run.label}.csv"
        if not summary_path.exists() or not speaker_csv_path.exists():
            continue
        summary = read_json(summary_path)
        speaker_stats = read_speaker_stats(speaker_csv_path)

        speaker_rows.append(
            {
                "run_label": run.label,
                "run_name": run.run_name,
                "train_unique_speakers": int(summary.get("train_unique_speakers", 0)),
                "validation_unique_speakers": int(summary.get("validation_unique_speakers", 0)),
                "overlap_speaker_count": int(summary.get("overlap_speaker_count", 0)),
                "overlap_ratio_validation_speakers": round_or_none(summary.get("overlap_ratio_validation_speakers")),
                "validation_samples_seen_speakers": int(summary.get("validation_samples_seen_speakers", 0)),
                "validation_samples_unseen_speakers": int(summary.get("validation_samples_unseen_speakers", 0)),
                "accuracy_seen_speakers": round_or_none(summary.get("accuracy_seen_speakers")),
                "accuracy_unseen_speakers": round_or_none(summary.get("accuracy_unseen_speakers")),
                "accuracy_gap_seen_minus_unseen": round_or_none(summary.get("accuracy_gap_seen_minus_unseen")),
                "warnings": " | ".join(summary.get("warnings", [])) if isinstance(summary.get("warnings", []), list) else "",
                "speaker_count": int(speaker_stats["speaker_count"]),
                "zero_accuracy_speakers": int(speaker_stats["zero_accuracy_speakers"]),
                "le_0_2_speakers": int(speaker_stats["le_0_2_speakers"]),
                "ge_0_8_speakers": int(speaker_stats["ge_0_8_speakers"]),
                "median_speaker_accuracy": round_or_none(speaker_stats["median_speaker_accuracy"]),
            }
        )

    write_csv(
        tables_dir / "speaker_summary_comparison.csv",
        speaker_rows,
        [
            "run_label",
            "run_name",
            "train_unique_speakers",
            "validation_unique_speakers",
            "overlap_speaker_count",
            "overlap_ratio_validation_speakers",
            "validation_samples_seen_speakers",
            "validation_samples_unseen_speakers",
            "accuracy_seen_speakers",
            "accuracy_unseen_speakers",
            "accuracy_gap_seen_minus_unseen",
            "warnings",
            "speaker_count",
            "zero_accuracy_speakers",
            "le_0_2_speakers",
            "ge_0_8_speakers",
            "median_speaker_accuracy",
        ],
    )

    missing_required = [r for r in missing_rows]

    print("Task 3 artifact generation complete.")
    print(f"- Reports directory: {reports_dir}")
    print(f"- Figures directory: {figures_dir}")
    print(f"- Tables directory: {tables_dir}")
    print(f"- Inventory CSV: {tables_dir / 'artifact_inventory.csv'}")
    print(f"- Missing required artifacts: {len(missing_required)}")

    if missing_required:
        for row in missing_required:
            print(
                "  - "
                f"{row['run_label']}::{row['artifact_key']} -> {row['reason']} "
                f"(weights={row['weight_files_found']}, checkpoints={row['checkpoint_dirs_found']})"
            )


if __name__ == "__main__":
    main()
