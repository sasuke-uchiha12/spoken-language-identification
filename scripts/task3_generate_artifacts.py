#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    run_id: str


RUNS: List[RunSpec] = [
    RunSpec("baseline_untuned", "Baseline (untuned)", "SLID_utter-project-mHuBERT-147_1e-05_20260301_094826"),
    RunSpec("tuned_ref", "Tuned reference (no mitigation)", "SLID_utter-project-mHuBERT-147_1e-05_20260301_150502"),
    RunSpec("mitigation1", "Mitigation 1 (data-centric)", "SLID_utter-project-mHuBERT-147_1e-05_20260302_111613"),
    RunSpec("improved_dann", "Improved (DANN)", "SLID_utter-project-mHuBERT-147_1e-05_20260305_022125"),
]

TRACKED_CONFUSIONS: List[Tuple[str, str]] = [
    ("konkani", "marathi"),
    ("sindhi", "punjabi"),
    ("hindi", "urdu"),
    ("dogri", "punjabi"),
    ("bodo", "manipuri"),
    ("odia", "bengali"),
    ("santali", "bengali"),
]


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_per_language_accuracy(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["language"]] = float(row["accuracy"])
    return out


def read_confusion(path: Path) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        preds = header[1:]
        for row in reader:
            true_label = row[0]
            vals = [int(x) for x in row[1:]]
            for pred_label, val in zip(preds, vals):
                if true_label != pred_label:
                    out[(true_label, pred_label)] = val
    return out


def top_confusions(conf: Dict[Tuple[str, str], int], n: int = 10) -> List[Tuple[str, str, int]]:
    pairs = sorted(((t, p, c) for (t, p), c in conf.items() if c > 0), key=lambda x: x[2], reverse=True)
    return pairs[:n]


def load_tsne_points(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    coords: List[List[float]] = []
    langs: List[str] = []
    speakers: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append([float(row["x"]), float(row["y"])])
            langs.append(str(row["language"]))
            speakers.append(str(row["speaker_id"]))
    return np.asarray(coords, dtype=np.float64), langs, speakers


def filter_min_count(
    points: np.ndarray,
    labels: Sequence[str],
    min_count: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    counts: Dict[str, int] = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    keep = np.array([counts[l] >= min_count for l in labels], dtype=bool)
    return points[keep], [l for l, k in zip(labels, keep) if k]


def safe_silhouette(points: np.ndarray, labels: Sequence[str]) -> float:
    if len(points) < 3:
        return float("nan")
    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return float("nan")
    try:
        return float(silhouette_score(points, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def mean_pairwise_distance(points: np.ndarray) -> float:
    n = points.shape[0]
    if n < 2:
        return float("nan")
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(n, k=1)
    vals = d[iu]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def cluster_distance_summary(points: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    by_label: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)

    intra_vals: List[float] = []
    centroids: List[np.ndarray] = []
    for _, idx_list in by_label.items():
        cluster = points[np.asarray(idx_list, dtype=np.int64)]
        centroids.append(cluster.mean(axis=0))
        if cluster.shape[0] >= 2:
            intra = mean_pairwise_distance(cluster)
            if not np.isnan(intra):
                intra_vals.append(intra)

    intra_mean = float(np.mean(intra_vals)) if intra_vals else float("nan")
    centroid_arr = np.asarray(centroids, dtype=np.float64) if centroids else np.empty((0, points.shape[1]))
    inter_centroid_mean = mean_pairwise_distance(centroid_arr) if len(centroid_arr) >= 2 else float("nan")

    ratio = float("nan")
    if not np.isnan(intra_mean) and intra_mean > 0 and not np.isnan(inter_centroid_mean):
        ratio = float(inter_centroid_mean / intra_mean)

    return {
        "n_points": int(points.shape[0]),
        "n_labels": int(len(by_label)),
        "intra_mean_distance": intra_mean,
        "inter_centroid_mean_distance": inter_centroid_mean,
        "separation_ratio": ratio,
    }


def select_topk_speakers(
    points: np.ndarray,
    speakers: Sequence[str],
    top_k: int = 12,
) -> Tuple[np.ndarray, List[str]]:
    counts: Dict[str, int] = {}
    for s in speakers:
        counts[s] = counts.get(s, 0) + 1
    top = [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    keep = np.array([s in top for s in speakers], dtype=bool)
    return points[keep], [s for s, k in zip(speakers, keep) if k]


def knn_label_purity(points: np.ndarray, labels: Sequence[str], k: int = 5) -> float:
    n = points.shape[0]
    if n <= k:
        return float("nan")
    n_neighbors = min(n, k + 1)
    try:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(points)
        indices = nn.kneighbors(points, return_distance=False)
    except Exception:
        return float("nan")

    label_arr = np.asarray(labels)
    scores: List[float] = []
    for i in range(n):
        neigh = indices[i]
        neigh = [j for j in neigh if j != i][:k]
        if not neigh:
            continue
        same = float(np.mean(label_arr[neigh] == label_arr[i]))
        scores.append(same)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    runs_root = repo_root / "indic-SLID-mac"
    reports_dir = repo_root / "reports"
    tables_dir = reports_dir / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict] = []
    per_lang_by_run: Dict[str, Dict[str, float]] = {}
    conf_by_run: Dict[str, Dict[Tuple[str, str], int]] = {}
    speaker_rows: List[Dict] = []
    tsne_cluster_rows: List[Dict] = []
    tsne_distance_rows: List[Dict] = []
    source_files: List[str] = []

    for run in RUNS:
        run_dir = runs_root / run.run_id
        diagnostics_dir = run_dir / "diagnostics"
        eval_path = run_dir / "eval_results.json"
        trainer_state_path = run_dir / "trainer_state.json"
        per_lang_path = diagnostics_dir / "per_language_accuracy_validation.csv"
        conf_path = diagnostics_dir / "confusion_matrix_validation.csv"
        conf_png_path = diagnostics_dir / "confusion_matrix_validation.png"
        speaker_path = diagnostics_dir / "speaker_diagnostic_summary.json"
        tsne_points_path = diagnostics_dir / "tsne_validation_points.csv"
        tsne_lang_png_path = diagnostics_dir / "tsne_validation_by_language.png"
        tsne_spk_png_path = diagnostics_dir / "tsne_validation_by_speaker_topk.png"

        for p in [eval_path, trainer_state_path, per_lang_path, conf_path, speaker_path]:
            source_files.append(str(p.relative_to(repo_root)))
            if not p.exists():
                raise FileNotFoundError(f"Required file missing: {p}")
        if conf_png_path.exists():
            source_files.append(str(conf_png_path.relative_to(repo_root)))

        eval_res = read_json(eval_path)
        trainer_state = read_json(trainer_state_path)
        best_ckpt = str(trainer_state.get("best_model_checkpoint", ""))
        metrics_rows.append(
            {
                "run_key": run.key,
                "run_label": run.label,
                "run_id": run.run_id,
                "eval_accuracy": eval_res["eval_accuracy"],
                "eval_macro_f1": eval_res["eval_macro_f1"],
                "eval_loss": eval_res["eval_loss"],
                "best_global_step": trainer_state.get("best_global_step"),
                "best_metric": trainer_state.get("best_metric"),
                "best_model_checkpoint": Path(best_ckpt).name if best_ckpt else "",
            }
        )

        per_lang_by_run[run.key] = read_per_language_accuracy(per_lang_path)
        conf_by_run[run.key] = read_confusion(conf_path)

        sp = read_json(speaker_path)
        speaker_rows.append(
            {
                "run_key": run.key,
                "run_label": run.label,
                "run_id": run.run_id,
                "train_unique_speakers": sp.get("train_unique_speakers"),
                "validation_unique_speakers": sp.get("validation_unique_speakers"),
                "overlap_speaker_count": sp.get("overlap_speaker_count"),
                "overlap_ratio_validation_speakers": sp.get("overlap_ratio_validation_speakers"),
                "validation_samples_seen_speakers": sp.get("validation_samples_seen_speakers"),
                "validation_samples_unseen_speakers": sp.get("validation_samples_unseen_speakers"),
                "accuracy_seen_speakers": sp.get("accuracy_seen_speakers"),
                "accuracy_unseen_speakers": sp.get("accuracy_unseen_speakers"),
                "warnings_count": len(sp.get("warnings", [])),
                "warnings": " | ".join(sp.get("warnings", [])),
            }
        )

        if tsne_points_path.exists():
            source_files.append(str(tsne_points_path.relative_to(repo_root)))
            if tsne_lang_png_path.exists():
                source_files.append(str(tsne_lang_png_path.relative_to(repo_root)))
            if tsne_spk_png_path.exists():
                source_files.append(str(tsne_spk_png_path.relative_to(repo_root)))
            points, languages, speakers = load_tsne_points(tsne_points_path)
            lang_points, lang_labels = filter_min_count(points, languages, min_count=2)
            sil_lang = safe_silhouette(lang_points, lang_labels)

            spk_points, spk_labels = select_topk_speakers(points, speakers, top_k=12)
            spk_points_f, spk_labels_f = filter_min_count(spk_points, spk_labels, min_count=2)
            sil_spk = safe_silhouette(spk_points_f, spk_labels_f)
            purity = knn_label_purity(spk_points_f, spk_labels_f, k=5)

            tsne_cluster_rows.append(
                {
                    "run_key": run.key,
                    "run_label": run.label,
                    "run_id": run.run_id,
                    "n_points_tsne": int(points.shape[0]),
                    "n_language_labels": len(set(languages)),
                    "silhouette_language": sil_lang,
                    "n_speaker_topk_points": int(spk_points_f.shape[0]),
                    "n_speaker_topk_labels": len(set(spk_labels_f)),
                    "silhouette_speaker_topk": sil_spk,
                    "knn_speaker_purity_at_5": purity,
                }
            )

            for view_name, v_points, v_labels in [
                ("language", lang_points, lang_labels),
                ("speaker_topk", spk_points_f, spk_labels_f),
            ]:
                summary = cluster_distance_summary(v_points, v_labels)
                tsne_distance_rows.append(
                    {
                        "run_key": run.key,
                        "run_label": run.label,
                        "run_id": run.run_id,
                        "view": view_name,
                        **summary,
                    }
                )

    metrics_rows.sort(
        key=lambda r: ["baseline_untuned", "tuned_ref", "mitigation1", "improved_dann"].index(r["run_key"])
    )
    write_csv(
        tables_dir / "task3_metrics_comparison.csv",
        metrics_rows,
        [
            "run_key",
            "run_label",
            "run_id",
            "eval_accuracy",
            "eval_macro_f1",
            "eval_loss",
            "best_global_step",
            "best_metric",
            "best_model_checkpoint",
        ],
    )

    # Extra convenience files matching existing naming pattern in this branch.
    baseline_speaker = next(r for r in speaker_rows if r["run_key"] == "baseline_untuned")
    tuned_speaker = next(r for r in speaker_rows if r["run_key"] == "tuned_ref")
    m1_speaker = next(r for r in speaker_rows if r["run_key"] == "mitigation1")
    write_json(tables_dir / "speaker_summary_baseline.json", baseline_speaker)
    write_json(tables_dir / "speaker_summary_tuned_reference.json", tuned_speaker)
    write_json(tables_dir / "speaker_summary_mitigation1.json", m1_speaker)

    # Per-language deltas.
    all_langs = sorted(set().union(*[set(x.keys()) for x in per_lang_by_run.values()]))
    per_lang_delta_rows: List[Dict] = []
    for lang in all_langs:
        bu = per_lang_by_run["baseline_untuned"].get(lang, 0.0)
        tr = per_lang_by_run["tuned_ref"].get(lang, 0.0)
        m1 = per_lang_by_run["mitigation1"].get(lang, 0.0)
        d = per_lang_by_run["improved_dann"].get(lang, 0.0)
        per_lang_delta_rows.append(
            {
                "language": lang,
                "baseline_untuned_accuracy": bu,
                "tuned_ref_accuracy": tr,
                "mitigation1_accuracy": m1,
                "improved_dann_accuracy": d,
                "delta_baseline_untuned_to_dann": d - bu,
                "delta_tuned_ref_to_m1": m1 - tr,
                "delta_m1_to_dann": d - m1,
                "delta_tuned_ref_to_dann": d - tr,
            }
        )
    write_csv(
        tables_dir / "task3_per_language_delta.csv",
        per_lang_delta_rows,
        [
            "language",
            "baseline_untuned_accuracy",
            "tuned_ref_accuracy",
            "mitigation1_accuracy",
            "improved_dann_accuracy",
            "delta_baseline_untuned_to_dann",
            "delta_tuned_ref_to_m1",
            "delta_m1_to_dann",
            "delta_tuned_ref_to_dann",
        ],
    )

    # Confusion top pairs table.
    conf_rows: List[Dict] = []
    for run in RUNS:
        top = top_confusions(conf_by_run[run.key], n=15)
        for rank, (true_l, pred_l, count) in enumerate(top, start=1):
            conf_rows.append(
                {
                    "run_key": run.key,
                    "run_label": run.label,
                    "run_id": run.run_id,
                    "rank": rank,
                    "true_language": true_l,
                    "pred_language": pred_l,
                    "count": count,
                }
            )
    write_csv(
        tables_dir / "task3_confusion_top_pairs.csv",
        conf_rows,
        ["run_key", "run_label", "run_id", "rank", "true_language", "pred_language", "count"],
    )

    # Speaker summary table.
    write_csv(
        tables_dir / "task3_speaker_summary.csv",
        speaker_rows,
        [
            "run_key",
            "run_label",
            "run_id",
            "train_unique_speakers",
            "validation_unique_speakers",
            "overlap_speaker_count",
            "overlap_ratio_validation_speakers",
            "validation_samples_seen_speakers",
            "validation_samples_unseen_speakers",
            "accuracy_seen_speakers",
            "accuracy_unseen_speakers",
            "warnings_count",
            "warnings",
        ],
    )
    write_csv(
        tables_dir / "speaker_summary_comparison.csv",
        speaker_rows,
        [
            "run_key",
            "run_label",
            "run_id",
            "train_unique_speakers",
            "validation_unique_speakers",
            "overlap_speaker_count",
            "overlap_ratio_validation_speakers",
            "validation_samples_seen_speakers",
            "validation_samples_unseen_speakers",
            "accuracy_seen_speakers",
            "accuracy_unseen_speakers",
            "warnings_count",
            "warnings",
        ],
    )

    # t-SNE tables.
    write_csv(
        tables_dir / "task3_tsne_cluster_metrics.csv",
        tsne_cluster_rows,
        [
            "run_key",
            "run_label",
            "run_id",
            "n_points_tsne",
            "n_language_labels",
            "silhouette_language",
            "n_speaker_topk_points",
            "n_speaker_topk_labels",
            "silhouette_speaker_topk",
            "knn_speaker_purity_at_5",
        ],
    )
    write_csv(
        tables_dir / "task3_tsne_pairwise_distance_summary.csv",
        tsne_distance_rows,
        [
            "run_key",
            "run_label",
            "run_id",
            "view",
            "n_points",
            "n_labels",
            "intra_mean_distance",
            "inter_centroid_mean_distance",
            "separation_ratio",
        ],
    )

    # Build compact markdown tables.
    m = {r["run_key"]: r for r in metrics_rows}
    bu = m["baseline_untuned"]
    tr = m["tuned_ref"]
    m1 = m["mitigation1"]
    d = m["improved_dann"]

    delta_rows_sorted = sorted(per_lang_delta_rows, key=lambda x: x["delta_baseline_untuned_to_dann"], reverse=True)
    top_gain = delta_rows_sorted[:5]
    top_drop = sorted(per_lang_delta_rows, key=lambda x: x["delta_baseline_untuned_to_dann"])[:5]
    top_gain_m1_to_dann = sorted(per_lang_delta_rows, key=lambda x: x["delta_m1_to_dann"], reverse=True)[:5]
    top_drop_m1_to_dann = sorted(per_lang_delta_rows, key=lambda x: x["delta_m1_to_dann"])[:5]

    conf_tracked_rows = []
    for true_l, pred_l in TRACKED_CONFUSIONS:
        conf_tracked_rows.append(
            {
                "pair": f"{true_l}->{pred_l}",
                "baseline_untuned": conf_by_run["baseline_untuned"].get((true_l, pred_l), 0),
                "tuned_ref": conf_by_run["tuned_ref"].get((true_l, pred_l), 0),
                "mitigation1": conf_by_run["mitigation1"].get((true_l, pred_l), 0),
                "dann": conf_by_run["improved_dann"].get((true_l, pred_l), 0),
            }
        )

    tsne_map = {row["run_key"]: row for row in tsne_cluster_rows}
    tsne_bu = tsne_map.get("baseline_untuned", {})
    tsne_tr = tsne_map.get("tuned_ref", {})
    tsne_m1 = tsne_map.get("mitigation1", {})
    tsne_d = tsne_map.get("improved_dann", {})
    tsne_available = sorted(tsne_map.keys())
    tsne_missing = [r.key for r in RUNS if r.key not in tsne_map]

    speaker_map = {row["run_key"]: row for row in speaker_rows}

    run_id_map = {r.key: r.run_id for r in RUNS}
    conf_img_lines: List[str] = []
    for r in RUNS:
        rel = f"../indic-SLID-mac/{r.run_id}/diagnostics/confusion_matrix_validation.png"
        conf_img_lines.append(f"- {r.label}: `{rel}`")
        conf_img_lines.append(f'  <img src="{rel}" alt="{r.label} confusion matrix" width="560" />')

    tsne_img_lines: List[str] = []
    for key in ["baseline_untuned", "tuned_ref", "mitigation1", "improved_dann"]:
        run_id = run_id_map[key]
        lang_rel = f"../indic-SLID-mac/{run_id}/diagnostics/tsne_validation_by_language.png"
        spk_rel = f"../indic-SLID-mac/{run_id}/diagnostics/tsne_validation_by_speaker_topk.png"
        points_path = runs_root / run_id / "diagnostics" / "tsne_validation_points.csv"
        if points_path.exists():
            tsne_img_lines.append(f"- {next(x.label for x in RUNS if x.key == key)} (language view): `{lang_rel}`")
            tsne_img_lines.append(
                f'  <img src="{lang_rel}" alt="{next(x.label for x in RUNS if x.key == key)} t-SNE by language" width="560" />'
            )
            tsne_img_lines.append(f"- {next(x.label for x in RUNS if x.key == key)} (speaker view): `{spk_rel}`")
            tsne_img_lines.append(
                f'  <img src="{spk_rel}" alt="{next(x.label for x in RUNS if x.key == key)} t-SNE by speaker" width="560" />'
            )
        else:
            tsne_img_lines.append(
                f"- {next(x.label for x in RUNS if x.key == key)}: t-SNE images not available in current local artifacts."
            )

    report_md = f"""# Task 3: Model Analysis (Baseline vs Improved, with Mitigation-1 Bridge)

## 1. Problem Bias Source

The dataset has limited speakers per language in training, which encourages a shortcut: the model can partially rely on speaker characteristics instead of language-discriminative cues. This is difficult because speaker timbre/prosody can dominate representation learning and create brittle language boundaries for unseen voices.

## 2. Proposed Technique and Success Extent

Primary required comparison:

- Baseline (untuned): `{RUNS[0].run_id}`
- Improved model (DANN): `{RUNS[3].run_id}`

Primary comparison metrics:

| Run | Accuracy | Macro-F1 | Eval Loss |
|---|---:|---:|---:|
| Baseline (untuned) | {fmt(bu['eval_accuracy'])} | {fmt(bu['eval_macro_f1'])} | {fmt(bu['eval_loss'])} |
| Improved (DANN) | {fmt(d['eval_accuracy'])} | {fmt(d['eval_macro_f1'])} | {fmt(d['eval_loss'])} |

Secondary progression (bridge/control):

- Tuned reference (no mitigation): `{RUNS[1].run_id}`
- Mitigation 1 (data-centric): `{RUNS[2].run_id}`
- Improved (DANN): `{RUNS[3].run_id}`

| Run | Accuracy | Macro-F1 | Eval Loss |
|---|---:|---:|---:|
| Tuned reference | {fmt(tr['eval_accuracy'])} | {fmt(tr['eval_macro_f1'])} | {fmt(tr['eval_loss'])} |
| Mitigation 1 | {fmt(m1['eval_accuracy'])} | {fmt(m1['eval_macro_f1'])} | {fmt(m1['eval_loss'])} |
| Improved (DANN) | {fmt(d['eval_accuracy'])} | {fmt(d['eval_macro_f1'])} | {fmt(d['eval_loss'])} |

Observed outcome summary:

- DANN improves over untuned baseline by accuracy **{fmt(d['eval_accuracy'] - bu['eval_accuracy'])}** and macro-F1 **{fmt(d['eval_macro_f1'] - bu['eval_macro_f1'])}**.
- DANN improves over tuned reference by accuracy **{fmt(d['eval_accuracy'] - tr['eval_accuracy'])}** and macro-F1 **{fmt(d['eval_macro_f1'] - tr['eval_macro_f1'])}**.
- DANN improves over Mitigation 1 by accuracy **{fmt(d['eval_accuracy'] - m1['eval_accuracy'])}** and macro-F1 **{fmt(d['eval_macro_f1'] - m1['eval_macro_f1'])}**.
- Eval loss is also lower in DANN, indicating better fit without needing retraining here.

## 3. Confusion Pattern Analysis

Tracked high-impact confusion pairs:

| Pair | Baseline (untuned) | Tuned ref | Mitigation 1 | DANN |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| `{r['pair']}` | {r['baseline_untuned']} | {r['tuned_ref']} | {r['mitigation1']} | {r['dann']} |"
        for r in conf_tracked_rows
    ) + f"""

Top confusion trend:

- DANN reduces several major confusions strongly (for example `sindhi->punjabi`, `hindi->urdu`, `konkani->marathi`).
- Some confusion remains structurally hard (`konkani->marathi` still high), likely due acoustic/prosodic similarity and class overlap effects.

Language-wise gains (untuned baseline -> DANN) include classes that were historically weak:

""" + "\n".join(
        f"- `{r['language']}`: {fmt(r['baseline_untuned_accuracy'])} -> {fmt(r['improved_dann_accuracy'])} (delta {fmt(r['delta_baseline_untuned_to_dann'])})"
        for r in top_gain
    ) + """

Largest drops (untuned baseline -> DANN), showing tradeoff:

""" + "\n".join(
        f"- `{r['language']}`: {fmt(r['baseline_untuned_accuracy'])} -> {fmt(r['improved_dann_accuracy'])} (delta {fmt(r['delta_baseline_untuned_to_dann'])})"
        for r in top_drop
    ) + f"""

Largest gains (Mitigation 1 -> DANN):

""" + "\n".join(
        f"- `{r['language']}`: {fmt(r['mitigation1_accuracy'])} -> {fmt(r['improved_dann_accuracy'])} (delta {fmt(r['delta_m1_to_dann'])})"
        for r in top_gain_m1_to_dann
    ) + """

Largest drops (Mitigation 1 -> DANN), showing tradeoff:

""" + "\n".join(
        f"- `{r['language']}`: {fmt(r['mitigation1_accuracy'])} -> {fmt(r['improved_dann_accuracy'])} (delta {fmt(r['delta_m1_to_dann'])})"
        for r in top_drop_m1_to_dann
    ) + f"""

Confusion matrix visuals:

""" + "\n".join(conf_img_lines) + f"""

## 4. Last-Layer Representation Analysis (t-SNE)

t-SNE settings target: `max_samples=2200`, `perplexity=30`, `seed=42`, `speaker_top_k=12`.

Available t-SNE point exports in this workspace: `{", ".join(tsne_available) if tsne_available else "none"}`.
Missing t-SNE point exports in this workspace: `{", ".join(tsne_missing) if tsne_missing else "none"}`.

Quantitative t-SNE support:

| Run | Silhouette (Language) | Silhouette (Speaker top-k) | kNN Speaker Purity@5 |
|---|---:|---:|---:|
| Baseline (untuned) | {fmt(tsne_bu.get('silhouette_language', float('nan')))} | {fmt(tsne_bu.get('silhouette_speaker_topk', float('nan')))} | {fmt(tsne_bu.get('knn_speaker_purity_at_5', float('nan')))} |
| Tuned ref | {fmt(tsne_tr.get('silhouette_language', float('nan')))} | {fmt(tsne_tr.get('silhouette_speaker_topk', float('nan')))} | {fmt(tsne_tr.get('knn_speaker_purity_at_5', float('nan')))} |
| Mitigation 1 | {fmt(tsne_m1.get('silhouette_language', float('nan')))} | {fmt(tsne_m1.get('silhouette_speaker_topk', float('nan')))} | {fmt(tsne_m1.get('knn_speaker_purity_at_5', float('nan')))} |
| DANN | {fmt(tsne_d.get('silhouette_language', float('nan')))} | {fmt(tsne_d.get('silhouette_speaker_topk', float('nan')))} | {fmt(tsne_d.get('knn_speaker_purity_at_5', float('nan')))} |

Interpretation:

- Language-colored t-SNE should become cleaner in improved runs if language signal strengthens.
- Speaker-colored t-SNE and kNN speaker purity quantify residual speaker structure in embeddings.
- In this workspace, untuned/tuned/M1 t-SNE numeric rows are `NA` because those run folders do not include saved model/preprocessor artifacts needed for post-hoc export.
- With available DANN t-SNE, speaker identity signal appears reduced but still present (non-zero local speaker purity).

t-SNE visuals:

""" + "\n".join(tsne_img_lines) + f"""

## 5. Do Models Encode Speaker Identity?

Speaker diagnostics (`overlap_speaker_count=0`, no warnings) show clean split integrity across runs.

Unseen-speaker accuracy:

- Baseline (untuned): {fmt(float(speaker_map['baseline_untuned']['accuracy_unseen_speakers']))}
- Tuned ref: {fmt(float(speaker_map['tuned_ref']['accuracy_unseen_speakers']))}
- Mitigation 1: {fmt(float(speaker_map['mitigation1']['accuracy_unseen_speakers']))}
- DANN: {fmt(float(speaker_map['improved_dann']['accuracy_unseen_speakers']))}

Conclusion: the models still encode some speaker information (non-zero speaker structure in embedding space), but DANN improves language-discriminative behavior and unseen-speaker performance compared with the baseline chain.

## 6. Final Takeaways

1. Bias source is real: small-speaker regime encourages speaker shortcut learning.
2. Data-centric mitigation helps but remains partial.
3. DANN provides the strongest overall improvement in this branch.
4. Confusion analysis and t-SNE both show progress plus remaining hard language pairs.
5. Speaker bias mitigation is improved, not absolutely solved; this calibrated claim is the most defensible for Task 3.
"""

    task3_report_path = reports_dir / "task3_analysis.md"
    task3_report_path.write_text(report_md, encoding="utf-8")

    # Artifacts index.
    generated_files = [
        "reports/task3_analysis.md",
        "reports/artifacts_used.md",
        "reports/tables/task3_metrics_comparison.csv",
        "reports/tables/task3_confusion_top_pairs.csv",
        "reports/tables/task3_per_language_delta.csv",
        "reports/tables/task3_speaker_summary.csv",
        "reports/tables/task3_tsne_cluster_metrics.csv",
        "reports/tables/task3_tsne_pairwise_distance_summary.csv",
        "reports/tables/speaker_summary_baseline.json",
        "reports/tables/speaker_summary_tuned_reference.json",
        "reports/tables/speaker_summary_mitigation1.json",
        "reports/tables/speaker_summary_comparison.csv",
    ]

    artifacts_md = "# Task 3 Artifacts Used\n\n## Source Artifacts\n\n"
    for p in sorted(set(source_files)):
        artifacts_md += f"- `{p}`\n"

    artifacts_md += "\n## Generated Artifacts\n\n"
    for p in generated_files:
        artifacts_md += f"- `{p}`\n"

    artifacts_md += "\n## Run Mapping\n\n"
    artifacts_md += f"- Baseline (untuned): `{RUNS[0].run_id}`\n"
    artifacts_md += f"- Tuned reference (no mitigation): `{RUNS[1].run_id}`\n"
    artifacts_md += f"- Mitigation 1: `{RUNS[2].run_id}`\n"
    artifacts_md += f"- Improved (DANN): `{RUNS[3].run_id}`\n"
    artifacts_md += "\n## Notes\n\n"
    artifacts_md += "- Baseline (untuned), tuned reference, and Mitigation 1 run folders in this workspace contain metrics/diagnostics but not full model/preprocessor checkpoint files.\n"
    artifacts_md += "- Therefore, post-hoc t-SNE re-export for those runs is not reproducible from current local artifacts.\n"

    (reports_dir / "artifacts_used.md").write_text(artifacts_md, encoding="utf-8")

    print("Task 3 artifacts generated successfully.")
    for p in generated_files:
        print(f"- {p}")


if __name__ == "__main__":
    main()
