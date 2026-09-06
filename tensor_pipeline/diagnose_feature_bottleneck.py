"""Test whether aggressive unsupervised compression is the main bottleneck."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(r"C:\Projects\AIBSI\tensor_pipeline\outputs\tensorization")
OUT = Path(r"C:\Projects\AIBSI\tensor_pipeline\outputs\feature_bottleneck_diagnostic")


def representations(tensor: np.ndarray, frequencies: np.ndarray) -> dict[str, np.ndarray]:
    bands = [(1, 4), (4, 8), (8, 13), (13, 30.01)]
    band_tensor = np.stack(
        [tensor[:, :, (frequencies >= lo) & (frequencies < hi), :].mean(axis=2) for lo, hi in bands],
        axis=2,
    )
    band_summary = np.concatenate(
        [band_tensor.mean(axis=3).reshape(len(tensor), -1), band_tensor.std(axis=3).reshape(len(tensor), -1)],
        axis=1,
    )
    return {
        "full_tensor_flat": tensor.reshape(len(tensor), -1),
        "channel_band_time": band_tensor.reshape(len(tensor), -1),
        "channel_band_mean_std": band_summary,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tensor = np.load(ROOT / "tensor_relative_sqrt_1_30hz.npy")
    frequencies = np.load(ROOT / "frequencies_hz.npy")
    labels = np.load(ROOT / "labels.npy")
    reps = representations(tensor, frequencies)
    splitter = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=5040).split(tensor, labels)
    )
    rows = []
    predictions = []
    for representation, features in reps.items():
        for k in [10, 20, 40]:
            if k > features.shape[1]:
                continue
            for classifier_name in ["logistic_regression", "linear_svm"]:
                repeat_scores = []
                repeat_f1 = []
                for repeat in range(3):
                    true = []
                    pred = []
                    for split_i in range(repeat * 5, repeat * 5 + 5):
                        train, test = splitter[split_i]
                        if classifier_name == "logistic_regression":
                            model = LogisticRegression(
                                C=0.1, class_weight="balanced", max_iter=5000, random_state=7000 + split_i
                            )
                        else:
                            model = SVC(
                                C=0.1, kernel="linear", class_weight="balanced", random_state=7000 + split_i
                            )
                        pipeline = make_pipeline(
                            SelectKBest(f_classif, k=k),
                            StandardScaler(),
                            model,
                        )
                        pipeline.fit(features[train], labels[train])
                        fold_pred = pipeline.predict(features[test])
                        true.extend(labels[test].tolist())
                        pred.extend(fold_pred.tolist())
                        for subject_i, prediction in zip(test, fold_pred):
                            predictions.append(
                                {
                                    "representation": representation,
                                    "selected_features": k,
                                    "classifier": classifier_name,
                                    "repeat": repeat,
                                    "subject_index": int(subject_i),
                                    "true_label": int(labels[subject_i]),
                                    "predicted_label": int(prediction),
                                }
                            )
                    repeat_scores.append(balanced_accuracy_score(true, pred))
                    repeat_f1.append(f1_score(true, pred, average="macro"))
                rows.append(
                    {
                        "representation": representation,
                        "input_features": features.shape[1],
                        "selected_features": k,
                        "classifier": classifier_name,
                        "balanced_accuracy_mean": float(np.mean(repeat_scores)),
                        "balanced_accuracy_sd": float(np.std(repeat_scores, ddof=1)),
                        "macro_f1_mean": float(np.mean(repeat_f1)),
                        "macro_f1_sd": float(np.std(repeat_f1, ddof=1)),
                    }
                )
    rows.sort(key=lambda row: row["balanced_accuracy_mean"], reverse=True)
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "purpose": "diagnose whether rank-3/5/8 unsupervised subject factors discard discriminative information",
        "validation": "3x repeated stratified 5-fold subject-level CV; feature selection fitted inside every training fold",
        "results": rows,
        "best": rows[0],
        "interpretation_rule": "A material improvement over 0.541 indicates that representation compression, rather than the classifier alone, is a major bottleneck.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows[:12]:
        print(row)


if __name__ == "__main__":
    main()
