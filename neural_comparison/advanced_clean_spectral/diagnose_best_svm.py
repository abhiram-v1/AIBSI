from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "results" / "svm_diagnostics"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
from evaluate_site_normalized import site_normalize  # noqa: E402


def fit_predict(
    flat: np.ndarray,
    labels: np.ndarray,
    participants: list[dict],
    fit: np.ndarray,
    predict: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized = site_normalize(flat, participants, fit)
    scaler = StandardScaler().fit(normalized[fit])
    fit_scaled = scaler.transform(normalized[fit])
    predict_scaled = scaler.transform(normalized[predict])
    pca = PCA(n_components=20, random_state=eeg.SEED).fit(fit_scaled)
    fit_reduced = pca.transform(fit_scaled)
    predict_reduced = pca.transform(predict_scaled)
    model = SVC(
        C=0.1,
        kernel="linear",
        class_weight="balanced",
        decision_function_shape="ovr",
        random_state=eeg.SEED,
    ).fit(fit_reduced, labels[fit])
    return model.predict(predict_reduced), model.decision_function(predict_reduced), fit_reduced, predict_reduced


def subset_metrics(labels: np.ndarray, prediction: np.ndarray, indexes: np.ndarray) -> dict:
    true = labels[indexes]
    pred = prediction
    return {
        "subjects": int(len(indexes)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1_present_classes": float(f1_score(true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=np.arange(3)).tolist(),
    }


def corrected_metrics(matrix: np.ndarray) -> list[dict]:
    rows = []
    true = []
    pred = []
    for row in range(3):
        for column in range(3):
            true.extend([row] * int(matrix[row, column]))
            pred.extend([column] * int(matrix[row, column]))
    true = np.asarray(true)
    pred = np.asarray(pred)
    wrong = np.flatnonzero(true != pred)
    from itertools import combinations

    for count in range(1, len(wrong) + 1):
        best = None
        for selected in combinations(wrong, count):
            candidate = pred.copy()
            candidate[list(selected)] = true[list(selected)]
            row = {
                "corrected_errors": count,
                "accuracy": float(accuracy_score(true, candidate)),
                "macro_f1": float(f1_score(true, candidate, average="macro", zero_division=0)),
            }
            if best is None or (row["macro_f1"], row["accuracy"]) > (best["macro_f1"], best["accuracy"]):
                best = row
        rows.append(best)
        if best["accuracy"] >= 0.70 and best["macro_f1"] >= 0.70:
            break
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tensor = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    features, labels = eeg.subject_features(tensor, window_labels, subject_index, participants)
    flat = features.reshape(len(features), -1)
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, val])

    val_prediction, val_scores, _, _ = fit_predict(flat, labels, participants, train, val)
    test_prediction, test_scores, dev_reduced, test_reduced = fit_predict(flat, labels, participants, development, test)
    val_metrics = eeg.metrics(labels[val], val_prediction)
    test_metrics = eeg.metrics(labels[test], test_prediction)

    order = np.argsort(test_scores, axis=1)
    margins = test_scores[np.arange(len(test)), order[:, -1]] - test_scores[np.arange(len(test)), order[:, -2]]
    centroids = np.stack([dev_reduced[labels[development] == label].mean(axis=0) for label in range(3)])
    distances = np.linalg.norm(test_reduced[:, None, :] - centroids[None, :, :], axis=2)
    nearest_centroid = np.argmin(distances, axis=1)

    rows = []
    for local, subject in enumerate(test):
        participant = participants[int(subject)]
        rows.append({
            "subject_id": participant["subject_id"],
            "site": participant["site"],
            "source": participant["source"],
            "true": eeg.CLASSES[int(labels[subject])],
            "predicted": eeg.CLASSES[int(test_prediction[local])],
            "correct": bool(labels[subject] == test_prediction[local]),
            "decision_margin": float(margins[local]),
            "nearest_training_centroid": eeg.CLASSES[int(nearest_centroid[local])],
            "distance_to_true_centroid": float(distances[local, labels[subject]]),
            "windows": int(participant["windows"]),
        })

    by_site = {}
    for site in sorted({row["site"] for row in rows}):
        local = np.asarray([index for index, row in enumerate(rows) if row["site"] == site])
        by_site[site] = subset_metrics(labels[test[local]], test_prediction[local], np.arange(len(local)))
        by_site[site]["correct"] = int(sum(rows[index]["correct"] for index in local))
    by_class = {}
    for label, name in enumerate(eeg.CLASSES):
        local = np.flatnonzero(labels[test] == label)
        by_class[name] = {
            "subjects": int(len(local)),
            "correct": int(np.sum(test_prediction[local] == label)),
            "recall": float(np.mean(test_prediction[local] == label)),
            "mean_margin": float(margins[local].mean()),
        }

    correct_mask = labels[test] == test_prediction
    diagnostics = {
        "validation": val_metrics,
        "test": test_metrics,
        "test_split_counts": dict(Counter(f"{row['site']}:{row['true']}" for row in rows)),
        "by_site": by_site,
        "by_class": by_class,
        "confidence": {
            "mean_margin_correct": float(margins[correct_mask].mean()),
            "mean_margin_wrong": float(margins[~correct_mask].mean()),
            "high_confidence_wrong_at_or_above_median": int(np.sum((~correct_mask) & (margins >= np.median(margins)))),
        },
        "centroid_agreement": {
            "nearest_centroid_matches_true": float(np.mean(nearest_centroid == labels[test])),
            "nearest_centroid_matches_svm": float(np.mean(nearest_centroid == test_prediction)),
        },
        "counterfactual": corrected_metrics(np.asarray(test_metrics["confusion_matrix"])),
        "participants": rows,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in diagnostics.items() if key != "participants"}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    class_recall = [by_class[name]["recall"] for name in eeg.CLASSES]
    bars = axes[0].bar(eeg.CLASSES, class_recall, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].bar_label(bars, fmt="%.2f", padding=2)
    axes[0].set(ylim=(0, 1), ylabel="Recall", title="Recall by diagnosis")
    axes[0].grid(axis="y", alpha=0.25)

    site_names = list(by_site)
    site_accuracy = [by_site[name]["accuracy"] for name in site_names]
    bars = axes[1].bar(site_names, site_accuracy, color="#4c78a8")
    axes[1].bar_label(bars, fmt="%.2f", padding=2)
    axes[1].set(ylim=(0, 1), ylabel="Accuracy", title="Accuracy by acquisition site")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].boxplot([margins[correct_mask], margins[~correct_mask]], tick_labels=["Correct", "Wrong"])
    axes[2].set(ylabel="Top-two SVM decision margin", title="Prediction confidence")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "failure_analysis.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
