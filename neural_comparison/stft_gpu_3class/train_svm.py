from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "prepared"
CNN_RUN = HERE / "outputs" / "run"
OUT = HERE / "outputs" / "svm"
CLASS_NAMES = ["AD", "FTD", "HC"]


def subject_features(
    x: np.ndarray, y: np.ndarray, subject_index: np.ndarray, participants: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize each participant's STFT without hand-crafted EEG bands."""
    features = []
    labels = []
    for subject in range(len(participants)):
        windows = x[subject_index == subject].astype(np.float32)
        # Retain the full 1--40 Hz resolution. Pool only repeated windows and
        # within-window STFT frames, then include variability across both.
        flattened_repeats = windows.transpose(0, 3, 1, 2).reshape(-1, windows.shape[1], windows.shape[2])
        mean_spectrum = flattened_repeats.mean(axis=0)
        std_spectrum = flattened_repeats.std(axis=0)
        features.append(np.concatenate([mean_spectrum.ravel(), std_spectrum.ravel()]))
        labels.append(int(y[np.flatnonzero(subject_index == subject)[0]]))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def calculate_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=np.arange(3), zero_division=0
    )
    return {
        "subjects": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(true, pred, labels=np.arange(3)).tolist(),
    }


def make_model(c: float, gamma: str | float, components: int, kernel: str) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=components, random_state=20261007)),
            (
                "svm",
                SVC(
                    C=c,
                    gamma=gamma,
                    kernel=kernel,
                    class_weight="balanced",
                    probability=True,
                    random_state=20261007,
                ),
            ),
        ]
    )


def plot_results(metrics: dict, comparison: dict) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="STFT-SVM participant test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    measures = ["precision", "recall", "f1"]
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for offset, measure in enumerate(measures):
        values = [metrics["per_class"][name][measure] for name in CLASS_NAMES]
        ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=measure.title())
    ax.set_xticks(positions, CLASS_NAMES)
    ax.set(ylim=(0, 1), ylabel="Score", title="STFT-SVM participant metrics by class")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "per_class_metrics.png", dpi=180)
    plt.close(fig)

    names = list(comparison)
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    x_pos = np.arange(len(names))
    for offset, measure in enumerate(measures):
        ax.bar(
            x_pos + (offset - 1) * 0.24,
            [comparison[name][measure] for name in names],
            0.24,
            label=measure.replace("_", " ").title(),
        )
    ax.set_xticks(x_pos, names)
    ax.set(ylim=(0, 1), ylabel="Participant-level score", title="Same split: GPU CNN and SVM")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(PREPARED / "spectrograms.npy", mmap_mode="r")
    y_windows = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = json.loads((CNN_RUN / "splits.json").read_text(encoding="utf-8"))
    features, labels = subject_features(x, y_windows, subject_index, participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)

    candidates = []
    for kernel in ["linear", "rbf"]:
        gammas: list[str | float] = ["scale"] if kernel == "linear" else ["scale", 0.001, 0.01]
        for components in [20, 40, 80]:
            for c in [0.1, 1.0, 10.0, 100.0]:
                for gamma in gammas:
                    model = make_model(c, gamma, components, kernel)
                    model.fit(features[train], labels[train])
                    pred = model.predict(features[val])
                    candidates.append(
                        {
                            "kernel": kernel,
                            "components": components,
                            "C": c,
                            "gamma": gamma,
                            "val_macro_f1": float(f1_score(labels[val], pred, average="macro", zero_division=0)),
                            "val_accuracy": float(accuracy_score(labels[val], pred)),
                        }
                    )
    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_accuracy"]), reverse=True)
    best = candidates[0]
    print("Best validation configuration:", best)

    train_val = np.concatenate([train, val])
    final_model = make_model(best["C"], best["gamma"], best["components"], best["kernel"])
    final_model.fit(features[train_val], labels[train_val])
    probabilities = final_model.predict_proba(features[test])
    prediction = final_model.classes_[np.argmax(probabilities, axis=1)]
    metrics = calculate_metrics(labels[test], prediction)
    payload = {"selected_on_validation": best, "test": metrics, "candidates": candidates}
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (OUT / "subject_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["subject_id", "source", "site", "true_label", "predicted_label", "prob_AD", "prob_FTD", "prob_HC", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, subject in enumerate(test):
            true = int(labels[subject])
            pred = int(prediction[position])
            participant = participants[int(subject)]
            probability_by_class = {int(cls): float(probabilities[position, i]) for i, cls in enumerate(final_model.classes_)}
            writer.writerow(
                {
                    "subject_id": participant["subject_id"],
                    "source": participant["source"],
                    "site": participant["site"],
                    "true_label": CLASS_NAMES[true],
                    "predicted_label": CLASS_NAMES[pred],
                    "prob_AD": probability_by_class.get(0, 0.0),
                    "prob_FTD": probability_by_class.get(1, 0.0),
                    "prob_HC": probability_by_class.get(2, 0.0),
                    "correct": true == pred,
                }
            )

    cnn_metrics = json.loads((CNN_RUN / "metrics.json").read_text(encoding="utf-8"))["test"]
    comparison = {"GPU CNN": cnn_metrics, "STFT SVM": metrics}
    plot_results(metrics, comparison)
    print(json.dumps(metrics, indent=2))
    print(f"Saved SVM artifacts to {OUT}")


if __name__ == "__main__":
    main()
