from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "results" / "svm_calibration"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
from evaluate_site_normalized import site_normalize  # noqa: E402


def model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=eeg.SEED)),
        ("svm", SVC(C=0.1, kernel="linear", class_weight="balanced", decision_function_shape="ovr", random_state=eeg.SEED)),
    ])


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

    normalized = site_normalize(flat, participants, train)
    validation_model = model().fit(normalized[train], labels[train])
    val_scores = validation_model.decision_function(normalized[val])
    grid = np.linspace(-0.75, 0.75, 61)
    candidates = []
    for ad_bias, ftd_bias in itertools.product(grid, grid):
        bias = np.asarray([ad_bias, ftd_bias, 0.0])
        prediction = np.argmax(val_scores + bias, axis=1)
        candidates.append({
            "bias": bias.tolist(),
            "val_macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
            "val_accuracy": float(accuracy_score(labels[val], prediction)),
            "magnitude": float(np.linalg.norm(bias)),
        })
    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_accuracy"], -row["magnitude"]), reverse=True)
    best = candidates[0]

    development = np.concatenate([train, val])
    normalized = site_normalize(flat, participants, development)
    final_model = model().fit(normalized[development], labels[development])
    uncalibrated_prediction = final_model.predict(normalized[test])
    calibrated_prediction = np.argmax(final_model.decision_function(normalized[test]) + np.asarray(best["bias"]), axis=1)
    uncalibrated = eeg.metrics(labels[test], uncalibrated_prediction)
    calibrated = eeg.metrics(labels[test], calibrated_prediction)
    payload = {
        "method": "class decision offsets selected only on validation macro F1",
        "selected_on_validation": best,
        "uncalibrated_test": uncalibrated,
        "calibrated_test": calibrated,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    matrix = np.asarray(calibrated["confusion_matrix"])
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="Calibrated site-normalized SVM")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    measures = ["accuracy", "macro_f1", "macro_recall"]
    positions = np.arange(2)
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            positions + (offset - 1) * 0.24,
            [uncalibrated[measure], calibrated[measure]],
            0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2)
    axes[1].set_xticks(positions, ["Uncalibrated", "Validation-calibrated"])
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Effect of decision calibration")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
