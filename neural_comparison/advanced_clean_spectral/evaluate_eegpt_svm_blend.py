from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MOD = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "deep_models" / "eegpt_attention"
sys.path.insert(0, str(HERE))
import evaluate_site_normalized as site_svm  # noqa: E402
import pipeline as eeg  # noqa: E402


def svm_scores(
    features: np.ndarray,
    labels: np.ndarray,
    participants: list[dict],
    fit: np.ndarray,
    predict: np.ndarray,
    config: dict,
) -> np.ndarray:
    normalized = site_svm.site_normalize(features, participants, fit)
    model = site_svm.direct_model(config)
    model.fit(normalized[fit], labels[fit])
    return model.decision_function(normalized[predict])


def main() -> None:
    tensor = np.load(MOD / "log_mel.npy", mmap_mode="r")
    window_labels = np.load(MOD / "labels.npy", mmap_mode="r")
    subject_index = np.load(MOD / "subject_index.npy", mmap_mode="r")
    participants = json.loads((MOD / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    features, labels = eeg.subject_features(tensor, window_labels, subject_index, participants)
    features = features.reshape(len(features), -1)

    selected = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["selected_on_validation"]
    val_svm = svm_scores(features, labels, participants, train, val, selected["config"])
    test_svm = svm_scores(features, labels, participants, np.concatenate([train, val]), test, selected["config"])
    eegpt = np.load(OUT / "probabilities.npz")
    if not np.array_equal(eegpt["val_true"], labels[val]) or not np.array_equal(eegpt["test_true"], labels[test]):
        raise RuntimeError("EEGPT and SVM participant ordering differs")

    candidates = []
    for svm_temperature in [0.5, 0.75, 1.0, 1.5, 2.0]:
        svm_probability = softmax(val_svm / svm_temperature, axis=1)
        for eegpt_temperature in [0.75, 1.0, 1.25, 1.5]:
            eegpt_probability = softmax(np.log(np.maximum(eegpt["val"], 1e-7)) / eegpt_temperature, axis=1)
            for svm_weight in np.linspace(0.0, 1.0, 21):
                probability = svm_weight * svm_probability + (1.0 - svm_weight) * eegpt_probability
                prediction = np.argmax(probability, axis=1)
                candidates.append({
                    "svm_temperature": svm_temperature,
                    "eegpt_temperature": eegpt_temperature,
                    "svm_weight": float(svm_weight),
                    "val_macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
                    "val_accuracy": float(accuracy_score(labels[val], prediction)),
                })
    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_accuracy"], -abs(row["svm_weight"] - 0.5)), reverse=True)
    best = candidates[0]
    svm_probability = softmax(test_svm / best["svm_temperature"], axis=1)
    eegpt_probability = softmax(np.log(np.maximum(eegpt["test"], 1e-7)) / best["eegpt_temperature"], axis=1)
    probability = best["svm_weight"] * svm_probability + (1.0 - best["svm_weight"]) * eegpt_probability
    metrics = eeg.metrics(labels[test], np.argmax(probability, axis=1))
    payload = {"selection": best, "test": metrics, "validation_candidates": candidates}
    (OUT / "svm_blend_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    svm_metrics = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    cnn_metrics = json.loads((HERE / "outputs" / "deep_models" / "attention_pooling" / "metrics.json").read_text(encoding="utf-8"))["ensemble_test"]
    eegpt_metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))["test"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    matrix = np.asarray(metrics["confusion_matrix"])
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="SVM + EEGPT blend: held-out participants")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    names = ["SVM", "CNN +\nattention", "EEGPT fine\ntune", "SVM +\nEEGPT"]
    model_metrics = [svm_metrics, cnn_metrics, eegpt_metrics, metrics]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    positions = np.arange(len(names))
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            positions + (offset - 1) * 0.24,
            [item[measure] for item in model_metrics],
            width=0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axes[1].set_xticks(positions, names)
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Same fixed participant split")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.savefig(OUT / "svm_eegpt_blend_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"selection": best, "test": metrics}, indent=2))


if __name__ == "__main__":
    main()
