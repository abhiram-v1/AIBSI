from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "raw_cnn_svm_fusion"
MODERATE = HERE / "outputs" / "moderate_log_mel" / "prepared"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
from evaluate_site_normalized import direct_model, site_normalize  # noqa: E402


def temperature_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.maximum(probabilities, 1e-9)) / temperature, axis=1)


def scores(true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "macro_f1": float(f1_score(true, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tensor = np.load(MODERATE / "log_mel.npy", mmap_mode="r")
    labels_windows = np.load(MODERATE / "labels.npy", mmap_mode="r")
    subject_index = np.load(MODERATE / "subject_index.npy", mmap_mode="r")
    participants = json.loads((MODERATE / "manifest.json").read_text(encoding="utf-8"))["participants"]
    features, labels = eeg.subject_features(tensor, labels_windows, subject_index, participants)
    flat = features.reshape(len(features), -1)
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    validation = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)

    config = {"pca": 20, "C": 0.1, "kernel": "linear"}
    normalized_validation = site_normalize(flat, participants, train)
    validation_model = direct_model(config)
    validation_model.fit(normalized_validation[train], labels[train])
    svm_validation_scores = validation_model.decision_function(normalized_validation[validation])

    train_validation = np.concatenate([train, validation])
    normalized_test = site_normalize(flat, participants, train_validation)
    test_model = direct_model(config)
    test_model.fit(normalized_test[train_validation], labels[train_validation])
    svm_test_scores = test_model.decision_function(normalized_test[test])

    window = np.load(
        HERE / "outputs" / "raw_1d_cnn_selected" / "participant_probabilities.npz"
    )
    mil = np.load(HERE / "outputs" / "raw_1d_cnn_mil" / "participant_probabilities.npz")
    if not np.array_equal(window["validation_indices"], validation):
        raise RuntimeError("Window CNN validation participant order differs")
    if not np.array_equal(mil["validation_indices"], validation):
        raise RuntimeError("MIL CNN validation participant order differs")
    if not np.array_equal(window["test_indices"], test) or not np.array_equal(mil["test_indices"], test):
        raise RuntimeError("Test participant order differs")

    candidates = []
    temperatures = [0.5, 0.75, 1.0, 1.5, 2.0]
    svm_temperatures = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    weight_values = np.arange(0.0, 1.01, 0.1)
    for svm_temperature in svm_temperatures:
        svm_probabilities = softmax(svm_validation_scores / svm_temperature, axis=1)
        for window_temperature in temperatures:
            window_probabilities = temperature_probabilities(
                window["selected_validation_probabilities"], window_temperature
            )
            for mil_temperature in temperatures:
                mil_probabilities = temperature_probabilities(
                    mil["validation_probabilities"], mil_temperature
                )
                for svm_weight in weight_values:
                    for window_weight in weight_values:
                        mil_weight = 1.0 - svm_weight - window_weight
                        if mil_weight < -1e-9:
                            continue
                        probabilities = (
                            svm_weight * svm_probabilities
                            + window_weight * window_probabilities
                            + mil_weight * mil_probabilities
                        )
                        metric = scores(labels[validation], probabilities)
                        candidates.append(
                            {
                                "svm_temperature": svm_temperature,
                                "window_temperature": window_temperature,
                                "mil_temperature": mil_temperature,
                                "svm_weight": float(svm_weight),
                                "window_weight": float(window_weight),
                                "mil_weight": float(mil_weight),
                                **metric,
                            }
                        )
    candidates.sort(
        key=lambda row: (row["macro_f1"], row["balanced_accuracy"], row["accuracy"]),
        reverse=True,
    )
    selected = candidates[0]

    svm_test_probabilities = softmax(svm_test_scores / selected["svm_temperature"], axis=1)
    window_test_probabilities = temperature_probabilities(
        window["selected_test_probabilities"], selected["window_temperature"]
    )
    mil_test_probabilities = temperature_probabilities(
        mil["test_probabilities"], selected["mil_temperature"]
    )
    test_probabilities = (
        selected["svm_weight"] * svm_test_probabilities
        + selected["window_weight"] * window_test_probabilities
        + selected["mil_weight"] * mil_test_probabilities
    )
    test_metrics = eeg.metrics(labels[test], test_probabilities.argmax(axis=1))

    payload = {
        "selection": "temperatures and three-model probability weights selected by validation macro F1",
        "test_used_for_selection": False,
        "selected_validation": selected,
        "test": test_metrics,
        "top_validation_candidates": candidates[:50],
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "probabilities.npz",
        validation_indices=validation,
        validation_true=labels[validation],
        test_indices=test,
        test_true=labels[test],
        test_probabilities=test_probabilities,
    )

    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Validation-selected raw CNN and SVM fusion")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    references = json.loads(
        (HERE / "outputs" / "raw_1d_cnn_mil" / "metrics.json").read_text(encoding="utf-8")
    )
    earlier = json.loads(
        (HERE / "outputs" / "raw_1d_cnn_selected" / "metrics.json").read_text(encoding="utf-8")
    )["selected_test"]
    rows = [earlier, references["selected_test"], test_metrics]
    names = ["Window CNN", "Participant MIL CNN", "Validation-selected fusion"]
    measures = ["accuracy", "balanced_accuracy", "macro_f1"]
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    for offset, measure in enumerate(measures):
        bars = ax.bar(
            positions + (offset - 1) * 0.23,
            [row[measure] for row in rows],
            0.23,
            label=measure.replace("_", " ").title(),
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(positions, names)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Raw representation model fusion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)

    recalls = {name: test_metrics["per_class"][name]["recall"] for name in eeg.CLASSES}
    report = [
        "# Validation-selected raw CNN and log-mel SVM fusion",
        "",
        "Temperatures and weights were selected solely on the 26 validation participants. "
        "The frozen configuration was then applied to the 27 held-out participants.",
        "",
        "## Selected validation configuration",
        "",
        f"- SVM / window CNN / MIL CNN weights: {selected['svm_weight']:.2f} / {selected['window_weight']:.2f} / {selected['mil_weight']:.2f}",
        f"- Validation accuracy: {selected['accuracy']:.4f}",
        f"- Validation balanced accuracy: {selected['balanced_accuracy']:.4f}",
        f"- Validation macro F1: {selected['macro_f1']:.4f}",
        "",
        "## Held-out participant result",
        "",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
        f"- Macro F1: {test_metrics['macro_f1']:.4f}",
        f"- Recall AD / FTD / HC: {recalls['AD']:.4f} / {recalls['FTD']:.4f} / {recalls['HC']:.4f}",
        f"- Confusion matrix: {test_metrics['confusion_matrix']}",
        "",
        "The fixed test cohort has been inspected repeatedly, so this remains exploratory even though the fusion rule never used test labels.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected_validation": selected, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
