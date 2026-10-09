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


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ADV = HERE / "outputs" / "prepared"
MOD = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "results"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


def site_normalize(features: np.ndarray, participants: list[dict], fit: np.ndarray) -> np.ndarray:
    result = np.empty_like(features, dtype=np.float32)
    sites = np.asarray([participant["site"] for participant in participants])
    for site in sorted(set(sites)):
        reference = features[fit[sites[fit] == site]]
        mean = reference.mean(axis=0, keepdims=True)
        std = np.maximum(reference.std(axis=0, keepdims=True), 1e-3)
        result[sites == site] = (features[sites == site] - mean) / std
    return result


def direct_model(config: dict) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if config["pca"] is not None:
        steps.append(("pca", PCA(n_components=config["pca"], random_state=eeg.SEED)))
    steps.append(
        (
            "svm",
            SVC(
                C=config["C"], kernel=config["kernel"], gamma="scale",
                class_weight="balanced", random_state=eeg.SEED,
            ),
        )
    )
    return Pipeline(steps)


def main() -> None:
    labels_windows = np.load(ADV / "labels.npy", mmap_mode="r")
    subject_index = np.load(ADV / "subject_index.npy", mmap_mode="r")
    participants = json.loads((ADV / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    configs = [
        {"pca": pca, "C": c, "kernel": kernel}
        for pca, c, kernel in itertools.product([None, 20, 40, 60], [0.01, 0.1, 1.0, 10.0], ["linear", "rbf"])
    ]
    candidates = []
    representation_data = {}
    for representation, prepared in [("advanced_log_mel", ADV), ("moderate_log_mel", MOD)]:
        tensor = np.load(prepared / "log_mel.npy", mmap_mode="r")
        if prepared == MOD:
            local_labels = np.load(prepared / "labels.npy", mmap_mode="r")
            local_subject = np.load(prepared / "subject_index.npy", mmap_mode="r")
            local_participants = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))["participants"]
        else:
            local_labels, local_subject, local_participants = labels_windows, subject_index, participants
        if [item["subject_id"] for item in local_participants] != [item["subject_id"] for item in participants]:
            raise RuntimeError(f"Participant order differs for {representation}")
        features, labels = eeg.subject_features(tensor, local_labels, local_subject, local_participants)
        flat = features.reshape(len(features), -1)
        normalized = site_normalize(flat, local_participants, train)
        representation_data[representation] = (flat, labels, local_participants)
        hierarchy_prediction = eeg.hierarchical_fit_predict(normalized[train], labels[train], normalized[val])
        candidates.append(
            {
                "representation": representation,
                "model": "hierarchical",
                "config": None,
                "val_macro_f1": float(f1_score(labels[val], hierarchy_prediction, average="macro", zero_division=0)),
                "val_accuracy": float(accuracy_score(labels[val], hierarchy_prediction)),
            }
        )
        for config in configs:
            model = direct_model(config)
            model.fit(normalized[train], labels[train])
            prediction = model.predict(normalized[val])
            candidates.append(
                {
                    "representation": representation,
                    "model": "direct",
                    "config": config,
                    "val_macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
                    "val_accuracy": float(accuracy_score(labels[val], prediction)),
                }
            )
    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_accuracy"]), reverse=True)
    best = candidates[0]
    flat, labels, selected_participants = representation_data[best["representation"]]
    train_val = np.concatenate([train, val])
    normalized = site_normalize(flat, selected_participants, train_val)
    if best["model"] == "hierarchical":
        prediction = eeg.hierarchical_fit_predict(normalized[train_val], labels[train_val], normalized[test])
    else:
        model = direct_model(best["config"])
        model.fit(normalized[train_val], labels[train_val])
        prediction = model.predict(normalized[test])
    test_metrics = eeg.metrics(labels[test], prediction)
    payload = {"selected_on_validation": best, "test": test_metrics, "validation_candidates": candidates}
    (OUT / "site_normalized_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_on_validation": best, "test": test_metrics}, indent=2))

    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title="Training-only site-normalized spectral model")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "site_normalized_confusion.png", dpi=180)
    plt.close(fig)

    earlier = json.loads(
        (ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "hierarchical_svm" / "metrics.json").read_text(encoding="utf-8")
    )["test"]
    cleaned_cnn = json.loads(
        (HERE / "outputs" / "deep_models" / "metrics.json").read_text(encoding="utf-8")
    )["results"]["log_mel"]["cnn"]["test"]
    moderate_ensemble = json.loads(
        (HERE / "outputs" / "deep_models" / "moderate_log_mel" / "metrics.json").read_text(encoding="utf-8")
    )["test_calibrated"]
    comparison_rows = [earlier, cleaned_cnn, moderate_ensemble, test_metrics]
    comparison_labels = [
        "Earlier STFT\nhierarchical SVM",
        "Advanced-cleaned\nlog-mel CNN",
        "Moderate-cleaned\nlog-mel CNN ensemble",
        "Site-normalized\nlog-mel SVM",
    ]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    positions = np.arange(len(comparison_rows))
    for offset, measure in enumerate(measures):
        bars = ax.bar(
            positions + (offset - 1) * 0.24,
            [row[measure] for row in comparison_rows],
            0.24,
            label=measure.replace("_", " ").title(),
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(positions, comparison_labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Effect of cleaning and acquisition-site normalization")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "site_normalized_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    class_positions = np.arange(3)
    recalls = [test_metrics["per_class"][name]["recall"] for name in eeg.CLASSES]
    f1_values = [test_metrics["per_class"][name]["f1"] for name in eeg.CLASSES]
    bars = ax.bar(class_positions - 0.18, recalls, 0.36, label="Recall")
    ax.bar_label(bars, fmt="%.2f", padding=2)
    bars = ax.bar(class_positions + 0.18, f1_values, 0.36, label="F1")
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(class_positions, eeg.CLASSES)
    ax.set(ylim=(0, 1), ylabel="Held-out score", title="Site-normalized log-mel SVM by class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "site_normalized_per_class.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
