from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from run_pipeline import CLASS_NAMES, PREPARED, make_subject_splits
from train_deep_compare import (
    CNNBiLSTM,
    CompactCNN,
    OUT,
    SEEDS,
    WindowDataset,
    calculate_metrics,
    predict_windows,
)


def split_indexes(subject_index: np.ndarray, subjects: list[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))


def subject_arrays(
    probabilities: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = np.asarray(sorted(set(subject_index[indexes].tolist())), dtype=np.int64)
    subject_probabilities = []
    labels = []
    for subject in subjects:
        use = indexes[subject_index[indexes] == subject]
        subject_probabilities.append(probabilities[use].mean(axis=0))
        labels.append(int(y[use[0]]))
    return np.asarray(subject_probabilities), np.asarray(labels), subjects


def rows_from_arrays(
    probabilities: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    participants: list[dict],
    multipliers: np.ndarray,
) -> list[dict]:
    adjusted = probabilities * multipliers[None, :]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    predictions = adjusted.argmax(axis=1)
    rows = []
    for position, subject in enumerate(subjects):
        true = int(labels[position])
        pred = int(predictions[position])
        rows.append(
            {
                **participants[int(subject)],
                "true_id": true,
                "true_label": CLASS_NAMES[true],
                "predicted_id": pred,
                "predicted_label": CLASS_NAMES[pred],
                "prob_AD": float(adjusted[position, 0]),
                "prob_FTD": float(adjusted[position, 1]),
                "prob_HC": float(adjusted[position, 2]),
                "correct": bool(true == pred),
            }
        )
    return rows


def select_multipliers(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    # AD is the reference. FTD and HC multipliers implement a validation-tuned
    # hierarchical decision without reading any held-out test labels.
    values = np.asarray([0.25, 0.4, 0.63, 1.0, 1.59, 2.52, 4.0], dtype=np.float32)
    candidates = []
    for ftd, hc in product(values, values):
        multipliers = np.asarray([1.0, ftd, hc], dtype=np.float32)
        prediction = np.argmax(probabilities * multipliers[None, :], axis=1)
        candidates.append(
            {
                "multipliers": multipliers,
                "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(labels, prediction)),
                "distance": float(np.square(np.log(multipliers)).sum()),
            }
        )
    candidates.sort(key=lambda row: (row["macro_f1"], row["accuracy"], -row["distance"]), reverse=True)
    best = candidates[0]
    return best["multipliers"], {
        "multipliers": best["multipliers"].tolist(),
        "validation_macro_f1": best["macro_f1"],
        "validation_accuracy": best["accuracy"],
    }


def evaluate_model(
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    participants: list[dict],
    splits: dict[str, list[int]],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict:
    indexes = {name: split_indexes(subject_index, splits[name]) for name in ["val", "test"]}
    loaders = {
        name: DataLoader(
            WindowDataset(x, y, indexes[name], mean, std),
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for name in indexes
    }
    probability_runs: dict[str, list[np.ndarray]] = {"val": [], "test": []}
    for seed in SEEDS:
        model = CompactCNN() if model_type == "cnn" else CNNBiLSTM()
        checkpoint = torch.load(
            OUT / model_type / f"seed_{seed}" / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        for split in ["val", "test"]:
            probabilities, _ = predict_windows(model, loaders[split], device, len(x))
            probability_runs[split].append(probabilities[indexes[split]])
        del model

    arrays = {}
    for split in ["val", "test"]:
        mean_probabilities = np.mean(np.stack(probability_runs[split]), axis=0)
        full = np.full((len(x), len(CLASS_NAMES)), np.nan, dtype=np.float32)
        full[indexes[split]] = mean_probabilities
        arrays[split] = subject_arrays(full, y, subject_index, indexes[split])

    val_probabilities, val_labels, val_subjects = arrays["val"]
    multipliers, selection = select_multipliers(val_probabilities, val_labels)
    test_probabilities, test_labels, test_subjects = arrays["test"]
    val_rows = rows_from_arrays(
        val_probabilities, val_labels, val_subjects, participants, multipliers
    )
    test_rows = rows_from_arrays(
        test_probabilities, test_labels, test_subjects, participants, multipliers
    )
    return {
        "selection": selection,
        "val": calculate_metrics(val_rows),
        "test": calculate_metrics(test_rows),
        "test_rows": test_rows,
    }


def save_predictions(path: Path, rows: list[dict]) -> None:
    fields = [
        "subject_id", "source", "site", "true_label", "predicted_label",
        "prob_AD", "prob_FTD", "prob_HC", "correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot(results: dict) -> None:
    models = ["cnn", "cnn_bilstm"]
    labels = ["CNN", "CNN–BiLSTM"]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    positions = np.arange(2)
    for offset, measure in enumerate(measures):
        values = [results[model]["test"][measure] for model in models]
        bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=measure.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Participant-level score", title="Validation-calibrated held-out comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "calibrated_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, model, label in zip(axes, models, labels):
        matrix = np.asarray(results[model]["test"]["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set(xlabel="Predicted", ylabel="True", title=label)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    fig.savefig(OUT / "calibrated_confusion_matrices.png", dpi=180)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda")
    x = np.load(PREPARED / "spectrograms.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = make_subject_splits(participants)
    scaler = np.load(OUT / "enhanced_train_scaler.npz")
    results = {}
    for model_type in ["cnn", "cnn_bilstm"]:
        result = evaluate_model(
            model_type,
            x,
            y,
            subject_index,
            participants,
            splits,
            scaler["mean"],
            scaler["std"],
            device,
        )
        save_predictions(OUT / f"{model_type}_calibrated_predictions.csv", result.pop("test_rows"))
        results[model_type] = result
        print(model_type, json.dumps(result, indent=2))
    (OUT / "calibrated_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results)


if __name__ == "__main__":
    main()
