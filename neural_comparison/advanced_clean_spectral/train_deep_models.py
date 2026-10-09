from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "prepared"
OUT = HERE / "outputs" / "deep_models"
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import train_deep_compare as deep  # noqa: E402


REPRESENTATIONS = ["log_stft", "log_mel"]
MODELS = ["cnn", "cnn_bilstm"]
SEED = 20261009


def run(epochs: int, bag_size: int, learning_rate: float) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the cleaned deep-model comparison")
    OUT.mkdir(parents=True, exist_ok=True)
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = deep.make_subject_splits(participants)
    split_indexes = {
        name: np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))
        for name, subjects in splits.items()
    }
    results: dict[str, dict] = {}
    for representation in REPRESENTATIONS:
        print(f"\n===== {representation} =====")
        representation_dir = OUT / representation
        representation_dir.mkdir(parents=True, exist_ok=True)
        deep.OUT = representation_dir
        x = np.load(PREPARED / f"{representation}.npy", mmap_mode="r")
        mean, std = deep.fit_scaler(x, split_indexes["train"])
        np.savez_compressed(representation_dir / "train_scaler.npz", mean=mean, std=std)
        representation_results: dict[str, dict] = {}
        for model_type in MODELS:
            result, _, _, _ = deep.train_one(
                model_type,
                SEED,
                x,
                y,
                subject_index,
                participants,
                splits,
                mean,
                std,
                epochs,
                bag_size,
                learning_rate,
            )
            representation_results[model_type] = result
            print(
                f"{representation} {model_type}: "
                f"val F1={result['val']['macro_f1']:.4f}, "
                f"test F1={result['test']['macro_f1']:.4f}"
            )
        results[representation] = representation_results

    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "seed": SEED,
        "epochs_max": epochs,
        "bag_size": bag_size,
        "learning_rate": learning_rate,
        "participants": len(participants),
        "split_participants": {name: len(values) for name, values in splits.items()},
        "results": results,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload)


def plot(payload: dict) -> None:
    labels = ["STFT\nCNN", "STFT\nCNN-BiLSTM", "Log-mel\nCNN", "Log-mel\nCNN-BiLSTM"]
    rows = [
        payload["results"][representation][model]["test"]
        for representation in REPRESENTATIONS
        for model in MODELS
    ]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(rows))
    for offset, measure in enumerate(measures):
        values = [row[measure] for row in rows]
        bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=measure.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Cleaned spectral deep-model comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    maximum = max(np.asarray(row["confusion_matrix"]).max() for row in rows)
    for ax, label, row in zip(axes.flat, labels, rows):
        matrix = np.asarray(row["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for true in range(3):
            for predicted in range(3):
                ax.text(predicted, true, str(matrix[true, predicted]), ha="center", va="center")
        ax.set_xticks(range(3), deep.CLASS_NAMES)
        ax.set_yticks(range(3), deep.CLASS_NAMES)
        ax.set(xlabel="Predicted", ylabel="True", title=label.replace("\n", " "))
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    fig.savefig(OUT / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    previous_path = ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "hierarchical_svm" / "metrics.json"
    classical_path = HERE / "outputs" / "results" / "metrics.json"
    if previous_path.exists() and classical_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))["test"]
        classical = json.loads(classical_path.read_text(encoding="utf-8"))["representations"]["log_mel"]["test"]
        overall_labels = ["Earlier STFT\nhierarchical SVM", "Cleaned log-mel\nhierarchical SVM", "Cleaned log-mel\nCNN", "Cleaned log-mel\nCNN-BiLSTM"]
        overall_rows = [previous, classical, payload["results"]["log_mel"]["cnn"]["test"], payload["results"]["log_mel"]["cnn_bilstm"]["test"]]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        positions = np.arange(len(overall_rows))
        for offset, measure in enumerate(measures):
            values = [row[measure] for row in overall_rows]
            bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=measure.replace("_", " ").title())
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_xticks(positions, overall_labels)
        ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Best earlier model versus cleaned log-mel models")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.savefig(OUT / "overall_best_comparison.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN and CNN-BiLSTM comparison on cleaned log-STFT and log-mel EEG")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bag-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    run(args.epochs, args.bag_size, args.learning_rate)


if __name__ == "__main__":
    main()
