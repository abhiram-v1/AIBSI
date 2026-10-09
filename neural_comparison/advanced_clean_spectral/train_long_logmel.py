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
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
SHORT_OUT = HERE / "outputs" / "deep_models" / "moderate_log_mel"
OUT = HERE / "outputs" / "deep_models" / "moderate_log_mel_long"
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import train_deep_compare as deep  # noqa: E402


SEEDS = [20261009, 20261019, 20261029]


def main() -> None:
    parser = argparse.ArgumentParser(description="Long log-mel CNN training with early stopping")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--bag-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    deep.OUT = OUT

    x = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = deep.make_subject_splits(participants)
    split_indexes = {
        name: np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))
        for name, subjects in splits.items()
    }
    scaler = np.load(SHORT_OUT / "train_scaler.npz")
    results = []
    validation_runs = []
    test_runs = []
    histories = {}
    for seed in SEEDS:
        result, val_probabilities, test_probabilities, history = deep.train_one(
            "cnn",
            seed,
            x,
            y,
            subject_index,
            participants,
            splits,
            scaler["mean"],
            scaler["std"],
            args.epochs,
            args.bag_size,
            args.learning_rate,
            patience=args.patience,
        )
        results.append(result)
        validation_runs.append(val_probabilities)
        test_runs.append(test_probabilities)
        histories[str(seed)] = history

    validation_rows = deep.ensemble_rows(
        validation_runs, y, subject_index, split_indexes["val"], participants
    )
    test_rows = deep.ensemble_rows(
        test_runs, y, subject_index, split_indexes["test"], participants
    )
    payload = {
        "device": torch.cuda.get_device_name(0),
        "epochs_ceiling": args.epochs,
        "early_stopping_patience": args.patience,
        "checkpoint_metric": "validation macro F1",
        "seeds": SEEDS,
        "individual_runs": results,
        "ensemble_validation": deep.calculate_metrics(validation_rows),
        "ensemble_test": deep.calculate_metrics(test_rows),
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    plot(payload, histories)


def plot(payload: dict, histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for seed, history in histories.items():
        axes[0].plot(
            [row["epoch"] for row in history],
            [row["val_macro_f1"] for row in history],
            marker="o",
            markersize=3,
            label=f"Seed {seed[-2:]}",
        )
        axes[1].plot(
            [row["epoch"] for row in history],
            [row["train_loss"] for row in history],
            label=f"Seed {seed[-2:]}",
        )
    axes[0].set(xlabel="Epoch", ylabel="Validation macro F1", title="Early-stopping metric", ylim=(0, 1))
    axes[1].set(xlabel="Epoch", ylabel="Training loss", title="Training loss")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(OUT / "training_curves.png", dpi=180)
    plt.close(fig)

    short = json.loads((SHORT_OUT / "metrics.json").read_text(encoding="utf-8"))["test_calibrated"]
    site = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    rows = [short, payload["ensemble_test"], site]
    labels = ["30-epoch CNN\nensemble", "Long CNN ensemble\nwith early stopping", "Site-normalized\nlog-mel SVM"]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(3)
    for offset, measure in enumerate(measures):
        bars = ax.bar(
            positions + (offset - 1) * 0.24,
            [row[measure] for row in rows],
            0.24,
            label=measure.replace("_", " ").title(),
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Effect of longer CNN training")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "long_training_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
