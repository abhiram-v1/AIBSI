from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "prepared"
OUT = HERE / "outputs" / "deep_models" / "log_mel"
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import train_deep_compare as deep  # noqa: E402


SEEDS = [20261009, 20261019, 20261029]


def participant_arrays(
    probabilities: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values, labels = [], []
    for subject in sorted(set(subject_index[indexes].tolist())):
        use = indexes[subject_index[indexes] == subject]
        values.append(probabilities[use].mean(axis=0))
        labels.append(int(y[use[0]]))
    return np.asarray(values), np.asarray(labels)


def metrics_from_arrays(true: np.ndarray, predicted: np.ndarray) -> dict:
    return deep.calculate_metrics(
        [{"true_id": int(t), "predicted_id": int(p)} for t, p in zip(true, predicted)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-seed cleaned log-mel CNN ensemble")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bag-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    x = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = deep.make_subject_splits(participants)
    split_indexes = {
        name: np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))
        for name, subjects in splits.items()
    }
    scaler = np.load(OUT / "train_scaler.npz")
    deep.OUT = OUT
    for seed in SEEDS[1:]:
        checkpoint = OUT / "cnn" / f"seed_{seed}" / "best_model.pt"
        if checkpoint.exists():
            print(f"Reusing {checkpoint}")
            continue
        deep.train_one(
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
        )

    device = torch.device("cuda")
    probability_runs: dict[str, dict[int, np.ndarray]] = {"val": {}, "test": {}}
    for seed in SEEDS:
        model = deep.CompactCNN().to(device)
        checkpoint = torch.load(
            OUT / "cnn" / f"seed_{seed}" / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        for split_name in ["val", "test"]:
            loader = DataLoader(
                deep.WindowDataset(x, y, split_indexes[split_name], scaler["mean"], scaler["std"]),
                batch_size=256,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            probability_runs[split_name][seed], _ = deep.predict_windows(model, loader, device, len(y))

    participant_probabilities = {}
    participant_labels = {}
    for split_name in ["val", "test"]:
        mean_windows = np.full_like(next(iter(probability_runs[split_name].values())), np.nan)
        mean_windows[split_indexes[split_name]] = np.mean(
            np.stack([probability_runs[split_name][seed][split_indexes[split_name]] for seed in SEEDS]),
            axis=0,
        )
        participant_probabilities[split_name], participant_labels[split_name] = participant_arrays(
            mean_windows, y, subject_index, split_indexes[split_name]
        )

    val_prob = participant_probabilities["val"]
    val_true = participant_labels["val"]
    candidates = []
    for ad, ftd, hc in itertools.product([0.75, 1.0, 1.25], [1.0, 1.25, 1.5, 2.0, 2.5, 3.0], [0.75, 1.0, 1.25]):
        weights = np.asarray([ad, ftd, hc])
        prediction = np.argmax(val_prob * weights, axis=1)
        candidates.append(
            {
                "weights": weights.tolist(),
                "macro_f1": float(f1_score(val_true, prediction, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(val_true, prediction, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(val_true, prediction)),
            }
        )
    candidates.sort(key=lambda row: (row["macro_f1"], row["macro_recall"], row["accuracy"]), reverse=True)
    selected = candidates[0]
    test_prob = participant_probabilities["test"]
    test_true = participant_labels["test"]
    uncalibrated = metrics_from_arrays(test_true, np.argmax(test_prob, axis=1))
    calibrated = metrics_from_arrays(test_true, np.argmax(test_prob * np.asarray(selected["weights"]), axis=1))
    payload = {
        "seeds": SEEDS,
        "selected_calibration_on_validation": selected,
        "validation_uncalibrated": metrics_from_arrays(val_true, np.argmax(val_prob, axis=1)),
        "test_uncalibrated": uncalibrated,
        "test_calibrated": calibrated,
    }
    (OUT / "cnn_three_seed_ensemble.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    plot(payload)


def plot(payload: dict) -> None:
    rows = [payload["test_uncalibrated"], payload["test_calibrated"]]
    labels = ["Uniform ensemble", "Validation-calibrated ensemble"]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    positions = np.arange(2)
    for offset, measure in enumerate(measures):
        bars = ax.bar(
            positions + (offset - 1) * 0.24,
            [row[measure] for row in rows],
            0.24,
            label=measure.replace("_", " ").title(),
        )
        ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Three-seed cleaned log-mel CNN ensemble")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "cnn_three_seed_ensemble.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
