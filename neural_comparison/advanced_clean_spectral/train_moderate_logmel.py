from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD_PREPARED = ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "prepared"
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "deep_models" / "moderate_log_mel"
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import train_deep_compare as deep  # noqa: E402
sys.path.insert(0, str(HERE))
import pipeline as advanced  # noqa: E402


SEEDS = [20261009, 20261019, 20261029]


def prepare_log_mel() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    output = PREPARED / "log_mel.npy"
    if output.exists():
        print(f"Reusing {output}")
        return
    stft = np.load(OLD_PREPARED / "spectrograms.npy", mmap_mode="r")
    converted = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float16,
        shape=(stft.shape[0], stft.shape[1], advanced.MEL_BINS, stft.shape[3]),
    )
    frequencies = np.linspace(1.0, 40.0, stft.shape[2])
    bank = advanced.mel_filter_bank(frequencies)
    for start in range(0, len(stft), 256):
        block = stft[start : start + 256].astype(np.float32)
        relative_power = np.power(10.0, block)
        mel_power = np.einsum("mf,ncft->ncmt", bank, relative_power, optimize=True)
        log_mel = np.log10(mel_power + 1e-20)
        log_mel -= np.median(log_mel, axis=(1, 2, 3), keepdims=True)
        converted[start : start + len(block)] = log_mel.astype(np.float16)
        print(f"Converted {min(start + len(block), len(stft))}/{len(stft)} windows")
    converted.flush()
    for name in ["labels.npy", "subject_index.npy", "manifest.json"]:
        shutil.copy2(OLD_PREPARED / name, PREPARED / name)


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


def metric(true: np.ndarray, predicted: np.ndarray) -> dict:
    return deep.calculate_metrics(
        [{"true_id": int(t), "predicted_id": int(p)} for t, p in zip(true, predicted)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Moderately cleaned log-mel three-seed CNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bag-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prepare_log_mel()
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = deep.make_subject_splits(participants)
    split_indexes = {
        name: np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))
        for name, subjects in splits.items()
    }
    scaler_path = OUT / "train_scaler.npz"
    if scaler_path.exists():
        scaler = np.load(scaler_path)
        mean, std = scaler["mean"], scaler["std"]
    else:
        mean, std = deep.fit_scaler(x, split_indexes["train"])
        np.savez_compressed(scaler_path, mean=mean, std=std)
    deep.OUT = OUT
    individual = {}
    probability_runs = {"val": [], "test": []}
    for seed in SEEDS:
        checkpoint_path = OUT / "cnn" / f"seed_{seed}" / "best_model.pt"
        if checkpoint_path.exists():
            print(f"Reusing {checkpoint_path}")
        else:
            deep.train_one(
                "cnn", seed, x, y, subject_index, participants, splits,
                mean, std, args.epochs, args.bag_size, args.learning_rate,
            )
        individual[str(seed)] = json.loads(
            (OUT / "cnn" / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8")
        )
        model = deep.CompactCNN().to("cuda")
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        for split_name in ["val", "test"]:
            loader = torch.utils.data.DataLoader(
                deep.WindowDataset(x, y, split_indexes[split_name], mean, std),
                batch_size=256, shuffle=False, num_workers=0, pin_memory=True,
            )
            probabilities, _ = deep.predict_windows(model, loader, torch.device("cuda"), len(y))
            probability_runs[split_name].append(probabilities)

    participant_probability = {}
    participant_label = {}
    for split_name in ["val", "test"]:
        average = np.full_like(probability_runs[split_name][0], np.nan)
        indexes = split_indexes[split_name]
        average[indexes] = np.mean(np.stack([run[indexes] for run in probability_runs[split_name]]), axis=0)
        participant_probability[split_name], participant_label[split_name] = participant_arrays(
            average, y, subject_index, indexes
        )

    val_probability, val_true = participant_probability["val"], participant_label["val"]
    candidates = []
    for ad, ftd, hc in itertools.product([0.75, 1.0, 1.25], [1.0, 1.25, 1.5, 2.0, 2.5, 3.0], [0.75, 1.0, 1.25]):
        weights = np.asarray([ad, ftd, hc])
        prediction = np.argmax(val_probability * weights, axis=1)
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
    test_probability, test_true = participant_probability["test"], participant_label["test"]
    uncalibrated = metric(test_true, np.argmax(test_probability, axis=1))
    calibrated = metric(test_true, np.argmax(test_probability * np.asarray(selected["weights"]), axis=1))
    payload = {
        "cleaning": "original moderate filtering and artifact rejection; converted from saved log-STFT to 24-bin log-mel",
        "seeds": SEEDS,
        "individual": individual,
        "validation_selected_calibration": selected,
        "test_uncalibrated": uncalibrated,
        "test_calibrated": calibrated,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_selected_calibration": selected, "test_uncalibrated": uncalibrated, "test_calibrated": calibrated}, indent=2))
    plot(payload)


def plot(payload: dict) -> None:
    cleaned = json.loads((HERE / "outputs" / "deep_models" / "metrics.json").read_text(encoding="utf-8"))["results"]["log_mel"]["cnn"]["test"]
    rows = [cleaned, payload["test_uncalibrated"], payload["test_calibrated"]]
    labels = ["Advanced cleaning\nsingle CNN", "Moderate cleaning\n3-seed ensemble", "Moderate cleaning\ncalibrated ensemble"]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(3)
    for offset, measure in enumerate(measures):
        bars = ax.bar(positions + (offset - 1) * 0.24, [row[measure] for row in rows], 0.24, label=measure.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Log-mel CNN cleaning ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "cleaning_ablation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
