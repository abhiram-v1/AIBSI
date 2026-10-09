from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "prepared"
OUT = HERE / "outputs" / "deep_models"
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import train_deep_compare as deep  # noqa: E402


def main() -> None:
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = deep.make_subject_splits(participants)
    test_indexes = np.flatnonzero(np.isin(subject_index, np.asarray(splits["test"], dtype=np.int32)))
    device = torch.device("cuda")
    probabilities: dict[str, np.ndarray] = {}
    for representation in ["log_stft", "log_mel"]:
        x = np.load(PREPARED / f"{representation}.npy", mmap_mode="r")
        scaler = np.load(OUT / representation / "train_scaler.npz")
        loader = DataLoader(
            deep.WindowDataset(x, y, test_indexes, scaler["mean"], scaler["std"]),
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for model_type in ["cnn", "cnn_bilstm"]:
            model = deep.CompactCNN() if model_type == "cnn" else deep.CNNBiLSTM()
            checkpoint = torch.load(
                OUT / representation / model_type / "seed_20261009" / "best_model.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            probs, _ = deep.predict_windows(model, loader, device, len(y))
            probabilities[f"{representation}_{model_type}"] = probs

    combinations = {
        "log_stft_cnn_bilstm": ["log_stft_cnn", "log_stft_cnn_bilstm"],
        "log_mel_cnn_bilstm": ["log_mel_cnn", "log_mel_cnn_bilstm"],
        "all_four_uniform": list(probabilities),
    }
    results = {}
    for name, members in combinations.items():
        average = np.full_like(next(iter(probabilities.values())), np.nan)
        average[test_indexes] = np.mean(
            np.stack([probabilities[member][test_indexes] for member in members]), axis=0
        )
        labels = np.full(len(y), -1, dtype=np.int64)
        labels[test_indexes] = y[test_indexes]
        rows = deep.aggregate_subjects(average, labels, subject_index, test_indexes, participants)
        results[name] = {"members": members, "test": deep.calculate_metrics(rows)}
        print(name, json.dumps(results[name]["test"], indent=2))
    (OUT / "ensemble_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
