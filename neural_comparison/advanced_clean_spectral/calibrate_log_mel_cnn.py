from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "prepared"
OUT = HERE / "outputs" / "deep_models"
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import train_deep_compare as deep  # noqa: E402


def participant_probabilities(
    probabilities: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    labels = []
    for subject in sorted(set(subject_index[indexes].tolist())):
        use = indexes[subject_index[indexes] == subject]
        values.append(probabilities[use].mean(axis=0))
        labels.append(int(y[use[0]]))
    return np.asarray(values), np.asarray(labels)


def main() -> None:
    x = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = deep.make_subject_splits(participants)
    scaler = np.load(OUT / "log_mel" / "train_scaler.npz")
    device = torch.device("cuda")
    model = deep.CompactCNN().to(device)
    checkpoint = torch.load(
        OUT / "log_mel" / "cnn" / "seed_20261009" / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    split_data = {}
    for name in ["val", "test"]:
        indexes = np.flatnonzero(np.isin(subject_index, np.asarray(splits[name], dtype=np.int32)))
        loader = DataLoader(
            deep.WindowDataset(x, y, indexes, scaler["mean"], scaler["std"]),
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        probabilities, _ = deep.predict_windows(model, loader, device, len(y))
        split_data[name] = participant_probabilities(probabilities, y, subject_index, indexes)

    val_probabilities, val_labels = split_data["val"]
    candidates = []
    for ad, ftd, hc in itertools.product([0.75, 1.0, 1.25], [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0], [0.75, 1.0, 1.25]):
        weights = np.asarray([ad, ftd, hc])
        prediction = np.argmax(val_probabilities * weights, axis=1)
        candidates.append(
            {
                "weights": weights.tolist(),
                "macro_f1": float(f1_score(val_labels, prediction, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(val_labels, prediction, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(val_labels, prediction)),
            }
        )
    candidates.sort(key=lambda item: (item["macro_f1"], item["macro_recall"], item["accuracy"]), reverse=True)
    best = candidates[0]
    test_probabilities, test_labels = split_data["test"]
    test_prediction = np.argmax(test_probabilities * np.asarray(best["weights"]), axis=1)
    rows = []
    for true, predicted in zip(test_labels, test_prediction):
        rows.append({"true_id": int(true), "predicted_id": int(predicted)})
    test = deep.calculate_metrics(rows)
    payload = {"selected_on_validation": best, "test": test, "validation_candidates": candidates}
    (OUT / "calibrated_log_mel_cnn.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_on_validation": best, "test": test}, indent=2))


if __name__ == "__main__":
    main()
