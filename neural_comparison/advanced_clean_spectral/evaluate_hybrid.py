from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ADV_PREPARED = HERE / "outputs" / "prepared"
DEEP_OUT = HERE / "outputs" / "deep_models" / "log_mel"
OLD = ROOT / "neural_comparison" / "stft_gpu_3class"
sys.path.insert(0, str(OLD))
import train_deep_compare as deep  # noqa: E402
import train_hierarchical_svm as hierarchy  # noqa: E402
from train_svm import PREPARED as OLD_PREPARED, subject_features  # noqa: E402


SEEDS = [20261009, 20261019, 20261029]


def svm_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    evaluate: np.ndarray,
    selected: dict,
    temperature: float,
) -> np.ndarray:
    stage1 = hierarchy.make_model(selected["stage1"]["config"], selected["stage1"]["boost"])
    stage1.fit(features[train], (labels[train] != 2).astype(int))
    disease = train[labels[train] != 2]
    stage2 = hierarchy.make_model(selected["stage2"]["config"], selected["stage2"]["boost"])
    stage2.fit(features[disease], labels[disease])
    disease_probability = expit(stage1.decision_function(features[evaluate]) / temperature)
    ftd_given_disease = expit(stage2.decision_function(features[evaluate]) / temperature)
    return np.column_stack(
        [
            disease_probability * (1.0 - ftd_given_disease),
            disease_probability * ftd_given_disease,
            1.0 - disease_probability,
        ]
    )


def cnn_subject_probabilities(split_name: str) -> dict[str, np.ndarray]:
    x = np.load(ADV_PREPARED / "log_mel.npy", mmap_mode="r")
    y = np.load(ADV_PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(ADV_PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((ADV_PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = deep.make_subject_splits(participants)
    indexes = np.flatnonzero(np.isin(subject_index, np.asarray(splits[split_name], dtype=np.int32)))
    scaler = np.load(DEEP_OUT / "train_scaler.npz")
    loader = DataLoader(
        deep.WindowDataset(x, y, indexes, scaler["mean"], scaler["std"]),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda")
    runs = []
    for seed in SEEDS:
        model = deep.CompactCNN().to(device)
        checkpoint = torch.load(
            DEEP_OUT / "cnn" / f"seed_{seed}" / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        probabilities, _ = deep.predict_windows(model, loader, device, len(y))
        runs.append(probabilities)
    mean_windows = np.full_like(runs[0], np.nan)
    mean_windows[indexes] = np.mean(np.stack([run[indexes] for run in runs]), axis=0)
    result = {}
    for subject in splits[split_name]:
        use = indexes[subject_index[indexes] == subject]
        result[participants[subject]["subject_id"]] = mean_windows[use].mean(axis=0)
    return result


def metric(true: np.ndarray, predicted: np.ndarray) -> dict:
    return deep.calculate_metrics(
        [{"true_id": int(t), "predicted_id": int(p)} for t, p in zip(true, predicted)]
    )


def main() -> None:
    old_x = np.load(OLD_PREPARED / "spectrograms.npy", mmap_mode="r")
    old_y_windows = np.load(OLD_PREPARED / "labels.npy", mmap_mode="r")
    old_subject_index = np.load(OLD_PREPARED / "subject_index.npy", mmap_mode="r")
    old_participants = json.loads((OLD_PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    old_splits = json.loads((OLD / "outputs" / "deep_compare" / "splits.json").read_text(encoding="utf-8"))
    selected = json.loads((OLD / "outputs" / "hierarchical_svm" / "metrics.json").read_text(encoding="utf-8"))["selected_on_validation"]
    features, labels = subject_features(old_x, old_y_windows, old_subject_index, old_participants)
    train = np.asarray(old_splits["train"], dtype=int)
    val = np.asarray(old_splits["val"], dtype=int)
    test = np.asarray(old_splits["test"], dtype=int)
    cnn_val_by_id = cnn_subject_probabilities("val")
    cnn_test_by_id = cnn_subject_probabilities("test")
    cnn_val = np.stack([cnn_val_by_id[old_participants[index]["subject_id"]] for index in val])
    cnn_test = np.stack([cnn_test_by_id[old_participants[index]["subject_id"]] for index in test])

    candidates = []
    svm_validation = {
        temperature: svm_probabilities(features, labels, train, val, selected, temperature)
        for temperature in [0.5, 1.0, 2.0]
    }
    for temperature, cnn_weight, ftd_weight in itertools.product([0.5, 1.0, 2.0], np.linspace(0.0, 1.0, 11), [1.0, 1.25, 1.5, 2.0]):
        svm_val = svm_validation[temperature]
        combined = cnn_weight * cnn_val + (1.0 - cnn_weight) * svm_val
        combined[:, 1] *= ftd_weight
        prediction = np.argmax(combined, axis=1)
        candidates.append(
            {
                "svm_temperature": temperature,
                "cnn_weight": float(cnn_weight),
                "ftd_weight": ftd_weight,
                "macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(labels[val], prediction, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(labels[val], prediction)),
            }
        )
    candidates.sort(key=lambda row: (row["macro_f1"], row["macro_recall"], row["accuracy"]), reverse=True)
    best = candidates[0]
    train_val = np.concatenate([train, val])
    svm_test = svm_probabilities(features, labels, train_val, test, selected, best["svm_temperature"])
    combined_test = best["cnn_weight"] * cnn_test + (1.0 - best["cnn_weight"]) * svm_test
    combined_test[:, 1] *= best["ftd_weight"]
    prediction = np.argmax(combined_test, axis=1)
    test_metrics = metric(labels[test], prediction)
    payload = {"selected_on_validation": best, "test": test_metrics, "validation_candidates": candidates}
    output = HERE / "outputs" / "deep_models" / "hybrid_svm_cnn.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_on_validation": best, "test": test_metrics}, indent=2))

    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3), deep.CLASS_NAMES)
    ax.set_yticks(range(3), deep.CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="Validation-selected SVM + log-mel CNN hybrid")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(HERE / "outputs" / "deep_models" / "hybrid_confusion_matrix.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
