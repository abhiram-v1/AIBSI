"""Compare log-STFT and log-mel with one fixed participant-level SVM setup.

The prepared arrays are read in place. Both representations use the same
participants, split, site normalization, PCA dimensions, and SVM parameters.
This is an exploratory representation comparison on the previously used test
split, not a fresh independent test of a newly selected model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STFT = ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "prepared"
MEL = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "results"
CLASSES = ["AD", "FTD", "HC"]
SEED = 20261009


def participant_features(tensor: np.ndarray, labels: np.ndarray, subjects: np.ndarray, count: int):
    """Mean and standard deviation for each electrode/frequency across frames and windows."""
    features = []
    targets = []
    for subject in range(count):
        indexes = np.flatnonzero(subjects == subject)
        if len(indexes) == 0:
            raise ValueError(f"Participant {subject} has no windows")
        block = np.asarray(tensor[indexes], dtype=np.float32)
        repeated = block.transpose(0, 3, 1, 2).reshape(-1, block.shape[1], block.shape[2])
        features.append(np.concatenate([repeated.mean(axis=0), repeated.std(axis=0)], axis=1).reshape(-1))
        targets.append(int(labels[indexes[0]]))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def split_participants(participants: list[dict]):
    ids = np.arange(len(participants))
    strata = np.asarray([p["source"] + ":" + p["label"] for p in participants])
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=20261007)
    train, remaining = next(outer.split(ids, strata))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=20261008)
    val_local, test_local = next(inner.split(remaining, strata[remaining]))
    return train, remaining[val_local], remaining[test_local]


def fit_evaluate(features: np.ndarray, labels: np.ndarray, participants: list[dict], development, test):
    sites = np.asarray([p["site"] for p in participants])
    normalized = np.empty_like(features, dtype=np.float32)
    for site in sorted(set(sites)):
        reference = features[development[sites[development] == site]]
        if len(reference) == 0:
            raise ValueError(f"No development participants at site {site}")
        mean = reference.mean(axis=0, keepdims=True)
        std = np.maximum(reference.std(axis=0, keepdims=True), 1e-3)
        normalized[sites == site] = (features[sites == site] - mean) / std

    model = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=SEED)),
        ("svm", SVC(C=0.1, kernel="linear", gamma="scale", class_weight="balanced", random_state=SEED)),
    ])
    model.fit(normalized[development], labels[development])
    truth = labels[test]
    prediction = model.predict(normalized[test])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "correct": int(np.sum(truth == prediction)),
        "test_participants": int(len(test)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASSES)
        },
    }


def save_figure(results: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    measures = ["accuracy", "macro_f1", "macro_recall"]
    names = ["Log-STFT", "Log-mel"]
    x = np.arange(len(measures))
    for j, name in enumerate(names):
        values = [results[name][key] for key in measures]
        axes[0].bar(x + (j - 0.5) * 0.36, values, 0.36, label=name)
    axes[0].set_xticks(x, ["Accuracy", "Macro F1", "Macro recall"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Held-out participant scores")
    axes[0].legend()
    for ax, name in zip(axes[1:], names):
        matrix = np.asarray(results[name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, max(np.max(results[n]["confusion_matrix"]) for n in names)))
        for row in range(3):
            for column in range(3):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set(xlabel="Predicted", ylabel="Actual", title=name + " confusion matrix")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    participants = json.loads((STFT / "manifest.json").read_text(encoding="utf-8"))["participants"]
    mel_participants = json.loads((MEL / "manifest.json").read_text(encoding="utf-8"))["participants"]
    if [p["subject_id"] for p in participants] != [p["subject_id"] for p in mel_participants]:
        raise ValueError("Participant order differs between representations")
    labels = np.load(STFT / "labels.npy", mmap_mode="r")
    subjects = np.load(STFT / "subject_index.npy", mmap_mode="r")
    if not np.array_equal(labels, np.load(MEL / "labels.npy", mmap_mode="r")):
        raise ValueError("Window labels differ between representations")
    if not np.array_equal(subjects, np.load(MEL / "subject_index.npy", mmap_mode="r")):
        raise ValueError("Participant indexes differ between representations")
    train, val, test = split_participants(participants)
    development = np.concatenate([train, val])

    results = {}
    for name, source in [("Log-STFT", STFT / "spectrograms.npy"), ("Log-mel", MEL / "log_mel.npy")]:
        tensor = np.load(source, mmap_mode="r")
        features, targets = participant_features(tensor, labels, subjects, len(participants))
        results[name] = fit_evaluate(features, targets, participants, development, test)
        results[name]["feature_count"] = int(features.shape[1])
        results[name]["source"] = str(source)
        print(f"{name}: {results[name]['correct']}/{len(test)} correct, "
              f"accuracy={results[name]['accuracy']:.3f}, macro F1={results[name]['macro_f1']:.3f}")

    payload = {
        "experiment": "Fixed-config, matched-participant log-STFT vs log-mel SVM",
        "interpretation": "Exploratory comparison on a previously used test split",
        "participants": len(participants),
        "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "split_seeds": {"outer": 20261007, "inner": 20261008},
        "model": {"site_normalization_fit": "train+validation only", "pca_components": 20,
                  "svm_kernel": "linear", "svm_C": 0.1, "class_weight": "balanced", "seed": SEED},
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "matched_log_stft_svm_metrics.json"
    image_path = OUT / "matched_log_stft_svm_comparison.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_figure(results, image_path)
    print(f"Saved {json_path}")
    print(f"Saved {image_path}")


if __name__ == "__main__":
    main()
