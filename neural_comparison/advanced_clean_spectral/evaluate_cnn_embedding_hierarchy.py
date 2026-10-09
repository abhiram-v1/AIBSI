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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "prepared"
MODEL_OUT = HERE / "outputs" / "deep_models" / "log_mel"
RESULT_OUT = HERE / "outputs" / "deep_models"
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import train_deep_compare as deep  # noqa: E402


SEED = 20261009


@torch.no_grad()
def extract_embeddings(
    x: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    model: deep.CompactCNN,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.arange(len(y))
    loader = DataLoader(
        deep.WindowDataset(x, y, indexes, mean, std),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda")
    window_embeddings = np.zeros((len(y), 96), dtype=np.float32)
    for values, _, batch_indexes in loader:
        values = values.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            embedding = model.encoder(values).mean(dim=1)
        window_embeddings[batch_indexes.numpy()] = embedding.float().cpu().numpy()
    subjects = int(subject_index.max()) + 1
    participant_embeddings = np.zeros((subjects, 96), dtype=np.float32)
    participant_labels = np.zeros(subjects, dtype=np.int64)
    for subject in range(subjects):
        use = np.flatnonzero(subject_index == subject)
        participant_embeddings[subject] = window_embeddings[use].mean(axis=0)
        participant_labels[subject] = int(y[use[0]])
    return participant_embeddings, participant_labels


def make_model(config: dict, class_weight: dict | str) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if config["pca"] is not None:
        steps.append(("pca", PCA(n_components=config["pca"], random_state=SEED)))
    steps.append(
        (
            "svm",
            SVC(
                C=config["C"],
                kernel=config["kernel"],
                gamma="scale",
                class_weight=class_weight,
                random_state=SEED,
            ),
        )
    )
    return Pipeline(steps)


def predict_hierarchy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.where(first == 0, 2, second).astype(np.int64)


def main() -> None:
    x = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    y_windows = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = deep.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    scaler = np.load(MODEL_OUT / "train_scaler.npz")
    model = deep.CompactCNN().to("cuda")
    checkpoint = torch.load(
        MODEL_OUT / "cnn" / f"seed_{SEED}" / "best_model.pt",
        map_location="cuda",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    features, labels = extract_embeddings(
        x, y_windows, subject_index, scaler["mean"], scaler["std"], model
    )

    configs = [
        {"pca": pca, "C": c, "kernel": kernel}
        for pca, c, kernel in itertools.product([None, 20, 40, 60], [0.01, 0.1, 1.0, 10.0], ["linear", "rbf"])
    ]
    first_runs = []
    for config, disease_boost in itertools.product(configs, [1.0, 1.5, 2.0]):
        first = make_model(config, {0: 1.0, 1: disease_boost})
        first.fit(features[train], (labels[train] != 2).astype(int))
        first_runs.append((config, disease_boost, first.predict(features[val])))
    disease_train = train[labels[train] != 2]
    second_runs = []
    for config, ftd_boost in itertools.product(configs, [1.0, 1.5, 2.0, 3.0, 4.0]):
        second = make_model(config, {0: 1.0, 1: ftd_boost})
        second.fit(features[disease_train], labels[disease_train])
        second_runs.append((config, ftd_boost, second.predict(features[val])))
    candidates = []
    for first, second in itertools.product(first_runs, second_runs):
        prediction = predict_hierarchy(first[2], second[2])
        candidates.append(
            {
                "stage1": {"config": first[0], "disease_boost": first[1]},
                "stage2": {"config": second[0], "ftd_boost": second[1]},
                "val_macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
                "val_accuracy": float(accuracy_score(labels[val], prediction)),
                "val_ftd_recall": float(np.mean(prediction[labels[val] == 1] == 1)),
            }
        )
    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_ftd_recall"], row["val_accuracy"]), reverse=True)
    best = candidates[0]
    train_val = np.concatenate([train, val])
    stage1 = make_model(best["stage1"]["config"], {0: 1.0, 1: best["stage1"]["disease_boost"]})
    stage1.fit(features[train_val], (labels[train_val] != 2).astype(int))
    disease_train_val = train_val[labels[train_val] != 2]
    stage2 = make_model(best["stage2"]["config"], {0: 1.0, 1: best["stage2"]["ftd_boost"]})
    stage2.fit(features[disease_train_val], labels[disease_train_val])
    prediction = predict_hierarchy(stage1.predict(features[test]), stage2.predict(features[test]))
    rows = [{"true_id": int(t), "predicted_id": int(p)} for t, p in zip(labels[test], prediction)]
    test_metrics = deep.calculate_metrics(rows)
    payload = {"selected_on_validation": best, "test": test_metrics}
    (RESULT_OUT / "cnn_embedding_hierarchy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(RESULT_OUT / "cnn_participant_embeddings.npy", features)
    print(json.dumps(payload, indent=2))

    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3), deep.CLASS_NAMES)
    ax.set_yticks(range(3), deep.CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="Hierarchical SVM on log-mel CNN embeddings")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(RESULT_OUT / "cnn_embedding_hierarchy_confusion.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
