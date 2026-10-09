from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_pipeline import CLASS_NAMES
from train_deep_compare import calculate_metrics
from train_svm import CNN_RUN, PREPARED, subject_features


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "hierarchical_svm"


def make_model(config: dict, minority_boost: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=config["components"], random_state=20261009)),
            (
                "svm",
                SVC(
                    C=config["C"],
                    gamma=config["gamma"],
                    kernel=config["kernel"],
                    class_weight={0: 1.0, 1: minority_boost},
                    probability=False,
                    random_state=20261009,
                ),
            ),
        ]
    )


def configurations() -> list[dict]:
    configs = []
    # The disease-only training fold has 79 participants, so PCA must stay
    # below that sample count.
    for components in [20, 40, 60]:
        for c in [0.1, 1.0, 10.0]:
            configs.append({"components": components, "C": c, "kernel": "linear", "gamma": "scale"})
            configs.append({"components": components, "C": c, "kernel": "rbf", "gamma": "scale"})
    return configs


def hierarchical_prediction(stage1: np.ndarray, stage2: np.ndarray) -> np.ndarray:
    # Stage 1: 0=healthy, 1=dementia. Stage 2: 0=AD, 1=FTD.
    return np.where(stage1 == 0, 2, stage2).astype(np.int64)


def metric_rows(true: np.ndarray, pred: np.ndarray, participants: list[dict], subjects: np.ndarray) -> list[dict]:
    return [
        {
            **participants[int(subject)],
            "true_id": int(true[i]),
            "true_label": CLASS_NAMES[int(true[i])],
            "predicted_id": int(pred[i]),
            "predicted_label": CLASS_NAMES[int(pred[i])],
            "correct": bool(true[i] == pred[i]),
        }
        for i, subject in enumerate(subjects)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(PREPARED / "spectrograms.npy", mmap_mode="r")
    y_windows = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = json.loads((CNN_RUN / "splits.json").read_text(encoding="utf-8"))
    features, labels = subject_features(x, y_windows, subject_index, participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    configs = configurations()

    stage1_runs = []
    stage1_train_labels = (labels[train] != 2).astype(int)
    stage1_val_labels = (labels[val] != 2).astype(int)
    for config, boost in product(configs, [1.0, 1.5, 2.0]):
        model = make_model(config, boost)
        model.fit(features[train], stage1_train_labels)
        stage1_runs.append({"config": config, "boost": boost, "prediction": model.predict(features[val])})

    disease_train = train[labels[train] != 2]
    stage2_runs = []
    for config, boost in product(configs, [1.0, 1.5, 2.0, 3.0]):
        model = make_model(config, boost)
        model.fit(features[disease_train], labels[disease_train])
        stage2_runs.append({"config": config, "boost": boost, "prediction": model.predict(features[val])})

    candidates = []
    for first, second in product(stage1_runs, stage2_runs):
        prediction = hierarchical_prediction(first["prediction"], second["prediction"])
        candidates.append(
            {
                "stage1": {"config": first["config"], "boost": first["boost"]},
                "stage2": {"config": second["config"], "boost": second["boost"]},
                "val_macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
                "val_ftd_recall": float(np.mean(prediction[labels[val] == 1] == 1)),
            }
        )
    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_ftd_recall"]), reverse=True)
    best = candidates[0]
    print("Selected:", best)

    train_val = np.concatenate([train, val])
    stage1 = make_model(best["stage1"]["config"], best["stage1"]["boost"])
    stage1.fit(features[train_val], (labels[train_val] != 2).astype(int))
    disease_train_val = train_val[labels[train_val] != 2]
    stage2 = make_model(best["stage2"]["config"], best["stage2"]["boost"])
    stage2.fit(features[disease_train_val], labels[disease_train_val])
    prediction = hierarchical_prediction(stage1.predict(features[test]), stage2.predict(features[test]))
    rows = metric_rows(labels[test], prediction, participants, test)
    metrics = calculate_metrics(rows)
    payload = {"selected_on_validation": best, "test": metrics}
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="Hierarchical STFT-SVM test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
