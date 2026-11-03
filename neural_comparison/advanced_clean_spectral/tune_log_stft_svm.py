"""Tune an EEG log-STFT SVM independently of the log-mel model.

Participant features come from the prepared log-STFT cache. Model selection
uses repeated participant-level cross-validation on development participants;
each fold fits site normalization, scaling, and PCA from its training rows.
The prior held-out participant split is evaluated once after selection.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARED = ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "prepared"
OUT = HERE / "outputs" / "results"
CLASSES = ["AD", "FTD", "HC"]
SEED = 20261009


def subject_features(tensor, window_labels, subject_index, participants):
    mean_rows, mean_std_rows, labels = [], [], []
    for subject in range(len(participants)):
        indexes = np.flatnonzero(subject_index == subject)
        if not len(indexes):
            raise ValueError(f"Participant {subject} has no windows")
        block = np.asarray(tensor[indexes], dtype=np.float32)
        frames = block.transpose(0, 3, 1, 2).reshape(-1, block.shape[1], block.shape[2])
        mean = frames.mean(axis=0)
        std = frames.std(axis=0)
        mean_rows.append(mean.reshape(-1))
        mean_std_rows.append(np.concatenate([mean, std], axis=1).reshape(-1))
        labels.append(int(window_labels[indexes[0]]))
    return {
        "mean": np.asarray(mean_rows, dtype=np.float32),
        "mean_std": np.asarray(mean_std_rows, dtype=np.float32),
    }, np.asarray(labels, dtype=np.int64)


def split_participants(participants):
    ids = np.arange(len(participants))
    strata = np.asarray([p["source"] + ":" + p["label"] for p in participants])
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=20261007)
    train, remaining = next(outer.split(ids, strata))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=20261008)
    val_local, test_local = next(inner.split(remaining, strata[remaining]))
    return np.concatenate([train, remaining[val_local]]), remaining[test_local], (len(train), len(val_local))


def site_transform(train_x, val_x, train_sites, val_sites, mode):
    if mode == "none":
        return train_x, val_x
    result_train = np.empty_like(train_x, dtype=np.float32)
    result_val = np.empty_like(val_x, dtype=np.float32)
    for site in sorted(set(train_sites) | set(val_sites)):
        reference = train_x[train_sites == site]
        if not len(reference):
            raise ValueError(f"No fitting data at site {site}")
        mean = reference.mean(axis=0)
        std = np.maximum(reference.std(axis=0), 1e-3)
        if mode == "center":
            result_train[train_sites == site] = train_x[train_sites == site] - mean
            result_val[val_sites == site] = val_x[val_sites == site] - mean
        elif mode == "zscore":
            result_train[train_sites == site] = (train_x[train_sites == site] - mean) / std
            result_val[val_sites == site] = (val_x[val_sites == site] - mean) / std
        else:
            raise ValueError(mode)
    return result_train, result_val


def prepared_fold(train_x, val_x, train_sites, val_sites, mode):
    train_x, val_x = site_transform(train_x, val_x, train_sites, val_sites, mode)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    # One shared 80-PC basis makes the candidate PCA dimensions comparable.
    pca = PCA(n_components=80, svd_solver="randomized", random_state=SEED)
    train_pc = pca.fit_transform(train_x)
    val_pc = pca.transform(val_x)
    return {None: (train_x, val_x), **{
        size: (train_pc[:, :size], val_pc[:, :size]) for size in (10, 20, 40, 80)
    }}


def candidate_grid():
    models = [("linear", c, "scale") for c in (0.01, 0.1, 1.0, 10.0)]
    models += [("rbf", c, gamma) for c, gamma in itertools.product(
        (0.01, 0.1, 1.0, 10.0), ("scale", 0.01)
    )]
    for feature_mode, site_mode, pca_size, (kernel, c, gamma) in itertools.product(
        ("mean", "mean_std"), ("none", "center", "zscore"),
        (None, 10, 20, 40, 80), models
    ):
        yield (feature_mode, site_mode, pca_size, kernel, c, gamma)


def tune(feature_sets, labels, sites, development, strata):
    candidates = list(candidate_grid())
    scores = {candidate: [] for candidate in candidates}
    accuracies = {candidate: [] for candidate in candidates}
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    for fold_index, (train_local, val_local) in enumerate(folds.split(development, strata[development]), 1):
        train_ids = development[train_local]
        val_ids = development[val_local]
        truth = labels[val_ids]
        for feature_mode, features in feature_sets.items():
            for site_mode in ("none", "center", "zscore"):
                projections = prepared_fold(
                    features[train_ids], features[val_ids], sites[train_ids], sites[val_ids], site_mode
                )
                for pca_size, (train_x, val_x) in projections.items():
                    for kernel, c, gamma in [("linear", c, "scale") for c in (0.01, 0.1, 1.0, 10.0)] + [
                        ("rbf", c, g) for c, g in itertools.product((0.01, 0.1, 1.0, 10.0), ("scale", 0.01))
                    ]:
                        key = (feature_mode, site_mode, pca_size, kernel, c, gamma)
                        model = SVC(C=c, kernel=kernel, gamma=gamma, class_weight="balanced", random_state=SEED)
                        model.fit(train_x, labels[train_ids])
                        prediction = model.predict(val_x)
                        scores[key].append(float(f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0)))
                        accuracies[key].append(float(accuracy_score(truth, prediction)))
        print(f"Completed development CV fold {fold_index}/10", flush=True)

    rows = []
    for key in candidates:
        feature_mode, site_mode, pca_size, kernel, c, gamma = key
        rows.append({
            "feature_mode": feature_mode, "site_mode": site_mode,
            "pca_components": pca_size, "kernel": kernel, "C": c, "gamma": gamma,
            "cv_macro_f1": float(np.mean(scores[key])),
            "cv_macro_f1_std": float(np.std(scores[key])),
            "cv_accuracy": float(np.mean(accuracies[key])),
        })
    rows.sort(key=lambda row: (row["cv_macro_f1"], row["cv_accuracy"]), reverse=True)
    return rows


def final_evaluate(config, feature_sets, labels, sites, participants, development, test):
    features = feature_sets[config["feature_mode"]]
    projections = prepared_fold(
        features[development], features[test], sites[development], sites[test], config["site_mode"]
    )
    train_x, test_x = projections[config["pca_components"]]
    model = SVC(
        C=config["C"], kernel=config["kernel"], gamma=config["gamma"],
        class_weight="balanced", random_state=SEED
    )
    model.fit(train_x, labels[development])
    truth = labels[test]
    prediction = model.predict(test_x)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_recall": float(np.mean(recall)),
        "correct": int(np.sum(truth == prediction)),
        "test_participants": int(len(test)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASSES)
        },
        "test_predictions": [
            {"subject_id": participants[index]["subject_id"], "actual": CLASSES[int(labels[index])],
             "predicted": CLASSES[int(pred)]}
            for index, pred in zip(test, prediction)
        ],
    }


def save_plot(best, test_metrics, path):
    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    values = [test_metrics[k] for k in ("accuracy", "macro_f1", "macro_recall")]
    axes[0].bar(["Accuracy", "Macro F1", "Macro recall"], values, color="#3476a6")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Held-out scores · {test_metrics['correct']}/27 correct")
    axes[1].imshow(matrix, cmap="Blues")
    for row in range(3):
        for col in range(3):
            axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center")
    axes[1].set_xticks(range(3), CLASSES)
    axes[1].set_yticks(range(3), CLASSES)
    axes[1].set(xlabel="Predicted", ylabel="Actual", title="Log-STFT SVM confusion matrix")
    site_label = "no site normalization" if best["site_mode"] == "none" else best["site_mode"] + " site normalization"
    pca_label = "no PCA" if best["pca_components"] is None else f"{best['pca_components']} PCs"
    fig.suptitle(f"Development CV selected: {best['feature_mode']}, {site_label}, "
                 f"{pca_label}, {best['kernel']} SVM")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    tensor = np.load(PREPARED / "spectrograms.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    feature_sets, labels = subject_features(tensor, window_labels, subject_index, participants)
    sites = np.asarray([p["site"] for p in participants])
    strata = np.asarray([p["source"] + ":" + p["label"] for p in participants])
    development, test, (train_count, val_count) = split_participants(participants)
    print(f"Participants: development={len(development)} ({train_count}+{val_count}), test={len(test)}", flush=True)
    rows = tune(feature_sets, labels, sites, development, strata)
    best = rows[0]
    print("Selected from development CV:", json.dumps(best), flush=True)
    test_metrics = final_evaluate(best, feature_sets, labels, sites, participants, development, test)
    print("Held-out:", json.dumps({k: test_metrics[k] for k in ("accuracy", "macro_f1", "macro_recall", "correct")}), flush=True)
    payload = {
        "experiment": "Log-STFT SVM selected independently of log-mel",
        "source": str(PREPARED / "spectrograms.npy"),
        "participants": len(participants),
        "development_participants": len(development),
        "test_participants": len(test),
        "selection": "2x repeated 5-fold participant CV, stratified by dataset source and class; macro F1 primary",
        "candidate_count": len(rows),
        "best_config": best,
        "test": test_metrics,
        "limitation": "The test split was used in earlier project experiments, so this remains exploratory.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "tuned_log_stft_svm_metrics.json"
    table_path = OUT / "tuned_log_stft_svm_cv.csv"
    plot_path = OUT / "tuned_log_stft_svm.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(table_path, index=False)
    save_plot(best, test_metrics, plot_path)
    print("Saved:", json_path, table_path, plot_path, sep="\n", flush=True)


if __name__ == "__main__":
    main()
