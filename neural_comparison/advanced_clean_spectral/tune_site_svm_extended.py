from __future__ import annotations

import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "site_svm_extended_tuning"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


PCA_COMPONENTS = [10, 15, 20, 30, 40, 60, 80, None]
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
RBF_GAMMAS: list[str | float] = ["scale", 0.001, 0.003, 0.01, 0.03, 0.1]
FTD_BOOSTS = [1.0, 1.25, 1.5, 2.0]
CV_SEEDS = [20261009, 20261029]


def site_normalize(
    features: np.ndarray,
    participants: list[dict],
    fit: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    sites = np.asarray([row["site"] for row in participants])
    result = np.empty_like(features, dtype=np.float32)
    parameters = {}
    for site in sorted(set(sites)):
        reference = features[fit[sites[fit] == site]].astype(np.float64)
        mean = reference.mean(axis=0)
        std = np.maximum(reference.std(axis=0), 1e-3)
        result[sites == site] = ((features[sites == site] - mean) / std).astype(np.float32)
        parameters[site] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return result, parameters


def class_weights(labels: np.ndarray, boost: float) -> dict[int, float]:
    counts = np.bincount(labels, minlength=3).astype(float)
    weights = len(labels) / (3.0 * counts)
    weights[1] *= boost
    return {index: float(value) for index, value in enumerate(weights)}


def configs() -> list[dict[str, Any]]:
    result = []
    for components, c, boost in itertools.product(PCA_COMPONENTS, C_VALUES, FTD_BOOSTS):
        result.append({"pca": components, "kernel": "linear", "C": c, "gamma": "scale", "ftd_boost": boost})
        for gamma in RBF_GAMMAS:
            result.append({"pca": components, "kernel": "rbf", "C": c, "gamma": gamma, "ftd_boost": boost})
    return result


def selection_key(row: dict[str, Any]) -> tuple:
    return (
        row["cv_macro_f1"],
        row["cv_macro_recall"],
        row["cv_accuracy"],
        row["kernel"] == "linear",
        -(row["pca"] if row["pca"] is not None else 10_000),
        -row["C"],
        -row["ftd_boost"],
    )


def main() -> None:
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    tensor = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    participant_features, labels = eeg.subject_features(tensor, window_labels, subject_index, participants)
    features = participant_features.reshape(len(participant_features), -1)
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    validation = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, validation])
    strata = np.asarray([f"{participants[index]['source']}:{labels[index]}" for index in development])
    choices = configs()
    predictions = np.full((len(choices), len(CV_SEEDS), len(development)), -1, dtype=np.int8)
    fold_rows = []

    for repeat, seed in enumerate(CV_SEEDS):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (fit_local, valid_local) in enumerate(folds.split(development, strata)):
            fit = development[fit_local]
            valid = development[valid_local]
            normalized, _ = site_normalize(features, participants, fit)
            scaler = StandardScaler().fit(normalized[fit])
            scaled = scaler.transform(normalized)
            representations: dict[int | None, np.ndarray] = {None: scaled}
            for components in [value for value in PCA_COMPONENTS if value is not None]:
                representations[components] = PCA(n_components=components, random_state=eeg.SEED).fit(scaled[fit]).transform(scaled)
            weights_cache = {boost: class_weights(labels[fit], boost) for boost in FTD_BOOSTS}
            for candidate_index, config in enumerate(choices):
                model = SVC(
                    C=config["C"],
                    kernel=config["kernel"],
                    gamma=config["gamma"],
                    class_weight=weights_cache[config["ftd_boost"]],
                    decision_function_shape="ovr",
                    random_state=eeg.SEED,
                )
                values = representations[config["pca"]]
                model.fit(values[fit], labels[fit])
                predictions[candidate_index, repeat, valid_local] = model.predict(values[valid]).astype(np.int8)
            fold_rows.append({"repeat": repeat, "fold": fold, "fit": fit.tolist(), "validation": valid.tolist()})
            print(f"extended SVM CV repeat={repeat + 1}/2 fold={fold + 1}/5 complete", flush=True)

    if np.any(predictions < 0):
        raise RuntimeError("Some development predictions are missing")
    truth = np.tile(labels[development], len(CV_SEEDS))
    candidates = []
    for config, prediction in zip(choices, predictions):
        flattened = prediction.reshape(-1)
        candidates.append(
            {
                **config,
                "cv_macro_f1": float(f1_score(truth, flattened, average="macro", zero_division=0)),
                "cv_macro_recall": float(recall_score(truth, flattened, average="macro", zero_division=0)),
                "cv_accuracy": float(accuracy_score(truth, flattened)),
            }
        )
    candidates.sort(key=selection_key, reverse=True)
    selected = candidates[0]
    print(f"selected={selected}", flush=True)

    normalized, normalization = site_normalize(features, participants, development)
    scaler = StandardScaler().fit(normalized[development])
    scaled = scaler.transform(normalized)
    pca = None
    transformed = scaled
    if selected["pca"] is not None:
        pca = PCA(n_components=selected["pca"], random_state=eeg.SEED).fit(scaled[development])
        transformed = pca.transform(scaled)
    final_model = SVC(
        C=selected["C"],
        kernel=selected["kernel"],
        gamma=selected["gamma"],
        class_weight=class_weights(labels[development], selected["ftd_boost"]),
        decision_function_shape="ovr",
        probability=False,
        random_state=eeg.SEED,
    ).fit(transformed[development], labels[development])
    prediction = final_model.predict(transformed[test])
    test_metrics = eeg.metrics(labels[test], prediction)
    artifact = {
        "normalization": normalization,
        "scaler": scaler,
        "pca": pca,
        "svm": final_model,
        "selected": selected,
        "development_subjects": development,
        "test_subjects": test,
    }
    joblib.dump(artifact, OUT / "model.joblib")

    original = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    blend = json.loads((HERE / "outputs" / "deep_models" / "eegpt_attention" / "svm_blend_metrics.json").read_text(encoding="utf-8"))["test"]
    payload = {
        "method": "extended tuning of the same site-normalized moderate-log-mel SVM",
        "candidate_count": len(choices),
        "development_cv": "2x5 stratified participant folds; site normalization, scaling, and PCA refitted per fold",
        "selected": selected,
        "test": test_metrics,
        "original_svm_test": original,
        "folds": fold_rows,
        "top_candidates": candidates[:50],
        "runtime_seconds": time.perf_counter() - began,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    matrix = np.asarray(test_metrics["confusion_matrix"])
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="Extended-tuned SVM: held-out participants")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    names = ["Original SVM", "Extended-tuned SVM", "SVM + EEGPT"]
    model_metrics = [original, test_metrics, blend]
    positions = np.arange(len(names))
    for offset, measure in enumerate(["accuracy", "macro_f1", "macro_recall"]):
        bars = axes[1].bar(
            positions + (offset - 1) * 0.24,
            [item[measure] for item in model_metrics],
            width=0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    axes[1].set_xticks(positions, names)
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="SVM hyperparameter tuning")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"selected": selected, "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
