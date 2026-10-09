from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "tensor_train_svm"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tensor_pipeline" / "v2"))
import pipeline as eeg  # noqa: E402
from models import TTProjection  # noqa: E402


RANKS = [4, 8, 16, 32]
CHANNEL_RANKS = [4, 8]
SPECTRAL_RANKS = [8, 16, 24]
SVM_C = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
SVM_KERNELS = ["linear", "rbf"]
CV_SEEDS = [20261009, 20261029]


def participant_tensors(
    windows: np.ndarray,
    labels: np.ndarray,
    subject_index: np.ndarray,
    participants: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    tensors, targets = [], []
    for subject in range(len(participants)):
        value = windows[subject_index == subject].astype(np.float32)
        # Resting-state frames and windows are repeated observations rather than
        # synchronized events. Summaries retain electrode x frequency structure.
        repeated = value.transpose(0, 3, 1, 2).reshape(-1, value.shape[1], value.shape[2])
        mean = repeated.mean(axis=0)
        std = repeated.std(axis=0)
        median = np.median(repeated, axis=0)
        iqr = np.quantile(repeated, 0.75, axis=0) - np.quantile(repeated, 0.25, axis=0)
        tensors.append(np.stack([mean, std, median, iqr], axis=-1))
        targets.append(int(labels[np.flatnonzero(subject_index == subject)[0]]))
    result = np.asarray(tensors, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Participant tensor contains nonfinite values")
    return result, np.asarray(targets, dtype=np.int64)


def site_normalize(
    tensor: np.ndarray,
    participants: list[dict],
    fit: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    sites = np.asarray([row["site"] for row in participants])
    normalized = np.empty_like(tensor, dtype=np.float32)
    parameters: dict[str, dict[str, np.ndarray]] = {}
    for site in sorted(set(sites)):
        reference = tensor[fit[sites[fit] == site]].astype(np.float64)
        if not len(reference):
            raise RuntimeError(f"No training participants from site {site}")
        mean = reference.mean(axis=0)
        std = np.maximum(reference.std(axis=0), 1e-3)
        normalized[sites == site] = ((tensor[sites == site] - mean) / std).astype(np.float32)
        parameters[site] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return normalized, parameters


def classifier(c: float, kernel: str) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    C=c,
                    kernel=kernel,
                    gamma="scale",
                    class_weight="balanced",
                    decision_function_shape="ovr",
                    random_state=eeg.SEED,
                ),
            ),
        ]
    )


def candidate_key(row: dict) -> tuple:
    return (
        row["cv_macro_f1"],
        row["cv_macro_recall"],
        row["cv_accuracy"],
        -row["rank"],
        -row["spectral_rank"],
        -row["channel_rank"],
        -row["C"],
        row["kernel"] == "linear",
    )


def main() -> None:
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    windows = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    tensors, targets = participant_tensors(windows, labels, subject_index, participants)
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    validation = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, validation])
    strata = np.asarray([f"{participants[index]['source']}:{targets[index]}" for index in development])

    configs = [
        {
            "rank": rank,
            "channel_rank": channel_rank,
            "spectral_rank": spectral_rank,
            "C": c,
            "kernel": kernel,
        }
        for rank, channel_rank, spectral_rank, c, kernel in itertools.product(
            RANKS, CHANNEL_RANKS, SPECTRAL_RANKS, SVM_C, SVM_KERNELS
        )
    ]
    predictions = [list() for _ in configs]
    truths = [list() for _ in configs]
    fold_rows = []

    for repeat, seed in enumerate(CV_SEEDS):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (fit_local, valid_local) in enumerate(cv.split(development, strata)):
            fit = development[fit_local]
            valid = development[valid_local]
            normalized, _ = site_normalize(tensors, participants, fit)
            feature_cache: dict[tuple[int, int, int], np.ndarray] = {}
            for rank, channel_rank, spectral_rank in itertools.product(RANKS, CHANNEL_RANKS, SPECTRAL_RANKS):
                key = (rank, channel_rank, spectral_rank)
                decomposition = TTProjection(
                    rank=rank,
                    channel_rank=channel_rank,
                    spectral_rank=spectral_rank,
                ).fit(normalized[fit])
                feature_cache[key] = decomposition.transform(normalized)
            for candidate_index, config in enumerate(configs):
                features = feature_cache[(config["rank"], config["channel_rank"], config["spectral_rank"])]
                model = classifier(config["C"], config["kernel"])
                model.fit(features[fit], targets[fit])
                prediction = model.predict(features[valid])
                predictions[candidate_index].extend(prediction.tolist())
                truths[candidate_index].extend(targets[valid].tolist())
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "fit_subjects": fit.tolist(),
                    "validation_subjects": valid.tolist(),
                }
            )
            print(f"development CV repeat={repeat + 1}/2 fold={fold + 1}/5 complete", flush=True)

    candidates = []
    for config, truth, prediction in zip(configs, truths, predictions):
        truth_array = np.asarray(truth)
        prediction_array = np.asarray(prediction)
        candidates.append(
            {
                **config,
                "cv_macro_f1": float(f1_score(truth_array, prediction_array, average="macro", zero_division=0)),
                "cv_macro_recall": float(recall_score(truth_array, prediction_array, average="macro", zero_division=0)),
                "cv_accuracy": float(accuracy_score(truth_array, prediction_array)),
            }
        )
    candidates.sort(key=candidate_key, reverse=True)
    selected = candidates[0]
    print(f"selected={selected}", flush=True)

    normalized, normalization = site_normalize(tensors, participants, development)
    decomposition = TTProjection(
        rank=selected["rank"],
        channel_rank=selected["channel_rank"],
        spectral_rank=selected["spectral_rank"],
    ).fit(normalized[development])
    features = decomposition.transform(normalized)
    model = classifier(selected["C"], selected["kernel"])
    model.fit(features[development], targets[development])
    prediction = model.predict(features[test])
    test_metrics = eeg.metrics(targets[test], prediction)

    artifact = {
        "normalization": normalization,
        "decomposition": decomposition,
        "classifier": model,
        "development_subjects": development,
        "test_subjects": test,
        "channels": manifest["channels"],
        "statistics": ["mean", "std", "median", "iqr"],
    }
    joblib.dump(artifact, OUT / "tt_svm_model.joblib")
    np.save(OUT / "participant_tensors.npy", tensors)
    payload: dict[str, Any] = {
        "method": "training-only common Tensor Train projection + class-balanced SVM",
        "participant_tensor_shape": list(tensors.shape),
        "participant_split": {key: len(value) for key, value in splits.items()},
        "development_cv": "2x5 stratified participant folds; every fold refits site normalization and TT",
        "candidate_count": len(configs),
        "selected": selected,
        "tt_feature_count": int(features.shape[1]),
        "tt_orthogonality_error": decomposition.orthogonality_error_,
        "tt_relative_centered_reconstruction_error": decomposition.relative_centered_reconstruction_error_,
        "test": test_metrics,
        "development_folds": fold_rows,
        "rank_summary": {
            str(rank): max(row["cv_macro_f1"] for row in candidates if row["rank"] == rank)
            for rank in RANKS
        },
        "top_candidates": candidates[:20],
        "runtime_seconds": time.perf_counter() - began,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    svm = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    blend = json.loads((HERE / "outputs" / "deep_models" / "eegpt_attention" / "svm_blend_metrics.json").read_text(encoding="utf-8"))["test"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    matrix = np.asarray(test_metrics["confusion_matrix"])
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="Tensor Train + SVM: held-out participants")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    names = ["Spectral SVM", "TT + SVM", "SVM + EEGPT"]
    values = [svm, test_metrics, blend]
    positions = np.arange(len(names))
    for offset, measure in enumerate(["accuracy", "macro_f1", "macro_recall"]):
        bars = axes[1].bar(
            positions + (offset - 1) * 0.24,
            [item[measure] for item in values],
            width=0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    axes[1].set_xticks(positions, names)
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Tensor decomposition comparison")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.savefig(OUT / "tt_svm_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    rank_scores = [payload["rank_summary"][str(rank)] for rank in RANKS]
    bars = ax.bar([str(rank) for rank in RANKS], rank_scores, color="#4c78a8")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylim=(0, 1), xlabel="Terminal TT rank", ylabel="Best development macro F1", title="Tensor Train rank comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "tt_rank_selection.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"selected": selected, "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
