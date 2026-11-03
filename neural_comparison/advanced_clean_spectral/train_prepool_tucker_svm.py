"""Training-only Tucker/HOSVD before participant pooling, followed by an SVM.

The decomposition is learned from training-participant log-mel windows only.
Ranks and classifier settings are selected on the existing validation subjects;
the selected pipeline is then refit on all development subjects and evaluated on
the unchanged participant-level test split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs/moderate_log_mel/prepared"
SPLITS = ROOT / "neural_comparison/stft_gpu_3class/outputs/run/splits.json"
OUT = HERE / "outputs/prepool_tucker_svm"

RANKS = ((4, 8, 3), (6, 12, 4), (8, 16, 5))
SUMMARIES = ("mean_std", "robust4")
NORMALIZATIONS = ("global", "site_center")
PCA_COMPONENTS = (20, 40, 80)
C_VALUES = (0.01, 0.1, 1.0, 10.0)
KERNELS = ("linear", "rbf")
CLASS_NAMES = ("AD", "FTD", "HC")
SEED = 20261007
BATCH = 128


def participant_labels(window_labels: np.ndarray, owners: np.ndarray, count: int) -> np.ndarray:
    labels = np.empty(count, dtype=np.int64)
    for subject in range(count):
        values = np.unique(window_labels[owners == subject])
        if len(values) != 1:
            raise RuntimeError(f"Participant {subject} has inconsistent labels: {values}")
        labels[subject] = values[0]
    return labels


def training_windows(owners: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(owners, subjects))


def fit_hosvd(
    tensor: np.ndarray, owners: np.ndarray, subjects: np.ndarray, ranks: tuple[int, int, int]
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    indexes = training_windows(owners, subjects)
    mean = np.zeros(tensor.shape[1:], dtype=np.float64)
    for start in range(0, len(indexes), BATCH):
        mean += tensor[indexes[start : start + BATCH]].astype(np.float32).sum(axis=0)
    mean = (mean / len(indexes)).astype(np.float32)

    channel_cov = np.zeros((tensor.shape[1], tensor.shape[1]), dtype=np.float64)
    frequency_cov = np.zeros((tensor.shape[2], tensor.shape[2]), dtype=np.float64)
    time_cov = np.zeros((tensor.shape[3], tensor.shape[3]), dtype=np.float64)
    for number, start in enumerate(range(0, len(indexes), BATCH), 1):
        block = tensor[indexes[start : start + BATCH]].astype(np.float32) - mean
        channel_cov += np.einsum("ncft,ndft->cd", block, block, optimize=True)
        frequency_cov += np.einsum("ncft,ncgt->fg", block, block, optimize=True)
        time_cov += np.einsum("ncft,ncfu->tu", block, block, optimize=True)
        if number % 10 == 0:
            print(f"  HOSVD covariance: {min(start + BATCH, len(indexes))}/{len(indexes)} windows", flush=True)

    factors = []
    retained: dict[str, float] = {}
    for name, covariance, rank in zip(
        ("channel", "frequency", "time"),
        (channel_cov, frequency_cov, time_cov),
        ranks,
        strict=True,
    ):
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        vectors = vectors[:, order]
        factors.append(vectors[:, :rank].astype(np.float32))
        retained[name] = float(values[:rank].sum() / max(values.sum(), 1e-12))
    return mean, tuple(factors), retained


def project(
    tensor: np.ndarray,
    mean: np.ndarray,
    factors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    channel, frequency, temporal = factors
    shape = (len(tensor), channel.shape[1], frequency.shape[1], temporal.shape[1])
    output = np.empty(shape, dtype=np.float32)
    for number, start in enumerate(range(0, len(tensor), BATCH), 1):
        block = tensor[start : start + BATCH].astype(np.float32) - mean
        output[start : start + len(block)] = np.einsum(
            "ncft,ca,fb,td->nabd",
            block,
            channel,
            frequency,
            temporal,
            optimize=True,
        )
        if number % 10 == 0:
            print(f"  Projecting: {min(start + BATCH, len(tensor))}/{len(tensor)} windows", flush=True)
    return output


def summarize(
    cores: np.ndarray,
    owners: np.ndarray,
    participant_count: int,
    ranks: tuple[int, int, int],
    method: str,
) -> np.ndarray:
    rows = []
    rc, rf, rt = ranks
    for subject in range(participant_count):
        values = cores[owners == subject, :rc, :rf, :rt].reshape(np.sum(owners == subject), -1)
        if method == "mean_std":
            row = np.concatenate([values.mean(axis=0), values.std(axis=0)])
        elif method == "robust4":
            row = np.concatenate(
                [
                    values.mean(axis=0),
                    values.std(axis=0),
                    np.median(values, axis=0),
                    np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0),
                ]
            )
        else:
            raise ValueError(method)
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite participant features")
    return matrix


def site_center(
    features: np.ndarray,
    fit: np.ndarray,
    sites: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "global":
        return features.copy()
    if method != "site_center":
        raise ValueError(method)
    result = features.copy()
    fallback = features[fit].mean(axis=0)
    for site in np.unique(sites):
        local = fit[sites[fit] == site]
        center = features[local].mean(axis=0) if len(local) else fallback
        result[sites == site] -= center
    return result


def make_model(components: int, c_value: float, kernel: str, rows: int, columns: int) -> Pipeline:
    count = min(components, rows - 1, columns)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=count, random_state=SEED)),
            (
                "svm",
                SVC(
                    C=c_value,
                    kernel=kernel,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(3), zero_division=0
    )
    return {
        "subjects": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(3)).tolist(),
    }


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    complexity = np.prod(row["ranks"]) * (4 if row["summary"] == "robust4" else 2)
    return (-row["macro_f1"], -row["balanced_accuracy"], -row["accuracy"], complexity, row["pca"], row["C"])


def plot_results(result: dict[str, Any]) -> None:
    baseline = json.loads((HERE / "outputs/results/site_shift/metrics.json").read_text(encoding="utf-8"))["baseline_test"]
    old_tt = json.loads((HERE / "outputs/tensor_train_svm/metrics.json").read_text(encoding="utf-8"))["test"]
    new = result["test"]
    labels = ["Log-mel SVM", "Summary-first TT", "Pre-pooling Tucker"]
    rows = [baseline, old_tt, new]
    measures = (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy"))
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.24
    colors = ("#4F81BD", "#C0504D", "#70AD47")
    for offset, ((key, title), color) in enumerate(zip(measures, colors, strict=True)):
        values = [row[key] for row in rows]
        bars = ax.bar(positions + (offset - 1) * width, values, width, label=title, color=color)
        ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=3, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Held-out participant score")
    ax.set_title("Decomposition order: fixed-split comparison")
    ax.legend(loc="upper center", ncols=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(new["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Pre-pooling Tucker + SVM")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    tensor = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    owners = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    train = np.asarray(splits["train"], dtype=int)
    validation = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, validation])
    labels = participant_labels(window_labels, owners, len(participants))
    sites = np.asarray([row["site"] for row in participants])

    print("Fitting train-only shared Tucker/HOSVD basis", flush=True)
    maximum = tuple(max(rank[axis] for rank in RANKS) for axis in range(3))
    mean, factors, train_retained = fit_hosvd(tensor, owners, train, maximum)
    cores = project(tensor, mean, factors)

    matrices = {
        (ranks, summary): summarize(cores, owners, len(participants), ranks, summary)
        for ranks in RANKS
        for summary in SUMMARIES
    }
    candidates = []
    for (ranks, summary), features in matrices.items():
        for normalization in NORMALIZATIONS:
            normalized = site_center(features, train, sites, normalization)
            for components in PCA_COMPONENTS:
                for c_value in C_VALUES:
                    for kernel in KERNELS:
                        model = make_model(components, c_value, kernel, len(train), normalized.shape[1])
                        model.fit(normalized[train], labels[train])
                        prediction = model.predict(normalized[validation])
                        score = metrics(labels[validation], prediction)
                        candidates.append(
                            {
                                "ranks": list(ranks),
                                "summary": summary,
                                "normalization": normalization,
                                "pca": components,
                                "C": c_value,
                                "kernel": kernel,
                                "feature_count": int(features.shape[1]),
                                "accuracy": score["accuracy"],
                                "balanced_accuracy": score["balanced_accuracy"],
                                "macro_f1": score["macro_f1"],
                            }
                        )
    candidates.sort(key=candidate_key)
    selected = candidates[0]
    print("Selected on validation:", json.dumps(selected, indent=2), flush=True)

    selected_ranks = tuple(selected["ranks"])
    print("Refitting selected Tucker basis on all development participants", flush=True)
    final_mean, final_factors, final_retained = fit_hosvd(tensor, owners, development, selected_ranks)
    final_cores = project(tensor, final_mean, final_factors)
    final_features = summarize(
        final_cores,
        owners,
        len(participants),
        selected_ranks,
        selected["summary"],
    )
    final_normalized = site_center(final_features, development, sites, selected["normalization"])
    model = make_model(
        selected["pca"], selected["C"], selected["kernel"], len(development), final_features.shape[1]
    )
    model.fit(final_normalized[development], labels[development])
    prediction = model.predict(final_normalized[test])
    test_metrics = metrics(labels[test], prediction)

    result = {
        "method": "training-only shared Tucker/HOSVD on window x electrode x mel x STFT-frame data before participant pooling",
        "input_window_shape": list(tensor.shape[1:]),
        "participant_split": {name: len(splits[name]) for name in ("train", "val", "test")},
        "selection_endpoint": "validation participant macro F1; balanced accuracy and accuracy are tie breakers",
        "candidate_count": len(candidates),
        "selected": selected,
        "train_basis_mode_energy_retained_at_maximum_ranks": train_retained,
        "final_basis_mode_energy_retained": final_retained,
        "test": test_metrics,
        "test_subjects": test.tolist(),
        "test_predictions": prediction.tolist(),
        "test_truth": labels[test].tolist(),
        "top_candidates": candidates[:20],
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "The fixed test set was inspected in earlier experiments, so this is exploratory rather than an unbiased publication estimate.",
            "Tucker/HOSVD is unsupervised and may retain acquisition-site variance rather than diagnosis-specific variance.",
            "Independent evidence remains the number of participants, not the number of EEG windows.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_results(result)
    print(json.dumps({"selected": selected, "test": test_metrics, "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
