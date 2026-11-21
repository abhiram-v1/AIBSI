from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "prepared"
OUT = HERE / "outputs" / "mfcc_svm"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


def site_normalize(features: np.ndarray, participants: list[dict], fit: np.ndarray) -> np.ndarray:
    result = np.empty_like(features, dtype=np.float32)
    sites = np.asarray([participant["site"] for participant in participants])
    for site in sorted(set(sites)):
        reference = features[fit[sites[fit] == site]]
        if not len(reference):
            raise RuntimeError(f"No training participant from site {site}")
        mean = reference.mean(axis=0, keepdims=True)
        std = np.maximum(reference.std(axis=0, keepdims=True), 1e-3)
        result[sites == site] = (features[sites == site] - mean) / std
    return result


def model(config: dict, probability: bool = False) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if config["pca"] is not None:
        steps.append(("pca", PCA(n_components=config["pca"], random_state=eeg.SEED)))
    steps.append(
        (
            "svm",
            SVC(
                C=config["C"],
                kernel=config["kernel"],
                gamma="scale",
                class_weight="balanced",
                probability=probability,
                random_state=eeg.SEED,
            ),
        )
    )
    return Pipeline(steps)


def extract_all_mfcc_summaries(
    log_mel: np.ndarray,
    labels_windows: np.ndarray,
    subject_index: np.ndarray,
    participants: list[dict],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    shape = (len(participants), len(eeg.CHANNELS), log_mel.shape[2])
    summaries = {
        "mean": np.empty(shape, dtype=np.float32),
        "std": np.empty(shape, dtype=np.float32),
        "median": np.empty(shape, dtype=np.float32),
        "iqr": np.empty(shape, dtype=np.float32),
    }
    subject_labels = np.empty(len(participants), dtype=np.int64)
    for subject in range(len(participants)):
        windows = np.asarray(log_mel[subject_index == subject], dtype=np.float32)
        # DCT-II along the mel-band axis creates 24 cepstral coefficients.
        coefficients = dct(windows, type=2, axis=2, norm="ortho")
        repeats = coefficients.transpose(0, 3, 1, 2).reshape(
            -1, coefficients.shape[1], coefficients.shape[2]
        )
        summaries["mean"][subject] = repeats.mean(axis=0)
        summaries["std"][subject] = repeats.std(axis=0)
        summaries["median"][subject] = np.median(repeats, axis=0)
        q75, q25 = np.percentile(repeats, [75, 25], axis=0)
        summaries["iqr"][subject] = q75 - q25
        subject_labels[subject] = int(labels_windows[np.flatnonzero(subject_index == subject)[0]])
    return summaries, subject_labels


def feature_matrix(
    summaries: dict[str, np.ndarray], n_mfcc: int, include_c0: bool, summary_set: str
) -> np.ndarray:
    start = 0 if include_c0 else 1
    stop = start + n_mfcc
    names = ["mean", "std"] if summary_set == "mean_std" else ["mean", "std", "median", "iqr"]
    blocks = [summaries[name][:, :, start:stop] for name in names]
    return np.concatenate(blocks, axis=2).reshape(len(blocks[0]), -1).astype(np.float32)


def score(true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "macro_f1": float(f1_score(true, prediction, average="macro", zero_division=0)),
    }


def save_figures(test_metrics: dict, baseline: dict | None) -> None:
    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="MFCC SVM: held-out participants")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    recalls = [test_metrics["per_class"][name]["recall"] for name in eeg.CLASSES]
    class_f1 = [test_metrics["per_class"][name]["f1"] for name in eeg.CLASSES]
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    bars = ax.bar(positions - 0.18, recalls, 0.36, label="Recall")
    ax.bar_label(bars, fmt="%.2f", padding=2)
    bars = ax.bar(positions + 0.18, class_f1, 0.36, label="F1")
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(positions, eeg.CLASSES)
    ax.set(ylim=(0, 1), ylabel="Score", title="MFCC SVM performance by class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "per_class.png", dpi=180)
    plt.close(fig)

    if baseline is not None:
        labels = ["Log-mel SVM", "MFCC SVM"]
        rows = [baseline, test_metrics]
        measures = ["accuracy", "balanced_accuracy", "macro_f1"]
        positions = np.arange(2)
        fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
        for offset, measure in enumerate(measures):
            bars = ax.bar(
                positions + (offset - 1) * 0.23,
                [row[measure] for row in rows],
                0.23,
                label=measure.replace("_", " ").title(),
            )
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        ax.set_xticks(positions, labels)
        ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Log-mel and MFCC comparison")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.savefig(OUT / "comparison.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_mel = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    labels_windows = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)

    summaries, labels = extract_all_mfcc_summaries(
        log_mel, labels_windows, subject_index, participants
    )

    representation_configs = [
        {"n_mfcc": n, "include_c0": c0, "summary_set": summary_set}
        for n, c0, summary_set in itertools.product(
            [8, 12, 16, 20], [False, True], ["mean_std", "four_statistics"]
        )
    ]
    svm_configs = [
        {"pca": pca, "C": c, "kernel": kernel}
        for pca, c, kernel in itertools.product(
            [None, 20, 40],
            [0.01, 0.03, 0.1, 0.3, 1.0],
            ["linear", "rbf"],
        )
    ]

    candidates = []
    matrices: dict[tuple, np.ndarray] = {}
    total = len(representation_configs) * len(svm_configs)
    completed = 0
    for representation in representation_configs:
        key = (representation["n_mfcc"], representation["include_c0"], representation["summary_set"])
        raw = feature_matrix(summaries, **representation)
        matrices[key] = raw
        normalized = site_normalize(raw, participants, train)
        for svm_config in svm_configs:
            estimator = model(svm_config)
            estimator.fit(normalized[train], labels[train])
            prediction = estimator.predict(normalized[val])
            metrics = score(labels[val], prediction)
            candidates.append({**representation, **svm_config, **metrics})
            completed += 1
        print(f"Completed {completed}/{total} validation configurations", flush=True)

    candidates.sort(
        key=lambda row: (row["macro_f1"], row["balanced_accuracy"], row["accuracy"]),
        reverse=True,
    )
    best = candidates[0]
    best_key = (best["n_mfcc"], best["include_c0"], best["summary_set"])
    raw = matrices[best_key]
    train_val = np.concatenate([train, val])
    normalized = site_normalize(raw, participants, train_val)
    final_config = {name: best[name] for name in ["pca", "C", "kernel"]}
    estimator = model(final_config, probability=True)
    estimator.fit(normalized[train_val], labels[train_val])
    prediction = estimator.predict(normalized[test])
    probabilities = estimator.predict_proba(normalized[test])
    test_metrics = eeg.metrics(labels[test], prediction)

    baseline_path = HERE / "outputs" / "results" / "site_normalized_metrics.json"
    baseline = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["test"]

    payload = {
        "protocol": {
            "participants": len(participants),
            "windows": int(len(log_mel)),
            "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
            "selection_metric": "validation macro F1",
            "test_used_for_selection": False,
            "input": "DCT-II of existing 24-band log-mel tensors",
        },
        "selected_on_validation": best,
        "selected_feature_columns": int(raw.shape[1]),
        "test": test_metrics,
        "baseline_log_mel_svm": baseline,
        "top_validation_candidates": candidates[:25],
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(OUT / "mfcc_subject_matrix.npy", raw)
    np.savez_compressed(
        OUT / "test_predictions.npz",
        test_indices=test,
        true=labels[test],
        predicted=prediction,
        probabilities=probabilities,
    )
    joblib.dump(
        {
            "model": estimator,
            "selected": best,
            "train_val_indices": train_val,
            "participants": participants,
        },
        OUT / "model.joblib",
    )
    save_figures(test_metrics, baseline)

    recall = {name: test_metrics["per_class"][name]["recall"] for name in eeg.CLASSES}
    report = [
        "# MFCC SVM experiment",
        "",
        "MFCCs were obtained by an orthonormal DCT-II across the existing 24 log-mel bands. "
        "All representation and SVM settings were chosen on the 26-person validation split; "
        "the 27-person test split was evaluated once after selection.",
        "",
        "## Selected configuration",
        "",
        f"- MFCC coefficients: {best['n_mfcc']}",
        f"- Include coefficient 0: {best['include_c0']}",
        f"- Summaries: {best['summary_set']}",
        f"- Feature columns: {raw.shape[1]}",
        f"- PCA: {best['pca']}",
        f"- SVM kernel: {best['kernel']}",
        f"- SVM C: {best['C']}",
        f"- Validation macro F1: {best['macro_f1']:.4f}",
        "",
        "## Held-out participant result",
        "",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
        f"- Macro F1: {test_metrics['macro_f1']:.4f}",
        f"- Recall AD / FTD / HC: {recall['AD']:.4f} / {recall['FTD']:.4f} / {recall['HC']:.4f}",
        f"- Confusion matrix: {test_metrics['confusion_matrix']}",
    ]
    if baseline is not None:
        report.extend(
            [
                "",
                "## Existing log-mel SVM reference",
                "",
                f"- Accuracy: {baseline['accuracy']:.4f}",
                f"- Balanced accuracy: {baseline['balanced_accuracy']:.4f}",
                f"- Macro F1: {baseline['macro_f1']:.4f}",
            ]
        )
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected_on_validation": best, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
