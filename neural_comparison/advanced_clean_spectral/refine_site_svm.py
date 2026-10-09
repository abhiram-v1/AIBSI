from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "results" / "svm_refinement"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


PAIRS = [(0, 1), (2, 3), (4, 8), (7, 11), (9, 13), (12, 16), (14, 15), (17, 18)]
BANDS = [(0, 4), (4, 8), (8, 13), (13, 18), (18, 24)]


def participant_feature_sets(tensor: np.ndarray, subject_index: np.ndarray, count: int) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    upper = np.triu_indices(19, 1)
    for subject in range(count):
        windows = np.asarray(tensor[subject_index == subject], dtype=np.float32)
        samples = windows.transpose(0, 3, 1, 2).reshape(-1, 19, 24)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        median = np.median(samples, axis=0)
        iqr = np.percentile(samples, 75, axis=0) - np.percentile(samples, 25, axis=0)
        moments = np.concatenate([mean, std], axis=1).reshape(-1)
        robust = np.concatenate([median, iqr], axis=1).reshape(-1)

        relative_samples = samples - samples.mean(axis=2, keepdims=True)
        relative_mean = relative_samples.mean(axis=0)
        relative_std = relative_samples.std(axis=0)
        relative = np.concatenate([relative_mean, relative_std], axis=1).reshape(-1)
        asymmetry = np.stack([relative_mean[left] - relative_mean[right] for left, right in PAIRS]).reshape(-1)

        connectivity = []
        for start, stop in BANDS:
            band_series = relative_samples[:, :, start:stop].mean(axis=2)
            corr = np.corrcoef(band_series, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.99, neginf=-0.99)
            connectivity.append(np.arctanh(np.clip(corr[upper], -0.99, 0.99)))
        connectivity = np.concatenate(connectivity)

        output["moments"].append(moments)
        output["robust"].append(robust)
        output["relative"].append(np.concatenate([relative, asymmetry]))
        output["relative_connectivity"].append(np.concatenate([relative, asymmetry, connectivity]))
        output["all_spectral"].append(np.concatenate([moments, robust, asymmetry]))
    return {name: np.asarray(rows, dtype=np.float32) for name, rows in output.items()}


def site_transform(
    train_x: np.ndarray,
    other_x: np.ndarray,
    train_sites: np.ndarray,
    other_sites: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    global_mean = train_x.mean(axis=0)
    global_std = np.maximum(train_x.std(axis=0), 1e-3)
    transformed_train = np.empty_like(train_x)
    transformed_other = np.empty_like(other_x)
    for site in sorted(set(np.concatenate([train_sites, other_sites]))):
        local = train_x[train_sites == site]
        if len(local) == 0:
            mean, std = global_mean, global_std
        elif method == "site_zscore":
            mean = local.mean(axis=0)
            std = np.maximum(local.std(axis=0), 1e-3)
        elif method == "site_center":
            mean = local.mean(axis=0)
            std = global_std
        elif method == "site_shrunk":
            weight = len(local) / (len(local) + 10.0)
            mean = weight * local.mean(axis=0) + (1.0 - weight) * global_mean
            variance = weight * local.var(axis=0) + (1.0 - weight) * global_std**2
            std = np.maximum(np.sqrt(variance), 1e-3)
        elif method == "global":
            mean, std = global_mean, global_std
        else:
            raise ValueError(method)
        transformed_train[train_sites == site] = (train_x[train_sites == site] - mean) / std
        transformed_other[other_sites == site] = (other_x[other_sites == site] - mean) / std
    return transformed_train, transformed_other


def class_weight(name: str) -> str | dict[int, float]:
    if name == "balanced":
        return "balanced"
    boost = float(name.replace("ftd", ""))
    return {0: 1.0, 1: boost, 2: 1.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tensor = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, val])
    labels = np.asarray([
        int(window_labels[np.flatnonzero(subject_index == subject)[0]]) for subject in range(len(participants))
    ])
    sites = np.asarray([row["site"] for row in participants])
    strata = np.asarray([f"{sites[index]}:{labels[index]}" for index in development])
    features = participant_feature_sets(tensor, subject_index, len(participants))

    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=eeg.SEED)
    folds = list(splitter.split(development, strata))
    pca_values = [10, 15, 20, 30]
    c_values = [0.01, 0.03, 0.1, 0.3, 1.0]
    normalizers = ["site_zscore", "site_center", "site_shrunk", "global"]
    weights = ["balanced", "ftd1.25", "ftd1.5"]
    records = []

    total_families = len(features) * len(normalizers)
    family_number = 0
    for feature_name, matrix in features.items():
        dev_x = matrix[development]
        for normalization in normalizers:
            family_number += 1
            print(f"[{family_number}/{total_families}] {feature_name} + {normalization}", flush=True)
            scores = {
                (pca_count, c, weight): np.zeros((len(development), 3), dtype=np.float64)
                for pca_count in pca_values for c in c_values for weight in weights
            }
            counts = np.zeros(len(development), dtype=np.float64)
            for fit_local, predict_local in folds:
                fit_global = development[fit_local]
                predict_global = development[predict_local]
                fit_x, predict_x = site_transform(
                    dev_x[fit_local], dev_x[predict_local], sites[fit_global], sites[predict_global], normalization
                )
                scaler = StandardScaler().fit(fit_x)
                fit_scaled = scaler.transform(fit_x)
                predict_scaled = scaler.transform(predict_x)
                for pca_count in pca_values:
                    pca = PCA(n_components=pca_count, svd_solver="randomized", random_state=eeg.SEED).fit(fit_scaled)
                    fit_reduced = pca.transform(fit_scaled)
                    predict_reduced = pca.transform(predict_scaled)
                    for c in c_values:
                        for weight in weights:
                            model = SVC(C=c, kernel="linear", class_weight=class_weight(weight), random_state=eeg.SEED)
                            model.fit(fit_reduced, labels[fit_global])
                            scores[(pca_count, c, weight)][predict_local] += model.decision_function(predict_reduced)
                counts[predict_local] += 1
            for (pca_count, c, weight), decision_sum in scores.items():
                prediction = np.argmax(decision_sum / counts[:, None], axis=1)
                records.append({
                    "feature_set": feature_name,
                    "normalization": normalization,
                    "pca": pca_count,
                    "C": c,
                    "class_weight": weight,
                    "cv_macro_f1": float(f1_score(labels[development], prediction, average="macro", zero_division=0)),
                    "cv_accuracy": float(accuracy_score(labels[development], prediction)),
                    "cv_prediction": prediction.tolist(),
                })

    records.sort(key=lambda row: (row["cv_macro_f1"], row["cv_accuracy"], -row["pca"]), reverse=True)
    best = records[0]
    selected_x = features[best["feature_set"]]
    fit_x, test_x = site_transform(
        selected_x[development], selected_x[test], sites[development], sites[test], best["normalization"]
    )
    scaler = StandardScaler().fit(fit_x)
    fit_scaled = scaler.transform(fit_x)
    test_scaled = scaler.transform(test_x)
    pca = PCA(n_components=best["pca"], svd_solver="randomized", random_state=eeg.SEED).fit(fit_scaled)
    model = SVC(
        C=best["C"], kernel="linear", class_weight=class_weight(best["class_weight"]), random_state=eeg.SEED
    ).fit(pca.transform(fit_scaled), labels[development])
    test_prediction = model.predict(pca.transform(test_scaled))
    test_metrics = eeg.metrics(labels[test], test_prediction)

    baseline = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    payload = {
        "selection": "5-fold repeated twice on the 147 development participants; held-out labels excluded",
        "selected": {key: value for key, value in best.items() if key != "cv_prediction"},
        "test": test_metrics,
        "baseline": baseline,
        "top_candidates": [{key: value for key, value in row.items() if key != "cv_prediction"} for row in records[:30]],
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected": payload["selected"], "test": test_metrics}, indent=2), flush=True)

    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="Refined SVM confusion matrix")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    measures = ["accuracy", "macro_f1", "macro_recall"]
    positions = np.arange(2)
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            positions + (offset - 1) * 0.24,
            [baseline[measure], test_metrics[measure]],
            0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2)
    axes[1].set_xticks(positions, ["Previous site-normalized\nSVM", "Cross-validated\nrefined SVM"])
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="SVM refinement")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
