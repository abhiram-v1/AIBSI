from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import oas
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "results" / "site_shift"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


def matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    values = np.maximum(values, 1e-5)
    return (vectors * (values**power)) @ vectors.T


def harmonize(
    train_x: np.ndarray,
    other_x: np.ndarray,
    train_sites: np.ndarray,
    other_sites: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "global":
        return train_x, other_x
    train_result = np.empty_like(train_x)
    other_result = np.empty_like(other_x)
    global_mean = train_x.mean(axis=0)
    global_std = np.maximum(train_x.std(axis=0), 1e-3)
    global_cov, _ = oas(train_x - global_mean, assume_centered=True)
    global_root = matrix_power(global_cov, 0.5)
    for site in sorted(set(np.concatenate([train_sites, other_sites]))):
        local = train_x[train_sites == site]
        if len(local) < 3:
            mean, std = global_mean, global_std
            train_result[train_sites == site] = (train_x[train_sites == site] - mean) / std
            other_result[other_sites == site] = (other_x[other_sites == site] - mean) / std
            continue
        mean = local.mean(axis=0)
        if method == "pca_center":
            train_result[train_sites == site] = train_x[train_sites == site] - mean
            other_result[other_sites == site] = other_x[other_sites == site] - mean
        elif method == "pca_zscore":
            std = np.maximum(local.std(axis=0), 1e-3)
            train_result[train_sites == site] = (train_x[train_sites == site] - mean) / std
            other_result[other_sites == site] = (other_x[other_sites == site] - mean) / std
        elif method == "coral":
            covariance, _ = oas(local - mean, assume_centered=True)
            transform = matrix_power(covariance, -0.5) @ global_root
            train_result[train_sites == site] = (train_x[train_sites == site] - mean) @ transform
            other_result[other_sites == site] = (other_x[other_sites == site] - mean) @ transform
        else:
            raise ValueError(method)
    return train_result, other_result


def prepare_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    tensor = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    features, labels = eeg.subject_features(tensor, window_labels, subject_index, participants)
    sites = np.asarray([row["site"] for row in participants])
    return features.reshape(len(features), -1), labels, sites, participants


def transform_fold(
    matrix: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    sites: np.ndarray,
    components: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(matrix[fit])
    fit_scaled = scaler.transform(matrix[fit])
    predict_scaled = scaler.transform(matrix[predict])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=eeg.SEED).fit(fit_scaled)
    fit_reduced = pca.transform(fit_scaled)
    predict_reduced = pca.transform(predict_scaled)
    return harmonize(fit_reduced, predict_reduced, sites[fit], sites[predict], method)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix, labels, sites, participants = prepare_features()
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, val])
    strata = np.asarray([f"{sites[index]}:{labels[index]}" for index in development])
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=eeg.SEED).split(development, strata))
    configs = []
    for method in ["global", "pca_center", "pca_zscore", "coral"]:
        for components in [10, 15, 20, 30, 40]:
            for c in [0.01, 0.03, 0.1, 0.3, 1.0]:
                for kernel in ["linear", "rbf"]:
                    configs.append({
                        "method": method,
                        "pca": components,
                        "C": c,
                        "kernel": kernel,
                        "scores": np.zeros((len(development), 3), dtype=np.float64),
                        "counts": np.zeros(len(development), dtype=np.float64),
                    })

    for fold_number, (fit_local, predict_local) in enumerate(folds, 1):
        print(f"Site-harmonization fold {fold_number}/{len(folds)}", flush=True)
        fit = development[fit_local]
        predict = development[predict_local]
        cache = {}
        for config in configs:
            key = (config["method"], config["pca"])
            if key not in cache:
                cache[key] = transform_fold(matrix, fit, predict, sites, config["pca"], config["method"])
            fit_x, predict_x = cache[key]
            model = SVC(
                C=config["C"], kernel=config["kernel"], gamma="scale", class_weight="balanced",
                decision_function_shape="ovr", random_state=eeg.SEED,
            ).fit(fit_x, labels[fit])
            config["scores"][predict_local] += model.decision_function(predict_x)
            config["counts"][predict_local] += 1

    ranked = []
    for config in configs:
        prediction = np.argmax(config["scores"] / config["counts"][:, None], axis=1)
        ranked.append({
            "method": config["method"], "pca": config["pca"], "C": config["C"], "kernel": config["kernel"],
            "cv_macro_f1": float(f1_score(labels[development], prediction, average="macro", zero_division=0)),
            "cv_accuracy": float(accuracy_score(labels[development], prediction)),
        })
    ranked.sort(key=lambda row: (row["cv_macro_f1"], row["cv_accuracy"]), reverse=True)
    best = ranked[0]
    fit_x, test_x = transform_fold(matrix, development, test, sites, best["pca"], best["method"])
    final = SVC(
        C=best["C"], kernel=best["kernel"], gamma="scale", class_weight="balanced", random_state=eeg.SEED
    ).fit(fit_x, labels[development])
    test_prediction = final.predict(test_x)
    test_metrics = eeg.metrics(labels[test], test_prediction)

    leave_site_out = {}
    for target_site in sorted(set(sites)):
        fit = np.flatnonzero(sites != target_site)
        predict = np.flatnonzero(sites == target_site)
        # An unseen site has no training reference. This intentionally uses only
        # global scaling to measure genuine cross-site transfer.
        fit_x, predict_x = transform_fold(matrix, fit, predict, sites, 20, "global")
        model = SVC(C=0.1, kernel="linear", class_weight="balanced", random_state=eeg.SEED).fit(fit_x, labels[fit])
        leave_site_out[target_site] = eeg.metrics(labels[predict], model.predict(predict_x))

    within_site = {}
    for site in sorted(set(sites)):
        indexes = np.flatnonzero(sites == site)
        splits_site = StratifiedKFold(n_splits=5, shuffle=True, random_state=eeg.SEED)
        prediction = np.full(len(indexes), -1, dtype=int)
        for fit_local, predict_local in splits_site.split(indexes, labels[indexes]):
            fit, predict = indexes[fit_local], indexes[predict_local]
            fit_x, predict_x = transform_fold(matrix, fit, predict, sites, 20, "global")
            model = SVC(C=0.1, kernel="linear", class_weight="balanced", random_state=eeg.SEED).fit(fit_x, labels[fit])
            prediction[predict_local] = model.predict(predict_x)
        within_site[site] = eeg.metrics(labels[indexes], prediction)

    baseline = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    payload = {
        "selected_harmonization": best,
        "harmonized_test": test_metrics,
        "baseline_test": baseline,
        "leave_one_site_out": leave_site_out,
        "within_site_cross_validation": within_site,
        "top_candidates": ranked[:30],
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "top_candidates"}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    sites_order = sorted(leave_site_out)
    x = np.arange(len(sites_order))
    width = 0.36
    bars = axes[0].bar(x - width / 2, [within_site[s]["accuracy"] for s in sites_order], width, label="Within-site CV")
    axes[0].bar_label(bars, fmt="%.2f", padding=2)
    bars = axes[0].bar(x + width / 2, [leave_site_out[s]["accuracy"] for s in sites_order], width, label="Site held out")
    axes[0].bar_label(bars, fmt="%.2f", padding=2)
    axes[0].set_xticks(x, sites_order)
    axes[0].set(ylim=(0, 1), ylabel="Accuracy", title="Within-site versus cross-site transfer")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    measures = ["accuracy", "macro_f1", "macro_recall"]
    rows = [baseline, test_metrics]
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            np.arange(2) + (offset - 1) * 0.24, [row[measure] for row in rows], 0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2)
    axes[1].set_xticks(np.arange(2), ["Current site-normalized", "PCA-space harmonized"])
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Site harmonization experiment")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT / "site_shift_analysis.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
