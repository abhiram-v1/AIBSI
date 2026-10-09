from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.covariance import oas
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD_DIR = ROOT / "neural_comparison" / "stft_gpu_3class"
OLD_PREPARED = OLD_DIR / "outputs" / "prepared"
OUT = HERE / "outputs" / "riemannian_svm"
PREPARED = OUT / "prepared"
sys.path.insert(0, str(OLD_DIR))
import run_pipeline as raw_eeg  # noqa: E402
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
]
BAND_SETS = {
    "all": [0, 1, 2, 3, 4],
    "slow": [0, 1, 2],
    "theta_alpha_beta": [1, 2, 3],
    "delta_to_beta": [0, 1, 2, 3],
    "theta": [1],
    "alpha": [2],
}


def sym(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) * 0.5


def spd_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(sym(matrix))
    values = np.maximum(values, 1e-8)
    return (vectors * (values**power)) @ vectors.T


def spd_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(sym(matrix))
    values = np.maximum(values, 1e-8)
    return (vectors * np.log(values)) @ vectors.T


def spd_exp(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(sym(matrix))
    return (vectors * np.exp(np.clip(values, -30, 30))) @ vectors.T


def log_euclidean_mean(matrices: np.ndarray) -> np.ndarray:
    return spd_exp(np.mean([spd_log(matrix) for matrix in matrices], axis=0))


def prepare_covariances() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    output = PREPARED / "covariances.npz"
    if output.exists():
        print(f"Reusing {output}")
        return
    mne.set_log_level("ERROR")
    manifest = json.loads((OLD_PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    records = {record.subject_id: record for record in raw_eeg.discover_records()}
    filters = [butter(4, [low, high], btype="bandpass", fs=raw_eeg.FS, output="sos") for _, low, high in BANDS]
    raw_covariances = []
    trace_covariances = []
    audits: list[dict[str, Any]] = []

    for participant in tqdm(participants, desc="Filter-bank covariance", unit="participant"):
        record = records[participant["subject_id"]]
        data, header = raw_eeg.load_harmonized(record)
        start = raw_eeg.EDGE_SECONDS * raw_eeg.FS
        stop = data.shape[1] - raw_eeg.EDGE_SECONDS * raw_eeg.FS
        offsets = []
        for offset in range(start, stop - raw_eeg.WINDOW_SAMPLES + 1, raw_eeg.WINDOW_SAMPLES):
            valid, _ = raw_eeg.valid_window(data[:, offset : offset + raw_eeg.WINDOW_SAMPLES])
            if valid:
                offsets.append(offset)
        offsets = raw_eeg.evenly_limit(offsets, raw_eeg.MAX_WINDOWS_PER_SUBJECT)
        if len(offsets) < 5:
            raise RuntimeError(f"{record.subject_id}: only {len(offsets)} valid windows")

        subject_raw = []
        subject_trace = []
        for filt in filters:
            filtered = sosfiltfilt(filt, data, axis=1)
            window_covariances = []
            window_trace_covariances = []
            for offset in offsets:
                segment = filtered[:, offset : offset + raw_eeg.WINDOW_SAMPLES].astype(np.float64) * 1e6
                segment -= segment.mean(axis=1, keepdims=True)
                covariance, _ = oas(segment.T, assume_centered=True)
                covariance = sym(covariance) + np.eye(covariance.shape[0]) * 1e-6
                window_covariances.append(covariance)
                window_trace_covariances.append(covariance / max(np.trace(covariance) / len(covariance), 1e-8))
            subject_raw.append(log_euclidean_mean(np.asarray(window_covariances)))
            subject_trace.append(log_euclidean_mean(np.asarray(window_trace_covariances)))
        raw_covariances.append(subject_raw)
        trace_covariances.append(subject_trace)
        audits.append({**asdict(record), **header, "retained_windows": len(offsets)})

    np.savez_compressed(
        output,
        raw=np.asarray(raw_covariances, dtype=np.float32),
        trace=np.asarray(trace_covariances, dtype=np.float32),
    )
    (PREPARED / "manifest.json").write_text(
        json.dumps({"bands": BANDS, "participants": participants, "audits": audits}, indent=2), encoding="utf-8"
    )
    print(f"Saved {output}")


def tangent_features(covariances: np.ndarray, reference_indexes: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(covariances.shape[-1])
    off_diagonal = upper[0] != upper[1]
    bands = []
    for band in range(covariances.shape[1]):
        reference = log_euclidean_mean(covariances[reference_indexes, band])
        inverse_root = spd_power(reference, -0.5)
        vectors = []
        for covariance in covariances[:, band]:
            tangent = spd_log(inverse_root @ covariance @ inverse_root)
            vector = tangent[upper]
            vector[off_diagonal] *= np.sqrt(2.0)
            vectors.append(vector)
        bands.append(np.asarray(vectors))
    return np.stack(bands, axis=1).astype(np.float32)


def site_transform(
    train_x: np.ndarray,
    other_x: np.ndarray,
    train_sites: np.ndarray,
    other_sites: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_result = np.empty_like(train_x)
    other_result = np.empty_like(other_x)
    global_mean = train_x.mean(axis=0)
    global_std = np.maximum(train_x.std(axis=0), 1e-3)
    for site in sorted(set(np.concatenate([train_sites, other_sites]))):
        local = train_x[train_sites == site]
        if method == "site_zscore" and len(local):
            mean, std = local.mean(axis=0), np.maximum(local.std(axis=0), 1e-3)
        elif method == "site_center" and len(local):
            mean, std = local.mean(axis=0), global_std
        else:
            mean, std = global_mean, global_std
        train_result[train_sites == site] = (train_x[train_sites == site] - mean) / std
        other_result[other_sites == site] = (other_x[other_sites == site] - mean) / std
    return train_result, other_result


def evaluate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = np.load(PREPARED / "covariances.npz")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    labels = np.asarray([eeg.LABEL_ID[row["label"]] for row in participants])
    sites = np.asarray([row["site"] for row in participants])
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, val])
    strata = np.asarray([f"{sites[index]}:{labels[index]}" for index in development])
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=eeg.SEED).split(development, strata))

    configs = []
    for covariance_name in ["raw", "trace"]:
        for band_set in BAND_SETS:
            for normalization in ["site_zscore", "site_center", "global"]:
                for pca_count in [10, 20, 30, 40, None]:
                    for c in [0.01, 0.1, 1.0, 10.0]:
                        for kernel in ["linear", "rbf"]:
                            configs.append({
                                "covariance": covariance_name,
                                "band_set": band_set,
                                "normalization": normalization,
                                "pca": pca_count,
                                "C": c,
                                "kernel": kernel,
                                "score_sum": np.zeros((len(development), 3), dtype=np.float64),
                                "counts": np.zeros(len(development), dtype=np.float64),
                            })

    for fold_number, (fit_local, predict_local) in enumerate(folds, 1):
        fit_global = development[fit_local]
        predict_global = development[predict_local]
        print(f"Cross-validation fold {fold_number}/{len(folds)}", flush=True)
        tangent_cache = {
            name: tangent_features(arrays[name].astype(np.float64), fit_global) for name in ["raw", "trace"]
        }
        transform_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        pca_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for config in configs:
            bands = BAND_SETS[config["band_set"]]
            transform_key = (config["covariance"], config["band_set"], config["normalization"])
            if transform_key not in transform_cache:
                matrix = tangent_cache[config["covariance"]][:, bands].reshape(len(participants), -1)
                fit_x, predict_x = site_transform(
                    matrix[fit_global], matrix[predict_global], sites[fit_global], sites[predict_global], config["normalization"]
                )
                scaler = StandardScaler().fit(fit_x)
                transform_cache[transform_key] = (scaler.transform(fit_x), scaler.transform(predict_x))
            fit_x, predict_x = transform_cache[transform_key]
            pca_key = (*transform_key, config["pca"])
            if pca_key not in pca_cache:
                if config["pca"] is None:
                    pca_cache[pca_key] = (fit_x, predict_x)
                else:
                    reducer = PCA(n_components=config["pca"], svd_solver="randomized", random_state=eeg.SEED).fit(fit_x)
                    pca_cache[pca_key] = (reducer.transform(fit_x), reducer.transform(predict_x))
            reduced_fit, reduced_predict = pca_cache[pca_key]
            model = SVC(
                C=config["C"], kernel=config["kernel"], gamma="scale", class_weight="balanced",
                decision_function_shape="ovr", random_state=eeg.SEED,
            ).fit(reduced_fit, labels[fit_global])
            config["score_sum"][predict_local] += model.decision_function(reduced_predict)
            config["counts"][predict_local] += 1

    ranked = []
    for config in configs:
        prediction = np.argmax(config["score_sum"] / config["counts"][:, None], axis=1)
        ranked.append({
            **{key: config[key] for key in ["covariance", "band_set", "normalization", "pca", "C", "kernel"]},
            "cv_macro_f1": float(f1_score(labels[development], prediction, average="macro", zero_division=0)),
            "cv_accuracy": float(accuracy_score(labels[development], prediction)),
        })
    ranked.sort(key=lambda row: (row["cv_macro_f1"], row["cv_accuracy"], row["pca"] is not None), reverse=True)
    best = ranked[0]

    covariances = arrays[best["covariance"]].astype(np.float64)
    tangent = tangent_features(covariances, development)
    matrix = tangent[:, BAND_SETS[best["band_set"]]].reshape(len(participants), -1)
    fit_x, test_x = site_transform(matrix[development], matrix[test], sites[development], sites[test], best["normalization"])
    scaler = StandardScaler().fit(fit_x)
    fit_x, test_x = scaler.transform(fit_x), scaler.transform(test_x)
    if best["pca"] is not None:
        reducer = PCA(n_components=best["pca"], svd_solver="randomized", random_state=eeg.SEED).fit(fit_x)
        fit_x, test_x = reducer.transform(fit_x), reducer.transform(test_x)
    model = SVC(
        C=best["C"], kernel=best["kernel"], gamma="scale", class_weight="balanced", random_state=eeg.SEED
    ).fit(fit_x, labels[development])
    prediction = model.predict(test_x)
    test_metrics = eeg.metrics(labels[test], prediction)
    baseline = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    payload = {
        "selection": "filter-bank settings selected with repeated five-fold development-only cross-validation",
        "selected": best,
        "test": test_metrics,
        "baseline": baseline,
        "top_candidates": ranked[:30],
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best, "test": test_metrics}, indent=2), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    confusion = np.asarray(test_metrics["confusion_matrix"])
    image = axes[0].imshow(confusion, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(confusion[row, column]), ha="center", va="center", fontsize=12)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="Filter-bank Riemannian SVM")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    measures = ["accuracy", "macro_f1", "macro_recall"]
    positions = np.arange(2)
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            positions + (offset - 1) * 0.24, [baseline[measure], test_metrics[measure]], 0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2)
    axes[1].set_xticks(positions, ["Log-mel SVM", "Riemannian SVM"])
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Participant-level SVM comparison")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter-bank Riemannian tangent-space SVM")
    parser.add_argument("command", choices=["prepare", "evaluate", "all"], nargs="?", default="all")
    args = parser.parse_args()
    if args.command in {"prepare", "all"}:
        prepare_covariances()
    if args.command in {"evaluate", "all"}:
        evaluate()


if __name__ == "__main__":
    main()
