from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MINIMAL = ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "prepared"
ADVANCED = HERE / "outputs" / "prepared"
OUT = HERE / "outputs" / "preprocessing_feature_audit"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


SEED = 20261014
OUTER_SPLITS = 5
INNER_SPLITS = 4
DEFAULT_REPEATS = 5
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.1),
}
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "P7", "P8", "Pz"],
    "temporal": ["T7", "T8"],
    "occipital": ["O1", "O2"],
}
ASYMMETRY_PAIRS = [
    ("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("C3", "C4"),
    ("P3", "P4"), ("P7", "P8"), ("T7", "T8"), ("O1", "O2"),
]
FEATURE_FAMILIES = ["relative_band", "spectral", "regional_connectivity", "all"]
NORMALIZERS = ["global", "site_zscore", "site_shrunk"]
PCA_COMPONENTS = [None, 20]
C_VALUES = [0.03, 0.1, 0.3]
FTD_BOOSTS = [1.0, 1.5]


def participant_labels(prepared: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    window_labels = np.load(prepared / "labels.npy", mmap_mode="r")
    subject_index = np.load(prepared / "subject_index.npy", mmap_mode="r")
    labels = np.asarray(
        [int(window_labels[np.flatnonzero(subject_index == subject)[0]]) for subject in range(len(participants))],
        dtype=np.int64,
    )
    sites = np.asarray([row["site"] for row in participants])
    return participants, labels, sites, subject_index


def alpha_peak(mean_spectrum: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    keep = np.flatnonzero((frequencies >= 7.0) & (frequencies <= 13.0))
    local = mean_spectrum[..., keep]
    peak = np.argmax(local, axis=-1)
    return frequencies[keep[peak]]


def safe_corr(values: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.95, neginf=-0.95)
    upper = np.triu_indices(correlation.shape[0], 1)
    return np.arctanh(np.clip(correlation[upper], -0.95, 0.95))


def extract_subject_features(prepared: Path, tensor_name: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], np.ndarray, np.ndarray]:
    participants, labels, sites, subject_index = participant_labels(prepared)
    tensor = np.load(prepared / tensor_name, mmap_mode="r")
    frequencies = np.linspace(1.0, 40.0, tensor.shape[2])
    channel_index = {name: index for index, name in enumerate(eeg.CHANNELS)}
    region_indexes = [[channel_index[name] for name in channels] for channels in REGIONS.values()]
    band_masks = [np.flatnonzero((frequencies >= low) & (frequencies < high)) for low, high in BANDS.values()]
    family_rows: dict[str, list[np.ndarray]] = defaultdict(list)

    for subject in range(len(participants)):
        windows = np.asarray(tensor[subject_index == subject], dtype=np.float32)
        frames = windows.transpose(0, 3, 1, 2).reshape(-1, len(eeg.CHANNELS), tensor.shape[2])
        stable = np.clip(frames - frames.max(axis=2, keepdims=True), -20.0, 0.0)
        power = np.power(10.0, stable)
        relative = power / np.maximum(power.sum(axis=2, keepdims=True), 1e-12)
        band_series = np.stack([relative[:, :, mask].sum(axis=2) for mask in band_masks], axis=2)
        log_band = np.log(np.maximum(band_series, 1e-10))

        band_mean = band_series.mean(axis=0)
        band_std = band_series.std(axis=0)
        ratio = log_band[:, :, 1] - log_band[:, :, 2]
        ratio_features = np.concatenate([ratio.mean(axis=0), ratio.std(axis=0)])
        mean_spectrum = relative.mean(axis=0)
        channel_iaf = alpha_peak(mean_spectrum, frequencies)
        asymmetry = np.asarray(
            [
                (log_band[:, channel_index[left], band] - log_band[:, channel_index[right], band]).mean()
                for left, right in ASYMMETRY_PAIRS
                for band in range(len(BANDS))
            ],
            dtype=np.float32,
        )

        regional_band = np.stack([band_series[:, indexes].mean(axis=1) for indexes in region_indexes], axis=1)
        regional_log_band = np.log(np.maximum(regional_band, 1e-10))
        regional_mean = regional_band.mean(axis=0)
        regional_std = regional_band.std(axis=0)
        regional_ratio = regional_log_band[:, :, 1] - regional_log_band[:, :, 2]
        regional_ratio_features = np.concatenate([regional_ratio.mean(axis=0), regional_ratio.std(axis=0)])
        regional_spectrum = np.stack([mean_spectrum[indexes].mean(axis=0) for indexes in region_indexes])
        regional_iaf = alpha_peak(regional_spectrum, frequencies)
        connectivity = np.concatenate([safe_corr(regional_log_band[:, :, band]) for band in range(len(BANDS))])

        relative_band = np.concatenate([band_mean.reshape(-1), band_std.reshape(-1)])
        spectral = np.concatenate([relative_band, ratio_features, channel_iaf, asymmetry])
        regional = np.concatenate(
            [regional_mean.reshape(-1), regional_std.reshape(-1), regional_ratio_features, regional_iaf, asymmetry]
        )
        family_rows["relative_band"].append(relative_band)
        family_rows["spectral"].append(spectral)
        family_rows["regional_connectivity"].append(np.concatenate([regional, connectivity]))
        family_rows["all"].append(np.concatenate([spectral, regional_mean.reshape(-1), regional_std.reshape(-1), regional_iaf, connectivity]))

        if (subject + 1) % 30 == 0 or subject + 1 == len(participants):
            print(f"  extracted {subject + 1}/{len(participants)} participants from {tensor_name}", flush=True)

    matrices = {
        name: np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=8.0, neginf=-8.0)
        for name, rows in family_rows.items()
    }
    return matrices, participants, labels, sites


def align_features(
    reference_participants: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    participants: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    lookup = {row["subject_id"]: index for index, row in enumerate(participants)}
    missing = [row["subject_id"] for row in reference_participants if row["subject_id"] not in lookup]
    if missing:
        raise RuntimeError(f"Participants missing from preprocessing version: {missing[:5]}")
    order = np.asarray([lookup[row["subject_id"]] for row in reference_participants], dtype=int)
    return {name: matrix[order] for name, matrix in features.items()}


def normalize_fold(
    matrix: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    evaluate: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.empty((len(train), matrix.shape[1]), dtype=np.float32)
    eval_x = np.empty((len(evaluate), matrix.shape[1]), dtype=np.float32)
    train_sites = sites[train]
    eval_sites = sites[evaluate]
    global_mean = matrix[train].mean(axis=0)
    global_std = np.maximum(matrix[train].std(axis=0), 1e-4)
    for site in sorted(set(train_sites) | set(eval_sites)):
        local = matrix[train][train_sites == site]
        if len(local) == 0 or method == "global":
            mean, std = global_mean, global_std
        elif method == "site_zscore":
            mean = local.mean(axis=0)
            std = np.maximum(local.std(axis=0), 1e-4)
        elif method == "site_shrunk":
            weight = len(local) / (len(local) + 12.0)
            mean = weight * local.mean(axis=0) + (1.0 - weight) * global_mean
            variance = weight * local.var(axis=0) + (1.0 - weight) * np.square(global_std)
            std = np.maximum(np.sqrt(variance), 1e-4)
        else:
            raise ValueError(method)
        train_x[train_sites == site] = (matrix[train][train_sites == site] - mean) / std
        eval_x[eval_sites == site] = (matrix[evaluate][eval_sites == site] - mean) / std
    return train_x, eval_x


def reduce_fold(train_x: np.ndarray, eval_x: np.ndarray, components: int | None) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_x)
    fit = scaler.transform(train_x)
    evaluate = scaler.transform(eval_x)
    if components is not None:
        count = min(components, len(train_x) - 1, train_x.shape[1])
        pca = PCA(n_components=count, svd_solver="randomized", random_state=SEED).fit(fit)
        fit, evaluate = pca.transform(fit), pca.transform(evaluate)
    return fit, evaluate


def weights_for(labels: np.ndarray, boost: float) -> dict[int, float]:
    counts = np.bincount(labels, minlength=len(eeg.CLASSES)).astype(float)
    base = len(labels) / (len(eeg.CLASSES) * np.maximum(counts, 1.0))
    base[1] *= boost
    return {index: float(value) for index, value in enumerate(base)}


def score(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, predicted, average="macro", zero_division=0)),
        "ftd_recall": float(recall_score(true == 1, predicted == 1, zero_division=0)),
    }


def tune(
    features: dict[str, np.ndarray], labels: np.ndarray, sites: np.ndarray, train: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strata = np.asarray([f"{sites[index]}:{labels[index]}" for index in train])
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    records: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for inner_fit_rel, inner_val_rel in splitter.split(train, strata):
        inner_fit = train[inner_fit_rel]
        inner_val = train[inner_val_rel]
        for family in FEATURE_FAMILIES:
            matrix = features[family]
            for normalization in NORMALIZERS:
                fit_x, val_x = normalize_fold(matrix, sites, inner_fit, inner_val, normalization)
                for components in PCA_COMPONENTS:
                    reduced_fit, reduced_val = reduce_fold(fit_x, val_x, components)
                    for c_value in C_VALUES:
                        for ftd_boost in FTD_BOOSTS:
                            model = SVC(
                                C=c_value,
                                kernel="linear",
                                class_weight=weights_for(labels[inner_fit], ftd_boost),
                                random_state=SEED,
                            ).fit(reduced_fit, labels[inner_fit])
                            prediction = model.predict(reduced_val)
                            key = (family, normalization, components, c_value, ftd_boost)
                            records[key].append(score(labels[inner_val], prediction))
    candidates = []
    for key, values in records.items():
        family, normalization, components, c_value, ftd_boost = key
        candidates.append(
            {
                "family": family,
                "normalization": normalization,
                "pca": components,
                "C": c_value,
                "ftd_boost": ftd_boost,
                **{name: float(np.mean([row[name] for row in values])) for name in values[0]},
            }
        )
    candidates.sort(
        key=lambda row: (row["macro_f1"], row["ftd_recall"], row["macro_recall"], row["accuracy"]),
        reverse=True,
    )
    return candidates[0], candidates


def fit_predict(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = features[config["family"]]
    train_x, test_x = normalize_fold(matrix, sites, train, test, config["normalization"])
    train_x, test_x = reduce_fold(train_x, test_x, config["pca"])
    model = SVC(
        C=config["C"],
        kernel="linear",
        class_weight=weights_for(labels[train], config["ftd_boost"]),
        random_state=SEED,
    ).fit(train_x, labels[train])
    return model.predict(test_x), model.decision_function(test_x)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    measures = ["accuracy", "macro_f1", "macro_recall", "ftd_recall"]
    summary = {
        name: {
            "mean": float(np.mean([fold["metrics"][name] for fold in folds])),
            "std": float(np.std([fold["metrics"][name] for fold in folds], ddof=1)),
        }
        for name in measures
    }
    true = np.concatenate([np.asarray(fold["true"], dtype=int) for fold in folds])
    prediction = np.concatenate([np.asarray(fold["prediction"], dtype=int) for fold in folds])
    summary["pooled_confusion_matrix"] = confusion_matrix(true, prediction, labels=[0, 1, 2]).tolist()
    summary["pooled_per_class"] = {
        eeg.CLASSES[label]: {
            "recall": float(recall_score(true == label, prediction == label, zero_division=0)),
            "f1": float(f1_score(true == label, prediction == label, zero_division=0)),
        }
        for label in range(3)
    }
    if folds and "selected" in folds[0]:
        summary["selected_families"] = dict(Counter(fold["selected"]["family"] for fold in folds))
        summary["selected_normalizers"] = dict(Counter(fold["selected"]["normalization"] for fold in folds))
    else:
        summary["selected_families"] = {}
        summary["selected_normalizers"] = {}
    return summary


def cleaning_audit() -> dict[str, Any]:
    rows = json.loads((ADVANCED / "audit.json").read_text(encoding="utf-8"))
    included = [row for row in rows if row["status"] == "included"]
    channels = Counter(channel for row in included for channel in row.get("bad_channels_interpolated", []))
    return {
        "participants": len(included),
        "participants_with_interpolation": sum(bool(row.get("bad_channels_interpolated")) for row in included),
        "total_interpolated_channels": sum(len(row.get("bad_channels_interpolated", [])) for row in included),
        "interpolation_by_channel": dict(channels),
        "participants_with_ica": sum(bool(row.get("ica_components_removed")) for row in included),
        "total_ica_components": sum(len(row.get("ica_components_removed", [])) for row in included),
        "candidate_windows": sum(row.get("candidate_windows", 0) for row in included),
        "retained_windows": sum(row.get("retained_windows", 0) for row in included),
    }


def make_plots(results: dict[str, Any], audit: dict[str, Any]) -> None:
    colors = {"minimal": "#4C78A8", "advanced": "#E45756", "fusion": "#59A14F"}
    measures = ["accuracy", "macro_f1", "macro_recall", "ftd_recall"]
    labels = ["Accuracy", "Macro F1", "Macro recall", "FTD recall"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(len(measures))
    versions = ["minimal", "advanced", "fusion"]
    width = 0.26
    for index, version in enumerate(versions):
        means = [results[version]["summary"][name]["mean"] for name in measures]
        errors = [results[version]["summary"][name]["std"] for name in measures]
        bars = ax.bar(positions + (index - 1) * width, means, width, yerr=errors, capsize=4, label=version.title(), color=colors[version])
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=3, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 0.75), ylabel="Participant-level score", title="Cleaning and compact-feature SVM comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "preprocessing_performance.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, version in zip(axes, versions):
        matrix = np.asarray(results[version]["summary"]["pooled_confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(3):
            for column in range(3):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
        ax.set_xticks(range(3), eeg.CLASSES)
        ax.set_yticks(range(3), eeg.CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=f"{version.title()} cleaning")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "pooled_confusion_matrices.png", dpi=190)
    plt.close(fig)

    counts = np.asarray([audit["interpolation_by_channel"].get(channel, 0) for channel in eeg.CHANNELS])
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.barh(np.asarray(eeg.CHANNELS)[order], counts[order], color="#E45756")
    ax.set(xlabel="Participants whose channel was interpolated", title="Advanced-cleaning channel interpolation")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUT / "interpolation_frequency.png", dpi=190)
    plt.close(fig)

    families = FEATURE_FAMILIES
    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for index, version in enumerate(["minimal", "advanced"]):
        counts = [results[version]["summary"]["selected_families"].get(name, 0) for name in families]
        ax.bar(x + (index - 0.5) * 0.36, counts, 0.36, label=version.title(), color=colors[version])
    ax.set_xticks(x, [name.replace("_", "\n") for name in families])
    ax.set(ylabel="Outer folds selected", title="Feature family selected inside training folds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "feature_selection_frequency.png", dpi=190)
    plt.close(fig)


def write_report(payload: dict[str, Any]) -> None:
    results = payload["results"]
    audit = payload["cleaning_audit"]
    lines = [
        "# Preprocessing and compact-feature SVM audit",
        "",
        "## Protocol",
        "",
        f"- {payload['outer_repeats']} repeats x {OUTER_SPLITS} participant-level outer folds, stratified by site and diagnosis.",
        f"- {INNER_SPLITS}-fold training-only selection of feature family, site normalization, PCA, SVM C, and FTD class weight.",
        "- Both preprocessing versions used identical participants and outer folds.",
        "- Primary selection metric: macro F1; FTD recall was the first tie-breaker.",
        "",
        "## Repeated cross-validation",
        "",
        "| Model | Accuracy | Macro F1 | Macro recall | FTD recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for version in ["minimal", "advanced", "fusion"]:
        summary = results[version]["summary"]
        values = [summary[name] for name in ["accuracy", "macro_f1", "macro_recall", "ftd_recall"]]
        lines.append(
            f"| {version.title()} | " + " | ".join(f"{value['mean']*100:.1f}% +/- {value['std']*100:.1f}" for value in values) + " |"
        )
    difference = payload["paired_difference"]
    lines.extend(
        [
            "",
            f"Advanced minus minimal cleaning changed macro F1 by {difference['macro_f1_mean']*100:+.2f} percentage points on average across paired folds.",
            f"It changed FTD recall by {difference['ftd_recall_mean']*100:+.2f} percentage points.",
            "",
            "## Pooled class results",
            "",
            "| Model | AD F1 / recall | FTD F1 / recall | HC F1 / recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for version in ["minimal", "advanced", "fusion"]:
        per_class = results[version]["summary"]["pooled_per_class"]
        cells = [f"{per_class[name]['f1']*100:.1f}% / {per_class[name]['recall']*100:.1f}%" for name in eeg.CLASSES]
        lines.append(f"| {version.title()} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Cleaning audit",
            "",
            f"- Advanced cleaning interpolated {audit['total_interpolated_channels']} channels across {audit['participants_with_interpolation']} of {audit['participants']} participants.",
            f"- It removed {audit['total_ica_components']} ICA components across {audit['participants_with_ica']} participants.",
            f"- It retained {audit['retained_windows']} of {audit['candidate_windows']} artifact-free candidate windows after limiting each participant to 40 windows.",
            "",
            "## Interpretation",
            "",
        ]
    )
    better = max(results, key=lambda name: results[name]["summary"]["macro_f1"]["mean"])
    lines.append(f"- {better.title()} produced the highest repeated-fold macro F1 in this experiment.")
    lines.append("- Channel interpolation is too frequent to treat as a harmless cleanup step; 114 of 174 participants were altered.")
    lines.append("- Results remain limited by FTD sample size and site shift. The repeated-fold estimate is more credible than the repeatedly inspected fixed test split.")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare preprocessing and compact spectral/connectivity SVMs")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Extracting minimal-cleaning features", flush=True)
    minimal_features, participants, labels, sites = extract_subject_features(MINIMAL, "spectrograms.npy")
    print("Extracting advanced-cleaning features", flush=True)
    advanced_features, advanced_participants, advanced_labels, advanced_sites = extract_subject_features(ADVANCED, "log_stft.npy")
    advanced_features = align_features(participants, advanced_features, advanced_participants)
    advanced_lookup = {row["subject_id"]: index for index, row in enumerate(advanced_participants)}
    advanced_order = np.asarray([advanced_lookup[row["subject_id"]] for row in participants])
    if not np.array_equal(labels, advanced_labels[advanced_order]) or not np.array_equal(sites, advanced_sites[advanced_order]):
        raise RuntimeError("Participant labels or sites differ between preprocessing versions")

    strata = np.asarray([f"{sites[index]}:{labels[index]}" for index in range(len(labels))])
    splitter = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=args.repeats, random_state=SEED)
    outer_folds = list(splitter.split(np.arange(len(labels)), strata))
    versions = {"minimal": minimal_features, "advanced": advanced_features}
    results: dict[str, Any] = {}
    for version, features in versions.items():
        print(f"Evaluating {version} preprocessing", flush=True)
        folds = []
        for fold_number, (train, test) in enumerate(outer_folds, 1):
            selected, _ = tune(features, labels, sites, train)
            prediction, decision = fit_predict(features, labels, sites, train, test, selected)
            metrics = score(labels[test], prediction)
            folds.append(
                {
                    "fold": fold_number,
                    "train": train.tolist(),
                    "test": test.tolist(),
                    "true": labels[test].tolist(),
                    "prediction": prediction.tolist(),
                    "decision": decision.tolist(),
                    "selected": selected,
                    "metrics": metrics,
                }
            )
            print(
                f"  {version} fold {fold_number}/{len(outer_folds)}: "
                f"F1={metrics['macro_f1']:.3f}, FTD recall={metrics['ftd_recall']:.3f}, "
                f"family={selected['family']}",
                flush=True,
            )
        results[version] = {"folds": folds, "summary": summarize(folds)}

    fusion_folds = []
    for minimal_fold, advanced_fold in zip(results["minimal"]["folds"], results["advanced"]["folds"]):
        if minimal_fold["test"] != advanced_fold["test"]:
            raise RuntimeError("Paired preprocessing folds do not match")
        probability = 0.5 * softmax(np.asarray(minimal_fold["decision"])) + 0.5 * softmax(np.asarray(advanced_fold["decision"]))
        prediction = np.argmax(probability, axis=1)
        true = np.asarray(minimal_fold["true"], dtype=int)
        fusion_folds.append(
            {
                "fold": minimal_fold["fold"],
                "test": minimal_fold["test"],
                "true": true.tolist(),
                "prediction": prediction.tolist(),
                "metrics": score(true, prediction),
            }
        )
    results["fusion"] = {"folds": fusion_folds, "summary": summarize(fusion_folds)}

    minimal_f1 = np.asarray([row["metrics"]["macro_f1"] for row in results["minimal"]["folds"]])
    advanced_f1 = np.asarray([row["metrics"]["macro_f1"] for row in results["advanced"]["folds"]])
    minimal_ftd = np.asarray([row["metrics"]["ftd_recall"] for row in results["minimal"]["folds"]])
    advanced_ftd = np.asarray([row["metrics"]["ftd_recall"] for row in results["advanced"]["folds"]])
    try:
        statistic, p_value = wilcoxon(advanced_f1, minimal_f1)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    paired = {
        "macro_f1_mean": float(np.mean(advanced_f1 - minimal_f1)),
        "macro_f1_median": float(np.median(advanced_f1 - minimal_f1)),
        "ftd_recall_mean": float(np.mean(advanced_ftd - minimal_ftd)),
        "wilcoxon_descriptive_statistic": float(statistic),
        "wilcoxon_descriptive_p": float(p_value),
        "note": "Repeated folds are correlated; the Wilcoxon value is descriptive and not a confirmatory significance test.",
    }
    audit = cleaning_audit()
    payload = {
        "participants": len(participants),
        "class_counts": dict(Counter(participant["label"] for participant in participants)),
        "site_counts": dict(Counter(sites)),
        "outer_repeats": args.repeats,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "feature_dimensions": {name: matrix.shape[1] for name, matrix in minimal_features.items()},
        "grid": {
            "feature_families": FEATURE_FAMILIES,
            "normalizers": NORMALIZERS,
            "pca": PCA_COMPONENTS,
            "C": C_VALUES,
            "ftd_boost": FTD_BOOSTS,
        },
        "cleaning_audit": audit,
        "paired_difference": paired,
        "results": results,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_plots(results, audit)
    write_report(payload)
    print(json.dumps({name: row["summary"] for name, row in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
