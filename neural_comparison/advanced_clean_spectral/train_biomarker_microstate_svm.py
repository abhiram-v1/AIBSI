"""Periodic/aperiodic spectral biomarkers plus training-only EEG microstates.

This fixed-split experiment uses raw four-second windows from all 174
participants. A robust log-log spectral background fit produces periodic and
aperiodic features. Four polarity-invariant microstate templates are learned
from training participants only. ANOVA feature selection and an SVM are chosen
on validation macro F1, refit on all development participants, and evaluated on
the unchanged participant test split.
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
from scipy.signal import find_peaks, welch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import HuberRegressor
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
PREPARED = HERE / "outputs/eegpt/prepared"
SPLITS = ROOT / "neural_comparison/stft_gpu_3class/outputs/run/splits.json"
OUT = HERE / "outputs/biomarker_microstate_svm"

FS = 256
MICROSTATE_FS = 128
MICROSTATE_COUNT = 4
PEAK_MAPS_PER_WINDOW = 12
MAX_TEMPLATE_MAPS = 60000
CLASS_NAMES = ("AD", "FTD", "HC")
CHANNEL_REGIONS = {
    "frontal": ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"),
    "temporal": ("T7", "T8", "P7", "P8"),
    "central": ("C3", "C4", "Cz"),
    "parietal": ("P3", "P4", "Pz"),
    "occipital": ("O1", "O2"),
}
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
PEAK_BANDS = {"theta": (4.0, 8.0), "alpha": (7.0, 14.0), "beta": (13.0, 30.0)}
SEED = 20261007


def participant_labels(window_labels: np.ndarray, owners: np.ndarray, count: int) -> np.ndarray:
    labels = np.empty(count, dtype=np.int64)
    for subject in range(count):
        values = np.unique(window_labels[owners == subject])
        if len(values) != 1:
            raise RuntimeError(f"Inconsistent labels for participant {subject}: {values}")
        labels[subject] = values[0]
    return labels


def peak_width(frequencies: np.ndarray, residual: np.ndarray, peak_index: int) -> float:
    height = residual[peak_index]
    if height <= 0:
        return 0.0
    threshold = height / 2.0
    left = peak_index
    right = peak_index
    while left > 0 and residual[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(residual) and residual[right + 1] >= threshold:
        right += 1
    return float(frequencies[right] - frequencies[left] + 0.5)


def spectral_feature_names(channels: list[str]) -> tuple[list[str], list[str]]:
    base = ["aperiodic_exponent", "aperiodic_offset", "aperiodic_fit_r2"]
    base += [f"relative_{band}_power" for band in BANDS]
    base += ["log_theta_alpha_ratio", "log_slow_fast_ratio"]
    for band in PEAK_BANDS:
        base += [f"{band}_peak_hz", f"{band}_peak_height", f"{band}_peak_width_hz"]
    names = [f"channel__{channel}__{feature}" for channel in channels for feature in base]
    names += [f"region__{region}__{feature}" for region in CHANNEL_REGIONS for feature in base]
    names += [f"global__channel_mean__{feature}" for feature in base]
    names += [f"global__channel_std__{feature}" for feature in base]
    return base, names


def extract_spectral_features(
    raw: np.ndarray, owners: np.ndarray, participant_count: int, channels: list[str]
) -> tuple[np.ndarray, list[str]]:
    base_names, names = spectral_feature_names(channels)
    channel_index = {channel: index for index, channel in enumerate(channels)}
    region_indexes = {
        region: [channel_index[channel] for channel in members]
        for region, members in CHANNEL_REGIONS.items()
    }
    rows = []
    for subject in range(participant_count):
        windows = raw[owners == subject].astype(np.float32)
        frequencies, psd = welch(
            windows,
            fs=FS,
            window="hann",
            nperseg=512,
            noverlap=256,
            detrend="constant",
            axis=-1,
        )
        spectrum = np.maximum(psd.mean(axis=0), 1e-12)
        keep = (frequencies >= 1.0) & (frequencies <= 40.0)
        frequencies_kept = frequencies[keep]
        spectrum = spectrum[:, keep]
        log_spectrum = np.log10(spectrum)
        total_power = np.maximum(spectrum.sum(axis=1), 1e-12)
        channel_rows = []
        for electrode in range(len(channels)):
            fit_mask = (frequencies_kept >= 2.0) & (frequencies_kept <= 40.0)
            fit_mask &= ~((frequencies_kept >= 7.0) & (frequencies_kept <= 14.0))
            x = np.log10(frequencies_kept[fit_mask])[:, None]
            y = log_spectrum[electrode, fit_mask]
            model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=200).fit(x, y)
            baseline = model.intercept_ + model.coef_[0] * np.log10(frequencies_kept)
            residual = log_spectrum[electrode] - baseline
            prediction = model.predict(x)
            denominator = np.sum(np.square(y - y.mean()))
            r2 = 1.0 - np.sum(np.square(y - prediction)) / max(denominator, 1e-12)
            relative = {}
            for band, (low, high) in BANDS.items():
                mask = (frequencies_kept >= low) & (frequencies_kept < high)
                relative[band] = float(spectrum[electrode, mask].sum() / total_power[electrode])
            values = [float(-model.coef_[0]), float(model.intercept_), float(r2)]
            values += [relative[band] for band in BANDS]
            values += [
                float(np.log10((relative["theta"] + 1e-8) / (relative["alpha"] + 1e-8))),
                float(
                    np.log10(
                        (relative["delta"] + relative["theta"] + 1e-8)
                        / (relative["alpha"] + relative["beta"] + 1e-8)
                    )
                ),
            ]
            for low, high in PEAK_BANDS.values():
                mask = (frequencies_kept >= low) & (frequencies_kept <= high)
                local_frequencies = frequencies_kept[mask]
                local_residual = residual[mask]
                peak = int(np.argmax(local_residual))
                values += [
                    float(local_frequencies[peak]),
                    float(max(local_residual[peak], 0.0)),
                    peak_width(local_frequencies, local_residual, peak),
                ]
            channel_rows.append(values)
        channel_matrix = np.asarray(channel_rows, dtype=np.float32)
        regional = np.concatenate(
            [channel_matrix[indexes].mean(axis=0) for indexes in region_indexes.values()]
        )
        row = np.concatenate(
            [channel_matrix.ravel(), regional, channel_matrix.mean(axis=0), channel_matrix.std(axis=0)]
        )
        rows.append(row)
        if (subject + 1) % 20 == 0:
            print(f"Spectral biomarkers: {subject + 1}/{participant_count} participants", flush=True)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid spectral feature matrix {matrix.shape}, names={len(names)}")
    return matrix, names


def normalized_maps(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = window[:, ::2].astype(np.float32)
    values -= values.mean(axis=0, keepdims=True)
    gfp = values.std(axis=0)
    maps = values.T
    maps /= np.maximum(np.linalg.norm(maps, axis=1, keepdims=True), 1e-8)
    return maps, gfp


def collect_training_peak_maps(raw: np.ndarray, owners: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    indexes = np.flatnonzero(np.isin(owners, subjects))
    maps = []
    for number, index in enumerate(indexes, 1):
        topographies, gfp = normalized_maps(raw[index])
        peaks, _ = find_peaks(gfp, distance=4)
        if not len(peaks):
            continue
        selected = peaks[np.argsort(gfp[peaks])[-PEAK_MAPS_PER_WINDOW:]]
        maps.append(topographies[selected])
        if number % 1000 == 0:
            print(f"Microstate peak maps: {number}/{len(indexes)} windows", flush=True)
    result = np.concatenate(maps).astype(np.float32)
    if len(result) > MAX_TEMPLATE_MAPS:
        rng = np.random.default_rng(SEED)
        result = result[rng.choice(len(result), MAX_TEMPLATE_MAPS, replace=False)]
    return result


def fit_microstate_templates(maps: np.ndarray, count: int = MICROSTATE_COUNT) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    best_templates = None
    best_score = -np.inf
    for initialization in range(5):
        templates = maps[rng.choice(len(maps), count, replace=False)].copy()
        assignment = np.full(len(maps), -1, dtype=np.int16)
        for _ in range(40):
            scores = maps @ templates.T
            updated = np.argmax(np.abs(scores), axis=1).astype(np.int16)
            if np.array_equal(updated, assignment):
                break
            assignment = updated
            for state in range(count):
                members = maps[assignment == state]
                if not len(members):
                    templates[state] = maps[rng.integers(len(maps))]
                    continue
                covariance = members.T @ members
                values, vectors = np.linalg.eigh(covariance)
                template = vectors[:, np.argmax(values)]
                if np.dot(template, templates[state]) < 0:
                    template = -template
                templates[state] = template / max(np.linalg.norm(template), 1e-8)
        fit = np.max(np.abs(maps @ templates.T), axis=1)
        score = float(np.mean(np.square(fit)))
        print(f"  Microstate initialization {initialization + 1}/5: peak-map GEV={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_templates = templates.copy()
    assert best_templates is not None
    return best_templates


def microstate_names(count: int) -> list[str]:
    names = []
    for statistic in ("coverage", "occurrence_per_second", "mean_duration_seconds", "gev"):
        names += [f"microstate_{state + 1}__{statistic}" for state in range(count)]
    names += [
        f"transition_{source + 1}_to_{target + 1}"
        for source in range(count)
        for target in range(count)
        if source != target
    ]
    return names


def extract_microstate_features(
    raw: np.ndarray, owners: np.ndarray, participant_count: int, templates: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    count = len(templates)
    rows = []
    for subject in range(participant_count):
        sample_counts = np.zeros(count, dtype=np.float64)
        segment_counts = np.zeros(count, dtype=np.float64)
        gev_numerator = np.zeros(count, dtype=np.float64)
        transitions = np.zeros((count, count), dtype=np.float64)
        total_samples = 0
        total_gfp_energy = 0.0
        for window in raw[owners == subject]:
            maps, gfp = normalized_maps(window)
            correlations = np.abs(maps @ templates.T)
            labels = np.argmax(correlations, axis=1)
            best = correlations[np.arange(len(labels)), labels]
            sample_counts += np.bincount(labels, minlength=count)
            total_samples += len(labels)
            total_gfp_energy += float(np.sum(np.square(gfp)))
            for state in range(count):
                local = labels == state
                gev_numerator[state] += float(np.sum(np.square(gfp[local]) * np.square(best[local])))
            starts = np.r_[0, 1 + np.flatnonzero(labels[1:] != labels[:-1])]
            segment_labels = labels[starts]
            segment_counts += np.bincount(segment_labels, minlength=count)
            if len(segment_labels) > 1:
                np.add.at(transitions, (segment_labels[:-1], segment_labels[1:]), 1)
        coverage = sample_counts / max(total_samples, 1)
        seconds = total_samples / MICROSTATE_FS
        occurrence = segment_counts / max(seconds, 1e-8)
        duration = sample_counts / np.maximum(segment_counts, 1) / MICROSTATE_FS
        gev = gev_numerator / max(total_gfp_energy, 1e-12)
        row_totals = transitions.sum(axis=1, keepdims=True)
        transition_probability = np.divide(
            transitions, row_totals, out=np.zeros_like(transitions), where=row_totals > 0
        )
        off_diagonal = transition_probability[~np.eye(count, dtype=bool)]
        rows.append(np.concatenate([coverage, occurrence, duration, gev, off_diagonal]))
        if (subject + 1) % 20 == 0:
            print(f"Microstate features: {subject + 1}/{participant_count} participants", flush=True)
    matrix = np.asarray(rows, dtype=np.float32)
    names = microstate_names(count)
    if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid microstate matrix {matrix.shape}")
    return matrix, names


def site_center(features: np.ndarray, fit: np.ndarray, sites: np.ndarray, method: str) -> np.ndarray:
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


def make_model(feature_count: int, k: int | str, c_value: float, kernel: str) -> Pipeline:
    selected = "all" if k == "all" else min(int(k), feature_count)
    return Pipeline(
        [
            ("select", SelectKBest(score_func=f_classif, k=selected)),
            ("scale", StandardScaler()),
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
    selected = row["selected_features"] if isinstance(row["selected_features"], int) else 10**9
    kernel_complexity = 0 if row["kernel"] == "linear" else 1
    return (-row["macro_f1"], -row["balanced_accuracy"], -row["accuracy"], selected, kernel_complexity, row["C"])


def plot_results(result: dict[str, Any]) -> None:
    baseline = json.loads((HERE / "outputs/results/site_shift/metrics.json").read_text(encoding="utf-8"))["baseline_test"]
    tucker = json.loads((HERE / "outputs/prepool_tucker_svm/metrics.json").read_text(encoding="utf-8"))["test"]
    new = result["test"]
    labels = ["Log-mel SVM", "Pre-pooling Tucker", "Biomarkers + microstates"]
    rows = [baseline, tucker, new]
    measures = (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy"))
    positions = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for offset, ((key, title), color) in enumerate(
        zip(measures, ("#4F81BD", "#C0504D", "#70AD47"), strict=True)
    ):
        values = [row[key] for row in rows]
        bars = ax.bar(positions + (offset - 1) * width, values, width, label=title, color=color)
        ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=3, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Held-out participant score")
    ax.set_title("Biologically targeted EEG features: fixed-split comparison")
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
    ax.set_title("Biomarker + microstate SVM")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    branch_labels = ["Spectral biomarkers", "Microstates", "Combined"]
    branch_rows = [result["branch_results"][name]["test"] for name in ("spectral", "microstates", "combined")]
    positions = np.arange(len(branch_labels))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for offset, ((key, title), color) in enumerate(
        zip(measures, ("#4F81BD", "#C0504D", "#70AD47"), strict=True)
    ):
        values = [row[key] for row in branch_rows]
        bars = ax.bar(positions + (offset - 1) * width, values, width, label=title, color=color)
        ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=3, fontsize=9)
    ax.set_xticks(positions, branch_labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Held-out participant score")
    ax.set_title("Feature-family ablation")
    ax.legend(loc="upper center", ncols=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "feature_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = np.load(PREPARED / "raw_windows.npy", mmap_mode="r")
    owners = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    channels = list(manifest["channels"])
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    train = np.asarray(splits["train"], dtype=int)
    validation = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, validation])
    labels = participant_labels(window_labels, owners, len(participants))
    sites = np.asarray([participant["site"] for participant in participants])

    print("Extracting periodic/aperiodic spectral biomarkers", flush=True)
    spectral, spectral_names = extract_spectral_features(raw, owners, len(participants), channels)

    print("Learning four microstate templates from training participants only", flush=True)
    train_maps = collect_training_peak_maps(raw, owners, train)
    train_templates = fit_microstate_templates(train_maps)
    microstates, microstate_feature_names = extract_microstate_features(
        raw, owners, len(participants), train_templates
    )

    matrices = {
        "spectral": (spectral, spectral_names),
        "microstates": (microstates, microstate_feature_names),
        "combined": (
            np.concatenate([spectral, microstates], axis=1),
            spectral_names + microstate_feature_names,
        ),
    }
    candidates = []
    for feature_set, (features, _) in matrices.items():
        unique_k: list[int | str] = []
        for value in (20, 40, 80, 160, "all"):
            effective = "all" if value == "all" else min(value, features.shape[1])
            if effective not in unique_k:
                unique_k.append(effective)
        for normalization in ("global", "site_center"):
            normalized = site_center(features, train, sites, normalization)
            for k in unique_k:
                for c_value in (0.01, 0.1, 1.0, 10.0):
                    for kernel in ("linear", "rbf"):
                        model = make_model(features.shape[1], k, c_value, kernel)
                        model.fit(normalized[train], labels[train])
                        prediction = model.predict(normalized[validation])
                        score = metrics(labels[validation], prediction)
                        candidates.append(
                            {
                                "feature_set": feature_set,
                                "normalization": normalization,
                                "selected_features": k,
                                "C": c_value,
                                "kernel": kernel,
                                "input_features": int(features.shape[1]),
                                "accuracy": score["accuracy"],
                                "balanced_accuracy": score["balanced_accuracy"],
                                "macro_f1": score["macro_f1"],
                            }
                        )
    candidates.sort(key=candidate_key)
    selected = candidates[0]
    best_by_feature_set = {
        feature_set: next(row for row in candidates if row["feature_set"] == feature_set)
        for feature_set in matrices
    }
    print("Selected on validation:", json.dumps(selected, indent=2), flush=True)

    print("Refitting microstate templates on all development participants", flush=True)
    development_maps = collect_training_peak_maps(raw, owners, development)
    final_templates = fit_microstate_templates(development_maps)
    final_microstates, final_microstate_names = extract_microstate_features(
        raw, owners, len(participants), final_templates
    )
    final_matrices = {
        "spectral": (spectral, spectral_names),
        "microstates": (final_microstates, final_microstate_names),
        "combined": (
            np.concatenate([spectral, final_microstates], axis=1),
            spectral_names + final_microstate_names,
        ),
    }
    branch_results = {}
    branch_models = {}
    for feature_set, configuration in best_by_feature_set.items():
        branch_features, branch_names = final_matrices[feature_set]
        branch_normalized = site_center(
            branch_features, development, sites, configuration["normalization"]
        )
        branch_model = make_model(
            branch_features.shape[1],
            configuration["selected_features"],
            configuration["C"],
            configuration["kernel"],
        )
        branch_model.fit(branch_normalized[development], labels[development])
        branch_prediction = branch_model.predict(branch_normalized[test])
        branch_results[feature_set] = {
            "validation_selected_configuration": configuration,
            "test": metrics(labels[test], branch_prediction),
            "test_predictions": branch_prediction.tolist(),
        }
        branch_models[feature_set] = (branch_model, branch_names)

    test_metrics = branch_results[selected["feature_set"]]["test"]
    prediction = np.asarray(branch_results[selected["feature_set"]]["test_predictions"])
    model, final_names = branch_models[selected["feature_set"]]
    selector = model.named_steps["select"]
    support = selector.get_support()
    selected_names = [name for name, keep in zip(final_names, support, strict=True) if keep]
    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    ranked = np.argsort(scores)[::-1]
    top_features = [
        {"name": final_names[index], "anova_f": float(scores[index])}
        for index in ranked[: min(30, len(ranked))]
        if np.isfinite(scores[index])
    ]

    np.savez_compressed(
        OUT / "features.npz",
        spectral=spectral,
        microstates=final_microstates,
        participants=np.asarray([p["subject_id"] for p in participants]),
        labels=labels,
        spectral_names=np.asarray(spectral_names),
        microstate_names=np.asarray(final_microstate_names),
        templates=final_templates,
    )
    result = {
        "method": "robust periodic/aperiodic spectral biomarkers plus four training-only polarity-invariant microstates, ANOVA feature selection, and class-balanced SVM",
        "participant_split": {name: len(splits[name]) for name in ("train", "val", "test")},
        "spectral_feature_count": int(spectral.shape[1]),
        "microstate_feature_count": int(final_microstates.shape[1]),
        "candidate_count": len(candidates),
        "selected": selected,
        "best_validation_configuration_by_feature_set": best_by_feature_set,
        "branch_results": branch_results,
        "selected_feature_names": selected_names,
        "top_anova_features": top_features,
        "test": test_metrics,
        "test_subjects": test.tolist(),
        "test_truth": labels[test].tolist(),
        "test_predictions": prediction.tolist(),
        "top_candidates": candidates[:20],
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "This uses a transparent robust log-log background fit inspired by periodic/aperiodic parameterization; the optional specparam package was not installed.",
            "The fixed test set was inspected in prior project iterations, so the score is exploratory rather than an unbiased publication estimate.",
            "The validation set contains only 26 participants and model selection can overfit it.",
            "Microstate templates are fold-safe for this fixed split, but repeated nested participant-level evaluation is required for a publication claim.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_results(result)
    print(
        json.dumps(
            {"selected": selected, "test": test_metrics, "runtime_seconds": result["runtime_seconds"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
