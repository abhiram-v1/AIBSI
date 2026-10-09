from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy.signal import stft, welch
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
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
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
from run_pipeline import make_subject_splits  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PREPARED = OUT / "prepared"
RESULTS = OUT / "results"

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
CLASSES = ["AD", "FTD", "HC"]
LABEL_ID = {name: index for index, name in enumerate(CLASSES)}
LEGACY = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

FS = 128
WINDOW_SECONDS = 8
WINDOW_SAMPLES = FS * WINDOW_SECONDS
STFT_SAMPLES = FS * 2
STFT_HOP = FS // 2
EDGE_SECONDS = 10
MAX_WINDOWS = 40
MEL_BINS = 24
SEED = 20261009


@dataclass
class Record:
    subject_id: str
    source: str
    site: str
    label: str
    path: str


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def discover() -> list[Record]:
    records: list[Record] = []
    table = pd.read_csv(ROOT / "dataset" / "ds004504" / "participants.tsv", sep="\t")
    labels = {"A": "AD", "F": "FTD", "C": "HC"}
    for row in table.itertuples(index=False):
        subject = str(row.participant_id)
        path = ROOT / "dataset" / "ds004504" / "derivatives" / subject / "eeg" / f"{subject}_task-eyesclosed_eeg.set"
        if path.exists():
            records.append(Record(f"ds004504:{subject}", "ds004504", "AHEPA", labels[row.Group], str(path)))
    brainlat = ROOT / "dataset" / "brainlat" / "EEG"
    for folder, label in {"1_AD": "AD", "2_bvFTD": "FTD", "5_HC": "HC"}.items():
        for path in sorted((brainlat / folder).rglob("*.set")):
            subject = path.parent.parent.name
            site = path.relative_to(brainlat / folder).parts[0]
            records.append(Record(f"brainlat:{site}:{subject}", "brainlat", site, label, str(path)))
    return records


def target_montage() -> mne.channels.DigMontage:
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()
    return mne.channels.make_dig_montage(
        ch_pos={name: positions["ch_pos"][name] for name in CHANNELS},
        coord_frame=positions["coord_frame"],
    )


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 0.6745 * (values - median) / max(mad, 1e-12)


def detect_bad_channels(data: np.ndarray, sfreq: float) -> tuple[list[str], dict[str, dict[str, float]]]:
    sd = data.std(axis=1) * 1e6
    peak = np.ptp(data, axis=1) * 1e6
    freqs, psd = welch(data, fs=sfreq, nperseg=min(1024, data.shape[1]), axis=-1)
    low = psd[:, (freqs >= 1) & (freqs <= 25)].mean(axis=1)
    high = psd[:, (freqs >= 30) & (freqs <= 40)].mean(axis=1)
    hf_ratio = high / np.maximum(low, 1e-20)
    # Use absolute agreement with the other electrodes.  Correlation against a
    # global median can incorrectly label valid frontal channels whose polarity
    # differs because of the acquisition reference.
    matrix = np.corrcoef(data)
    correlations = np.asarray([
        np.median(np.abs(np.delete(matrix[index], index))) if np.std(data[index]) > 0 else 0.0
        for index in range(len(data))
    ])
    scores = (
        np.maximum(np.abs(robust_z(np.log10(np.maximum(sd, 1e-4)))) - 3.0, 0)
        + np.maximum(robust_z(np.log10(np.maximum(peak, 1e-4))) - 3.0, 0)
        + np.maximum(robust_z(np.log10(np.maximum(hf_ratio, 1e-8))) - 3.0, 0)
        + np.maximum(0.15 - correlations, 0) * 10
    )
    absolute_bad = (sd < 0.1) | (sd > 150) | (peak > 600) | ~np.isfinite(correlations)
    candidate = absolute_bad | (scores > 1.0)
    ordered = np.argsort(scores + absolute_bad.astype(float) * 100)[::-1]
    selected = [int(index) for index in ordered if candidate[index]][:4]
    details = {
        CHANNELS[index]: {
            "sd_uv": float(sd[index]),
            "peak_to_peak_uv": float(peak[index]),
            "high_frequency_ratio": float(hf_ratio[index]),
            "median_correlation": float(correlations[index]),
            "noise_score": float(scores[index]),
        }
        for index in range(len(CHANNELS))
    }
    return [CHANNELS[index] for index in selected], details


def conservative_ica(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, list[int], list[dict[str, float]]]:
    fit_raw = raw.copy().filter(1.0, 40.0, verbose="ERROR")
    ica = ICA(n_components=15, method="fastica", max_iter=350, random_state=SEED)
    ica.fit(fit_raw, picks="eeg", decim=4, verbose="ERROR")
    sources = ica.get_sources(fit_raw).get_data()
    data = fit_raw.get_data()
    index = {name: fit_raw.ch_names.index(name) for name in ["Fp1", "Fp2", "F3", "F4"]}
    proxy = (data[index["Fp1"]] + data[index["Fp2"]]) / 2 - (data[index["F3"]] + data[index["F4"]]) / 2
    mixing = np.abs(ica.get_components())
    frontal = mixing[[fit_raw.ch_names.index("Fp1"), fit_raw.ch_names.index("Fp2")]].mean(axis=0)
    spatial_ratio = frontal / np.maximum(np.median(mixing, axis=0), 1e-9)
    source_kurtosis = kurtosis(sources, axis=1, fisher=False, nan_policy="omit")
    diagnostics = []
    candidates = []
    for component in range(sources.shape[0]):
        correlation = float(np.corrcoef(sources[component], proxy)[0, 1])
        diagnostics.append(
            {
                "component": component,
                "ocular_correlation": correlation,
                "frontal_spatial_ratio": float(spatial_ratio[component]),
                "kurtosis": float(source_kurtosis[component]),
            }
        )
        if abs(correlation) >= 0.50 and spatial_ratio[component] >= 1.6 and source_kurtosis[component] >= 4.0:
            candidates.append((abs(correlation) * spatial_ratio[component], component))
    excluded = [component for _, component in sorted(candidates, reverse=True)[:2]]
    if excluded:
        ica.exclude = excluded
        raw = ica.apply(raw.copy(), verbose="ERROR")
    return raw, excluded, diagnostics


def load_and_clean(record: Record) -> tuple[np.ndarray, dict[str, Any]]:
    raw = mne.io.read_raw_eeglab(record.path, preload=True, verbose="ERROR")
    metadata: dict[str, Any] = {
        "original_sfreq": float(raw.info["sfreq"]),
        "original_channels": int(raw.info["nchan"]),
        "original_seconds": float(raw.n_times / raw.info["sfreq"]),
    }
    if record.source == "brainlat":
        montage = mne.channels.make_standard_montage("biosemi128")
        if raw.info["nchan"] != 128 or set(raw.ch_names) != set(montage.ch_names):
            raise ValueError(f"Unexpected BrainLat montage: {raw.info['nchan']} channels")
        raw.set_montage(montage, on_missing="raise")
        raw = raw.interpolate_to(target_montage(), method="spline")
        # ``interpolate_to`` retains the source montage coordinate frame in some
        # MNE versions.  Reattach standard head-coordinate positions so later
        # recording-specific bad-channel interpolation has valid geometry.
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")
    else:
        rename = {old: new for old, new in LEGACY.items() if old in raw.ch_names}
        if rename:
            raw.rename_channels(rename)
        missing = sorted(set(CHANNELS) - set(raw.ch_names))
        if missing:
            raise ValueError(f"Missing channels: {missing}")
        raw.pick(CHANNELS)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")
    raw.reorder_channels(CHANNELS)
    raw.filter(1.0, 40.0, picks="eeg", verbose="ERROR")
    raw.resample(FS, npad="auto", verbose="ERROR")

    bads, channel_details = detect_bad_channels(raw.get_data(), FS)
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, method=dict(eeg="spline"), verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw, excluded, ica_diagnostics = conservative_ica(raw)
    metadata.update(
        {
            "bad_channels_interpolated": bads,
            "channel_diagnostics": channel_details,
            "ica_components_removed": excluded,
            "ica_diagnostics": ica_diagnostics,
        }
    )
    return raw.get_data().astype(np.float32, copy=False), metadata


def window_channel_reasons(window: np.ndarray) -> dict[str, list[str]]:
    values_uv = window * 1e6
    sd = values_uv.std(axis=1)
    peak = np.ptp(values_uv, axis=1)
    maximum = np.max(np.abs(values_uv), axis=1)
    derivative = np.diff(values_uv, axis=1).std(axis=1)
    shape = kurtosis(values_uv, axis=1, fisher=False, nan_policy="omit")
    reasons: dict[str, list[str]] = defaultdict(list)
    for index, channel in enumerate(CHANNELS):
        if not np.isfinite(values_uv[index]).all():
            reasons[channel].append("nonfinite")
        if sd[index] < 0.15:
            reasons[channel].append("flat")
        if sd[index] > 100 or peak[index] > 350 or maximum[index] > 250:
            reasons[channel].append("amplitude")
        if derivative[index] > 35:
            reasons[channel].append("high_frequency_artifact")
        if shape[index] > 18:
            reasons[channel].append("impulsive_artifact")
    return dict(reasons)


def mel_scale(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def inverse_mel(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10 ** (np.asarray(value) / 2595.0) - 1.0)


def mel_filter_bank(frequencies: np.ndarray) -> np.ndarray:
    points = inverse_mel(np.linspace(mel_scale(1.0), mel_scale(40.0), MEL_BINS + 2))
    bank = np.zeros((MEL_BINS, len(frequencies)), dtype=np.float32)
    for band in range(MEL_BINS):
        left, center, right = points[band : band + 3]
        bank[band] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-6),
                (right - frequencies) / max(right - center, 1e-6),
            ),
        )
        bank[band] /= max(bank[band].sum(), 1e-8)
    return bank


def spectral_tensors(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, _, complex_spectrum = stft(
        window,
        fs=FS,
        window="hann",
        nperseg=STFT_SAMPLES,
        noverlap=STFT_SAMPLES - STFT_HOP,
        nfft=STFT_SAMPLES,
        detrend="constant",
        boundary=None,
        padded=False,
        axis=-1,
    )
    keep = (frequencies >= 1.0) & (frequencies <= 40.0)
    frequencies = frequencies[keep]
    power = np.abs(complex_spectrum[:, keep, :]) ** 2
    log_stft = np.log10(power + 1e-20).astype(np.float32)
    log_stft -= np.median(log_stft)
    mel_power = np.einsum("mf,cft->cmt", mel_filter_bank(frequencies), power, optimize=True)
    log_mel = np.log10(mel_power + 1e-20).astype(np.float32)
    log_mel -= np.median(log_mel)
    return log_stft, log_mel


def evenly_limit(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    indexes = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[index] for index in indexes]


def prepare() -> None:
    mne.set_log_level("ERROR")
    seed_all()
    PREPARED.mkdir(parents=True, exist_ok=True)
    records = discover()
    stft_parts = []
    mel_parts = []
    label_parts = []
    subject_parts = []
    participants = []
    audits = []
    for record in tqdm(records, desc="Advanced EEG cleaning"):
        audit: dict[str, Any] = {**asdict(record), "status": "error"}
        try:
            data, details = load_and_clean(record)
            audit.update(details)
            start = EDGE_SECONDS * FS
            stop = data.shape[1] - EDGE_SECONDS * FS
            candidates = []
            channel_rejections = Counter()
            rejected_windows = Counter()
            for offset in range(start, stop - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                window = data[:, offset : offset + WINDOW_SAMPLES]
                reasons = window_channel_reasons(window)
                for channel, channel_reasons in reasons.items():
                    for reason in channel_reasons:
                        channel_rejections[f"{channel}:{reason}"] += 1
                if reasons:
                    rejected_windows["channel_artifact"] += 1
                    continue
                candidates.append(spectral_tensors(window))
            selected = evenly_limit(candidates, MAX_WINDOWS)
            if len(selected) < 5:
                raise ValueError(f"Only {len(selected)} clean windows")
            subject = len(participants)
            stft_parts.append(np.stack([item[0] for item in selected]).astype(np.float16))
            mel_parts.append(np.stack([item[1] for item in selected]).astype(np.float16))
            label_parts.append(np.full(len(selected), LABEL_ID[record.label], dtype=np.int64))
            subject_parts.append(np.full(len(selected), subject, dtype=np.int32))
            participants.append({"subject_index": subject, **asdict(record), "windows": len(selected)})
            audit.update(
                {
                    "status": "included",
                    "candidate_windows": len(candidates),
                    "retained_windows": len(selected),
                    "rejected_windows": dict(rejected_windows),
                    "channel_window_rejections": dict(channel_rejections),
                }
            )
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
        audits.append(audit)
    if not stft_parts:
        raise RuntimeError("No clean EEG data prepared")
    log_stft = np.concatenate(stft_parts)
    log_mel = np.concatenate(mel_parts)
    labels = np.concatenate(label_parts)
    subject_index = np.concatenate(subject_parts)
    np.save(PREPARED / "log_stft.npy", log_stft)
    np.save(PREPARED / "log_mel.npy", log_mel)
    np.save(PREPARED / "labels.npy", labels)
    np.save(PREPARED / "subject_index.npy", subject_index)
    manifest = {
        "created_unix": time.time(),
        "channels": CHANNELS,
        "classes": CLASSES,
        "sampling_rate": FS,
        "window_seconds": WINDOW_SECONDS,
        "log_stft_shape": list(log_stft.shape),
        "log_mel_shape": list(log_mel.shape),
        "mel_bins": MEL_BINS,
        "participants": participants,
    }
    (PREPARED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PREPARED / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"Prepared {len(labels)} clean windows from {len(participants)} participants")
    print("STFT:", log_stft.shape, "log-mel:", log_mel.shape)
    print("Included:", Counter((p["source"], p["label"]) for p in participants))
    print("Excluded:", sum(a["status"] != "included" for a in audits))


def subject_features(
    tensor: np.ndarray, labels: np.ndarray, subject_index: np.ndarray, participants: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    subject_labels = []
    for subject in range(len(participants)):
        windows = tensor[subject_index == subject].astype(np.float32)
        repeats = windows.transpose(0, 3, 1, 2).reshape(-1, windows.shape[1], windows.shape[2])
        mean = repeats.mean(axis=0)
        std = repeats.std(axis=0)
        features.append(np.concatenate([mean, std], axis=1))
        subject_labels.append(int(labels[np.flatnonzero(subject_index == subject)[0]]))
    return np.asarray(features, dtype=np.float32), np.asarray(subject_labels, dtype=np.int64)


def hierarchy_model(components: int, boost: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=components, random_state=SEED)),
            ("svm", SVC(C=0.1, kernel="linear", class_weight={0: 1.0, 1: boost}, random_state=SEED)),
        ]
    )


def hierarchical_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
) -> np.ndarray:
    components1 = min(20, len(train_y) - 1, train_x.shape[1])
    stage1 = hierarchy_model(components1, 1.0)
    stage1.fit(train_x, (train_y != 2).astype(int))
    disease = train_y != 2
    components2 = min(20, int(disease.sum()) - 1, train_x.shape[1])
    stage2 = hierarchy_model(components2, 3.0)
    stage2.fit(train_x[disease], train_y[disease])
    is_disease = stage1.predict(predict_x)
    subtype = stage2.predict(predict_x)
    return np.where(is_disease == 0, 2, subtype).astype(int)


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=np.arange(3), zero_division=0
    )
    return {
        "subjects": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(true, pred, labels=np.arange(3)).tolist(),
    }


def normalize_score(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return (values - low) / max(high - low, 1e-9)


def electrode_scores(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    frequency_features: int,
    audits: list[dict],
) -> list[dict[str, Any]]:
    included = [audit for audit in audits if audit["status"] == "included"]
    bad_counts = Counter(channel for audit in included for channel in audit.get("bad_channels_interpolated", []))
    rejection_counts = Counter()
    total_candidate_windows = 0
    for audit in included:
        total_candidate_windows += audit.get("candidate_windows", 0) + sum(audit.get("rejected_windows", {}).values())
        for key, count in audit.get("channel_window_rejections", {}).items():
            rejection_counts[key.split(":", 1)[0]] += count
    noise = np.asarray([
        bad_counts[channel] / max(len(included), 1)
        + rejection_counts[channel] / max(total_candidate_windows, 1)
        for channel in CHANNELS
    ])
    mi_values = []
    single_f1 = []
    for electrode in range(len(CHANNELS)):
        values = features[:, electrode, :]
        mi = mutual_info_classif(values[train], labels[train], random_state=SEED)
        mi_values.append(float(np.mean(np.sort(mi)[-max(4, len(mi) // 5) :])))
        prediction = hierarchical_fit_predict(values[train], labels[train], values[val])
        single_f1.append(float(f1_score(labels[val], prediction, average="macro", zero_division=0)))
    combined = 0.45 * normalize_score(np.asarray(mi_values)) + 0.55 * normalize_score(np.asarray(single_f1)) - 0.35 * normalize_score(noise)
    rows = []
    for index, channel in enumerate(CHANNELS):
        rows.append(
            {
                "channel": channel,
                "noise_rate": float(noise[index]),
                "mutual_information": mi_values[index],
                "single_electrode_val_macro_f1": single_f1[index],
                "combined_score": float(combined[index]),
                "frequency_features_per_statistic": frequency_features,
            }
        )
    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    return rows


def subset_flat(features: np.ndarray, ranking: list[str], count: int) -> np.ndarray:
    selected = [CHANNELS.index(channel) for channel in ranking[:count]]
    return features[:, selected, :].reshape(len(features), -1)


def analyze() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    labels_windows = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    audits = json.loads((PREPARED / "audit.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    representations = {
        "log_stft": np.load(PREPARED / "log_stft.npy", mmap_mode="r"),
        "log_mel": np.load(PREPARED / "log_mel.npy", mmap_mode="r"),
    }
    payload: dict[str, Any] = {
        "participants": len(participants),
        "splits": {name: len(values) for name, values in splits.items()},
        "representations": {},
    }
    ranking_payload = {}
    counts = [6, 8, 10, 12, 15, 19]
    for name, tensor in representations.items():
        features, labels = subject_features(tensor, labels_windows, subject_index, participants)
        ranking_rows = electrode_scores(features, labels, train, val, tensor.shape[2], audits)
        ranking = [row["channel"] for row in ranking_rows]
        candidates = []
        for count in counts:
            flat = subset_flat(features, ranking, count)
            prediction = hierarchical_fit_predict(flat[train], labels[train], flat[val])
            candidates.append(
                {
                    "electrodes": count,
                    "selected": ranking[:count],
                    "val_macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
                    "val_accuracy": float(accuracy_score(labels[val], prediction)),
                }
            )
        candidates.sort(key=lambda item: (item["val_macro_f1"], item["val_accuracy"], -item["electrodes"]), reverse=True)
        best = candidates[0]
        flat = subset_flat(features, ranking, best["electrodes"])
        train_val = np.concatenate([train, val])
        test_prediction = hierarchical_fit_predict(flat[train_val], labels[train_val], flat[test])
        representation_metrics = metrics(labels[test], test_prediction)
        payload["representations"][name] = {
            "selected_electrodes": best["selected"],
            "selected_count": best["electrodes"],
            "selection_val_macro_f1": best["val_macro_f1"],
            "selection_candidates": sorted(candidates, key=lambda item: item["electrodes"]),
            "test": representation_metrics,
        }
        ranking_payload[name] = ranking_rows
        print(name, best, json.dumps(representation_metrics, indent=2))
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS / "electrode_ranking.json").write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")
    plot_analysis(payload, ranking_payload, audits)


def plot_analysis(payload: dict, rankings: dict, audits: list[dict]) -> None:
    names = ["log_stft", "log_mel"]
    display = ["Log-STFT", "Log-mel"]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    positions = np.arange(2)
    for offset, measure in enumerate(measures):
        values = [payload["representations"][name]["test"][measure] for name in names]
        bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=measure.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(positions, display)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Cleaned EEG spectral representation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(RESULTS / "representation_comparison.png", dpi=180)
    plt.close(fig)

    previous_path = ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "hierarchical_svm" / "metrics.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))["test"]
        comparison_names = ["Earlier STFT\n(no advanced cleaning)", "Cleaned\nlog-STFT", "Cleaned\nlog-mel"]
        comparison_rows = [previous] + [payload["representations"][name]["test"] for name in names]
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        positions = np.arange(len(comparison_names))
        for offset, measure in enumerate(measures):
            values = [row[measure] for row in comparison_rows]
            bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=measure.replace("_", " ").title())
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
        ax.set_xticks(positions, comparison_names)
        ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Effect of advanced cleaning and spectral representation")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.savefig(RESULTS / "previous_best_comparison.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for name, label in zip(names, display):
        candidates = payload["representations"][name]["selection_candidates"]
        ax.plot([item["electrodes"] for item in candidates], [item["val_macro_f1"] for item in candidates], marker="o", label=label)
    ax.set(xlabel="Number of top-ranked electrodes", ylabel="Validation macro F1", title="Electrode subset selection using validation participants", ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(RESULTS / "electrode_subset_selection.png", dpi=180)
    plt.close(fig)

    ranking = rankings["log_stft"]
    ordered = [row["channel"] for row in ranking][::-1]
    score = [row["combined_score"] for row in ranking][::-1]
    noise = [row["noise_rate"] for row in ranking][::-1]
    single = [row["single_electrode_val_macro_f1"] for row in ranking][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True, constrained_layout=True)
    axes[0].barh(ordered, noise, color="#d55e00")
    axes[0].set(title="Noise burden", xlabel="Rate")
    axes[1].barh(ordered, single, color="#2878b5")
    axes[1].set(title="Single-electrode validation", xlabel="Macro F1", xlim=(0, 1))
    axes[2].barh(ordered, score, color="#59a14f")
    axes[2].set(title="Combined selection score", xlabel="Score")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(RESULTS / "electrode_analysis.png", dpi=180)
    plt.close(fig)

    included = [audit for audit in audits if audit["status"] == "included"]
    bad_recordings = sum(bool(audit.get("bad_channels_interpolated")) for audit in included)
    ica_recordings = sum(bool(audit.get("ica_components_removed")) for audit in included)
    rejected = sum(sum(audit.get("rejected_windows", {}).values()) for audit in included)
    retained = sum(audit.get("retained_windows", 0) for audit in included)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    bars = axes[0].bar(
        ["Bad-channel\ninterpolation", "Ocular ICA\nremoval"],
        [bad_recordings, ica_recordings],
        color=["#e15759", "#f28e2b"],
    )
    axes[0].bar_label(bars, padding=3)
    axes[0].set(ylabel="Recordings", title="Recording-level cleaning")
    bars = axes[1].bar(
        ["Artifact windows\nrejected", "Clean windows\nretained"],
        [rejected, retained],
        color=["#76b7b2", "#59a14f"],
    )
    axes[1].bar_label(bars, padding=3)
    axes[1].set(ylabel="Windows", title="Window-level cleaning")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Automated EEG cleaning summary")
    fig.savefig(RESULTS / "cleaning_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, name, label in zip(axes, names, display):
        matrix = np.asarray(payload["representations"][name]["test"]["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=label)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    channel = "C3"
    channel_index = CHANNELS.index(channel)
    stft_example = np.load(PREPARED / "log_stft.npy", mmap_mode="r")[0, channel_index].astype(np.float32)
    mel_example = np.load(PREPARED / "log_mel.npy", mmap_mode="r")[0, channel_index].astype(np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    image = axes[0].imshow(
        stft_example,
        origin="lower",
        aspect="auto",
        extent=[0, WINDOW_SECONDS, 1, 40],
        cmap="magma",
    )
    axes[0].set(title=f"Log-STFT example ({channel})", xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(image, ax=axes[0], label="Relative log power")
    image = axes[1].imshow(mel_example, origin="lower", aspect="auto", extent=[0, WINDOW_SECONDS, 0, MEL_BINS - 1], cmap="magma")
    tick_indexes = np.asarray([0, 5, 11, 17, 23])
    mel_centers = inverse_mel(np.linspace(mel_scale(1.0), mel_scale(40.0), MEL_BINS + 2))[1:-1]
    axes[1].set_yticks(tick_indexes, [f"{mel_centers[index]:.0f}" for index in tick_indexes])
    axes[1].set(title=f"Log-mel example ({channel})", xlabel="Time (s)", ylabel="Mel-band center (Hz)")
    fig.colorbar(image, ax=axes[1], label="Relative log power")
    fig.savefig(RESULTS / "spectral_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced EEG cleaning, STFT/log-mel preparation, and electrode analysis")
    parser.add_argument("command", choices=["prepare", "analyze", "all"])
    args = parser.parse_args()
    if args.command in {"prepare", "all"}:
        prepare()
    if args.command in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
