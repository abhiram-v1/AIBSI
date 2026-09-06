"""Read-only signal-quality audit for the balanced primary EEG windows.

The script never edits signals.npy. It writes window-, subject-, and channel-level
metrics that can be reviewed before a separate cleaning stage is created.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\Projects\AIBSI\preprocessing\02_balanced_primary\primary")
OUT = Path(r"C:\Projects\AIBSI\preprocessing\03_signal_qc_audit")
FS = 250.0
CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
FEATURE_NAMES = [
    "max_abs_uv", "max_ptp_uv", "min_channel_std_uv", "max_channel_std_uv",
    "median_channel_rms_uv", "max_step_uv", "flat_diff_fraction",
    "max_abs_channel_corr", "median_abs_channel_corr", "low_0p5_1_ratio",
    "delta_1_4_ratio", "theta_4_8_ratio", "alpha_8_13_ratio",
    "beta_13_30_ratio", "high_30_45_ratio", "line_48_52_ratio",
]


def band_sum(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    return power[..., mask].sum(axis=-1)


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(values)
    return 0.67448975 * (values - med) / mad


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999, 1]
    x = np.quantile(values[np.isfinite(values)], q)
    return {str(p): round(float(v), 8) for p, v in zip(q, x)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(ROOT / "signals.npy", mmap_mode="r")
    meta = np.load(ROOT / "metadata.npz")
    n, channels, samples = x.shape
    if channels != len(CHANNELS) or samples != 1000:
        raise ValueError(f"Unexpected signal shape {x.shape}")

    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    channel_sum = defaultdict(lambda: np.zeros(channels, dtype=np.float64))
    channel_count = Counter()
    hann = np.hanning(samples).astype(np.float32)
    freqs = np.fft.rfftfreq(samples, d=1.0 / FS)
    tri = np.triu_indices(channels, k=1)

    for start in range(0, n, 128):
        stop = min(start + 128, n)
        block = np.asarray(x[start:stop], dtype=np.float32)
        centered = block - block.mean(axis=-1, keepdims=True)
        std = centered.std(axis=-1)
        rms = np.sqrt(np.mean(centered * centered, axis=-1))
        steps = np.diff(block, axis=-1)
        max_abs = np.max(np.abs(block), axis=(1, 2))
        ptp = np.ptp(block, axis=-1)

        # Per-window channel correlations, protected against a zero variance.
        z = centered / np.maximum(std[..., None], 1e-12)
        corr = np.einsum("bct,bdt->bcd", z, z, optimize=True) / samples
        abs_corr = np.abs(corr[:, tri[0], tri[1]])

        # Windowed periodogram. Ratios avoid scale dominating the artifact metrics.
        fft = np.fft.rfft(centered * hann, axis=-1)
        power = (fft.real * fft.real + fft.imag * fft.imag).astype(np.float64)
        total = band_sum(power, freqs, 0.5, 45.0001)
        total_mean = np.maximum(total.mean(axis=1), 1e-18)
        ratios = []
        for lo, hi in [(0.5, 1), (1, 4), (4, 8), (8, 13), (13, 30), (30, 45.0001), (48, 52)]:
            ratios.append(band_sum(power, freqs, lo, hi).mean(axis=1) / total_mean)

        rows = np.column_stack([
            max_abs,
            ptp.max(axis=1),
            std.min(axis=1),
            std.max(axis=1),
            np.median(rms, axis=1),
            np.max(np.abs(steps), axis=(1, 2)),
            np.mean(np.abs(steps) < 1e-6, axis=(1, 2)),
            abs_corr.max(axis=1),
            np.median(abs_corr, axis=1),
            *ratios,
        ])
        features[start:stop] = rows

        for local_i, global_i in enumerate(range(start, stop)):
            group = str(meta["group"][global_i])
            channel_sum[(group, "std")] += std[local_i]
            channel_sum[(group, "rms")] += rms[local_i]
            channel_count[group] += 1

    # Data-adaptive indicators are deliberately conservative. They identify review
    # candidates, not automatic rejection decisions.
    f = {name: features[:, i] for i, name in enumerate(FEATURE_NAMES)}
    global_z = {name: robust_z(vals) for name, vals in f.items()}
    subject_ids = np.asarray(meta["subject"]).astype(str)
    groups = np.asarray(meta["group"]).astype(str)
    unique_subjects = sorted(set(subject_ids))

    within_subject_z = {name: np.zeros(n, dtype=np.float64) for name in FEATURE_NAMES}
    for subject in unique_subjects:
        idx = np.flatnonzero(subject_ids == subject)
        for name in FEATURE_NAMES:
            within_subject_z[name][idx] = robust_z(f[name][idx])

    flags: list[list[str]] = [[] for _ in range(n)]
    for i in range(n):
        if f["max_abs_uv"][i] >= 400:
            flags[i].append("amplitude_ge_400uv")
        elif f["max_abs_uv"][i] >= 250:
            flags[i].append("amplitude_ge_250uv_review")
        if f["min_channel_std_uv"][i] < 1:
            flags[i].append("near_flat_channel")
        if f["flat_diff_fraction"][i] > 0.05:
            flags[i].append("repeated_values")
        if f["max_abs_channel_corr"][i] > 0.995:
            flags[i].append("possible_channel_bridge")
        for name in ["max_step_uv", "high_30_45_ratio", "low_0p5_1_ratio", "line_48_52_ratio"]:
            if global_z[name][i] > 8 and within_subject_z[name][i] > 8:
                flags[i].append(f"isolated_{name}")

    window_path = OUT / "window_metrics.csv"
    with window_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["array_index", "sample_id", "subject", "group", *FEATURE_NAMES, "review_flags"])
        for i in range(n):
            writer.writerow([
                i, str(meta["sample_id"][i]), subject_ids[i], groups[i],
                *[f"{v:.9g}" for v in features[i]], ";".join(flags[i]),
            ])

    subject_rows = []
    for subject in unique_subjects:
        idx = np.flatnonzero(subject_ids == subject)
        row = {
            "subject": subject,
            "group": groups[idx[0]],
            "windows": int(len(idx)),
            "flagged_windows": int(sum(bool(flags[i]) for i in idx)),
        }
        for name in FEATURE_NAMES:
            row[f"median_{name}"] = float(np.median(f[name][idx]))
            row[f"p95_{name}"] = float(np.quantile(f[name][idx], 0.95))
        subject_rows.append(row)

    subject_path = OUT / "subject_metrics.csv"
    with subject_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)

    channel_rows = []
    for group in sorted(set(groups)):
        for ci, channel in enumerate(CHANNELS):
            channel_rows.append({
                "group": group,
                "channel": channel,
                "mean_window_std_uv": float(channel_sum[(group, "std")][ci] / channel_count[group]),
                "mean_window_rms_uv": float(channel_sum[(group, "rms")][ci] / channel_count[group]),
            })
    channel_path = OUT / "channel_metrics.csv"
    with channel_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(channel_rows[0]))
        writer.writeheader()
        writer.writerows(channel_rows)

    flag_counts = Counter(flag for row in flags for flag in row)
    flagged_by_group = {
        group: int(sum(bool(flags[i]) for i in np.flatnonzero(groups == group)))
        for group in sorted(set(groups))
    }
    group_quantiles = {
        group: {name: quantiles(f[name][groups == group]) for name in FEATURE_NAMES}
        for group in sorted(set(groups))
    }
    subject_medians = {
        name: np.array([row[f"median_{name}"] for row in subject_rows], dtype=float)
        for name in FEATURE_NAMES
    }
    subject_outliers = []
    for name, vals in subject_medians.items():
        zvals = robust_z(vals)
        for row, zval in zip(subject_rows, zvals):
            if abs(zval) > 4.5:
                subject_outliers.append({
                    "subject": row["subject"], "group": row["group"],
                    "metric": name, "robust_z": round(float(zval), 4),
                    "value": round(float(row[f"median_{name}"]), 8),
                })

    report = {
        "input": str(ROOT / "signals.npy"),
        "input_unchanged": True,
        "shape": list(x.shape),
        "sampling_hz": FS,
        "window_seconds": samples / FS,
        "source_preprocessing": "0.5-45 Hz Butterworth, A1-A2 rereference, ASR, RunICA, ICLabel eye/jaw rejection",
        "metric_quantiles_all": {name: quantiles(f[name]) for name in FEATURE_NAMES},
        "metric_quantiles_by_group": group_quantiles,
        "review_flag_counts": dict(flag_counts),
        "windows_with_any_review_flag": int(sum(bool(row) for row in flags)),
        "windows_with_any_review_flag_by_group": flagged_by_group,
        "subject_level_outliers_robust_z_gt_4p5": subject_outliers,
        "notes": [
            "Review flags are screening indicators and are not automatic rejection labels.",
            "Spectral ratios are computed from 4-second Hann-windowed periodograms.",
            "50 Hz power is expected to be tiny because source derivatives were low-pass filtered at 45 Hz.",
            "Group spectral differences can be disease signal; they must not be normalized away globally.",
        ],
        "outputs": {
            "window_metrics": str(window_path),
            "subject_metrics": str(subject_path),
            "channel_metrics": str(channel_path),
        },
    }
    with (OUT / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps({
        "windows": n,
        "windows_with_any_review_flag": report["windows_with_any_review_flag"],
        "review_flag_counts": report["review_flag_counts"],
        "flagged_by_group": flagged_by_group,
        "subject_level_outlier_count": len(subject_outliers),
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
