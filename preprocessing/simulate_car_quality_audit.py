"""Evaluate common-average re-referencing without writing transformed EEG."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


PRIMARY = Path(r"C:\Projects\AIBSI\preprocessing\02_balanced_primary\primary")
OUT = Path(r"C:\Projects\AIBSI\preprocessing\03_signal_qc_audit")
FS = 250.0


def q(values: np.ndarray) -> dict[str, float]:
    probs = [0, 0.5, 0.95, 0.99, 0.995, 0.999, 1]
    return {str(p): float(v) for p, v in zip(probs, np.quantile(values, probs))}


def main() -> None:
    x = np.load(PRIMARY / "signals.npy", mmap_mode="r")
    meta = np.load(PRIMARY / "metadata.npz")
    groups = np.asarray(meta["group"]).astype(str)
    subjects = np.asarray(meta["subject"]).astype(str)
    n, _, samples = x.shape

    freqs = np.fft.rfftfreq(samples, 1 / FS)
    taper = np.hanning(samples).astype(np.float32)
    total_mask = (freqs >= 0.5) & (freqs <= 45)
    high_mask = (freqs >= 30) & (freqs <= 45)

    common_fraction = np.empty(n)
    max_abs = np.empty(n)
    max_step = np.empty(n)
    min_std = np.empty(n)
    high_ratio = np.empty(n)

    for start in range(0, n, 128):
        stop = min(start + 128, n)
        original = np.asarray(x[start:stop], dtype=np.float32)
        common = original.mean(axis=1, keepdims=True)
        common_fraction[start:stop] = (
            np.var(common[:, 0, :], axis=1)
            / np.maximum(np.mean(np.var(original, axis=2), axis=1), 1e-12)
        )
        car = original - common
        car -= car.mean(axis=2, keepdims=True)
        max_abs[start:stop] = np.max(np.abs(car), axis=(1, 2))
        max_step[start:stop] = np.max(np.abs(np.diff(car, axis=2)), axis=(1, 2))
        min_std[start:stop] = np.std(car, axis=2).min(axis=1)
        spectrum = np.abs(np.fft.rfft(car * taper, axis=2)) ** 2
        total = spectrum[:, :, total_mask].sum(axis=(1, 2))
        high_ratio[start:stop] = spectrum[:, :, high_mask].sum(axis=(1, 2)) / np.maximum(total, 1e-20)

    hard = (max_abs >= 400) | (max_step >= 300) | (min_std < 1)
    abrupt_review = (~hard) & (max_step >= 150)
    all_step_candidates = hard | abrupt_review

    by_group = {}
    for group in sorted(set(groups)):
        mask = groups == group
        by_group[group] = {
            "windows": int(mask.sum()),
            "hard_candidates": int((hard & mask).sum()),
            "additional_abrupt_step_review": int((abrupt_review & mask).sum()),
            "all_step_or_hard_candidates": int((all_step_candidates & mask).sum()),
            "high_30_45_ratio_median": float(np.median(high_ratio[mask])),
            "high_30_45_ratio_p95": float(np.quantile(high_ratio[mask], 0.95)),
        }

    subject_rows = []
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        subject_rows.append({
            "subject": subject,
            "group": str(groups[mask][0]),
            "windows": int(mask.sum()),
            "hard_candidates": int((hard & mask).sum()),
            "all_step_or_hard_candidates": int((all_step_candidates & mask).sum()),
            "median_high_30_45_ratio": float(np.median(high_ratio[mask])),
            "p95_high_30_45_ratio": float(np.quantile(high_ratio[mask], 0.95)),
        })

    report = {
        "method": "in-memory common-average reference across the 19 scalp channels; stored primary signals unchanged",
        "common_mode_variance_fraction": q(common_fraction),
        "after_car": {
            "max_abs_uv": q(max_abs),
            "max_step_uv_per_4ms": q(max_step),
            "minimum_channel_std_uv": q(min_std),
            "high_30_45_power_ratio": q(high_ratio),
        },
        "screening_thresholds": {
            "hard_candidate": "max abs >= 400 uV OR 4-ms step >= 300 uV OR channel std < 1 uV",
            "additional_review": "4-ms step >= 150 uV",
            "high_frequency": "reported only, never automatically rejected because it is strongly subject/group dependent",
        },
        "candidate_counts": {
            "hard": int(hard.sum()),
            "additional_abrupt_step_review": int(abrupt_review.sum()),
            "all_step_or_hard": int(all_step_candidates.sum()),
            "by_group": by_group,
            "top_subjects": sorted(
                subject_rows,
                key=lambda row: (row["all_step_or_hard_candidates"], row["hard_candidates"]),
                reverse=True,
            )[:15],
        },
        "persistent_high_frequency_subjects_median_gt_0p2": [
            row for row in subject_rows if row["median_high_30_45_ratio"] > 0.2
        ],
        "interpretation": [
            "The shared A1-A2 reference dominates most window variance; common-average rereferencing should precede connectivity or graph construction.",
            "Abrupt-step candidates remain review/rejection candidates after rereferencing.",
            "30-45 Hz power is too diagnosis/subject dependent to use as a label-blind deletion rule; analyze 1-30 Hz primarily and retain 30-45 Hz as a sensitivity analysis.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "car_diagnostic_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["candidate_counts"], indent=2))


if __name__ == "__main__":
    main()
