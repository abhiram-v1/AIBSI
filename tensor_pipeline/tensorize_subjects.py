"""Build a non-negative subject x channel x frequency x time EEG tensor.

Input windows are already common-average referenced and signal-QC cleaned. This
step is deterministic and label-blind, so its output may be cached. Learned
decompositions and classifiers are fitted separately inside evaluation folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch


WORKSPACE = Path(r"C:\Projects\AIBSI")
SOURCE = WORKSPACE / "preprocessing" / "04_clean_primary" / "primary"
OUTPUT = WORKSPACE / "tensor_pipeline" / "outputs" / "tensorization"

FS = 250.0
NPERSEG = 500
NOVERLAP = 250
NFFT = 500
FREQ_MIN = 1.0
FREQ_MAX = 30.0
TIME_BINS = 10
CHANNELS = np.array([
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    signals = np.load(SOURCE / "signals.npy", mmap_mode="r")
    metadata = np.load(SOURCE / "metadata.npz")
    window_subjects = np.asarray(metadata["subject"]).astype(str)
    start_seconds = np.asarray(metadata["start_seconds"])
    groups = np.asarray(metadata["group"]).astype(str)

    subjects = np.array(sorted(set(window_subjects)))
    tensor = None
    frequencies = None
    bin_counts = np.zeros((len(subjects), TIME_BINS), dtype=np.int16)
    subject_labels = np.empty(len(subjects), dtype=np.int8)
    subject_groups = np.empty(len(subjects), dtype="<U1")
    subject_age = np.empty(len(subjects), dtype=np.int16)
    subject_sex = np.empty(len(subjects), dtype="<U1")
    subject_mmse = np.empty(len(subjects), dtype=np.int16)

    for subject_i, subject in enumerate(subjects):
        indices = np.flatnonzero(window_subjects == subject)
        indices = indices[np.argsort(start_seconds[indices], kind="stable")]
        group = str(groups[indices[0]])
        subject_groups[subject_i] = group
        subject_labels[subject_i] = {"A": 0, "F": 1, "C": 2}[group]
        subject_age[subject_i] = int(metadata["age"][indices[0]])
        subject_sex[subject_i] = str(metadata["sex"][indices[0]])
        subject_mmse[subject_i] = int(metadata["mmse"][indices[0]])

        windows = np.asarray(signals[indices], dtype=np.float32)
        freq, psd = welch(
            windows,
            fs=FS,
            window="hann",
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            nfft=NFFT,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=-1,
        )
        select = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
        freq = freq[select]
        psd = np.asarray(psd[..., select], dtype=np.float64)

        # Normalize each window over its complete spatial-spectral map. This
        # retains relative channel and frequency structure while suppressing
        # arbitrary overall recording gain. Square root reduces heavy tails and
        # preserves the non-negativity required by NMF/NTF methods.
        denominator = psd.sum(axis=(1, 2), keepdims=True)
        relative_sqrt_power = np.sqrt(psd / np.maximum(denominator, 1e-30))

        if tensor is None:
            frequencies = freq.astype(np.float32)
            tensor = np.empty(
                (len(subjects), signals.shape[1], len(frequencies), TIME_BINS),
                dtype=np.float32,
            )

        # Rank-based chronological deciles guarantee populated, comparable time
        # bins despite differences in recording duration or excluded windows.
        for time_i, local_indices in enumerate(np.array_split(np.arange(len(indices)), TIME_BINS)):
            if len(local_indices) == 0:
                raise RuntimeError(f"Empty time bin for {subject}")
            bin_counts[subject_i, time_i] = len(local_indices)
            tensor[subject_i, :, :, time_i] = np.median(
                relative_sqrt_power[local_indices], axis=0
            ).astype(np.float32)

        print(
            f"tensorized {subject_i + 1:02d}/{len(subjects)} {subject} "
            f"group={group} windows={len(indices)}",
            flush=True,
        )

    assert tensor is not None and frequencies is not None
    if not np.isfinite(tensor).all() or np.min(tensor) < 0:
        raise RuntimeError("Tensor must be finite and non-negative")
    if CounterLike(subject_groups) != {"A": 23, "F": 23, "C": 23}:
        raise RuntimeError("Unexpected subject-group composition")

    np.save(OUTPUT / "tensor_relative_sqrt_1_30hz.npy", tensor)
    np.save(OUTPUT / "subjects.npy", subjects)
    np.save(OUTPUT / "labels.npy", subject_labels)
    np.save(OUTPUT / "frequencies_hz.npy", frequencies)
    np.save(OUTPUT / "channels.npy", CHANNELS)
    np.savez_compressed(
        OUTPUT / "subject_metadata.npz",
        subject=subjects,
        group=subject_groups,
        label=subject_labels,
        age=subject_age,
        sex=subject_sex,
        mmse=subject_mmse,
        time_bin_window_counts=bin_counts,
    )

    report = {
        "source": str(SOURCE),
        "output_tensor": str(OUTPUT / "tensor_relative_sqrt_1_30hz.npy"),
        "shape": list(tensor.shape),
        "axes": ["subject", "channel", "frequency", "relative_recording_time"],
        "subjects": len(subjects),
        "subject_groups": CounterLike(subject_groups),
        "channels": CHANNELS.tolist(),
        "sampling_hz": FS,
        "window_seconds": signals.shape[-1] / FS,
        "spectral_estimator": {
            "method": "Welch",
            "window": "Hann",
            "nperseg_samples": NPERSEG,
            "segment_seconds": NPERSEG / FS,
            "overlap_samples": NOVERLAP,
            "overlap_fraction": NOVERLAP / NPERSEG,
            "nfft": NFFT,
            "frequency_resolution_hz": FS / NFFT,
            "retained_frequency_hz": [FREQ_MIN, FREQ_MAX],
            "retained_frequency_points": len(frequencies),
        },
        "nonnegative_transform": "sqrt(window PSD / total window channel-frequency PSD)",
        "time_axis": {
            "bins": TIME_BINS,
            "definition": "chronological rank deciles within each subject",
            "aggregation": "elementwise median of window maps",
            "minimum_windows_per_subject_bin": int(bin_counts.min()),
            "maximum_windows_per_subject_bin": int(bin_counts.max()),
        },
        "value_range": [float(tensor.min()), float(tensor.max())],
        "finite": bool(np.isfinite(tensor).all()),
        "nonnegative": bool(np.min(tensor) >= 0),
        "leakage_note": "Tensorization is fixed and label-blind. Decomposition, learned scaling, graph estimation, and classifiers must be fit within training folds.",
    }
    (OUTPUT / "tensorization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def CounterLike(values: np.ndarray) -> dict[str, int]:
    return {str(value): int(np.sum(values == value)) for value in sorted(set(values))}


if __name__ == "__main__":
    main()
