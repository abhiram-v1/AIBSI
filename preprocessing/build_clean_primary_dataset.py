"""Create a cleaned, exactly balanced derivative of the primary EEG dataset.

The existing balanced primary dataset is read-only. All output signals receive a
19-channel common-average reference and per-window/per-channel mean removal.
Signal-QC exclusions and balance-only trims are retained in a separate array.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


SOURCE = Path(r"C:\Projects\AIBSI\preprocessing\02_balanced_primary\primary")
OUTPUT = Path(r"C:\Projects\AIBSI\preprocessing\04_clean_primary")
KEEP_DIR = OUTPUT / "primary"
EXCLUDED_DIR = OUTPUT / "excluded"

MAX_ABS_UV = 400.0
REVIEW_STEP_UV = 150.0
HARD_STEP_UV = 300.0
MIN_CHANNEL_STD_UV = 1.0
KEEP_SHUFFLE_SEED = 4506
EXCLUDED_SHUFFLE_SEED = 4507
GROUPS = ("A", "F", "C")


def car_transform(block: np.ndarray) -> np.ndarray:
    transformed = np.asarray(block, dtype=np.float32).copy()
    transformed -= transformed.mean(axis=1, keepdims=True)
    transformed -= transformed.mean(axis=2, keepdims=True)
    return transformed


def signal_qc_reason(max_abs: float, max_step: float, min_std: float) -> str:
    reasons = []
    if min_std < MIN_CHANNEL_STD_UV:
        reasons.append("near_flat_channel_after_car")
    if max_abs >= MAX_ABS_UV:
        reasons.append("extreme_amplitude_after_car")
    if max_step >= HARD_STEP_UV:
        reasons.append("extreme_abrupt_step_after_car")
    elif max_step >= REVIEW_STEP_UV:
        reasons.append("abrupt_step_after_car")
    return ";".join(reasons)


def write_partition(
    directory: Path,
    source_signals: np.ndarray,
    source_meta: dict[str, np.ndarray],
    source_manifest: list[dict],
    order: np.ndarray,
    max_abs: np.ndarray,
    max_step: np.ndarray,
    min_std: np.ndarray,
    exclusion_reason: np.ndarray,
    is_primary: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    shape = (len(order), source_signals.shape[1], source_signals.shape[2])
    out = open_memmap(directory / "signals.npy", mode="w+", dtype=np.float32, shape=shape)
    for dst_start in range(0, len(order), 128):
        dst_stop = min(dst_start + 128, len(order))
        idx = order[dst_start:dst_stop]
        out[dst_start:dst_stop] = car_transform(source_signals[idx])
    out.flush()
    del out

    np.save(directory / "labels.npy", np.asarray(source_meta["label"])[order])
    payload = {key: np.asarray(value)[order] for key, value in source_meta.items()}
    payload.update(
        {
            "original_array_index": order.astype(np.int32),
            "max_abs_uv_after_car": max_abs[order].astype(np.float32),
            "max_step_uv_after_car": max_step[order].astype(np.float32),
            "min_channel_std_uv_after_car": min_std[order].astype(np.float32),
            "cleaning_status": np.full(len(order), "clean_primary" if is_primary else "excluded"),
            "exclusion_reason": exclusion_reason[order].astype(str),
        }
    )
    np.savez_compressed(directory / "metadata.npz", **payload)

    with (directory / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for new_index, old_index in enumerate(order):
            row = dict(source_manifest[int(old_index)])
            row.update(
                {
                    "array_index": new_index,
                    "original_array_index": int(old_index),
                    "reference": "common_average_19_scalp_channels",
                    "max_abs_uv_after_car": round(float(max_abs[old_index]), 6),
                    "max_step_uv_after_car": round(float(max_step[old_index]), 6),
                    "min_channel_std_uv_after_car": round(float(min_std[old_index]), 6),
                    "cleaning_status": "clean_primary" if is_primary else "excluded",
                    "exclusion_reason": str(exclusion_reason[old_index]),
                }
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")

    signals = np.load(SOURCE / "signals.npy", mmap_mode="r")
    loaded_meta = np.load(SOURCE / "metadata.npz")
    meta = {key: loaded_meta[key] for key in loaded_meta.files}
    groups = np.asarray(meta["group"]).astype(str)
    subjects = np.asarray(meta["subject"]).astype(str)
    buckets = np.asarray(meta["bucket"]).astype(str)
    n = len(signals)
    source_manifest = [
        json.loads(line)
        for line in (SOURCE / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source_manifest) != n:
        raise ValueError("Manifest and signal counts do not match")

    max_abs = np.empty(n, dtype=np.float64)
    max_step = np.empty(n, dtype=np.float64)
    min_std = np.empty(n, dtype=np.float64)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        block = car_transform(signals[start:stop])
        max_abs[start:stop] = np.max(np.abs(block), axis=(1, 2))
        max_step[start:stop] = np.max(np.abs(np.diff(block, axis=2)), axis=(1, 2))
        min_std[start:stop] = np.std(block, axis=2).min(axis=1)

    exclusion_reason = np.array(
        [signal_qc_reason(a, d, s) for a, d, s in zip(max_abs, max_step, min_std)],
        dtype="<U96",
    )
    signal_excluded = exclusion_reason != ""
    remaining_counts = {group: int(((groups == group) & ~signal_excluded).sum()) for group in GROUPS}
    target_per_group = min(remaining_counts.values())

    # The FTD loss is concentrated in its medium-duration bucket. To retain the
    # closest available duration distribution, balancing trims are therefore
    # taken from the medium bucket of AD and CN, spread nearly equally across
    # subjects. Within each subject, borderline high-step windows are trimmed first.
    balance_trim = np.zeros(n, dtype=bool)
    trim_plan = []
    for group in GROUPS:
        needed = remaining_counts[group] - target_per_group
        if needed == 0:
            continue
        eligible = np.flatnonzero(
            (groups == group) & (buckets == "medium") & ~signal_excluded
        )
        eligible_subjects = sorted(set(subjects[eligible]))
        if not eligible_subjects:
            raise RuntimeError(f"No medium-bucket subjects available for {group} balancing")
        base, extra = divmod(needed, len(eligible_subjects))
        subject_sizes = {
            subject: int(np.sum(subjects[eligible] == subject)) for subject in eligible_subjects
        }
        quota_order = sorted(eligible_subjects, key=lambda subject: (-subject_sizes[subject], subject))
        quota = {subject: base + (rank < extra) for rank, subject in enumerate(quota_order)}
        for subject in eligible_subjects:
            candidates = eligible[subjects[eligible] == subject]
            # Quality-first deterministic trimming, with original index as tie-breaker.
            ranked = sorted(
                candidates.tolist(),
                key=lambda i: (max_step[i] / REVIEW_STEP_UV + max_abs[i] / MAX_ABS_UV, i),
                reverse=True,
            )
            selected = ranked[: quota[subject]]
            balance_trim[selected] = True
            trim_plan.append(
                {"group": group, "bucket": "medium", "subject": subject, "windows": len(selected)}
            )
        if int(balance_trim[(groups == group)].sum()) != needed:
            raise RuntimeError(f"Incorrect balance trim count for {group}")

    exclusion_reason[balance_trim] = "balance_trim_medium_duration_quality_first"
    keep = ~signal_excluded & ~balance_trim
    if any(int((keep & (groups == group)).sum()) != target_per_group for group in GROUPS):
        raise RuntimeError("Final class counts are not equal")

    keep_idx = np.flatnonzero(keep)
    excluded_idx = np.flatnonzero(~keep)
    keep_idx = keep_idx[np.random.default_rng(KEEP_SHUFFLE_SEED).permutation(len(keep_idx))]
    excluded_idx = excluded_idx[
        np.random.default_rng(EXCLUDED_SHUFFLE_SEED).permutation(len(excluded_idx))
    ]

    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        write_partition(
            KEEP_DIR, signals, meta, source_manifest, keep_idx,
            max_abs, max_step, min_std, exclusion_reason, True,
        )
        write_partition(
            EXCLUDED_DIR, signals, meta, source_manifest, excluded_idx,
            max_abs, max_step, min_std, exclusion_reason, False,
        )
    except Exception:
        # Only clean a newly-created, task-specific directory after a failed build.
        shutil.rmtree(OUTPUT)
        raise

    def grouped(mask: np.ndarray) -> dict:
        result = {}
        for group in GROUPS:
            gm = mask & (groups == group)
            result[group] = {
                "windows": int(gm.sum()),
                "subjects": int(len(set(subjects[gm]))),
                "duration_seconds": int(gm.sum()) * 4,
                "buckets": {
                    bucket: int((gm & (buckets == bucket)).sum())
                    for bucket in ("short", "medium", "long")
                },
            }
        return result

    reason_counts = Counter()
    for reason in exclusion_reason[~keep]:
        for item in str(reason).split(";"):
            reason_counts[item] += 1

    # Verify the actual stored arrays and reference properties.
    stored = np.load(KEEP_DIR / "signals.npy", mmap_mode="r")
    stored_labels = np.load(KEEP_DIR / "labels.npy")
    stored_meta = np.load(KEEP_DIR / "metadata.npz")
    max_channel_mean = 0.0
    max_window_channel_average = 0.0
    all_finite = True
    for start in range(0, len(stored), 128):
        block = np.asarray(stored[start : start + 128])
        all_finite &= bool(np.isfinite(block).all())
        max_channel_mean = max(max_channel_mean, float(np.max(np.abs(block.mean(axis=2)))))
        max_window_channel_average = max(
            max_window_channel_average, float(np.max(np.abs(block.mean(axis=1))))
        )

    report = {
        "source": str(SOURCE),
        "source_unchanged": True,
        "output": str(OUTPUT),
        "transform": {
            "reference": "common average across all 19 scalp channels at every time sample",
            "window_transform": "per-channel temporal mean removal after rereferencing",
            "additional_filtering": "none; source derivatives already use 0.5-45 Hz filtering, ASR and ICA",
        },
        "qc_thresholds": {
            "maximum_absolute_amplitude_uv": MAX_ABS_UV,
            "abrupt_step_review_uv_per_4ms": REVIEW_STEP_UV,
            "extreme_abrupt_step_uv_per_4ms": HARD_STEP_UV,
            "minimum_channel_standard_deviation_uv": MIN_CHANNEL_STD_UV,
        },
        "before": grouped(np.ones(n, dtype=bool)),
        "after_signal_qc_before_balance": remaining_counts,
        "target_windows_per_group": target_per_group,
        "primary": grouped(keep),
        "excluded": grouped(~keep),
        "excluded_reason_counts": dict(reason_counts),
        "signal_qc_exclusions": int(signal_excluded.sum()),
        "balance_only_trims": int(balance_trim.sum()),
        "trim_plan": trim_plan,
        "shuffle_seeds": {"primary": KEEP_SHUFFLE_SEED, "excluded": EXCLUDED_SHUFFLE_SEED},
        "high_frequency_policy": "No windows removed from 30-45 Hz power alone; use 1-30 Hz primary analysis and 30-45 Hz sensitivity analysis.",
        "checks": {
            "stored_shape": list(stored.shape),
            "stored_labels": {str(label): int((stored_labels == label).sum()) for label in (0, 1, 2)},
            "stored_metadata_rows": int(len(stored_meta["sample_id"])),
            "equal_group_counts": len(set(int((keep & (groups == g)).sum()) for g in GROUPS)) == 1,
            "all_69_subjects_retained": len(set(subjects[keep])) == 69,
            "primary_excluded_partition_all_source_windows": int(keep.sum() + (~keep).sum()) == n,
            "all_finite": bool(all_finite),
            "maximum_absolute_per_channel_temporal_mean_uv": max_channel_mean,
            "maximum_absolute_across_channel_mean_uv": max_window_channel_average,
            "primary_has_no_qc_candidate": bool(
                np.all(max_abs[keep] < MAX_ABS_UV)
                and np.all(max_step[keep] < REVIEW_STEP_UV)
                and np.all(min_std[keep] >= MIN_CHANNEL_STD_UV)
            ),
        },
        "important_note": "All future validation/test splits must remain grouped by subject. Any learned normalization or tensor decomposition must be fitted using training subjects only.",
    }
    (OUTPUT / "processing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "target_per_group": target_per_group,
        "primary_windows": int(keep.sum()),
        "signal_qc_exclusions": int(signal_excluded.sum()),
        "balance_only_trims": int(balance_trim.sum()),
        "excluded_windows": int((~keep).sum()),
        "checks": report["checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
