"""Build balanced, duration-bucketed EEG window datasets from ds004504.

The source BIDS dataset is never modified. The primary dataset uses FTD as the
limiting reference and stores equal numbers of four-second windows for AD, FTD,
and CN. Every complete window not used by the primary dataset is retained in a
separate leftover dataset. Incomplete terminal segments are also preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.io import loadmat
from scipy.optimize import Bounds, LinearConstraint, milp


WORKSPACE = Path(r"C:\Projects\AIBSI")
SOURCE_ROOT = WORKSPACE / "dataset" / "ds004504"
OUTPUT_ROOT = WORKSPACE / "preprocessing" / "02_balanced_primary"

SOURCE_HZ = 500
TARGET_HZ = 250
WINDOW_SECONDS = 4
WINDOW_SAMPLES = TARGET_HZ * WINDOW_SECONDS
CHANNEL_COUNT = 19
SHUFFLE_SEED = 4504

# Conservative, label-blind QC flags. Flagged windows remain in leftovers.
MAX_ABS_UV = 500.0
MIN_CHANNEL_STD_UV = 0.1

GROUP_NAMES = {
    "A": "Alzheimer's disease",
    "F": "Frontotemporal dementia",
    "C": "Cognitively normal",
}
LABELS = {"A": 0, "F": 1, "C": 2}


def load_records() -> list[dict]:
    with (SOURCE_ROOT / "participants.tsv").open(encoding="utf-8") as stream:
        participants = list(csv.DictReader(stream, delimiter="\t"))

    records = []
    for participant in participants:
        subject = participant["participant_id"]
        stem = f"{subject}_task-eyesclosed"
        raw_dir = SOURCE_ROOT / subject / "eeg"
        derivative_path = SOURCE_ROOT / "derivatives" / subject / "eeg" / f"{stem}_eeg.set"
        metadata_path = raw_dir / f"{stem}_eeg.json"
        if not derivative_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing EEG or metadata for {subject}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records.append(
            {
                "subject": subject,
                "group": participant["Group"],
                "age": int(participant["Age"]),
                "sex": participant["Gender"],
                "mmse": int(participant["MMSE"].strip()),
                "duration_seconds": float(metadata["RecordingDuration"]),
                "derivative_path": derivative_path,
            }
        )
    return records


def load_downsampled(record: dict) -> np.ndarray:
    loaded = loadmat(
        record["derivative_path"],
        variable_names=["data", "srate", "nbchan"],
        squeeze_me=True,
        struct_as_record=False,
    )
    source_hz = int(round(float(loaded["srate"])))
    channels = int(loaded["nbchan"])
    if source_hz != SOURCE_HZ or channels != CHANNEL_COUNT:
        raise ValueError(
            f"Unexpected recording format for {record['subject']}: "
            f"{source_hz} Hz, {channels} channels"
        )
    data = np.asarray(loaded["data"], dtype=np.float32)
    if data.ndim != 2 or data.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"Unexpected data shape for {record['subject']}: {data.shape}")
    # The derivatives are already low-pass filtered at 45 Hz. Taking every
    # second sample is safe when moving from 500 Hz to a 250 Hz target rate.
    return np.ascontiguousarray(data[:, ::2])


def scan_recordings(records: list[dict]) -> dict[str, dict]:
    scan = {}
    for index, record in enumerate(records, start=1):
        data = load_downsampled(record)
        window_count = data.shape[1] // WINDOW_SAMPLES
        usable = data[:, : window_count * WINDOW_SAMPLES]
        windows = usable.reshape(CHANNEL_COUNT, window_count, WINDOW_SAMPLES).transpose(1, 0, 2)

        finite = np.isfinite(windows).all(axis=(1, 2))
        maximum = np.max(np.abs(windows), axis=(1, 2))
        minimum_channel_std = np.std(windows, axis=2).min(axis=1)
        statuses = np.full(window_count, "ok", dtype="<U24")
        statuses[~finite] = "non_finite"
        statuses[finite & (minimum_channel_std < MIN_CHANNEL_STD_UV)] = "flat_channel"
        statuses[finite & (minimum_channel_std >= MIN_CHANNEL_STD_UV) & (maximum > MAX_ABS_UV)] = (
            "extreme_amplitude"
        )

        scan[record["subject"]] = {
            "downsampled_samples": int(data.shape[1]),
            "complete_windows": int(window_count),
            "tail_samples": int(data.shape[1] - window_count * WINDOW_SAMPLES),
            "statuses": statuses.tolist(),
            "good_indices": np.flatnonzero(statuses == "ok").astype(int).tolist(),
            "maximum_abs_uv": float(maximum.max()) if window_count else 0.0,
            "minimum_channel_std_uv": float(minimum_channel_std.min()) if window_count else 0.0,
        }
        print(
            f"scan {index:02d}/{len(records)} {record['subject']} "
            f"windows={window_count} good={len(scan[record['subject']]['good_indices'])}",
            flush=True,
        )
    return scan


def set_reference_buckets(records: list[dict], scan: dict[str, dict]) -> tuple[list[dict], dict]:
    references = sorted(
        [dict(record) for record in records if record["group"] == "F"],
        key=lambda record: (record["duration_seconds"], record["subject"]),
    )
    if len(references) != 23:
        raise ValueError(f"Expected 23 FTD subjects, found {len(references)}")

    for index, reference in enumerate(references):
        reference["bucket"] = "short" if index < 8 else "medium" if index < 15 else "long"
        reference["good_windows"] = len(scan[reference["subject"]]["good_indices"])
        reference["primary_windows"] = reference["good_windows"]

    short_end = (references[7]["duration_seconds"] + references[8]["duration_seconds"]) / 2
    medium_end = (references[14]["duration_seconds"] + references[15]["duration_seconds"]) / 2
    thresholds = {
        "short_upper_seconds": short_end,
        "medium_upper_seconds": medium_end,
    }
    return references, thresholds


def source_duration_bucket(duration_seconds: float, thresholds: dict) -> str:
    if duration_seconds <= thresholds["short_upper_seconds"]:
        return "short"
    if duration_seconds <= thresholds["medium_upper_seconds"]:
        return "medium"
    return "long"


def match_group(
    records: list[dict],
    scan: dict[str, dict],
    references: list[dict],
    group: str,
    male_quota: int,
) -> list[dict]:
    candidates = sorted(
        [dict(record) for record in records if record["group"] == group],
        key=lambda record: record["subject"],
    )
    for candidate in candidates:
        candidate["good_windows"] = len(scan[candidate["subject"]]["good_indices"])

    reference_count = len(references)
    candidate_count = len(candidates)
    variable_count = reference_count * candidate_count

    objective = np.empty(variable_count, dtype=float)
    for ref_index, reference in enumerate(references):
        for candidate_index, candidate in enumerate(candidates):
            position = ref_index * candidate_count + candidate_index
            shortage = max(0, reference["good_windows"] - candidate["good_windows"])
            objective[position] = (
                shortage * 1_000
                + int(reference["sex"] != candidate["sex"]) * 300
                + abs(reference["age"] - candidate["age"]) * 10
                + abs(reference["good_windows"] - candidate["good_windows"]) * 0.1
            )

    rows = []
    lower = []
    upper = []

    for ref_index in range(reference_count):
        row = np.zeros(variable_count)
        start = ref_index * candidate_count
        row[start : start + candidate_count] = 1
        rows.append(row)
        lower.append(1)
        upper.append(1)

    for candidate_index in range(candidate_count):
        row = np.zeros(variable_count)
        row[candidate_index::candidate_count] = 1
        rows.append(row)
        lower.append(0)
        upper.append(1)

    male_row = np.zeros(variable_count)
    for candidate_index, candidate in enumerate(candidates):
        if candidate["sex"] == "M":
            male_row[candidate_index::candidate_count] = 1
    rows.append(male_row)
    lower.append(male_quota)
    upper.append(male_quota)

    for bucket in ["short", "medium", "long"]:
        capacity_row = np.zeros(variable_count)
        bucket_target = sum(
            reference["good_windows"] for reference in references if reference["bucket"] == bucket
        )
        for ref_index, reference in enumerate(references):
            if reference["bucket"] != bucket:
                continue
            for candidate_index, candidate in enumerate(candidates):
                position = ref_index * candidate_count + candidate_index
                capacity_row[position] = candidate["good_windows"]
        rows.append(capacity_row)
        lower.append(bucket_target)
        upper.append(np.inf)

    constraints = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=constraints,
        options={"time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(f"Matching failed for {group}: {result.message}")

    selected = []
    solution = result.x.reshape(reference_count, candidate_count)
    for ref_index, reference in enumerate(references):
        candidate_index = int(np.argmax(solution[ref_index]))
        if solution[ref_index, candidate_index] < 0.5:
            raise RuntimeError(f"No assignment for reference {reference['subject']}")
        candidate = dict(candidates[candidate_index])
        candidate.update(
            {
                "reference_subject": reference["subject"],
                "bucket": reference["bucket"],
                "reference_windows": reference["good_windows"],
                "primary_windows": min(reference["good_windows"], candidate["good_windows"]),
            }
        )
        selected.append(candidate)

    for bucket in ["short", "medium", "long"]:
        target = sum(
            reference["good_windows"] for reference in references if reference["bucket"] == bucket
        )
        members = [record for record in selected if record["bucket"] == bucket]
        deficit = target - sum(record["primary_windows"] for record in members)
        for record in sorted(
            members,
            key=lambda item: item["good_windows"] - item["primary_windows"],
            reverse=True,
        ):
            if deficit <= 0:
                break
            addition = min(deficit, record["good_windows"] - record["primary_windows"])
            record["primary_windows"] += addition
            deficit -= addition
        if deficit:
            raise RuntimeError(f"Insufficient {group} capacity in {bucket}: {deficit} windows")
    return selected


def build_selection_plan(records: list[dict], scan: dict[str, dict]) -> dict:
    references, thresholds = set_reference_buckets(records, scan)
    selected = {
        "F": references,
        # AD has only 12 male subjects. Keeping all 12 yields the closest feasible
        # sex composition without duplicating anyone.
        "A": match_group(records, scan, references, "A", male_quota=12),
        # Thirteen male and ten female CN subjects provides a middle-ground sex
        # ratio between the fixed FTD composition and the feasible AD composition.
        "C": match_group(records, scan, references, "C", male_quota=13),
    }
    return {"selected": selected, "bucket_thresholds": thresholds}


def even_temporal_selection(good_indices: list[int], count: int) -> list[int]:
    if count > len(good_indices):
        raise ValueError(f"Cannot select {count} from {len(good_indices)} good windows")
    if count == len(good_indices):
        return list(good_indices)
    positions = np.floor((np.arange(count) + 0.5) * len(good_indices) / count).astype(int)
    selected = [good_indices[position] for position in positions]
    if len(set(selected)) != count:
        raise RuntimeError("Temporal selection produced duplicate windows")
    return selected


def selected_subject_summary(group_records: list[dict]) -> dict:
    ages = [record["age"] for record in group_records]
    return {
        "subjects": len(group_records),
        "windows": sum(record["primary_windows"] for record in group_records),
        "duration_seconds": sum(record["primary_windows"] for record in group_records)
        * WINDOW_SECONDS,
        "sex_subjects": dict(Counter(record["sex"] for record in group_records)),
        "age": {
            "min": min(ages),
            "max": max(ages),
            "mean": round(float(np.mean(ages)), 2),
            "sd": round(float(np.std(ages, ddof=1)), 2),
        },
        "buckets": {
            bucket: {
                "subjects": sum(record["bucket"] == bucket for record in group_records),
                "windows": sum(
                    record["primary_windows"]
                    for record in group_records
                    if record["bucket"] == bucket
                ),
            }
            for bucket in ["short", "medium", "long"]
        },
    }


def plan_report(records: list[dict], scan: dict[str, dict], selection_plan: dict) -> dict:
    source = {}
    for group in ["A", "F", "C"]:
        members = [record for record in records if record["group"] == group]
        source[group] = {
            "subjects": len(members),
            "duration_seconds": round(sum(record["duration_seconds"] for record in members), 1),
            "complete_windows": sum(scan[record["subject"]]["complete_windows"] for record in members),
            "qc_good_windows": sum(len(scan[record["subject"]]["good_indices"]) for record in members),
            "qc_flagged_windows": sum(
                scan[record["subject"]]["complete_windows"]
                - len(scan[record["subject"]]["good_indices"])
                for record in members
            ),
        }
    return {
        "sample_unit": "one non-overlapping 4-second window with 19 channels and 1000 samples per channel",
        "source_hz": SOURCE_HZ,
        "target_hz": TARGET_HZ,
        "limiting_reference": "F",
        "source_summary": source,
        "bucket_thresholds": selection_plan["bucket_thresholds"],
        "expected_primary": {
            group: selected_subject_summary(selection_plan["selected"][group])
            for group in ["A", "F", "C"]
        },
    }


def make_entries(
    records: list[dict], scan: dict[str, dict], selection_plan: dict
) -> tuple[list[dict], list[dict]]:
    selected_by_subject = {}
    for group, group_records in selection_plan["selected"].items():
        for record in group_records:
            selected_by_subject[record["subject"]] = record

    primary_entries = []
    leftover_entries = []
    thresholds = selection_plan["bucket_thresholds"]

    for record in records:
        subject_scan = scan[record["subject"]]
        selected_record = selected_by_subject.get(record["subject"])
        good_indices = subject_scan["good_indices"]
        primary_indices = set()
        if selected_record:
            primary_indices = set(
                even_temporal_selection(good_indices, selected_record["primary_windows"])
            )

        for window_index, qc_status in enumerate(subject_scan["statuses"]):
            base = {
                "sample_id": f"{record['subject']}__w{window_index:04d}",
                "subject": record["subject"],
                "group": record["group"],
                "label": LABELS[record["group"]],
                "diagnosis": GROUP_NAMES[record["group"]],
                "age": record["age"],
                "sex": record["sex"],
                "mmse": record["mmse"],
                "window_index": window_index,
                "start_seconds": window_index * WINDOW_SECONDS,
                "duration_seconds": WINDOW_SECONDS,
                "source_duration_seconds": record["duration_seconds"],
                "bucket": selected_record["bucket"]
                if selected_record
                else source_duration_bucket(record["duration_seconds"], thresholds),
                "qc_status": qc_status,
            }
            if window_index in primary_indices:
                if qc_status != "ok":
                    raise RuntimeError(f"QC-flagged window entered primary: {base['sample_id']}")
                base["selection_reason"] = (
                    "limiting_reference"
                    if record["group"] == "F"
                    else "duration_age_sex_matched"
                )
                primary_entries.append(base)
            else:
                if qc_status != "ok":
                    reason = f"qc_flag:{qc_status}"
                elif not selected_record:
                    reason = "unselected_subject"
                else:
                    reason = "trimmed_excess_window"
                base["selection_reason"] = reason
                leftover_entries.append(base)
    return primary_entries, leftover_entries


def shuffle_entries(entries: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entries))
    return [entries[int(index)] for index in order]


def max_label_run(entries: list[dict]) -> int:
    longest = current = 0
    previous = None
    for entry in entries:
        if entry["group"] == previous:
            current += 1
        else:
            previous = entry["group"]
            current = 1
        longest = max(longest, current)
    return longest


def entry_statistics(entries: list[dict]) -> dict:
    subjects_by_group = defaultdict(set)
    for entry in entries:
        subjects_by_group[entry["group"]].add(entry["subject"])
    return {
        "windows": len(entries),
        "duration_seconds": len(entries) * WINDOW_SECONDS,
        "groups": {
            group: {
                "windows": sum(entry["group"] == group for entry in entries),
                "subjects_with_complete_windows": len(subjects_by_group[group]),
                "buckets": dict(
                    Counter(entry["bucket"] for entry in entries if entry["group"] == group)
                ),
                "reasons": dict(
                    Counter(
                        entry["selection_reason"]
                        for entry in entries
                        if entry["group"] == group
                    )
                ),
            }
            for group in ["A", "F", "C"]
        },
        "qc_status": dict(Counter(entry["qc_status"] for entry in entries)),
        "maximum_consecutive_same_group_after_shuffle": max_label_run(entries),
    }


def write_manifest(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")


def write_metadata_npz(path: Path, entries: list[dict]) -> None:
    np.savez_compressed(
        path,
        sample_id=np.array([entry["sample_id"] for entry in entries]),
        subject=np.array([entry["subject"] for entry in entries]),
        group=np.array([entry["group"] for entry in entries]),
        label=np.array([entry["label"] for entry in entries], dtype=np.int8),
        bucket=np.array([entry["bucket"] for entry in entries]),
        age=np.array([entry["age"] for entry in entries], dtype=np.int16),
        sex=np.array([entry["sex"] for entry in entries]),
        mmse=np.array([entry["mmse"] for entry in entries], dtype=np.int16),
        window_index=np.array([entry["window_index"] for entry in entries], dtype=np.int32),
        start_seconds=np.array([entry["start_seconds"] for entry in entries], dtype=np.int32),
        qc_status=np.array([entry["qc_status"] for entry in entries]),
        selection_reason=np.array([entry["selection_reason"] for entry in entries]),
    )


def write_signal_datasets(
    records: list[dict],
    scan: dict[str, dict],
    primary_entries: list[dict],
    leftover_entries: list[dict],
) -> list[dict]:
    primary_dir = OUTPUT_ROOT / "primary"
    leftover_dir = OUTPUT_ROOT / "leftover"
    partial_dir = leftover_dir / "partial_segments"
    primary_dir.mkdir(parents=True)
    partial_dir.mkdir(parents=True)

    primary_signals = open_memmap(
        primary_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(primary_entries), CHANNEL_COUNT, WINDOW_SAMPLES),
    )
    leftover_signals = open_memmap(
        leftover_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(leftover_entries), CHANNEL_COUNT, WINDOW_SAMPLES),
    )

    primary_locations = defaultdict(list)
    leftover_locations = defaultdict(list)
    for destination, entry in enumerate(primary_entries):
        primary_locations[entry["subject"]].append((destination, entry["window_index"]))
    for destination, entry in enumerate(leftover_entries):
        leftover_locations[entry["subject"]].append((destination, entry["window_index"]))

    partial_manifest = []
    for index, record in enumerate(records, start=1):
        data = load_downsampled(record)
        window_count = scan[record["subject"]]["complete_windows"]
        usable_samples = window_count * WINDOW_SAMPLES
        windows = data[:, :usable_samples].reshape(
            CHANNEL_COUNT, window_count, WINDOW_SAMPLES
        ).transpose(1, 0, 2)

        for destination, window_index in primary_locations[record["subject"]]:
            window = windows[window_index]
            primary_signals[destination] = window - window.mean(axis=1, keepdims=True)
        for destination, window_index in leftover_locations[record["subject"]]:
            window = windows[window_index]
            leftover_signals[destination] = window - window.mean(axis=1, keepdims=True)

        tail = data[:, usable_samples:]
        if tail.shape[1]:
            tail_path = partial_dir / f"{record['subject']}__partial.npy"
            np.save(tail_path, tail - tail.mean(axis=1, keepdims=True))
            partial_manifest.append(
                {
                    "subject": record["subject"],
                    "group": record["group"],
                    "samples_per_channel": int(tail.shape[1]),
                    "duration_seconds": tail.shape[1] / TARGET_HZ,
                    "path": str(tail_path.relative_to(WORKSPACE)).replace("\\", "/"),
                }
            )
        print(f"write {index:02d}/{len(records)} {record['subject']}", flush=True)

    primary_signals.flush()
    leftover_signals.flush()
    del primary_signals
    del leftover_signals

    np.save(primary_dir / "labels.npy", np.array([e["label"] for e in primary_entries], dtype=np.int8))
    np.save(leftover_dir / "labels.npy", np.array([e["label"] for e in leftover_entries], dtype=np.int8))
    write_metadata_npz(primary_dir / "metadata.npz", primary_entries)
    write_metadata_npz(leftover_dir / "metadata.npz", leftover_entries)
    write_manifest(primary_dir / "manifest.jsonl", primary_entries)
    write_manifest(leftover_dir / "manifest.jsonl", leftover_entries)
    (leftover_dir / "partial_segments_manifest.json").write_text(
        json.dumps(partial_manifest, indent=2), encoding="utf-8"
    )
    return partial_manifest


def serializable_selection_plan(selection_plan: dict) -> dict:
    return {
        "bucket_thresholds": selection_plan["bucket_thresholds"],
        "selected": {
            group: [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"derivative_path"}
                }
                for record in group_records
            ]
            for group, group_records in selection_plan["selected"].items()
        },
    }


def verify_outputs(primary_entries: list[dict], leftover_entries: list[dict], scan: dict) -> dict:
    primary_counts = Counter(entry["group"] for entry in primary_entries)
    primary_ids = {entry["sample_id"] for entry in primary_entries}
    leftover_ids = {entry["sample_id"] for entry in leftover_entries}
    source_complete = sum(subject_scan["complete_windows"] for subject_scan in scan.values())
    checks = {
        "equal_primary_group_windows": len(set(primary_counts.values())) == 1,
        "primary_has_three_groups": set(primary_counts) == {"A", "F", "C"},
        "primary_leftover_no_overlap": primary_ids.isdisjoint(leftover_ids),
        "complete_window_partition": len(primary_entries) + len(leftover_entries) == source_complete,
        "primary_only_qc_ok": all(entry["qc_status"] == "ok" for entry in primary_entries),
        "primary_signal_shape": list(np.load(OUTPUT_ROOT / "primary" / "signals.npy", mmap_mode="r").shape),
        "leftover_signal_shape": list(np.load(OUTPUT_ROOT / "leftover" / "signals.npy", mmap_mode="r").shape),
    }
    if not all(value is True or isinstance(value, list) for value in checks.values()):
        raise RuntimeError(f"Output verification failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    records = load_records()
    scan = scan_recordings(records)
    selection_plan = build_selection_plan(records, scan)
    expected = plan_report(records, scan, selection_plan)
    print("PLAN")
    print(json.dumps(expected, indent=2))
    if args.scan_only:
        return

    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    primary_entries, leftover_entries = make_entries(records, scan, selection_plan)
    primary_entries = shuffle_entries(primary_entries, SHUFFLE_SEED)
    leftover_entries = shuffle_entries(leftover_entries, SHUFFLE_SEED + 1)

    partial_manifest = write_signal_datasets(
        records, scan, primary_entries, leftover_entries
    )
    primary_stats = entry_statistics(primary_entries)
    leftover_stats = entry_statistics(leftover_entries)
    checks = verify_outputs(primary_entries, leftover_entries, scan)

    (OUTPUT_ROOT / "selection_plan.json").write_text(
        json.dumps(serializable_selection_plan(selection_plan), indent=2), encoding="utf-8"
    )
    report = {
        "configuration": {
            "source": str(SOURCE_ROOT),
            "source_derivatives": "artifact-cleaned EEGLAB .set files",
            "source_hz": SOURCE_HZ,
            "target_hz": TARGET_HZ,
            "downsampling": "every second sample; source derivatives are low-pass filtered at 45 Hz",
            "window_seconds": WINDOW_SECONDS,
            "window_samples_per_channel": WINDOW_SAMPLES,
            "channels": CHANNEL_COUNT,
            "window_transform": "per-channel mean removal; no cross-subject normalization",
            "qc": {
                "maximum_absolute_amplitude_uv": MAX_ABS_UV,
                "minimum_channel_standard_deviation_uv": MIN_CHANNEL_STD_UV,
                "non_finite_values": "flag",
            },
            "shuffle_seed_primary": SHUFFLE_SEED,
            "shuffle_seed_leftover": SHUFFLE_SEED + 1,
        },
        "plan": expected,
        "primary": primary_stats,
        "leftover": leftover_stats,
        "partial_segments": {
            "count": len(partial_manifest),
            "total_samples_per_channel": sum(item["samples_per_channel"] for item in partial_manifest),
            "total_duration_seconds": sum(item["duration_seconds"] for item in partial_manifest),
        },
        "checks": checks,
        "important_note": "Future train/validation/test splits must group by subject despite shuffled window order.",
    }
    (OUTPUT_ROOT / "processing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("FINAL_REPORT")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
