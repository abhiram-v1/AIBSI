from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_PREPARED = HERE / "outputs" / "prepared"
OUT = HERE / "outputs" / "eegpt" / "prepared"
PARTS = OUT / "participant_parts"

sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
import run_pipeline as source  # noqa: E402


INPUT_FS = 128
OUTPUT_FS = 256
WINDOW_SECONDS = 4
WINDOW_SAMPLES = INPUT_FS * WINDOW_SECONDS
EDGE_SECONDS = 10
MAX_WINDOWS_PER_SUBJECT = 40


def prepare_subject(row: dict) -> tuple[np.ndarray, dict]:
    record = source.Record(
        subject_id=row["subject_id"],
        source=row["source"],
        site=row["site"],
        label=row["label"],
        path=row["path"],
    )
    data, header = source.load_harmonized(record)
    start = EDGE_SECONDS * INPUT_FS
    stop = data.shape[1] - EDGE_SECONDS * INPUT_FS
    accepted: list[np.ndarray] = []
    rejected = Counter()
    for offset in range(start, stop - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        window = data[:, offset : offset + WINDOW_SAMPLES]
        ok, reason = source.valid_window(window)
        if not ok:
            rejected[reason] += 1
            continue
        # EEGPT downstream loaders use microvolts. Average reference was already
        # applied by load_harmonized; remove each channel's residual DC before
        # band-limited resampling from 128 Hz to EEGPT's native 256 Hz.
        window = (window - window.mean(axis=1, keepdims=True)) * 1e6
        window = resample_poly(window, OUTPUT_FS, INPUT_FS, axis=1)
        window = np.clip(window, -500.0, 500.0)
        accepted.append(window.astype(np.float16))
    accepted = source.evenly_limit(accepted, MAX_WINDOWS_PER_SUBJECT)
    if len(accepted) < 5:
        raise ValueError(f"Only {len(accepted)} valid four-second windows")
    return np.stack(accepted), {
        **header,
        "candidate_windows": int(sum(rejected.values()) + len(accepted)),
        "retained_windows": len(accepted),
        "rejected_windows": dict(rejected),
    }


def main() -> None:
    mne.set_log_level("ERROR")
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((SOURCE_PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = source_manifest["participants"]
    audits = []

    for subject, row in enumerate(tqdm(participants, desc="Preparing EEGPT raw windows")):
        part_path = PARTS / f"subject_{subject:03d}.npy"
        audit_path = PARTS / f"subject_{subject:03d}.json"
        if part_path.exists() and audit_path.exists():
            audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
            continue
        audit = {"subject_index": subject, "subject_id": row["subject_id"], "status": "error"}
        try:
            windows, details = prepare_subject(row)
            np.save(part_path, windows)
            audit.update(details)
            audit["status"] = "included"
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        audits.append(audit)

    failures = [item for item in audits if item["status"] != "included"]
    if failures:
        raise RuntimeError(f"Raw preparation failed for {len(failures)} participants: {failures[:3]}")

    arrays, labels, subject_indexes = [], [], []
    updated_participants = []
    for subject, row in enumerate(participants):
        values = np.load(PARTS / f"subject_{subject:03d}.npy")
        arrays.append(values)
        labels.append(np.full(len(values), source.LABEL_TO_ID[row["label"]], dtype=np.int64))
        subject_indexes.append(np.full(len(values), subject, dtype=np.int32))
        updated_participants.append({**row, "windows": len(values)})

    x = np.concatenate(arrays)
    y = np.concatenate(labels)
    subject_index = np.concatenate(subject_indexes)
    np.save(OUT / "raw_windows.npy", x)
    np.save(OUT / "labels.npy", y)
    np.save(OUT / "subject_index.npy", subject_index)
    manifest = {
        "created_unix": time.time(),
        "source_manifest": str(SOURCE_PREPARED / "manifest.json"),
        "channels": source.TARGET_CHANNELS,
        "classes": source.CLASS_NAMES,
        "input_sampling_rate": INPUT_FS,
        "sampling_rate": OUTPUT_FS,
        "window_seconds": WINDOW_SECONDS,
        "units": "microvolts",
        "tensor_shape": list(x.shape),
        "participants": updated_participants,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"Saved {len(x)} windows from {len(participants)} participants: {x.shape}", flush=True)


if __name__ == "__main__":
    main()
