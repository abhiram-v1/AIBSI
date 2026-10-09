"""Conventional subject-level EEG features directly from derivative .set files.

No tensor decomposition or tensor-pipeline artifacts are read. Each 4-second
window contributes only to its own participant's spectral and time summaries.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/ds004504"
OUT = ROOT / "conventional_ml/outputs/binary_eeg"
FS = 500
WINDOW = 4 * FS
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45.01))
REGIONS = (
    ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"),
    ("T3", "T4", "T5", "T6"),
    ("C3", "C4", "Cz"),
    ("P3", "P4", "Pz"),
    ("O1", "O2"),
)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_participants():
    with (DATA / "participants.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 88
    return sorted(rows, key=lambda r: r["participant_id"])


def boundary_samples(events):
    result = []
    for event in np.atleast_1d(events):
        if str(getattr(event, "type", "")).lower() == "boundary":
            result.append(float(event.latency) - 1.0)
    return np.asarray(result)


def window_features(windows):
    # Shape is clean windows x 19 channels x 2000 samples.
    freq, psd = welch(windows, fs=FS, nperseg=1000, noverlap=500, axis=-1)
    keep = (freq >= 1) & (freq <= 45)
    spectrum = psd[..., keep]
    powers = np.stack([psd[..., (freq >= lo) & (freq < hi)].sum(axis=-1) * .5
                       for lo, hi in BANDS], axis=-1)
    total = np.maximum(powers.sum(axis=-1, keepdims=True), 1e-12)
    relative = powers / total
    distribution = spectrum / np.maximum(spectrum.sum(axis=-1, keepdims=True), 1e-12)
    entropy = -(distribution * np.log(np.maximum(distribution, 1e-12))).sum(axis=-1)
    entropy /= np.log(distribution.shape[-1])
    alpha_mask = (freq >= 7) & (freq <= 13)
    alpha_peak = freq[alpha_mask][psd[..., alpha_mask].argmax(axis=-1)]
    ratios = np.stack([np.log10(np.maximum(powers[..., 1], 1e-12) /
                               np.maximum(powers[..., 2], 1e-12)),
                       np.log10(np.maximum(powers[..., 0], 1e-12) /
                                np.maximum(powers[..., 2], 1e-12))], axis=-1)
    diff = np.diff(windows, axis=-1)
    second_diff = np.diff(diff, axis=-1)
    variance = np.maximum(windows.var(axis=-1), 1e-12)
    diff_var = np.maximum(diff.var(axis=-1), 1e-12)
    mobility = np.sqrt(diff_var / variance) * FS
    complexity = np.sqrt(np.maximum(second_diff.var(axis=-1), 1e-12) / diff_var) * FS
    complexity /= np.maximum(mobility, 1e-12)
    line_length = np.log10(np.maximum(np.abs(diff).mean(axis=-1), 1e-12))
    crossing = (windows[..., 1:] * windows[..., :-1] < 0).mean(axis=-1)
    features = np.concatenate([
        np.log10(np.maximum(powers, 1e-12)), relative,
        entropy[..., None], alpha_peak[..., None], ratios,
        .5 * np.log10(variance)[..., None], mobility[..., None],
        complexity[..., None], line_length[..., None], crossing[..., None]], axis=-1)
    assert features.shape[1:] == (19, 19)
    return features


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserving existing output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_participants()
    expected_channels = None
    subjects, groups, counts, regional_rows, global_rows, channel_rows, audit = [], [], [], [], [], [], []
    source_hashes = {}
    for number, row in enumerate(rows, 1):
        subject = row["participant_id"]
        path = DATA / "derivatives" / subject / "eeg" / f"{subject}_task-eyesclosed_eeg.set"
        source_hashes[subject] = file_hash(path)
        d = loadmat(path, squeeze_me=True, struct_as_record=False,
                    variable_names=["data", "srate", "chanlocs", "event", "pnts"])
        assert int(d["srate"]) == FS
        signal = np.asarray(d["data"], dtype=np.float32)
        assert signal.shape == (19, int(d["pnts"]))
        channels = [str(c.labels) for c in np.atleast_1d(d["chanlocs"])]
        expected_channels = channels if expected_channels is None else expected_channels
        assert channels == expected_channels
        n = signal.shape[1] // WINDOW
        starts = np.arange(n) * WINDOW
        boundaries = boundary_samples(d.get("event", []))
        safe = np.ones(n, dtype=bool)
        for boundary in boundaries:
            safe &= ~((starts < boundary + FS // 2) & (starts + WINDOW > boundary - FS // 2))
        windows = signal[:, :n * WINDOW].reshape(19, n, WINDOW).transpose(1, 0, 2).copy()
        windows -= windows.mean(axis=1, keepdims=True)  # common-average reference
        windows -= windows.mean(axis=2, keepdims=True)  # channel mean removal
        finite = np.isfinite(windows).all(axis=(1, 2))
        amplitude = np.max(np.abs(windows), axis=(1, 2)) < 400
        step = np.max(np.abs(np.diff(windows, axis=-1)), axis=(1, 2)) < 150
        active = windows.std(axis=-1).min(axis=1) >= 1
        good = safe & finite & amplitude & step & active
        kept = windows[good]
        assert len(kept) >= 20, subject
        features = window_features(kept)
        mean = features.mean(axis=0)
        variation = features.std(axis=0)
        channels_summary = np.concatenate([mean.ravel(), variation.ravel()])
        regional = np.concatenate([
            np.stack([mean[[channels.index(c) for c in region]].mean(axis=0)
                      for region in REGIONS]).ravel(),
            np.stack([variation[[channels.index(c) for c in region]].mean(axis=0)
                      for region in REGIONS]).ravel()])
        global_features = np.concatenate([mean.mean(axis=0), mean.std(axis=0),
                                          variation.mean(axis=0)])
        assert np.isfinite(channels_summary).all() and np.isfinite(regional).all()
        subjects.append(subject)
        groups.append(row["Group"])
        counts.append(len(kept))
        channel_rows.append(channels_summary)
        regional_rows.append(regional)
        global_rows.append(global_features)
        audit.append({"subject": subject, "group": row["Group"], "complete_windows": n,
                      "boundary_excluded": int((~safe).sum()),
                      "quality_excluded": int((safe & ~good).sum()),
                      "retained_windows": int(good.sum())})
        print(f"{number}/88 {subject}: {int(good.sum())}/{n} windows", flush=True)
    assert sorted(groups).count("A") == 36 and groups.count("F") == 23 and groups.count("C") == 29
    np.savez_compressed(OUT / "subject_features.npz", subjects=np.array(subjects),
                        groups=np.array(groups), counts=np.array(counts),
                        channels=np.array(expected_channels),
                        channel_features=np.asarray(channel_rows),
                        regional_features=np.asarray(regional_rows),
                        global_features=np.asarray(global_rows))
    report = {"source": "Derivative EEGLAB .set files, not tensor-pipeline outputs",
              "subjects": len(subjects), "groups": {g: groups.count(g) for g in ("A", "F", "C")},
              "retained_windows": int(sum(counts)), "features_per_channel": 19,
              "feature_shapes": {"channel": [88, 722], "regional": [88, 190], "global": [88, 57]},
              "qc": "4 s non-overlap; 0.5 s boundary guard; CAR; nonfinite, 400 uV amplitude, 150 uV/sample step, 1 uV channel SD thresholds",
              "audit": audit, "source_sha256": source_hashes,
              "code_sha256": file_hash(Path(__file__))}
    (OUT / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(subjects)} subjects and {sum(counts)} windows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
