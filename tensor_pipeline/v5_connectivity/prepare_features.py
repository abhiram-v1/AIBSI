"""Build label-free subject bags with filter-bank covariance/connectivity features.

Earlier experiment artifacts are read only.  Every retained V3 window is used once
per requested bag partition.  Bags never cross participants.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tensor_pipeline/outputs/v3_waveform_full_cohort"
OUT = ROOT / "tensor_pipeline/outputs/v5_connectivity"
BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
BAG_COUNTS = (1, 4, 8)
CHANNELS = 19
FS = 250


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def upper(a: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(a.shape[-1], 1)
    return a[..., i, j]


def normalized_covariance(data: np.ndarray) -> np.ndarray:
    """Trace-normalized SPD covariance with light fixed shrinkage."""
    data = data - data.mean(axis=1, keepdims=True)
    cov = data @ data.T / max(data.shape[1] - 1, 1)
    scale = max(float(np.trace(cov)) / len(cov), 1e-12)
    cov = cov / scale
    return 0.95 * cov + 0.05 * np.eye(len(cov))


def band_features(filtered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return SPD covariance, four connectivity edge sets and power summaries.

    filtered has shape windows x channels x samples. Hilbert transforms stay
    inside each four-second window, avoiding artificial phase continuity across
    window boundaries.
    """
    n = filtered.shape[0]
    flat = filtered.transpose(1, 0, 2).reshape(CHANNELS, -1).astype(np.float64)
    cov = normalized_covariance(flat)
    corr = np.corrcoef(flat)

    analytic = hilbert(filtered, axis=-1)
    envelope = np.abs(analytic).transpose(1, 0, 2).reshape(CHANNELS, -1)
    aec = np.corrcoef(envelope)
    phase = analytic / np.maximum(np.abs(analytic), 1e-12)
    phase = phase.transpose(1, 0, 2).reshape(CHANNELS, -1)
    phase_cross = phase @ phase.conj().T / phase.shape[1]
    im_phase = np.abs(np.imag(phase_cross))

    # Weighted phase-lag index from instantaneous analytic cross-products.
    wpli = np.zeros((CHANNELS, CHANNELS), dtype=np.float64)
    z = analytic.transpose(1, 0, 2).reshape(CHANNELS, -1)
    for i in range(CHANNELS):
        im = np.imag(z[i, None, :] * z[i + 1 :, :].conj())
        val = np.abs(im.mean(axis=1)) / np.maximum(np.abs(im).mean(axis=1), 1e-12)
        wpli[i, i + 1 :] = val
        wpli[i + 1 :, i] = val

    for matrix in (corr, aec, im_phase, wpli):
        np.fill_diagonal(matrix, 0.0)
    connectivity = np.concatenate([upper(corr), upper(aec), upper(im_phase), upper(wpli)])

    variance = np.maximum(np.var(filtered, axis=-1), 1e-12)
    power = np.concatenate([np.log10(variance).mean(axis=0), np.log10(variance).std(axis=0)])
    assert power.shape == (2 * CHANNELS,)
    return cov, connectivity, power


def load_windows() -> tuple[dict[str, list[np.ndarray]], dict[str, int], dict]:
    sources = {}
    metadata = {}
    for origin in ("primary", "leftover"):
        base = ROOT / "preprocessing/02_balanced_primary" / origin
        sources[origin] = np.load(base / "signals.npy", mmap_mode="r")
        metadata[origin] = dict(np.load(base / "metadata.npz"))
    retained = json.loads((SOURCE / "retained_windows.json").read_text(encoding="utf-8"))
    by_subject: dict[str, list[tuple[float, np.ndarray]]] = {}
    labels: dict[str, int] = {}
    for row in retained:
        origin, index = row["origin"], int(row["source_index"])
        meta = metadata[origin]
        assert str(meta["sample_id"][index]) == row["sample_id"]
        subject, label = str(row["subject"]), int(row["label"])
        labels.setdefault(subject, label)
        assert labels[subject] == label
        by_subject.setdefault(subject, []).append(
            (float(row["start_seconds"]), np.asarray(sources[origin][index], dtype=np.float32))
        )
    ordered = {s: [x for _, x in sorted(rows, key=lambda item: item[0])] for s, rows in by_subject.items()}
    return ordered, labels, {"retained": retained}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target_paths = [OUT / f"bags_{n}.npz" for n in BAG_COUNTS]
    if any(p.exists() for p in target_paths):
        raise FileExistsError("Preserving existing V5 feature artifacts; remove only in a fresh project copy.")
    windows, labels, provenance = load_windows()
    assert len(windows) == 88 and Counter(labels.values()) == Counter({0: 36, 1: 23, 2: 29})
    filters = [butter(4, band, btype="bandpass", fs=FS, output="sos") for band in BANDS]
    stores = {n: {k: [] for k in ("subject", "label", "bag", "windows", "covariance", "connectivity", "power")}
              for n in BAG_COUNTS}

    for position, subject in enumerate(sorted(windows)):
        x = np.stack(windows[subject]).astype(np.float64)
        x -= x.mean(axis=-1, keepdims=True)
        filtered = [sosfiltfilt(sos, x, axis=-1) for sos in filters]
        for bag_count in BAG_COUNTS:
            partitions = np.array_split(np.arange(len(x)), bag_count)
            assert all(len(part) for part in partitions)
            for bag_id, part in enumerate(partitions):
                covs, conns, powers = [], [], []
                for band_data in filtered:
                    cov, conn, power = band_features(band_data[part])
                    covs.append(cov); conns.append(conn); powers.append(power)
                store = stores[bag_count]
                store["subject"].append(subject)
                store["label"].append(labels[subject])
                store["bag"].append(bag_id)
                store["windows"].append(len(part))
                store["covariance"].append(covs)
                store["connectivity"].append(np.concatenate(conns))
                # Band power plus within-bag relative log-power contrasts.
                p = np.stack(powers)
                rel = p[:, :CHANNELS] - np.log10(
                    np.maximum(np.sum(10 ** p[:, :CHANNELS], axis=0, keepdims=True), 1e-12)
                )
                store["power"].append(np.concatenate([p.ravel(), rel.ravel()]))
        print(f"Prepared {position + 1:02d}/88 {subject}: {len(x)} windows", flush=True)

    shapes = {}
    for bag_count, store in stores.items():
        arrays = {k: np.asarray(v) for k, v in store.items()}
        assert np.isfinite(arrays["covariance"]).all()
        assert np.isfinite(arrays["connectivity"]).all()
        assert np.isfinite(arrays["power"]).all()
        np.savez_compressed(OUT / f"bags_{bag_count}.npz", **arrays)
        shapes[str(bag_count)] = {k: list(v.shape) for k, v in arrays.items()}
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_fingerprint": sha256(SOURCE / "retained_windows.json"),
        "retained_windows": len(provenance["retained"]),
        "subjects": len(windows),
        "class_counts": dict(Counter("AD" if v == 0 else "FTD" if v == 1 else "CN" for v in labels.values())),
        "sampling_hz": FS,
        "bands_hz": BANDS,
        "bag_counts": BAG_COUNTS,
        "features": "Per-band shrinkage covariance; correlation, amplitude-envelope correlation, imaginary phase-locking component and wPLI edges; log and relative band power.",
        "label_use": "Labels are copied to bags but never used to construct signal features.",
        "shapes": shapes,
    }
    (OUT / "feature_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
