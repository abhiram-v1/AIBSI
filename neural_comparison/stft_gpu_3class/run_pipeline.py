from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
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
import torch
import torch.nn as nn
from scipy.signal import stft
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
PREPARED = OUT / "prepared"
RUN_DIR = OUT / "run"

TARGET_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
CLASS_NAMES = ["AD", "FTD", "HC"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
LEGACY_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

FS = 128
WINDOW_SECONDS = 8
WINDOW_SAMPLES = FS * WINDOW_SECONDS
STFT_SECONDS = 2
STFT_SAMPLES = FS * STFT_SECONDS
STFT_HOP_SECONDS = 0.5
STFT_HOP = int(FS * STFT_HOP_SECONDS)
MAX_WINDOWS_PER_SUBJECT = 40
EDGE_SECONDS = 10
SEED = 20261007


@dataclass
class Record:
    subject_id: str
    source: str
    site: str
    label: str
    path: str


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_records() -> list[Record]:
    records: list[Record] = []
    participants = pd.read_csv(ROOT / "dataset" / "ds004504" / "participants.tsv", sep="\t")
    ds_label = {"A": "AD", "F": "FTD", "C": "HC"}
    for row in participants.itertuples(index=False):
        subject = str(row.participant_id)
        path = (
            ROOT / "dataset" / "ds004504" / "derivatives" / subject / "eeg"
            / f"{subject}_task-eyesclosed_eeg.set"
        )
        if path.exists():
            records.append(Record(f"ds004504:{subject}", "ds004504", "AHEPA", ds_label[row.Group], str(path)))

    brainlat_root = ROOT / "dataset" / "brainlat" / "EEG"
    groups = {"1_AD": "AD", "2_bvFTD": "FTD", "5_HC": "HC"}
    for folder, label in groups.items():
        for path in sorted((brainlat_root / folder).rglob("*.set")):
            subject = path.parent.parent.name
            site = path.relative_to(brainlat_root / folder).parts[0]
            records.append(Record(f"brainlat:{site}:{subject}", "brainlat", site, label, str(path)))

    ids = [r.subject_id for r in records]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate participant recordings detected: {duplicates[:10]}")
    return records


def target_montage() -> mne.channels.DigMontage:
    standard = mne.channels.make_standard_montage("standard_1020")
    positions = standard.get_positions()
    return mne.channels.make_dig_montage(
        ch_pos={name: positions["ch_pos"][name] for name in TARGET_CHANNELS},
        coord_frame=positions["coord_frame"],
    )


def load_harmonized(record: Record) -> tuple[np.ndarray, dict[str, Any]]:
    raw = mne.io.read_raw_eeglab(record.path, preload=False, verbose="ERROR")
    original = {
        "original_sfreq": float(raw.info["sfreq"]),
        "original_channels": int(raw.info["nchan"]),
        "original_seconds": float(raw.n_times / raw.info["sfreq"]),
    }
    if record.source == "brainlat":
        if raw.info["nchan"] != 128 or set(raw.ch_names) != set(mne.channels.make_standard_montage("biosemi128").ch_names):
            raise ValueError(f"Unexpected BrainLat montage: {raw.info['nchan']} channels")
        # Some BrainLat cohorts store the signal in a companion .fdt file, so
        # EEGLAB initially returns a lazy Raw object. Spatial interpolation
        # requires the samples to be resident in memory for both storage forms.
        raw.load_data(verbose="ERROR")
        raw.set_montage(mne.channels.make_standard_montage("biosemi128"), on_missing="raise")
        raw = raw.interpolate_to(target_montage(), method="spline")
    else:
        rename = {old: new for old, new in LEGACY_RENAME.items() if old in raw.ch_names}
        if rename:
            raw.rename_channels(rename)
        missing = sorted(set(TARGET_CHANNELS) - set(raw.ch_names))
        if missing:
            raise ValueError(f"Missing ds004504 channels: {missing}")
        raw.pick(TARGET_CHANNELS)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")
        raw.load_data(verbose="ERROR")

    raw.reorder_channels(TARGET_CHANNELS)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(1.0, 40.0, picks="eeg", verbose="ERROR")
    raw.resample(FS, npad="auto", verbose="ERROR")
    data = raw.get_data().astype(np.float32, copy=False)
    original["harmonized_seconds"] = float(data.shape[1] / FS)
    return data, original


def valid_window(window: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(window).all():
        return False, "nonfinite"
    sd_uv = window.std(axis=1) * 1e6
    if np.any(sd_uv < 0.1):
        return False, "flat"
    if np.any(sd_uv > 200.0):
        return False, "high_variance"
    if np.max(np.abs(window)) * 1e6 > 500.0:
        return False, "extreme_amplitude"
    return True, "accepted"


def window_to_stft(window: np.ndarray) -> np.ndarray:
    frequencies, _, spectrum = stft(
        window,
        fs=FS,
        window="hann",
        nperseg=STFT_SAMPLES,
        noverlap=STFT_SAMPLES - STFT_HOP,
        nfft=STFT_SAMPLES,
        detrend="constant",
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )
    keep = (frequencies >= 1.0) & (frequencies <= 40.0)
    log_power = np.log10(np.abs(spectrum[:, keep, :]) ** 2 + 1e-20).astype(np.float32)
    # Remove only the window-wide gain. Relative spectral and scalp patterns remain.
    log_power -= np.median(log_power)
    return log_power


def evenly_limit(items: list[np.ndarray], limit: int) -> list[np.ndarray]:
    if len(items) <= limit:
        return items
    indexes = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in indexes]


def prepare() -> None:
    mne.set_log_level("ERROR")
    seed_everything()
    PREPARED.mkdir(parents=True, exist_ok=True)
    records = discover_records()
    print("Discovered participants:", Counter((r.source, r.label) for r in records))

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    participants: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Preparing participants"):
        audit: dict[str, Any] = {**asdict(record), "status": "error"}
        try:
            data, header = load_harmonized(record)
            audit.update(header)
            start = EDGE_SECONDS * FS
            stop = data.shape[1] - EDGE_SECONDS * FS
            candidates: list[np.ndarray] = []
            rejected = Counter()
            for offset in range(start, stop - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                window = data[:, offset : offset + WINDOW_SAMPLES]
                ok, reason = valid_window(window)
                if ok:
                    candidates.append(window_to_stft(window))
                else:
                    rejected[reason] += 1
            selected = evenly_limit(candidates, MAX_WINDOWS_PER_SUBJECT)
            if len(selected) < 5:
                raise ValueError(f"Only {len(selected)} valid windows")
            subject_index = len(participants)
            x_subject = np.stack(selected).astype(np.float16)
            x_parts.append(x_subject)
            y_parts.append(np.full(len(selected), LABEL_TO_ID[record.label], dtype=np.int64))
            subject_parts.append(np.full(len(selected), subject_index, dtype=np.int32))
            participants.append(
                {
                    "subject_index": subject_index,
                    **asdict(record),
                    "windows": len(selected),
                }
            )
            audit.update(
                {
                    "status": "included",
                    "candidate_windows": len(candidates),
                    "retained_windows": len(selected),
                    "rejected_windows": dict(rejected),
                }
            )
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
        audits.append(audit)

    if not x_parts:
        raise RuntimeError("No EEG windows were prepared")
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    subject_index = np.concatenate(subject_parts)
    np.save(PREPARED / "spectrograms.npy", x)
    np.save(PREPARED / "labels.npy", y)
    np.save(PREPARED / "subject_index.npy", subject_index)
    manifest = {
        "created_unix": time.time(),
        "class_names": CLASS_NAMES,
        "channels": TARGET_CHANNELS,
        "sampling_rate": FS,
        "window_seconds": WINDOW_SECONDS,
        "stft_seconds": STFT_SECONDS,
        "stft_hop_seconds": STFT_HOP_SECONDS,
        "frequency_min": 1.0,
        "frequency_max": 40.0,
        "tensor_shape": list(x.shape),
        "participants": participants,
    }
    (PREPARED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PREPARED / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    plot_example(x, y)

    included = Counter((p["source"], p["label"]) for p in participants)
    failures = [a for a in audits if a["status"] != "included"]
    print(f"Prepared {len(x)} windows from {len(participants)} participants; shape={x.shape}")
    print("Included:", included)
    print(f"Excluded participants: {len(failures)}")
    for failure in failures[:20]:
        print("  ", failure["subject_id"], failure.get("error"))


def plot_example(x: np.ndarray, y: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for class_id, ax in enumerate(axes):
        item = x[np.flatnonzero(y == class_id)[0]].astype(np.float32).mean(axis=0)
        image = ax.imshow(
            item,
            aspect="auto",
            origin="lower",
            extent=[0, WINDOW_SECONDS - STFT_SECONDS, 1, 40],
            cmap="magma",
        )
        ax.set_title(f"{CLASS_NAMES[class_id]}: mean over 19 channels")
        ax.set_xlabel("Time within 8 s window (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(image, ax=ax, label="Centered log power")
    fig.savefig(PREPARED / "example_stft.png", dpi=180)
    plt.close(fig)


def make_subject_splits(participants: list[dict[str, Any]]) -> dict[str, list[int]]:
    ids = np.arange(len(participants))
    strata = np.array([f"{p['source']}:{p['label']}" for p in participants])
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train, remaining = next(outer.split(ids, strata))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1)
    val_rel, test_rel = next(inner.split(remaining, strata[remaining]))
    return {
        "train": ids[train].tolist(),
        "val": remaining[val_rel].tolist(),
        "test": remaining[test_rel].tolist(),
    }


def window_indexes(subject_index: np.ndarray, subjects: list[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))


def fit_scaler(x: np.ndarray, indexes: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    total = None
    total_sq = None
    count = 0
    for start in tqdm(range(0, len(indexes), batch_size), desc="Fitting train-only scaler"):
        block = x[indexes[start : start + batch_size]].astype(np.float64)
        block_sum = block.sum(axis=(0, 3), keepdims=True)
        block_sq = np.square(block).sum(axis=(0, 3), keepdims=True)
        total = block_sum if total is None else total + block_sum
        total_sq = block_sq if total_sq is None else total_sq + block_sq
        count += block.shape[0] * block.shape[3]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indexes: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.x = x
        self.y = y
        self.indexes = indexes
        self.mean = mean[0]
        self.std = std[0]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indexes[item])
        value = self.x[index].astype(np.float32)
        value = np.clip((value - self.mean) / self.std, -8.0, 8.0)
        return torch.from_numpy(value), torch.tensor(int(self.y[index]), dtype=torch.long), index


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, pool: bool) -> None:
        super().__init__()
        stride = 2 if pool else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(self.norm2(self.conv2(x)))
        return self.activation(x + residual)


class STFTCNN(nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(len(TARGET_CHANNELS), 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.features = nn.Sequential(
            ResidualBlock(64, 64, 0.08, False),
            ResidualBlock(64, 96, 0.12, True),
            ResidualBlock(96, 160, 0.16, True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(160, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(self.stem(x)))


@torch.no_grad()
def predict_windows(
    model: nn.Module, loader: DataLoader, device: torch.device, total_windows: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = np.full((total_windows, len(CLASS_NAMES)), np.nan, dtype=np.float32)
    labels = np.full(total_windows, -1, dtype=np.int64)
    for values, target, indexes in loader:
        values = values.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(values)
        idx = indexes.numpy()
        probabilities[idx] = torch.softmax(logits, dim=1).cpu().numpy()
        labels[idx] = target.numpy()
    return probabilities, labels


def aggregate_subjects(
    probabilities: np.ndarray,
    labels: np.ndarray,
    subject_index: np.ndarray,
    indexes: np.ndarray,
    participants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for subject in sorted(set(subject_index[indexes].tolist())):
        use = indexes[subject_index[indexes] == subject]
        mean_probability = probabilities[use].mean(axis=0)
        true = int(labels[use[0]])
        predicted = int(np.argmax(mean_probability))
        rows.append(
            {
                **participants[subject],
                "true_id": true,
                "true_label": CLASS_NAMES[true],
                "predicted_id": predicted,
                "predicted_label": CLASS_NAMES[predicted],
                "prob_AD": float(mean_probability[0]),
                "prob_FTD": float(mean_probability[1]),
                "prob_HC": float(mean_probability[2]),
                "correct": bool(true == predicted),
            }
        )
    return rows


def subject_score(rows: list[dict[str, Any]]) -> float:
    true = [row["true_id"] for row in rows]
    pred = [row["predicted_id"] for row in rows]
    return float(f1_score(true, pred, average="macro", zero_division=0))


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true = np.asarray([row["true_id"] for row in rows])
    pred = np.asarray([row["predicted_id"] for row in rows])
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    return {
        "subjects": int(len(rows)),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(true, pred, labels=np.arange(len(CLASS_NAMES))).tolist(),
    }


def train(epochs: int, batch_size: int, learning_rate: float) -> None:
    seed_everything()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this experiment is configured for GPU training")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    x = np.load(PREPARED / "spectrograms.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = make_subject_splits(participants)
    (RUN_DIR / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    indexes = {name: window_indexes(subject_index, subjects) for name, subjects in splits.items()}
    mean, std = fit_scaler(x, indexes["train"])
    np.savez_compressed(RUN_DIR / "train_scaler.npz", mean=mean, std=std)

    datasets = {name: SpectrogramDataset(x, y, idx, mean, std) for name, idx in indexes.items()}
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=True, drop_last=False,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True),
    }

    device = torch.device("cuda")
    model = STFTCNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_subject_labels = [LABEL_TO_ID[participants[i]["label"]] for i in splits["train"]]
    counts = np.bincount(train_subject_labels, minlength=len(CLASS_NAMES))
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate / 30)
    scaler = torch.amp.GradScaler("cuda")

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    patience = 8
    epochs_without_improvement = 0
    best_path = RUN_DIR / "best_model.pt"

    print(f"Training on {torch.cuda.get_device_name(0)}; model parameters={parameter_count:,}")
    print("Participant split sizes:", {key: len(value) for key, value in splits.items()})
    print("Window split sizes:", {key: len(value) for key, value in indexes.items()})
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for values, target, _ in tqdm(loaders["train"], desc=f"Epoch {epoch:02d}", leave=False):
            values = values.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(values)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(values)
            seen += len(values)
        scheduler.step()

        val_probabilities, val_labels = predict_windows(model, loaders["val"], device, len(x))
        val_rows = aggregate_subjects(
            val_probabilities, val_labels, subject_index, indexes["val"], participants
        )
        val_f1 = subject_score(val_rows)
        item = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_macro_f1": val_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(item)
        print(
            f"epoch={epoch:02d} loss={item['train_loss']:.4f} "
            f"val_subject_macro_f1={val_f1:.4f} lr={item['learning_rate']:.2e}"
        )
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "parameter_count": parameter_count,
                    "channels": TARGET_CHANNELS,
                    "class_names": CLASS_NAMES,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch={best_epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    predictions: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "split_participants": {key: len(value) for key, value in splits.items()},
        "split_windows": {key: len(value) for key, value in indexes.items()},
    }
    for split in ["val", "test"]:
        probabilities, labels = predict_windows(model, loaders[split], device, len(x))
        rows = aggregate_subjects(probabilities, labels, subject_index, indexes[split], participants)
        predictions[split] = rows
        metrics[split] = metrics_from_rows(rows)
        if split == "test":
            for source in ["ds004504", "brainlat"]:
                source_rows = [row for row in rows if row["source"] == source]
                if source_rows:
                    metrics[f"test_{source}"] = metrics_from_rows(source_rows)

    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RUN_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (RUN_DIR / "subject_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = predictions["test"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(history, metrics["test"])
    print(json.dumps(metrics["test"], indent=2))
    print(f"Saved run artifacts to {RUN_DIR}")


def plot_results(history: list[dict[str, Any]], test_metrics: dict[str, Any]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", markersize=3)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Weighted cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], marker="o", markersize=3)
    axes[1].set(title="Validation participant macro F1", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    axes[1].grid(alpha=0.25)
    fig.savefig(RUN_DIR / "training_curves.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Participant-level test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(RUN_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    measures = ["precision", "recall", "f1"]
    x_positions = np.arange(len(CLASS_NAMES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for offset, measure in enumerate(measures):
        values = [test_metrics["per_class"][name][measure] for name in CLASS_NAMES]
        ax.bar(x_positions + (offset - 1) * width, values, width, label=measure.title())
    ax.set_xticks(x_positions, CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Participant-level test metrics by class")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(RUN_DIR / "per_class_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the three-class STFT EEG CNN")
    parser.add_argument("command", choices=["prepare", "train", "all"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.command in {"prepare", "all"}:
        prepare()
    if args.command in {"train", "all"}:
        train(args.epochs, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
