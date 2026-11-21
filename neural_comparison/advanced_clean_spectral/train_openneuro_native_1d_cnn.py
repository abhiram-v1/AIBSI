from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "ds004504"
OUT = Path(__file__).resolve().parent / "outputs" / "openneuro_native_500hz_1d_cnn"
PREPARED = OUT / "prepared"
CLASSES = ["AD", "FTD", "HC"]
GROUP_TO_ID = {"A": 0, "F": 1, "C": 2}
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
FS = 500
WINDOW_SECONDS = 8
WINDOW_SAMPLES = FS * WINDOW_SECONDS
MAX_WINDOWS = 40
SEEDS = [20261124, 20261125, 20261126]
BATCH_SIZE = 64
MAX_EPOCHS = 45
PATIENCE = 8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def prepare() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    PREPARED.mkdir(parents=True, exist_ok=True)
    manifest_path = PREPARED / "manifest.json"
    waveform_path = PREPARED / "waveforms.npy"
    if manifest_path.exists() and waveform_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        count = int(manifest["accepted_windows"])
        return (
            np.load(waveform_path, mmap_mode="r")[:count],
            np.load(PREPARED / "labels.npy", mmap_mode="r")[:count],
            np.load(PREPARED / "subject_index.npy", mmap_mode="r")[:count],
            manifest["participants"],
        )

    table = pd.read_csv(DATASET / "participants.tsv", sep="\t")
    participants = []
    for _, row in table.iterrows():
        subject = str(row["participant_id"])
        participants.append({"subject_id": subject, "label": CLASSES[GROUP_TO_ID[str(row["Group"])]]})
    capacity = len(participants) * MAX_WINDOWS
    waveforms = np.lib.format.open_memmap(
        waveform_path, mode="w+", dtype=np.float32, shape=(capacity, len(CHANNELS), WINDOW_SAMPLES)
    )
    labels = np.lib.format.open_memmap(PREPARED / "labels.npy", mode="w+", dtype=np.int64, shape=(capacity,))
    owners = np.lib.format.open_memmap(PREPARED / "subject_index.npy", mode="w+", dtype=np.int64, shape=(capacity,))
    cursor = 0
    for subject_index, participant in enumerate(participants):
        subject = participant["subject_id"]
        path = DATASET / "derivatives" / subject / "eeg" / f"{subject}_task-eyesclosed_eeg.set"
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        if float(raw.info["sfreq"]) != FS:
            raise RuntimeError(f"Unexpected sampling rate for {subject}: {raw.info['sfreq']}")
        if set(raw.ch_names) != set(CHANNELS):
            raise RuntimeError(f"Unexpected montage for {subject}: {raw.ch_names}")
        raw.reorder_channels(CHANNELS)
        signal = raw.get_data().astype(np.float32, copy=False)
        starts = np.arange(0, signal.shape[1] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES)
        accepted = []
        for start in starts:
            window_uv = signal[:, start : start + WINDOW_SAMPLES] * 1e6
            channel_sd = window_uv.std(axis=1)
            if np.isfinite(window_uv).all() and np.all((channel_sd >= 0.1) & (channel_sd <= 200.0)) and np.max(np.abs(window_uv)) <= 500.0:
                accepted.append(int(start))
        if len(accepted) > MAX_WINDOWS:
            positions = np.linspace(0, len(accepted) - 1, MAX_WINDOWS).round().astype(int)
            accepted = [accepted[position] for position in positions]
        label = GROUP_TO_ID[table.iloc[subject_index]["Group"]]
        for start in accepted:
            waveforms[cursor] = signal[:, start : start + WINDOW_SAMPLES]
            labels[cursor] = label
            owners[cursor] = subject_index
            cursor += 1
        participant["accepted_windows"] = len(accepted)
        participant["native_sampling_hz"] = FS
        print(f"prepared {subject_index + 1}/88 {subject}: {len(accepted)} windows", flush=True)
        del raw, signal
    waveforms.flush(); labels.flush(); owners.flush()
    manifest = {
        "participants": participants,
        "accepted_windows": cursor,
        "policy": {
            "source": "OpenNeuro ds004504 supplied artifact-cleaned derivatives only",
            "sampling_hz": FS,
            "electrodes": CHANNELS,
            "spatial_interpolation": False,
            "resampling": False,
            "additional_filtering": False,
            "rereferencing": False,
            "window_seconds": WINDOW_SECONDS,
            "overlap": False,
            "maximum_windows_per_participant": MAX_WINDOWS,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return waveforms[:cursor], labels[:cursor], owners[:cursor], participants


def split_subjects(subject_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = np.arange(len(subject_labels))
    first = StratifiedShuffleSplit(n_splits=1, test_size=14, random_state=20261124)
    development_position, test_position = next(first.split(subjects, subject_labels))
    development = subjects[development_position]
    test = subjects[test_position]
    second = StratifiedShuffleSplit(n_splits=1, test_size=14, random_state=20261125)
    train_position, val_position = next(second.split(development, subject_labels[development]))
    return development[train_position], development[val_position], test


def channel_scale(x: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    squares = np.zeros(19, dtype=np.float64)
    count = 0
    for start in range(0, len(indexes), 64):
        values = np.asarray(x[indexes[start : start + 64]], dtype=np.float32) * 1e6
        values -= values.mean(axis=2, keepdims=True)
        squares += np.square(values, dtype=np.float64).sum(axis=(0, 2))
        count += values.shape[0] * values.shape[2]
    return np.maximum(np.sqrt(squares / count), 1e-3).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, x, y, indexes, scale, augment):
        self.x, self.y = x, y
        self.indexes = np.asarray(indexes, dtype=np.int64)
        self.scale = scale[:, None]
        self.augment = augment

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        index = int(self.indexes[item])
        values = np.asarray(self.x[index], dtype=np.float32).copy() * 1e6
        values -= values.mean(axis=1, keepdims=True)
        values /= self.scale
        if self.augment:
            values *= np.float32(np.random.uniform(0.9, 1.1))
            values = np.roll(values, int(np.random.randint(-125, 126)), axis=1)
            if np.random.random() < 0.35:
                values[int(np.random.randint(0, 19))] = 0
            values += np.random.normal(0, 0.015, values.shape).astype(np.float32)
        return torch.from_numpy(values), torch.tensor(int(self.y[index]))


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels_in, channels_in, kernel, stride=stride, padding=kernel // 2, groups=channels_in, bias=False),
            nn.BatchNorm1d(channels_in), nn.GELU(),
            nn.Conv1d(channels_in, channels_out, 1, bias=False), nn.BatchNorm1d(channels_out), nn.GELU(), nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.net(x)


class Native500HzCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.filterbank = nn.Sequential(
            nn.Conv1d(19, 76, 251, padding=125, groups=19, bias=False),
            nn.BatchNorm1d(76), nn.GELU(),
            nn.Conv1d(76, 64, 1, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            nn.AvgPool1d(16), nn.Dropout(0.2),
        )
        self.encoder = nn.Sequential(Block(64, 96, 31, 2), Block(96, 128, 15, 2), Block(128, 128, 7, 2))
        self.classifier = nn.Sequential(nn.Linear(256, 96), nn.GELU(), nn.Dropout(0.4), nn.Linear(96, 3))

    def forward(self, x):
        values = self.encoder(self.filterbank(x))
        return self.classifier(torch.cat([values.mean(2), values.amax(2)], dim=1))


@torch.no_grad()
def predict(model, x, y, indexes, scale, device):
    loader = DataLoader(WindowDataset(x, y, indexes, scale, False), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    model.eval(); result = []
    for values, _ in loader:
        result.append(torch.softmax(model(values.to(device, non_blocking=True)), 1).cpu().numpy())
    return np.concatenate(result)


def aggregate(window_probability, indexes, owners, subjects):
    result = np.empty((len(subjects), 3), dtype=np.float32)
    window_owners = owners[indexes]
    for row, subject in enumerate(subjects):
        result[row] = window_probability[window_owners == subject].mean(0)
    return result


def metrics(true, probability):
    prediction = probability.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(true, prediction, labels=np.arange(3), zero_division=0)
    return {
        "subjects": len(true), "accuracy": float(accuracy_score(true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "macro_f1": float(f1_score(true, prediction, average="macro", zero_division=0)),
        "per_class": {CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i in range(3)},
        "confusion_matrix": confusion_matrix(true, prediction, labels=np.arange(3)).tolist(),
    }


def train_seed(seed, x, y, owners, subject_labels, train_subjects, val_subjects, device):
    seed_all(seed)
    train_indexes = np.flatnonzero(np.isin(owners, train_subjects))
    val_indexes = np.flatnonzero(np.isin(owners, val_subjects))
    scale = channel_scale(x, train_indexes)
    per_subject = np.bincount(owners[train_indexes], minlength=len(subject_labels))
    sample_weights = np.asarray([1 / per_subject[owners[index]] for index in train_indexes])
    loader = DataLoader(
        WindowDataset(x, y, train_indexes, scale, True), batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(sample_weights, len(train_indexes), replacement=True),
        num_workers=0, pin_memory=True, drop_last=True,
    )
    counts = np.bincount(subject_labels[train_subjects], minlength=3).astype(np.float32)
    class_weights = counts.sum() / (3 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    model = Native500HzCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None; best_score = (-math.inf, -math.inf); best_epoch = 0; stale = 0; history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); losses = []
        for values, labels in loader:
            values, labels = values.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(values), labels)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_probability = aggregate(predict(model, x, y, val_indexes, scale, device), val_indexes, owners, val_subjects)
        score_metrics = metrics(subject_labels[val_subjects], val_probability)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **score_metrics})
        score = (score_metrics["macro_f1"], score_metrics["balanced_accuracy"])
        print(f"seed={seed} epoch={epoch:02d} loss={np.mean(losses):.4f} val_f1={score[0]:.4f} val_bacc={score[1]:.4f}", flush=True)
        if score > best_score:
            best_score = score; best_epoch = epoch; stale = 0; best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, scale, best_epoch, history


def confusion_plot(matrix, path, title):
    fig, ax = plt.subplots(figsize=(5.4, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row][column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, owners, participants = prepare()
    subject_labels = np.asarray([CLASSES.index(row["label"]) for row in participants])
    train_subjects, val_subjects, test_subjects = split_subjects(subject_labels)
    split_payload = {name: {"indices": values.tolist(), "class_counts": {CLASSES[i]: int(np.sum(subject_labels[values] == i)) for i in range(3)}} for name, values in [("train", train_subjects), ("validation", val_subjects), ("test", test_subjects)]}
    (OUT / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_indexes = np.flatnonzero(np.isin(owners, test_subjects))
    val_indexes = np.flatnonzero(np.isin(owners, val_subjects))
    runs = []; test_probabilities = []; val_probabilities = []
    for seed in SEEDS:
        model, scale, epoch, history = train_seed(seed, x, y, owners, subject_labels, train_subjects, val_subjects, device)
        val_probability = aggregate(predict(model, x, y, val_indexes, scale, device), val_indexes, owners, val_subjects)
        test_probability = aggregate(predict(model, x, y, test_indexes, scale, device), test_indexes, owners, test_subjects)
        val_metric = metrics(subject_labels[val_subjects], val_probability)
        test_metric = metrics(subject_labels[test_subjects], test_probability)
        runs.append({"seed": seed, "best_epoch": epoch, "validation": val_metric, "test": test_metric, "history": history})
        val_probabilities.append(val_probability); test_probabilities.append(test_probability)
        torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "channel_scale": scale, "seed": seed, "best_epoch": epoch}, OUT / f"model_seed_{seed}.pt")
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    selected = max(range(len(runs)), key=lambda i: (runs[i]["validation"]["macro_f1"], runs[i]["validation"]["balanced_accuracy"]))
    ensemble_val = np.mean(val_probabilities, axis=0); ensemble_test = np.mean(test_probabilities, axis=0)
    payload = {
        "protocol": {"participants": 88, "classes": {CLASSES[i]: int(np.sum(subject_labels == i)) for i in range(3)}, "sampling_hz": FS, "electrodes": 19, "interpolation": False, "resampling": False, "train_participants": len(train_subjects), "validation_participants": len(val_subjects), "test_participants": len(test_subjects), "selection_uses_test": False},
        "selected_seed": runs[selected]["seed"], "selected_epoch": runs[selected]["best_epoch"],
        "selected_validation": runs[selected]["validation"], "selected_test": runs[selected]["test"],
        "three_seed_ensemble_validation": metrics(subject_labels[val_subjects], ensemble_val),
        "three_seed_ensemble_test": metrics(subject_labels[test_subjects], ensemble_test),
        "runs": runs,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "probabilities.npz", validation_indices=val_subjects, validation_true=subject_labels[val_subjects], validation_probabilities=np.asarray(val_probabilities), test_indices=test_subjects, test_true=subject_labels[test_subjects], test_probabilities=np.asarray(test_probabilities))
    confusion_plot(payload["selected_test"]["confusion_matrix"], OUT / "selected_test_confusion.png", "Native 500 Hz OpenNeuro CNN")
    labels = ["Selected validation", "Selected test", "Ensemble validation", "Ensemble test"]
    rows = [payload["selected_validation"], payload["selected_test"], payload["three_seed_ensemble_validation"], payload["three_seed_ensemble_test"]]
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True); positions = np.arange(4)
    for offset, measure in enumerate(["accuracy", "balanced_accuracy", "macro_f1"]):
        bars = ax.bar(positions + (offset - 1) * 0.23, [row[measure] for row in rows], 0.23, label=measure.replace("_", " ").title()); ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(positions, labels, rotation=8); ax.set(ylim=(0, 1), ylabel="Participant-level score", title="OpenNeuro-only native 500 Hz raw 1D CNN"); ax.grid(axis="y", alpha=0.25); ax.legend()
    fig.savefig(OUT / "performance.png", dpi=180); plt.close(fig)
    test_result = payload["selected_test"]; per_class = test_result["per_class"]
    report = ["# OpenNeuro-only native 500 Hz raw 1D CNN", "", "This experiment uses only the 88 OpenNeuro ds004504 participants and the supplied artifact-cleaned derivative EEG. The native 19-channel montage and 500 Hz sampling rate are preserved. No BrainLat signals, electrode interpolation, resampling, new filtering, or rereferencing are used.", "", "## Participant split", "", f"- Train: {len(train_subjects)}", f"- Validation: {len(val_subjects)}", f"- Test: {len(test_subjects)}", "", "## Validation-selected model", "", f"- Accuracy: {test_result['accuracy']:.4f}", f"- Balanced accuracy: {test_result['balanced_accuracy']:.4f}", f"- Macro F1: {test_result['macro_f1']:.4f}", f"- Recall AD / FTD / HC: {per_class['AD']['recall']:.4f} / {per_class['FTD']['recall']:.4f} / {per_class['HC']['recall']:.4f}", f"- Confusion matrix: {test_result['confusion_matrix']}", "", "The held-out set has only 14 people, so class-specific percentages have high uncertainty and should not be interpreted as external clinical performance."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["protocol", "selected_seed", "selected_epoch", "selected_validation", "selected_test", "three_seed_ensemble_validation", "three_seed_ensemble_test"]}, indent=2))


if __name__ == "__main__":
    main()
