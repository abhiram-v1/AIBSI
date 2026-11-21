from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "waveform_tucker_before_stft_2026-11-05" / "prepared"
OUT = HERE / "outputs" / "raw_1d_cnn"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


SEEDS = [20261116, 20261117, 20261118]
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def training_channel_scale(x: np.ndarray, indexes: np.ndarray, batch_size: int = 256) -> np.ndarray:
    sum_squares = np.zeros(x.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indexes), batch_size):
        values = np.asarray(x[indexes[start : start + batch_size]], dtype=np.float32) * 1e6
        values -= values.mean(axis=2, keepdims=True)
        sum_squares += np.square(values, dtype=np.float64).sum(axis=(0, 2))
        count += values.shape[0] * values.shape[2]
    scale = np.sqrt(sum_squares / max(count, 1))
    return np.maximum(scale, 1e-3).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indexes: np.ndarray,
        channel_scale: np.ndarray,
        augment: bool,
    ) -> None:
        self.x = x
        self.y = y
        self.indexes = np.asarray(indexes, dtype=np.int64)
        self.channel_scale = channel_scale[:, None]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indexes[item])
        values = np.asarray(self.x[index], dtype=np.float32).copy() * 1e6
        values -= values.mean(axis=1, keepdims=True)
        values /= self.channel_scale
        if self.augment:
            values *= np.float32(np.random.uniform(0.90, 1.10))
            shift = int(np.random.randint(-32, 33))
            values = np.roll(values, shift, axis=1)
            if np.random.random() < 0.35:
                channel = int(np.random.randint(0, values.shape[0]))
                values[channel] = 0.0
            values += np.random.normal(0.0, 0.015, size=values.shape).astype(np.float32)
        return torch.from_numpy(values), torch.tensor(int(self.y[index])), torch.tensor(index)


class SeparableBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel: int, stride: int) -> None:
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels_in,
                channels_in,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=channels_in,
                bias=False,
            ),
            nn.BatchNorm1d(channels_in),
            nn.GELU(),
            nn.Conv1d(channels_in, channels_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RawEEG1DCNN(nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        # Four learned temporal filters per electrode. Kernel 65 spans about 0.5 s at 128 Hz.
        self.learned_filterbank = nn.Sequential(
            nn.Conv1d(19, 76, kernel_size=65, padding=32, groups=19, bias=False),
            nn.BatchNorm1d(76),
            nn.GELU(),
            nn.Conv1d(76, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.20),
        )
        self.encoder = nn.Sequential(
            SeparableBlock(64, 96, kernel=31, stride=2),
            SeparableBlock(96, 128, kernel=15, stride=2),
            SeparableBlock(128, 128, kernel=7, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(96, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.encoder(self.learned_filterbank(x))
        mean = values.mean(dim=2)
        maximum = values.amax(dim=2)
        return self.classifier(torch.cat([mean, maximum], dim=1))


@torch.no_grad()
def predict_windows(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indexes: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    dataset = WindowDataset(x, y, indexes, scale, augment=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    output = []
    for values, _, _ in loader:
        logits = model(values.to(device, non_blocking=True))
        output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output)


def aggregate_participants(
    probabilities: np.ndarray,
    window_indexes: np.ndarray,
    subject_index: np.ndarray,
    subjects: np.ndarray,
) -> np.ndarray:
    owners = subject_index[window_indexes]
    result = np.empty((len(subjects), probabilities.shape[1]), dtype=np.float32)
    for row, subject in enumerate(subjects):
        result[row] = probabilities[owners == subject].mean(axis=0)
    return result


def participant_metrics(true: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "macro_f1": float(f1_score(true, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, prediction, labels=np.arange(3)).tolist(),
    }


def train_seed(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    subject_labels: np.ndarray,
    train_subjects: np.ndarray,
    val_subjects: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    seed_everything(seed)
    train_windows = np.flatnonzero(np.isin(subject_index, train_subjects))
    val_windows = np.flatnonzero(np.isin(subject_index, val_subjects))
    scale = training_channel_scale(x, train_windows)

    counts_by_subject = np.bincount(subject_index[train_windows], minlength=len(subject_labels))
    weights = np.asarray(
        [1.0 / max(counts_by_subject[int(subject_index[index])], 1) for index in train_windows],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(train_windows), replacement=True)
    train_loader = DataLoader(
        WindowDataset(x, y, train_windows, scale, augment=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    participant_counts = np.bincount(subject_labels[train_subjects], minlength=3).astype(np.float32)
    class_weights = participant_counts.sum() / np.maximum(3.0 * participant_counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    model = RawEEG1DCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = None
    best_score = (-math.inf, -math.inf)
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for values, labels, _ in train_loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(values)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_window_probabilities = predict_windows(model, x, y, val_windows, scale, device)
        val_probabilities = aggregate_participants(
            val_window_probabilities, val_windows, subject_index, val_subjects
        )
        metrics = participant_metrics(subject_labels[val_subjects], val_probabilities)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        score = (metrics["macro_f1"], metrics["balanced_accuracy"])
        print(
            f"seed={seed} epoch={epoch:02d} loss={np.mean(losses):.4f} "
            f"val_f1={metrics['macro_f1']:.4f} val_bacc={metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    val_window_probabilities = predict_windows(model, x, y, val_windows, scale, device)
    val_probabilities = aggregate_participants(
        val_window_probabilities, val_windows, subject_index, val_subjects
    )
    checkpoint = {
        "model_state": {name: value.cpu() for name, value in model.state_dict().items()},
        "channel_scale": scale,
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
        "architecture": "RawEEG1DCNN",
    }
    torch.save(checkpoint, OUT / f"model_seed_{seed}.pt")
    return checkpoint, val_probabilities, scale


def save_figures(metrics: dict, baseline: dict, mfcc: dict | None, histories: list[list[dict]]) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Raw 1D CNN: held-out participants")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    rows = [baseline]
    labels = ["Log-mel SVM"]
    if mfcc is not None:
        rows.append(mfcc)
        labels.append("MFCC SVM")
    rows.append(metrics)
    labels.append("Raw 1D CNN")
    measures = ["accuracy", "balanced_accuracy", "macro_f1"]
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.4, 5), constrained_layout=True)
    for offset, measure in enumerate(measures):
        bars = ax.bar(
            positions + (offset - 1) * 0.23,
            [row[measure] for row in rows],
            0.23,
            label=measure.replace("_", " ").title(),
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Learned and fixed spectral representations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for seed, history in zip(SEEDS, histories):
        ax.plot([row["epoch"] for row in history], [row["macro_f1"] for row in history], label=str(seed))
    ax.set(xlabel="Epoch", ylabel="Participant validation macro F1", ylim=(0, 1), title="Raw 1D CNN early stopping")
    ax.grid(alpha=0.25)
    ax.legend(title="Seed")
    fig.savefig(OUT / "training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.load(PREPARED / "waveforms.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "waveform_manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = eeg.make_subject_splits(participants)
    train_subjects = np.asarray(splits["train"], dtype=np.int64)
    val_subjects = np.asarray(splits["val"], dtype=np.int64)
    test_subjects = np.asarray(splits["test"], dtype=np.int64)
    subject_labels = np.asarray([eeg.LABEL_ID[row["label"]] for row in participants], dtype=np.int64)
    test_windows = np.flatnonzero(np.isin(subject_index, test_subjects))

    print(
        json.dumps(
            {
                "device": str(device),
                "windows": list(x.shape),
                "split": {
                    "train": len(train_subjects),
                    "validation": len(val_subjects),
                    "test": len(test_subjects),
                },
                "parameters": sum(p.numel() for p in RawEEG1DCNN().parameters()),
            },
            indent=2,
        ),
        flush=True,
    )

    checkpoints = []
    val_seed_probabilities = []
    test_seed_probabilities = []
    histories = []
    start = time.time()
    for seed in SEEDS:
        checkpoint, val_probabilities, scale = train_seed(
            seed,
            x,
            y,
            subject_index,
            subject_labels,
            train_subjects,
            val_subjects,
            device,
        )
        model = RawEEG1DCNN().to(device)
        model.load_state_dict(checkpoint["model_state"])
        test_window_probabilities = predict_windows(model, x, y, test_windows, scale, device)
        test_probabilities = aggregate_participants(
            test_window_probabilities, test_windows, subject_index, test_subjects
        )
        checkpoints.append({"seed": seed, "best_epoch": checkpoint["best_epoch"]})
        histories.append(checkpoint["history"])
        val_seed_probabilities.append(val_probabilities)
        test_seed_probabilities.append(test_probabilities)

    val_probabilities = np.mean(val_seed_probabilities, axis=0)
    test_probabilities = np.mean(test_seed_probabilities, axis=0)
    val_metrics = eeg.metrics(subject_labels[val_subjects], val_probabilities.argmax(axis=1))
    test_metrics = eeg.metrics(subject_labels[test_subjects], test_probabilities.argmax(axis=1))

    baseline = json.loads(
        (HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8")
    )["test"]
    mfcc_path = HERE / "outputs" / "mfcc_svm" / "metrics.json"
    mfcc = json.loads(mfcc_path.read_text(encoding="utf-8"))["test"] if mfcc_path.exists() else None
    payload = {
        "protocol": {
            "input": "raw 19-channel EEG; no STFT, mel filterbank or MFCC",
            "window_shape": list(x.shape[1:]),
            "sampling_rate": manifest["sampling_rate"],
            "window_seconds": manifest["window_seconds"],
            "split_sizes": {
                "train": len(train_subjects),
                "validation": len(val_subjects),
                "test": len(test_subjects),
            },
            "seeds": SEEDS,
            "early_stopping_metric": "participant validation macro F1",
            "test_used_for_selection": False,
            "device": str(device),
            "parameters": sum(p.numel() for p in RawEEG1DCNN().parameters()),
            "elapsed_seconds": time.time() - start,
        },
        "checkpoints": checkpoints,
        "validation_ensemble": val_metrics,
        "test_ensemble": test_metrics,
        "log_mel_svm_reference": baseline,
        "mfcc_svm_reference": mfcc,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "participant_probabilities.npz",
        validation_indices=val_subjects,
        validation_true=subject_labels[val_subjects],
        validation_probabilities=val_probabilities,
        test_indices=test_subjects,
        test_true=subject_labels[test_subjects],
        test_probabilities=test_probabilities,
    )
    save_figures(test_metrics, baseline, mfcc, histories)

    recalls = {name: test_metrics["per_class"][name]["recall"] for name in eeg.CLASSES}
    report = [
        "# Raw 1D CNN experiment",
        "",
        "The model received raw 19-channel, eight-second EEG windows at 128 Hz. "
        "It used learned temporal filters and no STFT, mel filterbank or MFCC transform. "
        "Three predetermined seeds were early-stopped using participant-level validation macro F1, "
        "then averaged before one evaluation on the held-out participants.",
        "",
        "## Protocol",
        "",
        f"- Input per window: {x.shape[1]} electrodes x {x.shape[2]} samples",
        f"- Model parameters: {payload['protocol']['parameters']:,}",
        f"- Train / validation / test participants: {len(train_subjects)} / {len(val_subjects)} / {len(test_subjects)}",
        f"- Selected epochs by seed: {', '.join(str(row['best_epoch']) for row in checkpoints)}",
        "",
        "## Held-out participant result",
        "",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
        f"- Macro F1: {test_metrics['macro_f1']:.4f}",
        f"- Recall AD / FTD / HC: {recalls['AD']:.4f} / {recalls['FTD']:.4f} / {recalls['HC']:.4f}",
        f"- Confusion matrix: {test_metrics['confusion_matrix']}",
        "",
        "## Reference results on the same fixed test participants",
        "",
        f"- Log-mel SVM: accuracy {baseline['accuracy']:.4f}, balanced accuracy {baseline['balanced_accuracy']:.4f}, macro F1 {baseline['macro_f1']:.4f}",
    ]
    if mfcc is not None:
        report.append(
            f"- MFCC SVM: accuracy {mfcc['accuracy']:.4f}, balanced accuracy {mfcc['balanced_accuracy']:.4f}, macro F1 {mfcc['macro_f1']:.4f}"
        )
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"validation": val_metrics, "test": test_metrics, "checkpoints": checkpoints}, indent=2))


if __name__ == "__main__":
    main()
