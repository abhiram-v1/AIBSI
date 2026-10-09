from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "deep_models" / "attention_pooling"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


SEEDS = [20261009, 20261019, 20261029]
SITES = ["AHEPA", "AR", "CL"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(value: np.ndarray) -> np.ndarray:
    value = value.copy()
    if np.random.random() < 0.45:
        width = np.random.randint(2, 5)
        start = np.random.randint(0, value.shape[1] - width + 1)
        value[:, start : start + width] = 0.0
    if np.random.random() < 0.35:
        width = np.random.randint(1, 3)
        start = np.random.randint(0, value.shape[2] - width + 1)
        value[:, :, start : start + width] = 0.0
    if np.random.random() < 0.25:
        value[np.random.randint(0, value.shape[0])] = 0.0
    if np.random.random() < 0.60:
        value += np.random.normal(0, 0.02, value.shape).astype(np.float32)
    return value


def fit_site_scalers(
    x: np.ndarray,
    subject_index: np.ndarray,
    participants: list[dict],
    train_subjects: list[int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_set = set(train_subjects)
    scalers = {}
    for site in SITES:
        subjects = [index for index, row in enumerate(participants) if index in train_set and row["site"] == site]
        indexes = np.flatnonzero(np.isin(subject_index, subjects))
        total = np.zeros(x.shape[1:3], dtype=np.float64)
        total_sq = np.zeros_like(total)
        count = 0
        for start in range(0, len(indexes), 256):
            block = x[indexes[start : start + 256]].astype(np.float32)
            total += block.sum(axis=(0, 3), dtype=np.float64)
            total_sq += np.square(block, dtype=np.float64).sum(axis=(0, 3))
            count += block.shape[0] * block.shape[3]
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-4))
        scalers[site] = (mean.astype(np.float32), std.astype(np.float32))
    return scalers


class ParticipantBagDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_index: np.ndarray,
        participants: list[dict],
        subjects: list[int],
        scalers: dict[str, tuple[np.ndarray, np.ndarray]],
        bag_size: int | None,
        training: bool,
    ) -> None:
        self.x = x
        self.y = y
        self.subject_index = subject_index
        self.participants = participants
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.scalers = scalers
        self.bag_size = bag_size
        self.training = training
        self.windows = {int(subject): np.flatnonzero(subject_index == subject) for subject in self.subjects}
        self.labels = {int(subject): int(y[self.windows[int(subject)][0]]) for subject in self.subjects}

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        subject = int(self.subjects[item])
        indexes = self.windows[subject]
        if self.bag_size is not None:
            indexes = np.random.choice(indexes, self.bag_size, replace=len(indexes) < self.bag_size)
        site = self.participants[subject]["site"]
        mean, std = self.scalers[site]
        values = []
        for index in indexes:
            value = self.x[int(index)].astype(np.float32)
            value = np.clip((value - mean[:, :, None]) / std[:, :, None], -6.0, 6.0)
            values.append(augment(value) if self.training else value)
        return (
            torch.from_numpy(np.stack(values)),
            torch.tensor(self.labels[subject], dtype=torch.long),
            torch.tensor(SITES.index(site), dtype=torch.long),
            subject,
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int], dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False) if in_channels != out_channels or stride != (1, 1) else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = torch.nn.functional.gelu(self.norm1(self.conv1(value)))
        value = self.dropout(self.norm2(self.conv2(value)))
        return torch.nn.functional.gelu(value + residual)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


class AttentionPoolingCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = len(eeg.CHANNELS)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, 12), nn.GELU(), nn.Linear(12, channels), nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualBlock(32, 48, (2, 1), 0.08),
            ResidualBlock(48, 64, (2, 1), 0.10),
            ResidualBlock(64, 96, (2, 1), 0.12),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(96, 96), nn.GELU(), nn.Dropout(0.20))
        self.window_attention = nn.Sequential(
            nn.Linear(96, 48), nn.Tanh(), nn.Linear(48, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(96), nn.Dropout(0.35), nn.Linear(96, 48), nn.GELU(), nn.Dropout(0.20), nn.Linear(48, 3)
        )
        self.site_classifier = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, bags: torch.Tensor, adversarial_strength: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, windows, channels, frequencies, frames = bags.shape
        values = bags.reshape(batch * windows, channels, frequencies, frames)
        channel_summary = values.mean(dim=(2, 3))
        channel_weights = 0.5 + self.channel_attention(channel_summary)
        values = values * channel_weights[:, :, None, None]
        embeddings = self.projection(self.encoder(values)).reshape(batch, windows, -1)
        attention = torch.softmax(self.window_attention(embeddings).squeeze(-1), dim=1)
        participant = (embeddings * attention[:, :, None]).sum(dim=1)
        site_input = GradientReversal.apply(participant, adversarial_strength)
        return self.classifier(participant), self.site_classifier(site_input), attention, channel_weights.reshape(batch, windows, channels)


def metrics(true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(true, prediction, labels=np.arange(3), zero_division=0)
    return {
        "subjects": int(len(true)),
        "accuracy": float(accuracy_score(true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "macro_f1": float(f1_score(true, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, prediction, average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(eeg.CLASSES)
        },
        "confusion_matrix": confusion_matrix(true, prediction, labels=np.arange(3)).tolist(),
    }


@torch.no_grad()
def predict(
    model: AttentionPoolingCNN,
    dataset: ParticipantBagDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    probabilities, labels, attention_rows = [], [], []
    for bags, target, _, subject in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True):
        bags = bags.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits, _, attention, channel_weights = model(bags)
        probabilities.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
        labels.append(int(target.item()))
        attention_rows.append({
            "subject": int(subject.item()),
            "window_attention": attention[0].cpu().numpy().tolist(),
            "mean_channel_attention": channel_weights[0].mean(dim=0).cpu().numpy().tolist(),
        })
    return np.asarray(probabilities), np.asarray(labels), attention_rows


def train_seed(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    participants: list[dict],
    splits: dict[str, list[int]],
    scalers: dict[str, tuple[np.ndarray, np.ndarray]],
    epochs: int,
    patience: int,
    bag_size: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    seed_everything(seed)
    device = torch.device("cuda")
    model = AttentionPoolingCNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_dataset = ParticipantBagDataset(x, y, subject_index, participants, splits["train"], scalers, bag_size, True)
    val_dataset = ParticipantBagDataset(x, y, subject_index, participants, splits["val"], scalers, None, False)
    test_dataset = ParticipantBagDataset(x, y, subject_index, participants, splits["test"], scalers, None, False)
    counts = Counter(train_dataset.labels.values())
    sample_weights = [1.0 / counts[train_dataset.labels[int(subject)]] for subject in train_dataset.subjects]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset) * 3, replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    class_loss = nn.CrossEntropyLoss(label_smoothing=0.08)
    site_loss = nn.CrossEntropyLoss()
    amp_scaler = torch.amp.GradScaler("cuda")
    run_dir = OUT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"
    history = []
    best_f1, best_epoch, stale = -1.0, 0, 0
    print(f"seed={seed} parameters={parameter_count:,}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        strength = 0.12 * min(epoch / 10.0, 1.0)
        for bags, target, site, _ in loader:
            bags = bags.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            site = site.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits, site_logits, _, _ = model(bags, strength)
                loss = class_loss(logits, target) + 0.12 * site_loss(site_logits, site)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total_loss += float(loss.item()) * len(target)
            seen += len(target)
        scheduler.step()
        val_probability, val_true, _ = predict(model, val_dataset, device)
        val_prediction = np.argmax(val_probability, axis=1)
        val_f1 = float(f1_score(val_true, val_prediction, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "val_macro_f1": val_f1})
        print(f"seed={seed} epoch={epoch:02d} loss={total_loss/seen:.4f} val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1 + 1e-4:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at {epoch}; best epoch={best_epoch}", flush=True)
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    val_probability, val_true, val_attention = predict(model, val_dataset, device)
    test_probability, test_true, test_attention = predict(model, test_dataset, device)
    result = {
        "seed": seed,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "val": metrics(val_true, np.argmax(val_probability, axis=1)),
        "test": metrics(test_true, np.argmax(test_probability, axis=1)),
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "attention.json").write_text(json.dumps({"validation": val_attention, "test": test_attention}), encoding="utf-8")
    return result, val_probability, test_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN with electrode and participant-window attention pooling")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--bag-size", type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    splits = eeg.make_subject_splits(participants)
    scalers = fit_site_scalers(x, subject_index, participants, splits["train"])
    runs, val_runs, test_runs = [], [], []
    for seed in SEEDS:
        result, val_probability, test_probability = train_seed(
            seed, x, y, subject_index, participants, splits, scalers,
            args.epochs, args.patience, args.bag_size,
        )
        runs.append(result)
        val_runs.append(val_probability)
        test_runs.append(test_probability)

    val_probability = np.mean(np.stack(val_runs), axis=0)
    test_probability = np.mean(np.stack(test_runs), axis=0)
    val_subjects = np.asarray(splits["val"], dtype=int)
    test_subjects = np.asarray(splits["test"], dtype=int)
    subject_labels = np.asarray([eeg.LABEL_ID[row["label"]] for row in participants])
    ensemble_val = metrics(subject_labels[val_subjects], np.argmax(val_probability, axis=1))
    ensemble_test = metrics(subject_labels[test_subjects], np.argmax(test_probability, axis=1))
    baseline = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    payload = {
        "architecture": "site-normalized log-mel CNN + electrode attention + window attention pooling + site-adversarial head",
        "seeds": SEEDS,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "bag_size": args.bag_size,
        "individual_runs": runs,
        "ensemble_validation": ensemble_val,
        "ensemble_test": ensemble_test,
        "svm_baseline": baseline,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ensemble_validation": ensemble_val, "ensemble_test": ensemble_test}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    confusion = np.asarray(ensemble_test["confusion_matrix"])
    image = axes[0].imshow(confusion, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(confusion[row, column]), ha="center", va="center", fontsize=12)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="CNN attention-pooling ensemble")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    measures = ["accuracy", "macro_f1", "macro_recall"]
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            np.arange(2) + (offset - 1) * 0.24,
            [baseline[measure], ensemble_test[measure]], 0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2)
    axes[1].set_xticks(np.arange(2), ["Site-normalized SVM", "CNN + attention pooling"])
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Attention pooling comparison")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
