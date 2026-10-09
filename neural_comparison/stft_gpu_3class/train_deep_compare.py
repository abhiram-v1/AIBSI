from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from run_pipeline import CLASS_NAMES, PREPARED, TARGET_CHANNELS, make_subject_splits


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "deep_compare"
SEEDS = [20261009, 20261019, 20261029]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spectral_shape(values: np.ndarray) -> np.ndarray:
    """Remove per-channel/per-frame gain while retaining spectral shape."""
    values = values.astype(np.float32, copy=True)
    values -= values.mean(axis=-2, keepdims=True)
    values /= np.maximum(values.std(axis=-2, keepdims=True), 0.15)
    return values


def fit_scaler(x: np.ndarray, indexes: np.ndarray, batch_size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    total = None
    total_sq = None
    count = 0
    for start in tqdm(range(0, len(indexes), batch_size), desc="Enhanced train scaler"):
        block = spectral_shape(x[indexes[start : start + batch_size]])
        block_sum = block.sum(axis=(0, 3), keepdims=True, dtype=np.float64)
        block_sq = np.square(block, dtype=np.float64).sum(axis=(0, 3), keepdims=True)
        total = block_sum if total is None else total + block_sum
        total_sq = block_sq if total_sq is None else total_sq + block_sq
        count += block.shape[0] * block.shape[3]
    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 1e-4)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def augment(value: np.ndarray) -> np.ndarray:
    value = value.copy()
    if np.random.random() < 0.65:
        width = np.random.randint(2, 8)
        start = np.random.randint(0, value.shape[1] - width + 1)
        value[:, start : start + width, :] = 0.0
    if np.random.random() < 0.55:
        width = np.random.randint(1, 4)
        start = np.random.randint(0, value.shape[2] - width + 1)
        value[:, :, start : start + width] = 0.0
    if np.random.random() < 0.35:
        channels = np.random.choice(value.shape[0], size=np.random.randint(1, 3), replace=False)
        value[channels] = 0.0
    if np.random.random() < 0.70:
        value += np.random.normal(0.0, 0.025, size=value.shape).astype(np.float32)
    return value


class SubjectBagDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_index: np.ndarray,
        subjects: list[int],
        mean: np.ndarray,
        std: np.ndarray,
        bag_size: int,
    ) -> None:
        self.x = x
        self.y = y
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.mean = mean[0]
        self.std = std[0]
        self.bag_size = bag_size
        self.windows = {int(s): np.flatnonzero(subject_index == s) for s in self.subjects}
        self.labels = {
            int(s): int(y[self.windows[int(s)][0]]) for s in self.subjects
        }

    def __len__(self) -> int:
        return len(self.subjects)

    def transform(self, index: int, training: bool = True) -> np.ndarray:
        value = spectral_shape(self.x[index])
        value = np.clip((value - self.mean) / self.std, -6.0, 6.0)
        return augment(value) if training else value

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        subject = int(self.subjects[item])
        candidates = self.windows[subject]
        chosen = np.random.choice(candidates, size=self.bag_size, replace=len(candidates) < self.bag_size)
        bag = np.stack([self.transform(int(index)) for index in chosen])
        return (
            torch.from_numpy(bag),
            torch.tensor(self.labels[subject], dtype=torch.long),
            subject,
        )


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indexes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.indexes = indexes
        self.mean = mean[0]
        self.std = std[0]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indexes[item])
        value = spectral_shape(self.x[index])
        value = np.clip((value - self.mean) / self.std, -6.0, 6.0)
        return torch.from_numpy(value), torch.tensor(int(self.y[index]), dtype=torch.long), index


class ResidualFrequencyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool, dropout: float) -> None:
        super().__init__()
        stride = (2, 1) if pool else (1, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if in_channels != out_channels or pool
            else nn.Identity()
        )
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = self.activation(self.norm1(self.conv1(values)))
        values = self.dropout(self.norm2(self.conv2(values)))
        return self.activation(values + residual)


class SpectralEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(len(TARGET_CHANNELS), 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualFrequencyBlock(32, 32, False, 0.08),
            ResidualFrequencyBlock(32, 48, True, 0.10),
            ResidualFrequencyBlock(48, 64, True, 0.12),
            ResidualFrequencyBlock(64, 96, True, 0.15),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [batch, 96, reduced-frequency, STFT-time] -> [batch, STFT-time, 96]
        return self.layers(values).mean(dim=2).transpose(1, 2)


class CompactCNN(nn.Module):
    name = "cnn"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SpectralEncoder()
        self.classifier = nn.Sequential(
            nn.LayerNorm(96),
            nn.Dropout(0.40),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(48, len(CLASS_NAMES)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(values).mean(dim=1))


class CNNBiLSTM(nn.Module):
    name = "cnn_bilstm"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SpectralEncoder()
        self.temporal = nn.LSTM(
            input_size=96,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(nn.Linear(128, 48), nn.Tanh(), nn.Linear(48, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(0.45),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.temporal(self.encoder(values))
        weights = torch.softmax(self.attention(sequence), dim=1)
        return self.classifier((sequence * weights).sum(dim=1))


@torch.no_grad()
def predict_windows(
    model: nn.Module, loader: DataLoader, device: torch.device, total_windows: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = np.full((total_windows, len(CLASS_NAMES)), np.nan, dtype=np.float32)
    labels = np.full(total_windows, -1, dtype=np.int64)
    for values, target, indexes in loader:
        values = values.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
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


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def train_one(
    model_type: str,
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    participants: list[dict[str, Any]],
    splits: dict[str, list[int]],
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    bag_size: int,
    learning_rate: float,
    patience: int = 12,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seed_everything(seed)
    device = torch.device("cuda")
    model: nn.Module = CompactCNN() if model_type == "cnn" else CNNBiLSTM()
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_dir = OUT / model_type / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SubjectBagDataset(
        x, y, subject_index, splits["train"], mean, std, bag_size
    )
    train_counts = Counter(train_dataset.labels.values())
    sample_weights = [1.0 / train_counts[train_dataset.labels[int(s)]] for s in train_dataset.subjects]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_dataset) * 2,
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    split_indexes = {
        name: np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))
        for name, subjects in splits.items()
    }
    eval_loaders = {
        name: DataLoader(
            WindowDataset(x, y, split_indexes[name], mean, std),
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for name in ["val", "test"]
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=learning_rate / 40
    )
    scaler = torch.amp.GradScaler("cuda")
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "best_model.pt"

    print(f"\n{model_type} seed={seed}: parameters={parameter_count:,}")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for bags, targets, _ in tqdm(train_loader, desc=f"{model_type} {seed} e{epoch:02d}", leave=False):
            batch, bag_count, channels, frequencies, frames = bags.shape
            bags = bags.to(device, non_blocking=True).reshape(
                batch * bag_count, channels, frequencies, frames
            )
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                window_logits = model(bags).reshape(batch, bag_count, -1)
                subject_logits = window_logits.mean(dim=1)
                loss = criterion(subject_logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * batch
            seen += batch
        scheduler.step()

        val_probabilities, val_labels = predict_windows(model, eval_loaders["val"], device, len(x))
        val_rows = aggregate_subjects(
            val_probabilities,
            val_labels,
            subject_index,
            split_indexes["val"],
            participants,
        )
        val_f1 = calculate_metrics(val_rows)["macro_f1"]
        item = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "val_macro_f1": val_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(item)
        print(
            f"{model_type} seed={seed} epoch={epoch:02d} "
            f"loss={item['train_loss']:.4f} val_macro_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": model_type,
                    "seed": seed,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "parameter_count": parameter_count,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop: best epoch={best_epoch}, val macro F1={best_f1:.4f}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_probabilities, val_labels = predict_windows(model, eval_loaders["val"], device, len(x))
    test_probabilities, test_labels = predict_windows(model, eval_loaders["test"], device, len(x))
    val_rows = aggregate_subjects(
        val_probabilities, val_labels, subject_index, split_indexes["val"], participants
    )
    test_rows = aggregate_subjects(
        test_probabilities, test_labels, subject_index, split_indexes["test"], participants
    )
    result = {
        "model": model_type,
        "seed": seed,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "val": calculate_metrics(val_rows),
        "test": calculate_metrics(test_rows),
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, val_probabilities, test_probabilities, history


def ensemble_rows(
    probability_runs: list[np.ndarray],
    y: np.ndarray,
    subject_index: np.ndarray,
    indexes: np.ndarray,
    participants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    probabilities = np.full_like(probability_runs[0], np.nan)
    probabilities[indexes] = np.mean(
        np.stack([run[indexes] for run in probability_runs]), axis=0
    )
    labels = np.full(len(y), -1, dtype=np.int64)
    labels[indexes] = y[indexes]
    return aggregate_subjects(probabilities, labels, subject_index, indexes, participants)


def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "subject_id", "source", "site", "true_label", "predicted_label",
        "prob_AD", "prob_FTD", "prob_HC", "correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot_comparison(results: dict[str, Any]) -> None:
    models = ["cnn", "cnn_bilstm"]
    labels = ["CNN", "CNN–BiLSTM"]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    colors = ["#2878b5", "#f28e2b", "#59a14f"]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    positions = np.arange(len(models))
    for offset, (measure, color) in enumerate(zip(measures, colors)):
        values = [results[model]["test"][measure] for model in models]
        bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, label=measure.replace("_", " ").title(), color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Participant-level score", title="Enhanced STFT models on the same held-out participants")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    maximum = max(np.asarray(results[model]["test"]["confusion_matrix"]).max() for model in models)
    for ax, model, label in zip(axes, models, labels):
        matrix = np.asarray(results[model]["test"]["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set(xlabel="Predicted", ylabel="True", title=label)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    fig.savefig(OUT / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
    class_positions = np.arange(3)
    for ax, model, label in zip(axes, models, labels):
        for offset, measure in enumerate(["recall", "f1"]):
            values = [results[model]["test"]["per_class"][name][measure] for name in CLASS_NAMES]
            bars = ax.bar(class_positions + (offset - 0.5) * 0.34, values, 0.34, label=measure.title())
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_xticks(class_positions, CLASS_NAMES)
        ax.set(ylim=(0, 1), title=label, ylabel="Score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.savefig(OUT / "per_class_recall_f1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare participant-trained CNN and CNN-BiLSTM models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bag-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seeds", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)

    x = np.load(PREPARED / "spectrograms.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = make_subject_splits(participants)
    split_indexes = {
        name: np.flatnonzero(np.isin(subject_index, np.asarray(subjects, dtype=np.int32)))
        for name, subjects in splits.items()
    }
    mean, std = fit_scaler(x, split_indexes["train"])
    np.savez_compressed(OUT / "enhanced_train_scaler.npz", mean=mean, std=std)
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    all_results: dict[str, Any] = {}
    for model_type in ["cnn", "cnn_bilstm"]:
        seed_results = []
        val_runs = []
        test_runs = []
        for seed in SEEDS[: args.seeds]:
            result, val_probabilities, test_probabilities, _ = train_one(
                model_type,
                seed,
                x,
                y,
                subject_index,
                participants,
                splits,
                mean,
                std,
                args.epochs,
                args.bag_size,
                args.learning_rate,
            )
            seed_results.append(result)
            val_runs.append(val_probabilities)
            test_runs.append(test_probabilities)

        val_rows = ensemble_rows(
            val_runs, y, subject_index, split_indexes["val"], participants
        )
        test_rows = ensemble_rows(
            test_runs, y, subject_index, split_indexes["test"], participants
        )
        model_result: dict[str, Any] = {
            "ensemble_seeds": SEEDS[: args.seeds],
            "individual_runs": seed_results,
            "val": calculate_metrics(val_rows),
            "test": calculate_metrics(test_rows),
        }
        for source in ["ds004504", "brainlat"]:
            source_rows = [row for row in test_rows if row["source"] == source]
            model_result[f"test_{source}"] = calculate_metrics(source_rows)
        all_results[model_type] = model_result
        save_predictions(OUT / f"{model_type}_subject_predictions.csv", test_rows)
        print(f"\n{model_type} ensemble test:")
        print(json.dumps(model_result["test"], indent=2))

    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "participants": len(participants),
        "split_participants": {name: len(values) for name, values in splits.items()},
        "bag_size": args.bag_size,
        "spectral_normalization": "per-channel per-frame frequency-shape z-score plus train-only frequency scaling",
        "results": all_results,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_comparison(all_results)
    print(f"\nSaved comparison to {OUT}")


if __name__ == "__main__":
    main()
