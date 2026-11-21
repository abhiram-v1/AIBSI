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


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "waveform_tucker_before_stft_2026-11-05" / "prepared"
OUT = HERE / "outputs" / "raw_1d_cnn_mil"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
from train_raw_1d_cnn import training_channel_scale  # noqa: E402


SEEDS = [20261121, 20261122, 20261123]
BAG_SIZE = 8
BATCH_SIZE = 16
MAX_EPOCHS = 80
PATIENCE = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_window(values: np.ndarray, scale: np.ndarray, augment: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy() * 1e6
    values -= values.mean(axis=1, keepdims=True)
    values /= scale[:, None]
    if augment:
        values *= np.float32(np.random.uniform(0.90, 1.10))
        values = np.roll(values, int(np.random.randint(-32, 33)), axis=1)
        if np.random.random() < 0.30:
            values[int(np.random.randint(0, values.shape[0]))] = 0.0
        values += np.random.normal(0, 0.012, values.shape).astype(np.float32)
    return values


class ParticipantBagDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        subject_index: np.ndarray,
        subject_labels: np.ndarray,
        subjects: np.ndarray,
        scale: np.ndarray,
        bag_size: int,
    ) -> None:
        self.x = x
        self.subject_labels = subject_labels
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.scale = scale
        self.bag_size = bag_size
        self.window_indexes = {
            int(subject): np.flatnonzero(subject_index == subject) for subject in self.subjects
        }

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subject = int(self.subjects[item])
        available = self.window_indexes[subject]
        chosen = np.random.choice(available, self.bag_size, replace=len(available) < self.bag_size)
        bag = np.stack([normalize_window(self.x[index], self.scale, True) for index in chosen])
        return torch.from_numpy(bag), torch.tensor(int(self.subject_labels[subject])), torch.tensor(subject)


class EvaluationWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        subject_index: np.ndarray,
        indexes: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        self.x = x
        self.subject_index = subject_index
        self.indexes = np.asarray(indexes, dtype=np.int64)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indexes[item])
        values = normalize_window(self.x[index], self.scale, False)
        return torch.from_numpy(values), torch.tensor(int(self.subject_index[index]))


class SeparableBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels_in,
                channels_in,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                groups=channels_in,
                bias=False,
            ),
            nn.BatchNorm1d(channels_in),
            nn.GELU(),
            nn.Conv1d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class MultiScaleWindowEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Two learned filters per electrode at four temporal scales.
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(19, 38, kernel, padding=kernel // 2, groups=19, bias=False),
                    nn.BatchNorm1d(38),
                    nn.GELU(),
                    nn.AvgPool1d(4),
                )
                for kernel in [17, 33, 65, 129]
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(152, 96, 1, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.encoder = nn.Sequential(
            SeparableBlock(96, 128, 31, 2),
            SeparableBlock(128, 160, 15, 2),
            SeparableBlock(160, 160, 7, 2),
        )
        self.projection = nn.Sequential(
            nn.Linear(320, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.cat([branch(values) for branch in self.branches], dim=1)
        values = self.encoder(self.fuse(values))
        pooled = torch.cat([values.mean(dim=2), values.amax(dim=2)], dim=1)
        return self.projection(pooled)


class ParticipantMILCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.window_encoder = MultiScaleWindowEncoder()
        self.attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.35), nn.Linear(64, 3)
        )

    def classify_embeddings(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attention(embeddings).squeeze(-1), dim=1)
        participant = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return self.classifier(participant), weights

    def forward(self, bags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, windows, channels, samples = bags.shape
        embeddings = self.window_encoder(bags.reshape(batch * windows, channels, samples))
        embeddings = embeddings.reshape(batch, windows, -1)
        return self.classify_embeddings(embeddings)


@torch.no_grad()
def participant_probabilities(
    model: ParticipantMILCNN,
    x: np.ndarray,
    subject_index: np.ndarray,
    subjects: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[int, list[float]]]:
    model.eval()
    indexes = np.flatnonzero(np.isin(subject_index, subjects))
    loader = DataLoader(
        EvaluationWindowDataset(x, subject_index, indexes, scale),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    embedding_parts = []
    owner_parts = []
    for windows, owners in loader:
        embedding_parts.append(
            model.window_encoder(windows.to(device, non_blocking=True)).cpu()
        )
        owner_parts.append(owners)
    all_embeddings = torch.cat(embedding_parts)
    all_owners = torch.cat(owner_parts).numpy()
    probabilities = []
    attentions: dict[int, list[float]] = {}
    for subject in subjects:
        embeddings = all_embeddings[all_owners == subject].to(device).unsqueeze(0)
        logits, weights = model.classify_embeddings(embeddings)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
        attentions[int(subject)] = weights.cpu().numpy()[0].tolist()
    return np.asarray(probabilities), attentions


def metrics(true: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    return eeg.metrics(true, prediction)


def train_seed(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    subject_labels: np.ndarray,
    train_subjects: np.ndarray,
    validation_subjects: np.ndarray,
    device: torch.device,
) -> dict:
    seed_everything(seed)
    train_windows = np.flatnonzero(np.isin(subject_index, train_subjects))
    scale = training_channel_scale(x, train_windows)
    dataset = ParticipantBagDataset(
        x, subject_index, subject_labels, train_subjects, scale, BAG_SIZE
    )
    class_counts = np.bincount(subject_labels[train_subjects], minlength=3)
    sample_weights = np.asarray(
        [1.0 / class_counts[subject_labels[subject]] for subject in train_subjects],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_subjects) * 4, replacement=True
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    model = ParticipantMILCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_score = (-math.inf, -math.inf)
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for bags, labels, _ in loader:
            bags = bags.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = model(bags)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_probabilities, _ = participant_probabilities(
            model, x, subject_index, validation_subjects, scale, device
        )
        validation_metrics = metrics(subject_labels[validation_subjects], validation_probabilities)
        score = (validation_metrics["macro_f1"], validation_metrics["balanced_accuracy"])
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), **validation_metrics}
        )
        print(
            f"seed={seed} epoch={epoch:02d} loss={np.mean(losses):.4f} "
            f"val_f1={validation_metrics['macro_f1']:.4f} "
            f"val_bacc={validation_metrics['balanced_accuracy']:.4f}",
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
    validation_probabilities, validation_attention = participant_probabilities(
        model, x, subject_index, validation_subjects, scale, device
    )
    checkpoint = {
        "model_state": {name: value.cpu() for name, value in model.state_dict().items()},
        "channel_scale": scale,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metrics": metrics(subject_labels[validation_subjects], validation_probabilities),
        "validation_probabilities": validation_probabilities,
        "validation_attention": validation_attention,
        "history": history,
    }
    torch.save(checkpoint, OUT / f"model_seed_{seed}.pt")
    return checkpoint


def save_figures(selected_test: dict, baseline: dict, earlier_cnn: dict, histories: list[list[dict]]) -> None:
    matrix = np.asarray(selected_test["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Multiscale participant-level 1D CNN")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    rows = [baseline, earlier_cnn, selected_test]
    labels = ["Log-mel SVM", "Window-trained 1D CNN", "Participant MIL CNN"]
    measures = ["accuracy", "balanced_accuracy", "macro_f1"]
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    for offset, measure in enumerate(measures):
        bars = ax.bar(
            positions + (offset - 1) * 0.23,
            [row[measure] for row in rows],
            0.23,
            label=measure.replace("_", " ").title(),
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Raw CNN training objectives")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for seed, history in zip(SEEDS, histories):
        ax.plot(
            [row["epoch"] for row in history],
            [row["macro_f1"] for row in history],
            label=str(seed),
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Validation macro F1",
        ylim=(0, 1),
        title="Participant-level raw CNN early stopping",
    )
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
    validation_subjects = np.asarray(splits["val"], dtype=np.int64)
    test_subjects = np.asarray(splits["test"], dtype=np.int64)
    subject_labels = np.asarray([eeg.LABEL_ID[row["label"]] for row in participants])
    parameters = sum(parameter.numel() for parameter in ParticipantMILCNN().parameters())
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": parameters,
                "split": [len(train_subjects), len(validation_subjects), len(test_subjects)],
            },
            indent=2,
        ),
        flush=True,
    )

    checkpoints = []
    start = time.time()
    for seed in SEEDS:
        checkpoints.append(
            train_seed(
                seed,
                x,
                y,
                subject_index,
                subject_labels,
                train_subjects,
                validation_subjects,
                device,
            )
        )

    selected = max(
        checkpoints,
        key=lambda row: (
            row["validation_metrics"]["macro_f1"],
            row["validation_metrics"]["balanced_accuracy"],
            row["validation_metrics"]["accuracy"],
        ),
    )
    model = ParticipantMILCNN().to(device)
    model.load_state_dict(selected["model_state"])
    test_probabilities, test_attention = participant_probabilities(
        model,
        x,
        subject_index,
        test_subjects,
        selected["channel_scale"],
        device,
    )
    test_metrics = metrics(subject_labels[test_subjects], test_probabilities)

    baseline = json.loads(
        (HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8")
    )["test"]
    earlier_cnn = json.loads(
        (HERE / "outputs" / "raw_1d_cnn_selected" / "metrics.json").read_text(encoding="utf-8")
    )["selected_test"]
    payload = {
        "protocol": {
            "input": "raw 19 x 1024 EEG windows",
            "training_objective": "participant classification from attention-pooled bags",
            "bag_size": BAG_SIZE,
            "parameters": parameters,
            "seeds": SEEDS,
            "selection": "participant validation macro F1",
            "test_used_for_selection": False,
            "elapsed_seconds": time.time() - start,
        },
        "selected_seed": selected["seed"],
        "selected_epoch": selected["best_epoch"],
        "selected_validation": selected["validation_metrics"],
        "selected_test": test_metrics,
        "all_validation": [
            {
                "seed": row["seed"],
                "epoch": row["best_epoch"],
                "metrics": row["validation_metrics"],
            }
            for row in checkpoints
        ],
        "log_mel_svm_reference": baseline,
        "window_cnn_reference": earlier_cnn,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "participant_probabilities.npz",
        selected_seed=selected["seed"],
        validation_indices=validation_subjects,
        validation_true=subject_labels[validation_subjects],
        validation_probabilities=selected["validation_probabilities"],
        test_indices=test_subjects,
        test_true=subject_labels[test_subjects],
        test_probabilities=test_probabilities,
    )
    (OUT / "test_attention.json").write_text(
        json.dumps(test_attention, indent=2), encoding="utf-8"
    )
    save_figures(test_metrics, baseline, earlier_cnn, [row["history"] for row in checkpoints])

    recalls = {name: test_metrics["per_class"][name]["recall"] for name in eeg.CLASSES}
    report = [
        "# Multiscale participant-level raw 1D CNN",
        "",
        "The CNN learned temporal filters at four scales directly from raw EEG. Training used "
        "eight-window participant bags and attention pooling, so the loss optimized participant "
        "classification rather than treating every window as an independent diagnosis.",
        "",
        "## Selection",
        "",
        f"- Selected seed: {selected['seed']}",
        f"- Selected epoch: {selected['best_epoch']}",
        f"- Validation accuracy: {selected['validation_metrics']['accuracy']:.4f}",
        f"- Validation balanced accuracy: {selected['validation_metrics']['balanced_accuracy']:.4f}",
        f"- Validation macro F1: {selected['validation_metrics']['macro_f1']:.4f}",
        "",
        "## Held-out participant result",
        "",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
        f"- Macro F1: {test_metrics['macro_f1']:.4f}",
        f"- Recall AD / FTD / HC: {recalls['AD']:.4f} / {recalls['FTD']:.4f} / {recalls['HC']:.4f}",
        f"- Confusion matrix: {test_metrics['confusion_matrix']}",
        "",
        "The fixed test set has already been inspected in earlier experiments. Treat this as an exploratory architecture comparison, not a new blind test estimate.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_seed": selected["seed"],
                "validation": selected["validation_metrics"],
                "test": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
