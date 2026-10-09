from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from functools import partial
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
PREPARED = HERE / "outputs" / "eegpt" / "prepared"
OUT = HERE / "outputs" / "deep_models" / "eegpt_attention"
CHECKPOINT = ROOT / "external" / "EEGPT" / "checkpoint" / "eegpt" / "eegpt_mcae_58chs_4s_large4E.ckpt"
WEIGHTS_PATH = CHECKPOINT.with_name("eegpt_target_encoder_weights.pt")
EEGPT_DOWNSTREAM = ROOT / "external" / "EEGPT" / "downstream"
sys.path.insert(0, str(EEGPT_DOWNSTREAM))
sys.path.insert(0, str(HERE))
from Modules.models.EEGPT_mcae import EEGTransformer  # noqa: E402
import pipeline as eeg  # noqa: E402


SEED = 20261009
EXPECTED_SHA256 = "9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_encoder() -> EEGTransformer:
    encoder = EEGTransformer(
        img_size=[19, 1024],
        patch_size=64,
        embed_num=4,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Sanitized target-encoder weights are missing: {WEIGHTS_PATH}")
    weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    result = encoder.load_state_dict(weights, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"EEGPT weight mismatch: {result}")
    print(f"Loaded {len(weights)} pretrained EEGPT tensors", flush=True)
    return encoder


def augment(window: np.ndarray) -> np.ndarray:
    value = window.copy()
    if np.random.random() < 0.6:
        value *= np.random.uniform(0.90, 1.10)
    if np.random.random() < 0.5:
        value += np.random.normal(0.0, 0.5, value.shape).astype(np.float32)
    if np.random.random() < 0.25:
        value[np.random.randint(value.shape[0])] = 0.0
    if np.random.random() < 0.3:
        value = np.roll(value, np.random.randint(-32, 33), axis=-1)
    return value


class ParticipantDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_index: np.ndarray,
        subjects: list[int],
        bag_size: int | None,
        training: bool,
    ) -> None:
        self.x = x
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.bag_size = bag_size
        self.training = training
        self.windows = {int(subject): np.flatnonzero(subject_index == subject) for subject in self.subjects}
        self.labels = {int(subject): int(y[self.windows[int(subject)][0]]) for subject in self.subjects}

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        subject = int(self.subjects[item])
        indexes = self.windows[subject]
        if self.bag_size is not None:
            indexes = np.random.choice(indexes, self.bag_size, replace=len(indexes) < self.bag_size)
        windows = []
        for index in indexes:
            value = self.x[int(index)].astype(np.float32)
            windows.append(augment(value) if self.training else value)
        return torch.from_numpy(np.stack(windows)), torch.tensor(self.labels[subject]), subject


class EEGPTAttention(nn.Module):
    def __init__(self, encoder: EEGTransformer, channels: list[str]) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("channel_ids", encoder.prepare_chan_ids(channels), persistent=False)
        self.projection = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, 256), nn.GELU(), nn.Dropout(0.25)
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 96), nn.Tanh(), nn.Dropout(0.10), nn.Linear(96, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Dropout(0.35), nn.Linear(256, 96), nn.GELU(), nn.Dropout(0.20), nn.Linear(96, 3)
        )

    def encode_windows(self, windows: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
        outputs = []
        for start in range(0, len(windows), chunk_size):
            encoded = self.encoder(windows[start : start + chunk_size], self.channel_ids)
            # 16 temporal patches x 4 learned summary tokens x 512 dimensions.
            outputs.append(encoded.mean(dim=1).flatten(1))
        return self.projection(torch.cat(outputs, dim=0))

    def forward(self, bags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, windows, channels, samples = bags.shape
        embeddings = self.encode_windows(bags.reshape(batch * windows, channels, samples))
        embeddings = embeddings.reshape(batch, windows, -1)
        attention = torch.softmax(self.attention(embeddings).squeeze(-1), dim=1)
        participant = (embeddings * attention[:, :, None]).sum(dim=1)
        return self.classifier(participant), attention


def set_stage(model: EEGPTAttention, fine_tune: bool) -> tuple[int, int]:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    if fine_tune:
        for block in model.encoder.blocks[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for module in [model.encoder.norm, model.encoder.chan_embed]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        model.encoder.summary_token.requires_grad_(True)
    encoder_trainable = sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad)
    total_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return encoder_trainable, total_trainable


def score(true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
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
def predict(model: EEGPTAttention, dataset: ParticipantDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    model.eval()
    probabilities, labels, rows = [], [], []
    for bags, label, subject in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True):
        with torch.amp.autocast("cuda"):
            logits, attention = model(bags.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1)[0].float().cpu().numpy())
        labels.append(int(label.item()))
        rows.append({"subject": int(subject.item()), "window_attention": attention[0].float().cpu().numpy().tolist()})
    return np.asarray(probabilities), np.asarray(labels), rows


def make_optimizer(model: EEGPTAttention, fine_tune: bool) -> torch.optim.Optimizer:
    head = list(model.projection.parameters()) + list(model.attention.parameters()) + list(model.classifier.parameters())
    if not fine_tune:
        return torch.optim.AdamW(head, lr=3e-4, weight_decay=3e-3)
    encoder = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        [{"params": encoder, "lr": 1e-5}, {"params": head, "lr": 1.5e-4}],
        weight_decay=3e-3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Participant-level EEGPT fine tuning with window attention pooling")
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--bag-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    checkpoint_hash = sha256(CHECKPOINT)
    if checkpoint_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")
    seed_everything(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    x = np.load(PREPARED / "raw_windows.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    splits = eeg.make_subject_splits(participants)
    train_data = ParticipantDataset(x, y, subject_index, splits["train"], args.bag_size, True)
    val_data = ParticipantDataset(x, y, subject_index, splits["val"], None, False)
    test_data = ParticipantDataset(x, y, subject_index, splits["test"], None, False)

    counts = Counter(train_data.labels.values())
    weights = [1.0 / counts[train_data.labels[int(subject)]] for subject in train_data.subjects]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_data) * 2, replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = DataLoader(train_data, batch_size=4, sampler=sampler, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    model = EEGPTAttention(build_encoder(), manifest["channels"]).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.06)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_f1, best_epoch, stale = -1.0, 0, 0
    best_path = OUT / "best_model.pt"
    global_epoch = 0

    for stage_name, stage_epochs, fine_tune in [
        ("frozen_probe", args.warmup_epochs, False),
        ("last_two_blocks", args.finetune_epochs, True),
    ]:
        encoder_trainable, total_trainable = set_stage(model, fine_tune)
        optimizer = make_optimizer(model, fine_tune)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs, eta_min=2e-6)
        stale = 0
        print(
            f"stage={stage_name} encoder_trainable={encoder_trainable:,} total_trainable={total_trainable:,}",
            flush=True,
        )
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            model.train()
            total_loss, seen = 0.0, 0
            for bags, target, _ in loader:
                bags = bags.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits, _ = model(bags)
                    loss = loss_function(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item()) * len(target)
                seen += len(target)
            scheduler.step()
            val_probability, val_true, _ = predict(model, val_data, device)
            val_f1 = float(f1_score(val_true, np.argmax(val_probability, axis=1), average="macro", zero_division=0))
            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "train_loss": total_loss / seen,
                "val_macro_f1": val_f1,
            }
            history.append(row)
            print(
                f"epoch={global_epoch:02d} stage={stage_name} loss={row['train_loss']:.4f} val_macro_f1={val_f1:.4f}",
                flush=True,
            )
            if val_f1 > best_f1 + 1e-4:
                best_f1, best_epoch, stale = val_f1, global_epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": global_epoch, "val_macro_f1": val_f1}, best_path)
            else:
                stale += 1
                if fine_tune and stale >= args.patience:
                    print(f"early stop at epoch {global_epoch}; best epoch={best_epoch}", flush=True)
                    break

    saved = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    val_probability, val_true, val_attention = predict(model, val_data, device)
    test_probability, test_true, test_attention = predict(model, test_data, device)
    val_metrics = score(val_true, np.argmax(val_probability, axis=1))
    test_metrics = score(test_true, np.argmax(test_probability, axis=1))
    baseline = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    attention_cnn = json.loads((HERE / "outputs" / "deep_models" / "attention_pooling" / "metrics.json").read_text(encoding="utf-8"))["ensemble_test"]
    result = {
        "architecture": "pretrained EEGPT encoder + participant window attention pooling",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "pretrained_tensors_loaded": 102,
        "total_parameters": total_parameters,
        "seed": SEED,
        "split_participants": {key: len(value) for key, value in splits.items()},
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
        "svm_baseline": baseline,
        "cnn_attention_baseline": attention_cnn,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "attention.json").write_text(json.dumps({"validation": val_attention, "test": test_attention}), encoding="utf-8")
    np.savez_compressed(OUT / "probabilities.npz", val=val_probability, test=test_probability, val_true=val_true, test_true=test_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    matrix = np.asarray(test_metrics["confusion_matrix"])
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=12)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="EEGPT fine tune: held-out participants")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    names = ["SVM", "CNN + attention", "EEGPT + attention"]
    models = [baseline, attention_cnn, test_metrics]
    measures = ["accuracy", "macro_f1", "macro_recall"]
    locations = np.arange(len(names))
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            locations + (offset - 1) * 0.24,
            [item[measure] for item in models],
            width=0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axes[1].set_xticks(locations, names)
    axes[1].set(ylim=(0, 1), ylabel="Score", title="Same participant split")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
