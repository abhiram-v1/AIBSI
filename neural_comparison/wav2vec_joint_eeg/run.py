"""Participant-held-out comparison of supervised and masked-context EEG encoders.

Uses the cached 13-frame log-mel windows, so this is wav2vec-inspired rather
than a raw-waveform wav2vec 2.0 reproduction. The diagnosis head trains from
the first step in every arm. No outer-test windows enter fitting or selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "neural_comparison/advanced_clean_spectral/outputs/prepared"
GNTF = ROOT / "neural_comparison/graph_ntf_svm/outputs/full/oof_predictions.json"
CLASSES = ("AD", "FTD", "HC")
ARMS = ("supervised", "masked", "masked_contrastive")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(19, 32, (5, 3), padding=(2, 1), bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, (3, 3), padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.norm = nn.LayerNorm(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.cnn(x).mean(dim=2).transpose(1, 2))


class JointEEG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.position = nn.Parameter(torch.zeros(1, 13, 64))
        nn.init.trunc_normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=.1, activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.mask_token = nn.Parameter(torch.zeros(64))
        self.predictor = nn.Linear(64, 64)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Dropout(.2), nn.Linear(64, 3))

    def window_embedding(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.context(z + self.position).mean(dim=1)

    def masked_loss(self, z: torch.Tensor) -> torch.Tensor:
        # One contiguous three-frame span in each eight-second window.
        n, frames, _ = z.shape
        starts = torch.randint(frames - 2, (n,), device=z.device)
        mask = torch.zeros(n, frames, dtype=torch.bool, device=z.device)
        rows = torch.arange(n, device=z.device)
        for offset in range(3):
            mask[rows, starts + offset] = True
        masked = torch.where(mask.unsqueeze(-1), self.mask_token, z)
        predicted = self.predictor(self.context(masked + self.position))
        return F.smooth_l1_loss(predicted[mask], z.detach()[mask])

    def training_outputs(self, x: torch.Tensor, bag_size: int, arm: str):
        z = self.encoder(x)
        windows = self.context(z + self.position).mean(dim=1)
        subject = windows.reshape(-1, bag_size, 64).mean(dim=1)
        logits = self.head(subject)
        auxiliary = self.masked_loss(z) if arm != "supervised" else logits.new_zeros(())
        return logits, subject, auxiliary


def supervised_contrastive(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = .2) -> torch.Tensor:
    vectors = F.normalize(embeddings, dim=1)
    similarity = vectors @ vectors.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    similarity = similarity.masked_fill(diagonal, -1e4)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    return -((log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)).mean()


def normalized_windows(x: np.ndarray, subject_index: np.ndarray, train_subjects: np.ndarray) -> np.ndarray:
    train_indices = np.flatnonzero(np.isin(subject_index, train_subjects))
    total = np.zeros((19, 24), dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = 0
    for start in range(0, len(train_indices), 128):
        block = x[train_indices[start:start + 128]].astype(np.float32)
        block -= block.mean(axis=2, keepdims=True)
        block /= np.maximum(block.std(axis=2, keepdims=True), .15)
        total += block.sum(axis=(0, 3), dtype=np.float64)
        total_sq += np.square(block, dtype=np.float64).sum(axis=(0, 3))
        count += block.shape[0] * block.shape[3]
    mean = (total / count).astype(np.float32)
    std = np.sqrt(np.maximum(total_sq / count - (total / count) ** 2, 1e-4)).astype(np.float32)
    transformed = np.empty(x.shape, dtype=np.float16)
    for start in range(0, len(x), 128):
        block = x[start:start + 128].astype(np.float32)
        block -= block.mean(axis=2, keepdims=True)
        block /= np.maximum(block.std(axis=2, keepdims=True), .15)
        transformed[start:start + len(block)] = np.clip((block - mean[None, :, :, None]) / std[None, :, :, None], -6, 6)
    return transformed


def splits_from_gntf(participants: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    records = json.loads(GNTF.read_text(encoding="utf-8"))
    chosen = sorted((r for r in records if r["method"] == "gntf" and r["repeat"] == 0), key=lambda r: r["fold"])
    ids = {p["subject_id"]: i for i, p in enumerate(participants)}
    if len(chosen) != 5:
        raise ValueError("Expected five first-repeat G-NTF outer folds")
    folds = []
    for record in chosen:
        train = np.array([ids[s] for s in record["train_ids"]], dtype=np.int64)
        test = np.array([ids[s] for s in record["test_ids"]], dtype=np.int64)
        if set(train) & set(test) or len(train) + len(test) != len(participants):
            raise ValueError("Invalid participant split")
        folds.append((train, test))
    if sorted(i for _, test in folds for i in test) != list(range(len(participants))):
        raise ValueError("Outer tests do not cover each participant exactly once")
    return folds


@torch.no_grad()
def evaluate(model: JointEEG, data: np.ndarray, subjects: np.ndarray, windows: list[np.ndarray], y: np.ndarray,
             device: torch.device, batch: int = 192) -> tuple[np.ndarray, dict]:
    model.eval()
    use = np.concatenate([windows[int(s)] for s in subjects])
    embeddings = np.empty((len(use), 64), dtype=np.float32)
    for start in range(0, len(use), batch):
        idx = use[start:start + batch]
        xb = torch.from_numpy(data[idx].astype(np.float32)).to(device)
        embeddings[start:start + len(idx)] = model.window_embedding(xb).cpu().numpy()
    by_subject = []
    cursor = 0
    for subject in subjects:
        count = len(windows[int(subject)])
        by_subject.append(embeddings[cursor:cursor + count].mean(axis=0))
        cursor += count
    logits = model.head(torch.from_numpy(np.stack(by_subject)).to(device)).cpu().numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    truth = y[subjects]
    pred = probabilities.argmax(axis=1)
    return probabilities, {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "recall": [float(v) for v in recall_score(truth, pred, labels=[0, 1, 2], average=None, zero_division=0)],
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
    }


def train_one(arm: str, fold: int, seed: int, x: np.ndarray, subject_index: np.ndarray,
              participants: list[dict], y: np.ndarray, train_outer: np.ndarray, test: np.ndarray,
              epochs: int, bag_size: int, out: Path) -> dict:
    seed_all(seed + fold)
    strata = np.array([f"{participants[int(s)]['label']}:{participants[int(s)]['source']}" for s in train_outer])
    if min(Counter(strata).values()) < 2:
        strata = y[train_outer]
    inner_train_index, val_index = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=seed + 41 * fold).split(train_outer, strata))
    train, val = train_outer[inner_train_index], train_outer[val_index]
    data = normalized_windows(x, subject_index, train)
    windows = [np.flatnonzero(subject_index == i) for i in range(len(participants))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointEEG().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    rng = np.random.default_rng(seed + 101 * fold)
    pools = {label: train[y[train] == label] for label in range(3)}
    best_score = (-1.0, -1.0, float("-inf"))
    best_epoch = 0
    stale = 0
    history = []
    checkpoint = out / f"fold_{fold:02d}_{arm}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(math.ceil(len(train) / 12)):
            subjects = np.concatenate([rng.choice(pools[label], 4, replace=len(pools[label]) < 4) for label in range(3)])
            rng.shuffle(subjects)
            selected = np.stack([rng.choice(windows[int(s)], bag_size, replace=len(windows[int(s)]) < bag_size) for s in subjects])
            xb = torch.from_numpy(data[selected.reshape(-1)].astype(np.float32)).to(device)
            targets = torch.from_numpy(y[subjects]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, representation, auxiliary = model.training_outputs(xb, bag_size, arm)
            ce = F.cross_entropy(logits, targets)
            contrast = supervised_contrastive(representation, targets) if arm == "masked_contrastive" else ce.new_zeros(())
            loss = ce + .2 * auxiliary + .1 * contrast
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append([float(ce.detach()), float(auxiliary.detach()), float(contrast.detach())])
        _, metrics = evaluate(model, data, val, windows, y, device)
        score = (metrics["macro_f1"], metrics["balanced_accuracy"], -metrics["log_loss"])
        history.append({"epoch": epoch, "train_losses": np.mean(losses, axis=0).tolist(), "validation": metrics})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale >= 8:
            print(f"fold={fold} arm={arm} epoch={epoch} val_f1={metrics['macro_f1']:.3f} best={best_score[0]:.3f} stale={stale}", flush=True)
        if stale >= 8:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    probabilities, metrics = evaluate(model, data, test, windows, y, device)
    print(f"RESULT fold={fold} arm={arm} epoch={best_epoch} test_f1={metrics['macro_f1']:.3f} test_ftd={metrics['recall'][1]:.3f}", flush=True)
    rows = [{"fold": fold, "arm": arm, "subject_id": participants[int(s)]["subject_id"],
             "true": CLASSES[int(y[s])], "predicted": CLASSES[int(np.argmax(p))],
             "p_AD": float(p[0]), "p_FTD": float(p[1]), "p_HC": float(p[2])}
            for s, p in zip(test, probabilities)]
    result = {"fold": fold, "arm": arm, "seed": seed + fold, "train_subjects": train.tolist(),
              "validation_subjects": val.tolist(), "test_subjects": test.tolist(),
              "best_epoch": best_epoch, "validation": history[best_epoch - 1]["validation"],
              "test": metrics, "history": history, "predictions": rows}
    (out / f"fold_{fold:02d}_{arm}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate(results: list[dict], participants: list[dict], out: Path) -> None:
    prediction_rows = [row for result in results for row in result["predictions"]]
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "arm", "subject_id", "true", "predicted", "p_AD", "p_FTD", "p_HC"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    summary = {}
    for arm in ARMS:
        rows = [r for r in prediction_rows if r["arm"] == arm]
        if len(rows) != len(participants):
            continue
        if len({r["subject_id"] for r in rows}) != len(participants):
            raise ValueError("An outer-test participant is missing or duplicated")
        true = np.array([CLASSES.index(r["true"]) for r in rows])
        probabilities = np.array([[r[f"p_{name}"] for name in CLASSES] for r in rows])
        pred = probabilities.argmax(axis=1)
        summary[arm] = {
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "recall": dict(zip(CLASSES, [float(v) for v in recall_score(true, pred, labels=[0, 1, 2], average=None, zero_division=0)])),
            "confusion_matrix": confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
            "log_loss": float(log_loss(true, probabilities, labels=[0, 1, 2])),
            "best_epochs": [r["best_epoch"] for r in results if r["arm"] == arm],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--bag-size", type=int, default=6)
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--out", type=Path, default=HERE / "outputs/full")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    x = np.load(DATA / "log_mel.npy", mmap_mode="r")
    subject_index = np.load(DATA / "subject_index.npy")
    window_labels = np.load(DATA / "labels.npy")
    y = np.array([CLASSES.index(p["label"]) for p in participants])
    if not (x.shape == (6719, 19, 24, 13) and len(subject_index) == len(x) and np.array_equal(window_labels, y[subject_index])):
        raise ValueError("Prepared data shape or labels differ from expected protocol")
    folds = splits_from_gntf(participants)
    protocol = {"input": str(DATA / "log_mel.npy"), "input_shape": list(x.shape), "classes": CLASSES,
                "outer_split": "same participants as G-NTF repeat 0", "inner_validation": "20% of outer training participants, stratified by label and source when feasible",
                "selection": "best participant-level validation macro F1, then balanced accuracy, then log loss",
                "arms": args.arms, "epochs_max": args.epochs, "bag_size": args.bag_size, "seed": args.seed,
                "loss": "CE + 0.2 masked latent SmoothL1 + 0.1 supervised contrastive (terms included by arm)",
                "mask": "one contiguous span of 3/13 frames per window", "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
                "caveat": "wav2vec-inspired on cached log-mel frames; no raw waveform or quantizer"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for fold, (train, test) in enumerate(folds[:args.fold_limit]):
        for arm in args.arms:
            saved = args.out / f"fold_{fold:02d}_{arm}.json"
            if saved.exists():
                results.append(json.loads(saved.read_text(encoding="utf-8")))
                continue
            results.append(train_one(arm, fold, args.seed, x, subject_index, participants, y, train, test,
                                     args.epochs, args.bag_size, args.out))
    aggregate(results, participants, args.out)


if __name__ == "__main__":
    main()
