from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "waveform_tucker_before_stft_2026-11-05" / "prepared"
MODERATE = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "raw_cnn_svm_fusion_refit"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
import train_raw_1d_cnn as windowcnn  # noqa: E402
import train_raw_1d_cnn_mil as milcnn  # noqa: E402
from evaluate_site_normalized import direct_model, site_normalize  # noqa: E402


def train_window_model(
    x: np.ndarray,
    y: np.ndarray,
    subject_index: np.ndarray,
    subject_labels: np.ndarray,
    subjects: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[windowcnn.RawEEG1DCNN, np.ndarray]:
    windowcnn.seed_everything(seed)
    indexes = np.flatnonzero(np.isin(subject_index, subjects))
    scale = windowcnn.training_channel_scale(x, indexes)
    counts_by_subject = np.bincount(subject_index[indexes], minlength=len(subject_labels))
    weights = np.asarray(
        [1.0 / max(counts_by_subject[int(subject_index[index])], 1) for index in indexes]
    )
    loader = DataLoader(
        windowcnn.WindowDataset(x, y, indexes, scale, augment=True),
        batch_size=windowcnn.BATCH_SIZE,
        sampler=WeightedRandomSampler(weights, num_samples=len(indexes), replacement=True),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    participant_counts = np.bincount(subject_labels[subjects], minlength=3).astype(np.float32)
    class_weights = participant_counts.sum() / np.maximum(3.0 * participant_counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    model = windowcnn.RawEEG1DCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=windowcnn.LEARNING_RATE, weight_decay=windowcnn.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=windowcnn.MAX_EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, labels, _ in loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(values), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        print(f"window refit epoch={epoch}/{epochs} loss={np.mean(losses):.4f}", flush=True)
    torch.save(
        {"model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "channel_scale": scale},
        OUT / "window_cnn_refit.pt",
    )
    return model, scale


def train_mil_model(
    x: np.ndarray,
    subject_index: np.ndarray,
    subject_labels: np.ndarray,
    subjects: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[milcnn.ParticipantMILCNN, np.ndarray]:
    milcnn.seed_everything(seed)
    indexes = np.flatnonzero(np.isin(subject_index, subjects))
    scale = windowcnn.training_channel_scale(x, indexes)
    dataset = milcnn.ParticipantBagDataset(
        x, subject_index, subject_labels, subjects, scale, milcnn.BAG_SIZE
    )
    class_counts = np.bincount(subject_labels[subjects], minlength=3)
    weights = np.asarray([1.0 / class_counts[subject_labels[s]] for s in subjects])
    loader = DataLoader(
        dataset,
        batch_size=milcnn.BATCH_SIZE,
        sampler=WeightedRandomSampler(weights, num_samples=len(subjects) * 4, replacement=True),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    model = milcnn.ParticipantMILCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=milcnn.MAX_EPOCHS, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
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
        print(f"MIL refit epoch={epoch}/{epochs} loss={np.mean(losses):.4f}", flush=True)
    torch.save(
        {"model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "channel_scale": scale},
        OUT / "mil_cnn_refit.pt",
    )
    return model, scale


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.load(PREPARED / "waveforms.npy", mmap_mode="r")
    y = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads(
        (PREPARED / "waveform_manifest.json").read_text(encoding="utf-8")
    )["participants"]
    splits = eeg.make_subject_splits(participants)
    development = np.asarray(splits["train"] + splits["val"], dtype=np.int64)
    test = np.asarray(splits["test"], dtype=np.int64)
    subject_labels = np.asarray([eeg.LABEL_ID[row["label"]] for row in participants])
    test_windows = np.flatnonzero(np.isin(subject_index, test))

    window_selection = json.loads(
        (HERE / "outputs" / "raw_1d_cnn_selected" / "metrics.json").read_text(encoding="utf-8")
    )
    mil_selection = json.loads(
        (HERE / "outputs" / "raw_1d_cnn_mil" / "metrics.json").read_text(encoding="utf-8")
    )
    fusion_selection = json.loads(
        (HERE / "outputs" / "raw_cnn_svm_fusion" / "metrics.json").read_text(encoding="utf-8")
    )["selected_validation"]

    window_model, window_scale = train_window_model(
        x,
        y,
        subject_index,
        subject_labels,
        development,
        int(window_selection["selected_best_epoch"]),
        int(window_selection["selected_seed"]),
        device,
    )
    window_probabilities = windowcnn.aggregate_participants(
        windowcnn.predict_windows(window_model, x, y, test_windows, window_scale, device),
        test_windows,
        subject_index,
        test,
    )

    mil_model, mil_scale = train_mil_model(
        x,
        subject_index,
        subject_labels,
        development,
        int(mil_selection["selected_epoch"]),
        int(mil_selection["selected_seed"]),
        device,
    )
    mil_probabilities, attention = milcnn.participant_probabilities(
        mil_model, x, subject_index, test, mil_scale, device
    )

    tensor = np.load(MODERATE / "log_mel.npy", mmap_mode="r")
    labels_windows = np.load(MODERATE / "labels.npy", mmap_mode="r")
    moderate_subject = np.load(MODERATE / "subject_index.npy", mmap_mode="r")
    moderate_participants = json.loads(
        (MODERATE / "manifest.json").read_text(encoding="utf-8")
    )["participants"]
    features, labels = eeg.subject_features(
        tensor, labels_windows, moderate_subject, moderate_participants
    )
    flat = features.reshape(len(features), -1)
    normalized = site_normalize(flat, moderate_participants, development)
    svm = direct_model({"pca": 20, "C": 0.1, "kernel": "linear"})
    svm.fit(normalized[development], labels[development])
    svm_probabilities = softmax(
        svm.decision_function(normalized[test]) / fusion_selection["svm_temperature"], axis=1
    )
    window_scaled = softmax(
        np.log(np.maximum(window_probabilities, 1e-9)) / fusion_selection["window_temperature"],
        axis=1,
    )
    mil_scaled = softmax(
        np.log(np.maximum(mil_probabilities, 1e-9)) / fusion_selection["mil_temperature"],
        axis=1,
    )
    fusion_probabilities = (
        fusion_selection["svm_weight"] * svm_probabilities
        + fusion_selection["window_weight"] * window_scaled
        + fusion_selection["mil_weight"] * mil_scaled
    )
    window_metrics = eeg.metrics(subject_labels[test], window_probabilities.argmax(axis=1))
    mil_metrics = eeg.metrics(subject_labels[test], mil_probabilities.argmax(axis=1))
    fusion_metrics = eeg.metrics(subject_labels[test], fusion_probabilities.argmax(axis=1))

    payload = {
        "protocol": {
            "development_participants": len(development),
            "test_participants": len(test),
            "window_epochs": int(window_selection["selected_best_epoch"]),
            "mil_epochs": int(mil_selection["selected_epoch"]),
            "fusion_configuration": fusion_selection,
            "test_used_for_selection": False,
        },
        "refit_window_cnn": window_metrics,
        "refit_mil_cnn": mil_metrics,
        "refit_fusion": fusion_metrics,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "test_probabilities.npz",
        test_indices=test,
        test_true=subject_labels[test],
        window_probabilities=window_probabilities,
        mil_probabilities=mil_probabilities,
        fusion_probabilities=fusion_probabilities,
    )
    (OUT / "test_attention.json").write_text(json.dumps(attention, indent=2), encoding="utf-8")

    rows = [window_metrics, mil_metrics, fusion_metrics]
    names = ["Refit window CNN", "Refit MIL CNN", "Refit fusion"]
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
    ax.set_xticks(positions, names)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Refit on all 147 development participants")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(fusion_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Refit raw CNN and SVM fusion")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    recalls = {name: fusion_metrics["per_class"][name]["recall"] for name in eeg.CLASSES}
    report = [
        "# Raw CNN and SVM refit",
        "",
        "After architecture, epoch counts, temperatures and fusion weights were fixed from validation, "
        "the two CNNs and the SVM were refitted using all 147 development participants. The 27 test "
        "participants did not influence these choices.",
        "",
        "## Held-out fusion result",
        "",
        f"- Accuracy: {fusion_metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {fusion_metrics['balanced_accuracy']:.4f}",
        f"- Macro F1: {fusion_metrics['macro_f1']:.4f}",
        f"- Recall AD / FTD / HC: {recalls['AD']:.4f} / {recalls['FTD']:.4f} / {recalls['HC']:.4f}",
        f"- Confusion matrix: {fusion_metrics['confusion_matrix']}",
        "",
        f"Window CNN alone: {window_metrics['accuracy']:.4f} accuracy, {window_metrics['macro_f1']:.4f} macro F1.",
        f"MIL CNN alone: {mil_metrics['accuracy']:.4f} accuracy, {mil_metrics['macro_f1']:.4f} macro F1.",
        "",
        "The test cohort has been inspected in earlier experiments, so this remains exploratory rather than a fresh blind estimate.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
