from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_openneuro_native_1d_cnn as native


OUT = native.OUT / "refit"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, owners, participants = native.prepare()
    subject_labels = np.asarray([native.CLASSES.index(row["label"]) for row in participants])
    train, validation, test = native.split_subjects(subject_labels)
    development = np.concatenate([train, validation])
    selection = json.loads((native.OUT / "metrics.json").read_text(encoding="utf-8"))
    seed = int(selection["selected_seed"])
    epochs = int(selection["selected_epoch"])
    native.seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indexes = np.flatnonzero(np.isin(owners, development))
    test_indexes = np.flatnonzero(np.isin(owners, test))
    scale = native.channel_scale(x, indexes)
    windows_per_subject = np.bincount(owners[indexes], minlength=len(participants))
    weights = np.asarray([1 / windows_per_subject[owners[index]] for index in indexes])
    loader = DataLoader(
        native.WindowDataset(x, y, indexes, scale, True),
        batch_size=native.BATCH_SIZE,
        sampler=WeightedRandomSampler(weights, len(indexes), replacement=True),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    counts = np.bincount(subject_labels[development], minlength=3).astype(np.float32)
    class_weights = counts.sum() / (3 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    model = native.Native500HzCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=native.MAX_EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for values, labels in loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(values), labels)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        scheduler.step(); history.append(float(np.mean(losses)))
        print(f"native refit epoch={epoch}/{epochs} loss={history[-1]:.4f}", flush=True)
    probability = native.aggregate(
        native.predict(model, x, y, test_indexes, scale, device), test_indexes, owners, test
    )
    result = native.metrics(subject_labels[test], probability)
    payload = {
        "protocol": {
            "development_participants": len(development),
            "test_participants": len(test),
            "sampling_hz": native.FS,
            "electrodes": len(native.CHANNELS),
            "interpolation": False,
            "resampling": False,
            "selected_seed": seed,
            "fixed_epochs": epochs,
            "test_used_for_selection": False,
        },
        "test": result,
        "training_loss": history,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "test_probabilities.npz",
        test_indices=test,
        test_true=subject_labels[test],
        test_probabilities=probability,
    )
    torch.save(
        {"model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "channel_scale": scale, "seed": seed, "epochs": epochs},
        OUT / "model.pt",
    )
    native.confusion_plot(result["confusion_matrix"], OUT / "confusion_matrix.png", "OpenNeuro native 500 Hz CNN refit")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(np.arange(1, epochs + 1), history, marker="o", markersize=3)
    ax.set(xlabel="Epoch", ylabel="Training loss", title="Native OpenNeuro CNN refit")
    ax.grid(alpha=0.25); fig.savefig(OUT / "training_loss.png", dpi=180); plt.close(fig)
    per_class = result["per_class"]
    report = [
        "# Native OpenNeuro 500 Hz CNN refit", "",
        "The validation-selected configuration was refitted on all 74 development participants. "
        "Only the 88-person OpenNeuro cohort was used; montage interpolation and resampling were disabled.", "",
        f"- Accuracy: {result['accuracy']:.4f}",
        f"- Balanced accuracy: {result['balanced_accuracy']:.4f}",
        f"- Macro F1: {result['macro_f1']:.4f}",
        f"- Recall AD / FTD / HC: {per_class['AD']['recall']:.4f} / {per_class['FTD']['recall']:.4f} / {per_class['HC']['recall']:.4f}",
        f"- Confusion matrix: {result['confusion_matrix']}", "",
        "The test set contains only 14 participants (6 AD, 4 FTD, 4 HC), so uncertainty is high.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
