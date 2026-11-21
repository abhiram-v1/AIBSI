from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_eegpt_attention as eegpt
import train_openneuro_native_1d_cnn as native


OUT = native.OUT / "eegpt_blend"


def temperature(probability: np.ndarray, value: float) -> np.ndarray:
    return softmax(np.log(np.maximum(probability, 1e-9)) / value, axis=1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    OUT.mkdir(parents=True, exist_ok=True)
    seed = 20261127
    eegpt.seed_everything(seed)
    native_x, _, _, native_participants = native.prepare()
    del native_x
    subject_labels = np.asarray([native.CLASSES.index(row["label"]) for row in native_participants])
    train_subjects, val_subjects, test_subjects = native.split_subjects(subject_labels)

    x = np.load(eegpt.PREPARED / "raw_windows.npy", mmap_mode="r")
    y = np.load(eegpt.PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(eegpt.PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((eegpt.PREPARED / "manifest.json").read_text(encoding="utf-8"))
    if any(row["source"] != "ds004504" for row in manifest["participants"][:88]):
        raise RuntimeError("The first 88 EEGPT participants are not the OpenNeuro cohort")
    train_data = eegpt.ParticipantDataset(x, y, subject_index, train_subjects.tolist(), 8, True)
    val_data = eegpt.ParticipantDataset(x, y, subject_index, val_subjects.tolist(), None, False)
    test_data = eegpt.ParticipantDataset(x, y, subject_index, test_subjects.tolist(), None, False)
    counts = Counter(train_data.labels.values())
    weights = [1.0 / counts[train_data.labels[int(subject)]] for subject in train_data.subjects]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_data) * 2, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(train_data, batch_size=4, sampler=sampler, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    model = eegpt.EEGPTAttention(eegpt.build_encoder(), manifest["channels"], extra_blocks=0).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.06)
    scaler = torch.amp.GradScaler("cuda")
    best_f1 = -1.0; best_bacc = -1.0; best_epoch = 0; global_epoch = 0; history = []
    best_path = OUT / "best_eegpt_model.pt"
    for stage, epochs, fine_tune in [("frozen_probe", 8, False), ("last_two_blocks", 30, True)]:
        eegpt.set_stage(model, fine_tune)
        optimizer = eegpt.make_optimizer(model, fine_tune)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-6)
        stale = 0
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1; model.train(); total_loss = 0.0; seen = 0
            for bags, target, _ in loader:
                bags = bags.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits, _ = model(bags); loss = criterion(logits, target)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer); scaler.update()
                total_loss += float(loss.item()) * len(target); seen += len(target)
            scheduler.step()
            val_probability, val_true, _ = eegpt.predict(model, val_data, device)
            prediction = val_probability.argmax(1)
            val_f1 = float(f1_score(val_true, prediction, average="macro", zero_division=0))
            val_bacc = float(balanced_accuracy_score(val_true, prediction))
            history.append({"epoch": global_epoch, "stage": stage, "loss": total_loss / seen, "val_macro_f1": val_f1, "val_balanced_accuracy": val_bacc})
            print(f"epoch={global_epoch:02d} stage={stage} loss={total_loss/seen:.4f} val_f1={val_f1:.4f} val_bacc={val_bacc:.4f}", flush=True)
            if (val_f1, val_bacc) > (best_f1, best_bacc):
                best_f1, best_bacc, best_epoch, stale = val_f1, val_bacc, global_epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": global_epoch}, best_path)
            else:
                stale += 1
                if fine_tune and stale >= 8:
                    break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model_state"])
    eegpt_val, val_true, _ = eegpt.predict(model, val_data, device)
    eegpt_test, test_true, _ = eegpt.predict(model, test_data, device)

    cnn_file = np.load(native.OUT / "probabilities.npz")
    cnn_metrics = json.loads((native.OUT / "metrics.json").read_text(encoding="utf-8"))
    selected_seed = int(cnn_metrics["selected_seed"])
    seed_position = native.SEEDS.index(selected_seed)
    if not np.array_equal(cnn_file["validation_indices"], val_subjects) or not np.array_equal(cnn_file["test_indices"], test_subjects):
        raise RuntimeError("CNN and EEGPT participant splits do not align")
    cnn_val = cnn_file["validation_probabilities"][seed_position]
    cnn_test = cnn_file["test_probabilities"][seed_position]

    temperatures = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    candidates = []
    for cnn_temperature in temperatures:
        scaled_cnn = temperature(cnn_val, cnn_temperature)
        for eegpt_temperature in temperatures:
            scaled_eegpt = temperature(eegpt_val, eegpt_temperature)
            for cnn_weight in np.linspace(0, 1, 21):
                probability = cnn_weight * scaled_cnn + (1 - cnn_weight) * scaled_eegpt
                prediction = probability.argmax(1)
                candidates.append({
                    "cnn_temperature": cnn_temperature, "eegpt_temperature": eegpt_temperature,
                    "cnn_weight": float(cnn_weight), "eegpt_weight": float(1 - cnn_weight),
                    "macro_f1": float(f1_score(val_true, prediction, average="macro", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(val_true, prediction)),
                    "accuracy": float(np.mean(prediction == val_true)),
                })
    selected = max(candidates, key=lambda row: (row["macro_f1"], row["balanced_accuracy"], row["accuracy"], -abs(row["cnn_weight"] - 0.5)))
    blend_val = selected["cnn_weight"] * temperature(cnn_val, selected["cnn_temperature"]) + selected["eegpt_weight"] * temperature(eegpt_val, selected["eegpt_temperature"])
    blend_test = selected["cnn_weight"] * temperature(cnn_test, selected["cnn_temperature"]) + selected["eegpt_weight"] * temperature(eegpt_test, selected["eegpt_temperature"])
    result = {
        "protocol": {"cohort": "OpenNeuro ds004504 only", "train": 60, "validation": 14, "test": 14, "cnn_sampling_hz": 500, "eegpt_required_sampling_hz": 256, "selection_uses_test": False},
        "eegpt_best_epoch": best_epoch,
        "eegpt_validation": eegpt.score(val_true, eegpt_val.argmax(1)),
        "eegpt_test": eegpt.score(test_true, eegpt_test.argmax(1)),
        "cnn_validation": native.metrics(val_true, cnn_val),
        "cnn_test": native.metrics(test_true, cnn_test),
        "selected_blend": selected,
        "blend_validation": native.metrics(val_true, blend_val),
        "blend_test": native.metrics(test_true, blend_test),
        "history": history,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "probabilities.npz", validation_indices=val_subjects, validation_true=val_true, cnn_validation=cnn_val, eegpt_validation=eegpt_val, blend_validation=blend_val, test_indices=test_subjects, test_true=test_true, cnn_test=cnn_test, eegpt_test=eegpt_test, blend_test=blend_test)
    names = ["Native CNN", "EEGPT", "CNN + EEGPT"]
    rows = [result["cnn_test"], result["eegpt_test"], result["blend_test"]]
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True); positions = np.arange(3)
    for offset, measure in enumerate(["accuracy", "balanced_accuracy", "macro_f1"]):
        bars = ax.bar(positions + (offset - 1) * 0.23, [row[measure] for row in rows], 0.23, label=measure.replace("_", " ").title()); ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(positions, names); ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="OpenNeuro-only CNN and EEGPT blend"); ax.grid(axis="y", alpha=0.25); ax.legend(); fig.savefig(OUT / "comparison.png", dpi=180); plt.close(fig)
    native.confusion_plot(result["blend_test"]["confusion_matrix"], OUT / "blend_confusion_matrix.png", "OpenNeuro CNN + EEGPT blend")
    recalls = result["blend_test"]["per_class"]
    report = ["# OpenNeuro-only native CNN + EEGPT blend", "", "EEGPT was trained only on the same 60 OpenNeuro training participants. Blend temperature and weight were selected only on the same 14 validation participants, then frozen for the 14-person test set.", "", f"- CNN weight: {selected['cnn_weight']:.2f}", f"- EEGPT weight: {selected['eegpt_weight']:.2f}", f"- Test accuracy: {result['blend_test']['accuracy']:.4f}", f"- Test balanced accuracy: {result['blend_test']['balanced_accuracy']:.4f}", f"- Test macro F1: {result['blend_test']['macro_f1']:.4f}", f"- Recall AD / FTD / HC: {recalls['AD']['recall']:.4f} / {recalls['FTD']['recall']:.4f} / {recalls['HC']['recall']:.4f}", "", "The validation and test sets each contain only 14 participants, so this remains exploratory."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["protocol", "eegpt_best_epoch", "eegpt_validation", "eegpt_test", "cnn_validation", "cnn_test", "selected_blend", "blend_validation", "blend_test"]}, indent=2))


if __name__ == "__main__":
    main()
