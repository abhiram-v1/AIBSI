from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "waveform_tucker_before_stft_2026-11-05" / "prepared"
SOURCE = HERE / "outputs" / "raw_1d_cnn"
OUT = HERE / "outputs" / "raw_1d_cnn_selected"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
import train_raw_1d_cnn as rawcnn  # noqa: E402


def plot_confusion(metrics: dict) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Validation-selected raw 1D CNN")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_comparison(selected: dict, baseline: dict, mfcc: dict, ensemble: dict) -> None:
    rows = [baseline, mfcc, ensemble, selected]
    labels = ["Log-mel SVM", "MFCC SVM", "CNN 3-seed average", "CNN selected seed"]
    measures = ["accuracy", "balanced_accuracy", "macro_f1"]
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for offset, measure in enumerate(measures):
        bars = ax.bar(
            positions + (offset - 1) * 0.23,
            [row[measure] for row in rows],
            0.23,
            label=measure.replace("_", " ").title(),
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylim=(0, 1), ylabel="Held-out participant score", title="Raw 1D CNN result and representation baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
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
    validation_subjects = np.asarray(splits["val"], dtype=np.int64)
    test_subjects = np.asarray(splits["test"], dtype=np.int64)
    subject_labels = np.asarray([eeg.LABEL_ID[row["label"]] for row in participants], dtype=np.int64)

    seed_rows = []
    val_probabilities_all = []
    test_probabilities_all = []
    for seed in rawcnn.SEEDS:
        checkpoint = torch.load(
            SOURCE / f"model_seed_{seed}.pt", map_location="cpu", weights_only=False
        )
        model = rawcnn.RawEEG1DCNN().to(device)
        model.load_state_dict(checkpoint["model_state"])
        probabilities_by_split = {}
        metrics_by_split = {}
        for name, subjects in [("validation", validation_subjects), ("test", test_subjects)]:
            windows = np.flatnonzero(np.isin(subject_index, subjects))
            window_probabilities = rawcnn.predict_windows(
                model, x, y, windows, checkpoint["channel_scale"], device
            )
            probabilities = rawcnn.aggregate_participants(
                window_probabilities, windows, subject_index, subjects
            )
            probabilities_by_split[name] = probabilities
            metrics_by_split[name] = eeg.metrics(subject_labels[subjects], probabilities.argmax(axis=1))
        val_probabilities_all.append(probabilities_by_split["validation"])
        test_probabilities_all.append(probabilities_by_split["test"])
        seed_rows.append(
            {
                "seed": seed,
                "best_epoch": checkpoint["best_epoch"],
                "validation": metrics_by_split["validation"],
                "test": metrics_by_split["test"],
            }
        )

    selected = max(
        seed_rows,
        key=lambda row: (
            row["validation"]["macro_f1"],
            row["validation"]["balanced_accuracy"],
            row["validation"]["accuracy"],
        ),
    )
    selected_index = next(i for i, row in enumerate(seed_rows) if row["seed"] == selected["seed"])
    selected_val_probabilities = val_probabilities_all[selected_index]
    selected_test_probabilities = test_probabilities_all[selected_index]
    ensemble_val_probabilities = np.mean(val_probabilities_all, axis=0)
    ensemble_test_probabilities = np.mean(test_probabilities_all, axis=0)
    ensemble_validation = eeg.metrics(
        subject_labels[validation_subjects], ensemble_val_probabilities.argmax(axis=1)
    )
    ensemble_test = eeg.metrics(subject_labels[test_subjects], ensemble_test_probabilities.argmax(axis=1))

    baseline = json.loads(
        (HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8")
    )["test"]
    mfcc = json.loads(
        (HERE / "outputs" / "mfcc_svm" / "metrics.json").read_text(encoding="utf-8")
    )["test"]
    payload = {
        "selection_rule": "highest participant-level validation macro F1 among three predetermined seeds",
        "test_used_for_selection": False,
        "selected_seed": selected["seed"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_validation": selected["validation"],
        "selected_test": selected["test"],
        "ensemble_validation": ensemble_validation,
        "ensemble_test": ensemble_test,
        "all_seeds": seed_rows,
        "log_mel_svm_reference": baseline,
        "mfcc_svm_reference": mfcc,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "participant_probabilities.npz",
        selected_seed=selected["seed"],
        validation_indices=validation_subjects,
        validation_true=subject_labels[validation_subjects],
        selected_validation_probabilities=selected_val_probabilities,
        ensemble_validation_probabilities=ensemble_val_probabilities,
        test_indices=test_subjects,
        test_true=subject_labels[test_subjects],
        selected_test_probabilities=selected_test_probabilities,
        ensemble_test_probabilities=ensemble_test_probabilities,
    )
    plot_confusion(selected["test"])
    plot_comparison(selected["test"], baseline, mfcc, ensemble_test)

    recalls = {name: selected["test"]["per_class"][name]["recall"] for name in eeg.CLASSES}
    report = [
        "# Validation-selected raw 1D CNN",
        "",
        "A compact 75,463-parameter 1D CNN received raw 19-channel, eight-second EEG windows at 128 Hz. "
        "No STFT, mel filterbank or MFCC transform was used. The checkpoint was selected only by "
        "participant-level validation macro F1 among three predetermined random seeds.",
        "",
        "## Selection",
        "",
        f"- Selected seed: {selected['seed']}",
        f"- Selected epoch: {selected['best_epoch']}",
        f"- Validation macro F1: {selected['validation']['macro_f1']:.4f}",
        f"- Validation balanced accuracy: {selected['validation']['balanced_accuracy']:.4f}",
        "",
        "## Held-out participant result",
        "",
        f"- Accuracy: {selected['test']['accuracy']:.4f}",
        f"- Balanced accuracy: {selected['test']['balanced_accuracy']:.4f}",
        f"- Macro F1: {selected['test']['macro_f1']:.4f}",
        f"- Recall AD / FTD / HC: {recalls['AD']:.4f} / {recalls['FTD']:.4f} / {recalls['HC']:.4f}",
        f"- Confusion matrix: {selected['test']['confusion_matrix']}",
        "",
        "## Controls on the same fixed split",
        "",
        f"- Three-seed probability average: accuracy {ensemble_test['accuracy']:.4f}, macro F1 {ensemble_test['macro_f1']:.4f}",
        f"- Log-mel SVM: accuracy {baseline['accuracy']:.4f}, macro F1 {baseline['macro_f1']:.4f}",
        f"- MFCC SVM: accuracy {mfcc['accuracy']:.4f}, macro F1 {mfcc['macro_f1']:.4f}",
        "",
        "This remains an exploratory fixed-split result. Repeated nested participant-level validation is required before treating it as the expected generalization score.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "ensemble_test": ensemble_test}, indent=2))


if __name__ == "__main__":
    main()
