from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "waveform_tucker_before_stft_2026-11-05" / "prepared"
OUT = HERE / "outputs" / "raw_1d_cnn_cv_ensemble"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
import refit_raw_cnn_fusion as refit  # noqa: E402
import train_raw_1d_cnn as cnn  # noqa: E402


def subject_probabilities(model, x, y, subject_index, subjects, scale, device):
    windows = np.flatnonzero(np.isin(subject_index, subjects))
    return cnn.aggregate_participants(
        cnn.predict_windows(model, x, y, windows, scale, device),
        windows,
        subject_index,
        subjects,
    )


def confusion_figure(matrix: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), eeg.CLASSES)
    ax.set_yticks(range(3), eeg.CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    selection = json.loads(
        (HERE / "outputs" / "raw_1d_cnn_selected" / "metrics.json").read_text(encoding="utf-8")
    )
    epochs = int(selection["selected_best_epoch"])

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261122)
    oof = np.zeros((len(development), 3), dtype=np.float32)
    test_predictions = []
    folds = []
    for fold, (train_position, val_position) in enumerate(
        splitter.split(development, subject_labels[development]), start=1
    ):
        train_subjects = development[train_position]
        val_subjects = development[val_position]
        fold_dir = OUT / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        refit.OUT = fold_dir
        seed = 20261122 + fold
        model, scale = refit.train_window_model(
            x, y, subject_index, subject_labels, train_subjects, epochs, seed, device
        )
        val_probability = subject_probabilities(
            model, x, y, subject_index, val_subjects, scale, device
        )
        test_probability = subject_probabilities(
            model, x, y, subject_index, test, scale, device
        )
        oof[val_position] = val_probability
        test_predictions.append(test_probability)
        fold_metrics = eeg.metrics(subject_labels[val_subjects], val_probability.argmax(axis=1))
        folds.append({"fold": fold, "train": len(train_subjects), "validation": len(val_subjects), **fold_metrics})
        print(
            f"fold={fold}/5 oof_f1={fold_metrics['macro_f1']:.4f} "
            f"oof_bacc={fold_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    test_probability = np.mean(test_predictions, axis=0)
    oof_metrics = eeg.metrics(subject_labels[development], oof.argmax(axis=1))
    test_metrics = eeg.metrics(subject_labels[test], test_probability.argmax(axis=1))
    payload = {
        "protocol": {
            "full_participants": len(participants),
            "development_participants": len(development),
            "test_participants": len(test),
            "folds": 5,
            "fixed_epochs_per_fold": epochs,
            "test_used_for_selection": False,
        },
        "fold_validation": folds,
        "out_of_fold_development": oof_metrics,
        "held_out_test_ensemble": test_metrics,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "probabilities.npz",
        development_indices=development,
        development_true=subject_labels[development],
        out_of_fold_probabilities=oof,
        test_indices=test,
        test_true=subject_labels[test],
        test_probabilities=test_probability,
        fold_test_probabilities=np.asarray(test_predictions),
    )
    confusion_figure(
        np.asarray(oof_metrics["confusion_matrix"]),
        OUT / "oof_confusion_matrix.png",
        "Five-fold out-of-fold development predictions",
    )
    confusion_figure(
        np.asarray(test_metrics["confusion_matrix"]),
        OUT / "test_confusion_matrix.png",
        "Five-CNN ensemble on held-out participants",
    )

    measures = ["accuracy", "balanced_accuracy", "macro_f1"]
    x_axis = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for offset, (name, metrics) in enumerate(
        [("OOF development", oof_metrics), ("Held-out test", test_metrics)]
    ):
        bars = ax.bar(x_axis + (offset - 0.5) * 0.34, [metrics[m] for m in measures], 0.34, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_xticks(x_axis, [m.replace("_", " ").title() for m in measures])
    ax.set(ylim=(0, 1), ylabel="Participant-level score", title="Five-fold raw 1D CNN ensemble")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "validation_test_comparison.png", dpi=180)
    plt.close(fig)

    recalls = test_metrics["per_class"]
    report = [
        "# Five-fold raw 1D CNN ensemble",
        "",
        "Five CNNs were trained on participant-stratified folds of the 147-person development set. "
        "The epoch count and architecture were frozen before this run. Out-of-fold predictions cover "
        "each development participant once; the untouched 27-person test probabilities are averaged.",
        "",
        "## Results",
        "",
        f"- OOF development: accuracy {oof_metrics['accuracy']:.4f}, balanced accuracy "
        f"{oof_metrics['balanced_accuracy']:.4f}, macro F1 {oof_metrics['macro_f1']:.4f}",
        f"- Held-out test: accuracy {test_metrics['accuracy']:.4f}, balanced accuracy "
        f"{test_metrics['balanced_accuracy']:.4f}, macro F1 {test_metrics['macro_f1']:.4f}",
        f"- Test recall AD / FTD / HC: {recalls['AD']['recall']:.4f} / "
        f"{recalls['FTD']['recall']:.4f} / {recalls['HC']['recall']:.4f}",
        "",
        "The test set has been inspected in prior experiments, so it is an exploratory held-out result, "
        "not a pristine external validation estimate.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
