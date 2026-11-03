from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent

NAVY = "#17324D"
BLUE = "#3977B8"
TEAL = "#168C8C"
ORANGE = "#D98236"
RED = "#C44E52"
GREEN = "#4C956C"
GREY = "#6B7280"
LIGHT = "#E8EEF4"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C9D3DD",
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "text.color": NAVY,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        value = bar.get_width() if bar.get_width() != 0 else bar.get_height()
        if bar.get_width() != 0:
            ax.text(
                bar.get_width() + 0.7,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
                fontsize=9,
                color=NAVY,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                fmt.format(value),
                ha="center",
                fontsize=9,
                color=NAVY,
            )


def original_tensor_progress() -> None:
    groups = {
        "V1 · exploratory spectral tensors": [
            ("PCA/NMF + logistic", 54.1),
            ("Tucker", 53.6),
            ("CP", 52.2),
            ("Graph NCP", 51.7),
        ],
        "V2 · corrected nested tensor study": [
            ("Tucker/HOSVD", 56.0),
            ("TT + spectral", 54.1),
            ("Graph-smoothed TT", 53.1),
            ("Spectral control", 51.7),
            ("Window TT", 51.2),
            ("Subject TT", 50.2),
        ],
        "V3 · full-cohort waveform study": [
            ("Spectral control", 55.8),
            ("Waveform Tucker", 51.8),
            ("Waveform TT", 50.9),
            ("Supervised TT", 49.9),
        ],
        "V4 · bounded classifier tuning": [
            ("RBF SVM", 59.7),
            ("Random forest", 58.3),
            ("XGBoost", 57.6),
            ("Logistic regression", 54.6),
            ("Supervised TT", 50.9),
        ],
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, (title, rows) in zip(axes.flat, groups.items()):
        labels = [row[0] for row in rows][::-1]
        values = [row[1] for row in rows][::-1]
        colors = [TEAL if ("TT" in label or "Tucker" in label or "CP" in label) else BLUE for label in labels]
        bars = ax.barh(labels, values, color=colors, height=0.65)
        ax.axvline(33.3, color=GREY, linestyle="--", linewidth=1, label="Chance-level balanced accuracy")
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlim(30, 64)
        ax.grid(axis="x", alpha=0.2)
        label_bars(ax, bars)
    axes[1, 0].set_xlabel("Balanced accuracy (%)")
    axes[1, 1].set_xlabel("Balanced accuracy (%)")
    fig.suptitle("Original OpenNeuro cohort: tensor and classifier development", fontsize=18, fontweight="bold", x=0.5)
    fig.text(
        0.5,
        0.02,
        "Values summarize separate versioned experiments; cohort and protocol changed across versions, so bars are milestones—not a controlled learning curve.",
        ha="center",
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    finish(fig, "01_original_tensor_progress.png")


def dataset_growth() -> None:
    cohorts = ["Original OpenNeuro", "Combined with BrainLat"]
    ad = np.array([36, 71])
    ftd = np.array([23, 42])
    healthy = np.array([29, 61])
    totals = ad + ftd + healthy
    x = np.arange(len(cohorts))
    fig, ax = plt.subplots(figsize=(10, 7))
    b1 = ax.bar(x, ad, color=BLUE, width=0.58, label="AD")
    b2 = ax.bar(x, ftd, bottom=ad, color=ORANGE, width=0.58, label="FTD")
    b3 = ax.bar(x, healthy, bottom=ad + ftd, color=TEAL, width=0.58, label="Healthy")
    for bars, values in [(b1, ad), (b2, ftd), (b3, healthy)]:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    for index, total in enumerate(totals):
        ax.text(index, total + 4, f"Total {total}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x, cohorts)
    ax.set_ylim(0, 195)
    ax.set_ylabel("Usable participants")
    ax.set_title("Independent-participant growth after adding BrainLat", loc="left")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, loc="upper left")
    ax.text(
        0.0,
        -0.12,
        "The paper-matched binary experiment used only the original 88 participants. The later STFT, log-mel, tensor-SVM and EEGPT study used the combined 174-participant cohort.",
        transform=ax.transAxes,
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, "00_dataset_growth.png")


def binary_replication() -> None:
    models = [
        "Reference-paper SVM",
        "Tensor TT/Tucker SVM",
        "Direct EEG SVM",
        "Our final nested pipeline",
    ]
    accuracy = np.array([85.0, 68.6, 73.1, 75.0])
    macro_f1 = np.array([46.0, 64.9, 70.4, 71.8])
    balanced = np.array([50.0, 65.4, 71.2, 72.0])
    cn_recall = np.array([0.0, 56.3, 65.5, 63.2])
    y = np.arange(len(models))
    h = 0.19
    fig, ax = plt.subplots(figsize=(13, 7))
    for offset, values, color, label in [
        (-1.5 * h, accuracy, BLUE, "Accuracy"),
        (-0.5 * h, macro_f1, TEAL, "Macro F1"),
        (0.5 * h, balanced, ORANGE, "Balanced accuracy"),
        (1.5 * h, cn_recall, GREEN, "Healthy-control recall"),
    ]:
        bars = ax.barh(y + offset, values, height=h, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(value + 0.7, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    ax.set_title("Binary dementia vs healthy control: headline accuracy versus balanced evidence", loc="left")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncol=2, loc="lower right")
    ax.text(
        0.0,
        -0.17,
        "Reference-paper SVM predicted every test participant as dementia: 85% accuracy but 0% healthy-control recall. Our results use participant-level evaluation.",
        transform=ax.transAxes,
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, "02_binary_replication.png")


def combined_model_progress() -> None:
    rows = [
        ("STFT CNN-BiLSTM", 44.4, 40.9),
        ("STFT CNN", 48.1, 39.0),
        ("Earlier STFT SVM", 55.6, 52.5),
        ("Log-mel CNN-BiLSTM", 55.6, 48.9),
        ("Log-mel CNN", 59.3, 52.1),
        ("Attention CNN", 55.6, 49.7),
        ("Tensor Train + SVM", 51.9, 46.4),
        ("EEGPT", 63.0, 55.0),
        ("Site-normalized log-mel SVM", 66.7, 64.3),
        ("SVM + EEGPT blend", 70.4, 67.8),
    ]
    labels = [row[0] for row in rows]
    accuracy = np.array([row[1] for row in rows])
    macro_f1 = np.array([row[2] for row in rows])
    y = np.arange(len(rows))
    h = 0.36
    fig, ax = plt.subplots(figsize=(13, 9))
    b1 = ax.barh(y - h / 2, accuracy, height=h, color=BLUE, label="Accuracy")
    b2 = ax.barh(y + h / 2, macro_f1, height=h, color=TEAL, label="Macro F1")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(30, 76)
    ax.set_xlabel("Held-out participant score (%)")
    ax.set_title("Combined 174-participant cohort: model progression", loc="left")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    ax.text(
        0.0,
        -0.10,
        "Exploratory fixed split: 121 train, 26 validation, 27 test. These bars compare the saved test predictions, not independent publication-grade estimates.",
        transform=ax.transAxes,
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, "03_combined_model_progress.png")


def class_recall() -> None:
    models = ["Log-mel SVM", "Tensor Train + SVM", "EEGPT", "SVM + EEGPT"]
    recalls = np.array(
        [
            [72.7, 50.0, 70.0],
            [63.6, 16.7, 60.0],
            [72.7, 16.7, 80.0],
            [72.7, 50.0, 80.0],
        ]
    )
    x = np.arange(len(models))
    w = 0.24
    fig, ax = plt.subplots(figsize=(12, 7))
    for index, (class_name, color) in enumerate([("AD", BLUE), ("FTD", ORANGE), ("Healthy", TEAL)]):
        bars = ax.bar(x + (index - 1) * w, recalls[:, index], width=w, color=color, label=class_name)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Where each main model succeeded and failed", loc="left")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, loc="upper left")
    ax.text(
        0.0,
        -0.13,
        "FTD remained the bottleneck. EEGPT improved healthy-control recall, while the direct SVM preserved stronger FTD recognition; their blend retained both advantages.",
        transform=ax.transAxes,
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, "04_main_model_class_recall.png")


def tensor_tradeoff() -> None:
    labels = ["Direct log-mel\nSVM", "Tensor Train\n+ SVM"]
    source_features = [912, 1824]
    retained_features = [20, 4]
    accuracy = [66.7, 51.9]
    macro_f1 = [64.3, 46.4]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    x = np.arange(2)
    w = 0.35
    b1 = axes[0].bar(x - w / 2, source_features, width=w, color=LIGHT, label="Before reduction")
    b2 = axes[0].bar(x + w / 2, retained_features, width=w, color=TEAL, label="Model input after reduction")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Feature count (log scale)")
    axes[0].set_title("Compression", loc="left")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].bar_label(b1, fmt="%.0f", padding=2)
    axes[0].bar_label(b2, fmt="%.0f", padding=2)
    b3 = axes[1].bar(x - w / 2, accuracy, width=w, color=BLUE, label="Accuracy")
    b4 = axes[1].bar(x + w / 2, macro_f1, width=w, color=TEAL, label="Macro F1")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 80)
    axes[1].set_ylabel("Held-out score (%)")
    axes[1].set_title("Classification trade-off", loc="left")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].bar_label(b3, fmt="%.1f", padding=2)
    axes[1].bar_label(b4, fmt="%.1f", padding=2)
    fig.suptitle("Tensor decomposition achieved extreme compression, but removed diagnostic information", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Direct branch: 912 mean/SD spectral features → PCA 20. Tensor branch: 1,824 electrode × mel × statistic values → 4 TT coefficients.",
        ha="center",
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    finish(fig, "05_tensor_compression_tradeoff.png")


def robustness() -> None:
    labels = ["Original fixed split", "Repeated participant folds"]
    accuracy = [66.7, 51.9]
    accuracy_sd = [0.0, 6.7]
    macro_f1 = [64.3, 49.0]
    f1_sd = [0.0, 6.6]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(10, 7))
    b1 = ax.bar(x - w / 2, accuracy, width=w, color=BLUE, yerr=accuracy_sd, capsize=5, label="Accuracy")
    b2 = ax.bar(x + w / 2, macro_f1, width=w, color=TEAL, yerr=f1_sd, capsize=5, label="Macro F1")
    ax.bar_label(b1, fmt="%.1f", padding=4)
    ax.bar_label(b2, fmt="%.1f", padding=4)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Participant-level score (%)")
    ax.set_title("Why the 66.7% SVM result must be called exploratory", loc="left")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    ax.text(
        0.0,
        -0.14,
        "Repeated result shows mean ± fold SD across 50 participant test folds. The original 27-person test split was near the optimistic end of the observed distribution.",
        transform=ax.transAxes,
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, "06_validation_robustness.png")


def preprocessing_ablation() -> None:
    labels = ["Moderate cleaning", "Advanced cleaning", "50/50 feature fusion"]
    accuracy = [54.3, 51.6, 54.3]
    accuracy_sd = [7.0, 7.7, 7.8]
    macro_f1 = [50.9, 48.9, 51.4]
    f1_sd = [7.6, 8.0, 8.9]
    macro_recall = [51.6, 49.6, 52.4]
    recall_sd = [7.2, 7.8, 8.3]
    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(12, 7))
    for offset, values, errors, color, label in [
        (-w, accuracy, accuracy_sd, BLUE, "Accuracy"),
        (0, macro_f1, f1_sd, TEAL, "Macro F1"),
        (w, macro_recall, recall_sd, ORANGE, "Macro recall"),
    ]:
        bars = ax.bar(x + offset, values, width=w, color=color, yerr=errors, capsize=4, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(25, 70)
    ax.set_ylabel("Repeated participant-CV score (%)")
    ax.set_title("More aggressive EEG cleaning did not improve generalization", loc="left")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, loc="upper center")
    ax.text(
        0.0,
        -0.14,
        "Mean ± fold SD over 5 × 5 participant folds. Advanced cleaning interpolated 238 channels and removed 87 ICA components, reducing macro F1 by 2.0 points.",
        transform=ax.transAxes,
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, "07_preprocessing_ablation.png")


def main() -> None:
    setup()
    dataset_growth()
    original_tensor_progress()
    binary_replication()
    combined_model_progress()
    class_recall()
    tensor_tradeoff()
    robustness()
    preprocessing_ablation()
    print(f"Wrote charts to {OUT}")


if __name__ == "__main__":
    main()
