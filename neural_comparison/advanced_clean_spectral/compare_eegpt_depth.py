from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEEP = HERE / "outputs" / "deep_models"
OUT = DEEP / "eegpt_attention_plus2"


def main() -> None:
    svm = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    original = json.loads((DEEP / "eegpt_attention" / "metrics.json").read_text(encoding="utf-8"))["test"]
    deeper = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))["test"]
    blend = json.loads((DEEP / "eegpt_attention" / "svm_blend_metrics.json").read_text(encoding="utf-8"))["test"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    matrix = np.asarray(deeper["confusion_matrix"])
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    axes[0].set_xticks(range(3), ["AD", "FTD", "HC"])
    axes[0].set_yticks(range(3), ["AD", "FTD", "HC"])
    axes[0].set(xlabel="Predicted", ylabel="True", title="EEGPT with two new blocks")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    names = ["Spectral\nSVM", "Original\nEEGPT", "EEGPT + 2\nnew blocks", "SVM +\nEEGPT"]
    models = [svm, original, deeper, blend]
    positions = np.arange(len(names))
    for offset, measure in enumerate(["accuracy", "macro_f1", "macro_recall"]):
        bars = axes[1].bar(
            positions + (offset - 1) * 0.24,
            [item[measure] for item in models],
            width=0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axes[1].set_xticks(positions, names)
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Effect of adding transformer depth")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.savefig(OUT / "depth_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
