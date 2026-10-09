from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    original_cnn = load(OUTPUTS / "run" / "metrics.json")["test"]
    flat_svm = load(OUTPUTS / "svm" / "metrics.json")["test"]
    calibrated = load(OUTPUTS / "deep_compare" / "calibrated_metrics.json")
    hierarchical = load(OUTPUTS / "hierarchical_svm" / "metrics.json")["test"]
    results = {
        "Original CNN": original_cnn,
        "Flat SVM": flat_svm,
        "Enhanced CNN": calibrated["cnn"]["test"],
        "CNN–BiLSTM": calibrated["cnn_bilstm"]["test"],
        "Hierarchical SVM": hierarchical,
    }
    (OUTPUTS / "final_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    measures = ["accuracy", "macro_f1", "macro_recall"]
    colors = ["#2878b5", "#f28e2b", "#59a14f"]
    names = list(results)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5.3), constrained_layout=True)
    for offset, (measure, color) in enumerate(zip(measures, colors)):
        values = [results[name][measure] for name in names]
        bars = ax.bar(positions + (offset - 1) * 0.24, values, 0.24, color=color, label=measure.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(positions, names, rotation=10)
    ax.set(ylim=(0, 0.75), ylabel="Held-out participant score", title="Three-class EEG model comparison on the same test participants")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, loc="upper center")
    fig.savefig(OUTPUTS / "final_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    class_names = ["AD", "FTD", "HC"]
    width = 0.15
    class_positions = np.arange(3)
    for model_offset, name in enumerate(names):
        values = [results[name]["per_class"][class_name]["recall"] for class_name in class_names]
        bars = ax.bar(class_positions + (model_offset - 2) * width, values, width, label=name)
        if name == "Hierarchical SVM":
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(class_positions, class_names)
    ax.set(ylim=(0, 1), ylabel="Recall", title="Per-class recall on held-out participants")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.savefig(OUTPUTS / "final_per_class_recall.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
