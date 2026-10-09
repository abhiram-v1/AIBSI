"""Plain Matplotlib plots for the three RBF SVM presentation runs."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_progress/output"
THREE = ROOT / "tensor_pipeline/outputs/v4_tuning/nested_report.json"
TENSOR = ROOT / "presentation_progress/svm_runs/tensor_binary/summary.json"
DIRECT = ROOT / "presentation_progress/svm_runs/direct_binary/summary.json"


def load():
    three = json.loads(THREE.read_text(encoding="utf-8"))["results"]["clf_svm"]
    tensor = json.loads(TENSOR.read_text(encoding="utf-8"))
    direct = json.loads(DIRECT.read_text(encoding="utf-8"))
    assert len(three["repeats"]) == len(tensor["repeat_metrics"]) == len(direct["repeat_metrics"]) == 3
    assert tensor["completed_folds"] == direct["completed_folds"] == 15
    return three, tensor, direct


def metrics_plot(three, tensor, direct):
    names = ["3-class V4 SVM\n(tuned features)",
             "2-class tensor SVM\n(TT/Tucker)",
             "2-class direct EEG SVM\n(no tensors)"]
    values = 100 * np.array([
        [three["mean_accuracy"], three["mean_balanced_accuracy"], three["mean_macro_f1"]],
        [tensor["mean_accuracy"], tensor["mean_balanced_accuracy"], tensor["mean_macro_f1"]],
        [direct["mean_accuracy"], direct["mean_balanced_accuracy"], direct["mean_macro_f1"]],
    ])
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(3)
    width = .24
    for i, name in enumerate(("Accuracy", "Balanced accuracy", "Macro F1")):
        bars = ax.bar(x + (i - 1) * width, values[:, i], width, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Held-out score (%)")
    ax.set_title("RBF SVM results: subject-level cross-validation")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    fig.text(.5, .025, "The 3-class and 2-class scores are different tasks; all three use an RBF SVM classifier.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(OUT / "svm_metrics.png", dpi=200)
    plt.close(fig)


def confusion_plot(three, tensor, direct):
    three_cm = np.sum([r["confusion_matrix"] for r in three["repeats"]], axis=0).astype(int)
    def binary_cm(result):
        tn, fp, fn, tp = np.sum([r["confusion_tn_fp_fn_tp"]
                                 for r in result["repeat_metrics"]], axis=0).astype(int)
        return np.array([[tn, fp], [fn, tp]])
    matrices = [three_cm, binary_cm(tensor), binary_cm(direct)]
    labels = [["AD", "FTD", "CN"], ["CN", "Dementia"], ["CN", "Dementia"]]
    titles = ["3-class V4 SVM", "2-class tensor SVM", "2-class direct EEG SVM"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1))
    for ax, cm, names, title in zip(axes, matrices, labels, titles):
        ConfusionMatrixDisplay(cm, display_labels=names).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Confusion matrices: true diagnosis versus SVM prediction", fontsize=14)
    fig.text(.5, .025, "Three 5-fold CV repetitions; each of 88 people contributes one held-out prediction per repetition.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .13, 1, .94])
    fig.savefig(OUT / "svm_confusion_matrices.png", dpi=200)
    plt.close(fig)


def recall_plot(three, tensor, direct):
    recalls_3 = 100 * np.mean([r["recall"] for r in three["repeats"]], axis=0)
    recalls_2 = 100 * np.array([
        [tensor["mean_dementia_recall"], tensor["mean_cn_recall"]],
        [direct["mean_dementia_recall"], direct["mean_cn_recall"]],
    ])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bars = left.bar(["AD", "FTD", "CN"], recalls_3)
    left.bar_label(bars, fmt="%.1f", padding=3)
    left.set_title("3-class V4 SVM")
    left.set_ylabel("Recall (%)")
    x = np.arange(2)
    for i, name in enumerate(("Tensor SVM", "Direct EEG SVM")):
        bars = right.bar(x + (i - .5) * .35, recalls_2[i], .35, label=name)
        right.bar_label(bars, fmt="%.1f", padding=3)
    right.set_xticks(x, ["Dementia", "CN"])
    right.set_title("2-class SVM runs")
    right.legend(fontsize=9)
    for ax in (left, right):
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    fig.suptitle("SVM recall by diagnosis", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(OUT / "svm_recall.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load()
    metrics_plot(*runs)
    confusion_plot(*runs)
    recall_plot(*runs)
    for name in ("svm_metrics.png", "svm_confusion_matrices.png", "svm_recall.png"):
        print(OUT / name)


if __name__ == "__main__":
    main()
