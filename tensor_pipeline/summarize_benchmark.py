"""Consolidate rank-sweep benchmark outputs and generate comparison figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"C:\Projects\AIBSI\tensor_pipeline\outputs\benchmark_rank_sweep")
OUTPUT = Path(r"C:\Projects\AIBSI\tensor_pipeline\outputs\summary")
RANKS = [3, 5, 8]
METHOD_LABELS = {
    "pca": "PCA",
    "nmf": "NMF",
    "nonnegative_cp": "Nonnegative CP",
    "graph_nonnegative_cp": "Graph NCP",
    "nonnegative_tucker": "Nonnegative Tucker",
}
CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic regression",
    "rbf_svm": "RBF SVM",
    "hist_gradient_boosting": "Histogram gradient boosting",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = {}
    for rank in RANKS:
        report = json.loads((ROOT / f"rank_{rank}" / "benchmark_report.json").read_text())
        reports[rank] = report
        for result in report["results"]:
            rows.append({"rank": rank, **result})
    rows.sort(key=lambda row: row["repeat_mean_balanced_accuracy"], reverse=True)

    csv_fields = [
        "rank", "method", "classifier", "repeat_mean_balanced_accuracy",
        "repeat_sd_balanced_accuracy", "repeat_mean_macro_f1", "repeat_sd_macro_f1",
        "balanced_accuracy", "macro_f1", "macro_ovr_auc", "log_loss",
    ]
    with (OUTPUT / "all_rank_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    methods = list(METHOD_LABELS)
    classifiers = list(CLASSIFIER_LABELS)
    fig, axes = plt.subplots(1, len(RANKS), figsize=(17, 5.5), constrained_layout=True)
    for ax, rank in zip(axes, RANKS):
        matrix = np.array([
            [
                next(
                    row["repeat_mean_balanced_accuracy"]
                    for row in rows
                    if row["rank"] == rank and row["method"] == method and row["classifier"] == classifier
                )
                for classifier in classifiers
            ]
            for method in methods
        ])
        image = ax.imshow(matrix, vmin=0.30, vmax=0.60, cmap="viridis")
        ax.set_title(f"Rank {rank}")
        ax.set_xticks(range(len(classifiers)), [CLASSIFIER_LABELS[c] for c in classifiers], rotation=30, ha="right")
        ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white" if matrix[i, j] < 0.47 else "black")
    fig.colorbar(image, ax=axes, label="Mean balanced accuracy", shrink=0.8)
    fig.suptitle("Decomposition × classifier benchmark (3× repeated stratified 5-fold CV)")
    fig.savefig(OUTPUT / "balanced_accuracy_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for method in methods:
        values = [
            next(
                row["repeat_mean_balanced_accuracy"]
                for row in rows
                if row["rank"] == rank
                and row["method"] == method
                and row["classifier"] == "logistic_regression"
            )
            for rank in RANKS
        ]
        ax.plot(RANKS, values, marker="o", linewidth=2, label=METHOD_LABELS[method])
    ax.axhline(1 / 3, color="black", linestyle="--", linewidth=1, label="Chance (0.333)")
    ax.set_xlabel("Decomposition rank")
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_xticks(RANKS)
    ax.set_ylim(0.30, 0.60)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    ax.set_title("Rank sensitivity with logistic regression")
    fig.savefig(OUTPUT / "rank_sensitivity_logistic.png", dpi=180)
    plt.close(fig)

    selected = [
        rows[0],
        max((row for row in rows if row["method"] in {"nonnegative_cp", "nonnegative_tucker"}), key=lambda row: row["repeat_mean_balanced_accuracy"]),
        max((row for row in rows if row["method"] == "graph_nonnegative_cp"), key=lambda row: row["repeat_mean_balanced_accuracy"]),
    ]
    # Remove duplicate selections while keeping order.
    unique = []
    seen = set()
    for row in selected:
        key = (row["rank"], row["method"], row["classifier"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    fig, axes = plt.subplots(1, len(unique), figsize=(5 * len(unique), 4.5), constrained_layout=True)
    if len(unique) == 1:
        axes = [axes]
    for ax, row in zip(axes, unique):
        cm = np.asarray(row["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks(range(3), ["AD", "FTD", "CN"])
        ax.set_yticks(range(3), ["AD", "FTD", "CN"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"{METHOD_LABELS[row['method']]}\n{CLASSIFIER_LABELS[row['classifier']]}, rank {row['rank']}"
        )
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(OUTPUT / "selected_confusion_matrices.png", dpi=180)
    plt.close(fig)

    diagnostics = {}
    for rank, report in reports.items():
        diagnostics[rank] = {}
        for method in ["nonnegative_cp", "graph_nonnegative_cp", "nonnegative_tucker"]:
            entries = [run["details"] for run in report["decomposition_runs"] if run["method"] == method]
            diagnostics[rank][method] = {
                "mean_relative_squared_reconstruction_error": float(
                    np.mean([entry["relative_squared_reconstruction_error"] for entry in entries])
                ),
                "mean_channel_graph_smoothness": (
                    float(np.mean([entry["channel_graph_smoothness"] for entry in entries]))
                    if "channel_graph_smoothness" in entries[0]
                    else None
                ),
            }

    summary = {
        "ranks": RANKS,
        "combinations": len(rows),
        "best_overall": rows[0],
        "top_ten": rows[:10],
        "decomposition_diagnostics": diagnostics,
        "plots": [
            str(OUTPUT / "balanced_accuracy_heatmap.png"),
            str(OUTPUT / "rank_sensitivity_logistic.png"),
            str(OUTPUT / "selected_confusion_matrices.png"),
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for row in rows[:20]:
        print(
            f"rank={row['rank']} {row['method']:24s} {row['classifier']:24s} "
            f"BA={row['repeat_mean_balanced_accuracy']:.6f}±{row['repeat_sd_balanced_accuracy']:.6f} "
            f"F1={row['repeat_mean_macro_f1']:.6f}"
        )


if __name__ == "__main__":
    main()
