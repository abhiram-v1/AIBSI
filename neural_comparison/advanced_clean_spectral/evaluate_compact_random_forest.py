from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import audit_preprocessing_feature_svm as audit


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "compact_random_forest"
SEED = audit.SEED
INNER_TREES = 32
FINAL_TREES = 800

FAMILIES = audit.FEATURE_FAMILIES
NORMALIZERS = ["global", "site_shrunk"]
MIN_LEAVES = [2, 5]
MAX_FEATURES = ["sqrt", 0.5]


def configs() -> list[dict[str, Any]]:
    return [
        {
            "family": family,
            "normalization": normalization,
            "min_samples_leaf": leaf,
            "max_features": max_features,
        }
        for family, normalization, leaf, max_features in itertools.product(
            FAMILIES, NORMALIZERS, MIN_LEAVES, MAX_FEATURES
        )
    ]


def make_model(labels: np.ndarray, config: dict[str, Any], trees: int, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        criterion="gini",
        max_features=config["max_features"],
        min_samples_leaf=config["min_samples_leaf"],
        class_weight=audit.weights_for(labels, 1.25),
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )


def fit_predict(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    config: dict[str, Any],
    trees: int,
    seed: int,
) -> tuple[np.ndarray, RandomForestClassifier]:
    matrix = features[config["family"]]
    train_x, test_x = audit.normalize_fold(matrix, sites, train, test, config["normalization"])
    model = make_model(labels[train], config, trees, seed).fit(train_x, labels[train])
    return model.predict(test_x), model


def tune(
    features: dict[str, np.ndarray], labels: np.ndarray, sites: np.ndarray, train: np.ndarray, fold_seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strata = np.asarray([f"{sites[index]}:{labels[index]}" for index in train])
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=fold_seed)
    candidates = configs()
    values: list[list[dict[str, float]]] = [[] for _ in candidates]
    for inner_number, (fit_rel, val_rel) in enumerate(splitter.split(train, strata)):
        fit = train[fit_rel]
        val = train[val_rel]
        for index, config in enumerate(candidates):
            prediction, _ = fit_predict(
                features, labels, sites, fit, val, config, INNER_TREES, fold_seed + inner_number * 100 + index
            )
            values[index].append(audit.score(labels[val], prediction))
    records = []
    for config, rows in zip(candidates, values):
        records.append(
            {
                **config,
                **{name: float(np.mean([row[name] for row in rows])) for name in rows[0]},
            }
        )
    records.sort(
        key=lambda row: (row["macro_f1"], row["ftd_recall"], row["macro_recall"], row["accuracy"]),
        reverse=True,
    )
    return records[0], records


def fixed_split_evaluation(
    features: dict[str, np.ndarray], participants: list[dict[str, Any]], labels: np.ndarray, sites: np.ndarray
) -> dict[str, Any]:
    splits = audit.eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    validation_rows = []
    for index, config in enumerate(configs()):
        prediction, _ = fit_predict(features, labels, sites, train, val, config, FINAL_TREES, SEED + index)
        validation_rows.append({**config, **audit.score(labels[val], prediction)})
    validation_rows.sort(
        key=lambda row: (row["macro_f1"], row["ftd_recall"], row["macro_recall"], row["accuracy"]),
        reverse=True,
    )
    selected = {key: validation_rows[0][key] for key in ["family", "normalization", "min_samples_leaf", "max_features"]}
    train_val = np.concatenate([train, val])
    prediction, _ = fit_predict(features, labels, sites, train_val, test, selected, FINAL_TREES, SEED + 9000)
    return {
        "splits": {name: len(indexes) for name, indexes in splits.items()},
        "selected_on_validation": validation_rows[0],
        "test_true": labels[test].tolist(),
        "test_prediction": prediction.tolist(),
        "test": audit.score(labels[test], prediction),
        "confusion_matrix": confusion_matrix(labels[test], prediction, labels=[0, 1, 2]).tolist(),
        "per_class": {
            audit.eeg.CLASSES[label]: {
                "recall": float(audit.recall_score(labels[test] == label, prediction == label, zero_division=0)),
                "f1": float(audit.f1_score(labels[test] == label, prediction == label, zero_division=0)),
            }
            for label in range(3)
        },
        "validation_candidates": validation_rows,
    }


def load_benchmarks() -> dict[str, dict[str, float]]:
    result_dir = HERE / "outputs"
    files = {
        "SVM": result_dir / "results" / "site_normalized_metrics.json",
        "EEGPT": result_dir / "deep_models" / "eegpt_attention" / "metrics.json",
        "SVM + EEGPT": result_dir / "deep_models" / "eegpt_attention" / "svm_blend_metrics.json",
    }
    return {name: json.loads(path.read_text(encoding="utf-8"))["test"] for name, path in files.items()}


def plot_results(payload: dict[str, Any]) -> None:
    fixed = {**payload["benchmarks"], "Random Forest": payload["fixed_split"]["test"]}
    metrics = ["accuracy", "macro_f1", "macro_recall"]
    labels = ["Accuracy", "Macro F1", "Macro recall"]
    colors = ["#4C78A8", "#F58518", "#59A14F", "#E45756"]
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for index, (name, row) in enumerate(fixed.items()):
        values = [row[metric] for metric in metrics]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=name, color=colors[index])
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set(ylim=(0, 0.82), ylabel="Held-out participant score", title="Fixed-split benchmark comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(OUT / "fixed_split_benchmark.png", dpi=190)
    plt.close(fig)

    matrix = np.asarray(payload["fixed_split"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.8, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), audit.eeg.CLASSES)
    ax.set_yticks(range(3), audit.eeg.CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title="Random Forest: fixed held-out participants")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "random_forest_confusion.png", dpi=190)
    plt.close(fig)

    repeated = payload["repeated"]["summary"]
    svm = payload["compact_svm_fusion"]
    x = np.arange(4)
    metric_names = ["accuracy", "macro_f1", "macro_recall", "ftd_recall"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index, (name, summary, color) in enumerate(
        [("Compact SVM fusion", svm, "#4C78A8"), ("Random Forest", repeated, "#E45756")]
    ):
        means = [summary[metric]["mean"] for metric in metric_names]
        errors = [summary[metric]["std"] for metric in metric_names]
        bars = ax.bar(x + (index - 0.5) * 0.36, means, 0.36, yerr=errors, capsize=4, label=name, color=color)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=3, fontsize=9)
    ax.set_xticks(x, ["Accuracy", "Macro F1", "Macro recall", "FTD recall"])
    ax.set(ylim=(0, 0.75), ylabel="Participant-level score", title="Repeated participant-level evaluation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "repeated_comparison.png", dpi=190)
    plt.close(fig)


def write_report(payload: dict[str, Any]) -> None:
    repeated = payload["repeated"]["summary"]
    fixed = payload["fixed_split"]
    benchmarks = {**payload["benchmarks"], "Random Forest": fixed["test"]}
    lines = [
        "# Compact-feature Random Forest",
        "",
        "## Repeated participant-level evaluation",
        "",
        "The feature family, site normalization, minimum leaf size, and feature-subsampling rate were selected using four-fold training-only cross-validation inside every outer fold.",
        "",
        "| Accuracy | Macro F1 | Macro recall | FTD recall |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(
            f"{repeated[name]['mean']*100:.1f}% +/- {repeated[name]['std']*100:.1f}%"
            for name in ["accuracy", "macro_f1", "macro_recall", "ftd_recall"]
        ) + " |",
        "",
        f"Compared with the compact SVM fusion on the identical folds, Random Forest changed macro F1 by {payload['paired_macro_f1_change']*100:+.2f} percentage points.",
        "",
        "## Original fixed split",
        "",
        "| Model | Accuracy | Macro F1 | Macro recall |",
        "|---|---:|---:|---:|",
    ]
    for name, row in benchmarks.items():
        lines.append(f"| {name} | {row['accuracy']*100:.1f}% | {row['macro_f1']*100:.1f}% | {row['macro_recall']*100:.1f}% |")
    selected = fixed["selected_on_validation"]
    lines.extend(
        [
            "",
            f"Validation selected `{selected['family']}` features, `{selected['normalization']}` normalization, minimum leaf size {selected['min_samples_leaf']}, and `{selected['max_features']}` feature subsampling.",
            "",
            "## Decision",
            "",
            "The Random Forest is retained as a classical-model comparison. The SVM + EEGPT blend remains the final exploratory benchmark because it has the strongest fixed-split accuracy and macro F1. Its 70.4% score must still be described as fixed-split performance rather than expected external performance.",
        ]
    )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate compact spectral Random Forest")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Extracting compact features from moderate preprocessing", flush=True)
    features, participants, labels, sites = audit.extract_subject_features(audit.MINIMAL, "spectrograms.npy")
    strata = np.asarray([f"{sites[index]}:{labels[index]}" for index in range(len(labels))])
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=SEED)
    folds = []
    for fold_number, (train, test) in enumerate(splitter.split(np.arange(len(labels)), strata), 1):
        selected, _ = tune(features, labels, sites, train, SEED + fold_number)
        prediction, _ = fit_predict(features, labels, sites, train, test, selected, FINAL_TREES, SEED + 5000 + fold_number)
        metrics = audit.score(labels[test], prediction)
        folds.append(
            {
                "fold": fold_number,
                "train": train.tolist(),
                "test": test.tolist(),
                "true": labels[test].tolist(),
                "prediction": prediction.tolist(),
                "selected": selected,
                "metrics": metrics,
            }
        )
        print(
            f"  fold {fold_number}/{args.repeats * 5}: F1={metrics['macro_f1']:.3f}, "
            f"FTD recall={metrics['ftd_recall']:.3f}, family={selected['family']}",
            flush=True,
        )

    repeated = {"folds": folds, "summary": audit.summarize(folds)}
    fixed = fixed_split_evaluation(features, participants, labels, sites)
    svm_payload = json.loads((audit.OUT / "metrics.json").read_text(encoding="utf-8"))
    svm_folds = svm_payload["results"]["fusion"]["folds"]
    if [row["test"] for row in svm_folds] != [row["test"] for row in folds]:
        raise RuntimeError("Random Forest and compact SVM outer folds do not match")
    paired_change = float(
        np.mean([rf["metrics"]["macro_f1"] - svm["metrics"]["macro_f1"] for rf, svm in zip(folds, svm_folds)])
    )
    payload = {
        "participants": len(participants),
        "class_counts": dict(Counter(row["label"] for row in participants)),
        "protocol": {
            "outer_repeats": args.repeats,
            "outer_folds": 5,
            "inner_folds": 4,
            "inner_trees": INNER_TREES,
            "final_trees": FINAL_TREES,
            "configs": configs(),
        },
        "repeated": repeated,
        "compact_svm_fusion": svm_payload["results"]["fusion"]["summary"],
        "paired_macro_f1_change": paired_change,
        "fixed_split": fixed,
        "benchmarks": load_benchmarks(),
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_results(payload)
    write_report(payload)
    print(json.dumps({"repeated": repeated["summary"], "fixed_split": fixed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
