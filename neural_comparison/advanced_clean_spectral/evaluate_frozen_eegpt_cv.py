"""Repeated nested participant-level CV for spectral, frozen-EEGPT and fused models.

Uses the same 5 x 5 outer folds (same seed and site:diagnosis strata) as the
compact-feature SVM and random-forest audits, so the numbers are directly
comparable. Every normalization, PCA, scaler and hyperparameter choice is fit
inside the training participants of each fold; the EEGPT encoder is frozen and
label-free, so its embeddings are computed once.
"""
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import audit_preprocessing_feature_svm as audit

eeg = audit.eeg
HERE = Path(__file__).resolve().parent
MOD = HERE / "outputs" / "moderate_log_mel" / "prepared"
EEGPT_PREPARED = HERE / "outputs" / "eegpt" / "prepared"
EMBEDDINGS = HERE / "outputs" / "eegpt" / "frozen_embeddings" / "window_embeddings.npy"
OUT = HERE / "outputs" / "frozen_eegpt_cv"
SEED = audit.SEED

NORMALIZERS = ["global", "site_zscore", "site_shrunk"]
PCA_SINGLE = [10, 20, 40]
PCA_FUSED = [10, 20]
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0]
FTD_BOOSTS = [1.0, 1.5]
MODEL_BLOCKS = {
    "spectral": ["spectral"],
    "eegpt2048": ["eegpt2048"],
    "eegpt512": ["eegpt512"],
    "early_fusion": ["spectral", "eegpt2048"],
}
BASELINE = "spectral"


def load_data() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    participants, labels, sites, eegpt_index = audit.participant_labels(EEGPT_PREPARED)
    mod_participants, mod_labels, mod_sites, mod_index = audit.participant_labels(MOD)
    if [p["subject_id"] for p in participants] != [p["subject_id"] for p in mod_participants]:
        raise RuntimeError("Participant order differs between EEGPT and log-mel preparation")
    if not np.array_equal(labels, mod_labels) or not np.array_equal(sites, mod_sites):
        raise RuntimeError("Labels or sites differ between preparations")
    tensor = np.load(MOD / "log_mel.npy", mmap_mode="r")
    window_labels = np.load(MOD / "labels.npy", mmap_mode="r")
    spectral, spectral_labels = eeg.subject_features(tensor, window_labels, mod_index, mod_participants)
    if not np.array_equal(spectral_labels, labels):
        raise RuntimeError("Spectral subject labels differ")
    windows = np.load(EMBEDDINGS)
    if len(windows) != len(eegpt_index):
        raise RuntimeError("Embedding count does not match EEGPT windows")
    pooled = np.stack([windows[eegpt_index == subject].mean(axis=0) for subject in range(len(participants))])
    blocks = {
        "spectral": np.nan_to_num(spectral.reshape(len(spectral), -1)),
        "eegpt2048": pooled.astype(np.float32),
        "eegpt512": pooled.reshape(len(pooled), 4, 512).mean(axis=1).astype(np.float32),
    }
    return blocks, labels, sites, participants


def prepare(
    blocks: dict[str, np.ndarray], names: list[str], sites: np.ndarray,
    train: np.ndarray, evaluate: np.ndarray, normalization: str, components: int,
) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for name in names:
        fit_x, eval_x = audit.normalize_fold(blocks[name], sites, train, evaluate, normalization)
        parts.append(audit.reduce_fold(fit_x, eval_x, components))
    if len(parts) > 1:
        # Equalize block energy so one block cannot dominate the fused linear SVM.
        energy = [float((fit_x ** 2).sum(axis=1).mean()) for fit_x, _ in parts]
        target = float(np.mean(energy))
        parts = [(fx * np.sqrt(target / e), ex * np.sqrt(target / e)) for (fx, ex), e in zip(parts, energy)]
    return np.hstack([p[0] for p in parts]), np.hstack([p[1] for p in parts])


def classifier(labels: np.ndarray, c_value: float, boost: float) -> SVC:
    return SVC(C=c_value, kernel="linear", class_weight=audit.weights_for(labels, boost), random_state=SEED)


def tune(
    blocks: dict[str, np.ndarray], names: list[str], labels: np.ndarray, sites: np.ndarray, train: np.ndarray
) -> dict[str, Any]:
    strata = np.asarray([f"{sites[i]}:{labels[i]}" for i in train])
    inner = StratifiedKFold(n_splits=audit.INNER_SPLITS, shuffle=True, random_state=SEED)
    component_grid = PCA_FUSED if len(names) > 1 else PCA_SINGLE
    records: dict[tuple, list[dict[str, float]]] = {}
    for fit_rel, val_rel in inner.split(train, strata):
        fit, val = train[fit_rel], train[val_rel]
        for normalization, components in itertools.product(NORMALIZERS, component_grid):
            fit_x, val_x = prepare(blocks, names, sites, fit, val, normalization, components)
            for c_value, boost in itertools.product(C_VALUES, FTD_BOOSTS):
                prediction = classifier(labels[fit], c_value, boost).fit(fit_x, labels[fit]).predict(val_x)
                records.setdefault((normalization, components, c_value, boost), []).append(
                    audit.score(labels[val], prediction)
                )
    candidates = []
    for (normalization, components, c_value, boost), rows in records.items():
        candidates.append({
            "normalization": normalization, "pca": components, "C": c_value, "ftd_boost": boost,
            **{name: float(np.mean([row[name] for row in rows])) for name in rows[0]},
        })
    candidates.sort(
        key=lambda row: (row["macro_f1"], row["ftd_recall"], row["macro_recall"], row["accuracy"]), reverse=True
    )
    return candidates[0]


def fit_predict(
    blocks: dict[str, np.ndarray], names: list[str], labels: np.ndarray, sites: np.ndarray,
    train: np.ndarray, test: np.ndarray, config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    fit_x, test_x = prepare(blocks, names, sites, train, test, config["normalization"], config["pca"])
    model = classifier(labels[train], config["C"], config["ftd_boost"]).fit(fit_x, labels[train])
    return model.predict(test_x), model.decision_function(test_x)


def fast_macro_f1(true: np.ndarray, prediction: np.ndarray) -> float:
    matrix = np.bincount(true * 3 + prediction, minlength=9).reshape(3, 3)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.where(denominator > 0, 2 * np.diag(matrix) / np.maximum(denominator, 1), 0.0).mean())


def run_cv(
    blocks: dict[str, np.ndarray], labels: np.ndarray, sites: np.ndarray, repeats: int, model_names: list[str]
) -> dict[str, Any]:
    strata = np.asarray([f"{sites[i]}:{labels[i]}" for i in range(len(labels))])
    splitter = RepeatedStratifiedKFold(n_splits=audit.OUTER_SPLITS, n_repeats=repeats, random_state=SEED)
    folds = list(splitter.split(np.arange(len(labels)), strata))
    results: dict[str, Any] = {name: {"folds": []} for name in model_names}
    if "spectral" in model_names and "eegpt2048" in model_names:
        results["late_fusion"] = {"folds": []}
    for number, (train, test) in enumerate(folds, 1):
        decisions = {}
        for name in model_names:
            names = MODEL_BLOCKS[name]
            config = tune(blocks, names, labels, sites, train)
            prediction, decision = fit_predict(blocks, names, labels, sites, train, test, config)
            decisions[name] = decision
            results[name]["folds"].append({
                "fold": number, "test": test.tolist(), "prediction": prediction.tolist(),
                "metrics": audit.score(labels[test], prediction), "selected": config,
            })
        if "late_fusion" in results:
            blend = 0.5 * audit.softmax(decisions["spectral"]) + 0.5 * audit.softmax(decisions["eegpt2048"])
            prediction = np.argmax(blend, axis=1)
            results["late_fusion"]["folds"].append({
                "fold": number, "test": test.tolist(), "prediction": prediction.tolist(),
                "metrics": audit.score(labels[test], prediction), "selected": None,
            })
        line = ", ".join(f"{name}={results[name]['folds'][-1]['metrics']['macro_f1']:.2f}" for name in results)
        print(f"  fold {number}/{len(folds)} macro-F1: {line}", flush=True)
    for name, entry in results.items():
        predictions = np.zeros((repeats, len(labels)), dtype=int)
        for number, fold in enumerate(entry["folds"]):
            predictions[number // audit.OUTER_SPLITS, fold["test"]] = fold["prediction"]
        entry["oof_predictions"] = predictions
    return results


def summarize(entry: dict[str, Any], labels: np.ndarray) -> dict[str, Any]:
    measures = ["accuracy", "macro_f1", "macro_recall", "ftd_recall"]
    summary: dict[str, Any] = {
        name: {
            "mean": float(np.mean([f["metrics"][name] for f in entry["folds"]])),
            "std": float(np.std([f["metrics"][name] for f in entry["folds"]], ddof=1)),
        }
        for name in measures
    }
    predictions = entry["oof_predictions"]
    per_repeat = [fast_macro_f1(labels, row) for row in predictions]
    summary["pooled_macro_f1_per_repeat"] = {
        "values": per_repeat, "mean": float(np.mean(per_repeat)), "std": float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0,
    }
    matrix = sum(np.bincount(labels * 3 + row, minlength=9).reshape(3, 3) for row in predictions)
    summary["pooled_confusion_matrix_all_repeats"] = matrix.tolist()
    summary["pooled_recall"] = {eeg.CLASSES[i]: float(matrix[i, i] / matrix[i].sum()) for i in range(3)}
    if entry["folds"][0]["selected"] is not None:
        summary["selected_normalization"] = dict(Counter(f["selected"]["normalization"] for f in entry["folds"]))
        summary["selected_pca"] = dict(Counter(str(f["selected"]["pca"]) for f in entry["folds"]))
        summary["selected_C"] = dict(Counter(str(f["selected"]["C"]) for f in entry["folds"]))
    return summary


def paired_bootstrap(results: dict[str, Any], labels: np.ndarray, baseline: str, draws: int = 2000) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(0, len(labels), size=(draws, len(labels)))

    def score(name: str, index: np.ndarray) -> float:
        return float(np.mean([fast_macro_f1(labels[index], row[index]) for row in results[name]["oof_predictions"]]))

    base = np.asarray([score(baseline, index) for index in resamples])
    output = {}
    for name in results:
        if name == baseline:
            continue
        diff = np.asarray([score(name, index) for index in resamples]) - base
        output[name] = {
            "mean_difference": float(diff.mean()),
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "probability_better": float((diff > 0).mean()),
        }
    return output


def site_predictability(blocks: dict[str, np.ndarray], sites: np.ndarray) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    output = {}
    for name, matrix in blocks.items():
        model = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=SEED), LogisticRegression(max_iter=3000))
        output[name] = float(cross_val_score(model, matrix, sites, cv=splitter, scoring="balanced_accuracy").mean())
    return output


def permuted_labels(labels: np.ndarray, sites: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    shuffled = labels.copy()
    for site in sorted(set(sites)):
        index = np.flatnonzero(sites == site)
        shuffled[index] = rng.permutation(labels[index])
    return shuffled


def plot(summaries: dict[str, Any], path: Path) -> None:
    names = list(summaries)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    positions = np.arange(len(names))
    means = [summaries[n]["pooled_macro_f1_per_repeat"]["mean"] for n in names]
    errors = [summaries[n]["pooled_macro_f1_per_repeat"]["std"] for n in names]
    bars = ax.bar(positions, means, yerr=errors, capsize=4, color="#4c78a8")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.axhline(0.678, color="#d62728", linestyle=":", linewidth=1.2, label="old fixed-split SVM+EEGPT blend (27 participants)")
    ax.set_xticks(positions, names)
    ax.set(ylim=(0, 0.8), ylabel="Macro F1, pooled over 174 participants (mean +/- SD of 5 repeats)",
           title="Repeated nested participant-level CV")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen EEGPT + spectral repeated nested CV")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--permutation-check", action="store_true", help="Shuffle labels within site; expect ~chance")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, labels, sites, _ = load_data()
    print({name: matrix.shape for name, matrix in blocks.items()}, flush=True)

    if args.permutation_check:
        shuffled = permuted_labels(labels, sites)
        results = run_cv(blocks, shuffled, sites, 1, ["spectral", "eegpt2048"])
        summary = {name: summarize(entry, shuffled) for name, entry in results.items()}
        (OUT / "permutation_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name, row in summary.items():
            print(f"{name}: shuffled-label macro-F1 per outer fold {row['macro_f1']['mean']:.3f} +/- {row['macro_f1']['std']:.3f}")
        return

    results = run_cv(blocks, labels, sites, args.repeats, list(MODEL_BLOCKS))
    summaries = {name: summarize(entry, labels) for name, entry in results.items()}
    bootstrap = paired_bootstrap(results, labels, BASELINE)
    site_scores = site_predictability(blocks, sites)
    payload = {
        "protocol": {
            "outer": f"{args.repeats} x 5 RepeatedStratifiedKFold, strata=site:diagnosis, seed={SEED}",
            "inner": f"{audit.INNER_SPLITS}-fold StratifiedKFold tuning inside each outer training set",
            "grid": {"normalizers": NORMALIZERS, "pca_single": PCA_SINGLE, "pca_fused": PCA_FUSED,
                     "C": C_VALUES, "ftd_boost": FTD_BOOSTS},
            "late_fusion": "fixed 0.5/0.5 softmax of spectral and eegpt2048 decisions, nothing tuned",
        },
        "summary": summaries,
        "paired_bootstrap_vs_spectral": bootstrap,
        "site_predictability_balanced_accuracy_chance_0.333": site_scores,
        "folds": {name: [{k: v for k, v in f.items()} for f in entry["folds"]] for name, entry in results.items()},
    }
    (OUT / "frozen_eegpt_cv.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(summaries, OUT / "frozen_eegpt_cv.png")

    print("\nOuter-fold mean +/- SD (25 folds) | pooled macro-F1 per repeat")
    for name, row in summaries.items():
        pooled = row["pooled_macro_f1_per_repeat"]
        print(
            f"{name:13s} acc {row['accuracy']['mean']:.3f}+/-{row['accuracy']['std']:.3f}  "
            f"F1 {row['macro_f1']['mean']:.3f}+/-{row['macro_f1']['std']:.3f}  "
            f"FTD recall {row['ftd_recall']['mean']:.3f}  | pooled F1 {pooled['mean']:.3f}+/-{pooled['std']:.3f}  "
            f"recall {row['pooled_recall']}"
        )
    print("\nPaired participant bootstrap vs spectral (pooled macro-F1):")
    for name, row in bootstrap.items():
        print(f"  {name:13s} diff {row['mean_difference']:+.3f}  95% CI [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]  P(better) {row['probability_better']:.2f}")
    print("\nSite predictability (balanced accuracy, chance 0.333):", site_scores)


if __name__ == "__main__":
    main()
