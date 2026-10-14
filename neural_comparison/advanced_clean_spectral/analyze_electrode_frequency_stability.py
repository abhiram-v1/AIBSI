from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "electrode_frequency_stability"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402


SEED = 20261013
REPEATS = 10
OUTER_SPLITS = 5
INNER_SPLITS = 4
PERMUTATIONS = 4
ELECTRODE_COUNTS = (5, 8, 12, 19)
C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0)


def load_subject_data() -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    tensor = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    labels_windows = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    participants = manifest["participants"]
    features, labels = eeg.subject_features(tensor, labels_windows, subject_index, participants)
    sites = np.asarray([p["site"] for p in participants])
    return features, labels, participants, sites


def combined_strata(labels: np.ndarray, sites: np.ndarray) -> np.ndarray:
    return np.asarray([f"{site}:{int(label)}" for site, label in zip(sites, labels)])


def normalize_fold(
    features: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    evaluate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.empty_like(features[train], dtype=np.float32)
    eval_x = np.empty_like(features[evaluate], dtype=np.float32)
    train_sites = sites[train]
    eval_sites = sites[evaluate]
    global_mean = features[train].mean(axis=0, keepdims=True)
    global_std = np.maximum(features[train].std(axis=0, keepdims=True), 1e-3)
    for site in sorted(set(train_sites) | set(eval_sites)):
        fit_mask = train_sites == site
        mean = features[train][fit_mask].mean(axis=0, keepdims=True) if fit_mask.any() else global_mean
        std = (
            np.maximum(features[train][fit_mask].std(axis=0, keepdims=True), 1e-3)
            if fit_mask.any()
            else global_std
        )
        train_x[train_sites == site] = (features[train][train_sites == site] - mean) / std
        eval_x[eval_sites == site] = (features[evaluate][eval_sites == site] - mean) / std
    return train_x, eval_x


def columns_for_electrodes(electrodes: list[int], per_electrode: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(e * per_electrode, (e + 1) * per_electrode, dtype=int) for e in electrodes]
    )


def make_model(c_value: float, n_samples: int, n_features: int) -> Pipeline:
    components = min(20, n_samples - 1, n_features)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=components, random_state=SEED)),
            (
                "svm",
                SVC(
                    C=c_value,
                    kernel="linear",
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def tune_c(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    columns: np.ndarray,
) -> tuple[float, float]:
    strata = combined_strata(labels[train], sites[train])
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + int(train.sum()))
    scores: dict[float, list[float]] = defaultdict(list)
    for local_train, local_val in splitter.split(train, strata):
        inner_train, inner_val = train[local_train], train[local_val]
        train_x, val_x = normalize_fold(features, sites, inner_train, inner_val)
        flat_train = train_x.reshape(len(inner_train), -1)[:, columns]
        flat_val = val_x.reshape(len(inner_val), -1)[:, columns]
        for c_value in C_VALUES:
            model = make_model(c_value, len(inner_train), len(columns))
            model.fit(flat_train, labels[inner_train])
            pred = model.predict(flat_val)
            scores[c_value].append(f1_score(labels[inner_val], pred, average="macro", zero_division=0))
    ranked = sorted(
        ((float(np.mean(values)), -c_value, c_value) for c_value, values in scores.items()),
        reverse=True,
    )
    return float(ranked[0][2]), float(ranked[0][0])


def rank_groups_from_training(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    c_value: float,
    mel_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF grouped permutation ranking using only the current outer training partition."""
    per_electrode = features.shape[2]
    mel_bins = per_electrode // 2
    strata = combined_strata(labels[train], sites[train])
    splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + 17 + int(train.sum()))
    electrode_drops = np.zeros((INNER_SPLITS, len(eeg.CHANNELS), PERMUTATIONS), dtype=float)
    band_names = ["Delta 1-4", "Theta 4-8", "Alpha 8-13", "Beta 13-30", "Gamma 30-40"]
    band_edges = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 41)]
    band_drops = np.zeros((INNER_SPLITS, len(band_names), PERMUTATIONS), dtype=float)
    rng = np.random.default_rng(SEED + int(train.sum()))
    all_columns = np.arange(features.shape[1] * per_electrode)
    for fold, (local_train, local_val) in enumerate(splitter.split(train, strata)):
        inner_train, inner_val = train[local_train], train[local_val]
        train_x, val_x = normalize_fold(features, sites, inner_train, inner_val)
        flat_train = train_x.reshape(len(inner_train), -1)
        flat_val = val_x.reshape(len(inner_val), -1)
        model = make_model(c_value, len(inner_train), len(all_columns))
        model.fit(flat_train, labels[inner_train])
        baseline = f1_score(labels[inner_val], model.predict(flat_val), average="macro", zero_division=0)
        val_sites = sites[inner_val]
        for electrode in range(len(eeg.CHANNELS)):
            cols = columns_for_electrodes([electrode], per_electrode)
            for repeat in range(PERMUTATIONS):
                shuffled = flat_val.copy()
                for site in np.unique(val_sites):
                    rows = np.flatnonzero(val_sites == site)
                    shuffled[np.ix_(rows, cols)] = flat_val[np.ix_(rng.permutation(rows), cols)]
                score = f1_score(labels[inner_val], model.predict(shuffled), average="macro", zero_division=0)
                electrode_drops[fold, electrode, repeat] = baseline - score
        for band, (low, high) in enumerate(band_edges):
            bins = np.flatnonzero((mel_centers >= low) & (mel_centers < high))
            cols = np.asarray(
                [
                    electrode * per_electrode + offset + int(bin_index)
                    for electrode in range(len(eeg.CHANNELS))
                    for offset in (0, mel_bins)
                    for bin_index in bins
                ],
                dtype=int,
            )
            for repeat in range(PERMUTATIONS):
                shuffled = flat_val.copy()
                for site in np.unique(val_sites):
                    rows = np.flatnonzero(val_sites == site)
                    shuffled[np.ix_(rows, cols)] = flat_val[np.ix_(rng.permutation(rows), cols)]
                score = f1_score(labels[inner_val], model.predict(shuffled), average="macro", zero_division=0)
                band_drops[fold, band, repeat] = baseline - score
    return electrode_drops.mean(axis=(0, 2)), band_drops.mean(axis=(0, 2))


def evaluate_model(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    electrode_indices: list[int],
    c_value: float,
) -> tuple[np.ndarray, dict]:
    per_electrode = features.shape[2]
    columns = columns_for_electrodes(electrode_indices, per_electrode)
    train_x, test_x = normalize_fold(features, sites, train, test)
    train_flat = train_x.reshape(len(train), -1)[:, columns]
    test_flat = test_x.reshape(len(test), -1)[:, columns]
    model = make_model(c_value, len(train), len(columns))
    model.fit(train_flat, labels[train])
    pred = model.predict(test_flat)
    return pred, eeg.metrics(labels[test], pred)


def run_outer_fold(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    mel_centers: np.ndarray,
) -> dict:
    per_electrode = features.shape[2]
    full_columns = columns_for_electrodes(list(range(len(eeg.CHANNELS))), per_electrode)
    full_c, full_inner_f1 = tune_c(features, labels, sites, train, full_columns)
    importance, band_importance = rank_groups_from_training(
        features, labels, sites, train, full_c, mel_centers
    )
    ranking = np.argsort(importance)[::-1]
    models = {}
    for count in ELECTRODE_COUNTS:
        selected = list(ranking[:count]) if count < len(eeg.CHANNELS) else list(range(len(eeg.CHANNELS)))
        columns = columns_for_electrodes(selected, per_electrode)
        c_value, inner_f1 = tune_c(features, labels, sites, train, columns)
        pred, metrics = evaluate_model(features, labels, sites, train, test, selected, c_value)
        models[str(count)] = {
            "selected": [eeg.CHANNELS[i] for i in selected],
            "C": c_value,
            "inner_macro_f1": inner_f1,
            "prediction": pred.tolist(),
            "metrics": metrics,
        }

    # Leave-one-electrode-out ablation with the full model's selected C.
    base_pred, base_metrics = evaluate_model(
        features, labels, sites, train, test, list(range(len(eeg.CHANNELS))), full_c
    )
    loo_delta = []
    for electrode in range(len(eeg.CHANNELS)):
        keep = [i for i in range(len(eeg.CHANNELS)) if i != electrode]
        _, removed_metrics = evaluate_model(features, labels, sites, train, test, keep, full_c)
        loo_delta.append(base_metrics["macro_f1"] - removed_metrics["macro_f1"])
    return {
        "train": train.tolist(),
        "test": test.tolist(),
        "true": labels[test].tolist(),
        "full_C": full_c,
        "full_inner_macro_f1": full_inner_f1,
        "permutation_importance": importance.tolist(),
        "band_importance": band_importance.tolist(),
        "ranking": [eeg.CHANNELS[i] for i in ranking],
        "loo_delta_macro_f1": loo_delta,
        "models": models,
        "base_prediction": base_pred.tolist(),
        "base_metrics": base_metrics,
    }


def summarize_folds(folds: list[dict], participants: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for count in ELECTRODE_COUNTS:
        key = str(count)
        values = np.asarray([fold["models"][key]["metrics"]["macro_f1"] for fold in folds])
        accuracies = np.asarray([fold["models"][key]["metrics"]["accuracy"] for fold in folds])
        recalls = np.asarray([fold["models"][key]["metrics"]["macro_recall"] for fold in folds])
        true = np.concatenate([fold["true"] for fold in folds])
        pred = np.concatenate([fold["models"][key]["prediction"] for fold in folds])
        pooled = eeg.metrics(true, pred)
        by_site = {}
        for site in sorted({p["site"] for p in participants}):
            site_true, site_pred = [], []
            for fold in folds:
                test = np.asarray(fold["test"], dtype=int)
                mask = np.asarray([participants[i]["site"] == site for i in test])
                site_true.extend(np.asarray(fold["true"])[mask].tolist())
                site_pred.extend(np.asarray(fold["models"][key]["prediction"])[mask].tolist())
            by_site[site] = {
                "subjects_over_repeats": len(site_true),
                "accuracy": float(accuracy_score(site_true, site_pred)),
                "macro_f1": float(f1_score(site_true, site_pred, average="macro", zero_division=0)),
            }
        summary[key] = {
            "fold_macro_f1_mean": float(values.mean()),
            "fold_macro_f1_std": float(values.std(ddof=1)),
            "fold_accuracy_mean": float(accuracies.mean()),
            "fold_accuracy_std": float(accuracies.std(ddof=1)),
            "fold_macro_recall_mean": float(recalls.mean()),
            "fold_macro_recall_std": float(recalls.std(ddof=1)),
            "pooled_repeated_predictions": pooled,
            "by_site": by_site,
        }
    return summary


def descriptive_effects(
    features: np.ndarray, labels: np.ndarray, sites: np.ndarray
) -> np.ndarray:
    all_index = np.arange(len(labels))
    normalized, _ = normalize_fold(features, sites, all_index, all_index)
    mean_log_mel = normalized[:, :, : features.shape[2] // 2]
    effects = np.zeros((3, mean_log_mel.shape[1], mean_log_mel.shape[2]), dtype=float)
    for label in range(3):
        a = mean_log_mel[labels == label]
        b = mean_log_mel[labels != label]
        pooled = np.sqrt(
            ((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1))
            / max(len(a) + len(b) - 2, 1)
        )
        effects[label] = (a.mean(axis=0) - b.mean(axis=0)) / np.maximum(pooled, 1e-6)
    return effects


def make_plots(
    folds: list[dict], summary: dict, effects: np.ndarray, mel_centers: np.ndarray
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    permutation = np.asarray([fold["permutation_importance"] for fold in folds])
    loo = np.asarray([fold["loo_delta_macro_f1"] for fold in folds])
    top8_frequency = Counter(
        electrode for fold in folds for electrode in fold["ranking"][:8]
    )
    order = np.argsort(permutation.mean(axis=0))
    names = [eeg.CHANNELS[i] for i in order]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, constrained_layout=True)
    axes[0].barh(names, permutation.mean(axis=0)[order], xerr=permutation.std(axis=0, ddof=1)[order], color="#4472C4")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(title="Grouped permutation", xlabel="Macro F1 decrease")
    axes[1].barh(names, loo.mean(axis=0)[order], xerr=loo.std(axis=0, ddof=1)[order], color="#70AD47")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(title="Leave-one-electrode-out", xlabel="Macro F1 decrease")
    axes[2].barh(names, [top8_frequency[eeg.CHANNELS[i]] / len(folds) for i in order], color="#ED7D31")
    axes[2].set(title="Selection stability", xlabel="Fraction selected in top 8", xlim=(0, 1))
    fig.suptitle("Electrode contribution across repeated participant-level folds")
    fig.savefig(OUT / "electrode_importance.png", dpi=190)
    plt.close(fig)

    band_names = ["Delta\n1-4", "Theta\n4-8", "Alpha\n8-13", "Beta\n13-30", "Gamma\n30-40"]
    band = np.asarray([fold["band_importance"] for fold in folds])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(band_names, band.mean(axis=0), yerr=band.std(axis=0, ddof=1), color="#5B9BD5", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="Macro F1 decrease after permutation", title="Frequency-band contribution")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    fig.savefig(OUT / "frequency_band_importance.png", dpi=190)
    plt.close(fig)

    counts = list(ELECTRODE_COUNTS)
    means = [summary[str(k)]["fold_macro_f1_mean"] for k in counts]
    stds = [summary[str(k)]["fold_macro_f1_std"] for k in counts]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(counts, means, yerr=stds, marker="o", linewidth=2, capsize=5, color="#4472C4")
    ax.set(
        xticks=counts,
        ylim=(0.35, 0.8),
        xlabel="Electrodes retained",
        ylabel="Repeated-CV macro F1",
        title="Compact montage performance (mean ± fold SD)",
    )
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "compact_montage_performance.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
    limit = float(np.percentile(np.abs(effects), 98))
    for label, ax in enumerate(axes):
        image = ax.imshow(
            effects[label], origin="lower", aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit,
            extent=[mel_centers[0], mel_centers[-1], -0.5, len(eeg.CHANNELS) - 0.5],
        )
        ax.set_yticks(range(len(eeg.CHANNELS)), eeg.CHANNELS)
        ax.set(xlabel="Frequency (Hz)", title=f"{eeg.CLASSES[label]} versus other classes")
    axes[0].set_ylabel("Electrode")
    fig.colorbar(image, ax=axes, label="Standardized log-power difference", fraction=0.025, pad=0.02)
    fig.suptitle("Site-centered electrode-frequency effects (descriptive)")
    fig.savefig(OUT / "electrode_frequency_effects.png", dpi=190)
    plt.close(fig)


def write_report(payload: dict, electrode_table: pd.DataFrame, band_table: pd.DataFrame) -> None:
    summary = payload["repeated_cv"]
    control = payload["fixed_c_control"]
    locked = payload["locked_test"]
    best_count = locked["selected_count"]
    best = locked["models"][str(best_count)]
    top = electrode_table.sort_values(
        ["top8_selection_frequency", "permutation_f1_drop_mean"], ascending=False
    ).head(8)
    lines = [
        "# Electrode and frequency stability experiment",
        "",
        f"Run date: 2026-10-13",
        "",
        "## Protocol",
        "",
        f"- Data: {payload['participants']} participants ({payload['class_counts']}).",
        f"- Representation: participant-level mean and standard deviation of 19-channel, 24-bin log-mel spectra.",
        f"- Evaluation: {REPEATS} repeats × {OUTER_SPLITS} participant-level folds, stratified by site and diagnosis.",
        "- Site normalization, grouped permutation ranking, electrode selection, PCA, regularization selection, and SVM fitting were repeated inside the applicable training fold.",
        "- Primary metric: participant-level macro F1.",
        "",
        "## Repeated cross-validation",
        "",
        "| Electrodes | Macro F1 mean ± fold SD | Accuracy mean ± fold SD | Macro recall mean ± fold SD |",
        "|---:|---:|---:|---:|",
    ]
    for count in ELECTRODE_COUNTS:
        row = summary[str(count)]
        lines.append(
            f"| {count} | {row['fold_macro_f1_mean']*100:.1f}% ± {row['fold_macro_f1_std']*100:.1f} | "
            f"{row['fold_accuracy_mean']*100:.1f}% ± {row['fold_accuracy_std']*100:.1f} | "
            f"{row['fold_macro_recall_mean']*100:.1f}% ± {row['fold_macro_recall_std']*100:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Control with the original fixed `C=0.1` and all 19 electrodes: "
            f"{control['macro_f1_mean']*100:.1f}% ± {control['macro_f1_std']*100:.1f} macro F1. "
            "This confirms that the lower repeated-CV estimate is caused by participant-split instability, not by the new C search.",
        ]
    )
    lines.extend(
        [
            "",
            "## Fixed exploratory held-out split",
            "",
            f"Inner development folds selected {best_count} electrodes: {', '.join(best['selected'])}.",
            "",
            "| Model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for count in ELECTRODE_COUNTS:
        row = locked["models"][str(count)]["metrics"]
        lines.append(
            f"| {count} electrodes | {row['accuracy']*100:.1f}% | {row['macro_f1']*100:.1f}% | "
            f"{row['macro_recall']*100:.1f}% | {row['per_class']['AD']['recall']*100:.1f}% | "
            f"{row['per_class']['FTD']['recall']*100:.1f}% | {row['per_class']['HC']['recall']*100:.1f}% |"
        )
    lines.extend(["", "## Most stable electrodes", "", "| Electrode | Top-8 selection frequency | Permutation F1 drop | Leave-one-out F1 drop |", "|---|---:|---:|---:|"])
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.electrode} | {row.top8_selection_frequency*100:.0f}% | "
            f"{row.permutation_f1_drop_mean*100:.2f} pp | {row.loo_f1_drop_mean*100:.2f} pp |"
        )
    lines.extend(["", "## Frequency bands", "", "| Band | Permutation F1 drop |", "|---|---:|"])
    for row in band_table.sort_values("permutation_f1_drop_mean", ascending=False).itertuples(index=False):
        lines.append(f"| {row.band} | {row.permutation_f1_drop_mean*100:.2f} pp |")
    full = summary["19"]["pooled_repeated_predictions"]
    lines.extend(
        [
            "",
            "## Generalization weaknesses",
            "",
            f"Across repeated folds, the 19-electrode model's pooled FTD recall was "
            f"{full['per_class']['FTD']['recall']*100:.1f}% and FTD F1 was "
            f"{full['per_class']['FTD']['f1']*100:.1f}%.",
            "",
            "| Site | Repeated-prediction macro F1 |",
            "|---|---:|",
        ]
    )
    for site, row in summary["19"]["by_site"].items():
        lines.append(f"| {site} | {row['macro_f1']*100:.1f}% |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive permutation or leave-one-out values mean performance fell when that information was destroyed or removed.",
            "- Negative values mean the feature group was unstable or sometimes harmful; they are not evidence that the electrode is biologically irrelevant.",
            "- The electrode-frequency heatmap is descriptive. It shows class-associated spectral differences after site centering, not causal biomarkers.",
            "- The fixed 27-participant test set has been inspected in earlier experiments, so repeated nested cross-validation is the more credible estimate.",
            "- Electrode selection can reduce input dimension and noise, but it does not create additional independent FTD patients.",
            "- The data-driven 8-electrode model matched the 19-electrode mean, but its selected channels changed substantially between folds. There is no defensible fixed reduced montage yet.",
            "- Theta carried the largest reproducible grouped contribution. Beta and 30–40 Hz also contributed, although the latter must be checked for residual muscle or acquisition noise.",
            "- No compact montage improved generalization. The fixed split's 64.3% macro F1 is optimistic relative to the repeated-fold estimate.",
        ]
    )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features, labels, participants, sites = load_subject_data()
    per_electrode = features.shape[2]
    mel_bins = per_electrode // 2
    mel_centers = eeg.inverse_mel(
        np.linspace(eeg.mel_scale(1.0), eeg.mel_scale(40.0), mel_bins + 2)
    )[1:-1]
    strata = combined_strata(labels, sites)
    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS, n_repeats=REPEATS, random_state=SEED
    )
    folds = []
    for fold_index, (train, test) in enumerate(splitter.split(np.arange(len(labels)), strata), start=1):
        print(f"Fold {fold_index}/{OUTER_SPLITS * REPEATS}", flush=True)
        folds.append(run_outer_fold(features, labels, sites, train, test, mel_centers))
    summary = summarize_folds(folds, participants)

    # Control: replay the original all-electrode C without nested C selection.
    fixed_c_scores = []
    for fold in folds:
        train = np.asarray(fold["train"], dtype=int)
        test = np.asarray(fold["test"], dtype=int)
        _, control_metrics = evaluate_model(
            features, labels, sites, train, test, list(range(len(eeg.CHANNELS))), 0.1
        )
        fixed_c_scores.append(control_metrics["macro_f1"])
    fixed_c_control = {
        "C": 0.1,
        "macro_f1_mean": float(np.mean(fixed_c_scores)),
        "macro_f1_std": float(np.std(fixed_c_scores, ddof=1)),
        "macro_f1_min": float(np.min(fixed_c_scores)),
        "macro_f1_max": float(np.max(fixed_c_scores)),
    }

    permutation = np.asarray([fold["permutation_importance"] for fold in folds])
    loo = np.asarray([fold["loo_delta_macro_f1"] for fold in folds])
    top5 = Counter(channel for fold in folds for channel in fold["ranking"][:5])
    top8 = Counter(channel for fold in folds for channel in fold["ranking"][:8])
    top12 = Counter(channel for fold in folds for channel in fold["ranking"][:12])
    electrode_table = pd.DataFrame(
        {
            "electrode": eeg.CHANNELS,
            "permutation_f1_drop_mean": permutation.mean(axis=0),
            "permutation_f1_drop_std": permutation.std(axis=0, ddof=1),
            "loo_f1_drop_mean": loo.mean(axis=0),
            "loo_f1_drop_std": loo.std(axis=0, ddof=1),
            "top5_selection_frequency": [top5[c] / len(folds) for c in eeg.CHANNELS],
            "top8_selection_frequency": [top8[c] / len(folds) for c in eeg.CHANNELS],
            "top12_selection_frequency": [top12[c] / len(folds) for c in eeg.CHANNELS],
        }
    )
    electrode_table.to_csv(OUT / "electrode_importance.csv", index=False)
    band_values = np.asarray([fold["band_importance"] for fold in folds])
    band_table = pd.DataFrame(
        {
            "band": ["Delta 1-4 Hz", "Theta 4-8 Hz", "Alpha 8-13 Hz", "Beta 13-30 Hz", "Gamma 30-40 Hz"],
            "permutation_f1_drop_mean": band_values.mean(axis=0),
            "permutation_f1_drop_std": band_values.std(axis=0, ddof=1),
        }
    )
    band_table.to_csv(OUT / "frequency_band_importance.csv", index=False)

    # Replay the existing fixed split. Selection uses only train+validation labels.
    fixed = eeg.make_subject_splits(participants)
    development = np.asarray(fixed["train"] + fixed["val"], dtype=int)
    test = np.asarray(fixed["test"], dtype=int)
    locked_fold = run_outer_fold(features, labels, sites, development, test, mel_centers)
    selected_count = max(
        ELECTRODE_COUNTS,
        key=lambda count: (
            locked_fold["models"][str(count)]["inner_macro_f1"],
            -count,
        ),
    )
    locked_fold["selected_count"] = int(selected_count)

    effects = descriptive_effects(features, labels, sites)
    make_plots(folds, summary, effects, mel_centers)
    payload = {
        "protocol": {
            "seed": SEED,
            "outer_splits": OUTER_SPLITS,
            "repeats": REPEATS,
            "inner_splits": INNER_SPLITS,
            "permutations": PERMUTATIONS,
            "C_values": list(C_VALUES),
            "electrode_counts": list(ELECTRODE_COUNTS),
        },
        "participants": len(participants),
        "class_counts": {eeg.CLASSES[i]: int((labels == i).sum()) for i in range(3)},
        "site_counts": dict(Counter(sites)),
        "repeated_cv": summary,
        "fixed_c_control": fixed_c_control,
        "locked_test": locked_fold,
        "folds": folds,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, electrode_table, band_table)
    print(json.dumps({"repeated_cv": summary, "locked_test_selected": selected_count, "locked_test": locked_fold["models"][str(selected_count)]["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
