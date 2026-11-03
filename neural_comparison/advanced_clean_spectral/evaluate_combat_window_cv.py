"""ComBat harmonization and window-level training, on the same 5 x 5 participant folds.

Stages (run separately, then merge):
  --stage subject   participant-level SVMs with site_zscore / ComBat(dx) / ComBat(dx+age+sex)
  --stage window    window-level logistic regression, participant decision = mean window log-probability
  --stage report    merge stages (+ the earlier frozen-EEGPT run), paired participant bootstrap, figure
  --selftest        synthetic check of the ComBat implementation

ComBat is fit on training participants only (site location/scale with empirical-Bayes
shrinkage, diagnosis/age/sex protected during estimation) and then applied to held-out
participants using only their site, never their label or demographics.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import audit_preprocessing_feature_svm as audit
import evaluate_frozen_eegpt_cv as base

eeg = audit.eeg
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs" / "combat_window_cv"
SEED = audit.SEED
NORMALIZER_SETS = ["site_zscore", "combat_dx", "combat_dx_age_sex"]
WINDOW_NORMALIZERS = ["site_zscore", "combat_dx"]
PCA_SINGLE = [10, 20, 40]
PCA_FUSED = [10, 20]
PCA_WINDOW = [20, 50]
WINDOW_C = [0.003, 0.03, 0.3]
FTD_BOOSTS = [1.0, 1.5]


class ComBat:
    """Parametric empirical-Bayes ComBat that can be applied to new participants."""

    def fit(self, x: np.ndarray, sites: np.ndarray, covariates: np.ndarray) -> "ComBat":
        x = np.asarray(x, dtype=np.float64)
        self.levels = sorted(set(sites))
        site_design = np.stack([sites == level for level in self.levels], axis=1).astype(np.float64)
        design = np.hstack([site_design, covariates])
        beta = np.linalg.lstsq(design, x, rcond=None)[0]
        counts = site_design.sum(axis=0)
        k = len(self.levels)
        self.grand = (counts / counts.sum()) @ beta[:k]
        covariate_beta = beta[k:]
        # Held-out participants have no usable label, so they are centred at the average covariate profile.
        self.cov_shift = covariates.mean(axis=0) @ covariate_beta
        self.variance = np.maximum(((x - design @ beta) ** 2).mean(axis=0), 1e-8)
        z = (x - (self.grand + covariates @ covariate_beta)) / np.sqrt(self.variance)
        self.gamma, self.delta = {}, {}
        for level in self.levels:
            zs = z[sites == level]
            n = len(zs)
            g_hat, d_hat = zs.mean(axis=0), zs.var(axis=0, ddof=1)
            g_bar, tau2 = g_hat.mean(), max(g_hat.var(ddof=1), 1e-12)
            m, s2 = d_hat.mean(), max(d_hat.var(ddof=1), 1e-12)
            a_prior, b_prior = (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2
            d_star = d_hat.copy()
            for _ in range(200):
                g_star = (n * tau2 * g_hat + d_star * g_bar) / (n * tau2 + d_star)
                d_new = (0.5 * ((zs - g_star) ** 2).sum(axis=0) + b_prior) / (n / 2 + a_prior - 1)
                change = float(np.max(np.abs(d_new - d_star) / np.maximum(d_star, 1e-12)))
                d_star = d_new
                if change < 1e-4:
                    break
            self.gamma[level] = (n * tau2 * g_hat + d_star * g_bar) / (n * tau2 + d_star)
            self.delta[level] = d_star
        return self

    def transform(self, x: np.ndarray, sites: np.ndarray) -> np.ndarray:
        centre = self.grand + self.cov_shift
        root = np.sqrt(self.variance)
        z = (np.asarray(x, dtype=np.float64) - centre) / root
        for level in self.levels:
            mask = sites == level
            z[mask] = (z[mask] - self.gamma[level]) / np.sqrt(self.delta[level])
        return (z * root + centre).astype(np.float32)


def selftest() -> None:
    rng = np.random.default_rng(1)
    n, p = 240, 300
    sites = np.repeat(np.array(["A", "B", "C"]), n // 3)
    label = rng.integers(0, 2, n)
    signal = np.zeros(p)
    signal[:20] = 1.0
    shift = {"A": 0.0, "B": 1.5, "C": -1.0}
    scale = {"A": 1.0, "B": 2.0, "C": 0.6}
    x = np.stack([rng.normal(shift[s], scale[s], p) for s in sites]) + label[:, None] * signal
    train = np.arange(n) % 4 != 0
    test = ~train
    model = ComBat().fit(x[train], sites[train], label[train, None].astype(float))
    adj_train, adj_test = model.transform(x[train], sites[train]), model.transform(x[test], sites[test])
    for name, values, mask in [("train", adj_train, train), ("held-out", adj_test, test)]:
        means = [float(values[sites[mask] == s][:, 40:].mean()) for s in ["A", "B", "C"]]
        stds = [float(values[sites[mask] == s][:, 40:].std()) for s in ["A", "B", "C"]]
        print(f"{name}: null-feature site means {np.round(means, 2)}, stds {np.round(stds, 2)} (raw means 0/1.5/-1, stds 1/2/0.6)")
    effect = adj_test[label[test] == 1][:, :20].mean() - adj_test[label[test] == 0][:, :20].mean()
    print(f"held-out diagnosis effect on signal features after ComBat: {effect:.2f} (true 1.0)")


def load_demographics(participants: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    root = ROOT / "dataset"
    ds = pd.read_csv(root / "ds004504" / "participants.tsv", sep="\t")
    frames = [
        pd.read_csv(root / "brainlat" / "EEG" / folder / f"Demographics_{name}_EEG_data.csv")
        for folder, name in [("1_AD", "AD"), ("2_bvFTD", "bvFTD"), ("5_HC", "HC")]
    ]
    brainlat = pd.concat(frames).drop_duplicates("id EEG").set_index("id EEG")
    age, male = np.full(len(participants), np.nan), np.full(len(participants), np.nan)
    for i, row in enumerate(participants):
        sid = row["subject_id"].split(":")[-1]
        if row["source"] == "ds004504":
            match = ds[ds.participant_id == sid]
            if len(match):
                age[i], male[i] = float(match.Age.iloc[0]), float(match.Gender.iloc[0] == "M")
        elif sid in brainlat.index:
            age[i], male[i] = float(brainlat.loc[sid, "Age"]), float(brainlat.loc[sid, "sex"])
    return age, male


def covariates(method: str, labels: np.ndarray, age: np.ndarray, male: np.ndarray, train: np.ndarray) -> np.ndarray:
    columns = [(labels[train] == 1).astype(float), (labels[train] == 2).astype(float)]
    if method == "combat_dx_age_sex":
        a, m = age[train], male[train]
        a = np.where(np.isnan(a), np.nanmean(a), a)
        m = np.where(np.isnan(m), np.nanmean(m), m)
        columns += [(a - a.mean()) / max(a.std(), 1e-6), m - m.mean()]
    return np.stack(columns, axis=1)


def harmonize(
    matrix: np.ndarray, sites: np.ndarray, labels: np.ndarray, age: np.ndarray, male: np.ndarray,
    train: np.ndarray, evaluate: np.ndarray, method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not method.startswith("combat"):
        return audit.normalize_fold(matrix, sites, train, evaluate, method)
    model = ComBat().fit(matrix[train], sites[train], covariates(method, labels, age, male, train))
    return model.transform(matrix[train], sites[train]), model.transform(matrix[evaluate], sites[evaluate])


def prepare_blocks(
    blocks: dict[str, np.ndarray], names: list[str], sites, labels, age, male, train, evaluate, method, components,
) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for name in names:
        fit_x, eval_x = harmonize(blocks[name], sites, labels, age, male, train, evaluate, method)
        parts.append(audit.reduce_fold(fit_x, eval_x, components))
    if len(parts) > 1:
        energy = [float((fx ** 2).sum(axis=1).mean()) for fx, _ in parts]
        target = float(np.mean(energy))
        parts = [(fx * np.sqrt(target / e), ex * np.sqrt(target / e)) for (fx, ex), e in zip(parts, energy)]
    return np.hstack([p[0] for p in parts]), np.hstack([p[1] for p in parts])


def rank(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    candidates.sort(key=lambda r: (r["macro_f1"], r["ftd_recall"], r["macro_recall"], r["accuracy"]), reverse=True)
    return candidates[0]


def average_records(records: dict[tuple, list[dict[str, float]]]) -> list[dict[str, Any]]:
    return [
        {"pca": key[0], "C": key[1], "ftd_boost": key[2], **{n: float(np.mean([r[n] for r in rows])) for n in rows[0]}}
        for key, rows in records.items()
    ]


def inner_folds(labels: np.ndarray, sites: np.ndarray, train: np.ndarray):
    strata = np.asarray([f"{sites[i]}:{labels[i]}" for i in train])
    return [
        (train[a], train[b])
        for a, b in StratifiedKFold(n_splits=audit.INNER_SPLITS, shuffle=True, random_state=SEED).split(train, strata)
    ]


def subject_job(blocks, names, labels, sites, age, male, method: str, components_grid: list, c_values: list) -> Callable:
    def run(train: np.ndarray, test: np.ndarray):
        records: dict[tuple, list] = {}
        for fit, val in inner_folds(labels, sites, train):
            for components in components_grid:
                fit_x, val_x = prepare_blocks(blocks, names, sites, labels, age, male, fit, val, method, components)
                for c_value, boost in itertools.product(c_values, FTD_BOOSTS):
                    prediction = base.classifier(labels[fit], c_value, boost).fit(fit_x, labels[fit]).predict(val_x)
                    records.setdefault((components, c_value, boost), []).append(audit.score(labels[val], prediction))
        best = rank(average_records(records))
        fit_x, test_x = prepare_blocks(blocks, names, sites, labels, age, male, train, test, method, best["pca"])
        model = base.classifier(labels[train], best["C"], best["ftd_boost"]).fit(fit_x, labels[train])
        return {"normalization": method, **best}, model.predict(test_x), model.decision_function(test_x)

    return run


def window_job(features, subject_of_window, labels, sites, age, male, method: str) -> Callable:
    counts = np.bincount(subject_of_window, minlength=len(labels)).astype(float)

    def window_fit_predict(train, test, components, c_value, boost):
        train_w = np.flatnonzero(np.isin(subject_of_window, train))
        test_w = np.flatnonzero(np.isin(subject_of_window, test))
        owner_train, owner_test = subject_of_window[train_w], subject_of_window[test_w]
        fit_x, test_x = harmonize_windows(features, subject_of_window, labels, sites, age, male, train, train_w, test_w, method)
        fit_x, test_x = audit.reduce_fold(fit_x, test_x, components)
        class_weight = audit.weights_for(labels[train], boost)
        weights = np.asarray([class_weight[labels[s]] for s in owner_train]) / counts[owner_train] * counts[train].mean()
        model = LogisticRegression(C=c_value, max_iter=400).fit(fit_x, labels[owner_train], sample_weight=weights)
        log_probability = np.log(np.maximum(model.predict_proba(test_x), 1e-9))
        decision = np.zeros((len(labels), 3))
        np.add.at(decision, owner_test, log_probability)
        decision = decision[test] / counts[test, None]
        return decision

    def run(train: np.ndarray, test: np.ndarray):
        records: dict[tuple, list] = {}
        for fit, val in inner_folds(labels, sites, train):
            for components, c_value, boost in itertools.product(PCA_WINDOW, WINDOW_C, FTD_BOOSTS):
                decision = window_fit_predict(fit, val, components, c_value, boost)
                records.setdefault((components, c_value, boost), []).append(
                    audit.score(labels[val], np.argmax(decision, axis=1))
                )
        best = rank(average_records(records))
        decision = window_fit_predict(train, test, best["pca"], best["C"], best["ftd_boost"])
        return {"normalization": method, **best}, np.argmax(decision, axis=1), decision

    return run


def harmonize_windows(features, owner, labels, sites, age, male, train, train_w, test_w, method):
    """Harmonize windows; ComBat covariates come from each window's training participant only."""
    site_w = sites[owner]
    if not method.startswith("combat"):
        return audit.normalize_fold(features, site_w, train_w, test_w, method)
    per_subject = covariates(method, labels, age, male, train)
    table = np.zeros((len(labels), per_subject.shape[1]))
    table[train] = per_subject
    model = ComBat().fit(features[train_w], site_w[train_w], table[owner[train_w]])
    return model.transform(features[train_w], site_w[train_w]), model.transform(features[test_w], site_w[test_w])


def load_windows(participants: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    embeddings = np.load(base.EMBEDDINGS).astype(np.float32)
    eegpt_owner = np.load(base.EEGPT_PREPARED / "subject_index.npy").astype(int)
    tensor = np.load(base.MOD / "log_mel.npy", mmap_mode="r")
    spectral_owner = np.load(base.MOD / "subject_index.npy").astype(int)
    pieces = []
    for start in range(0, len(tensor), 500):
        chunk = np.asarray(tensor[start : start + 500], dtype=np.float32)
        pieces.append(np.concatenate([chunk.mean(axis=3), chunk.std(axis=3)], axis=2).reshape(len(chunk), -1))
    return {
        "win_spectral": (np.nan_to_num(np.concatenate(pieces)), spectral_owner),
        "win_eegpt": (embeddings, eegpt_owner),
    }


def run_stage(
    jobs: dict[str, Callable], late_pairs: dict[str, tuple[str, str]], labels, sites, repeats: int
) -> dict[str, Any]:
    strata = np.asarray([f"{sites[i]}:{labels[i]}" for i in range(len(labels))])
    folds = list(RepeatedStratifiedKFold(n_splits=audit.OUTER_SPLITS, n_repeats=repeats, random_state=SEED)
                 .split(np.arange(len(labels)), strata))
    results = {key: {"folds": []} for key in [*jobs, *late_pairs]}
    for number, (train, test) in enumerate(folds, 1):
        decisions = {}
        for key, job in jobs.items():
            config, prediction, decision = job(train, test)
            decisions[key] = decision
            results[key]["folds"].append({"fold": number, "test": test.tolist(), "prediction": prediction.tolist(),
                                          "metrics": audit.score(labels[test], prediction), "selected": config})
        for key, (a, b) in late_pairs.items():
            blend = 0.5 * audit.softmax(decisions[a]) + 0.5 * audit.softmax(decisions[b])
            prediction = np.argmax(blend, axis=1)
            results[key]["folds"].append({"fold": number, "test": test.tolist(), "prediction": prediction.tolist(),
                                          "metrics": audit.score(labels[test], prediction), "selected": None})
        print(f"  fold {number}/{len(folds)}: " + ", ".join(
            f"{k}={results[k]['folds'][-1]['metrics']['macro_f1']:.2f}" for k in results), flush=True)
    return results


def oof(folds: list[dict[str, Any]], n: int, repeats: int) -> np.ndarray:
    predictions = np.zeros((repeats, n), dtype=int)
    for number, fold in enumerate(folds):
        predictions[number // audit.OUTER_SPLITS, fold["test"]] = fold["prediction"]
    return predictions


def save_stage(name: str, results: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stage_{name}.json").write_text(json.dumps({k: v["folds"] for k, v in results.items()}), encoding="utf-8")


def stage_subject(repeats: int) -> None:
    blocks, labels, sites, participants = base.load_data()
    age, male = load_demographics(participants)
    print(f"demographics missing: age={int(np.isnan(age).sum())}, sex={int(np.isnan(male).sum())}", flush=True)
    blocks["demographics"] = np.column_stack(
        [np.nan_to_num(age, nan=np.nanmedian(age)), np.nan_to_num(male, nan=np.nanmedian(male))]
        + [(sites == s).astype(float) for s in sorted(set(sites))]
    ).astype(np.float32)
    jobs, late = {}, {}
    for method in NORMALIZER_SETS:
        jobs[f"spectral|{method}"] = subject_job(blocks, ["spectral"], labels, sites, age, male, method, PCA_SINGLE, base.C_VALUES)
        jobs[f"eegpt2048|{method}"] = subject_job(blocks, ["eegpt2048"], labels, sites, age, male, method, PCA_SINGLE, base.C_VALUES)
        jobs[f"early_fusion|{method}"] = subject_job(blocks, ["spectral", "eegpt2048"], labels, sites, age, male, method, PCA_FUSED, base.C_VALUES)
        late[f"late_fusion|{method}"] = (f"spectral|{method}", f"eegpt2048|{method}")
    jobs["demographics_only"] = subject_job(blocks, ["demographics"], labels, sites, age, male, "global", [None], base.C_VALUES)
    save_stage("subject", run_stage(jobs, late, labels, sites, repeats))


def stage_window(repeats: int) -> None:
    blocks, labels, sites, participants = base.load_data()
    age, male = load_demographics(participants)
    windows = load_windows(participants)
    jobs, late = {}, {}
    for method in WINDOW_NORMALIZERS:
        for name, (features, owner) in windows.items():
            jobs[f"{name}|{method}"] = window_job(features, owner, labels, sites, age, male, method)
        late[f"win_late_fusion|{method}"] = (f"win_spectral|{method}", f"win_eegpt|{method}")
    save_stage("window", run_stage(jobs, late, labels, sites, repeats))


def bootstrap_pairs(predictions: dict[str, np.ndarray], labels: np.ndarray, pairs: list[tuple[str, str]], draws=2000):
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(0, len(labels), size=(draws, len(labels)))

    def score(key: str, index: np.ndarray) -> float:
        return float(np.mean([base.fast_macro_f1(labels[index], row[index]) for row in predictions[key]]))

    cache: dict[str, np.ndarray] = {}
    out = {}
    for a, b in pairs:
        for key in (a, b):
            if key not in cache:
                cache[key] = np.asarray([score(key, index) for index in resamples])
        diff = cache[a] - cache[b]
        out[f"{a} - {b}"] = {"mean_difference": float(diff.mean()),
                              "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                              "probability_a_better": float((diff > 0).mean())}
    return out


def report(repeats: int) -> None:
    _, labels, _, _ = base.load_data()
    folds: dict[str, list] = {}
    for stage in ["subject", "window"]:
        path = OUT / f"stage_{stage}.json"
        if path.exists():
            folds.update(json.loads(path.read_text(encoding="utf-8")))
    previous = json.loads((base.OUT / "frozen_eegpt_cv.json").read_text(encoding="utf-8"))["folds"]
    for key in ["spectral", "eegpt2048", "early_fusion"]:
        folds[f"previous_selected|{key}"] = previous[key]
    predictions, summaries = {}, {}
    for key, entries in folds.items():
        predictions[key] = oof(entries, len(labels), repeats)
        summaries[key] = base.summarize({"folds": entries, "oof_predictions": predictions[key]}, labels)
    pairs = []
    for model in ["spectral", "eegpt2048", "early_fusion", "late_fusion"]:
        pairs += [(f"{model}|combat_dx", f"{model}|site_zscore"), (f"{model}|combat_dx_age_sex", f"{model}|combat_dx")]
    for name in ["win_spectral", "win_eegpt", "win_late_fusion"]:
        for method in WINDOW_NORMALIZERS:
            subject_key = {"win_spectral": "spectral", "win_eegpt": "eegpt2048", "win_late_fusion": "late_fusion"}[name]
            pairs.append((f"{name}|{method}", f"{subject_key}|{method}"))
        pairs.append((f"{name}|combat_dx", f"{name}|site_zscore"))
    pairs += [(key, "spectral|site_zscore") for key in summaries if key not in ("spectral|site_zscore",) and "demographics" not in key]
    pairs = [(a, b) for a, b in pairs if a in predictions and b in predictions]
    bootstrap = bootstrap_pairs(predictions, labels, pairs)
    (OUT / "combat_window_cv.json").write_text(
        json.dumps({"summary": summaries, "paired_bootstrap": bootstrap}, indent=2), encoding="utf-8")

    lines = ["| Model | Accuracy | Macro F1 (25 folds) | Pooled F1 (5 repeats) | FTD recall |", "|---|---:|---:|---:|---:|"]
    for key, row in summaries.items():
        pooled = row["pooled_macro_f1_per_repeat"]
        lines.append(f"| {key} | {row['accuracy']['mean']:.3f} | {row['macro_f1']['mean']:.3f} +/- {row['macro_f1']['std']:.3f} "
                     f"| {pooled['mean']:.3f} | {row['ftd_recall']['mean']:.3f} |")
    lines += ["", "| Paired comparison (pooled macro F1) | Difference | 95% CI | P(first better) |", "|---|---:|---|---:|"]
    for name, row in bootstrap.items():
        lines.append(f"| {name} | {row['mean_difference']:+.3f} | [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}] | {row['probability_a_better']:.2f} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    keys = [k for k in summaries if k != "demographics_only"]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.28 * len(keys))), constrained_layout=True)
    values = [summaries[k]["pooled_macro_f1_per_repeat"]["mean"] for k in keys]
    errors = [summaries[k]["pooled_macro_f1_per_repeat"]["std"] for k in keys]
    ax.barh(range(len(keys)), values, xerr=errors, color="#4c78a8", capsize=2)
    ax.axvline(1 / 3, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="Pooled macro F1 over 174 participants (mean +/- SD of 5 repeats)", title="ComBat and window-level ablations")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUT / "combat_window_cv.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="ComBat and window-level repeated nested CV")
    parser.add_argument("--stage", choices=["subject", "window", "report"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.stage == "subject":
        stage_subject(args.repeats)
    elif args.stage == "window":
        stage_window(args.repeats)
    elif args.stage == "report":
        report(args.repeats)
    else:
        parser.error("choose --stage or --selftest")


if __name__ == "__main__":
    main()
