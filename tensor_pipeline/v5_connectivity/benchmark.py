"""Nested subject-level benchmark for connectivity, Riemannian and bag models."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "tensor_pipeline/outputs/v5_connectivity"
sys.path.insert(0, str(HERE))
from models import (HierarchicalModel, LogEuclideanTangent, aligned_scores,
                    candidate_configs, fit_pipeline)


def signature() -> str:
    h = hashlib.sha256()
    for path in (Path(__file__), HERE / "models.py", HERE / "prepare_features.py", OUT / "feature_report.json"):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "recall": recall_score(y, prediction, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "confusion": [[int(np.sum((y == i) & (prediction == j))) for j in range(3)] for i in range(3)],
    }


def load_all():
    data = {n: dict(np.load(OUT / f"bags_{n}.npz")) for n in (1, 4, 8)}
    subjects = data[1]["subject"].astype(str)
    labels = data[1]["label"].astype(int)
    assert len(subjects) == len(np.unique(subjects)) == 88
    assert np.array_equal(np.bincount(labels), [36, 23, 29])
    for n, d in data.items():
        counts = Counter(d["subject"].astype(str))
        assert set(counts.values()) == {n}
        for s, label in zip(subjects, labels):
            assert np.all(d["label"][d["subject"].astype(str) == s] == label)
    return data, subjects, labels


def bag_mask(d: dict, selected_subjects: np.ndarray) -> np.ndarray:
    return np.isin(d["subject"].astype(str), selected_subjects)


def feature_bank(d: dict, train_mask: np.ndarray):
    mapper = LogEuclideanTangent().fit(d["covariance"][train_mask])
    riemann = mapper.transform(d["covariance"])
    power = d["power"].astype(float)
    connectivity = d["connectivity"].astype(float)
    return {
        "power": power,
        "riemann": riemann,
        "connectivity": connectivity,
        "hybrid": np.concatenate([riemann, connectivity, power], axis=1),
    }, mapper


def subject_scores(scores: np.ndarray, bag_subjects: np.ndarray, ordered_subjects: np.ndarray) -> np.ndarray:
    result = []
    for subject in ordered_subjects:
        ix = bag_subjects == subject
        assert np.any(ix)
        result.append(scores[ix].mean(axis=0))
    return np.asarray(result)


def predict_subjects(model, x: np.ndarray, bag_subjects: np.ndarray, ordered_subjects: np.ndarray):
    score = model.decision_function(x) if isinstance(model, HierarchicalModel) else aligned_scores(model, x)
    aggregated = subject_scores(score, bag_subjects, ordered_subjects)
    return aggregated.argmax(axis=1), aggregated


def fit_config(cfg: dict, bank: dict, d: dict, train_mask: np.ndarray):
    x = bank[cfg["representation"]]
    y = d["label"].astype(int)
    subjects = d["subject"].astype(str)
    if cfg["mode"] == "hierarchical":
        return HierarchicalModel(cfg).fit(x[train_mask], y[train_mask], subjects[train_mask])
    return fit_pipeline(cfg, x[train_mask], y[train_mask], subjects[train_mask])


def rank_rows(rows: list[dict]) -> list[dict]:
    rep_order = {"power": 0, "riemann": 1, "connectivity": 2, "hybrid": 3}
    clf_order = {"logistic": 0, "linear_svm": 1, "svm": 2, "xgboost": 3}
    def key(row):
        cfg = row["config"]
        complexity = (cfg["bags"], rep_order[cfg["representation"]], clf_order[cfg["classifier"]],
                      cfg.get("selection") or 0, cfg.get("C") or 0, cfg["mode"])
        return (-row["metrics"]["balanced_accuracy"], -row["metrics"]["macro_f1"], complexity,
                json.dumps(cfg, sort_keys=True))
    return sorted(rows, key=key)


def selections(rows: list[dict]) -> dict:
    ranked = rank_rows(rows)
    selected = {"primary": ranked[0]}
    for rep in ("power", "riemann", "connectivity", "hybrid"):
        selected[f"rep_{rep}"] = next(r for r in ranked if r["config"]["representation"] == rep)
    for mode in ("multiclass", "hierarchical"):
        selected[f"mode_{mode}"] = next(r for r in ranked if r["config"]["mode"] == mode)
    for bags in (1, 4, 8):
        selected[f"bags_{bags}"] = next(r for r in ranked if r["config"]["bags"] == bags)
    return selected


def summarize(folds: list[dict]) -> dict:
    names = list(folds[0]["predictions"]) if folds else []
    results = {}
    for name in names:
        repeated = []
        for repeat in range(3):
            group = [f for f in folds if f["repeat"] == repeat]
            if len(group) != 5:
                continue
            truth = np.concatenate([f["truth"] for f in group])
            pred = np.concatenate([f["predictions"][name]["prediction"] for f in group])
            repeated.append(metric(truth, pred))
        if repeated:
            results[name] = {
                "repeats": repeated,
                "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in repeated])),
                "sd_balanced_accuracy": float(np.std([r["balanced_accuracy"] for r in repeated], ddof=1)) if len(repeated) > 1 else None,
                "mean_accuracy": float(np.mean([r["accuracy"] for r in repeated])),
                "mean_macro_f1": float(np.mean([r["macro_f1"] for r in repeated])),
                "mean_recall": np.mean([r["recall"] for r in repeated], axis=0).tolist(),
            }
    result = {"signature": signature(), "completed_folds": len(folds), "results": results,
              "total_seconds": float(sum(f["seconds"] for f in folds))}
    (OUT / "nested_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    data, subjects, y = load_all()
    configs = candidate_configs()
    sig = signature()
    (OUT / "models").mkdir(exist_ok=True)
    protocol = {
        "signature": sig,
        "date": "2026-09-06",
        "subjects": subjects.tolist(),
        "class_counts": np.bincount(y).tolist(),
        "retained_windows": 16824,
        "outer": {"folds": 5, "repeats": 3, "seed": 9060},
        "inner": {"folds": 3, "seed": "90650 + outer_index"},
        "candidate_count": len(configs),
        "candidates": configs,
        "primary_endpoint": "Mean participant-level balanced accuracy of the complete inner-selected pipeline.",
        "aggregation": "All bags from a participant remain in one fold. Decision scores are averaged across held-out bags before participant prediction.",
        "balance": "Equal total training weight per diagnosis and per participant; no oversampling or duplication.",
        "riemannian_provenance": "Log-Euclidean reference and tangent mapper fitted only on the current training participants.",
        "feature_selection": "ANOVA feature selection and scaling fitted only on current training bags.",
        "limits": "Same previously explored 88 participants and outer split seed as V3/V4; no external validation and no claim of independent confirmation.",
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    folds = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9060)
    for split, (train, test) in enumerate(outer.split(subjects, y)):
        path = OUT / f"fold_{split:02d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["signature"] == sig and row["train"] == train.tolist() and row["test"] == test.tolist()
            folds.append(row)
            continue
        started = time.perf_counter()
        print(f"Outer {split + 1}/15: {len(train)} train, {len(test)} held out, {len(configs)} fixed candidates", flush=True)
        oof = np.full((len(configs), len(subjects)), -1, dtype=int)
        inner_records = []
        inner_cv = StratifiedKFold(3, shuffle=True, random_state=90650 + split)
        for inner, (a, b) in enumerate(inner_cv.split(train, y[train])):
            inner_train, validation = train[a], train[b]
            assert not set(inner_train) & set(validation) and not (set(inner_train) | set(validation)) & set(test)
            banks = {}
            for bags, d in data.items():
                mask = bag_mask(d, subjects[inner_train])
                banks[bags] = feature_bank(d, mask)[0]
            for index, cfg in enumerate(configs):
                d, bank = data[cfg["bags"]], banks[cfg["bags"]]
                train_mask = bag_mask(d, subjects[inner_train])
                valid_mask = bag_mask(d, subjects[validation])
                model = fit_config(cfg, bank, d, train_mask)
                pred, _ = predict_subjects(model, bank[cfg["representation"]][valid_mask],
                                           d["subject"].astype(str)[valid_mask], subjects[validation])
                oof[index, validation] = pred
            inner_records.append({"train": inner_train.tolist(), "validation": validation.tolist()})
            print(f"  Inner {inner + 1}/3 complete", flush=True)
        assert np.all(oof[:, train] >= 0)
        scored = [{"config": cfg, "metrics": metric(y[train], oof[i, train])} for i, cfg in enumerate(configs)]
        chosen = selections(scored)

        outer_banks, mappers = {}, {}
        for bags, d in data.items():
            mask = bag_mask(d, subjects[train])
            outer_banks[bags], mappers[bags] = feature_bank(d, mask)
        fitted, predictions, stored = {}, {}, {}
        for name, choice in chosen.items():
            cfg = choice["config"]
            key = json.dumps(cfg, sort_keys=True)
            d, bank = data[cfg["bags"]], outer_banks[cfg["bags"]]
            train_mask = bag_mask(d, subjects[train]); test_mask = bag_mask(d, subjects[test])
            if key not in fitted:
                fitted[key] = fit_config(cfg, bank, d, train_mask)
            model = fitted[key]
            pred, score = predict_subjects(model, bank[cfg["representation"]][test_mask],
                                           d["subject"].astype(str)[test_mask], subjects[test])
            predictions[name] = {"config": cfg, "prediction": pred.tolist(), "scores": score.tolist(),
                                 "inner_metrics": choice["metrics"]}
            stored[name] = {"config": cfg, "model": model, "mapper": mappers[cfg["bags"]]}

        # Fixed diverse hard-vote ensemble; the primary selection breaks ties.
        voters = [predictions[f"rep_{rep}"]["prediction"] for rep in ("power", "riemann", "connectivity", "hybrid")]
        primary = np.asarray(predictions["primary"]["prediction"])
        vote_prediction = []
        for column, fallback in zip(np.asarray(voters).T, primary):
            counts = np.bincount(column, minlength=3)
            winners = np.flatnonzero(counts == counts.max())
            vote_prediction.append(int(fallback if fallback in winners else winners[0]))
        predictions["diverse_vote"] = {"prediction": vote_prediction,
                                         "members": [f"rep_{r}" for r in ("power", "riemann", "connectivity", "hybrid")]}
        joblib.dump({"signature": sig, "train": train, "test": test, "subjects_train": subjects[train],
                     "subjects_test": subjects[test], "pipelines": stored}, OUT / "models" / f"fold_{split:02d}.joblib")
        row = {"signature": sig, "split": split, "repeat": split // 5, "train": train.tolist(), "test": test.tolist(),
               "test_subjects": subjects[test].tolist(), "truth": y[test].tolist(), "inner": inner_records,
               "inner_results": scored, "predictions": predictions, "seconds": time.perf_counter() - started}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        folds.append(row); summarize(folds)
        print(f"Outer {split + 1} saved in {row['seconds']:.1f}s", flush=True)
    report = summarize(folds)
    print(json.dumps({k: {m: v for m, v in row.items() if m != "repeats"} for k, row in report["results"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
