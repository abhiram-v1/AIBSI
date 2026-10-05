"""Nested participant-level dementia (AD+FTD) versus CN benchmark.

Uses previously computed, diagnosis-blind subject features. All selection,
scaling, and classifier fitting are confined to each training partition.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tensor_pipeline/outputs/v6_binary"
V3 = ROOT / "tensor_pipeline/outputs/v3_waveform_full_cohort"
V5 = ROOT / "tensor_pipeline/outputs/v5_connectivity"


def fingerprint(paths):
    result = {}
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        result[str(path.relative_to(ROOT))] = h.hexdigest()
    return result


def load():
    index = np.load(V3 / "index.npz")
    subjects, labels = index["subjects"].astype(str), index["labels"].astype(int)
    bags = np.load(V5 / "bags_1.npz")
    assert len(subjects) == len(set(subjects)) == 88
    assert np.array_equal(np.bincount(labels), [36, 23, 29])
    assert np.array_equal(subjects, bags["subject"].astype(str))
    assert np.array_equal(labels, bags["label"].astype(int))
    x = {"spectral": np.load(V3 / "spectral_controls.npy"),
         "power": bags["power"], "connectivity": bags["connectivity"]}
    assert all(a.shape[0] == 88 and np.isfinite(a).all() for a in x.values())
    return subjects, labels, x


def candidates():
    return [dict(features=features, k=k, model=model, C=c)
            for features, ks in (("spectral", (16, 32, 64)),
                                 ("power", (16, 32, 64)),
                                 ("connectivity", (32, 64, 128)))
            for k in ks for model, cs in (("logistic", (0.1, 1.0)), ("rbf_svm", (1.0, 10.0)))
            for c in cs]


def make_model(config):
    classifier = (LogisticRegression(C=config["C"], class_weight="balanced", max_iter=3000)
                  if config["model"] == "logistic" else
                  SVC(C=config["C"], kernel="rbf", gamma="scale", class_weight="balanced"))
    return Pipeline([("select", SelectKBest(f_classif, k=config["k"])),
                     ("scale", StandardScaler()), ("classifier", classifier)])


def metrics(y, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
            "dementia_f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
            "dementia_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
            "dementia_precision": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
            "cn_specificity": float(tn / (tn + fp)),
            "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}


def run():
    start = time.perf_counter()
    subjects, original, x = load()
    y = (original != 2).astype(int)  # AD + FTD = dementia; CN = negative.
    configs = candidates()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [V3 / "index.npz", V3 / "spectral_controls.npy", V5 / "bags_1.npz", Path(__file__)]
    protocol = {"task": "AD+FTD (positive) versus CN (negative)",
                "participants": {"AD": 36, "FTD": 23, "CN": 29},
                "inputs_sha256": fingerprint(inputs),
                "outer": "3 repeats x 5 stratified participant folds, seed 9066",
                "inner": "3 stratified participant folds, seed 90660 + outer split",
                "selection": "Maximize inner pooled macro F1; tie-break by dementia recall, then balanced accuracy, then candidate order.",
                "threshold": "Default classifier predict threshold; no outer-test threshold tuning.",
                "features": "Previously computed label-blind participant summaries; ANOVA selection and scaling fit inside training folds.",
                "candidates": configs,
                "interpretation": "Internal nested CV on previously explored participants, not independent external validation."}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    records = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9066)
    for fold, (train, test) in enumerate(outer.split(subjects, y)):
        oof = np.full((len(configs), len(train)), -1, dtype=int)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=90660 + fold)
        for inside, (a, b) in enumerate(inner.split(train, y[train])):
            for j, config in enumerate(configs):
                model = make_model(config)
                matrix = x[config["features"]]
                model.fit(matrix[train[a]], y[train[a]])
                oof[j, b] = model.predict(matrix[train[b]])
        assert np.all(oof >= 0)
        scores = [metrics(y[train], row) for row in oof]
        chosen = min(range(len(configs)), key=lambda j: (-scores[j]["macro_f1"],
                       -scores[j]["dementia_recall"], -scores[j]["balanced_accuracy"], j))
        config = configs[chosen]
        model = make_model(config)
        matrix = x[config["features"]]
        model.fit(matrix[train], y[train])
        predicted = model.predict(matrix[test])
        records.append({"fold": fold, "repeat": fold // 5,
                        "train_subjects": subjects[train].tolist(),
                        "test_subjects": subjects[test].tolist(),
                        "true": y[test].tolist(), "predicted": predicted.tolist(),
                        "chosen_config": config, "chosen_inner_metrics": scores[chosen],
                        "all_inner_metrics": scores, "outer_metrics": metrics(y[test], predicted)})
        print(f"Completed fold {fold + 1}/15: macro F1 {records[-1]['outer_metrics']['macro_f1']:.3f}", flush=True)
    repeats = []
    for repeat in range(3):
        rows = [row for row in records if row["repeat"] == repeat]
        truth = np.concatenate([row["true"] for row in rows])
        predicted = np.concatenate([row["predicted"] for row in rows])
        assert len(truth) == 88 and np.array_equal(np.bincount(truth), [29, 59])
        repeats.append(metrics(truth, predicted))
    fields = ("accuracy", "balanced_accuracy", "macro_f1", "dementia_f1",
              "dementia_recall", "dementia_precision", "cn_specificity")
    summary = {"mean_" + field: float(np.mean([r[field] for r in repeats])) for field in fields}
    summary.update({"sd_" + field: float(np.std([r[field] for r in repeats], ddof=1)) for field in fields})
    summary.update({"repeat_metrics": repeats, "selected_config_counts": dict(Counter(
        json.dumps(row["chosen_config"], sort_keys=True) for row in records)),
        "seconds": time.perf_counter() - start})
    (OUT / "folds.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
