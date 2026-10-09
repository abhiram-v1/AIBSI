"""Nested binary classification using conventional EEG features only."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conventional_ml/outputs/binary_eeg"
FEATURES = OUT / "subject_features.npz"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    d = np.load(FEATURES)
    subjects = d["subjects"].astype(str)
    groups = d["groups"].astype(str)
    assert len(set(subjects)) == 88 and np.array_equal(
        [np.sum(groups == g) for g in ("A", "F", "C")], [36, 23, 29])
    matrices = {"global": d["global_features"],
                "regional": d["regional_features"],
                "channel": d["channel_features"],
                "global_regional": np.concatenate([d["global_features"],
                                                   d["regional_features"]], axis=1)}
    assert all(x.shape[0] == 88 and np.isfinite(x).all() for x in matrices.values())
    return subjects, (groups != "C").astype(int), matrices


def candidates():
    representations = (("global", (None, 16, 32)),
                       ("regional", (16, 32, 64)),
                       ("channel", (16, 32, 64)),
                       ("global_regional", (16, 32, 64)))
    classifiers = (("logistic", 0.1), ("logistic", 1.0),
                   ("rbf_svm", 0.1), ("rbf_svm", 1.0), ("rbf_svm", 10.0),
                   ("shrinkage_lda", None), ("random_forest", 3))
    return [dict(features=rep, k=k, model=model, C=c)
            for rep, ks in representations for k in ks for model, c in classifiers]


def make_model(cfg):
    kind = cfg["model"]
    if kind == "logistic":
        clf = LogisticRegression(C=cfg["C"], max_iter=3000, class_weight="balanced")
    elif kind == "rbf_svm":
        clf = SVC(C=cfg["C"], kernel="rbf", gamma="scale", class_weight="balanced")
    elif kind == "shrinkage_lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5])
    else:
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=int(cfg["C"]),
                                     max_features="sqrt", class_weight="balanced",
                                     random_state=9100, n_jobs=1)
    steps = []
    if cfg["k"] is not None: steps.append(("select", SelectKBest(f_classif, k=cfg["k"])))
    if kind != "random_forest": steps.append(("scale", StandardScaler()))
    steps.append(("classifier", clf))
    return Pipeline(steps)


def metric(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "dementia_f1": float(f1_score(y, p, zero_division=0)),
            "dementia_recall": float(recall_score(y, p, zero_division=0)),
            "dementia_precision": float(precision_score(y, p, zero_division=0)),
            "cn_specificity": float(tn / (tn + fp)),
            "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}


def summary(rows):
    fields = ("accuracy", "balanced_accuracy", "macro_f1", "dementia_f1",
              "dementia_recall", "dementia_precision", "cn_specificity")
    result = {"completed_folds": len(rows), "endpoints": {}}
    for name in ("inner_selected", "fixed_global_logistic"):
        repeats = []
        for repeat in range(3):
            fold_rows = [r for r in rows if r["repeat"] == repeat]
            if len(fold_rows) != 5: continue
            y = np.concatenate([r["true"] for r in fold_rows])
            p = np.concatenate([r["predictions"][name] for r in fold_rows])
            assert len(y) == 88 and np.array_equal(np.bincount(y), [29, 59])
            repeats.append(metric(y, p))
        if repeats:
            result["endpoints"][name] = {
                "repeat_metrics": repeats,
                **{"mean_" + f: float(np.mean([r[f] for r in repeats])) for f in fields},
                **{"sd_" + f: float(np.std([r[f] for r in repeats], ddof=1))
                   if len(repeats) > 1 else None for f in fields}}
    return result


def main():
    began = time.perf_counter()
    subjects, y, matrices = load()
    configs = candidates()
    protocol = {"task": "AD+FTD (positive) versus CN (negative), 88 people",
                "source": "Conventional time and spectral features extracted directly from derivative .set EEG; no tensor-pipeline input.",
                "outer": "3 repetitions x 5 participant folds, seed 9066",
                "inner": "5 participant folds, seed 91000 + outer index",
                "selection": "Pooled inner macro F1, then dementia recall, balanced accuracy, candidate order",
                "fixed_control": "Global features with class-balanced logistic regression C=1, no feature selection",
                "learned_stages": "ANOVA feature selection and scaling fit inside each training fold",
                "threshold": "Default classifier prediction; no test-label threshold tuning",
                "feature_sha256": sha(FEATURES), "code_sha256": sha(Path(__file__)),
                "candidates": configs,
                "limits": "Same previously explored 88-person cohort; this is internal validation, not independent confirmation."}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9066)
    fixed = dict(features="global", k=None, model="logistic", C=1.0)
    for fold, (train, test) in enumerate(outer.split(subjects, y)):
        path = OUT / f"fold_{fold:02d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["train_subjects"] == subjects[train].tolist()
            assert row["test_subjects"] == subjects[test].tolist()
            rows.append(row)
            continue
        oof = np.full((len(configs), len(train)), -1, dtype=np.int8)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=91000 + fold)
        for a, b in inner.split(train, y[train]):
            for j, cfg in enumerate(configs):
                matrix = matrices[cfg["features"]]
                fitted = make_model(cfg).fit(matrix[train[a]], y[train[a]])
                oof[j, b] = fitted.predict(matrix[train[b]])
        assert np.all(oof >= 0)
        scored = [metric(y[train], p) for p in oof]
        chosen = min(range(len(configs)), key=lambda j:
                     (-scored[j]["macro_f1"], -scored[j]["dementia_recall"],
                      -scored[j]["balanced_accuracy"], j))
        cfg = configs[chosen]
        selected_model = make_model(cfg).fit(matrices[cfg["features"]][train], y[train])
        fixed_model = make_model(fixed).fit(matrices["global"][train], y[train])
        predictions = {"inner_selected": selected_model.predict(matrices[cfg["features"]][test]).astype(int).tolist(),
                       "fixed_global_logistic": fixed_model.predict(matrices["global"][test]).astype(int).tolist()}
        row = {"fold": fold, "repeat": fold // 5,
               "train_subjects": subjects[train].tolist(), "test_subjects": subjects[test].tolist(),
               "true": y[test].tolist(), "predictions": predictions,
               "selected_config": cfg, "selected_inner_metrics": scored[chosen],
               "outer_metrics": {name: metric(y[test], p) for name, p in predictions.items()}}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        (OUT / "summary.json").write_text(json.dumps(summary(rows), indent=2), encoding="utf-8")
        print(f"Outer {fold + 1}/15 saved: selected macro F1 {row['outer_metrics']['inner_selected']['macro_f1']:.3f}", flush=True)
    result = summary(rows)
    result["seconds_this_run"] = time.perf_counter() - began
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in endpoint.items() if k != "repeat_metrics"}
                      for name, endpoint in result["endpoints"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
