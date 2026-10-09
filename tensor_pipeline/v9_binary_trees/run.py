"""Bounded tree-model comparison for the existing binary EEG cohort."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "tensor_pipeline/outputs/v3_waveform_full_cohort"
V5 = ROOT / "tensor_pipeline/outputs/v5_connectivity"
V6 = ROOT / "tensor_pipeline/outputs/v6_binary"
V8_SCRIPT = ROOT / "tensor_pipeline/v8_binary_compact/run.py"
OUT = ROOT / "tensor_pipeline/outputs/v9_binary_trees"
spec = importlib.util.spec_from_file_location("v8_binary_features", V8_SCRIPT)
v8 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v8)


def load():
    subjects, y, features = v8.load()
    features["power"] = np.load(V5 / "bags_1.npz")["power"]
    features["connectivity"] = np.load(V5 / "bags_1.npz")["connectivity"]
    features["spectral_power"] = np.concatenate([features["raw_spectral"], features["power"]], axis=1)
    return subjects, y, features


def configs():
    reps = (("raw_spectral", (32, 64)), ("power", (32, 64)),
            ("connectivity", (64, 128)), ("spectral_power", (32, 64)),
            ("graph_summary", (32, 64)))
    models = (("extra_trees", 2), ("extra_trees", 4),
              ("random_forest", 3), ("xgboost", 1), ("xgboost", 2))
    return [dict(features=rep, k=k, model=model, complexity=c)
            for rep, ks in reps for k in ks for model, c in models]


def make_model(cfg):
    kind, c = cfg["model"], cfg["complexity"]
    if kind == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=c,
                                   max_features="sqrt", n_jobs=1, random_state=9090)
    elif kind == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=c,
                                     max_features="sqrt", n_jobs=1, random_state=9090)
    else:
        clf = XGBClassifier(n_estimators=150, max_depth=c, learning_rate=.05,
                            min_child_weight=3, subsample=.8, colsample_bytree=.8,
                            reg_lambda=10, reg_alpha=.1, tree_method="hist", device="cpu",
                            objective="binary:logistic", eval_metric="logloss",
                            n_jobs=1, random_state=9090, verbosity=0)
    return Pipeline([("select", SelectKBest(f_classif, k=cfg["k"])), ("clf", clf)])


def balanced_weights(y):
    count = np.bincount(y, minlength=2)
    return len(y) / (2 * count[y])


def main():
    started = time.perf_counter()
    subjects, y, features = load()
    choices = configs()
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [V3 / "index.npz", V3 / "spectral_controls.npy",
                    V5 / "bags_1.npz", V8_SCRIPT, Path(__file__)]
    protocol = {"task": "AD+FTD positive versus CN negative; 88 people",
                "design": "Exploratory tree comparison after V6-V8 outcomes were inspected.",
                "outer": "Same V6 participant folds, 3x5 stratified, seed 9066.",
                "inner": "3 stratified participant folds, seed 90900 + outer index.",
                "selection": "Inner pooled macro F1, dementia recall, balanced accuracy; default threshold.",
                "weights": "Current training fold class-balanced observation weights.",
                "inputs_sha256": {str(p.relative_to(ROOT)): v8.sha(p) for p in source_paths},
                "candidates": choices,
                "limits": "Same adaptively explored 88 subjects; no external confirmation."}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9066)
    for fold, (train, test) in enumerate(outer.split(subjects, y)):
        path = OUT / f"fold_{fold:02d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["train_subjects"] == subjects[train].tolist()
            assert row["test_subjects"] == subjects[test].tolist()
            rows.append(row)
            continue
        oof = np.full((len(choices), len(train)), -1, dtype=np.int8)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=90900 + fold)
        for a, b in inner.split(train, y[train]):
            inner_train, inner_test = train[a], train[b]
            for j, cfg in enumerate(choices):
                matrix = features[cfg["features"]]
                model = make_model(cfg)
                model.fit(matrix[inner_train], y[inner_train],
                          clf__sample_weight=balanced_weights(y[inner_train]))
                oof[j, b] = model.predict(matrix[inner_test])
        assert np.all(oof >= 0)
        scores = [v8.metric(y[train], p) for p in oof]
        chosen = min(range(len(choices)), key=lambda j:
                     (-scores[j]["macro_f1"], -scores[j]["dementia_recall"],
                      -scores[j]["balanced_accuracy"], j))
        cfg = choices[chosen]
        matrix = features[cfg["features"]]
        model = make_model(cfg)
        model.fit(matrix[train], y[train], clf__sample_weight=balanced_weights(y[train]))
        predicted = model.predict(matrix[test]).astype(int)
        row = {"fold": fold, "repeat": fold // 5,
               "train_subjects": subjects[train].tolist(), "test_subjects": subjects[test].tolist(),
               "true": y[test].tolist(), "predicted": predicted.tolist(),
               "selected_config": cfg, "selected_inner_metrics": scores[chosen],
               "outer_metrics": v8.metric(y[test], predicted)}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        (OUT / "summary.json").write_text(json.dumps(v8.report(rows), indent=2), encoding="utf-8")
        print(f"Outer {fold + 1}/15 saved; macro F1 {row['outer_metrics']['macro_f1']:.3f}", flush=True)
    result = v8.report(rows)
    baseline = json.loads((V6 / "summary.json").read_text(encoding="utf-8"))
    result["difference_vs_v6_mean"] = {f: result["mean_" + f] - baseline["mean_" + f]
                                        for f in ("accuracy", "balanced_accuracy", "macro_f1",
                                                  "dementia_f1", "dementia_recall", "dementia_precision",
                                                  "cn_specificity")}
    result["seconds_this_run"] = time.perf_counter() - started
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "repeat_metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
