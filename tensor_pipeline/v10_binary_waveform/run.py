"""Binary TT/Tucker EEG benchmark with nested subject-level CV."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "tensor_pipeline/outputs/v3_waveform_full_cohort"
OUT = ROOT / "tensor_pipeline/outputs/v10_binary_waveform"
sys.path.insert(0, str(ROOT / "tensor_pipeline/v3_waveform"))
from wave_models import energies, top  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    idx = np.load(V3 / "index.npz")
    subjects = idx["subjects"].astype(str)
    original = idx["labels"].astype(int)
    moments = np.load(V3 / "delay_moments.npy", mmap_mode="r")
    assert moments.shape == (88, 304, 304)
    assert np.array_equal(np.bincount(original), [36, 23, 29])
    return subjects, (original != 2).astype(int), moments


def configs():
    reps = [("tt", 8), ("tt", 16), ("tucker", 4), ("tucker", 6)]
    classifiers = [("logistic", .1), ("logistic", 1.0),
                   ("rbf_svm", 1.0), ("rbf_svm", 10.0)]
    return [dict(decomposition=rep, rank=rank, classifier=clf, C=c)
            for rep, rank in reps for clf, c in classifiers]


def basis_for(moments, y, cfg):
    # Each class contributes half the training moment. No test participant
    # contributes to the TT/Tucker basis.
    a = .5 * np.mean(moments[y == 0], axis=0) + .5 * np.mean(moments[y == 1], axis=0)
    tensor = a.reshape(19, 16, 19, 16)
    if cfg["decomposition"] == "tt":
        channel = top(np.einsum("cldl->cd", tensor), 6)
        left = np.kron(channel, np.eye(16))
        temporal = top(left.T @ a @ left, cfg["rank"])
        basis = left @ temporal
    else:
        channel = top(np.einsum("cldl->cd", tensor), cfg["rank"])
        lag = top(np.einsum("clcm->lm", tensor), cfg["rank"])
        basis = np.kron(channel, lag)
    return basis


def transform(moments, basis):
    return np.log(np.maximum(energies(np.asarray(moments), basis), 1e-8))


def model_for(cfg):
    clf = (LogisticRegression(C=cfg["C"], max_iter=3000, class_weight="balanced")
           if cfg["classifier"] == "logistic" else
           SVC(C=cfg["C"], kernel="rbf", gamma="scale", class_weight="balanced"))
    return Pipeline([("scale", StandardScaler()), ("classifier", clf)])


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


def summarize(rows):
    fields = ("accuracy", "balanced_accuracy", "macro_f1", "dementia_f1",
              "dementia_recall", "dementia_precision", "cn_specificity")
    repeats = []
    for repeat in range(3):
        group = [row for row in rows if row["repeat"] == repeat]
        if len(group) != 5: continue
        truth = np.concatenate([row["true"] for row in group])
        pred = np.concatenate([row["predicted"] for row in group])
        assert len(truth) == 88 and np.array_equal(np.bincount(truth), [29, 59])
        repeats.append(metric(truth, pred))
    result = {"completed_folds": len(rows), "repeat_metrics": repeats}
    if repeats:
        result.update({"mean_" + key: float(np.mean([r[key] for r in repeats])) for key in fields})
        result.update({"sd_" + key: float(np.std([r[key] for r in repeats], ddof=1))
                       if len(repeats) > 1 else None for key in fields})
    return result


def main():
    began = time.perf_counter()
    subjects, y, moments = load()
    choices = configs()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [V3 / "index.npz", V3 / "delay_moments.npy", Path(__file__),
              ROOT / "tensor_pipeline/v3_waveform/wave_models.py"]
    protocol = {"task": "AD+FTD positive versus CN negative; 88 people",
                "input": "V3 cleaned waveform delay moments; no spectral or connectivity control features",
                "decomposition": "Training-only class-balanced waveform TT/Tucker basis",
                "outer": "3x5 stratified participant CV, seed 9066",
                "inner": "3 stratified participant folds, seed 91100 + outer index",
                "selection": "Pooled inner macro F1, then dementia recall, balanced accuracy",
                "classifiers": "Class-balanced logistic regression or RBF SVM; scaler fitted in training fold",
                "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                "candidates": choices,
                "limits": "Existing previously explored 88-person cohort; no external validation."}
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
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=91100 + fold)
        for a, b in inner.split(train, y[train]):
            train_inner, valid_inner = train[a], train[b]
            for j, cfg in enumerate(choices):
                basis = basis_for(moments[train_inner], y[train_inner], cfg)
                z_train = transform(moments[train_inner], basis)
                z_valid = transform(moments[valid_inner], basis)
                model = model_for(cfg).fit(z_train, y[train_inner])
                oof[j, b] = model.predict(z_valid)
        assert np.all(oof >= 0)
        scores = [metric(y[train], p) for p in oof]
        chosen = min(range(len(choices)), key=lambda j:
                     (-scores[j]["macro_f1"], -scores[j]["dementia_recall"],
                      -scores[j]["balanced_accuracy"], j))
        cfg = choices[chosen]
        basis = basis_for(moments[train], y[train], cfg)
        model = model_for(cfg).fit(transform(moments[train], basis), y[train])
        p = model.predict(transform(moments[test], basis)).astype(int)
        row = {"fold": fold, "repeat": fold // 5,
               "train_subjects": subjects[train].tolist(), "test_subjects": subjects[test].tolist(),
               "true": y[test].tolist(), "predicted": p.tolist(),
               "selected_config": cfg, "selected_inner_metrics": scores[chosen],
               "outer_metrics": metric(y[test], p)}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        (OUT / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
        print(f"Outer {fold + 1}/15 saved: macro F1 {row['outer_metrics']['macro_f1']:.3f}", flush=True)
    result = summarize(rows)
    result["seconds_this_run"] = time.perf_counter() - began
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "repeat_metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
