"""Participant-level binary RBF SVM runs for the presentation comparison.

Usage: python run_binary_svm.py tensor|direct
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "tensor_pipeline/outputs/v3_waveform_full_cohort"
DIRECT = ROOT / "conventional_ml/outputs/binary_eeg/subject_features.npz"
BASE = ROOT / "presentation_progress/svm_runs"
CS = (0.1, 1.0, 10.0, 100.0)
REPS = (("tt", 8), ("tt", 16), ("tucker", 4), ("tucker", 6))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(mode):
    if mode == "tensor":
        d = np.load(V3 / "index.npz")
        subjects = d["subjects"].astype(str)
        groups = d["labels"].astype(int)
        x = np.load(V3 / "delay_moments.npy", mmap_mode="r")
        assert x.shape == (88, 304, 304)
        y = (groups != 2).astype(int)
    else:
        d = np.load(DIRECT)
        subjects = d["subjects"].astype(str)
        groups = d["groups"].astype(str)
        x = {"global": d["global_features"],
             "regional": d["regional_features"],
             "global_regional": np.concatenate(
                 [d["global_features"], d["regional_features"]], axis=1)}
        assert all(np.isfinite(a).all() for a in x.values())
        y = (groups != "C").astype(int)
    assert len(set(subjects)) == 88
    assert np.array_equal(np.bincount(y), [29, 59])
    return subjects, y, x


def top(a, rank):
    _, u = eigh(a, subset_by_index=(len(a) - rank, len(a) - 1),
                check_finite=False)
    u = u[:, ::-1]
    sign = np.sign(u[np.abs(u).argmax(0), np.arange(rank)])
    return u * np.where(sign == 0, 1, sign)


def tensor_basis(x, y, rep):
    # Equal weight per diagnosis; only subjects in this training fold enter.
    a = .5 * np.mean(x[y == 0], axis=0) + .5 * np.mean(x[y == 1], axis=0)
    tensor = a.reshape(19, 16, 19, 16)
    kind, rank = rep
    if kind == "tt":
        channel = top(np.einsum("cldl->cd", tensor), 6)
        left = np.kron(channel, np.eye(16))
        return left @ top(left.T @ a @ left, rank)
    channel = top(np.einsum("cldl->cd", tensor), rank)
    lag = top(np.einsum("clcm->lm", tensor), rank)
    return np.kron(channel, lag)


def tensor_features(x, basis):
    a = np.asarray(x)
    projected = (a.reshape(-1, a.shape[-1]) @ basis).reshape(len(a),
                                                               a.shape[1], -1)
    energy = np.einsum("ik,nik->nk", basis, projected, optimize=True)
    energy /= np.sum(basis * basis, axis=0)
    return np.log(np.maximum(energy, 1e-8))


def configs(mode):
    if mode == "tensor":
        return [dict(decomposition=kind, rank=rank, C=c)
                for kind, rank in REPS for c in CS]
    dimensions = (("global", (16, 32, None)),
                  ("regional", (16, 32, 64)),
                  ("global_regional", (16, 32, 64)))
    return [dict(features=name, k=k, C=c)
            for name, ks in dimensions for k in ks for c in CS]


def model(cfg, mode):
    steps = []
    if mode == "direct" and cfg["k"] is not None:
        steps.append(("select", SelectKBest(f_classif, k=cfg["k"])))
    steps.extend([("scale", StandardScaler()),
                  ("svm", SVC(kernel="rbf", gamma="scale", C=cfg["C"],
                              class_weight="balanced"))])
    return Pipeline(steps)


def metric(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "dementia_precision": float(precision_score(y, p, zero_division=0)),
            "dementia_recall": float(recall_score(y, p, zero_division=0)),
            "dementia_f1": float(f1_score(y, p, zero_division=0)),
            "cn_recall": float(tn / (tn + fp)),
            "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}


def summary(rows):
    fields = ("accuracy", "balanced_accuracy", "macro_f1",
              "dementia_precision", "dementia_recall", "dementia_f1",
              "cn_recall")
    repeats = []
    for repeat in range(3):
        group = [r for r in rows if r["repeat"] == repeat]
        if len(group) != 5:
            continue
        truth = np.concatenate([r["true"] for r in group])
        pred = np.concatenate([r["predicted"] for r in group])
        ids = [subject for r in group for subject in r["test_subjects"]]
        assert len(truth) == 88 and len(set(ids)) == 88
        repeats.append(metric(truth, pred))
    result = {"completed_folds": len(rows), "repeat_metrics": repeats}
    if repeats:
        result.update({"mean_" + key: float(np.mean([r[key] for r in repeats]))
                       for key in fields})
        result.update({"sd_" + key: float(np.std([r[key] for r in repeats], ddof=1))
                       for key in fields})
    return result


def main(mode):
    started = time.perf_counter()
    subjects, y, x = load(mode)
    choices = configs(mode)
    out = BASE / ("tensor_binary" if mode == "tensor" else "direct_binary")
    out.mkdir(parents=True, exist_ok=True)
    input_paths = ([V3 / "index.npz", V3 / "delay_moments.npy"]
                   if mode == "tensor" else [DIRECT])
    protocol = {"mode": mode, "classifier": "RBF SVM only, class balanced",
                "task": "AD+FTD versus CN on 88 participants",
                "outer": "3 repetitions of 5 stratified subject folds; seed 9066",
                "inner": "3 stratified subject folds; seed 91100 + outer index",
                "selection": "Pooled inner macro F1, then dementia recall, balanced accuracy",
                "basis": "TT/Tucker fitted within training fold" if mode == "tensor"
                         else "Direct EEG features; selection and scaling within training fold",
                "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in input_paths},
                "code_sha256": sha(Path(__file__)), "candidates": choices,
                "limit": "Previously explored cohort; no independent external validation"}
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                    random_state=9066)
    for fold, (train, test) in enumerate(outer.split(subjects, y)):
        path = out / f"fold_{fold:02d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["train_subjects"] == subjects[train].tolist()
            assert row["test_subjects"] == subjects[test].tolist()
            assert set(row["train_subjects"]).isdisjoint(row["test_subjects"])
            rows.append(row)
            continue
        oof = np.full((len(choices), len(train)), -1, dtype=np.int8)
        inner = StratifiedKFold(n_splits=3, shuffle=True,
                                random_state=91100 + fold)
        for a, b in inner.split(train, y[train]):
            fit, valid = train[a], train[b]
            if mode == "tensor":
                for i, rep in enumerate(REPS):
                    basis = tensor_basis(x[fit], y[fit], rep)
                    z_fit = tensor_features(x[fit], basis)
                    z_valid = tensor_features(x[valid], basis)
                    for j, c in enumerate(CS):
                        cfg = choices[i * len(CS) + j]
                        oof[i * len(CS) + j, b] = model(cfg, mode).fit(
                            z_fit, y[fit]).predict(z_valid)
            else:
                for j, cfg in enumerate(choices):
                    data = x[cfg["features"]]
                    oof[j, b] = model(cfg, mode).fit(
                        data[fit], y[fit]).predict(data[valid])
        assert np.all(oof >= 0)
        scores = [metric(y[train], p) for p in oof]
        selected = min(range(len(choices)), key=lambda j: (
            -scores[j]["macro_f1"], -scores[j]["dementia_recall"],
            -scores[j]["balanced_accuracy"], j))
        cfg = choices[selected]
        if mode == "tensor":
            basis = tensor_basis(x[train], y[train],
                                 (cfg["decomposition"], cfg["rank"]))
            z_train, z_test = (tensor_features(x[train], basis),
                               tensor_features(x[test], basis))
        else:
            data = x[cfg["features"]]
            z_train, z_test = data[train], data[test]
        pred = model(cfg, mode).fit(z_train, y[train]).predict(z_test)
        row = {"fold": fold, "repeat": fold // 5,
               "train_subjects": subjects[train].tolist(),
               "test_subjects": subjects[test].tolist(),
               "true": y[test].tolist(), "predicted": pred.astype(int).tolist(),
               "selected_config": cfg, "inner_metrics": scores[selected],
               "outer_metrics": metric(y[test], pred)}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        (out / "summary.json").write_text(json.dumps(summary(rows), indent=2),
                                           encoding="utf-8")
        print(f"{mode} outer {fold + 1}/15 saved", flush=True)
    result = summary(rows)
    result["seconds_this_run"] = time.perf_counter() - started
    (out / "summary.json").write_text(json.dumps(result, indent=2),
                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "repeat_metrics"},
                     indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tensor", "direct"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.mode)
