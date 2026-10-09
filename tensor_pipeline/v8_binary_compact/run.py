"""Compact physiological EEG features for exploratory binary optimization."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "tensor_pipeline/outputs/v3_waveform_full_cohort"
V5 = ROOT / "tensor_pipeline/outputs/v5_connectivity"
V6 = ROOT / "tensor_pipeline/outputs/v6_binary"
OUT = ROOT / "tensor_pipeline/outputs/v8_binary_compact"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spectral_regions(controls, channels):
    assert controls.shape == (88, 418)
    # V3 order: log band power, relative band power, entropy, peak, ratio;
    # each has a channel-wise mean followed by within-window SD.
    mean_maps = np.concatenate([
        controls[:, 0:76].reshape(88, 19, 4),
        controls[:, 152:228].reshape(88, 19, 4),
        controls[:, 304:323, None], controls[:, 342:361, None],
        controls[:, 380:399, None]], axis=2)
    assert mean_maps.shape == (88, 19, 11)
    regions = [
        ("frontal", ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz")),
        ("temporal", ("T3", "T4", "T5", "T6")),
        ("central", ("C3", "C4", "Cz")),
        ("parietal", ("P3", "P4", "Pz")),
        ("occipital", ("O1", "O2")),
    ]
    names = list(channels)
    regional = np.stack([mean_maps[:, [names.index(c) for c in group]].mean(axis=1)
                         for _, group in regions], axis=1)
    global_features = np.concatenate([mean_maps.mean(axis=1), mean_maps.std(axis=1)], axis=1)
    return regional.reshape(88, -1), global_features


def graph_features(connectivity, channels):
    n = len(channels)
    iu = np.triu_indices(n, 1)
    edges = connectivity.reshape(88, 4, 4, len(iu[0]))
    regions = [("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"),
               ("T3", "T4", "T5", "T6"), ("C3", "C4", "Cz"),
               ("P3", "P4", "Pz"), ("O1", "O2")]
    names = list(channels)
    result = []
    for band in range(4):
        for kind in range(4):
            e = edges[:, band, kind]
            abs_e = np.abs(e)
            # Whole-network edge distribution.
            summary = np.column_stack([e.mean(axis=1), abs_e.mean(axis=1),
                                       e.std(axis=1), np.quantile(abs_e, .75, axis=1)])
            adjacency = np.zeros((88, n, n), dtype=float)
            adjacency[:, iu[0], iu[1]] = abs_e
            adjacency[:, iu[1], iu[0]] = abs_e
            strength = adjacency.mean(axis=2)
            regional = np.stack([strength[:, [names.index(c) for c in group]].mean(axis=1)
                                 for group in regions], axis=1)
            eigen = np.linalg.eigvalsh(adjacency)[:, -3:]
            result.append(np.concatenate([summary, regional, eigen], axis=1))
    return np.concatenate(result, axis=1)


def covariance_spectrum(covariance):
    eig = np.maximum(np.linalg.eigvalsh(covariance), 1e-10)
    log_eig = np.log(eig)
    probabilities = eig / eig.sum(axis=-1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities)).sum(axis=-1)
    anisotropy = log_eig[..., -1] - log_eig[..., 0]
    return np.concatenate([log_eig.reshape(88, -1), entropy, anisotropy], axis=1)


def load():
    idx = np.load(V3 / "index.npz")
    subjects = idx["subjects"].astype(str)
    original = idx["labels"].astype(int)
    channels = idx["channels"].astype(str)
    d = np.load(V5 / "bags_1.npz")
    assert np.array_equal(d["subject"].astype(str), subjects)
    assert np.array_equal(d["label"], original)
    assert np.array_equal(np.bincount(original), [36, 23, 29])
    spectral = np.load(V3 / "spectral_controls.npy")
    regional, global_features = spectral_regions(spectral, channels)
    graph = graph_features(d["connectivity"], channels)
    cov = covariance_spectrum(d["covariance"])
    x = {"global_spectral": global_features, "regional_spectral": regional,
         "graph_summary": graph, "cov_spectrum": cov, "raw_spectral": spectral,
         "regional_graph": np.concatenate([regional, graph], axis=1),
         "regional_cov": np.concatenate([regional, cov], axis=1)}
    assert all(a.shape[0] == 88 and np.isfinite(a).all() for a in x.values())
    return subjects, (original != 2).astype(int), x


def candidates():
    specs = [("global_spectral", (None,)),
             ("regional_spectral", (None, 16, 32)),
             ("graph_summary", (16, 32, 64)),
             ("cov_spectrum", (16, 32, None)),
             ("regional_graph", (16, 32, 64)),
             ("regional_cov", (16, 32, 64)),
             ("raw_spectral", (16, 32, 64))]
    classifiers = (("logistic", .1), ("logistic", 1.0),
                   ("rbf_svm", 1.0), ("rbf_svm", 10.0), ("shrinkage_lda", None))
    return [dict(features=rep, k=k, model=model, C=c)
            for rep, ks in specs for k in ks for model, c in classifiers]


def make_model(cfg):
    if cfg["model"] == "logistic":
        classifier = LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=3000)
    elif cfg["model"] == "rbf_svm":
        classifier = SVC(C=cfg["C"], gamma="scale", kernel="rbf", class_weight="balanced")
    else:
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                                 priors=[.5, .5])
    steps = []
    if cfg["k"] is not None: steps.append(("select", SelectKBest(f_classif, k=cfg["k"])))
    steps.extend([("scale", StandardScaler()), ("classifier", classifier)])
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


def report(rows):
    fields = ("accuracy", "balanced_accuracy", "macro_f1", "dementia_f1",
              "dementia_recall", "dementia_precision", "cn_specificity")
    repeats = []
    for repeat in range(3):
        group = [r for r in rows if r["repeat"] == repeat]
        if len(group) != 5: continue
        y = np.concatenate([r["true"] for r in group])
        p = np.concatenate([r["predicted"] for r in group])
        assert len(y) == 88 and np.array_equal(np.bincount(y), [29, 59])
        repeats.append(metric(y, p))
    result = {"completed_folds": len(rows), "repeat_metrics": repeats}
    if repeats:
        result.update({"mean_" + f: float(np.mean([r[f] for r in repeats])) for f in fields})
        result.update({"sd_" + f: float(np.std([r[f] for r in repeats], ddof=1))
                       if len(repeats) > 1 else None for f in fields})
    return result


def main():
    began = time.perf_counter()
    subjects, y, matrices = load()
    configs = candidates()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [V3 / "index.npz", V3 / "spectral_controls.npy", V5 / "bags_1.npz", Path(__file__)]
    protocol = {"task": "AD+FTD positive versus CN negative; 88 participants",
                "design": "Exploratory compact feature experiment after inspecting V6 and V7.",
                "outer": "Same V6/V7 participant folds: 3x5 stratified, seed 9066.",
                "inner": "5 stratified participant folds, seed 90800 + outer index.",
                "selection": "Pooled inner macro F1, then dementia recall, then balanced accuracy; default thresholds.",
                "features": "Fixed regional spectral averages, network summaries, and covariance spectra; no labels used in feature construction.",
                "feature_selection_scaling": "Fit within each training partition only.",
                "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                "candidates": configs,
                "limits": "Same adaptively explored 88 people; no independent confirmation."}
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
        oof = np.full((len(configs), len(train)), -1, dtype=np.int8)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=90800 + fold)
        for a, b in inner.split(train, y[train]):
            for j, cfg in enumerate(configs):
                matrix = matrices[cfg["features"]]
                m = make_model(cfg).fit(matrix[train[a]], y[train[a]])
                oof[j, b] = m.predict(matrix[train[b]])
        assert np.all(oof >= 0)
        scores = [metric(y[train], p) for p in oof]
        chosen = min(range(len(configs)), key=lambda j:
                     (-scores[j]["macro_f1"], -scores[j]["dementia_recall"],
                      -scores[j]["balanced_accuracy"], j))
        cfg = configs[chosen]
        matrix = matrices[cfg["features"]]
        model = make_model(cfg).fit(matrix[train], y[train])
        p = model.predict(matrix[test]).astype(int)
        row = {"fold": fold, "repeat": fold // 5,
               "train_subjects": subjects[train].tolist(), "test_subjects": subjects[test].tolist(),
               "true": y[test].tolist(), "predicted": p.tolist(), "selected_config": cfg,
               "selected_inner_metrics": scores[chosen], "outer_metrics": metric(y[test], p)}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        (OUT / "summary.json").write_text(json.dumps(report(rows), indent=2), encoding="utf-8")
        print(f"Outer {fold + 1}/15 saved; macro F1 {row['outer_metrics']['macro_f1']:.3f}", flush=True)
    result = report(rows)
    old = json.loads((V6 / "summary.json").read_text(encoding="utf-8"))
    result["difference_vs_v6_mean"] = {f: result["mean_" + f] - old["mean_" + f]
                                        for f in ("accuracy", "balanced_accuracy", "macro_f1",
                                                  "dementia_f1", "dementia_recall", "dementia_precision",
                                                  "cn_specificity")}
    result["seconds_this_run"] = time.perf_counter() - began
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "repeat_metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
