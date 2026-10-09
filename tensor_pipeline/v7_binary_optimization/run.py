"""Exploratory binary optimization with nested participant-level selection.

The 88-person cohort is already researcher-explored. Outer folds are held out
from this script's fits and selection, but results are not external validation.
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
V3 = ROOT / "tensor_pipeline/outputs/v3_waveform_full_cohort"
V5 = ROOT / "tensor_pipeline/outputs/v5_connectivity"
V6 = ROOT / "tensor_pipeline/outputs/v6_binary"
OUT = ROOT / "tensor_pipeline/outputs/v7_binary_optimization"
FEATURE_FILES = [V3 / "index.npz", V3 / "spectral_controls.npy",
                 *(V5 / f"bags_{n}.npz" for n in (1, 4, 8))]


def hashes(paths):
    out = {}
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        out[str(path.relative_to(ROOT))] = h.hexdigest()
    return out


def matrix_log_features(cov):
    """Per-observation SPD matrix log; no cohort reference or fitted state."""
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, 1e-10)
    logs = (vectors * np.log(values)[..., None, :]) @ np.swapaxes(vectors, -1, -2)
    upper = np.triu_indices(cov.shape[-1])
    result = logs[..., upper[0], upper[1]]
    result[..., upper[0] != upper[1]] *= np.sqrt(2)
    return result.reshape(len(cov), -1)


def load():
    idx = np.load(V3 / "index.npz")
    subjects, old_y = idx["subjects"].astype(str), idx["labels"].astype(int)
    assert len(set(subjects)) == 88 and np.array_equal(np.bincount(old_y), [36, 23, 29])
    matrices = {"spectral": np.load(V3 / "spectral_controls.npy")}
    for n in (1, 4, 8):
        d = np.load(V5 / f"bags_{n}.npz")
        assert np.array_equal(d["subject"].astype(str), np.repeat(subjects, n))
        assert np.array_equal(d["label"].astype(int), np.repeat(old_y, n))
        power = d["power"].reshape(88, n, -1)
        conn = d["connectivity"].reshape(88, n, -1)
        if n == 1:
            matrices["power"] = power[:, 0]
            matrices["connectivity"] = conn[:, 0]
            matrices["logcov"] = matrix_log_features(d["covariance"])
        else:
            matrices[f"power_bag{n}"] = np.concatenate([power.mean(axis=1), power.std(axis=1)], axis=1)
            if n == 4:
                matrices["connectivity_bag4"] = np.concatenate([conn.mean(axis=1), conn.std(axis=1)], axis=1)
    matrices["spectral_power"] = np.concatenate([matrices["spectral"], matrices["power"]], axis=1)
    matrices["spectral_connectivity"] = np.concatenate([matrices["spectral"], matrices["connectivity"]], axis=1)
    matrices["spectral_logcov"] = np.concatenate([matrices["spectral"], matrices["logcov"]], axis=1)
    matrices["power_logcov"] = np.concatenate([matrices["power"], matrices["logcov"]], axis=1)
    for name, x in matrices.items():
        assert x.shape[0] == 88 and np.isfinite(x).all(), name
    return subjects, (old_y != 2).astype(int), matrices


def family(rep):
    if rep == "spectral": return "spectral"
    if rep.startswith("power") or rep == "spectral_power": return "power"
    return "graph"


def candidates():
    # All choices fixed before inspecting V7 outcomes.
    specs = [
        ("spectral", (16, 32, 64)),
        ("power", (16, 32, 64)),
        ("connectivity", (32, 64, 128)),
        ("logcov", (32, 64, 128)),
        ("spectral_power", (32, 64, 128)),
        ("spectral_connectivity", (32, 64, 128)),
        ("spectral_logcov", (32, 64, 128)),
        ("power_logcov", (32, 64, 128)),
        ("power_bag4", (32, 64, 128)),
        ("power_bag8", (32, 64, 128)),
        ("connectivity_bag4", (32, 64, 128)),
    ]
    return [dict(features=rep, k=k, model=model, C=c)
            for rep, ks in specs for k in ks
            for model, cs in (("logistic", (0.1, 1.0)), ("rbf_svm", (1.0, 10.0)))
            for c in cs]


def model_for(cfg):
    classifier = (LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=3000)
                  if cfg["model"] == "logistic" else
                  SVC(C=cfg["C"], kernel="rbf", gamma="scale", class_weight="balanced"))
    return Pipeline([("select", SelectKBest(f_classif, k=cfg["k"])),
                     ("scale", StandardScaler()), ("classifier", classifier)])


def metric(y, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
            "dementia_f1": float(f1_score(y, prediction, zero_division=0)),
            "dementia_recall": float(recall_score(y, prediction, zero_division=0)),
            "dementia_precision": float(precision_score(y, prediction, zero_division=0)),
            "cn_specificity": float(tn / (tn + fp)),
            "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}


def rank(j, scores):
    m = scores[j]
    return (-m["macro_f1"], -m["dementia_recall"], -m["balanced_accuracy"], j)


def summaries(records):
    names = ("selected_pipeline", "family_vote")
    fields = ("accuracy", "balanced_accuracy", "macro_f1", "dementia_f1",
              "dementia_recall", "dementia_precision", "cn_specificity")
    result = {}
    for name in names:
        repeats = []
        for repeat in range(3):
            rows = [r for r in records if r["repeat"] == repeat]
            if len(rows) != 5: continue
            y = np.concatenate([r["true"] for r in rows])
            p = np.concatenate([r["predictions"][name] for r in rows])
            assert len(y) == 88 and np.array_equal(np.bincount(y), [29, 59])
            repeats.append(metric(y, p))
        if repeats:
            result[name] = {"repeats": repeats, **{
                "mean_" + f: float(np.mean([r[f] for r in repeats])) for f in fields},
                **{"sd_" + f: float(np.std([r[f] for r in repeats], ddof=1))
                   if len(repeats) > 1 else None for f in fields}}
    return result


def main():
    start = time.perf_counter()
    subjects, y, x = load()
    configs = candidates()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"task": "AD+FTD positive versus CN negative; 88 participants",
                "design": "Exploratory V7 optimization following inspected V6 results.",
                "outer": "Same V6 3x5 stratified participant folds; seed 9066, enabling paired comparison.",
                "inner": "3 stratified participant folds; seed 90660 + outer index.",
                "selection": "Pooled inner macro F1, then dementia recall, then balanced accuracy; default prediction thresholds.",
                "secondary_endpoint": "Fixed three-family hard majority vote: inner-best spectral, power and graph pipelines.",
                "inputs_sha256": hashes([*FEATURE_FILES, Path(__file__)]),
                "features": "Label-blind subject summaries, single-observation SPD matrix log and bag mean/std; ANOVA and scaler fit in each train split.",
                "candidates": configs,
                "limits": "Repeated use of the same 88 subjects is adaptively biased; no external test cohort."}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    records = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9066)
    for fold, (train, test) in enumerate(outer.split(subjects, y)):
        path = OUT / f"fold_{fold:02d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["train_subjects"] == subjects[train].tolist()
            assert row["test_subjects"] == subjects[test].tolist()
            records.append(row)
            continue
        oof = np.full((len(configs), len(train)), -1, dtype=np.int8)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=90660 + fold)
        for a, b in inner.split(train, y[train]):
            for j, cfg in enumerate(configs):
                m = model_for(cfg)
                matrix = x[cfg["features"]]
                m.fit(matrix[train[a]], y[train[a]])
                oof[j, b] = m.predict(matrix[train[b]])
        assert np.all(oof >= 0)
        scores = [metric(y[train], p) for p in oof]
        chosen = {"selected_pipeline": min(range(len(configs)), key=lambda j: rank(j, scores))}
        for group in ("spectral", "power", "graph"):
            eligible = [j for j, cfg in enumerate(configs) if family(cfg["features"]) == group]
            chosen[group] = min(eligible, key=lambda j: rank(j, scores))
        predictions = {}
        fitted = {}
        for name, j in chosen.items():
            if j not in fitted:
                cfg = configs[j]
                m = model_for(cfg)
                matrix = x[cfg["features"]]
                m.fit(matrix[train], y[train])
                fitted[j] = m.predict(matrix[test]).astype(int).tolist()
            predictions[name] = fitted[j]
        vote = np.asarray([predictions[g] for g in ("spectral", "power", "graph")]).sum(axis=0)
        predictions["family_vote"] = (vote >= 2).astype(int).tolist()
        row = {"fold": fold, "repeat": fold // 5,
               "train_subjects": subjects[train].tolist(), "test_subjects": subjects[test].tolist(),
               "true": y[test].tolist(), "predictions": predictions,
               "selected_configs": {name: configs[j] for name, j in chosen.items()},
               "selected_inner_metrics": {name: scores[j] for name, j in chosen.items()},
               "outer_metrics": {name: metric(y[test], predictions[name])
                                 for name in ("selected_pipeline", "family_vote")}}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        records.append(row)
        (OUT / "summary.json").write_text(json.dumps(summaries(records), indent=2), encoding="utf-8")
        print(f"Outer {fold + 1}/15 saved; selected F1 {row['outer_metrics']['selected_pipeline']['macro_f1']:.3f}; vote F1 {row['outer_metrics']['family_vote']['macro_f1']:.3f}", flush=True)
    result = summaries(records)
    baseline = json.loads((V6 / "summary.json").read_text(encoding="utf-8"))
    for endpoint in result.values():
        endpoint["difference_vs_v6_mean"] = {
            k: endpoint["mean_" + k] - baseline["mean_" + k]
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "dementia_f1",
                      "dementia_recall", "dementia_precision", "cn_specificity")}
    result["completed_folds"] = len(records)
    result["seconds_this_run"] = time.perf_counter() - start
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in endpoint.items() if k != "repeats"}
                      for name, endpoint in result.items() if isinstance(endpoint, dict)}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
