"""Recompute headline metrics from saved per-participant predictions.

This script does not retrain anything. It reads prediction files written by the
original experiment runs, recomputes accuracy / balanced accuracy / macro F1 with
scikit-learn, and compares them with the values printed in the research report.

The prediction files live under git-ignored output folders (they contain
participant-level records derived from OpenNeuro and BrainLat), so this script only
works on a machine where the experiments were run. Missing inputs are reported as
"missing", never as passed.

Usage (from the repository root):
    py -3.10 verification/recompute_metrics.py
Writes verification/recomputed_metrics.json and prints a table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "neural_comparison"
CLASSES = ["AD", "FTD", "HC"]
TOL = 0.0015  # report values are rounded to 0.1 percentage point / 3 decimals


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def as_index(values):
    out = []
    for v in values:
        out.append(CLASSES.index(v) if isinstance(v, str) else int(v))
    return np.asarray(out)


checks: list[dict] = []


def check(name, report_section, computed: dict | None, expected: dict, source: Path):
    row = {"experiment": name, "report_section": report_section, "source": str(source.relative_to(ROOT))}
    if computed is None:
        row["status"] = "missing input"
        checks.append(row)
        return
    row["values"] = {}
    ok = True
    for key, exp in expected.items():
        got = float(computed[key])
        match = abs(got - exp) <= TOL
        ok &= match
        row["values"][key] = {"recomputed": round(got, 4), "report": exp, "match": match}
    row["status"] = "match" if ok else "MISMATCH"
    checks.append(row)


# 1. Literature adaptations (five participant folds, one prediction per person) -----
paper = NC / "paper_benchmark" / "outputs" / "full_174"
for fname, label, exp in [
    ("li_cp6_gnb_oof.json", "Li CP6 + naive Bayes", {"accuracy": 0.603, "balanced_accuracy": 0.565}),
    ("rostamikia_hierarchical_svm_oof.json", "Rostamikia hierarchical SVM", {"accuracy": 0.552, "balanced_accuracy": 0.525}),
    ("dice_net_3class_oof.json", "DICE-Net (3-class)", {"accuracy": 0.511, "balanced_accuracy": 0.483}),
    ("jiang_microstate_cascade_oof.json", "Jiang microstate cascade", {"accuracy": 0.414, "balanced_accuracy": 0.427}),
]:
    f = paper / fname
    comp = None
    if f.exists():
        rows = json.loads(f.read_text(encoding="utf-8"))
        comp = metrics(as_index([r["true"] for r in rows]), as_index([r["predicted"] for r in rows]))
    check(label, "Sec. 9 / Table 9-tensor", comp, exp, f)

# 2. Raw-waveform 1D CNN, five-fold ensemble ----------------------------------------
f = NC / "advanced_clean_spectral" / "outputs" / "raw_1d_cnn_cv_ensemble" / "probabilities.npz"
if f.exists():
    z = np.load(f, allow_pickle=False)
    oof = metrics(z["development_true"], z["out_of_fold_probabilities"].argmax(1))
    test = metrics(z["test_true"], z["test_probabilities"].argmax(1))
else:
    oof = test = None
check("Raw 1D CNN 5-fold ensemble, out-of-fold (147 dev.)", "Sec. 12.4", oof,
      {"accuracy": 0.660, "balanced_accuracy": 0.653, "macro_f1": 0.643}, f)
check("Raw 1D CNN 5-fold ensemble, test (27)", "Sec. 12.4", test,
      {"accuracy": 0.667, "balanced_accuracy": 0.693, "macro_f1": 0.667}, f)

# 3. Binary HC vs dementia, 3 x 5 nested CV ------------------------------------------
f = NC / "binary_logmel_search" / "outputs" / "nested" / "oof_predictions.json"
comp = None
if f.exists():
    rows = json.loads(f.read_text(encoding="utf-8"))
    per_repeat = []
    for rep in sorted({r["repeat"] for r in rows}):
        yt = np.concatenate([r["true"] for r in rows if r["repeat"] == rep])
        yp = np.concatenate([r["predictions"] for r in rows if r["repeat"] == rep])
        per_repeat.append(metrics(yt, yp))
    comp = {k: np.mean([m[k] for m in per_repeat]) for k in per_repeat[0]}
check("Binary log-mel SVM (pooled per repeat, mean of 3)", "Sec. 7 / Table binary", comp,
      {"accuracy": 0.720, "balanced_accuracy": 0.703, "macro_f1": 0.699}, f)

# 4. CP / G-NTF / TT repeated nested CV ----------------------------------------------
f = NC / "graph_ntf_svm" / "outputs" / "full" / "oof_predictions.json"
expected = {
    "direct": {"balanced_accuracy": 0.487, "macro_f1": 0.451},
    "ntf": {"balanced_accuracy": 0.468, "macro_f1": 0.423},
    "gntf": {"balanced_accuracy": 0.472, "macro_f1": 0.411},
    "shuffled": {"balanced_accuracy": 0.485},
    "tt": {"balanced_accuracy": 0.505, "macro_f1": 0.472},
}
rows = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
for method, exp in expected.items():
    comp = None
    if rows is not None:
        # Same aggregation as graph_ntf_svm/report.py: pool the out-of-fold predictions of
        # each repeat (every participant once), score the repeat, then average repeats.
        mrows = [r for r in rows if r["method"] == method]
        rec = []
        for rep in sorted({r["repeat"] for r in mrows}):
            yt = np.concatenate([as_index(r["true"]) for r in mrows if r["repeat"] == rep])
            yp = np.concatenate([np.asarray(r["probabilities"]).argmax(1) for r in mrows if r["repeat"] == rep])
            rec.append(metrics(yt, yp))
        comp = {k: np.mean([m[k] for m in rec]) for k in rec[0]} if rec else None
    check(f"Tensor CV: {method}", "Sec. 11.5 / Table tensor", comp, exp, f)

out = ROOT / "verification" / "recomputed_metrics.json"
out.write_text(json.dumps(checks, indent=2, default=bool), encoding="utf-8")
width = max(len(c["experiment"]) for c in checks)
for c in checks:
    vals = ", ".join(f"{k}={v['recomputed']:.3f} (report {v['report']:.3f})" for k, v in c.get("values", {}).items())
    print(f"{c['status']:<14} {c['experiment']:<{width}}  {vals}")
print(f"\nwrote {out.relative_to(ROOT)}")
