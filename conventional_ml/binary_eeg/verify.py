"""Replay saved outer predictions and audit participant separation."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conventional_ml/outputs/binary_eeg"
TRAIN = Path(__file__).with_name("train.py")
spec = importlib.util.spec_from_file_location("binary_eeg_train", TRAIN)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def main():
    subjects, y, matrices = module.load()
    rows = [json.loads((OUT / f"fold_{i:02d}.json").read_text(encoding="utf-8"))
            for i in range(15)]
    fixed = dict(features="global", k=None, model="logistic", C=1.0)
    replayed = 0
    for fold, (train, test) in enumerate(RepeatedStratifiedKFold(
            n_splits=5, n_repeats=3, random_state=9066).split(subjects, y)):
        row = rows[fold]
        assert row["train_subjects"] == subjects[train].tolist()
        assert row["test_subjects"] == subjects[test].tolist()
        assert set(row["train_subjects"]).isdisjoint(row["test_subjects"])
        assert row["true"] == y[test].tolist()
        for name, cfg in (("fixed_global_logistic", fixed),
                          ("inner_selected", row["selected_config"])):
            matrix = matrices[cfg["features"]]
            model = module.make_model(cfg).fit(matrix[train], y[train])
            prediction = model.predict(matrix[test]).astype(int).tolist()
            assert prediction == row["predictions"][name]
            assert module.metric(y[test], prediction) == row["outer_metrics"][name]
            replayed += 1
    for repeat in range(3):
        group = rows[repeat * 5:(repeat + 1) * 5]
        assert len(set(sum((r["test_subjects"] for r in group), []))) == 88
    recomputed = module.summary(rows)
    saved = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert recomputed["endpoints"] == saved["endpoints"]
    result = {"verified": True, "outer_folds": 15, "model_predictions_replayed": replayed,
              "participants_per_repeat": 88, "subject_leakage": False,
              "metrics_recomputed": True,
              "source": "Direct derivative .set feature preparation; no tensor pipeline input"}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
