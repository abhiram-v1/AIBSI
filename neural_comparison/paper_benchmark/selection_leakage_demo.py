"""Feature-selection leakage demo on pure noise (no EEG data needed).

Mimics the recipe described in the Methods of Rostamikia et al. 2024
(Cognitive Neurodynamics, doi:10.1007/s11571-024-10152-7): univariate
t-test feature screening using ALL participants' labels, followed by SVM
cross-validation on the surviving features. As written, the paper's feature
selection precedes the train/test split.

Here the "features" are random numbers and unrelated to the labels, so any
accuracy above chance is produced by the procedure itself. The class sizes
match ds004504 dementia (AD+FTD = 59) vs healthy (29). The correct recipe
repeats the screening inside each training fold only.

This is a demonstration of the mechanism, not a re-analysis of the paper's
data: it cannot say how much of the published 87.8% / 93.5% is real signal.

Run:  py -3.10 neural_comparison/paper_benchmark/selection_leakage_demo.py
Writes nothing; prints a table (saved copy: selection_leakage_demo_output.txt).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

N_DEMENTIA, N_HC, N_FEATURES, REPEATS = 59, 29, 2000, 30
LABELS = np.r_[np.ones(N_DEMENTIA), np.zeros(N_HC)].astype(int)


def select(x: np.ndarray, y: np.ndarray, p_threshold: float) -> np.ndarray:
    test = ttest_ind(x[y == 1], x[y == 0], axis=0)
    return np.flatnonzero(test.pvalue < p_threshold)


def cross_validate(x, y, p_threshold: float, select_inside_folds: bool, seed: int):
    folds = StratifiedKFold(10, shuffle=True, random_state=seed)
    prediction = np.empty(len(y), dtype=int)
    # Leaky: every person's label is used once, up front, to choose features.
    global_columns = None if select_inside_folds else select(x, y, p_threshold)
    for train, test in folds.split(x, y):
        columns = select(x[train], y[train], p_threshold) if select_inside_folds else global_columns
        if len(columns) == 0:  # nothing survived screening: predict the majority class
            prediction[test] = 1
            continue
        model = SVC(kernel="linear").fit(x[train][:, columns], y[train])
        prediction[test] = model.predict(x[test][:, columns])
    kept = np.nan if select_inside_folds else len(global_columns)
    return (prediction == y).mean(), balanced_accuracy_score(y, prediction), kept


def main() -> None:
    rng = np.random.default_rng(0)
    for p_threshold in (0.05, 0.005):
        results = {False: [], True: []}
        for repeat in range(REPEATS):
            x = rng.standard_normal((len(LABELS), N_FEATURES))  # no real signal
            for inside in (False, True):
                results[inside].append(cross_validate(x, LABELS, p_threshold, inside, repeat))
        for inside in (False, True):
            a = np.array(results[inside])
            name = "selection INSIDE folds (correct)" if inside else "selection on ALL people first (leaky)"
            extra = "" if inside else f"  (features kept: {a[:, 2].mean():.0f} of {N_FEATURES})"
            print(f"p<{p_threshold}: {name:40s} accuracy {100 * a[:, 0].mean():5.1f}%  "
                  f"balanced accuracy {100 * a[:, 1].mean():5.1f}%{extra}")
    print(f"majority-class accuracy = {100 * N_DEMENTIA / len(LABELS):.1f}%, chance balanced accuracy = 50%")


if __name__ == "__main__":
    main()
