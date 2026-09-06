"""CPU launcher using scikit-learn histogram boosting as the tree model.

The main benchmark module contains the decomposition implementation. This
launcher avoids importing the unavailable external XGBoost wheel and replaces
only the classifier factory; all fold-local decomposition behavior is unchanged.
"""

from __future__ import annotations

import sys
import types

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Allow the main module to load without the optional external package. The dummy
# is never instantiated because classifier() is replaced below.
stub = types.ModuleType("xgboost")
stub.XGBClassifier = object
sys.modules.setdefault("xgboost", stub)

import run_decomposition_benchmark as benchmark  # noqa: E402


benchmark.CLASSIFIERS = (
    "logistic_regression",
    "rbf_svm",
    "hist_gradient_boosting",
)


def cpu_classifier(name: str, seed: int):
    if name == "logistic_regression":
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        )
        return make_pipeline(StandardScaler(), estimator)
    if name == "rbf_svm":
        estimator = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )
        return make_pipeline(StandardScaler(), estimator)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=2,
            learning_rate=0.05,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(name)


benchmark.classifier = cpu_classifier


if __name__ == "__main__":
    benchmark.main()
