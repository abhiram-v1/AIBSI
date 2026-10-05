"""Training-only Riemannian mapping and participant-bag classifiers."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


LABELS = np.array([0, 1, 2])


def spd_function(a: np.ndarray, fn) -> np.ndarray:
    values, vectors = eigh(a, check_finite=False)
    values = np.maximum(values, 1e-10)
    return (vectors * fn(values)) @ vectors.T


class LogEuclideanTangent:
    """Training-only log-Euclidean reference and congruent tangent mapping."""
    def fit(self, covariance: np.ndarray):
        # covariance: observations x bands x channels x channels
        logs = [[spd_function(c, np.log) for c in covariance[:, band]] for band in range(covariance.shape[1])]
        self.reference_ = np.stack([spd_function(np.mean(x, axis=0), np.exp) for x in logs])
        self.inv_sqrt_ = np.stack([spd_function(r, lambda v: v ** -0.5) for r in self.reference_])
        return self

    def transform(self, covariance: np.ndarray) -> np.ndarray:
        rows = []
        for observation in covariance:
            bands = []
            for c, inv in zip(observation, self.inv_sqrt_):
                tangent = spd_function(inv @ c @ inv, np.log)
                i, j = np.triu_indices(len(tangent))
                vec = tangent[i, j]
                vec[i != j] *= np.sqrt(2.0)
                bands.append(vec)
            rows.append(np.concatenate(bands))
        return np.asarray(rows)


def sample_weights(y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Equal total weight per class and per participant within class."""
    result = np.zeros(len(y), dtype=float)
    for label in LABELS:
        members = np.unique(subjects[y == label])
        for subject in members:
            ix = np.flatnonzero(subjects == subject)
            result[ix] = 1.0 / (len(LABELS) * len(members) * len(ix))
    result *= len(result) / result.sum()
    return result


def make_classifier(cfg: dict):
    kind = cfg["classifier"]
    if kind == "logistic":
        clf = LogisticRegression(C=cfg["C"], max_iter=5000, tol=1e-5)
    elif kind == "svm":
        clf = SVC(C=cfg["C"], kernel="rbf", gamma=cfg.get("gamma", "scale"), probability=False)
    elif kind == "linear_svm":
        clf = SVC(C=cfg["C"], kernel="linear", probability=False)
    else:
        clf = XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=10.0, reg_alpha=0.1, tree_method="hist", device="cpu",
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=9065, n_jobs=1, verbosity=0,
        )
    steps = []
    if cfg.get("selection"):
        steps.append(("select", SelectKBest(f_classif, k=cfg["selection"])))
    steps.extend([("scale", StandardScaler()), ("clf", clf)])
    return Pipeline(steps)


def fit_pipeline(cfg: dict, x: np.ndarray, y: np.ndarray, subjects: np.ndarray):
    model = make_classifier(cfg)
    weights = sample_weights(y, subjects)
    fit_params = {"scale__sample_weight": weights, "clf__sample_weight": weights}
    model.fit(x, y, **fit_params)
    return model


def aligned_scores(model, x: np.ndarray, expected: np.ndarray = LABELS) -> np.ndarray:
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x))
        if score.ndim == 1:
            score = np.column_stack([-score, score])
    else:
        score = np.asarray(model.predict_proba(x))
    result = np.full((len(x), len(expected)), -1e6, dtype=float)
    classes = np.asarray(model.classes_)
    for j, label in enumerate(classes):
        result[:, np.flatnonzero(expected == label)[0]] = score[:, j]
    return result


class HierarchicalModel:
    """CN-v-dementia gate followed by AD-v-FTD specialist."""
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, x: np.ndarray, y: np.ndarray, subjects: np.ndarray):
        gate_y = (y != 2).astype(int)
        specialist_ix = y != 2
        self.gate_ = make_classifier(self.cfg)
        gate_w = sample_weights_binary(gate_y, subjects)
        self.gate_.fit(x, gate_y, scale__sample_weight=gate_w, clf__sample_weight=gate_w)
        self.specialist_ = make_classifier(self.cfg)
        specialist_y = y[specialist_ix]
        specialist_subjects = subjects[specialist_ix]
        specialist_w = sample_weights_binary(specialist_y, specialist_subjects)
        self.specialist_.fit(x[specialist_ix], specialist_y,
                             scale__sample_weight=specialist_w, clf__sample_weight=specialist_w)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        gate = binary_score(self.gate_, x)
        specialist = binary_score(self.specialist_, x)
        # Higher gate means dementia. Within dementia, positive means FTD.
        return np.column_stack([gate - specialist, gate + specialist, -gate])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.decision_function(x).argmax(axis=1)


def binary_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x)).reshape(-1)
    p = model.predict_proba(x)
    return np.log(np.maximum(p[:, 1], 1e-9) / np.maximum(p[:, 0], 1e-9))


def sample_weights_binary(y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    result = np.zeros(len(y), dtype=float)
    classes = np.unique(y)
    for label in classes:
        members = np.unique(subjects[y == label])
        for subject in members:
            ix = np.flatnonzero(subjects == subject)
            result[ix] = 1.0 / (len(classes) * len(members) * len(ix))
    result *= len(result) / result.sum()
    return result


def candidate_configs() -> list[dict]:
    configs = []
    for bags in (1, 4, 8):
        for representation, selections in (
            ("power", (None,)),
            ("riemann", (64, 128)),
            ("connectivity", (64, 128)),
            ("hybrid", (128,)),
        ):
            for selection in selections:
                for classifier, c in (("logistic", 0.1), ("logistic", 1.0), ("svm", 1.0), ("svm", 10.0)):
                    for mode in ("multiclass", "hierarchical"):
                        configs.append(dict(bags=bags, representation=representation, selection=selection,
                                            classifier=classifier, C=c, mode=mode))
        # Two conservative tree controls per bag partition.
        for representation, selection in (("connectivity", 128), ("hybrid", 128)):
            configs.append(dict(bags=bags, representation=representation, selection=selection,
                                classifier="xgboost", C=None, mode="multiclass"))
    return configs
