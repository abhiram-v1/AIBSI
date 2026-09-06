"""Leakage-safe decomposition-to-classifier benchmark for the EEG tensor.

Each unsupervised decomposition is fitted only on the training subjects of an
outer repeated stratified fold. Held-out subjects are projected onto the fixed
training components before classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorly as tl
from scipy.optimize import nnls
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorly.decomposition import non_negative_tucker
from tensorly.tucker_tensor import tucker_to_tensor
from xgboost import XGBClassifier


WORKSPACE = Path(r"C:\Projects\AIBSI")
TENSOR_DIR = WORKSPACE / "tensor_pipeline" / "outputs" / "tensorization"
OUTPUT = WORKSPACE / "tensor_pipeline" / "outputs" / "benchmark"
LABEL_NAMES = {0: "AD", 1: "FTD", 2: "CN"}
METHODS = ("pca", "nmf", "nonnegative_cp", "graph_nonnegative_cp", "nonnegative_tucker")
CLASSIFIERS = ("logistic_regression", "rbf_svm", "xgboost")


def anatomical_graph(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    edges = [
        ("Fp1", "Fp2"), ("Fp1", "F7"), ("Fp1", "F3"),
        ("Fp2", "F4"), ("Fp2", "F8"),
        ("F7", "F3"), ("F7", "T3"), ("F3", "Fz"), ("F3", "C3"),
        ("Fz", "F4"), ("Fz", "Cz"), ("F4", "F8"), ("F4", "C4"),
        ("F8", "T4"), ("T3", "C3"), ("T3", "T5"),
        ("C3", "Cz"), ("C3", "P3"), ("Cz", "C4"), ("Cz", "Pz"),
        ("C4", "T4"), ("C4", "P4"), ("T4", "T6"),
        ("T5", "P3"), ("T5", "O1"), ("P3", "Pz"), ("P3", "O1"),
        ("Pz", "P4"), ("Pz", "O1"), ("Pz", "O2"),
        ("P4", "T6"), ("P4", "O2"), ("T6", "O2"),
    ]
    lookup = {str(channel): i for i, channel in enumerate(channels)}
    adjacency = np.zeros((len(channels), len(channels)), dtype=np.float64)
    for left, right in edges:
        i, j = lookup[left], lookup[right]
        adjacency[i, j] = adjacency[j, i] = 1.0
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    normalized_adjacency = inv_sqrt @ adjacency @ inv_sqrt
    normalized_degree = np.eye(len(channels))
    return normalized_adjacency, normalized_degree, edges


def normalize_cp(factors: list[np.ndarray]) -> list[np.ndarray]:
    for mode in range(1, len(factors)):
        norms = np.linalg.norm(factors[mode], axis=0)
        norms = np.maximum(norms, 1e-12)
        factors[mode] /= norms
        factors[0] *= norms
    return factors


def cp_reconstruct(factors: list[np.ndarray]) -> np.ndarray:
    return np.einsum("nr,ir,jr,kr->nijk", *factors, optimize=True)


def fit_nonnegative_cp_mu(
    tensor: np.ndarray,
    rank: int,
    seed: int,
    graph_lambda: float,
    adjacency: np.ndarray,
    graph_degree: np.ndarray,
    iterations: int,
    initializations: int = 2,
) -> tuple[list[np.ndarray], dict]:
    x = np.asarray(tensor, dtype=np.float64)
    eps = 1e-12
    best = None
    best_objective = math.inf
    best_info = None
    x_norm_sq = float(np.sum(x * x))

    for init in range(initializations):
        rng = np.random.default_rng(seed + init * 1009)
        factors = [rng.random((size, rank)) + 0.05 for size in x.shape]
        factors = normalize_cp(factors)
        previous = math.inf
        completed = 0
        for iteration in range(iterations):
            a, b, c, d = factors
            gram = (b.T @ b) * (c.T @ c) * (d.T @ d)
            numerator = np.einsum("nijk,ir,jr,kr->nr", x, b, c, d, optimize=True)
            a *= numerator / np.maximum(a @ gram, eps)

            gram = (a.T @ a) * (c.T @ c) * (d.T @ d)
            numerator = np.einsum("nijk,nr,jr,kr->ir", x, a, c, d, optimize=True)
            if graph_lambda > 0:
                numerator += graph_lambda * (adjacency @ b)
            denominator = b @ gram
            if graph_lambda > 0:
                denominator += graph_lambda * (graph_degree @ b)
            b *= numerator / np.maximum(denominator, eps)

            gram = (a.T @ a) * (b.T @ b) * (d.T @ d)
            numerator = np.einsum("nijk,nr,ir,kr->jr", x, a, b, d, optimize=True)
            c *= numerator / np.maximum(c @ gram, eps)

            gram = (a.T @ a) * (b.T @ b) * (c.T @ c)
            numerator = np.einsum("nijk,nr,ir,jr->kr", x, a, b, c, optimize=True)
            d *= numerator / np.maximum(d @ gram, eps)
            factors = normalize_cp([a, b, c, d])
            completed = iteration + 1

            if iteration % 10 == 9 or iteration == iterations - 1:
                residual = x - cp_reconstruct(factors)
                reconstruction = float(np.sum(residual * residual) / max(x_norm_sq, eps))
                laplacian = graph_degree - adjacency
                smoothness = float(np.trace(factors[1].T @ laplacian @ factors[1]))
                objective = reconstruction + graph_lambda * smoothness / rank
                if abs(previous - objective) < 1e-7:
                    break
                previous = objective

        residual = x - cp_reconstruct(factors)
        reconstruction = float(np.sum(residual * residual) / max(x_norm_sq, eps))
        laplacian = graph_degree - adjacency
        smoothness = float(np.trace(factors[1].T @ laplacian @ factors[1]))
        objective = reconstruction + graph_lambda * smoothness / rank
        if objective < best_objective:
            best_objective = objective
            best = [factor.copy() for factor in factors]
            best_info = {
                "relative_squared_reconstruction_error": reconstruction,
                "channel_graph_smoothness": smoothness,
                "objective": objective,
                "iterations": completed,
                "initialization": init,
            }
    assert best is not None and best_info is not None
    return best, best_info


def project_nnls(tensors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    features = np.empty((len(tensors), basis.shape[1]), dtype=np.float64)
    for i, sample in enumerate(tensors):
        features[i], _ = nnls(basis, sample.reshape(-1), maxiter=1000)
    return features


def decompose(
    method: str,
    train: np.ndarray,
    test: np.ndarray,
    rank: int,
    seed: int,
    adjacency: np.ndarray,
    graph_degree: np.ndarray,
    cp_iterations: int,
    tucker_iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_flat = train.reshape(len(train), -1)
    test_flat = test.reshape(len(test), -1)

    if method == "pca":
        model = PCA(n_components=rank, svd_solver="full", random_state=seed)
        train_features = model.fit_transform(train_flat)
        test_features = model.transform(test_flat)
        return train_features, test_features, {
            "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum())
        }

    if method == "nmf":
        model = NMF(
            n_components=rank,
            init="nndsvda",
            solver="cd",
            max_iter=1000,
            tol=1e-5,
            random_state=seed,
        )
        train_features = model.fit_transform(train_flat)
        test_features = model.transform(test_flat)
        return train_features, test_features, {
            "reconstruction_error": float(model.reconstruction_err_),
            "iterations": int(model.n_iter_),
        }

    if method in {"nonnegative_cp", "graph_nonnegative_cp"}:
        graph_lambda = 0.0 if method == "nonnegative_cp" else 0.1
        factors, info = fit_nonnegative_cp_mu(
            train,
            rank=rank,
            seed=seed,
            graph_lambda=graph_lambda,
            adjacency=adjacency,
            graph_degree=graph_degree,
            iterations=cp_iterations,
        )
        _, channel, frequency, temporal = factors
        basis = np.einsum(
            "ir,jr,kr->rijk", channel, frequency, temporal, optimize=True
        ).reshape(rank, -1).T
        return project_nnls(train, basis), project_nnls(test, basis), info

    if method == "nonnegative_tucker":
        ranks = [rank, min(8, train.shape[1]), min(12, train.shape[2]), min(5, train.shape[3])]
        decomposition = non_negative_tucker(
            tl.tensor(train, dtype=tl.float64),
            rank=ranks,
            n_iter_max=tucker_iterations,
            init="svd",
            tol=1e-5,
            random_state=seed,
            verbose=False,
        )
        core, factors = decomposition
        subject_rank = core.shape[0]
        basis_tensor = tucker_to_tensor(
            (core, [np.eye(subject_rank), factors[1], factors[2], factors[3]])
        )
        basis = np.asarray(basis_tensor).reshape(subject_rank, -1).T
        train_features = project_nnls(train, basis)
        test_features = project_nnls(test, basis)
        reconstruction = tucker_to_tensor((core, factors))
        relative_error = float(
            np.sum((train - reconstruction) ** 2) / np.maximum(np.sum(train * train), 1e-12)
        )
        return train_features, test_features, {
            "ranks": ranks,
            "relative_squared_reconstruction_error": relative_error,
        }

    raise ValueError(f"Unknown method: {method}")


def classifier(name: str, seed: int):
    if name == "logistic_regression":
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        )
    elif name == "rbf_svm":
        estimator = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device="cpu",
            n_jobs=4,
            random_state=seed,
        )
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), estimator)


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    specificity = []
    for i in range(3):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity.append(float(tn / max(tn + fp, 1)))
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_ovr_auc": auc,
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(precision[i]),
                "recall_sensitivity": float(recall[i]),
                "specificity": specificity[i],
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cp-iterations", type=int, default=80)
    parser.add_argument("--tucker-iterations", type=int, default=60)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "feature_cache"
    cache.mkdir(exist_ok=True)
    tensor = np.load(TENSOR_DIR / "tensor_relative_sqrt_1_30hz.npy").astype(np.float64)
    labels = np.load(TENSOR_DIR / "labels.npy")
    subjects = np.load(TENSOR_DIR / "subjects.npy").astype(str)
    channels = np.load(TENSOR_DIR / "channels.npy").astype(str)
    adjacency, graph_degree, graph_edges = anatomical_graph(channels)

    splitter = RepeatedStratifiedKFold(
        n_splits=args.splits, n_repeats=args.repeats, random_state=5040
    )
    prediction_rows = []
    decomposition_rows = []
    start_all = time.perf_counter()

    for split_i, (train_idx, test_idx) in enumerate(splitter.split(tensor, labels)):
        repeat_i = split_i // args.splits
        fold_i = split_i % args.splits
        split_seed = 6000 + split_i
        for method in METHODS:
            cache_path = cache / f"split_{split_i:02d}__{method}__rank_{args.rank}.npz"
            method_start = time.perf_counter()
            if cache_path.exists():
                cached = np.load(cache_path, allow_pickle=False)
                train_features = cached["train_features"]
                test_features = cached["test_features"]
                info = json.loads(str(cached["info_json"]))
                cached_result = True
            else:
                train_features, test_features, info = decompose(
                    method,
                    tensor[train_idx],
                    tensor[test_idx],
                    rank=args.rank,
                    seed=split_seed,
                    adjacency=adjacency,
                    graph_degree=graph_degree,
                    cp_iterations=args.cp_iterations,
                    tucker_iterations=args.tucker_iterations,
                )
                np.savez_compressed(
                    cache_path,
                    train_features=train_features,
                    test_features=test_features,
                    info_json=json.dumps(info),
                )
                cached_result = False
            elapsed = time.perf_counter() - method_start
            decomposition_rows.append(
                {
                    "split": split_i,
                    "repeat": repeat_i,
                    "fold": fold_i,
                    "method": method,
                    "rank": args.rank,
                    "train_subjects": len(train_idx),
                    "test_subjects": len(test_idx),
                    "seconds": elapsed,
                    "cached": cached_result,
                    "details": info,
                }
            )

            for classifier_name in CLASSIFIERS:
                model = classifier(classifier_name, split_seed)
                model.fit(train_features, labels[train_idx])
                probability = model.predict_proba(test_features)
                model_classes = model.classes_
                aligned_probability = np.zeros((len(test_idx), 3), dtype=float)
                aligned_probability[:, model_classes.astype(int)] = probability
                predicted = np.argmax(aligned_probability, axis=1)
                for local_i, subject_i in enumerate(test_idx):
                    prediction_rows.append(
                        {
                            "split": split_i,
                            "repeat": repeat_i,
                            "fold": fold_i,
                            "method": method,
                            "classifier": classifier_name,
                            "rank": args.rank,
                            "subject": subjects[subject_i],
                            "true_label": int(labels[subject_i]),
                            "true_group": LABEL_NAMES[int(labels[subject_i])],
                            "predicted_label": int(predicted[local_i]),
                            "predicted_group": LABEL_NAMES[int(predicted[local_i])],
                            "probability_AD": float(aligned_probability[local_i, 0]),
                            "probability_FTD": float(aligned_probability[local_i, 1]),
                            "probability_CN": float(aligned_probability[local_i, 2]),
                        }
                    )
            print(
                f"split {split_i + 1:02d}/{args.splits * args.repeats} "
                f"method={method} seconds={elapsed:.2f} cached={cached_result}",
                flush=True,
            )

    predictions_path = OUTPUT / "fold_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    summaries = []
    repeat_metrics = []
    grouped = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["method"], row["classifier"])].append(row)

    for (method, classifier_name), rows in grouped.items():
        for repeat_i in range(args.repeats):
            selected = [row for row in rows if row["repeat"] == repeat_i]
            selected.sort(key=lambda row: row["subject"])
            y_true = np.array([row["true_label"] for row in selected])
            proba = np.array(
                [[row["probability_AD"], row["probability_FTD"], row["probability_CN"]] for row in selected]
            )
            y_pred = np.argmax(proba, axis=1)
            metrics = aggregate_metrics(y_true, y_pred, proba)
            repeat_metrics.append(
                {"method": method, "classifier": classifier_name, "repeat": repeat_i, **metrics}
            )

        # Average each subject's probabilities over the repeated outer CV runs.
        per_subject = defaultdict(list)
        true_by_subject = {}
        for row in rows:
            per_subject[row["subject"]].append(
                [row["probability_AD"], row["probability_FTD"], row["probability_CN"]]
            )
            true_by_subject[row["subject"]] = row["true_label"]
        ordered_subjects = sorted(per_subject)
        proba = np.array([np.mean(per_subject[subject], axis=0) for subject in ordered_subjects])
        y_true = np.array([true_by_subject[subject] for subject in ordered_subjects])
        y_pred = np.argmax(proba, axis=1)
        aggregate = aggregate_metrics(y_true, y_pred, proba)
        repeats_for_pair = [
            row for row in repeat_metrics if row["method"] == method and row["classifier"] == classifier_name
        ]
        aggregate["repeat_mean_balanced_accuracy"] = float(
            np.mean([row["balanced_accuracy"] for row in repeats_for_pair])
        )
        aggregate["repeat_sd_balanced_accuracy"] = float(
            np.std([row["balanced_accuracy"] for row in repeats_for_pair], ddof=1)
        )
        aggregate["repeat_mean_macro_f1"] = float(
            np.mean([row["macro_f1"] for row in repeats_for_pair])
        )
        aggregate["repeat_sd_macro_f1"] = float(
            np.std([row["macro_f1"] for row in repeats_for_pair], ddof=1)
        )
        summaries.append({"method": method, "classifier": classifier_name, **aggregate})

    summaries.sort(key=lambda row: row["repeat_mean_balanced_accuracy"], reverse=True)
    summary_csv = OUTPUT / "results_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method", "classifier", "accuracy", "balanced_accuracy", "macro_f1",
            "macro_ovr_auc", "log_loss", "repeat_mean_balanced_accuracy",
            "repeat_sd_balanced_accuracy", "repeat_mean_macro_f1", "repeat_sd_macro_f1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "evaluation": {
            "design": f"{args.repeats}x repeated stratified {args.splits}-fold subject-level CV",
            "subjects": len(subjects),
            "class_counts": {LABEL_NAMES[i]: int((labels == i).sum()) for i in range(3)},
            "fixed_rank": args.rank,
            "hyperparameter_policy": "Predeclared fixed settings; no test-fold tuning",
            "decomposition_fit_scope": "training subjects separately inside every outer fold",
            "heldout_projection": "fixed training basis with non-negative least squares where applicable",
        },
        "tensor": {
            "path": str(TENSOR_DIR / "tensor_relative_sqrt_1_30hz.npy"),
            "shape": list(tensor.shape),
        },
        "methods": list(METHODS),
        "classifiers": list(CLASSIFIERS),
        "graph": {
            "type": "fixed normalized anatomical 10-20 adjacency",
            "edges": graph_edges,
            "graph_lambda": 0.1,
        },
        "rank": args.rank,
        "iterations": {"cp": args.cp_iterations, "tucker": args.tucker_iterations},
        "results": summaries,
        "repeat_metrics": repeat_metrics,
        "decomposition_runs": decomposition_rows,
        "runtime_seconds": time.perf_counter() - start_all,
        "notes": [
            "All splits are at subject level because each tensor row is one subject.",
            "Results are comparative benchmark estimates, not an independently validated clinical model.",
            "PCA and flattened NMF are decomposition baselines; CP and Tucker preserve tensor modes.",
            "Graph CP regularizes the channel factor using a fixed anatomical graph, which does not inspect labels or test data.",
        ],
    }
    (OUTPUT / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFINAL RANKING")
    for row in summaries:
        print(
            f"{row['method']:24s} {row['classifier']:20s} "
            f"balanced_accuracy={row['repeat_mean_balanced_accuracy']:.3f}±{row['repeat_sd_balanced_accuracy']:.3f} "
            f"macro_f1={row['repeat_mean_macro_f1']:.3f}±{row['repeat_sd_macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
