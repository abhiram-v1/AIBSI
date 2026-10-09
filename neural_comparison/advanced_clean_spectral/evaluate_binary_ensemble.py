from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
PREPARED = HERE / "outputs" / "moderate_log_mel" / "prepared"
OUT = HERE / "outputs" / "results" / "binary_ensemble"
sys.path.insert(0, str(HERE))
import pipeline as eeg  # noqa: E402
from evaluate_site_normalized import site_normalize  # noqa: E402


def make_model(config: dict, probability: bool) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=config["pca"], svd_solver="randomized", random_state=eeg.SEED)),
        ("svm", SVC(
            C=config["C"], kernel=config["kernel"], gamma="scale", class_weight="balanced",
            probability=probability, decision_function_shape="ovr", random_state=eeg.SEED,
        )),
    ])


def select_binary_config(x: np.ndarray, y: np.ndarray, fit: np.ndarray, classes: tuple[int, int]) -> dict:
    use = fit[np.isin(y[fit], classes)]
    configs = [
        {"pca": pca, "C": c, "kernel": kernel}
        for pca, c, kernel in itertools.product([10, 20, 30, 40], [0.01, 0.03, 0.1, 0.3, 1.0], ["linear", "rbf"])
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=eeg.SEED)
    rows = []
    for config in configs:
        score = cross_val_score(make_model(config, False), x[use], y[use], cv=cv, scoring="f1_macro").mean()
        rows.append({**config, "train_cv_macro_f1": float(score)})
    rows.sort(key=lambda row: (row["train_cv_macro_f1"], -row["pca"]), reverse=True)
    return rows[0]


def fit_binary(
    x: np.ndarray,
    y: np.ndarray,
    fit: np.ndarray,
    classes: tuple[int, int],
    config: dict,
) -> Pipeline:
    use = fit[np.isin(y[fit], classes)]
    return make_model(config, True).fit(x[use], y[use])


def probability_for(model: Pipeline, x: np.ndarray, class_id: int) -> np.ndarray:
    classes = model.named_steps["svm"].classes_
    column = int(np.flatnonzero(classes == class_id)[0])
    return model.predict_proba(x)[:, column]


def fuse(p3: np.ndarray, p_ad: np.ndarray, p_ftd: np.ndarray, alpha: float, ftd_bias: float, hc_bias: float) -> np.ndarray:
    epsilon = 1e-7
    binary = np.column_stack([
        p_ad,
        p_ftd,
        np.sqrt(np.maximum(1.0 - p_ad, epsilon) * np.maximum(1.0 - p_ftd, epsilon)),
    ])
    scores = alpha * np.log(np.maximum(p3, epsilon)) + (1.0 - alpha) * np.log(np.maximum(binary, epsilon))
    scores[:, 1] += ftd_bias
    scores[:, 2] += hc_bias
    return np.argmax(scores, axis=1)


def hard_fusions(
    three: np.ndarray,
    ad: np.ndarray,
    ftd: np.ndarray,
    p_ad: np.ndarray,
    p_ftd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gate = []
    majority = []
    for index, (ad_vote, ftd_vote, three_vote) in enumerate(zip(ad, ftd, three)):
        if ad_vote == 2 and ftd_vote == 2:
            gate_vote = 2
        elif ad_vote == 0 and ftd_vote == 2:
            gate_vote = 0
        elif ad_vote == 2 and ftd_vote == 1:
            gate_vote = 1
        elif three_vote in (0, 1):
            gate_vote = int(three_vote)
        else:
            gate_vote = 0 if p_ad[index] >= p_ftd[index] else 1
        gate.append(gate_vote)
        votes = [int(ad_vote), int(ftd_vote), int(three_vote)]
        counts = [votes.count(class_id) for class_id in range(3)]
        winners = [class_id for class_id, count in enumerate(counts) if count == max(counts)]
        majority.append(int(three_vote) if int(three_vote) in winners else winners[0])
    return np.asarray(gate), np.asarray(majority)


def train_probabilities(
    flat: np.ndarray,
    labels: np.ndarray,
    participants: list[dict],
    fit: np.ndarray,
    predict: np.ndarray,
    ad_config: dict,
    ftd_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    normalized = site_normalize(flat, participants, fit)
    three_config = {"pca": 20, "C": 0.1, "kernel": "linear"}
    three = make_model(three_config, True).fit(normalized[fit], labels[fit])
    ad = fit_binary(normalized, labels, fit, (0, 2), ad_config)
    ftd = fit_binary(normalized, labels, fit, (1, 2), ftd_config)
    p3 = three.predict_proba(normalized[predict])
    p_ad = probability_for(ad, normalized[predict], 0)
    p_ftd = probability_for(ftd, normalized[predict], 1)
    details = {
        "three_class_prediction": three.predict(normalized[predict]),
        "ad_binary_prediction": ad.predict(normalized[predict]),
        "ftd_binary_prediction": ftd.predict(normalized[predict]),
    }
    return p3, p_ad, p_ftd, details


def binary_metrics(y: np.ndarray, prediction: np.ndarray, indexes: np.ndarray, classes: tuple[int, int]) -> dict:
    use = np.flatnonzero(np.isin(y[indexes], classes))
    return {
        "subjects": int(len(use)),
        "accuracy": float(accuracy_score(y[indexes][use], prediction[use])),
        "macro_f1": float(f1_score(y[indexes][use], prediction[use], average="macro", zero_division=0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tensor = np.load(PREPARED / "log_mel.npy", mmap_mode="r")
    window_labels = np.load(PREPARED / "labels.npy", mmap_mode="r")
    subject_index = np.load(PREPARED / "subject_index.npy", mmap_mode="r")
    participants = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))["participants"]
    features, labels = eeg.subject_features(tensor, window_labels, subject_index, participants)
    flat = features.reshape(len(features), -1)
    splits = eeg.make_subject_splits(participants)
    train = np.asarray(splits["train"], dtype=int)
    val = np.asarray(splits["val"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)
    development = np.concatenate([train, val])

    train_normalized = site_normalize(flat, participants, train)
    ad_config = select_binary_config(train_normalized, labels, train, (0, 2))
    ftd_config = select_binary_config(train_normalized, labels, train, (1, 2))
    print("AD vs HC:", ad_config, flush=True)
    print("FTD vs HC:", ftd_config, flush=True)

    val_p3, val_ad, val_ftd, val_details = train_probabilities(
        flat, labels, participants, train, val, ad_config, ftd_config
    )
    candidates = []
    for alpha, ftd_bias, hc_bias in itertools.product(
        np.linspace(0.0, 1.0, 21), np.linspace(-0.4, 0.4, 9), np.linspace(-0.3, 0.3, 7)
    ):
        prediction = fuse(val_p3, val_ad, val_ftd, float(alpha), float(ftd_bias), float(hc_bias))
        candidates.append({
            "alpha_three_class": float(alpha),
            "ftd_bias": float(ftd_bias),
            "hc_bias": float(hc_bias),
            "val_macro_f1": float(f1_score(labels[val], prediction, average="macro", zero_division=0)),
            "val_accuracy": float(accuracy_score(labels[val], prediction)),
            "bias_magnitude": float(abs(ftd_bias) + abs(hc_bias)),
        })
    candidates.sort(
        key=lambda row: (row["val_macro_f1"], row["val_accuracy"], -row["bias_magnitude"], row["alpha_three_class"]),
        reverse=True,
    )
    selected = candidates[0]

    test_p3, test_ad, test_ftd, test_details = train_probabilities(
        flat, labels, participants, development, test, ad_config, ftd_config
    )
    ensemble_prediction = fuse(
        test_p3, test_ad, test_ftd,
        selected["alpha_three_class"], selected["ftd_bias"], selected["hc_bias"],
    )
    ensemble_metrics = eeg.metrics(labels[test], ensemble_prediction)
    gate_prediction, majority_prediction = hard_fusions(
        test_details["three_class_prediction"], test_details["ad_binary_prediction"],
        test_details["ftd_binary_prediction"], test_ad, test_ftd,
    )
    gate_metrics = eeg.metrics(labels[test], gate_prediction)
    majority_metrics = eeg.metrics(labels[test], majority_prediction)
    baseline = json.loads((HERE / "outputs" / "results" / "site_normalized_metrics.json").read_text(encoding="utf-8"))["test"]
    payload = {
        "design": "AD-vs-HC SVM + FTD-vs-HC SVM + existing three-class SVM; fusion selected on validation macro F1",
        "ad_vs_hc_config": ad_config,
        "ftd_vs_hc_config": ftd_config,
        "validation_selected_fusion": selected,
        "validation_binary_metrics": {
            "ad_vs_hc": binary_metrics(labels, val_details["ad_binary_prediction"], val, (0, 2)),
            "ftd_vs_hc": binary_metrics(labels, val_details["ftd_binary_prediction"], val, (1, 2)),
        },
        "test_binary_metrics": {
            "ad_vs_hc": binary_metrics(labels, test_details["ad_binary_prediction"], test, (0, 2)),
            "ftd_vs_hc": binary_metrics(labels, test_details["ftd_binary_prediction"], test, (1, 2)),
        },
        "baseline_three_class": baseline,
        "probability_ensemble_test": ensemble_metrics,
        "hard_gate_ensemble_test": gate_metrics,
        "majority_vote_ensemble_test": majority_metrics,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    confusion = np.asarray(majority_metrics["confusion_matrix"])
    image = axes[0].imshow(confusion, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(confusion[row, column]), ha="center", va="center", fontsize=12)
    axes[0].set_xticks(range(3), eeg.CLASSES)
    axes[0].set_yticks(range(3), eeg.CLASSES)
    axes[0].set(xlabel="Predicted", ylabel="True", title="Binary-specialist majority vote")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    measures = ["accuracy", "macro_f1", "macro_recall"]
    comparison_rows = [baseline, majority_metrics, gate_metrics, ensemble_metrics]
    for offset, measure in enumerate(measures):
        bars = axes[1].bar(
            np.arange(4) + (offset - 1) * 0.24,
            [row[measure] for row in comparison_rows],
            0.24,
            label=measure.replace("_", " ").title(),
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=2)
    axes[1].set_xticks(
        np.arange(4), ["Three-class\nSVM", "Majority\nvote", "Hard\ngate", "Probability\nfusion"]
    )
    axes[1].set(ylim=(0, 1), ylabel="Held-out participant score", title="Does the binary ensemble help?")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
