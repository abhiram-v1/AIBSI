"""Summarize the three paired-seed, five-fold experiment without rerunning models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DIRECTORIES = ("full", "seed_20261117", "seed_20261118")
ARMS = ("supervised", "masked", "masked_contrastive")
NAMES = ("Diagnosis only", "Diagnosis + mask", "Diagnosis + mask + contrast")


def main() -> None:
    summaries = [json.loads((OUT / name / "summary.json").read_text(encoding="utf-8")) for name in DIRECTORIES]
    means = {}
    for arm in ARMS:
        means[arm] = {
            "accuracy": float(np.mean([s[arm]["accuracy"] for s in summaries])),
            "macro_f1": float(np.mean([s[arm]["macro_f1"] for s in summaries])),
            "ftd_recall": float(np.mean([s[arm]["recall"]["FTD"] for s in summaries])),
            "balanced_accuracy": float(np.mean([s[arm]["balanced_accuracy"] for s in summaries])),
        }
    gntf_records = json.loads((HERE.parent / "graph_ntf_svm/outputs/full/oof_predictions.json").read_text(encoding="utf-8"))
    gntf = [r for r in gntf_records if r["method"] == "gntf" and r["repeat"] == 0]
    true = np.concatenate([r["true"] for r in gntf])
    predicted = np.concatenate([np.asarray(r["probabilities"]).argmax(axis=1) for r in gntf])
    matched_gntf = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "ftd_recall": float(recall_score(true, predicted, labels=[1], average=None, zero_division=0)[0]),
    }
    payload = {"seed_directories": DIRECTORIES, "arms": ARMS, "means_across_seeds": means,
               "matched_subjects_gntf_first_repeat": matched_gntf,
               "note": "Same 174 outer-test people per seed. Seed runs are correlated, not independent cohorts."}
    (OUT / "three_seed_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("macro_f1", "ftd_recall"), ("Macro F1", "FTD recall")):
        x = np.arange(3)
        for index, summary in enumerate(summaries):
            values = [summary[arm][metric] if metric == "macro_f1" else summary[arm]["recall"]["FTD"] for arm in ARMS]
            ax.plot(x, values, marker="o", alpha=.52, lw=1.2, label=f"Seed {index + 1}")
        ax.plot(x, [means[arm][metric] for arm in ARMS], color="#174A63", marker="s", lw=2.6, label="Mean")
        ax.set_xticks(x, NAMES, fontsize=8)
        ax.set_ylim(0, .65)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "three_seed_results.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Joint masked-context EEG experiment",
        "",
        "## Question",
        "",
        "Does a diagnosis head trained from the start improve AD/FTD/HC classification when paired with masked-context learning and a class-contrastive objective?",
        "",
        "## Protocol",
        "",
        "- Input: 6,719 existing eight-second log-mel EEG windows, each 19 electrodes x 24 frequency bins x 13 time frames. The fourth statistic mode used by G-NTF is not used here.",
        "- People: 174 (71 AD, 42 FTD, 61 HC). Five held-out participant folds match the first G-NTF repeat. All windows from a person stay in one fold.",
        "- Model: 88,899-parameter temporal/frequency CNN, two-layer context Transformer, participant-bag pooling, three-class diagnosis head from step one.",
        "- Arms: diagnosis cross-entropy (CE); CE + 0.2 masked-latent Smooth L1; CE + 0.2 masked loss + 0.1 supervised contrastive loss between participant embeddings.",
        "- Training: class-balanced subject bags, six sampled windows per person, up to 35 epochs. Twenty percent of outer-training people form an inner validation set. Choose the best epoch by participant macro F1, then balanced accuracy, then log loss. Outer-test people are evaluated only after checkpoint selection.",
        "- Seeds: 20261116, 20261117, 20261118. Same five outer-test folds, different training/validation seeds. Each seed predicts all 174 people once per arm.",
        "- This is wav2vec-inspired masked latent prediction on cached log-mel frames. It does not implement raw-waveform wav2vec 2.0 or vector quantization.",
        "",
        "## Held-out results",
        "",
        "Each entry is measured at participant level. The mean is across three correlated seed runs, not 522 independent patients.",
        "",
        "| Arm | Accuracy, seed 1 / 2 / 3 | Mean accuracy | Macro F1, seed 1 / 2 / 3 | Mean macro F1 | Mean FTD recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, name in zip(ARMS, NAMES):
        acc = " / ".join(f"{s[arm]['accuracy']:.3f}" for s in summaries)
        f1 = " / ".join(f"{s[arm]['macro_f1']:.3f}" for s in summaries)
        lines.append(f"| {name} | {acc} | {means[arm]['accuracy']:.3f} | {f1} | {means[arm]['macro_f1']:.3f} | {means[arm]['ftd_recall']:.3f} |")
    lines += [
        "",
        "![Three-seed macro F1 and FTD recall](three_seed_results.png)",
        "",
        "The diagnosis-only arm has the highest mean macro F1 (0.478) and mean FTD recall (0.254). Masking alone has similar mean accuracy (0.515 versus 0.511), but lower macro F1 (0.469). The added class-contrastive objective lowers both mean macro F1 (0.456) and FTD recall (0.198). These small, variable differences do not establish a reliable gain from masking or contrastive learning.",
        "",
        "Across the three supervised runs, 32 of 126 repeated FTD predictions are correct, 40 are called AD, and 54 are called HC. These are three predictions for each of 42 people, not 126 independent people. FTD confusion involves both AD and HC.",
        "",
        f"For context, G-NTF on these same outer-test people in its first repeat gave accuracy {matched_gntf['accuracy']:.3f}, macro F1 {matched_gntf['macro_f1']:.3f}, and FTD recall {matched_gntf['ftd_recall']:.3f}. This is a descriptive matched-subject reference: the representation, model selection, calibration, and number of seeds differ. Its full three-repeat summary is a separate result.",
        "",
        "## Interpretation and limits",
        "",
        "The joint objectives did not resolve the AD/FTD confusion in this experiment. Validation-selected epochs vary widely, and the cohort is small for training a neural encoder from scratch. The cached log-mel input retains only 13 time frames per window, which may limit masked-context learning. A larger unlabeled raw-EEG corpus and true two-stage pretraining would test a different hypothesis. No result here is external clinical validation.",
        "",
        "The cached moderate-preprocessing input inherits the source pipeline's limitation that EEGLAB discontinuity annotations were not explicitly rejected. Outputs in each seed directory contain protocol, per-fold checkpoints, histories, predictions, and summary metrics.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
