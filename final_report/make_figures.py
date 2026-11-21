"""Figures for the final report. Reads only saved project outputs; no model is refit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OUTS = ROOT / "neural_comparison" / "advanced_clean_spectral" / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "figure.dpi": 150})
BLUE, RED, GOLD, GREEN, GRAY = "#276FBF", "#C74B50", "#D89B2B", "#40826D", "#8A97A3"
PALE = "#DCEAF7"

manifest = json.load(open(OUTS / "waveform_tucker_before_stft_2026-11-05" / "prepared" / "waveform_manifest.json"))
P = manifest["participants"]

# ---- Fig 2: CNN variants, validation vs test macro F1 (horizontal)
rows = [("Window CNN, 3-seed mean", 0.522, 0.575),
        ("Window CNN, selected seed", 0.614, 0.696),
        ("Attention-MIL CNN", 0.720, 0.664),
        ("CNN + MIL + SVM fusion", 0.776, 0.693),
        ("5-fold CNN ensemble*", 0.643, 0.667)]
fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
y = np.arange(len(rows))[::-1]; h = 0.36
ax.barh(y + h/2, [r[1] for r in rows], h, color=GRAY, label="Validation (26)")
ax.barh(y - h/2, [r[2] for r in rows], h, color=BLUE, label="Test (27)")
ax.axvline(0.643, color=RED, ls="--", lw=1)
for yi, r in zip(y, rows):
    ax.text(r[1] + 0.01, yi + h/2, f"{r[1]:.2f}", va="center", fontsize=6)
    ax.text(r[2] + 0.01, yi - h/2, f"{r[2]:.2f}", va="center", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, 4.6); ax.set_xlabel("Participant macro F1", fontsize=7)
ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.savefig(FIG / "fig_cnn_variants.pdf"); plt.close(fig)

# ---- Fig 3/4: per-site analysis of the five-fold CNN ensemble (OOF for 147 dev + averaged test for 27)
d = np.load(OUTS / "raw_1d_cnn_cv_ensemble" / "probabilities.npz", allow_pickle=True)
idx = np.concatenate([d["development_indices"], d["test_indices"]])
true = np.concatenate([d["development_true"], d["test_true"]])
pred = np.concatenate([d["out_of_fold_probabilities"], d["test_probabilities"]]).argmax(1)
site = np.array([P[i]["site"] for i in idx])
def cm(mask):
    m = np.zeros((3, 3), int)
    for a, b in zip(true[mask], pred[mask]): m[a, b] += 1
    return m
names = ["AD", "FTD", "HC"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
for ax, (title, mask) in zip(axes, [("All 174", site == site), ("AHEPA (88)", site == "AHEPA"), ("BrainLat (86)", site != "AHEPA")]):
    m = cm(mask); rn = m / m.sum(1, keepdims=True)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]}", ha="center", va="center", color="white" if rn[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=9)
axes[0].set_ylabel("True")
fig.savefig(FIG / "fig_confusion_sites.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
sites = ["AHEPA", "AR", "CL"]; w = 0.26
for k, (lab, c) in enumerate(zip(names, [RED, GOLD, BLUE])):
    rec = []
    for s in sites:
        mk = (site == s) & (true == k)
        rec.append((pred[mk] == k).mean())
    b = ax.bar(np.arange(3) + (k - 1) * w, rec, w, color=c, label=lab)
    ax.bar_label(b, labels=[f"{r:.2f}" for r in rec], fontsize=6, padding=1)
ax.set_xticks(range(3)); ax.set_xticklabels(sites); ax.set_ylim(0, 1.3)
ax.set_ylabel("Recall"); ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")
fig.savefig(FIG / "fig_site_recall.pdf"); plt.close(fig)

# ---- Binary task: headline metrics and site-wise macro F1
binary = json.load(open(ROOT / "neural_comparison" / "binary_logmel_search" / "outputs" / "nested" / "summary.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
metric_names = ["Macro F1", "Balanced\naccuracy", "Accuracy", "ROC AUC"]
metric_values = [binary["mean"]["macro_f1"], binary["mean"]["balanced_accuracy"],
                 binary["mean"]["accuracy"], binary["mean"]["roc_auc"]]
bars = axes[0].bar(np.arange(4), metric_values, color=[BLUE, GREEN, GOLD, RED], width=0.68)
axes[0].axhline(binary["always_dementia"]["macro_f1"], color=GRAY, ls="--", lw=1,
                label="Always-dementia macro F1")
axes[0].bar_label(bars, labels=[f"{v:.3f}" for v in metric_values], fontsize=7, padding=2)
axes[0].set_xticks(np.arange(4), metric_names, fontsize=7)
axes[0].set_ylim(0, 0.9); axes[0].set_ylabel("Participant-level score")
axes[0].set_title("Nested CV on 174 participants", fontsize=9)
axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
site_rows = binary["groups"]["site"]
site_values = [site_rows[s]["mean"]["macro_f1"] for s in ["AHEPA", "AR", "CL"]]
bars = axes[1].bar(np.arange(3), site_values, color=[GRAY, GOLD, BLUE], width=0.62)
axes[1].bar_label(bars, labels=[f"{v:.3f}" for v in site_values], fontsize=7, padding=2)
axes[1].axhline(binary["mean"]["macro_f1"], color=RED, ls="--", lw=1, label="Overall")
axes[1].set_xticks(np.arange(3), ["AHEPA\n(n=88)", "AR\n(n=48)", "CL\n(n=38)"], fontsize=7)
axes[1].set_ylim(0, 0.9); axes[1].set_ylabel("Macro F1")
axes[1].set_title("Performance differs by site", fontsize=9)
axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
for ax in axes: ax.grid(axis="y", alpha=0.2)
fig.savefig(FIG / "fig_binary_results.pdf"); plt.close(fig)

# ---- Three-class progression: keep fixed-split and CV/OOF protocols visually separate
fixed = [
    ("Log-mel\nSVM", 0.643, BLUE), ("EEGPT", 0.550, GRAY),
    ("SVM +\nEEGPT", 0.678, GREEN), ("TT", 0.464, GOLD),
    ("Tucker", 0.576, GOLD), ("Raw\nCNN", 0.696, RED),
]
cv = [
    ("Log-mel\nSVM", 0.490, BLUE), ("Random\nforest", 0.517, GREEN),
    ("CP6 +\nNB*", 0.565, GOLD), ("Raw CNN\nOOF**", 0.643, RED),
]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
for ax, rows, title in [(axes[0], fixed, "Fixed 27-person test (exploratory)"),
                         (axes[1], cv, "Participant CV / out-of-fold")]:
    values = [r[1] for r in rows]
    bars = ax.bar(np.arange(len(rows)), values, color=[r[2] for r in rows], width=0.68)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=6.5, padding=2)
    ax.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=6.5)
    ax.set_ylim(0.3, 0.78); ax.set_ylabel("Macro F1"); ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
axes[1].text(0.02, -0.28, "* 5-fold, one repeat; ** OOF on 147 development participants",
             transform=axes[1].transAxes, fontsize=6, va="top")
fig.savefig(FIG / "fig_threeclass_progression.pdf"); plt.close(fig)

# ---- Fixed-split confusion matrices for the most discussed three-class models
cms = [
    ("Log-mel SVM\nF1 0.643", np.array([[8, 1, 2], [1, 3, 2], [1, 2, 7]])),
    ("SVM + EEGPT\nF1 0.678", np.array([[8, 1, 2], [1, 3, 2], [1, 1, 8]])),
    ("Raw 1D CNN\nF1 0.696", np.array([[7, 4, 0], [2, 4, 0], [0, 2, 8]])),
]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
for ax, (title, matrix) in zip(axes, cms):
    normalised = matrix / matrix.sum(1, keepdims=True)
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if normalised[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(3), names, fontsize=7); ax.set_yticks(range(3), names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7); ax.set_title(title, fontsize=8.5)
axes[0].set_ylabel("True", fontsize=7)
fig.savefig(FIG / "fig_threeclass_confusions.pdf"); plt.close(fig)

# ---- Tensor ablations: separate fixed-split and repeated-CV evidence
tensor_fixed = [("Undecomp.\nSVM", .643), ("TT", .464), ("Pre-pool\nTucker", .576),
                ("Waveform\nTucker", .519)]
tensor_cv = [("Direct", .451), ("NTF", .423), ("G-NTF", .411), ("TT", .472), ("CP6 + NB*", .565)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
for ax, rows, title in [(axes[0], tensor_fixed, "Fixed 27-person test"),
                         (axes[1], tensor_cv, "Participant-level cross-validation")]:
    values = [r[1] for r in rows]
    colors = [BLUE if i == 0 else GOLD for i in range(len(rows))]
    bars = ax.bar(np.arange(len(rows)), values, color=colors, width=0.68)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=6.5, padding=2)
    ax.set_xticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=6.5)
    ax.set_ylim(0.3, 0.7); ax.set_ylabel("Macro F1"); ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
axes[1].text(0.02, -0.24, "* Five folds, one repeat; optimiser failed to converge in 3/5 folds",
             transform=axes[1].transAxes, fontsize=6, va="top")
fig.savefig(FIG / "fig_tensor_results.pdf"); plt.close(fig)

# ---- Combined site audit to keep the related evidence on one report page
fig = plt.figure(figsize=(7.0, 3.0), constrained_layout=True)
gs = GridSpec(1, 4, figure=fig, width_ratios=[1.25, 1, 1, 1])
ax = fig.add_subplot(gs[0, 0]); sites = ["AHEPA", "AR", "CL"]; w = 0.24
for k, (lab, color) in enumerate(zip(names, [RED, GOLD, BLUE])):
    rec = []
    for s in sites:
        mask = (site == s) & (true == k)
        rec.append((pred[mask] == k).mean())
    bars = ax.bar(np.arange(3) + (k - 1) * w, rec, w, color=color, label=lab)
    ax.bar_label(bars, labels=[f"{r:.2f}" for r in rec], fontsize=5.5, padding=1)
ax.set_xticks(range(3), sites, fontsize=7); ax.set_ylim(0, 1.18); ax.set_ylabel("Recall")
ax.set_title("Recall by site", fontsize=9); ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")
ax.grid(axis="y", alpha=0.2)
for pos, (title, mask) in enumerate([("All 174", site == site), ("AHEPA", site == "AHEPA"), ("BrainLat", site != "AHEPA")], start=1):
    ax = fig.add_subplot(gs[0, pos]); matrix = cm(mask); normalised = matrix / matrix.sum(1, keepdims=True)
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if normalised[i, j] > 0.55 else "black", fontsize=8)
    ax.set_xticks(range(3), names, fontsize=6.5); ax.set_yticks(range(3), names, fontsize=6.5)
    ax.set_xlabel("Predicted", fontsize=6.5); ax.set_title(title, fontsize=8.5)
    if pos == 1: ax.set_ylabel("True", fontsize=6.5)
fig.savefig(FIG / "fig_site_audit.pdf"); plt.close(fig)
print("done")
