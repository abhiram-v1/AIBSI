"""Interpretability analysis of the saved tensor models (CP/G-NTF, Tensor Train, Tucker/HOSVD).

Reads only saved project artefacts; refits only the deterministic HOSVD covariance
step (fast).  All class comparisons are descriptive full-cohort summaries.
Run with:  py -3.10 final_report/tensor_interpretability.py
"""
import json, sys, importlib
from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.stats import kruskal, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "neural_comparison" / "advanced_clean_spectral"
GNTF = ROOT / "neural_comparison" / "graph_ntf_svm" / "outputs" / "full" / "final_gntf"
FIG = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ADV)); sys.path.insert(0, str(ROOT / "tensor_pipeline" / "v2"))
import pipeline as eeg
tucker = importlib.import_module("train_prepool_tucker_svm")

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
CLS = ["AD", "FTD", "HC"]; COL = ["#C74B50", "#D89B2B", "#276FBF"]
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 25), ("gamma", 25, 40.01)]
REGIONS = {"Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"], "Central": ["C3", "C4", "Cz"],
           "Temporal": ["T7", "T8"], "Parietal": ["P3", "P4", "Pz", "P7", "P8"], "Occipital": ["O1", "O2"]}

manifest = json.loads((ADV / "outputs/moderate_log_mel/prepared/manifest.json").read_text(encoding="utf-8"))
P = manifest["participants"]
channels = manifest["channels"]
labels = np.array([CLS.index(p["label"]) for p in P])
sites = np.array([p["site"] for p in P])
mel_hz = eeg.inverse_mel(np.linspace(eeg.mel_scale(1.0), eeg.mel_scale(40.0), eeg.MEL_BINS + 2))[1:-1]
band_of = np.array([next(i for i, (_, lo, hi) in enumerate(BANDS) if lo <= f < hi) for f in mel_hz])
info = mne.create_info(channels, 128.0, "eeg"); info.set_montage(mne.channels.make_standard_montage("standard_1020"))
OUT = {}

def topo(ax, values, cmap="viridis", sym=False):
    v = np.asarray(values, float); lim = np.abs(v).max() if sym else None
    mne.viz.plot_topomap(v, info, axes=ax, show=False, contours=0, cmap=cmap,
                         vlim=(-lim, lim) if sym else (v.min(), v.max()), sensors=True, names=None)

def auc(a, b):
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))

def bh(p):
    p = np.asarray(p); o = np.argsort(p); r = np.empty_like(o); r[o] = np.arange(1, len(p) + 1)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1); return out

# ===================================================================== CP / G-NTF
f = np.load(GNTF / "interpretation" / "factors.npz", allow_pickle=True)
B, F, S, A = f["electrodes"], f["mel"], f["statistics"], f["activation"]
ids = [l.split(",")[0] for l in (GNTF / "interpretation" / "gntf_component_activations.csv").read_text().splitlines()[1:]]
pid = [p["subject_id"] for p in P]
assert ids == pid, "participant order mismatch"
R = A.shape[1]
rows = []
kw_p = []
for r in range(R):
    groups = [A[labels == k, r] for k in range(3)]
    kw_p.append(kruskal(*groups).pvalue)
q = bh(kw_p)
stab = json.loads((GNTF / "interpretation" / "stability.json").read_text())
sim = np.array([fo["matched_similarities"] for fo in stab["folds"]])
clin = json.loads((GNTF / "interpretation" / "clinical_correlations.json").read_text())["correlations"]
for r in range(R):
    top = np.argsort(B[:, r])[::-1][:3]
    row = {"component": r + 1, "peak_hz": float(mel_hz[np.argmax(F[:, r])]),
           "top_electrodes": [channels[i] for i in top],
           "stat_loading": dict(zip(["mean", "sd", "median", "iqr"], np.round(S[:, r] / S[:, r].sum(), 3).tolist())),
           "median_activation": {c: float(np.median(A[labels == k, r])) for k, c in enumerate(CLS)},
           "auc_AD_vs_HC": auc(A[labels == 0, r], A[labels == 2, r]),
           "auc_FTD_vs_HC": auc(A[labels == 1, r], A[labels == 2, r]),
           "auc_AD_vs_FTD": auc(A[labels == 0, r], A[labels == 1, r]),
           "kruskal_p": float(kw_p[r]), "kruskal_q": float(q[r]),
           "stability_matched_fraction": float((sim[:, r] > 0).mean()),
           "stability_median_similarity_when_matched": float(np.median(sim[sim[:, r] > 0, r])) if (sim[:, r] > 0).any() else None}
    for inst in ["MMSE", "MoCA"]:
        c = [x for x in clin if x["instrument"] == inst and x["group"] == "all" and x["component"] == r + 1]
        if c: row[f"{inst}_rho"] = c[0]["rho"]; row[f"{inst}_q"] = c[0]["q_fdr"]; row[f"{inst}_n"] = c[0]["n"]
    rows.append(row)
OUT["cp"] = rows
OUT["cp_stability_folds"] = int(sim.shape[0])

fig, axes = plt.subplots(4, R, figsize=(7.2, 6.4), constrained_layout=True)
for r in range(R):
    topo(axes[0, r], B[:, r]); axes[0, r].set_title(f"Component {r + 1}", fontsize=8)
    ax = axes[1, r]; ax.plot(mel_hz, F[:, r], color="#17324D", lw=1.5); ax.fill_between(mel_hz, F[:, r], alpha=.2, color="#276FBF")
    for _, lo, _hi in BANDS[1:]: ax.axvline(lo, color="gray", lw=.4, ls=":")
    ax.set_xscale("log"); ax.set_xticks([2, 4, 8, 13, 25, 40]); ax.set_xticklabels(["2", "4", "8", "13", "25", "40"], fontsize=6)
    ax.set_xlabel("Hz", fontsize=6.5); ax.tick_params(labelsize=6)
    ax = axes[2, r]; ax.bar(range(4), S[:, r], color="#40826D"); ax.set_xticks(range(4)); ax.set_xticklabels(["mean", "SD", "med", "IQR"], fontsize=6, rotation=45)
    ax.tick_params(labelsize=6)
    ax = axes[3, r]
    bp = ax.boxplot([A[labels == k, r] for k in range(3)], patch_artist=True, widths=.6, showfliers=False, medianprops=dict(color="k", lw=1))
    for patch, c in zip(bp["boxes"], COL): patch.set_facecolor(c); patch.set_alpha(.75)
    ax.set_xticks(range(1, 4)); ax.set_xticklabels(CLS, fontsize=6.5); ax.tick_params(labelsize=6)
axes[0, 0].set_ylabel("Electrode\nweight $B$", fontsize=7); axes[1, 0].set_ylabel("Frequency\nweight $F$", fontsize=7)
axes[2, 0].set_ylabel("Statistic\nweight $S$", fontsize=7); axes[3, 0].set_ylabel("Activation $A$", fontsize=7)
fig.savefig(FIG / "fig_interp_cp.pdf"); plt.close(fig)

# ===================================================================== Tensor Train
art = joblib.load(ADV / "outputs/tensor_train_svm/tt_svm_model.joblib")  # trusted local artefact
tt = art["decomposition"]; norm = art["normalization"]; clf = art["classifier"]; dev = np.asarray(art["development_subjects"])
X = np.load(ADV / "outputs/tensor_train_svm/participant_tensors.npy").astype(np.float64)
Xn = np.empty_like(X)
for s in sorted(set(sites)):
    m = sites == s
    Xn[m] = (X[m] - norm[s]["mean"]) / norm[s]["std"]
Z = tt.transform(Xn)                                     # (174, 4) TT features
basis = tt.basis_.reshape(19, 24, 4, -1)                 # electrode x mel x statistic x feature
nf = basis.shape[-1]
var_total = ((Xn[dev] - tt.mean_) ** 2).sum()
var_expl = [float((Z[dev, d] ** 2).sum() / var_total) for d in range(nf)]
e_el = (basis ** 2).sum((1, 2)); e_el /= e_el.sum(0)
e_fr = (basis ** 2).sum((0, 2)); e_fr /= e_fr.sum(0)
e_st = (basis ** 2).sum((0, 1)); e_st /= e_st.sum(0)
svm = clf.named_steps["svm"]; sc = clf.named_steps["scale"]
pairs = [("AD", "FTD"), ("AD", "HC"), ("FTD", "HC")]
W = np.einsum("ijkd,pd->ijkp", basis, svm.coef_ / sc.scale_)      # input-space weight per pair
Wsum = W.sum(2)                                                  # sum over 4 statistics -> (19, 24, 3)
tt_rows = []
for d in range(nf):
    top = np.argsort(e_el[:, d])[::-1][:3]
    kwp = kruskal(*[Z[dev][labels[dev] == k, d] for k in range(3)]).pvalue
    tt_rows.append({"feature": d + 1, "variance_explained": var_expl[d],
                    "top_electrodes": [channels[i] for i in top], "top_electrode_share": [float(e_el[i, d]) for i in top],
                    "peak_hz": float(mel_hz[np.argmax(e_fr[:, d])]),
                    "band_energy_share": {BANDS[b][0]: float(e_fr[band_of == b, d].sum()) for b in range(5)},
                    "stat_energy_share": dict(zip(["mean", "sd", "median", "iqr"], np.round(e_st[:, d], 3).tolist())),
                    "svm_coef_scaled": {f"{a}-{b}": float(svm.coef_[p, d]) for p, (a, b) in enumerate(pairs)},
                    "median_score": {c: float(np.median(Z[dev][labels[dev] == k, d])) for k, c in enumerate(CLS)},
                    "kruskal_p_dev": float(kwp)})
OUT["tt"] = {"features": tt_rows, "total_variance_explained": float(sum(var_expl)),
             "reconstruction_error": float(tt.relative_centered_reconstruction_error_)}

def region_band(Wp):
    M = np.zeros((len(REGIONS), len(BANDS)))
    for i, (_, els) in enumerate(REGIONS.items()):
        idx = [channels.index(e) for e in els]
        for b in range(len(BANDS)):
            M[i, b] = Wp[np.ix_(idx, np.flatnonzero(band_of == b))].sum()
    return M
RB = {f"{a}-{b}": region_band(Wsum[:, :, p]) for p, (a, b) in enumerate(pairs)}
OUT["tt"]["region_band_weight"] = {k: {"regions": list(REGIONS), "bands": [b[0] for b in BANDS], "values": v.round(4).tolist()} for k, v in RB.items()}
# band-only totals
OUT["tt"]["band_weight_total"] = {k: dict(zip([b[0] for b in BANDS], v.sum(0).round(4).tolist())) for k, v in RB.items()}

fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
gs = fig.add_gridspec(3, 4)
for d in range(nf):
    ax = fig.add_subplot(gs[0, d]); topo(ax, e_el[:, d]); ax.set_title(f"TT feature {d + 1}\n({100 * var_expl[d]:.1f}% var.)", fontsize=7.5)
    ax = fig.add_subplot(gs[1, d]); ax.bar(range(24), e_fr[:, d], color=["#276FBF", "#2AA9B8", "#40826D", "#D89B2B", "#C74B50"][0] if False else [["#8DA0CB", "#66C2A5", "#FC8D62", "#E78AC3", "#A6D854"][b] for b in band_of])
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels([f"{mel_hz[i]:.0f}" for i in [0, 6, 12, 18, 23]], fontsize=6); ax.tick_params(labelsize=6); ax.set_xlabel("Hz", fontsize=6.5)
    if d == 0: ax.set_ylabel("Energy share", fontsize=7)
for p, (a, b) in enumerate(pairs):
    ax = fig.add_subplot(gs[2, p]); M = RB[f"{a}-{b}"]; lim = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([x[0][:3] for x in BANDS], fontsize=6.5)
    ax.set_yticks(range(5)); ax.set_yticklabels([r[:4] for r in REGIONS], fontsize=6.5)
    ax.set_title(f"{a} vs {b}\n(red: toward {a}, blue: toward {b})", fontsize=7)
ax = fig.add_subplot(gs[2, 3])
tot = np.array([RB[f"{a}-{b}"].sum(0) for a, b in pairs])
for p, (a, b) in enumerate(pairs): ax.bar(np.arange(5) + (p - 1) * .27, tot[p], .27, label=f"{a}-{b}")
ax.axhline(0, color="k", lw=.5); ax.set_xticks(range(5)); ax.set_xticklabels([x[0][:3] for x in BANDS], fontsize=6.5); ax.tick_params(labelsize=6)
ax.legend(fontsize=5.5, frameon=False); ax.set_title("Band totals", fontsize=7)
fig.savefig(FIG / "fig_interp_tt.pdf"); plt.close(fig)

# ===================================================================== Tucker / HOSVD
tensor = np.load(tucker.PREPARED / "log_mel.npy", mmap_mode="r")
owners = np.load(tucker.PREPARED / "subject_index.npy", mmap_mode="r")
splits = json.loads(tucker.SPLITS.read_text(encoding="utf-8"))
dev_t = np.concatenate([splits["train"], splits["val"]]).astype(int)
ranks = (8, 16, 5)
mean_t, (Uc, Uf, Ut), retained = tucker.fit_hosvd(tensor, owners, dev_t, ranks)
cores = tucker.project(tensor, mean_t, (Uc, Uf, Ut))                # windows x 8 x 16 x 5
sub_core = np.stack([cores[owners == s][..., 0].mean(0) for s in range(len(P))])   # participant mean of first time mode
# per mode variance
idx = np.flatnonzero(np.isin(owners, dev_t))
cov_c = np.zeros((19, 19)); cov_f = np.zeros((24, 24))
for st in range(0, len(idx), 128):
    blk = tensor[idx[st:st + 128]].astype(np.float32) - mean_t
    cov_c += np.einsum("ncft,ndft->cd", blk, blk); cov_f += np.einsum("ncft,ncgt->fg", blk, blk)
vc = np.sort(np.linalg.eigvalsh(cov_c))[::-1]; vf = np.sort(np.linalg.eigvalsh(cov_f))[::-1]
OUT["tucker"] = {"retained": retained, "channel_mode_share": (vc[:8] / vc.sum()).round(4).tolist(),
                 "frequency_mode_share": (vf[:16] / vf.sum()).round(4).tolist()}
# class relevance of mean core coefficients (dev people only)
KW = np.ones((8, 16))
dm = np.isin(np.arange(len(P)), dev_t)
for a in range(8):
    for b in range(16):
        v = sub_core[dm, a, b]
        KW[a, b] = kruskal(*[v[labels[dm] == k] for k in range(3)]).pvalue
qs = bh(KW.ravel()).reshape(8, 16)
top = np.dstack(np.unravel_index(np.argsort(KW.ravel())[:5], KW.shape))[0]
OUT["tucker"]["n_cells_q_lt_0.05"] = int((qs < .05).sum()); OUT["tucker"]["n_cells"] = int(KW.size)
OUT["tucker"]["top_cells"] = [{"channel_mode": int(a) + 1, "frequency_mode": int(b) + 1, "p": float(KW[a, b]), "q": float(qs[a, b])} for a, b in top]
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), constrained_layout=True)
for a in range(3):
    topo(axes[0, a], Uc[:, a], cmap="RdBu_r", sym=True); axes[0, a].set_title(f"Electrode mode {a + 1}\n({100 * vc[a] / vc.sum():.0f}% var.)", fontsize=7.5)
ax = axes[0, 3]
for b in range(4): ax.plot(mel_hz, Uf[:, b], lw=1.2, label=f"mode {b + 1} ({100 * vf[b] / vf.sum():.0f}%)")
ax.set_xscale("log"); ax.set_xticks([2, 4, 8, 13, 25, 40]); ax.set_xticklabels(["2", "4", "8", "13", "25", "40"], fontsize=6); ax.set_xlabel("Hz", fontsize=6.5)
ax.legend(fontsize=5.5, frameon=False); ax.set_title("Frequency modes", fontsize=7.5); ax.tick_params(labelsize=6)
ax = axes[1, 0]; ax.bar(range(1, 9), vc[:8] / vc.sum(), color="#276FBF"); ax.set_title("Electrode-mode variance", fontsize=7.5); ax.tick_params(labelsize=6)
ax = axes[1, 1]; ax.bar(range(1, 17), vf[:16] / vf.sum(), color="#2AA9B8"); ax.set_title("Frequency-mode variance", fontsize=7.5); ax.tick_params(labelsize=6)
ax = axes[1, 2:4];
gs2 = axes[1, 2].get_gridspec(); axes[1, 2].remove(); axes[1, 3].remove()
ax = fig.add_subplot(gs2[1, 2:4]); im = ax.imshow(-np.log10(KW), cmap="magma", aspect="auto")
ax.set_xlabel("Frequency mode", fontsize=7); ax.set_ylabel("Electrode mode", fontsize=7); ax.set_title(r"Class relevance $-\log_{10}p$ (Kruskal--Wallis)", fontsize=7.5)
ax.set_xticks(range(0, 16, 3)); ax.set_xticklabels(range(1, 17, 3), fontsize=6); ax.set_yticks(range(8)); ax.set_yticklabels(range(1, 9), fontsize=6)
fig.colorbar(im, ax=ax, fraction=.04)
fig.savefig(FIG / "fig_interp_tucker.pdf"); plt.close(fig)

(Path(__file__).resolve().parent / "interp_summary.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print(json.dumps(OUT, indent=1)[:9000])
