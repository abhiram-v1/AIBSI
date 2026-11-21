"""Waveform Tucker-2 reconstruction BEFORE the STFT, then the existing spectral SVM + EEGPT blend.

Per 8 s / 128 Hz / 19-channel window X (19, 1024) a delay tensor
H[c, lag, k] = X[c, k + lag]  (19, L=32, 993) is projected with shared,
training-participant-only electrode and delay factors,

    H_rec[:, :, k] = Pc @ H[:, :, k] @ Pl.T,

and X_rec[c, t] is the average of every H_rec[c, l, k] with l + k = t.
X_rec then goes through the SAME STFT / log-mel (float16 quantisation
included) / participant summary / site-normalised SVM as the existing
baseline and is blended with the already fine-tuned EEGPT probabilities.
EEGPT is never retrained.

Stages (all cached, nothing existing is overwritten):
  prepare : rebuild the 174-participant 8 s waveform cache and audit it
  run     : unit checks, factor fitting, validation search, frozen test evaluation
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import stft
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "neural_comparison" / "stft_gpu_3class"))
sys.path.insert(0, str(HERE))
import run_pipeline as rp  # noqa: E402
import pipeline as eeg  # noqa: E402
import evaluate_site_normalized as site_svm  # noqa: E402
import evaluate_tensor_eegpt_blend as common  # noqa: E402
import search_tensor_eegpt_joint as joint  # noqa: E402

OUT = HERE / "outputs" / "waveform_tucker_before_stft_2026-11-05"
PREP = OUT / "prepared"
CACHE = OUT / "cache"
OLD_STFT = ROOT / "neural_comparison" / "stft_gpu_3class" / "outputs" / "prepared"
MOD = HERE / "outputs" / "moderate_log_mel" / "prepared"
EEGPT = common.EEGPT
CLASSES = ["AD", "FTD", "HC"]
CODE_VERSION = "waveform-tucker-v1"

FS, WS, L = rp.FS, rp.WINDOW_SAMPLES, 32
K = WS - L + 1
CHANNEL_RANKS = [8, 12, 19]
DELAY_RANKS = [8, 16, 32]
RANK_GRID = list(itertools.product(CHANNEL_RANKS, DELAY_RANKS))
IDENTITY = (19, 32)
HISTORICAL_SVM = {"pca": 20, "C": 0.1, "kernel": "linear"}
HISTORICAL_BLEND = {"svm_temperature": 0.5, "eegpt_temperature": 0.75, "svm_weight": 0.5}
SVM_CONFIGS = [{"pca": p, "C": c, "kernel": k}
               for p, c, k in itertools.product([20, 40, None], [0.03, 0.1, 0.3, 1.0, 3.0], ["linear", "rbf"])]
BLENDS = joint.BLENDS


# --------------------------------------------------------------------------- helpers
def sha(path):
    return common.digest(path)


def jdump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(*a):
    print(*a, flush=True)


def counts_vector():
    t = np.arange(WS)
    return (np.minimum(L - 1, t) - np.maximum(0, t - (K - 1)) + 1).astype(np.float64)


COUNTS = counts_vector()


# --------------------------------------------------------------------------- preparation
def prepare_participant(item):
    import mne
    mne.set_log_level("ERROR")
    record = rp.Record(item["subject_id"], item["source"], item["site"], item["label"], item["path"])
    data, header = rp.load_harmonized(record)
    start, stop = rp.EDGE_SECONDS * FS, data.shape[1] - rp.EDGE_SECONDS * FS
    accepted, rejected = [], []
    for offset in range(start, stop - WS + 1, WS):
        ok, reason = rp.valid_window(data[:, offset:offset + WS])
        (accepted if ok else rejected).append((offset, reason))
    kept = rp.evenly_limit([o for o, _ in accepted], rp.MAX_WINDOWS_PER_SUBJECT)
    windows = np.stack([data[:, o:o + WS] for o in kept]).astype(np.float32)
    sd = windows.std(axis=2)
    return dict(subject_id=item["subject_id"], windows=windows, offsets=[int(o) for o in kept],
                accepted_offsets=[int(o) for o, _ in accepted],
                rejected=[[int(o), r] for o, r in rejected], header=header, samples=int(data.shape[1]),
                median_channel_sd_volts=float(np.median(sd)))


def batch_log_stft(windows):
    """Vectorised rp.window_to_stft (same arguments, per-window median)."""
    freqs, _, spec = stft(windows, fs=FS, window="hann", nperseg=rp.STFT_SAMPLES,
                          noverlap=rp.STFT_SAMPLES - rp.STFT_HOP, nfft=rp.STFT_SAMPLES, detrend="constant",
                          return_onesided=True, boundary=None, padded=False, axis=-1)
    keep = (freqs >= 1.0) & (freqs <= 40.0)
    lp = np.log10(np.abs(spec[..., keep, :]) ** 2 + 1e-20).astype(np.float32)
    lp -= np.median(lp, axis=(1, 2, 3), keepdims=True)
    return lp


BANK = eeg.mel_filter_bank(np.linspace(1.0, 40.0, 79))


def logmel_from_waveforms(windows, batch=128):
    """Waveform -> float16 log-STFT -> float16 log-mel, exactly as the stored baseline (incl. quantisation)."""
    out = np.empty((len(windows), 19, eeg.MEL_BINS, 13), dtype=np.float16)
    for s in range(0, len(windows), batch):
        lp = batch_log_stft(np.asarray(windows[s:s + batch], dtype=np.float32)).astype(np.float16)
        power = np.power(10.0, lp.astype(np.float32))
        mel = np.einsum("mf,ncft->ncmt", BANK, power, optimize=True)
        lm = np.log10(mel + 1e-20)
        lm -= np.median(lm, axis=(1, 2, 3), keepdims=True)
        out[s:s + batch] = lm.astype(np.float16)
    return out


def stage_prepare(jobs):
    PREP.mkdir(parents=True, exist_ok=True)
    manifest = jload(MOD / "manifest.json")
    participants = manifest["participants"]
    fp_path = PREP / "fingerprint.json"
    wanted = dict(code=CODE_VERSION, ids=[p["subject_id"] for p in participants],
                  run_pipeline=sha(Path(rp.__file__)), old_manifest=sha(MOD / "manifest.json"))
    if fp_path.exists() and jload(fp_path)["inputs"] == wanted and (PREP / "waveforms.npy").exists():
        log("Reusing waveform cache", PREP)
        return
    log(f"Rebuilding waveforms for {len(participants)} participants with {jobs} workers ...")
    t0 = time.perf_counter()
    results = joblib.Parallel(n_jobs=jobs, verbose=5)(joblib.delayed(prepare_participant)(p) for p in participants)
    assert [r["subject_id"] for r in results] == wanted["ids"]
    x = np.concatenate([r["windows"] for r in results])
    owners = np.concatenate([np.full(len(r["windows"]), i, dtype=np.int32) for i, r in enumerate(results)])
    labels = np.concatenate([np.full(len(r["windows"]), eeg.LABEL_ID[p["label"]], dtype=np.int64)
                             for r, p in zip(results, participants)])
    np.save(PREP / "waveforms.npy", x)
    np.save(PREP / "subject_index.npy", owners)
    np.save(PREP / "labels.npy", labels)
    rows = []
    for i, (r, p) in enumerate(zip(results, participants)):
        rows.append(dict(subject_index=i, subject_id=p["subject_id"], label=p["label"], source=p["source"], site=p["site"],
                         path=p["path"], original_cached_windows=p["windows"], rebuilt_windows=len(r["windows"]),
                         selected_offsets=r["offsets"], accepted_offsets=r["accepted_offsets"],
                         rejected_windows=r["rejected"], header=r["header"], harmonized_samples=r["samples"],
                         median_channel_sd_volts=r["median_channel_sd_volts"]))
    jdump(PREP / "waveform_manifest.json", dict(
        units="volts (float32), as returned by run_pipeline.load_harmonized", sampling_rate=FS,
        window_samples=WS, window_seconds=rp.WINDOW_SECONDS, channels=rp.TARGET_CHANNELS, edge_trim_seconds=rp.EDGE_SECONDS,
        max_windows=rp.MAX_WINDOWS_PER_SUBJECT, shape=list(x.shape), participants=rows))
    # ---- audit against the saved 6721-window caches
    old_labels = np.load(MOD / "labels.npy")
    old_owner = np.load(MOD / "subject_index.npy")
    audit = dict(rebuild_seconds=time.perf_counter() - t0, windows_rebuilt=int(len(x)), windows_cached=int(len(old_owner)))
    audit["owners_identical"] = bool(np.array_equal(owners, old_owner))
    audit["labels_identical"] = bool(np.array_equal(labels, old_labels))
    audit["participants_with_count_mismatch"] = [r["subject_id"] for r in rows if r["original_cached_windows"] != r["rebuilt_windows"]]
    if audit["owners_identical"]:
        old_stft = np.load(OLD_STFT / "spectrograms.npy", mmap_mode="r")
        old_mel = np.load(MOD / "log_mel.npy", mmap_mode="r")
        max_stft = max_mel = 0.0
        eq_stft = eq_mel = tot_stft = tot_mel = 0
        for s in range(0, len(x), 256):
            new_stft = batch_log_stft(x[s:s + 256]).astype(np.float16)
            ref = np.asarray(old_stft[s:s + 256])
            d = np.abs(new_stft.astype(np.float32) - ref.astype(np.float32))
            max_stft = max(max_stft, float(d.max()))
            eq_stft += int((new_stft == ref).sum())
            tot_stft += ref.size
            new_mel = logmel_from_waveforms(x[s:s + 256])
            ref = np.asarray(old_mel[s:s + 256])
            d = np.abs(new_mel.astype(np.float32) - ref.astype(np.float32))
            max_mel = max(max_mel, float(d.max()))
            eq_mel += int((new_mel == ref).sum())
            tot_mel += ref.size
        audit.update(stft_max_abs_diff=max_stft, stft_fraction_bitwise_equal=eq_stft / tot_stft,
                     log_mel_max_abs_diff=max_mel, log_mel_fraction_bitwise_equal=eq_mel / tot_mel)
        # single-window equivalence of the vectorised STFT with run_pipeline.window_to_stft
        for i in (0, len(x) // 2, len(x) - 1):
            assert np.allclose(batch_log_stft(x[i:i + 1])[0], rp.window_to_stft(x[i]), atol=1e-6)
        audit["vectorised_stft_matches_window_to_stft"] = True
    jdump(PREP / "rebuild_audit.json", audit)
    jdump(fp_path, dict(inputs=wanted, waveforms_sha256=sha(PREP / "waveforms.npy"), shape=list(x.shape)))
    log(json.dumps(audit, indent=2))


# --------------------------------------------------------------------------- Tucker-2 on delay tensor
def hankel_views(x):
    """(B, 19, T) -> (B, 19, K, L) view; element [b,c,k,l] = x[b,c,k+l]."""
    return sliding_window_view(x, L, axis=-1)


def window_covariances(x):
    """Un-centred second-moment matrices of mode-1 (19x19) and mode-2 (LxL) unfoldings, summed over lag/position."""
    h = np.ascontiguousarray(hankel_views(x))  # contiguous so that matmul uses BLAS
    cc = np.zeros((x.shape[0], 19, 19))
    for l in range(L):
        xs = np.ascontiguousarray(x[:, :, l:l + K])
        cc += xs @ xs.transpose(0, 2, 1)
    cl = np.matmul(h.transpose(0, 1, 3, 2), h).sum(axis=1)
    return cc, cl


def fit_factors(windows, owners, subjects):
    """Shared electrode/delay eigenbases from the given participants only. Each participant contributes the mean
    of its window covariances scaled to unit trace, so neither long recordings nor high-amplitude sites dominate."""
    cc_total, cl_total = np.zeros((19, 19)), np.zeros((L, L))
    for s in subjects:
        idx = np.flatnonzero(owners == s)
        cc_s, cl_s = np.zeros((19, 19)), np.zeros((L, L))
        for b in range(0, len(idx), 16):
            cc, cl = window_covariances(np.asarray(windows[idx[b:b + 16]], dtype=np.float64) * 1e6)  # micro-volt scale for conditioning
            cc_s += cc.sum(axis=0)
            cl_s += cl.sum(axis=0)
        assert np.isclose(np.trace(cc_s), np.trace(cl_s), rtol=1e-8)
        cc_total += cc_s / np.trace(cc_s)
        cl_total += cl_s / np.trace(cl_s)
    wc, vc = np.linalg.eigh(cc_total)
    wl, vl = np.linalg.eigh(cl_total)
    oc, ol = np.argsort(wc)[::-1], np.argsort(wl)[::-1]
    return dict(channel_vectors=vc[:, oc], channel_eigenvalues=wc[oc], delay_vectors=vl[:, ol], delay_eigenvalues=wl[ol],
                fit_subjects=np.asarray(subjects, dtype=np.int64))


def projectors(factors, rc, rl):
    uc, ul = factors["channel_vectors"][:, :rc], factors["delay_vectors"][:, :rl]
    return uc @ uc.T, ul @ ul.T


def reconstruct(x, pc, pl):
    """Project H = hankel(x) by Pc (electrode) and Pl (delay), then average anti-diagonals back to (B, 19, T)."""
    y = np.matmul(pc, np.asarray(x, dtype=np.float64))
    out = np.empty(y.shape)
    for s in range(0, len(y), 8):
        yb = y[s:s + 8]
        z = np.ascontiguousarray(hankel_views(yb)) @ pl.T  # [b,c,k,l] = sum_l' Pl[l,l'] y[b,c,k+l']
        acc = np.zeros(yb.shape)
        for l in range(L):
            acc[:, :, l:l + K] += z[:, :, :, l]
        out[s:s + 8] = acc / COUNTS
    return out


def reference_reconstruct(x, pc, pl):
    """Slow, explicit implementation used only in unit checks."""
    c, n = x.shape
    h = np.empty((c, L, n - L + 1))
    for lag in range(L):
        for k in range(n - L + 1):
            h[:, lag, k] = x[:, k + lag]
    hr = np.stack([pc @ h[:, :, k] @ pl.T for k in range(n - L + 1)], axis=2)
    out, cnt = np.zeros((c, n)), np.zeros(n)
    for lag in range(L):
        for k in range(n - L + 1):
            out[:, lag + k] += hr[:, lag, k]
            cnt[lag + k] += 1
    return out / cnt, cnt


def unit_checks(factors, windows):
    rng = np.random.default_rng(0)
    checks = {}
    x = rng.standard_normal((19, WS))
    a = rng.standard_normal((19, 19))
    b = rng.standard_normal((L, L))
    pc, _ = np.linalg.qr(a)
    pc = pc[:, :7] @ pc[:, :7].T
    pl, _ = np.linalg.qr(b)
    pl = pl[:, :11] @ pl[:, :11].T
    ref, cnt = reference_reconstruct(x, pc, pl)
    fast = reconstruct(x[None], pc, pl)[0]
    checks["fast_matches_explicit_hankel_diagonal_average_max_abs"] = float(np.abs(ref - fast).max())
    checks["edge_counts_match_explicit"] = bool(np.array_equal(cnt, COUNTS))
    checks["edge_counts_first_last"] = [float(COUNTS[0]), float(COUNTS[-1]), float(COUNTS[L - 1]), float(COUNTS.max())]
    const = np.full((1, 19, WS), 3.7)
    checks["constant_signal_identity_projectors_max_abs"] = float(np.abs(reconstruct(const, np.eye(19), np.eye(L)) - const).max())
    checks["output_samples_unchanged"] = list(fast.shape) == [19, WS]
    xs = np.asarray(windows[:32], dtype=np.float64)
    pc_full, pl_full = projectors(factors, 19, L)
    rec = reconstruct(xs, pc_full, pl_full)
    checks["full_rank_eigenbasis_identity_max_abs_volts"] = float(np.abs(rec - xs).max())
    checks["full_rank_eigenbasis_identity_max_rel"] = float(np.abs(rec - xs).max() / np.abs(xs).max())
    checks["full_rank_float32_bitwise_equal_fraction"] = float((rec.astype(np.float32) == xs.astype(np.float32)).mean())
    checks["orthonormal_channel_basis_error"] = float(np.abs(factors["channel_vectors"].T @ factors["channel_vectors"] - np.eye(19)).max())
    checks["orthonormal_delay_basis_error"] = float(np.abs(factors["delay_vectors"].T @ factors["delay_vectors"] - np.eye(L)).max())
    assert checks["fast_matches_explicit_hankel_diagonal_average_max_abs"] < 1e-9
    assert checks["edge_counts_match_explicit"] and checks["constant_signal_identity_projectors_max_abs"] < 1e-12
    assert checks["full_rank_eigenbasis_identity_max_rel"] < 1e-9
    return checks


# --------------------------------------------------------------------------- arms (features)
def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def arm_features(ctx, factors, ranks, tag, chunk=256):
    """Reconstruct every window with frozen factors and return participant features (174, 19, 48) + retention stats."""
    windows, owners, labels_w, participants = ctx["windows"], ctx["owners"], ctx["labels_w"], ctx["participants"]
    key = f"{tag}_{'raw' if ranks is None else '%d_%d' % ranks}"
    fp = dict(code=CODE_VERSION, wave=ctx["wave_fp"], ranks=ranks, fit=array_digest(factors["fit_subjects"]) if factors is not None else None)
    cache = CACHE / f"features_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if json.loads(str(z["fingerprint"])) == json.loads(json.dumps(fp)):
            return z["features"], dict(energy_ratio=z["energy_ratio"], residual_ratio=z["residual_ratio"])
    tensor = np.empty((len(windows), 19, eeg.MEL_BINS, 13), dtype=np.float16)
    e_in = np.zeros(len(windows))
    e_out = np.zeros(len(windows))
    e_res = np.zeros(len(windows))
    pc = pl = None
    if ranks is not None:
        pc, pl = projectors(factors, *ranks)
    for s in range(0, len(windows), chunk):
        block = np.asarray(windows[s:s + chunk])
        if ranks is None:
            rec = block
        else:
            rec64 = reconstruct(block, pc, pl)
            rec = rec64.astype(np.float32)
            x64 = block.astype(np.float64)
            e_in[s:s + chunk] = (x64 ** 2).sum(axis=(1, 2))
            e_out[s:s + chunk] = (rec64 ** 2).sum(axis=(1, 2))
            e_res[s:s + chunk] = ((x64 - rec64) ** 2).sum(axis=(1, 2))
        tensor[s:s + chunk] = logmel_from_waveforms(rec)
    features, _ = eeg.subject_features(tensor, labels_w, owners, participants)
    per_p_energy = np.array([e_out[owners == i].sum() / max(e_in[owners == i].sum(), 1e-30) for i in range(len(participants))])
    per_p_res = np.array([e_res[owners == i].sum() / max(e_in[owners == i].sum(), 1e-30) for i in range(len(participants))])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=features, energy_ratio=per_p_energy, residual_ratio=per_p_res,
                        fingerprint=json.dumps(fp))
    return features, dict(energy_ratio=per_p_energy, residual_ratio=per_p_res)


def flat(features):
    return features.reshape(len(features), -1)


# --------------------------------------------------------------------------- downstream
def site_stats(features, participants, fit):
    sites = np.asarray([p["site"] for p in participants])
    stats = {}
    for site in sorted(set(sites)):
        ref = features[fit[sites[fit] == site]]
        stats[site] = (ref.mean(axis=0, keepdims=True), np.maximum(ref.std(axis=0, keepdims=True), 1e-3))
    return stats


def apply_site(features, participants, stats):
    sites = np.asarray([p["site"] for p in participants])
    out = np.empty_like(features, dtype=np.float32)
    for site, (mean, std) in stats.items():
        out[sites == site] = (features[sites == site] - mean) / std
    return out


def fit_scores(features, labels, participants, fit, predict, config):
    """Decision scores of the historical site-normalised SVM pipeline, refitted on `fit` only."""
    normalized = site_svm.site_normalize(features, participants, fit)
    model = site_svm.direct_model(config)
    model.fit(normalized[fit], labels[fit])
    assert np.array_equal(model.classes_, np.arange(3))
    return model.decision_function(normalized[predict]), model.predict(normalized[predict]), model


def blend_grid_scores(scores, eegpt_p, true):
    svm_p = np.stack([common.probability(scores, t) for t in joint.SVMT])
    ep = np.stack([common.eegpt_probability(eegpt_p, t) for t in joint.EEGPTT])
    w = joint.WEIGHTS
    probs = svm_p[:, None, None] * w[None, None, :, None, None] + ep[None, :, None] * (1 - w)[None, None, :, None, None]
    pred = probs.reshape(-1, len(true), 3).argmax(axis=2)
    return joint.vector_scores(pred, true)


def rank_key(arm):
    return 0 if arm is None else 0 + arm[0] + arm[1]


def selection_key(f1, ba, acc, inner, arm, svm, blend):
    return (float(f1), float(ba), float(acc), float(inner), -abs(blend["svm_weight"] - 0.5),
            rank_key(arm) if arm is not None else IDENTITY[0] + IDENTITY[1],  # ties favour the least-compressed arm
            svm["kernel"] == "linear", -svm["C"], -(svm["pca"] if svm["pca"] else 999))


def metrics3(y, pred):
    return eeg.metrics(y, pred)


# --------------------------------------------------------------------------- main run
def load_context():
    wmani = jload(PREP / "waveform_manifest.json")
    old = jload(MOD / "manifest.json")
    participants = old["participants"]
    assert [p["subject_id"] for p in wmani["participants"]] == [p["subject_id"] for p in participants]
    raw_p = jload(HERE / "outputs/eegpt/prepared/manifest.json")["participants"]
    assert [p["subject_id"] for p in raw_p] == [p["subject_id"] for p in participants]
    splits = eeg.make_subject_splits(participants)
    saved_split = jload(HERE / "outputs/tensor_eegpt_blend_2026-11-05/splits.json")
    for k in ("train", "val", "test"):
        assert splits[k] == [r["index"] for r in saved_split[k]], f"split {k} differs from saved split"
        assert [participants[i]["subject_id"] for i in splits[k]] == [r["id"] for r in saved_split[k]]
    assert tuple(len(splits[k]) for k in ("train", "val", "test")) == (121, 26, 27)
    sets = {k: set(v) for k, v in splits.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    windows = np.load(PREP / "waveforms.npy", mmap_mode="r")
    owners = np.load(PREP / "subject_index.npy")
    labels_w = np.load(PREP / "labels.npy")
    assert np.array_equal(owners, np.load(MOD / "subject_index.npy")) and np.array_equal(labels_w, np.load(MOD / "labels.npy"))
    assert np.isfinite(np.asarray(windows[:64])).all()
    # EEGPT alignment: subject lists, labels, finite non-negative probabilities
    attention = jload(EEGPT / "attention.json")
    eegpt = np.load(EEGPT / "probabilities.npz")
    for name, key in (("val", "validation"), ("test", "test")):
        assert [r["subject"] for r in attention[key]] == splits[name], "EEGPT subject order differs"
        p = eegpt[name]
        assert p.shape == (len(splits[name]), 3) and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-3)
    labels = np.array([eeg.LABEL_ID[p["label"]] for p in participants])
    assert np.array_equal(eegpt["val_true"], labels[splits["val"]]) and np.array_equal(eegpt["test_true"], labels[splits["test"]])
    return dict(windows=windows, owners=owners, labels_w=labels_w, participants=participants, splits=splits, labels=labels,
                eegpt={"val": eegpt["val"], "test": eegpt["test"]}, wave_fp=jload(PREP / "fingerprint.json")["waveforms_sha256"][:16])


def write_protocol(ctx):
    protocol = dict(
        task="AD / FTD / HC participant-level classification; class order 0=AD 1=FTD 2=HC (HC = CN)",
        written_before_test_evaluation=True, code_version=CODE_VERSION,
        pipeline="harmonised 8 s/128 Hz/19-ch windows -> delay tensor (19,32,993) -> training-only shared Tucker-2 (electrode, delay) "
                 "-> anti-diagonal averaging -> same STFT/log-mel/float16/summary/site-normalised SVM -> blend with unchanged fine-tuned EEGPT",
        cohort=dict(participants=174, AD=71, FTD=42, HC=61, windows=int(len(ctx["owners"]))),
        units="volts float32 in the waveform cache; covariances fitted on micro-volt-scaled copies (scale-equivariant)",
        window=dict(seconds=8, fs=FS, samples=WS, edge_trim_seconds=rp.EDGE_SECONDS, max_windows=rp.MAX_WINDOWS_PER_SUBJECT, qc="run_pipeline.valid_window"),
        stft=dict(window="hann", nperseg=256, hop=64, nfft=256, detrend="constant", boundary=None, padded=False, band_hz=[1, 40], bins=79, frames=13,
                  power="log10(|S|^2+1e-20) minus window-wide median, stored float16"),
        log_mel=dict(bands=eeg.MEL_BINS, source="10**float16 log-STFT -> pipeline.mel_filter_bank -> log10 -> window median -> float16"),
        participant_features="pipeline.subject_features: mean+SD over windows and STFT frames = 19*24*2 = 912",
        tucker=dict(delay_length=L, positions=K, modes_decomposed=["electrode", "delay"], position_mode="left intact (never factorised)",
                    signed=True, centring="none (un-centred second moment)",
                    factor_fit="eigenvectors of participant-balanced summed mode-1 (19x19) and mode-2 (LxL) covariances; each participant contributes the mean of its window covariances scaled to unit trace",
                    fit_membership="train participants for validation search and inner folds; train+val (147) for the final refit",
                    channel_ranks=CHANNEL_RANKS, delay_ranks=DELAY_RANKS, rank_grid=RANK_GRID, identity_control=list(IDENTITY),
                    reconstruction="diagonal average with exact edge counts", pilot_grid_frozen=True),
        downstream=dict(site_normalisation="training-fold fitted per site", scaler="StandardScaler", pca=[20, 40, None], C=[0.03, 0.1, 0.3, 1.0, 3.0],
                        kernel=["linear", "rbf"], gamma="scale", class_weight="balanced", historical_control=HISTORICAL_SVM,
                        n_svm_configs=len(SVM_CONFIGS), same_budget_for_no_decomposition_arm=True),
        blend=dict(formula="w*softmax(score/T_svm)+(1-w)*softmax(log(max(p_eegpt,1e-7))/T_eegpt)", svm_temperatures=joint.SVMT,
                   eegpt_temperatures=joint.EEGPTT, svm_weights=joint.WEIGHTS.tolist(), n_blends=len(BLENDS),
                   historical_fixed_blend=HISTORICAL_BLEND, eegpt_retrained=False),
        splits=dict(train=ctx["splits"]["train"], val=ctx["splits"]["val"], test=ctx["splits"]["test"],
                    ids={k: [ctx["participants"][i]["subject_id"] for i in v] for k, v in ctx["splits"].items()}),
        inner_cv="5-fold source/diagnosis-stratified CV on the 121 training participants; Tucker factors, site normalisation, scaler, PCA and SVM refitted inside every fold. Standalone SVM F1 only (EEGPT has no outputs for training participants); used as tie-break.",
        selection=dict(target="validation ensemble macro F1", tie_break=["balanced accuracy", "accuracy", "inner-CV standalone SVM macro F1",
                                                                          "svm weight closest to 0.5", "less compression (larger ranks)", "linear kernel", "smaller C", "smaller PCA"]),
        procedures_evaluated_once_on_test=dict(
            P2_primary_joint_search="ranks x 30 SVM configs x 420 blends, validation ensemble F1",
            P1_fixed_downstream="ranks x historical SVM x 420 blends, validation ensemble F1",
            C1_no_decomposition_fixed="rebuilt waveforms, historical SVM, blend chosen from 420 on validation",
            C2_no_decomposition_matched_search="rebuilt waveforms x 30 SVM configs x 420 blends",
            C3_identity_full_rank_fixed="full-rank (19,32) reconstruction, historical SVM, 420 blends",
            C4_identity_full_rank_matched_search="full-rank reconstruction x 30 SVM x 420 blends"),
        final_refit="Tucker factors, site stats, scaler/PCA/SVM refit on 147 development participants after settings are frozen; test used once per procedure",
        test_set_note="27-person test set (11 AD, 6 FTD, 10 HC) was inspected in many earlier experiments: exploratory, not an untouched external estimate",
        caveat="Validation maxima on 26 participants are selection scores, not expected performance.")
    jdump(OUT / "protocol.json", protocol)
    return protocol


def hash_raw_sources():
    manifest = jload(PREP / "waveform_manifest.json")
    hashes = {}
    for p in manifest["participants"]:
        path = Path(p["path"])
        for f in (path, path.with_suffix(".fdt")):
            if f.exists():
                hashes[str(f)] = sha(f)
    jdump(PREP / "raw_source_hashes.json", hashes)
    log(f"Hashed {len(hashes)} raw files")


def arm_name(arm):
    return "raw" if arm is None else f"c{arm[0]}_d{arm[1]}"


def evaluate_procedure(F, ctx, fit, predict_idx, svm, blend, split):
    scores, pred, model = fit_scores(F, ctx["labels"], ctx["participants"], fit, predict_idx, svm)
    p = common.combined(scores, ctx["eegpt"][split], blend)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)
    return dict(scores=scores, standalone_pred=pred, probabilities=p, model=model)


def person_rows(ctx, idx, p, scores=None):
    return [dict(subject_id=ctx["participants"][i]["subject_id"], subject_index=int(i), actual=CLASSES[ctx["labels"][i]],
                 predicted=CLASSES[int(np.argmax(row))], probabilities={c: float(row[j]) for j, c in enumerate(CLASSES)})
            for i, row in zip(idx, p)]


def run(args):
    t0 = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    if (OUT / "metrics.json").exists():
        raise RuntimeError("Completed experiment already exists in this directory; preserve it or choose another output.")
    ctx = load_context()
    windows, owners, labels, participants = ctx["windows"], ctx["owners"], ctx["labels"], ctx["participants"]
    train, val, test = (np.asarray(ctx["splits"][k], dtype=int) for k in ("train", "val", "test"))
    dev = np.concatenate([train, val])
    protocol = write_protocol(ctx)
    source_paths = [Path(__file__), PREP / "waveforms.npy", PREP / "waveform_manifest.json", MOD / "log_mel.npy", MOD / "manifest.json",
                    MOD / "labels.npy", MOD / "subject_index.npy", EEGPT / "probabilities.npz", EEGPT / "attention.json",
                    EEGPT / "best_model.pt", EEGPT / "svm_blend_metrics.json", HERE / "outputs/results/site_normalized_metrics.json",
                    Path(rp.__file__), Path(eeg.__file__), Path(site_svm.__file__), Path(common.__file__), Path(joint.__file__)]
    hashes_before = {str(p.resolve()): sha(p) for p in source_paths}
    log("Inputs aligned (IDs, labels, EEGPT order). Protocol written before any evaluation.")

    # ---------------- 1. factors + unit checks (train participants only)
    factors_train = fit_factors(windows, owners, train)
    checks = unit_checks(factors_train, windows)
    log("Unit checks:", json.dumps(checks, indent=1))
    np.savez_compressed(OUT / "factors_train121.npz", **factors_train)

    # ---------------- 2. baseline reproduction using the SAVED log-mel cache (historical result)
    stored = np.load(MOD / "log_mel.npy", mmap_mode="r")
    stored_feat, _ = eeg.subject_features(stored, ctx["labels_w"], owners, participants)
    selected_hist = jload(HERE / "outputs/results/site_normalized_metrics.json")["selected_on_validation"]["config"]
    assert selected_hist == HISTORICAL_SVM
    old_blend = jload(EEGPT / "svm_blend_metrics.json")
    sc_hist, pr_hist, _ = fit_scores(flat(stored_feat), labels, participants, dev, test, HISTORICAL_SVM)
    hist_svm_only = metrics3(labels[test], pr_hist)
    hist_blend_p = common.combined(sc_hist, ctx["eegpt"]["test"], old_blend["selection"])
    hist_blend = metrics3(labels[test], hist_blend_p.argmax(1))
    for m in ("accuracy", "macro_f1", "balanced_accuracy"):
        assert np.isclose(hist_blend[m], old_blend["test"][m]), "historical direct SVM + EEGPT did not reproduce"
    log(f"Historical stored-cache baseline reproduced: SVM+EEGPT acc={hist_blend['accuracy']:.4f} F1={hist_blend['macro_f1']:.4f}; SVM alone F1={hist_svm_only['macro_f1']:.4f}")

    # ---------------- 3. arms for the validation phase (factors fitted on 121 training participants)
    ARMS = [None] + RANK_GRID
    feats = {}
    retention = {}
    for arm in ARMS:
        f, ret = arm_features(ctx, None if arm is None else factors_train, arm, "train")
        feats[arm] = flat(f)
        retention[arm_name(arm)] = ret
        log(f"features {arm_name(arm)} done ({time.perf_counter()-t0:.0f}s)")
    feat_diff = float(np.abs(feats[None] - flat(stored_feat)).max())
    identity_diff = float(np.abs(feats[IDENTITY] - flat(stored_feat)).max())
    log(f"Rebuilt raw vs stored participant features: max abs diff {feat_diff:.5f}; full-rank identity vs stored: {identity_diff:.5f}")

    # ---------------- 4. training-only inner CV (everything refitted per fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=eeg.SEED)
    strata = np.asarray([f"{participants[i]['source']}:{labels[i]}" for i in train])
    oof = {arm: np.empty((len(SVM_CONFIGS), len(train)), dtype=np.int64) for arm in ARMS}
    folds = []
    for fi, (fit_l, valid_l) in enumerate(cv.split(train, strata)):
        fit_f, valid_f = train[fit_l], train[valid_l]
        factors_f = fit_factors(windows, owners, fit_f)
        for arm in ARMS:
            F = feats[None] if arm is None else flat(arm_features(ctx, factors_f, arm, f"fold{fi}")[0])
            for si, svm in enumerate(SVM_CONFIGS):
                _, pred, _ = fit_scores(F, labels, participants, fit_f, valid_f, svm)
                oof[arm][si, valid_l] = pred
        folds.append(dict(fold=fi, fit=fit_f.tolist(), validation=valid_f.tolist()))
        log(f"inner CV fold {fi+1}/5 done ({time.perf_counter()-t0:.0f}s)")
    inner = {arm: joint.vector_scores(oof[arm], labels[train]) for arm in ARMS}
    jdump(OUT / "inner_cv.json", dict(folds=folds, candidates=[
        dict(arm=arm_name(arm), svm=SVM_CONFIGS[si], inner_macro_f1=float(inner[arm][0][si]),
             inner_balanced_accuracy=float(inner[arm][1][si]), inner_accuracy=float(inner[arm][2][si]))
        for arm in ARMS for si in range(len(SVM_CONFIGS))]))
    np.savez_compressed(OUT / "inner_oof.npz", subjects=train, true=labels[train],
                        **{f"pred_{arm_name(arm)}": oof[arm] for arm in ARMS})

    # ---------------- 5. validation search (train-fitted everything); every candidate persisted
    groups = ["P2", "P1", "C1", "C2", "C3", "C4"]
    best = {g: None for g in groups}
    parameter_best = {}
    total = 0
    svm_table = []
    hist_index = SVM_CONFIGS.index(HISTORICAL_SVM)
    with gzip.open(OUT / "all_validation_candidates.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as sink:
        for arm in ARMS:
            for si, svm in enumerate(SVM_CONFIGS):
                scores, pred, _ = fit_scores(feats[arm], labels, participants, train, val, svm)
                f1, ba, acc = blend_grid_scores(scores, ctx["eegpt"]["val"], labels[val])
                stand = metrics3(labels[val], pred)
                local = None
                for bi, blend in enumerate(BLENDS):
                    row = dict(arm=arm_name(arm), channel_rank=None if arm is None else arm[0], delay_rank=None if arm is None else arm[1],
                               svm_index=si, **{f"svm_{k}": v for k, v in svm.items()}, blend_index=bi, **blend,
                               val_macro_f1=float(f1[bi]), val_balanced_accuracy=float(ba[bi]), val_accuracy=float(acc[bi]),
                               inner_svm_macro_f1=float(inner[arm][0][si]))
                    sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                    total += 1
                    key = selection_key(f1[bi], ba[bi], acc[bi], inner[arm][0][si], arm, svm, blend)
                    targets = []
                    if arm is None:
                        targets += ["C2"] + (["C1"] if si == hist_index else [])
                    else:
                        targets += ["P2"] + (["P1"] if si == hist_index else [])
                        if arm == IDENTITY:
                            targets += ["C4"] + (["C3"] if si == hist_index else [])
                    for g in targets:
                        if best[g] is None or key > best[g][0]:
                            best[g] = (key, dict(row=row, arm=arm, svm=svm, blend=blend))
                    if arm is not None:
                        for param, value in (("channel_rank", arm[0]), ("delay_rank", arm[1]), ("pca", svm["pca"]), ("C", svm["C"]),
                                             ("kernel", svm["kernel"]), ("svm_temperature", blend["svm_temperature"]),
                                             ("eegpt_temperature", blend["eegpt_temperature"]), ("svm_weight", blend["svm_weight"])):
                            g = (param, json.dumps(value))
                            if g not in parameter_best or key > parameter_best[g][0]:
                                parameter_best[g] = (key, row)
                    if local is None or key > local[0]:
                        local = (key, row)
                svm_table.append(dict(arm=arm_name(arm), svm=svm, inner_macro_f1=float(inner[arm][0][si]),
                                      standalone_validation=stand, best_validation_blend=local[1]))
            log(f"validation search arm {arm_name(arm)} complete; {total:,} candidates ({time.perf_counter()-t0:.0f}s)")
    expected = len(ARMS) * len(SVM_CONFIGS) * len(BLENDS)
    assert total == expected
    jdump(OUT / "svm_search.json", svm_table)
    jdump(OUT / "parameter_best_results.json", [dict(parameter=p, value=json.loads(v), winner=r[1]) for (p, v), r in sorted(parameter_best.items())])
    selection = {g: dict(arm=None if best[g][1]["arm"] is None else list(best[g][1]["arm"]), svm=best[g][1]["svm"],
                         blend=best[g][1]["blend"], validation=best[g][1]["row"]) for g in groups}
    jdump(OUT / "selection.json", selection)  # frozen BEFORE test predictions
    log("Selections frozen on validation:")
    for g in groups:
        r = best[g][1]["row"]
        log(f"  {g}: arm={r['arm']} svm={best[g][1]['svm']} blend={best[g][1]['blend']} valF1={r['val_macro_f1']:.4f} valAcc={r['val_accuracy']:.4f}")

    # ---------------- 6. final refit on 147 development participants; test evaluated once per declared procedure
    factors_dev = fit_factors(windows, owners, dev)
    np.savez_compressed(OUT / "factors_dev147.npz", **factors_dev)
    dev_feats = {None: feats[None]}
    results, model_store, probabilities_json = {}, {}, {}
    for g in groups:
        arm = best[g][1]["arm"]
        if arm not in dev_feats:
            f, ret = arm_features(ctx, factors_dev, arm, "dev")
            dev_feats[arm] = flat(f)
            retention[arm_name(arm) + "_dev147"] = ret
        out = evaluate_procedure(dev_feats[arm], ctx, dev, test, best[g][1]["svm"], best[g][1]["blend"], "test")
        results[g] = dict(selection=selection[g], test=metrics3(labels[test], out["probabilities"].argmax(1)),
                          test_svm_alone=metrics3(labels[test], out["standalone_pred"]))
        model_store[g] = out
        probabilities_json[g] = person_rows(ctx, test, out["probabilities"])
        val_out = evaluate_procedure(feats[arm], ctx, train, val, best[g][1]["svm"], best[g][1]["blend"], "val")
        results[g]["validation"] = metrics3(labels[val], val_out["probabilities"].argmax(1))
        probabilities_json[g + "_validation"] = person_rows(ctx, val, val_out["probabilities"])
    results["EEGPT_alone"] = dict(test=metrics3(labels[test], ctx["eegpt"]["test"].argmax(1)),
                                  validation=metrics3(labels[val], ctx["eegpt"]["val"].argmax(1)))
    results["Historical_stored_cache_SVM"] = dict(test=hist_svm_only)
    results["Historical_stored_cache_SVM_plus_EEGPT"] = dict(test=hist_blend)

    # ---------------- 7. reloadable artifact for the primary procedure (P2) and P1; reload check
    artifacts = {}
    verification = dict(unit_checks=checks, eegpt_retrained=False, test_not_used_for_selection=True,
                        participant_ids_labels_and_eegpt_order_aligned=True, split_matches_saved_split=True)
    for g in ("P2", "P1"):
        arm = best[g][1]["arm"]
        F = dev_feats[arm]
        stats = site_stats(F, participants, dev)
        assert np.allclose(apply_site(F, participants, stats), site_svm.site_normalize(F, participants, dev), atol=1e-5)
        pc, pl = (None, None) if arm is None else projectors(factors_dev, *arm)
        artifact = dict(procedure=g, arm=arm, delay_length=L, projector_channel=pc, projector_delay=pl,
                        channel_vectors=factors_dev["channel_vectors"], delay_vectors=factors_dev["delay_vectors"],
                        channel_eigenvalues=factors_dev["channel_eigenvalues"], delay_eigenvalues=factors_dev["delay_eigenvalues"],
                        site_statistics=stats, svm_config=best[g][1]["svm"], pipeline=model_store[g]["model"],
                        blend=best[g][1]["blend"], classes=CLASSES, fit_participants=dev, subject_ids=[p["subject_id"] for p in participants],
                        stft=protocol["stft"], log_mel=protocol["log_mel"], units="volts")
        joblib.dump(artifact, OUT / f"model_{g}.joblib")
        reloaded = joblib.load(OUT / f"model_{g}.joblib")
        # recompute test probabilities from the saved waveforms with only the saved objects
        recomputed = []
        for i in test:
            w = np.asarray(windows[owners == i])
            rec = w if reloaded["projector_channel"] is None else reconstruct(w, reloaded["projector_channel"], reloaded["projector_delay"]).astype(np.float32)
            lm = logmel_from_waveforms(rec).astype(np.float32).transpose(0, 3, 1, 2).reshape(-1, 19, eeg.MEL_BINS)
            recomputed.append(np.concatenate([lm.mean(0), lm.std(0)], axis=1).reshape(-1))
        recomputed = np.asarray(recomputed, dtype=np.float32)
        full = np.zeros((len(participants), recomputed.shape[1]), dtype=np.float32)
        full[test] = recomputed
        norm = apply_site(full, participants, reloaded["site_statistics"])
        scr = reloaded["pipeline"].decision_function(norm[test])
        p_re = common.combined(scr, ctx["eegpt"]["test"], reloaded["blend"])
        verification[f"reload_{g}_max_abs_probability_difference"] = float(np.abs(p_re - model_store[g]["probabilities"]).max())
        verification[f"reload_{g}_prediction_identical"] = bool(np.array_equal(p_re.argmax(1), model_store[g]["probabilities"].argmax(1)))
        verification[f"reload_{g}_metrics_identical"] = metrics3(labels[test], p_re.argmax(1)) == results[g]["test"]
        assert verification[f"reload_{g}_max_abs_probability_difference"] < 1e-4, "saved model failed to reproduce probabilities"
    np.savez_compressed(OUT / "probabilities.npz", classes=np.asarray(CLASSES), test_subjects=test, val_subjects=val,
                        test_true=labels[test], val_true=labels[val], eegpt_test=ctx["eegpt"]["test"], eegpt_val=ctx["eegpt"]["val"],
                        **{f"test_{g}": model_store[g]["probabilities"] for g in groups},
                        **{f"test_svm_scores_{g}": model_store[g]["scores"] for g in groups})
    jdump(OUT / "participant_probabilities.json", probabilities_json)

    # ---------------- 8. final verification, hashes, metrics
    hashes_after = {str(p.resolve()): sha(p) for p in source_paths}
    verification["source_hashes_unchanged_during_run"] = hashes_before == hashes_after
    verification["all_validation_candidates_saved"] = total
    verification["expected_validation_candidates"] = expected
    verification["rebuilt_raw_vs_stored_participant_feature_max_abs_diff"] = feat_diff
    verification["full_rank_identity_vs_stored_participant_feature_max_abs_diff"] = identity_diff
    verification["probabilities_finite_and_normalised"] = True
    verification["inner_cv_refit_per_fold"] = "Tucker factors, site stats, scaler, PCA and SVM refitted in each of 5 folds"
    jdump(OUT / "source_hashes.json", hashes_before)
    jdump(OUT / "verification.json", verification)
    jdump(OUT / "retention.json", {k: dict(energy_ratio_mean_train=float(v["energy_ratio"][train].mean()),
                                           energy_ratio_mean_val=float(v["energy_ratio"][val].mean()),
                                           energy_ratio_mean_test=float(v["energy_ratio"][test].mean()),
                                           residual_ratio_mean_all=float(v["residual_ratio"].mean()))
                                   for k, v in retention.items() if k != "raw"})
    jdump(OUT / "metrics.json", dict(protocol=protocol, selection=selection, results=results, verification=verification,
                                     runtime_seconds=time.perf_counter() - t0))
    np.savez_compressed(OUT / 'retention_per_participant.npz', **{f'{k}__{m}': v[m] for k, v in retention.items() if k != 'raw' for m in ('energy_ratio', 'residual_ratio')})
    log("\nTEST RESULTS (settings frozen on validation before scoring):")
    for name, r in results.items():
        t = r["test"]
        log(f"  {name:42s} acc={100*t['accuracy']:.2f} F1={100*t['macro_f1']:.2f} BA={100*t['balanced_accuracy']:.2f}")
    for g in groups:
        log(f"\nPer-participant test probabilities, procedure {g}:")
        for r in probabilities_json[g]:
            log(" ", r["subject_id"], "actual=" + r["actual"], " ".join(f"{c}={100*v:.2f}%" for c, v in r["probabilities"].items()))
    log(f"Completed in {time.perf_counter()-t0:.0f}s; outputs in {OUT}")


PROC_LABEL = {
    "C1": "No decomposition\nfixed SVM", "C2": "No decomposition\nmatched search",
    "C3": "Full-rank identity\nfixed SVM", "C4": "Full-rank identity\nmatched search",
    "P1": "Tucker before STFT\nfixed SVM", "P2": "Tucker before STFT\njoint search (primary)",
}
PROC_ORDER = ["C1", "C2", "C3", "C4", "P1", "P2"]


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = jload(OUT / "metrics.json")
    res, sel = m["results"], m["selection"]
    ctx = load_context()
    labels = ctx["labels"]
    val, test = np.asarray(ctx["splits"]["val"]), np.asarray(ctx["splits"]["test"])
    ft = np.load(OUT / "factors_train121.npz")
    retention = jload(OUT / "retention.json")
    pct = lambda v: f"{100 * v:.2f}"  # noqa: E731

    # ---- 1. waveform reconstruction (first validation participant, first window, Cz)
    idx = np.flatnonzero(ctx["owners"] == val[0])[0]
    x = np.asarray(ctx["windows"][idx:idx + 1], dtype=np.float64)
    arm_p2 = tuple(sel["P2"]["arm"])
    arms = [arm_p2, (8, 8) if arm_p2 != (8, 8) else (8, 16)]
    t = np.arange(WS) / FS
    ch = rp.TARGET_CHANNELS.index("Cz")
    fig, axes = plt.subplots(3, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
    recs = []
    for col, arm in enumerate(arms):
        fac = {k: ft[k] for k in ("channel_vectors", "delay_vectors")}
        pc, pl = projectors(fac, *arm)
        rec = reconstruct(x, pc, pl)
        recs.append(rec)
        axes[0, col].plot(t, x[0, ch] * 1e6, color="#1f4e79", lw=0.8)
        axes[0, col].set_title(f"Electrode Cz, electrode rank {arm[0]}, delay rank {arm[1]}")
        axes[1, col].plot(t, rec[0, ch] * 1e6, color="#c0504d", lw=0.8)
        axes[2, col].plot(t, (x - rec)[0, ch] * 1e6, color="#555555", lw=0.8)
        for r, name in enumerate(["Original (µV)", "Reconstructed (µV)", "Residual = removed part (µV)"]):
            axes[r, col].set_ylabel(name)
            axes[r, col].grid(alpha=0.2)
        axes[2, col].set_xlabel("Time within the 8 s window (s)")
    fig.suptitle(f"Waveform before and after training-only Tucker-2 reconstruction ({ctx['participants'][val[0]]['subject_id']}, window 1)")
    fig.savefig(OUT / "waveform_reconstruction.png", dpi=170)
    plt.close(fig)

    # ---- 2. spectrogram before/after
    before = batch_log_stft(x.astype(np.float32))[0, ch]
    after = batch_log_stft(recs[0].astype(np.float32))[0, ch]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    ext = [0, 6, 1, 40]
    vmax = max(abs(before).max(), abs(after).max())
    for ax, img, title in zip(axes, (before, after, after - before), ("Original", f"Reconstructed (rank {arm_p2[0]}x{arm_p2[1]})", "Difference (reconstructed - original)")):
        im = ax.imshow(img, origin="lower", aspect="auto", extent=ext, cmap="magma" if "Diff" not in title else "coolwarm",
                       vmin=None if "Diff" in title else -vmax, vmax=None if "Diff" in title else vmax)
        ax.set(title=title, xlabel="Frame start time (s)", ylabel="Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Centred log10 power")
    fig.savefig(OUT / "spectrogram_before_after.png", dpi=170)
    plt.close(fig)

    # ---- 3. score comparison
    names = PROC_ORDER + ["EEGPT_alone"]
    pretty = [PROC_LABEL[g] for g in PROC_ORDER] + ["EEGPT alone\n(fine-tuned)"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for k, (key, lab) in enumerate((("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy"))):
        bars = axes[0].bar(np.arange(len(names)) + (k - 1) * 0.26, [100 * res[g]["test"][key] for g in names], 0.26, label=lab)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axes[0].set_xticks(range(len(names)), pretty, fontsize=8)
    axes[0].set(ylabel="Score (%)", ylim=(0, 100), title="Held-out TEST (27 participants; previously inspected, exploratory). Settings frozen on validation first.")
    axes[0].legend(ncols=3, loc="upper left")
    axes[0].grid(axis="y", alpha=0.2)
    for k, (key, lab) in enumerate((("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy"))):
        bars = axes[1].bar(np.arange(len(names)) + (k - 1) * 0.26, [100 * res[g]["validation"][key] for g in names], 0.26, label=lab)
        axes[1].bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axes[1].set_xticks(range(len(names)), pretty, fontsize=8)
    axes[1].set(ylabel="Score (%)", ylim=(0, 100), title="VALIDATION (26 participants) — these were used to choose the settings, so they are optimistic selection scores")
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "score_comparison.png", dpi=170)
    plt.close(fig)

    # ---- 4. class recalls
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for k, c in enumerate(CLASSES):
        vals = [100 * res[g]["test"]["per_class"][c]["recall"] for g in names]
        bars = ax.bar(np.arange(len(names)) + (k - 1) * 0.26, vals, 0.26, label=f"{c} recall")
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set_xticks(range(len(names)), pretty, fontsize=8)
    ax.set(ylabel="Recall on test (%)", ylim=(0, 105), title="Per-class recall on the 27-participant test set (11 AD, 6 FTD, 10 HC)")
    ax.legend(ncols=3)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "class_recalls.png", dpi=170)
    plt.close(fig)

    # ---- 5. confusion matrices
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for ax, g in zip(axes.ravel(), names + [None]):
        if g is None:
            ax.axis("off")
            continue
        cm = np.asarray(res[g]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=11)
        for i, j in itertools.product(range(3), repeat=2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual",
               title=PROC_LABEL.get(g, "EEGPT alone").replace("\n", " ") + f"\n{cm.trace()}/27 correct")
        ax.title.set_fontsize(9)
    fig.savefig(OUT / "confusion_matrices.png", dpi=170)
    plt.close(fig)

    # ---- 6. rank grid + eigenvalue spectra
    svm_table = jload(OUT / "svm_search.json")
    best_f1 = np.full((len(CHANNEL_RANKS), len(DELAY_RANKS)), np.nan)
    energy = np.full_like(best_f1, np.nan)
    for i, rc in enumerate(CHANNEL_RANKS):
        for j, rl in enumerate(DELAY_RANKS):
            name = f"c{rc}_d{rl}"
            best_f1[i, j] = 100 * max(r["best_validation_blend"]["val_macro_f1"] for r in svm_table if r["arm"] == name)
            energy[i, j] = 100 * retention[name]["energy_ratio_mean_val"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    for ax, data, title, fmt in ((axes[0], best_f1, "Best validation ensemble macro F1 (%)\n(over 30 SVM x 420 blends; selection score)", "%.1f"),
                                 (axes[1], energy, "Waveform energy kept (%), validation participants\n(not diagnostic information)", "%.1f")):
        im = ax.imshow(data, cmap="viridis", origin="lower")
        for i, j in itertools.product(range(3), range(3)):
            ax.text(j, i, fmt % data[i, j], ha="center", va="center", color="w", fontsize=10)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=DELAY_RANKS, yticklabels=CHANNEL_RANKS, xlabel="Delay rank", ylabel="Electrode rank", title=title)
        fig.colorbar(im, ax=ax)
    axes[2].plot(np.arange(1, 20), np.cumsum(ft["channel_eigenvalues"]) / ft["channel_eigenvalues"].sum() * 100, "o-")
    axes[2].set(title="Electrode factor: cumulative energy", xlabel="Components", ylabel="%")
    axes[3].plot(np.arange(1, L + 1), np.cumsum(ft["delay_eigenvalues"]) / ft["delay_eigenvalues"].sum() * 100, "o-", color="#c0504d")
    axes[3].set(title="Delay factor: cumulative energy", xlabel="Components", ylabel="%")
    for ax in axes[2:]:
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "rank_grid_and_spectra.png", dpi=170)
    plt.close(fig)

    # ---- parameter table for the search record (P2 validation winners)
    pb = jload(OUT / "parameter_best_results.json")
    column = dict(channel_rank="channel_rank", delay_rank="delay_rank", pca="svm_pca", C="svm_C", kernel="svm_kernel",
                  svm_temperature="svm_temperature", eegpt_temperature="eegpt_temperature", svm_weight="svm_weight")
    matching = {}
    with gzip.open(OUT / "all_validation_candidates.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] == "raw":
                continue
            for p, c in column.items():
                matching[(p, json.dumps(r[c]))] = matching.get((p, json.dumps(r[c])), 0) + 1
    names_param = dict(channel_rank="Waveform Tucker electrode rank", delay_rank="Waveform Tucker delay rank (L=32)", pca="SVM PCA components",
                       C="SVM C", kernel="SVM kernel", svm_temperature="SVM temperature", eegpt_temperature="EEGPT temperature",
                       svm_weight="SVM blend weight")
    rows = []
    for e in pb:
        w = e["winner"]
        rows.append("| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:,} | {}/svm {}/blend {} |".format(
            names_param[e["parameter"]], "None" if e["value"] is None else e["value"], 100 * w["val_macro_f1"],
            100 * w["val_balanced_accuracy"], 100 * w["val_accuracy"], matching[(e["parameter"], json.dumps(e["value"]))],
            w["arm"], w["svm_index"], w["blend_index"]))
    record = ["", "## Waveform Tucker-2 reconstruction BEFORE STFT + fine-tuned EEGPT: 174 participants", "",
              f"All {m['verification']['all_validation_candidates_saved']:,} saved candidates, including the no-decomposition arm; the rows below use only the "
              f"{9 * len(SVM_CONFIGS) * len(BLENDS):,} Tucker-reconstruction candidates (9 rank pairs x {len(SVM_CONFIGS)} SVM settings x {len(BLENDS)} blends). "
              "Scores are **validation (26 participants) ensemble selection scores, not test performance**. Training-only factors/SVM; EEGPT unchanged. "
              "Each row's macro F1, BA and accuracy come from the same winning configuration; other parameters vary. Actual test results are kept separately in "
              "[the experiment report](C:/Projects/AIBSI/neural_comparison/advanced_clean_spectral/outputs/waveform_tucker_before_stft_2026-11-05/REPORT.md).", "",
              "Source: [all_validation_candidates.jsonl.gz](C:/Projects/AIBSI/neural_comparison/advanced_clean_spectral/outputs/waveform_tucker_before_stft_2026-11-05/all_validation_candidates.jsonl.gz).", "",
              "| Parameter | Value | Best macro F1 (%) | BA / macro recall (%) | Accuracy (%) | Matching saved configs | Winning candidate |",
              "|---|---|---|---|---|---|---|"] + rows + [""]
    (OUT / "search_record_section.md").write_text("\n".join(record), encoding="utf-8")

    # ---- REPORT.md
    def row(g, label):
        t, v = res[g]["test"], res[g].get("validation")
        recalls = " / ".join(f"{100 * t['per_class'][c]['recall']:.0f}" for c in CLASSES)
        correct = int(np.trace(np.asarray(t["confusion_matrix"])))
        vs = f"{pct(v['macro_f1'])}" if v else "n/a"
        return f"| {label} | {pct(t['accuracy'])} ({correct}/27) | {pct(t['macro_f1'])} | {pct(t['balanced_accuracy'])} | {recalls} | {vs} |"

    def cfg(g):
        s = sel[g]
        arm = "none (rebuilt raw waveforms)" if s["arm"] is None else f"electrode rank {s['arm'][0]}, delay rank {s['arm'][1]}"
        return (f"`{g}`: {arm}; SVM `{json.dumps(s['svm'])}`; blend `{json.dumps(s['blend'])}`; validation ensemble F1 "
                f"{pct(s['validation']['val_macro_f1'])}%, accuracy {pct(s['validation']['val_accuracy'])}%")

    interp = (OUT / "interpretation.md")
    lines = ["# Waveform Tucker-2 reconstruction BEFORE the STFT, then SVM + fine-tuned EEGPT", "",
             "Everything below was run by `experiment_waveform_tucker_before_stft.py`. EEGPT was **not** retrained (saved probabilities reused); existing experiments and caches were not modified.", "",
             "```text", "rebuilt 8 s / 128 Hz / 19-ch windows -> delay tensor (19, 32, 993) -> training-only shared Tucker-2 (electrode x delay)",
             "-> anti-diagonal averaging back to (19, 1024) -> same STFT / log-mel (float16) / mean+SD summaries (912) -> site-normalised SVM",
             "-> probability blend with the unchanged fine-tuned EEGPT -> participant-level AD / FTD / HC", "```", "",
             "## Data and split (unchanged)", "",
             "174 independent participants (71 AD, 42 FTD, 61 HC; 88 AHEPA + 86 BrainLat). Historical split: 121 train / 26 validation / 27 test, identical to the saved split file. "
             "The 27-person test set (11 AD, 6 FTD, 10 HC) has been inspected in many earlier experiments, so **all test numbers are exploratory**. One test participant is worth 3.70 accuracy points.", "",
             "Waveforms were rebuilt from the original recordings with the project's own harmonisation, QC and window-selection code. They match the saved 6,721-window cache: "
             f"identical window ownership/counts/labels; rebuilt STFT values bitwise equal in {100*jload(PREP/'rebuild_audit.json')['stft_fraction_bitwise_equal']:.3f}% of entries (maximum difference "
             f"{jload(PREP/'rebuild_audit.json')['stft_max_abs_diff']:.4f}, one float16 step). The historical direct-SVM + EEGPT result was reproduced exactly from the stored cache "
             f"(accuracy {pct(res['Historical_stored_cache_SVM_plus_EEGPT']['test']['accuracy'])}%, macro F1 {pct(res['Historical_stored_cache_SVM_plus_EEGPT']['test']['macro_f1'])}%).", "",
             "## Test results (settings chosen on validation only, then frozen)", "",
             "Recalls are AD / FTD / HC (%). The last column is the validation macro F1 used for selection (optimistic).", "",
             "| Procedure | Accuracy % (correct) | Macro F1 % | Balanced accuracy % | Recall AD/FTD/HC | Validation macro F1 % |", "|---|---:|---:|---:|---|---:|"]
    for g in PROC_ORDER:
        lines.append(row(g, PROC_LABEL[g].replace("\n", " ")))
    lines.append(row("EEGPT_alone", "EEGPT alone (fine-tuned, unchanged)"))
    h = res["Historical_stored_cache_SVM_plus_EEGPT"]["test"]
    lines.append(f"| Historical direct log-mel SVM + EEGPT (stored cache, reproduced) | {pct(h['accuracy'])} ({int(np.trace(np.asarray(h['confusion_matrix'])))}/27) | {pct(h['macro_f1'])} | {pct(h['balanced_accuracy'])} | "
                 + " / ".join(f"{100*h['per_class'][c]['recall']:.0f}" for c in CLASSES) + " | 71.00 |")
    lines += ["", "Procedures: **C1** rebuilt waveforms with the historical SVM (PCA20, C=0.1, linear) and a validation-chosen blend; **C2** the same input with the 30-setting SVM search; "
              "**C3/C4** full-rank (19, 32) Tucker reconstruction, which must equal the input (identity control), with fixed / searched SVM; **P1** Tucker rank pair chosen on validation with the fixed historical SVM; "
              "**P2 (primary)** rank pair, SVM and blend chosen jointly on validation. All six were declared in `protocol.json` before test scoring.", "",
             "Selected settings:", ""] + [f"- {cfg(g)}" for g in PROC_ORDER] + ["",
             "## Comparison with the earlier tensor work (same 27-person test set; earlier numbers copied from the handover, not re-run)", "",
             "| Method | Accuracy % | Macro F1 % | Balanced accuracy % |", "|---|---:|---:|---:|",
             "| Direct log-mel SVM + EEGPT (no tensor) | 70.37 | 67.82 | 67.58 |",
             "| Summary-first Tensor Train → SVM alone | 51.85 | 46.35 | 46.77 |",
             "| Tucker AFTER log-mel, BEFORE pooling → SVM | 59.26 | 57.56 | 58.18 |",
             "| Jointly selected TT/SVM + EEGPT | 62.96 | 58.73 | 58.69 |",
             f"| **New: waveform Tucker BEFORE STFT + EEGPT (P2)** | {pct(res['P2']['test']['accuracy'])} | {pct(res['P2']['test']['macro_f1'])} | {pct(res['P2']['test']['balanced_accuracy'])} |",
             f"| New: waveform Tucker BEFORE STFT → SVM alone (P2 SVM) | {pct(res['P2']['test_svm_alone']['accuracy'])} | {pct(res['P2']['test_svm_alone']['macro_f1'])} | {pct(res['P2']['test_svm_alone']['balanced_accuracy'])} |", "",
             "The earlier rows used the same 174 people and split but different intermediate representations, and different amounts of search; they are context, not a controlled ablation. "
             "Note that the TT/Tucker-after-log-mel rows operate on summarised spectral tensors, whereas the new method changes the waveform before any spectral processing.", "",
             "## Search that was run", "",
             f"- Tucker pilot grid (frozen before examining results): electrode ranks {CHANNEL_RANKS} x delay ranks {DELAY_RANKS}, delay length {L}; the (19, 32) cell is the identity control.",
             f"- Downstream SVM grid for **every** arm, including the no-decomposition arm: PCA {{20, 40, none}} x C {{0.03, 0.1, 0.3, 1, 3}} x {{linear, RBF}} = {len(SVM_CONFIGS)}.",
             f"- Blend grid: 5 SVM temperatures x 4 EEGPT temperatures x 21 weights = {len(BLENDS)} per SVM.",
             f"- {m['verification']['all_validation_candidates_saved']:,} joint validation candidates were scored and saved (`all_validation_candidates.jsonl.gz`). The Tucker arms receive 9x as many candidates as the no-decomposition arm, "
             "which inflates their best validation score in expectation; C4 (identity arm) and the fixed-SVM controls P1/C1/C3 are included so that effect can be read off.",
             "- A 5-fold training-only CV (factors, site statistics, scaler, PCA and SVM refit inside each fold) supplied the standalone-SVM tie-break score. It is **not** a nested evaluation of the ensemble, because the fine-tuned EEGPT already saw labels of its original training participants.", "",
             "## Plots (plain-language explanation)", "",
             "- `waveform_reconstruction.png`: one 8 s signal at electrode Cz — original, rebuilt, and the part that was removed. Rank 8x8 smooths more aggressively than the selected setting. Energy kept is a statement about signal size, **not** about diagnostic information.",
             "- `spectrogram_before_after.png`: the same window's log-power spectrogram before/after, and their difference. Differences are concentrated where the low-rank filter attenuated frequencies.",
             "- `score_comparison.png`: accuracy, macro F1 and balanced accuracy on the test set (top) and validation set (bottom). The validation panel is a selection score and is higher by construction.",
             "- `class_recalls.png`: how many AD, FTD and HC participants were found by each procedure. FTD has only 6 test participants, so one person moves FTD recall by 16.7 points.",
             "- `confusion_matrices.png`: rows are the true class, columns the predicted class; the title shows participants correct out of 27.",
             "- `rank_grid_and_spectra.png`: validation F1 and kept waveform energy for each rank pair, and the cumulative energy of the electrode and delay factors.", "",
             "![Waveform](waveform_reconstruction.png)", "", "![Spectrogram](spectrogram_before_after.png)", "", "![Scores](score_comparison.png)", "",
             "![Recalls](class_recalls.png)", "", "![Confusion](confusion_matrices.png)", "", "![Ranks](rank_grid_and_spectra.png)", ""]
    if interp.exists():
        lines += ["## Interpretation", "", interp.read_text(encoding="utf-8"), ""]
    lines += ["## Limitations", "",
              "- Fixed historical split, 27 test participants, test set previously inspected: exploratory only. A large joint search on 26 validation participants overfits; the best validation score is not expected performance.",
              "- A single fine-tuned EEGPT checkpoint cannot be treated as leakage-free in arbitrary new folds; no nested CV of the ensemble was performed.",
              "- The cohorts differ strongly in site, equipment and amplitude (a BrainLat amplitude confound has been observed before). A gain on this split would not establish clinical portability.",
              "- Temperature-adjusted SVM softmax values are blend scores, not calibrated clinical risks.",
              "- Low-rank reconstruction can remove weak disease-related signal as well as noise, and can keep strong artefacts.", "",
              "## Files", "", "`protocol.json` (written before test scoring), `selection.json`, `metrics.json`, `verification.json`, `all_validation_candidates.jsonl.gz`, `svm_search.json`, "
              "`parameter_best_results.json`, `inner_cv.json`, `inner_oof.npz`, `factors_train121.npz`, `factors_dev147.npz`, `model_P2.joblib`, `model_P1.joblib`, `probabilities.npz`, "
              "`participant_probabilities.json`, `retention.json`, `source_hashes.json`, `prepared/` (waveform cache, `waveform_manifest.json` with offsets/QC, `rebuild_audit.json`, fingerprint), `cache/` (per-arm feature caches).", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log("Report written:", OUT / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "run", "hashraw", "report"], default="run")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        if args.stage == "prepare":
            stage_prepare(args.jobs)
            return
        if args.stage == "report":
            report()
            return
        if args.stage == "hashraw":
            hash_raw_sources()
            return
        run(args)


if __name__ == "__main__":
    main()
