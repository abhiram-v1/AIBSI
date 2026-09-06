"""Full-cohort primary + leftovers; unchanged QC; waveform delay moments."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly, spectrogram
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tensor_pipeline/outputs/v3_waveform_full_cohort'
PARTS = [ROOT / 'preprocessing/02_balanced_primary' / s for s in ['primary', 'leftover']]
LAGS = 16


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def spectral_features(x):
    f, _, p = spectrogram(x, fs=250, window='hann', nperseg=500, noverlap=250,
                          nfft=500, detrend='constant', scaling='density', mode='psd', axis=-1)
    keep = (f >= 1) & (f <= 30)
    f, p = f[keep], p[:, :, keep].mean(-1).astype(float)
    bp = np.stack([p[:, :, (f >= lo) & (f < hi)].sum(-1)*.5
                   for lo, hi in [(1,4), (4,8), (8,13), (13,30.01)]], -1)
    rel = bp / np.maximum(bp.sum(-1, keepdims=True), 1e-12)
    dist = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    ent = -(dist*np.log(np.maximum(dist, 1e-12))).sum(-1)/np.log(len(f))
    af = (f >= 7) & (f <= 13)
    peak = f[af][p[:, :, af].argmax(-1)]
    ratio = np.log10(np.maximum(bp[:, :, :2].sum(-1), 1e-12)/np.maximum(bp[:, :, 2:].sum(-1), 1e-12))
    result = []
    for a in [np.log10(np.maximum(bp, 1e-12)), rel, ent, peak, ratio]:
        result.extend([a.mean(0).ravel(), a.std(0).ravel()])
    return np.concatenate(result)


def delay_moment(x):
    """Exact mean of within-window delay covariance; each window equal weight.

    x = already-cleaned 19 x 1000 @250Hz. Anti-aliased resampling gives 500
    samples @125Hz. Delay view is channel x lag x local position (19x16x485).
    Moment suffices to evaluate mean squared ANY linear delay-tensor filter.
    """
    y = resample_poly(np.asarray(x, dtype=float), 1, 2, axis=-1)
    z = sliding_window_view(y, LAGS, axis=-1).transpose(0,2,1).reshape(19*LAGS, -1)
    z = z-z.mean(-1, keepdims=True)
    return z@z.T/z.shape[1]


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserving existing experiment: {OUT}')
    arrays = [np.load(p/'signals.npy', mmap_mode='r') for p in PARTS]
    metas = [dict(np.load(p/'metadata.npz')) for p in PARTS]
    ids = np.concatenate([m['sample_id'] for m in metas])
    assert len(ids) == len(set(ids)) == 17419
    subjects = np.unique(np.concatenate([m['subject'] for m in metas]))
    assert len(subjects) == 88
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for d in PARTS for p in [d/'signals.npy',d/'metadata.npz']}
    old_clean = dict(np.load(ROOT/'preprocessing/04_clean_primary/primary/metadata.npz'))
    clean_ids = set(old_clean['sample_id'])
    old_ex = dict(np.load(ROOT/'preprocessing/04_clean_primary/excluded/metadata.npz'))
    old_bad = set(old_ex['sample_id'][old_ex['exclusion_reason'] != 'balance_trim_medium_duration_quality_first'])
    old_balance = set(old_ex['sample_id'][old_ex['exclusion_reason'] == 'balance_trim_medium_duration_quality_first'])
    old_v2 = set(np.load(ROOT/'tensor_pipeline/outputs/v2_tensor_train/index.npz')['sample_id'])
    covs, controls, labels, records, exclusions, subjects_info = [], [], [], [], [], []
    channels = None
    headers = {}
    # Source data remain read-only; new features have their own directory.
    OUT.mkdir(parents=True)
    for s in subjects:
        path = ROOT/f'dataset/ds004504/derivatives/{s}/eeg/{s}_task-eyesclosed_eeg.set'
        d = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=['chanlocs','event','srate','pnts'])
        names = [str(c.labels) for c in d['chanlocs']]
        channels = names if channels is None else channels
        assert channels == names and d['srate'] == 500
        cuts = [(float(e.latency)-.5)/500 for e in np.atleast_1d(d.get('event', []))
                if str(getattr(e, 'type', '')).lower() == 'boundary']
        headers[str(s)] = dict(channels=names, cuts_seconds=cuts, pnts=int(d['pnts']))
        pieces, rows = [], []
        for part, (a,m) in enumerate(zip(arrays,metas)):
            ix = np.flatnonzero(m['subject'] == s)
            x = np.asarray(a[ix], dtype=np.float32).copy()
            x -= x.mean(axis=1, keepdims=True)
            x -= x.mean(axis=2, keepdims=True)
            for j,i in enumerate(ix):
                sample_id, start = str(m['sample_id'][i]), float(m['start_seconds'][i])
                reasons = []
                if str(m['qc_status'][i]) != 'ok': reasons.append('original_qc_not_ok')
                if not np.isfinite(x[j]).all(): reasons.append('nonfinite')
                if np.max(np.abs(x[j])) >= 400: reasons.append('amplitude_ge_400uv')
                if np.max(np.abs(np.diff(x[j], axis=-1))) >= 150: reasons.append('step_ge_150uv_per_4ms')
                if x[j].std(-1).min() < 1: reasons.append('channel_std_lt_1uv')
                if any(start < t+.5 and start+4 > t-.5 for t in cuts): reasons.append('boundary_0.5s_guard')
                row = dict(sample_id=sample_id, subject=str(s), label=int(m['label'][i]),
                           origin=['primary','leftover'][part], source_index=int(i), start_seconds=start)
                if reasons:
                    exclusions.append(dict(**row, reasons=reasons))
                else:
                    assert sample_id not in old_bad
                    pieces.append(x[j])
                    rows.append(row)
        order = np.argsort([r['start_seconds'] for r in rows])
        assert len(order) > 0
        x = np.stack(pieces)[order]
        rows = [rows[i] for i in order]
        assert len(set(r['label'] for r in rows)) == 1
        cov = np.zeros((19*LAGS,19*LAGS), dtype=float)
        for w in x: cov += delay_moment(w)
        cov /= len(x)
        covs.append(cov)
        controls.append(spectral_features(x))
        labels.append(rows[0]['label'])
        records.extend(rows)
        info = dict(subject=str(s), label=labels[-1], windows=len(x), seconds=4*len(x),
                    origins=dict(Counter(r['origin'] for r in rows)))
        subjects_info.append(info)
        print(f'Prepared {s}: {len(x)} windows; {info["origins"]}', flush=True)
    kept_ids = {r['sample_id'] for r in records}
    assert old_v2 <= kept_ids and not old_bad & kept_ids
    labels = np.array(labels)
    assert np.array_equal(np.bincount(labels), [36,23,29])
    covs, controls = np.array(covs), np.array(controls)
    assert np.isfinite(covs).all() and np.isfinite(controls).all()
    np.save(OUT/'delay_moments.npy', covs)
    np.save(OUT/'spectral_controls.npy', controls)
    np.savez_compressed(OUT/'index.npz', subjects=subjects, labels=labels,
                        counts=np.array([s['windows'] for s in subjects_info]), channels=np.array(channels))
    (OUT/'retained_windows.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    report = dict(source_hashes=source_hashes, source_headers=headers, subjects=subjects_info,
                  initial_windows=len(ids), retained_windows=len(records), excluded_windows=len(exclusions),
                  exclusions=exclusions, excluded_reason_counts=dict(Counter(k for r in exclusions for k in r['reasons'])),
                  kept_origins=dict(Counter(r['origin'] for r in records)),
                  retained_windows_per_class={g:sum(r['label']==i for r in records) for i,g in enumerate(['AD','FTD','CN'])},
                  subjects_per_class={'AD':36,'FTD':23,'CN':29}, recovered_balance_only_windows=len(old_balance & kept_ids),
                  all_v2_windows_retained=True, former_artifact_exclusions_not_reintroduced=True,
                  delay_tensor_shape_per_window=[19,16,485], delay_sampling_hz=125, lag_span_ms=120,
                  moments_shape=list(covs.shape), spectral_control_shape=list(controls.shape),
                  note='Moment computation is an exact sufficient-statistic implementation of mean-squared raw delay-filter responses; no Fourier transform in waveform arms. Local-position and window ordering are deliberately pooled, not modeled as long-range sequence dynamics. Partial tails remain excluded.')
    (OUT/'preparation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    manifest = {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), OUT/'delay_moments.npy',OUT/'spectral_controls.npy',OUT/'index.npz',OUT/'preparation_report.json',OUT/'retained_windows.json']}
    (OUT/'input_fingerprints.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ['retained_windows','excluded_windows','kept_origins','subjects_per_class','retained_windows_per_class','recovered_balance_only_windows']}, indent=2))


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4): main()
