"""Version 2: verified channels, EEGLAB boundaries, absolute window spectra."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'preprocessing/04_clean_primary/primary'
OUT = ROOT / 'tensor_pipeline/outputs/v2_tensor_train'
FS = 250
GUARD = 0.5

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(SOURCE / 'signals.npy', mmap_mode='r')
    m = np.load(SOURCE / 'metadata.npz')
    subjects = np.unique(m['subject'])
    keep = np.ones(len(x), bool)
    boundary_info = []
    channels = None
    coordinates = None
    exclusions = []
    source_checks = []
    for sub in subjects:
        path = ROOT / f'dataset/ds004504/derivatives/{sub}/eeg/{sub}_task-eyesclosed_eeg.set'
        d = loadmat(path, squeeze_me=True, struct_as_record=False,
                    variable_names=['chanlocs','event','srate','pnts'])
        names = [str(c.labels) for c in d['chanlocs']]
        if channels is None:
            channels = names
            coordinates = np.array([[c.X,c.Y,c.Z] for c in d['chanlocs']], float)
        assert channels == names, f'Inconsistent channel order: {sub}'
        assert float(d['srate']) == 500
        cuts = [(float(e.latency)-0.5)/500 for e in np.atleast_1d(d.get('event', []))
                if str(getattr(e, 'type', '')).lower() == 'boundary']
        indices = np.flatnonzero(m['subject'] == sub)
        for i in indices:
            start = float(m['start_seconds'][i])
            hit = [t for t in cuts if start < t + GUARD and start+4 > t-GUARD]
            if hit:
                keep[i] = False
                exclusions.append(dict(source_index=int(i), sample_id=str(m['sample_id'][i]),
                                       subject=str(sub), reason='EEGLAB_boundary_with_0.5s_guard', cuts=hit))
        boundary_info.append(dict(subject=str(sub), boundary_count=len(cuts), original_windows=len(indices),
                                  retained_windows=int(keep[indices].sum()), removed=int((~keep[indices]).sum())))
    assert all(r['retained_windows'] >= 24 for r in boundary_info)
    # Check stored array against actual source transformations, including channel order.
    for sub in [subjects[0], 'sub-065', 'sub-086']:
        path = ROOT / f'dataset/ds004504/derivatives/{sub}/eeg/{sub}_task-eyesclosed_eeg.set'
        raw = loadmat(path, variable_names=['data'])['data'].astype('float32')
        i = np.flatnonzero((m['subject']==sub)&keep)[0]
        j = int(m['window_index'][i])*2000
        expected = raw[:, j:j+2000:2].copy()
        expected -= expected.mean(axis=1, keepdims=True)
        expected -= expected.mean(axis=0, keepdims=True)
        expected -= expected.mean(axis=1, keepdims=True)
        err = float(np.max(np.abs(expected-x[i])))
        assert err < 0.001
        source_checks.append(dict(subject=str(sub), max_source_transform_error_uv=err))
    indices = np.flatnonzero(keep)
    # Store by subject and derivative-window time; temporal coordinates are local,
    # not falsely described as event-aligned across resting-state participants.
    indices = np.array(sorted(indices, key=lambda i:(str(m['subject'][i]),int(m['start_seconds'][i]))))
    spectra = np.lib.format.open_memmap(OUT/'window_logpsd.npy', mode='w+', dtype='float32',
                                       shape=(len(indices),19,59,3))
    subjects_tensor = np.empty((len(subjects),19,59,3), 'float32')
    controls = []
    control_names = None
    counts = []
    for si, sub in enumerate(subjects):
        positions = np.flatnonzero(m['subject'][indices] == sub)
        src_indices = indices[positions]
        f,t,p = spectrogram(np.asarray(x[src_indices]), fs=FS, window='hann', nperseg=500,
                            noverlap=250, nfft=500, detrend='constant', scaling='density', mode='psd', axis=-1)
        mask=(f>=1)&(f<=30)
        p=p[:,:,mask,:].astype(float)
        lp=np.log10(np.maximum(p,1e-12))
        spectra[positions]=lp
        # Subject tensor: no arbitrary early/middle/late alignment. Statistics
        # capture all retained short-time spectra, retaining absolute power.
        subjects_tensor[si]=np.stack([lp.mean(axis=(0,3)), np.median(lp,axis=(0,3)),
                                      lp.std(axis=(0,3))],axis=-1)
        meanpsd=p.mean(axis=-1)
        freqs=f[mask]
        bands=[(1,4),(4,8),(8,13),(13,30.01)]
        bp=np.stack([meanpsd[:,:,(freqs>=lo)&(freqs<hi)].sum(axis=-1)*.5 for lo,hi in bands],axis=-1)
        rel=bp/np.maximum(bp.sum(axis=-1,keepdims=True),1e-12)
        feature_maps=[np.log10(np.maximum(bp,1e-12)),rel]
        summaries=[]
        names=[]
        for kind,fm in zip(['log_absolute_bandpower','relative_bandpower'],feature_maps):
            for stat,vals in [('mean',fm.mean(0)),('std',fm.std(0))]:
                summaries.extend(vals.ravel().tolist())
                names.extend([f'{kind}/{stat}/{ch}/{lo}-{hi}' for ch in channels for lo,hi in bands])
        distribution=meanpsd/np.maximum(meanpsd.sum(-1,keepdims=True),1e-12)
        entropy=-(distribution*np.log(np.maximum(distribution,1e-12))).sum(-1)/np.log(len(freqs))
        alpha=meanpsd[:,:,(freqs>=7)&(freqs<=13)]
        peak=freqs[(freqs>=7)&(freqs<=13)][np.argmax(alpha,axis=-1)]
        ratio=np.log10(np.maximum(bp[:,:,:2].sum(-1),1e-12)/np.maximum(bp[:,:,2:].sum(-1),1e-12))
        for kind,vals in [('entropy',entropy),('alpha_peak_hz',peak),('log_slow_fast_ratio',ratio)]:
            for stat,vec in [('mean',vals.mean(0)),('std',vals.std(0))]:
                summaries.extend(vec.tolist())
                names.extend([f'{kind}/{stat}/{ch}' for ch in channels])
        control_names=names
        controls.append(summaries)
        counts.append(len(src_indices))
        print(f'Prepared {si+1}/69 {sub}: {len(src_indices)} boundary-safe windows',flush=True)
    spectra.flush()
    np.save(OUT/'subject_spectral_tensor.npy',subjects_tensor)
    np.save(OUT/'spectral_controls.npy',np.array(controls))
    np.save(OUT/'coordinates.npy',coordinates)
    np.save(OUT/'channels.npy',np.array(channels))
    label=np.array([m['label'][np.flatnonzero(m['subject']==s)[0]] for s in subjects])
    win_subject=np.searchsorted(subjects,m['subject'][indices])
    np.savez_compressed(OUT/'index.npz', subjects=subjects, labels=label, window_subject=win_subject,
                        source_indices=indices, sample_id=m['sample_id'][indices],
                        start_seconds=m['start_seconds'][indices], counts=np.array(counts),
                        frequencies_hz=freqs, segment_centers_seconds=t,
                        control_names=np.array(control_names))
    report=dict(source=str(SOURCE), source_hashes={name:sha(SOURCE/name) for name in ['signals.npy','metadata.npz']},
                boundary_guard_seconds=GUARD, original_windows=len(x),retained_windows=len(indices),
                removed_windows=int((~keep).sum()),excluded=exclusions,subjects=boundary_info,
                actual_channel_order=channels,original_tensor_channel_order_wrong=True, source_checks=source_checks,
                window_tensor_shape=list(spectra.shape),subject_tensor_shape=list(subjects_tensor.shape),
                subject_statistics=['mean logPSD','median logPSD','std logPSD'],
                controls_shape=list(np.array(controls).shape),
                retained_per_group={str(g):int(np.sum(m['label'][indices]==g)) for g in (0,1,2)},
                software=dict(python=sys.version,platform=platform.platform()),
                caveats=['Shared-reference variance is not proof of artifact; CAR is retained for comparability.',
                         'No clinical label, MMSE, age, sex, duration or window count is a predictor.',
                         'Original subject-matched cohort is reused; no external independent test cohort exists.',
                         'No 30-45 Hz features in primary run.'])
    (OUT/'preparation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ['original_windows','retained_windows','removed_windows','window_tensor_shape','subject_tensor_shape','retained_per_group']},indent=2))

if __name__=='__main__':
    main()
