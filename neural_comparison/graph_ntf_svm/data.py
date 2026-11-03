from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
MOD=ROOT/'neural_comparison/advanced_clean_spectral/outputs/moderate_log_mel/prepared'
CHANNELS=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
BANDS=[(1,4),(4,8),(8,13),(13,30),(30,40)]
CLASSES=['AD','FTD','HC']
STATS=['mean','SD','median','IQR']


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def summarize(logmel):
    if logmel.ndim!=4 or logmel.shape[1:]!=(19,24,13) or not np.isfinite(logmel).all():
        raise ValueError('Expected finite window x 19 x 24 x 13 log-mel data')
    l=np.asarray(logmel,np.float64)
    # Stabilized inverse log: any per-window centering cancels in frame normalization.
    power=10**(l-l.max(axis=(1,2),keepdims=True))
    power=np.sqrt(power/np.maximum(power.sum(axis=(1,2),keepdims=True),1e-15))
    values=power.transpose(0,3,1,2).reshape(-1,19,24)
    tensor=np.stack([values.mean(0),values.std(0),np.median(values,axis=0),
                     np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0)],axis=-1)
    original=l.transpose(0,3,1,2).reshape(-1,19,24)
    direct=np.concatenate([original.mean(0),original.std(0)],axis=1).ravel()
    return tensor.astype(np.float32),direct.astype(np.float32)


def prepare(destination):
    destination=Path(destination)
    manifest=json.loads((MOD/'manifest.json').read_text())
    if manifest['channels']!=CHANNELS:
        raise ValueError('Unexpected saved electrode order')
    windows=np.load(MOD/'log_mel.npy',mmap_mode='r')
    subject=np.load(MOD/'subject_index.npy')
    labels=np.load(MOD/'labels.npy')
    people=manifest['participants']
    if len({p['subject_id'] for p in people})!=len(people):
        raise ValueError('Duplicate participant IDs')
    tensors,direct,targets=[],[],[]
    for i,p in enumerate(people):
        indices=np.flatnonzero(subject==i)
        if len(indices)!=p['windows'] or len(np.unique(labels[indices]))!=1:
            raise ValueError('Invalid participant-window mapping')
        x,d=summarize(windows[indices])
        tensors.append(x);direct.append(d);targets.append(int(labels[indices[0]]))
    import mne
    montage=mne.channels.make_standard_montage('standard_1020')
    pos=np.stack([montage.get_positions()['ch_pos'][c] for c in CHANNELS])
    source_hashes={name:fingerprint(MOD/name) for name in ['manifest.json','log_mel.npy','subject_index.npy','labels.npy']}
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'dataset.npz').exists() and (destination/'manifest.json').exists():
        raise FileExistsError('Prepared dataset exists; reuse it or choose another directory')
    np.savez_compressed(destination/'dataset.npz',x=np.asarray(tensors),direct=np.asarray(direct),y=targets,
                        sites=np.array([p['site'] for p in people]),positions=pos)
    clinical=clinical_scores(people)
    write_json(destination/'manifest.json',{'participants':people,'classes':CLASSES,'channels':CHANNELS,
        'statistics':STATS,'tensor_shape':list(np.asarray(tensors).shape),'window_count':len(windows),
        'source_hashes':source_hashes,'transform':'inverse centered log power; per-frame spatial/spectral relative power; sqrt; mean,SD,median,IQR',
        'caveat':'Uses existing moderate window selection. Original preparation did not explicitly reject EEGLAB boundary annotations. Interpret as a legacy-preprocessing experiment.',
        'clinical':clinical})
    print(f'Prepared {len(people)} participants: {np.asarray(tensors).shape}; all non-negative',flush=True)


def clinical_scores(people):
    original=pd.read_csv(ROOT/'dataset/ds004504/participants.tsv',sep='\t').set_index('participant_id')
    frames=[]
    for folder,name in [('1_AD','AD'),('2_bvFTD','bvFTD'),('5_HC','HC')]:
        frames.append(pd.read_csv(ROOT/f'dataset/brainlat/EEG/{folder}/Cognition_{name}_EEG_data.csv'))
    brain=pd.concat(frames).set_index('id EEG')
    records=[]
    for p in people:
        sid=p['subject_id'].split(':')[-1]
        if p['source']=='ds004504':
            row=original.loc[sid]
            records.append({'instrument':'MMSE','score':float(row.MMSE),'age':float(row.Age),'sex':str(row.Gender)})
        else:
            entries=pd.to_numeric(brain.loc[[sid],'moca_total'],errors='coerce').dropna().unique() if sid in brain.index else []
            if len(entries)>1:
                raise ValueError(f'Conflicting cognitive scores for {sid}')
            val=entries[0] if len(entries) else np.nan
            records.append({'instrument':'MoCA','score':None if pd.isna(val) else float(val),'age':None,'sex':None})
    return records


def raw_windows(path,source,site):
    sys.path.insert(0,str(ROOT/'neural_comparison/stft_gpu_3class'))
    import run_pipeline as old
    record=old.Record('inference',source,site,'unknown',str(path))
    data,header=old.load_harmonized(record)
    candidates=[]
    for offset in range(10*128,data.shape[1]-10*128-1024+1,1024):
        window=data[:,offset:offset+1024]
        if old.valid_window(window)[0]:
            candidates.append(window)
    selected=old.evenly_limit(candidates,40)
    if len(selected)<5:
        raise ValueError('At least five valid eight-second windows required')
    return np.stack(selected),header


def mel_from_raw(windows):
    sys.path.insert(0,str(ROOT/'neural_comparison/stft_gpu_3class'))
    import run_pipeline as old
    sys.path.insert(0,str(ROOT/'neural_comparison/advanced_clean_spectral'))
    import pipeline as spectral
    stft=np.stack([old.window_to_stft(w) for w in windows]).astype(np.float16).astype(np.float32)
    bank=spectral.mel_filter_bank(np.linspace(1,40,79))
    p=np.einsum('mf,ncft->ncmt',bank,10**stft,optimize=True)
    l=np.log10(p+1e-20)
    l-=np.median(l,axis=(1,2,3),keepdims=True)
    return l.astype(np.float16)


def connectivity_from_raw(windows):
    """Non-negative wPLI from within-window analytic cross products; no phase joins."""
    output=np.zeros((19,19,len(BANDS)),np.float64)
    for k,band in enumerate(BANDS):
        filt=butter(4,band,btype='bandpass',fs=128,output='sos')
        z=hilbert(sosfiltfilt(filt,windows.astype(float),axis=-1),axis=-1)
        for i in range(19):
            for j in range(i+1,19):
                v=np.imag(z[:,i]*z[:,j].conj())
                value=abs(v.mean())/max(float(np.abs(v).mean()),1e-30)
                output[i,j,k]=output[j,i,k]=value
    return output.astype(np.float32)


def prepare_connectivity(destination):
    destination=Path(destination)
    manifest=json.loads((destination/'manifest.json').read_text())
    parts=destination/'connectivity_parts';parts.mkdir(exist_ok=True)
    tensors=[]
    for i,p in enumerate(manifest['participants']):
        file=parts/f'{i:03d}.npz'
        key={'subject_id':p['subject_id'],'set_sha256':fingerprint(p['path'])}
        fdt=Path(p['path']).with_suffix('.fdt')
        if fdt.exists():key['fdt_sha256']=fingerprint(fdt)
        if file.exists() and json.loads(file.with_suffix('.json').read_text())==key:
            c=np.load(file)['connectivity']
        else:
            windows,_=raw_windows(p['path'],p['source'],p['site'])
            if len(windows)!=p['windows']:
                raise ValueError(f'Window count differs from frozen spectral cohort: {p["subject_id"]}')
            c=connectivity_from_raw(windows)
            np.savez_compressed(file,connectivity=c)
            write_json(file.with_suffix('.json'),key)
        tensors.append(c)
        print(f'Connectivity {i+1}/{len(manifest["participants"])}',flush=True)
    np.save(destination/'connectivity.npy',np.asarray(tensors))
    write_json(destination/'connectivity_protocol.json',{'shape':list(np.asarray(tensors).shape),
        'bands_hz':BANDS,'measure':'wPLI','diagonal':'masked in factorization',
        'windowing':'Same moderate 8-second accepted windows, max40; analytic phase computed within each window',
        'limitations':'19-channel BrainLat spatial interpolation can affect connectivity; sensor-space associations are not anatomical connections.'})
