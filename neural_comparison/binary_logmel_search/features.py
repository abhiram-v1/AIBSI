"""Participant-local EEG preprocessing and configurable log-mel summaries."""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
from scipy.signal import stft
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORT = ROOT / 'neural_comparison/graph_ntf_svm/outputs/prepared/manifest.json'
FS = 256
CHANNELS = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
POLICY = {'source_resample_hz': FS, 'candidate_sampling_hz': [128,256], 'filter_hz': [1,40], 'reference': 'average', 'edge_seconds': 10,
          'boundary_guard_seconds': 2, 'duration_cap_seconds': 320,
          'minimum_windows': 3, 'segment_overlap': 0,
          'quality': {'minimum_channel_sd_uv': .1, 'maximum_channel_sd_uv': 200, 'maximum_absolute_uv': 500},
          'gain': 'subtract each segment log-mel median', 'statistics': ['mean','SD','median','IQR'],
          'mel_filter': 'triangular, unit-sum per band, 1-40 Hz',
          'stft': 'Hann, constant detrend, no padding, nfft=nperseg',
          'sources': 'existing derivatives; upstream cleaning differs across datasets'}


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:20]


def representation(config):
    return {k:config[k] for k in ['sampling_hz','segment_seconds','stft_seconds','hop_seconds','mel_bins']}


def mel_bank(frequencies, bins):
    mel=lambda hz:2595*np.log10(1+hz/700)
    points=700*(10**(np.linspace(mel(1),mel(40),bins+2)/2595)-1)
    bank=np.maximum(0,np.minimum((frequencies[None]-points[:-2,None])/(points[1:-1]-points[:-2])[:,None],
                                (points[2:,None]-frequencies[None])/(points[2:]-points[1:-1])[:,None]))
    totals=bank.sum(1,keepdims=True)
    if np.any(totals<=0): raise ValueError('Empty mel filter: increase STFT length or reduce mel count')
    return bank/totals


def load_eeg(person):
    """Reuse the verified montage conversion; preserve original bad/boundary times."""
    sys.path.insert(0,str(ROOT/'neural_comparison/stft_gpu_3class'))
    import run_pipeline as old
    old.FS=FS  # Direct native-source -> 256 Hz, before preparing the 128-Hz control.
    with contextlib.redirect_stdout(sys.stderr), warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mne.set_log_level('ERROR')
        header=mne.io.read_raw_eeglab(person['path'],preload=False,verbose='ERROR')
        intervals=[]
        for onset,duration,description in zip(header.annotations.onset,header.annotations.duration,header.annotations.description):
            if 'boundary' in description.lower() or description.lower().startswith(('bad','edge')):
                onset=float(onset-header.first_time)
                intervals.append((onset-2,onset+max(float(duration),0)+2))
        del header
        record=old.Record(person['subject_id'],person['source'],person['site'],person.get('label','unknown'),person['path'])
        data,details=old.load_harmonized(record)
    return data,intervals,details


def segment_windows(data,seconds,intervals,fs=FS):
    length=int(round(seconds*fs)); accepted=[]; rejected={'boundary':0,'quality':0}
    for start in range(10*fs,data.shape[1]-10*fs-length+1,length):
        if any(start/fs<b and (start+length)/fs>a for a,b in intervals):
            rejected['boundary']+=1; continue
        w=data[:,start:start+length]
        sd=w.std(1)*1e6
        if not np.isfinite(w).all() or np.any(sd<.1) or np.any(sd>200) or np.max(np.abs(w))*1e6>500:
            rejected['quality']+=1; continue
        accepted.append(start)
    cap=int(320//seconds)
    if len(accepted)>cap:
        selected=np.linspace(0,len(accepted)-1,cap).round().astype(int)
        accepted=[accepted[i] for i in selected]
    if len(accepted)<3: raise ValueError(f'Only {len(accepted)} accepted {seconds}s windows')
    return np.stack([data[:,i:i+length] for i in accepted]),{'accepted':len(accepted),'seconds':len(accepted)*seconds,**rejected}


def spectral_power(windows,config):
    fs=config['sampling_hz']
    size=int(round(config['stft_seconds']*fs)); hop=int(round(config['hop_seconds']*fs))
    if hop<1 or hop>size or windows.shape[-1]<size+2*hop: raise ValueError('Require at least three STFT frames')
    freq,_,z=stft(windows,fs=fs,window='hann',nperseg=size,noverlap=size-hop,nfft=size,
                 detrend='constant',boundary=None,padded=False,axis=-1)
    keep=(freq>=1)&(freq<=40)
    return freq[keep],np.abs(z[:,:,keep,:])**2


def pool_power(frequencies,power,bins):
    values=np.log10(np.einsum('mf,wcft->wcmt',mel_bank(frequencies,bins),power,optimize=True)+1e-20)
    values-=np.median(values,axis=(1,2,3),keepdims=True)
    frames=values.transpose(0,3,1,2).reshape(-1,len(CHANNELS),bins)
    return np.stack([frames.mean(0),frames.std(0),np.median(frames,axis=0),
                     np.quantile(frames,.75,axis=0)-np.quantile(frames,.25,axis=0)],axis=-1).astype(np.float32)


def summarize_recording(data,intervals,config):
    if config['sampling_hz']==128: data=resample_poly(data,1,2,axis=-1).astype(np.float32)
    windows,audit=segment_windows(data,config['segment_seconds'],intervals,config['sampling_hz'])
    freq,power=spectral_power(windows,config)
    return pool_power(freq,power,config['mel_bins']),audit


def source_hashes(person):
    path=Path(person['path']); values={str(path):sha(path)}
    if path.with_suffix('.fdt').exists(): values[str(path.with_suffix('.fdt'))]=sha(path.with_suffix('.fdt'))
    return values


def prepare_person(index,person,reps,destination):
    from threadpoolctl import threadpool_limits
    destination=Path(destination); file=destination/'parts'/f'{index:03d}.npz'; info=file.with_suffix('.json')
    signature={'subject_id':person['subject_id'],'source_hashes':source_hashes(person),'reps':reps,
               'policy':POLICY,'features_code_sha256':sha(__file__)}
    if file.exists() and info.exists() and json.loads(info.read_text())['signature']==signature:
        return index,'cached'
    with threadpool_limits(limits=1):
        data,intervals,details=load_eeg(person)
        arrays={}; audits={}
        for fs,seconds in sorted({(r['sampling_hz'],r['segment_seconds']) for r in reps}):
            signal=data if fs==256 else resample_poly(data,1,2,axis=-1).astype(np.float32)
            windows,audit=segment_windows(signal,seconds,intervals,fs); audits[f'{fs}Hz_{seconds}s']=audit
            local=[r for r in reps if r['segment_seconds']==seconds and r['sampling_hz']==fs]
            for length,hop in sorted({(r['stft_seconds'],r['hop_seconds']) for r in local}):
                chosen=[r for r in local if (r['stft_seconds'],r['hop_seconds'])==(length,hop)]
                freq,power=spectral_power(windows,chosen[0])
                for config in chosen: arrays[key(config)]=pool_power(freq,power,config['mel_bins'])
        file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(file,**arrays)
        dump(info,{'signature':signature,'windows':audits,'annotation_intervals':intervals,'header':details})
    return index,'prepared'
