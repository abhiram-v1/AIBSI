"""Matched BrainLat montage ablation; participant-local, no diagnosis-derived features."""
from __future__ import annotations

import contextlib
import hashlib
import json
import warnings
from pathlib import Path

import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix
from mne.bem import fit_sphere_to_headshape
from scipy.optimize import linear_sum_assignment
from scipy.signal import stft

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COHORT=ROOT/'neural_comparison/binary_logmel_search/outputs/prepared/plan.json'
ARMS=['native128','native19','legacy19','aligned19']
TARGET=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
POLICY={'sampling_hz':512,'filter_hz':[1,40],'reference':'native128 average before branching',
        'segment_seconds':8,'stft_seconds':1,'hop_seconds':1,'mel_bins':32,
        'segment_overlap':0,'maximum_seconds':320,'minimum_windows':3,'edge_seconds':10,
        'boundary_guard_seconds':2,'quality':'all native channels SD 0.1-200 uV, abs <=500 uV; common windows in all arms',
        'log_gain':'common per-window median of native128 log-mel applied to all four arms',
        'statistics':['mean','SD','median','IQR'],'spline_regularization':0.0,
        'scope':'86 already-included BrainLat people, internal binary HC versus AD+FTD validation'}


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def geometry():
    """Explicit source/target frames; also reproduce installed MNE legacy mapping."""
    source=mne.channels.make_standard_montage('biosemi128')
    source_raw=mne.io.RawArray(np.eye(128),mne.create_info(source.ch_names,512,'eeg'),verbose='ERROR')
    source_raw.set_montage(source)
    target=mne.channels.make_standard_montage('standard_1020')
    target_raw=mne.io.RawArray(np.eye(19),mne.create_info(TARGET,512,'eeg'),verbose='ERROR')
    target_raw.set_montage(target)
    src=np.array([c['loc'][:3] for c in source_raw.info['chs']])
    dst=np.array([c['loc'][:3] for c in target_raw.info['chs']])
    with warnings.catch_warnings(),contextlib.redirect_stdout(__import__('sys').stderr):
        warnings.simplefilter('ignore',RuntimeWarning)
        _,origin,_=fit_sphere_to_headshape(source_raw.info,verbose='ERROR')
        original_positions=target.get_positions()
        legacy_montage=mne.channels.make_dig_montage(ch_pos={c:original_positions['ch_pos'][c] for c in TARGET},coord_frame=original_positions['coord_frame'])
        legacy=source_raw.interpolate_to(legacy_montage,method='spline').get_data()
    # Spherical splines need BOTH sensor sets in the same frame and sphere origin.
    aligned=_make_interpolation_matrix(src-origin,dst-origin,alpha=0.)
    unit=lambda x:x/np.linalg.norm(x,axis=1,keepdims=True)
    distances=np.arccos(np.clip(unit(dst-origin)@unit(src-origin).T,-1,1))
    rows,cols=linear_sum_assignment(distances)
    selected=np.zeros((19,128));selected[rows,cols]=1
    mapping={'native128':np.eye(128),'native19':selected,'legacy19':legacy,'aligned19':aligned}
    metadata={'source_channels':source.ch_names,'target_channels':TARGET,
              'source_head_positions_m':src.tolist(),'target_head_positions_m':dst.tolist(),'sphere_origin_head_m':origin.tolist(),
              'native19_assignment':[{'target':TARGET[i],'native':source.ch_names[j],'angular_distance_degrees':float(np.degrees(distances[i,j]))} for i,j in zip(rows,cols)],
              'legacy_target_coordinate_frame':original_positions['coord_frame'],
              'aligned_target_coordinate_frame':'head, with same sphere origin subtracted from both layouts',
              'legacy_mapping':'installed MNE interpolate_to on standard1020 target positions, reproducing earlier spatial mapping; common128 reference here',
              'operator_audit':{a:{'shape':list(w.shape),'rank':int(np.linalg.matrix_rank(w)),
                    'maximum_absolute_coefficient':float(np.abs(w).max()),'maximum_row_abs_sum':float(np.abs(w).sum(1).max()),
                    'constant_preservation_error':float(np.max(np.abs(w.sum(1)-1)))} for a,w in mapping.items()}}
    return mapping,metadata


def native_recording(person):
    raw=mne.io.read_raw_eeglab(person['path'],preload=True,verbose='ERROR')
    expected=mne.channels.make_standard_montage('biosemi128').ch_names
    if set(raw.ch_names)!=set(expected) or len(raw.ch_names)!=128:raise ValueError('Unexpected native channels')
    if raw.info['sfreq']!=512:raise ValueError('Protocol requires native512 Hz; do not silently resample')
    raw.reorder_channels(expected);raw.set_montage('biosemi128')
    annotations=[]
    for onset,duration,desc in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description):
        if 'boundary' in desc.lower() or desc.lower().startswith(('bad','edge')):
            onset=float(onset-raw.first_time);annotations.append((onset-2,onset+max(float(duration),0)+2))
    source_bads=list(raw.info['bads'])
    raw.set_eeg_reference('average',projection=False,verbose='ERROR')
    raw.filter(1,40,picks='eeg',verbose='ERROR')
    data=raw.get_data().astype(np.float32)
    return data,annotations,{'original_hz':float(raw.info['sfreq']),'source_bad_channel_flags':source_bads,'duration_seconds':data.shape[1]/512}


def accepted_windows(data,annotations):
    fs=512;length=8*fs;starts=[];rejected={'boundary':0,'quality':0}
    for start in range(10*fs,data.shape[1]-10*fs-length+1,length):
        if any(start/fs<b and (start+length)/fs>a for a,b in annotations):rejected['boundary']+=1;continue
        w=data[:,start:start+length];sd=w.std(1)*1e6
        if not np.isfinite(w).all() or np.any(sd<.1) or np.any(sd>200) or np.max(np.abs(w))*1e6>500:
            rejected['quality']+=1;continue
        starts.append(start)
    if len(starts)>40:starts=[starts[i] for i in np.linspace(0,len(starts)-1,40).round().astype(int)]
    if len(starts)<3:raise ValueError(f'Only {len(starts)} accepted native128 windows; cohort must not silently change')
    return np.stack([data[:,i:i+length] for i in starts]),{'starts_samples':starts,'accepted':len(starts),'accepted_seconds':8*len(starts),**rejected}


def mel_bank(freq):
    mel=lambda x:2595*np.log10(1+x/700)
    points=700*(10**(np.linspace(mel(1),mel(40),34)/2595)-1)
    bank=np.maximum(0,np.minimum((freq[None]-points[:-2,None])/(points[1:-1]-points[:-2])[:,None],
             (points[2:,None]-freq[None])/(points[2:]-points[1:-1])[:,None]))
    return bank/bank.sum(1,keepdims=True)


def logmel(windows):
    freq,_,z=stft(windows,fs=512,window='hann',nperseg=512,noverlap=0,nfft=512,detrend='constant',boundary=None,padded=False,axis=-1)
    keep=(freq>=1)&(freq<=40)
    return np.log10(np.einsum('mf,wcft->wcmt',mel_bank(freq[keep]),np.abs(z[:,:,keep])**2,optimize=True)+1e-20)


def pool(values):
    frame=values.transpose(0,3,1,2).reshape(-1,values.shape[1],32)
    return np.stack([frame.mean(0),frame.std(0),np.median(frame,axis=0),
                     np.quantile(frame,.75,axis=0)-np.quantile(frame,.25,axis=0)],axis=-1).astype(np.float32)


def extract(person,mapping):
    data,annotations,header=native_recording(person)
    windows,audit=accepted_windows(data,annotations)
    native=logmel(windows);gain=np.median(native,axis=(1,2,3),keepdims=True)
    tensors={'native128':pool(native-gain)}
    for arm in ARMS[1:]:
        transformed=np.einsum('kc,wct->wkt',mapping[arm],windows,optimize=True).astype(np.float32)
        tensors[arm]=pool(logmel(transformed)-gain)
    return tensors,{'header':header,'windows':audit,'annotations':annotations}


def prepare_one(index,person,mapping,destination,signature):
    from threadpoolctl import threadpool_limits
    path=Path(destination)/'parts'/f'{index:03d}.npz';meta=path.with_suffix('.json')
    src={person['path']:sha(person['path'])};fdt=Path(person['path']).with_suffix('.fdt')
    if fdt.exists():src[str(fdt)]=sha(fdt)
    token={**signature,'subject_id':person['subject_id'],'source_hashes':src}
    if path.exists() and meta.exists() and json.loads(meta.read_text())['signature']==token:return index,'cached'
    with threadpool_limits(limits=1):
        tensors,audit=extract(person,mapping)
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**tensors)
        dump(meta,{'signature':token,**audit})
    return index,'prepared'
