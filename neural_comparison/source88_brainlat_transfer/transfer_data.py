"""Exact old feature definitions, compatible units/order, fixed BrainLat mapping."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly,butter,sosfiltfilt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CHANNELS=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


V6=module('source88_reference_v6',ROOT/'tensor_pipeline/v6_binary/run.py')
V3=module('source88_reference_spectral',ROOT/'tensor_pipeline/v3_waveform/prepare.py')
V5=module('source88_reference_connectivity',ROOT/'tensor_pipeline/v5_connectivity/prepare_features.py')


def feature_banks(raw_windows,car_windows):
    spectral=V3.spectral_features(car_windows)
    raw=raw_windows.astype(float);raw-=raw.mean(axis=-1,keepdims=True)
    powers=[];connections=[]
    for band in V5.BANDS:
        filtered=sosfiltfilt(butter(4,band,btype='bandpass',fs=250,output='sos'),raw,axis=-1)
        _,conn,power=V5.band_features(filtered)
        connections.append(conn);powers.append(power)
    p=np.stack(powers)
    relative=p[:,:19]-np.log10(np.maximum(np.sum(10**p[:,:19],axis=0,keepdims=True),1e-12))
    return {'spectral':spectral,'power':np.concatenate([p.ravel(),relative.ravel()]),'connectivity':np.concatenate(connections)}


def discover():
    people=[]
    for folder,label in [('1_AD','AD'),('2_bvFTD','FTD'),('5_HC','HC')]:
        base=ROOT/'dataset/brainlat/EEG'/folder
        for file in sorted(base.rglob('*.set')):
            site=file.relative_to(base).parts[0];sid=file.parent.parent.name
            people.append({'subject_id':f'brainlat:{site}:{sid}','site':site,'label':label,'path':str(file)})
    if len({p['subject_id'] for p in people})!=len(people):raise ValueError('Duplicate target participant IDs')
    return people


def windows_from_recording(person,mapping,minimum_sd_uv=1.):
    raw=mne.io.read_raw_eeglab(person['path'],preload=True,verbose='ERROR')
    expected=mne.channels.make_standard_montage('biosemi128').ch_names
    if len(raw.ch_names)!=128 or set(raw.ch_names)!=set(expected):raise ValueError('Unexpected BrainLat channel set')
    raw.reorder_channels(expected)
    sf=float(raw.info['sfreq'])
    if sf!=512:raise ValueError('Frozen conversion expects512Hz native BrainLat EEG')
    cuts=[]
    for onset,duration,desc in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description):
        if 'boundary' in desc.lower() or desc.lower().startswith(('bad','edge')):
            onset=float(onset-raw.first_time);cuts.append((onset-.5,onset+max(float(duration),0)+.5))
    # The old classifier expects19 analysis channels. Keep the pre-existing
    # legacy spatial operator fixed; do not select a montage on target outcomes.
    mapped=mapping@raw.get_data();del raw
    # Old training arrays are MICROVOLTS at250Hz, not MNE volts.
    data=resample_poly(mapped*1e6,125,256,axis=-1).astype(np.float32)
    n=data.shape[-1]//1000
    windows=data[:,:n*1000].reshape(19,n,1000).transpose(1,0,2).copy()
    initial=np.isfinite(windows).all(axis=(1,2))&(np.max(np.abs(windows),axis=(1,2))<=500)&(windows.std(-1).min(1)>=.1)
    car=windows.copy();car-=car.mean(axis=1,keepdims=True);car-=car.mean(axis=2,keepdims=True)
    finite=np.isfinite(car).all(axis=(1,2));amplitude=np.max(np.abs(car),axis=(1,2))<400
    step=np.max(np.abs(np.diff(car,axis=-1)),axis=(1,2))<150;active=car.std(-1).min(1)>=minimum_sd_uv
    boundary=np.zeros(n,bool)
    for i in range(n):boundary[i]=any(i*4<b and i*4+4>a for a,b in cuts)
    keep=initial&finite&amplitude&step&active&~boundary
    audit={'native_hz':sf,'analysis_hz':250,'units':'microvolts','complete_four_second_windows':n,
           'retained_windows':int(keep.sum()),'retained_seconds':int(keep.sum()*4),'minimum_channel_sd_uv_used':minimum_sd_uv,
           'rejected_by_criterion':{'initial_qc':int((~initial).sum()),'nonfinite':int((~finite).sum()),
                                  'amplitude':int((~amplitude).sum()),'step':int((~step).sum()),
                                  'low_channel_sd':int((~active).sum()),'boundary':int(boundary.sum())}}
    if keep.sum()<3:raise ValueError('Fewer than3 source-policy-clean windows: '+json.dumps(audit))
    return windows[keep],car[keep],audit


def target_one(index,person,destination,mapping):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        try:
            a,b,audit=windows_from_recording(person,mapping)
            features=feature_banks(a,b)
            if not all(np.isfinite(x).all() for x in features.values()):raise ValueError('Non-finite feature vector')
            file=Path(destination)/'target_parts'/f'{index:03d}.npz';file.parent.mkdir(exist_ok=True)
            np.savez_compressed(file,**features)
            return {'index':index,**person,'status':'scored_features','audit':audit}
        except FileNotFoundError as error:
            return {'index':index,**person,'status':'missing_signal_file','reason':str(error)}
        except Exception as error:
            return {'index':index,**person,'status':'unscorable','reason':f'{type(error).__name__}: {error}'}


def check_source_replay():
    """Recompute all three banks for one original person from the old windows."""
    retained=json.loads((V3.OUT/'retained_windows.json').read_text())
    subjects,labels,banks=V6.load();person=subjects[0]
    rows=sorted([r for r in retained if r['subject']==person],key=lambda r:r['start_seconds'])
    arrays={o:np.load(ROOT/f'preprocessing/02_balanced_primary/{o}/signals.npy',mmap_mode='r') for o in ['primary','leftover']}
    raw=np.stack([arrays[r['origin']][r['source_index']] for r in rows]).astype(np.float32)
    car=raw.copy();car-=car.mean(axis=1,keepdims=True);car-=car.mean(axis=2,keepdims=True)
    replay=feature_banks(raw,car)
    errors={k:float(np.max(np.abs(replay[k]-banks[k][0]))) for k in replay}
    for k in errors:np.testing.assert_allclose(replay[k],banks[k][0],atol=1e-7,rtol=1e-7)
    return {'participant':str(person),'maximum_feature_errors':errors,'units':'microvolts','sampling_hz':250}
