"""Participant-level paired signal audit and deterministic common EEG preparation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly,stft,welch
from scipy.stats import kurtosis

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
COHORT=ROOT/'neural_comparison/binary_logmel_search/outputs/prepared/plan.json'
OPERATORS=ROOT/'neural_comparison/brainlat_native_ablation/outputs/prepared/operators.npz'
CHANNELS=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
BANDS=[(1,4),(4,8),(8,13),(13,30),(30,40.01)]
POLICY={'fs':256,'channels':CHANNELS,'brainlat_mapping':'fixed pre-existing legacy19; operator hash saved',
        'common_reference':'mean of the same19 analysis channels in both cohorts',
        'filter_hz':[1,40],'filter_order':'same MNE FIR at256Hz AFTER anti-aliased resampling in both cohorts',
        'epoch_seconds':8,'epoch_overlap':0,'maximum_seconds':320,'edge_seconds':10,'boundary_guard_seconds':2,
        'qc':'common-reference filtered signals: all channel SD0.1–200uV; max abs500uV; finite; min3 windows',
        'spectra':'2-second Hann STFT,0.5-second hop,24unit-sum mel bands1–40Hz',
        'features':'per-person mean and SD over all frames;19x24x2=912 features',
        'upstream':'existing derivatives; prior ICA/ASR/filter differences cannot be undone'}


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def people():
    subjects=json.loads(COHORT.read_text())['participants']
    original=pd.read_csv(ROOT/'dataset/ds004504/participants.tsv',sep='\t').set_index('participant_id')
    demographics={}
    for folder,name in [('1_AD','AD'),('2_bvFTD','bvFTD'),('5_HC','HC')]:
        frame=pd.read_csv(ROOT/f'dataset/brainlat/EEG/{folder}/Demographics_{name}_EEG_data.csv')
        for _,r in frame.iterrows():
            key=(str(r['path']).split('/')[-1],str(r['id EEG']))
            age=pd.to_numeric(r['Age'],errors='coerce')
            demographics.setdefault(key,[]).append(float(age) if pd.notna(age) else None)
    result=[]
    for p in subjects:
        sid=p['subject_id'].split(':')[-1]
        if p['source']=='ds004504':age=float(original.loc[sid,'Age'])
        else:
            values={v for v in demographics.get((p['site'],sid),[]) if v is not None}
            age=next(iter(values)) if len(values)==1 else None
        result.append({**p,'age':age})
    if len(result)!=174:raise ValueError('Expected fixed174-person cohort')
    return result


def load_record(p):
    raw=mne.io.read_raw_eeglab(p['path'],preload=True,verbose='ERROR')
    annotations=[]
    for onset,duration,desc in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description):
        if 'boundary' in desc.lower() or desc.lower().startswith(('bad','edge')):
            onset=float(onset-raw.first_time);annotations.append((onset-2,onset+max(float(duration),0)+2))
    sf=float(raw.info['sfreq']);header={'native_hz':sf,'native_channels':len(raw.ch_names),'seconds':raw.n_times/sf,'source_bad_flags':list(raw.info['bads'])}
    if p['source']=='brainlat':
        names=mne.channels.make_standard_montage('biosemi128').ch_names
        if set(names)!=set(raw.ch_names):raise ValueError('BrainLat montage differs')
        raw.reorder_channels(names);mapped=np.load(OPERATORS)['legacy19']@raw.get_data()
    else:
        aliases={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
        raw.rename_channels({k:v for k,v in aliases.items() if k in raw.ch_names})
        raw.pick(CHANNELS);raw.reorder_channels(CHANNELS);mapped=raw.get_data()
    del raw
    from fractions import Fraction
    if sf!=int(sf):raise ValueError('Expected integer acquisition sampling rate')
    # Use exact integer ratio; binary float Fraction otherwise creates huge factors.
    ratio=Fraction(256,int(sf))
    acquisition=resample_poly(mapped,ratio.numerator,ratio.denominator,axis=-1)
    car=acquisition-acquisition.mean(0,keepdims=True)
    common=mne.filter.filter_data(car,256,1,40,verbose='ERROR')
    return acquisition.astype(np.float32),common.astype(np.float32),annotations,header


def accepted(common,annotations):
    length=2048;starts=[];rejected={'boundary':0,'quality':0}
    for start in range(2560,common.shape[1]-2560-length+1,length):
        if any(start/256<b and (start+length)/256>a for a,b in annotations):rejected['boundary']+=1;continue
        w=common[:,start:start+length];sd=w.std(1)*1e6
        if not np.isfinite(w).all() or np.any(sd<.1) or np.any(sd>200) or np.max(np.abs(w))*1e6>500:
            rejected['quality']+=1;continue
        starts.append(start)
    if len(starts)>40:starts=[starts[i] for i in np.linspace(0,len(starts)-1,40).round().astype(int)]
    if len(starts)<3:raise ValueError('Fewer than3 common-policy epochs; do not change cohort silently')
    return starts,{'accepted_windows':len(starts),'accepted_seconds':8*len(starts),**rejected}


def bank(freq):
    mel=lambda x:2595*np.log10(1+x/700)
    points=700*(10**(np.linspace(mel(1),mel(40),26)/2595)-1)
    w=np.maximum(0,np.minimum((freq[None]-points[:-2,None])/(points[1:-1]-points[:-2])[:,None],
                              (points[2:,None]-freq[None])/(points[2:]-points[1:-1])[:,None]))
    return w/w.sum(1,keepdims=True)


def spectrum(windows):
    # Work in microvolts for interpretable absolute powers and RMS.
    x=windows.astype(float)*1e6
    f,_,z=stft(x,fs=256,window='hann',nperseg=512,noverlap=384,nfft=512,detrend='constant',boundary=None,padded=False,axis=-1)
    keep=(f>=1)&(f<=40);mel=np.einsum('mf,wcft->wcmt',bank(f[keep]),np.abs(z[:,:,keep])**2,optimize=True)
    log=np.log10(mel+1e-20)
    freq,psd=welch(x,fs=256,nperseg=512,noverlap=256,nfft=512,axis=-1)
    mean_psd=psd.mean(0);bp=np.stack([mean_psd[:,(freq>=lo)&(freq<hi)].sum(1)*.5 for lo,hi in BANDS],axis=-1)
    selected=(freq>=1)&(freq<=40);profile=mean_psd[:,selected].mean(0)
    flat=x.transpose(1,0,2).reshape(19,-1)
    corr=np.corrcoef(flat);edges=corr[np.triu_indices(19,1)]
    total=np.maximum(np.sum(mean_psd[:,selected],axis=1)*.5,1e-20)
    line={str(hz):float(np.median(mean_psd[:,(freq>=hz-1)&(freq<=hz+1)].sum(1)*.5/total)) for hz in [50,60]}
    common_var=float(np.var(flat.mean(0)));mean_var=float(np.mean(np.var(flat,axis=1)))
    metrics={'median_channel_rms_uv':float(np.median(np.sqrt(np.mean(x*x,axis=(0,2))))),
             'median_channel_sd_uv':float(np.median(flat.std(1))),
             'common_mode_variance_fraction':common_var/max(mean_var,1e-20),
             'median_pair_correlation':float(np.median(edges)),
             'median_excess_kurtosis':float(np.median(kurtosis(x,axis=-1,fisher=True))),
             'high_frequency_30_40_fraction':float(np.median(bp[:,-1]/np.maximum(bp.sum(1),1e-20))),
             'residual_line50_ratio':line['50'],'residual_line60_ratio':line['60'],
             'windows_below_old1uv_floor':int(np.sum(x.std(-1).min(1)<1.))}
    return log,bp,profile,metrics


def pool(values):
    frames=values.transpose(0,3,1,2).reshape(-1,19,24)
    return np.stack([frames.mean(0),frames.std(0)],axis=-1).ravel().astype(np.float32)


def prepare_one(index,p,destination):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        dest=Path(destination);path=dest/'parts'/f'{index:03d}.npz';meta=path.with_suffix('.json')
        source_hash={p['path']:sha(p['path'])};fdt=Path(p['path']).with_suffix('.fdt')
        if fdt.exists():source_hash[str(fdt)]=sha(fdt)
        signature={'policy':POLICY,'data_code_sha256':sha(__file__),'source_hashes':source_hash,'operators_sha256':sha(OPERATORS)}
        if path.exists() and meta.exists() and json.loads(meta.read_text())['signature']==signature:return index,'cached'
        raw,common,annotations,header=load_record(p);starts,audit=accepted(common,annotations)
        a=np.stack([raw[:,i:i+2048] for i in starts]);b=np.stack([common[:,i:i+2048] for i in starts])
        rawlog,rawbp,rawprofile,rawmetrics=spectrum(a)
        log,bp,profile,metrics=spectrum(b)
        gain=log-np.median(log,axis=(1,2,3),keepdims=True)
        # Relative per-channel mel power. No across-participant estimates.
        from scipy.special import logsumexp
        relative=log-logsumexp(log*np.log(10),axis=2,keepdims=True)/np.log(10)
        arrays={'raw':pool(rawlog),'common':pool(log),'gain':pool(gain),'relative':pool(relative),
                'raw_bandpower':rawbp,'common_bandpower':bp,'raw_profile':rawprofile,'common_profile':profile}
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite audit output')
        path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**arrays)
        dump(meta,{'signature':signature,'participant':p,'header':header,'windows':audit,'raw_metrics':rawmetrics,'common_metrics':metrics})
        return index,'prepared'
