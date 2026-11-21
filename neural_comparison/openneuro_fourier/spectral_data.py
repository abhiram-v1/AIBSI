"""Fourier/Welch power features on fixed participant-level OpenNeuro recordings."""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.signal import welch

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent/'dataset_compatibility'
sys.path.insert(0,str(AUDIT))
from audit_data import CHANNELS,load_record,accepted,dump,sha

BANDS=[('delta_1_4',1,4),('theta_4_8',4,8),('alpha_8_13',8,13),('beta_13_30',13,30),('high_30_40',30,40.01)]
DESCRIPTORS=[*[f'log_power_{b}' for b,_,_ in BANDS],*[f'relative_{b}' for b,_,_ in BANDS],
             'log_slow_fast_ratio','log_theta_alpha_ratio','alpha_peak_7_13_hz','centroid_hz',
             'spectral_entropy','spectral_edge95_hz','log_total_power','aperiodic_exponent_proxy','alpha_above_background_db']
RELATIVE_INDICES=[i for i,n in enumerate(DESCRIPTORS) if not n.startswith('log_power_') and n!='log_total_power']
REGIONS={'frontal':['Fp1','Fp2','F3','F4','F7','F8','Fz'],'central':['C3','C4','Cz'],
         'temporal':['T7','T8','P7','P8'],'parieto_occipital':['P3','P4','Pz','O1','O2']}
POLICY={'source':'88 OpenNeuro supplied derivatives, no BrainLat participants',
        'signal':'same prior compatibility policy:256Hz,19-channel CAR,1–40Hz FIR; no new ICA',
        'windows':'same8s nonoverlapping accepted windows, edge10s/boundary2s guard, at most40/person',
        'qc':'all-channel SD0.1–200uV, absolute amplitude<=500uV, finite, at least3 windows',
        'fourier':'Welch averaged Hann periodograms,4s windows/2s hop,1024-point FFT,0.25Hz bins',
        'frequency_range_hz':[1,40],'delta_definition':'1–4Hz; below1Hz is outside this fixed processing policy',
        'power_units':'uV²/Hz, using EEGLAB/MNE voltage convention',
        'pooling':'mean PSD over Welch subwindows and accepted8s windows; one sample per participant',
        'background':'log-log least-squares slope over2–30Hz excluding7–13Hz; descriptive proxy, not FOOOF or source separation',
        'clinical_inputs':'age/sex used only for descriptive adjusted analyses; MMSE only severity correlation, never classifier input'}


def descriptors(psd,freq):
    """psd[...,frequency], either channel, region, or global spectra."""
    df=float(freq[1]-freq[0]);p=np.maximum(psd,1e-20)
    bp=np.stack([p[..., (freq>=a)&(freq<b)].sum(-1)*df for _,a,b in BANDS],axis=-1)
    total=bp.sum(-1);relative=bp/total[...,None]
    density=p/p.sum(-1,keepdims=True)
    alpha=(freq>=7)&(freq<=13)
    peak=freq[alpha][np.argmax(p[...,alpha],axis=-1)]
    entropy=-(density*np.log(density)).sum(-1)/np.log(len(freq))
    centroid=(density*freq).sum(-1)
    edge=freq[np.argmax(np.cumsum(density,axis=-1)>=.95,axis=-1)]
    fit=(freq>=2)&(freq<=30)&~alpha
    design=np.column_stack([np.ones(fit.sum()),np.log10(freq[fit])])
    coef=np.log10(p[...,fit])@np.linalg.pinv(design).T
    background=coef[...,0,None]+coef[...,1,None]*np.log10(freq[alpha])
    prominence=10*np.max(np.log10(p[...,alpha])-background,axis=-1)
    result=np.concatenate([np.log10(bp),relative,np.stack([
        np.log10((bp[...,0]+bp[...,1])/(bp[...,2]+bp[...,3])),np.log10(bp[...,1]/bp[...,2]),peak,
        centroid,entropy,edge,np.log10(total),-coef[...,1],prominence],axis=-1)],axis=-1)
    assert result.shape[-1]==len(DESCRIPTORS)
    return result


def prepare_one(i,person,destination):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        dest=Path(destination);path=dest/'parts'/f'{i:03d}.npz';meta=path.with_suffix('.json')
        signature={'policy':POLICY,'code_sha256':sha(__file__),'audit_data_sha256':sha(AUDIT/'audit_data.py'),
                   'source_sha256':sha(person['path'])}
        fdt=Path(person['path']).with_suffix('.fdt')
        if fdt.exists():signature['source_fdt_sha256']=sha(fdt)
        if path.exists() and meta.exists() and json.loads(meta.read_text())['signature']==signature:return i,'cached'
        acquisition,common,annotations,header=load_record(person);del acquisition
        starts,audit=accepted(common,annotations)
        x=np.stack([common[:,s:s+2048] for s in starts]).astype(float)*1e6
        freq,psd=welch(x,fs=256,window='hann',nperseg=1024,noverlap=512,nfft=1024,detrend='constant',scaling='density',axis=-1)
        keep=(freq>=1)&(freq<=40);freq=freq[keep];psd=psd[:,:,keep].mean(0)
        total=psd.sum(1,keepdims=True)*.25;relative=psd/total
        summary=descriptors(psd,freq);global_summary=descriptors(psd.mean(0),freq)
        regional=np.stack([descriptors(psd[[CHANNELS.index(ch) for ch in names]].mean(0),freq) for names in REGIONS.values()])
        rms=float(np.median(np.sqrt(np.mean(x*x,axis=(0,2)))))
        arrays={'psd':psd,'relative_psd':relative,'frequency':freq,'summary_tensor':summary,
                'summary':summary.ravel(),'relative_summary':summary[:,RELATIVE_INDICES].ravel(),
                'absolute_spectrum':np.log10(np.maximum(psd,1e-20)).ravel(),
                'relative_spectrum':np.log10(np.maximum(relative,1e-20)).ravel(),
                'amplitude':np.array([np.log10(rms)]),'global_summary':global_summary,'regional_summary':regional}
        assert all(np.isfinite(a).all() for a in arrays.values())
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
        dump(meta,{'signature':signature,'participant':person,'header':header,'windows':audit,'rms_uv':rms,'starts_samples':starts})
        return i,'prepared'


def feature_names(frequency):
    return {'amplitude':['global:log10_median_channel_RMS_uv'],
            'summary':[f'{ch}:{name}' for ch in CHANNELS for name in DESCRIPTORS],
            'relative_summary':[f'{ch}:{DESCRIPTORS[i]}' for ch in CHANNELS for i in RELATIVE_INDICES],
            'absolute_spectrum':[f'{ch}:log_PSD_{f:g}Hz' for ch in CHANNELS for f in frequency],
            'relative_spectrum':[f'{ch}:log_relative_PSD_{f:g}Hz' for ch in CHANNELS for f in frequency]}
