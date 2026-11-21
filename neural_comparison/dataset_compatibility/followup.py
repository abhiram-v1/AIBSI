"""Exploratory follow-up prompted by the observed BrainLat label/amplitude gap.

Do not merge these results with the predeclared harmonization selection endpoint.
"""
import json
import re
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import mne
from pymatreader import read_mat
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from audit_data import HERE,dump,sha
from compat_model import metrics,select,train_predict


def header_native(p):
    with threadpool_limits(limits=1):
        d=read_mat(p['path']);d=d.get('EEG',d);sf=float(d['srate']);channels=int(d['nbchan'])
        # One fixed60s segment, independent of diagnosis and of the interpolated signals.
        start=int(10*sf);stop=min(int(70*sf),int(d['pnts']))
        if isinstance(d['data'],str):
            path=Path(p['path']).parent/d['data']
            arr=np.memmap(path,dtype='<f4',mode='r',shape=(int(d['pnts']),channels)).T
        else:arr=np.asarray(d['data'])
        x=np.array(arr[:,start:stop],dtype=float)*1e-6
        if x.shape[0]!=channels:raise ValueError('Unexpected native data shape')
        car=x-x.mean(0,keepdims=True);car=mne.filter.filter_data(car,sf,1,40,verbose='ERROR')[:,int(2*sf):-int(2*sf)]
        rms=float(np.median(np.sqrt(np.mean(car*car,axis=1)))*1e6)
        etc=d.get('etc',{});etc=etc if isinstance(etc,dict) else {}
        return {'subject_id':p['subject_id'],'site':p['site'],'label':p['label'],'source':p['source'],
                'native_channels':channels,'native_hz':sf,'header_reference':str(d.get('ref','unknown')),
                'header_eeglab_version':str(etc.get('eeglabvers','unknown')),
                'history_eeglab_versions':re.findall(r"eeglabvers\s*=\s*'([^']+)'",str(d.get('history',''))),
                'native_car_rms_uv_fixed56s':rms,
                'native_segment_policy':'original native channels;10–70s,1–40Hz FIR and native-channel CAR, discard2s at each edge; no interpolation or label-dependent selection',
                'channel_unit_fields':{k:str(d.get('chanlocs',{}).get(k,'ABSENT'))[:100] for k in ('unit','units')} if isinstance(d.get('chanlocs'),dict) else {},
                'root_unit_fields':{k:str(d.get(k,'ABSENT'))[:100] for k in ('unit','units')}}


def main():
    dest=HERE/'outputs/followup';dest.mkdir(parents=True,exist_ok=True)
    prep=HERE/'outputs/prepared';ps=json.loads((prep/'plan.json').read_text())['participants']
    policy={'status':'exploratory follow-up prompted by amplitude separation observed in the main audit',
            'native_audit':'fixed60s native montage segment per record; no interpolation; metadata versions are provenance clues, not causal proof',
            'probe':'single log10 matched median-channel RMS feature, same nested SVM grid and thresholds as main audit',
            'cv':'3repeats5outer3inner, stratify site x original diagnosis; separate OpenNeuro,BrainLat and pooled runs',
            'code_sha256':sha(__file__),'compat_model_sha256':sha(HERE/'compat_model.py')}
    dump(dest/'protocol.json',policy)
    native_path=dest/'native_metadata.json'
    if native_path.exists():native=json.loads(native_path.read_text())
    else:
        native=[None]*len(ps)
        with ProcessPoolExecutor(2) as pool:
            pending={pool.submit(header_native,p):i for i,p in enumerate(ps)}
            for n,future in enumerate(as_completed(pending),1):
                native[pending[future]]=future.result()
                if n%10==0 or n==len(ps):print(f'Native montage/header audit{n}/{len(ps)}',flush=True)
        dump(native_path,native)
    x=np.log10(np.array([json.loads((prep/'parts'/f'{i:03d}.json').read_text())['common_metrics']['median_channel_rms_uv'] for i in range(len(ps))]))[:,None]
    y=np.array([p['label']!='HC' for p in ps],int);sites=np.array([p['site'] for p in ps]);dx=np.array([p['label'] for p in ps])
    source=np.array([p['source'] for p in ps]);strata=np.array([f'{s}_{d}' for s,d in zip(sites,dx)])
    results={}
    for name,ix in [('OpenNeuro',np.flatnonzero(source=='ds004504')),('BrainLat',np.flatnonzero(source=='brainlat')),('Pooled',np.arange(len(ps)))]:
        allpred=np.full((3,len(ix)),-1,int);records=[]
        for r,seed in enumerate([2026110501,2026110502,2026110503]):
            for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(ix,strata[ix])):
                train=ix[tr];test=ix[te];best,_=select(x[train],y[train],sites[train],strata[train],'raw',seed+100+f)
                prediction,score=train_predict(x[train],y[train],sites[train],x[test],sites[test],'raw',best['config'])
                allpred[r,te]=prediction;records.append({'repeat':r,'fold':f,'training':train.tolist(),'test':test.tolist(),'best':best,'prediction':prediction.tolist()})
        runs=[metrics(y[ix],p) for p in allpred]
        results[name]={'indices':ix.tolist(),'per_repeat':runs,'mean':{k:float(np.mean([m[k] for m in runs])) for k in runs[0] if k!='confusion_matrix'},'records':records}
        print(f'{name} single RMS-feature diagnosis probe: {results[name]["mean"]}',flush=True)
    dump(dest/'amplitude_probe.json',results)
    print('Exploratory native-montage and amplitude-only audit complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
