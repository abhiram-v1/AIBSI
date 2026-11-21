"""Repeated nested participant CV, residual dataset probes, and source-only transfer."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from audit_data import HERE,dump,sha
from compat_model import ARMS,GRID,Transform,fit_svm,feature_key,metrics,select,train_predict

SEEDS=[2026110501,2026110502,2026110503]


def bootstrap(y,p,q=None):
    rng=np.random.default_rng(2026110577);values=[]
    # Resample participants together with all repeated held-out predictions.
    groups=[np.flatnonzero(y==c) for c in (0,1)]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);row=[]
        for r in range(p.shape[0]):
            a=metrics(y[ix],p[r,ix]);b=metrics(y[ix],q[r,ix]) if q is not None else None
            row.append([a[k]-(b[k] if b is not None else 0) for k in ('macro_f1','balanced_accuracy')])
        values.append(np.mean(row,axis=0))
    bounds=np.percentile(values,[2.5,97.5],axis=0)
    return {k:bounds[:,i].tolist() for i,k in enumerate(('macro_f1','balanced_accuracy'))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=3);args=parser.parse_args()
    dest=HERE/'outputs/evaluation';dest.mkdir(parents=True,exist_ok=True)
    prep=HERE/'outputs/prepared';plan=json.loads((prep/'plan.json').read_text());ps=plan['participants']
    arrays=dict(np.load(prep/'features.npz'));y=np.array([p['label']!='HC' for p in ps],int)
    sites=np.array([p['site'] for p in ps]);dx=np.array([p['label'] for p in ps]);domain=np.array([p['source']=='brainlat' for p in ps],int)
    strata=np.array([f'{s}_{d}' for s,d in zip(sites,dx)]);reps=args.repeats
    protocol={'seeds':SEEDS[:reps],'outer_folds':5,'inner_folds':3,'arms':ARMS,'grid':GRID,'thresholds':[-.5,0,.5],
              'pca_dimensions':20,'coral_covariance_shrinkage':.3,'primary_metric':'macro_f1','tie_break':'balanced_accuracy then smaller C',
              'harmonization_input':'site metadata only, never test diagnosis; all learned transforms fit on training participants',
              'domain_probe':'balanced linear and RBF SVM C1 after each transform; dataset0=OpenNeuro,1=BrainLat; site labels allowed in harmonizer',
              'transfer':'select deterministic arm and SVM within source cohort only; target never used to fit normalization or select settings',
              'caveat':'Both cohorts explored earlier; not an untouched prospective external validation',
              'features_sha256':sha(prep/'features.npz'),'plan_sha256':sha(prep/'plan.json')}
    sources=dest/'source_snapshot'
    protocol['code_sha256']={f:sha(HERE/f) for f in ('audit_data.py','prepare.py','compat_model.py','evaluate.py')}
    if (dest/'protocol.json').exists() and json.loads((dest/'protocol.json').read_text())!=protocol:
        raise ValueError('Protocol/code/features changed: use a fresh output directory instead of mixing runs')
    sources.mkdir(exist_ok=True)
    for f in ('audit_data.py','prepare.py','compat_model.py','evaluate.py'):
        shutil.copy2(HERE/f,sources/f)
    dump(dest/'protocol.json',protocol)
    predictions={a:np.full((reps,len(y)),-1,int) for a in (*ARMS,'selected')}
    scores={a:np.full((reps,len(y)),np.nan) for a in ARMS}
    probes={a:{k:np.full((reps,len(y)),-1,int) for k in ('linear','rbf')} for a in ARMS}
    conditional={a:{k:np.full((reps,len(y)),-1,int) for k in ('linear','rbf')} for a in ARMS}
    records=[]
    for r,seed in enumerate(SEEDS[:reps]):
        for f,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(y,strata)):
            record_path=dest/f'fold_{r}_{f}.json'
            if record_path.exists():
                rec=json.loads(record_path.read_text())
                if rec['train']!=train.tolist() or rec['test']!=test.tolist():raise ValueError('Stale folds')
                print(f'Cached repeat{r+1} fold{f+1}',flush=True)
            else:
                rec={'repeat':r,'fold':f,'train':train.tolist(),'test':test.tolist(),'arms':{}}
                for arm in ARMS:
                    x=arrays[feature_key(arm)];best,trials=select(x[train],y[train],sites[train],strata[train],arm,seed+100+f)
                    pred,score=train_predict(x[train],y[train],sites[train],x[test],sites[test],arm,best['config'])
                    trans=Transform(arm).fit(x[train],sites[train]);a=trans.transform(x[train],sites[train]);b=trans.transform(x[test],sites[test])
                    dp={k:fit_svm(a,domain[train],{'kernel':k,'C':1.}).predict(b).tolist() for k in ('linear','rbf')}
                    # Diagnostic-label-conditioned probes are descriptive tests, not deployable diagnosis models.
                    cp={k:np.full(len(test),-1,int) for k in ('linear','rbf')}
                    for label in ('AD','FTD','HC'):
                        tr=train[dx[train]==label];mask=dx[test]==label;te=test[mask]
                        t=Transform(arm).fit(x[tr],sites[tr]);aa=t.transform(x[tr],sites[tr]);bb=t.transform(x[te],sites[te])
                        for k in cp:cp[k][mask]=fit_svm(aa,domain[tr],{'kernel':k,'C':1.}).predict(bb)
                    rec['arms'][arm]={'best':best,'trials':trials,'prediction':pred.tolist(),'score':score.tolist(),
                                      'domain':dp,'conditional_domain':{k:v.tolist() for k,v in cp.items()}}
                    print(f'Repeat{r+1}/3 fold{f+1}/5 {arm}: F1={metrics(y[test],pred)["macro_f1"]:.3f}',flush=True)
                rec['selected_arm']=max(ARMS,key=lambda a:(rec['arms'][a]['best']['macro_f1'],rec['arms'][a]['best']['balanced_accuracy']))
                dump(record_path,rec)
            for arm in ARMS:
                entry=rec['arms'][arm];predictions[arm][r,test]=entry['prediction'];scores[arm][r,test]=entry['score']
                for kernel in ('linear','rbf'):
                    probes[arm][kernel][r,test]=entry['domain'][kernel]
                    conditional[arm][kernel][r,test]=entry['conditional_domain'][kernel]
            predictions['selected'][r,test]=rec['arms'][rec['selected_arm']]['prediction'];records.append(rec)
    summary={}
    for arm,pred in predictions.items():
        if (pred<0).any():raise ValueError('Missing predictions')
        runs=[metrics(y,p) for p in pred]
        summary[arm]={'per_repeat':runs,'mean':{k:float(np.mean([m[k] for m in runs])) for k in runs[0] if k!='confusion_matrix'},
                      'conditional_bootstrap95':bootstrap(y,pred),'difference_vs_raw95':bootstrap(y,pred,predictions['raw'])}
        if arm in ARMS:
            summary[arm]['domain_balanced_accuracy']={k:float(np.mean([metrics(domain,p)['balanced_accuracy'] for p in probes[arm][k]])) for k in ('linear','rbf')}
            summary[arm]['conditional_domain_balanced_accuracy']={d:{k:float(np.mean([metrics(domain[dx==d],p[dx==d])['balanced_accuracy'] for p in conditional[arm][k]])) for k in ('linear','rbf')} for d in ('AD','FTD','HC')}
    dump(dest/'summary.json',summary)
    np.savez_compressed(dest/'predictions.npz',y=y,domain=domain,**{f'diagnosis_{k}':v for k,v in predictions.items()},
                        **{f'domain_{a}_{k}':v for a in ARMS for k,v in probes[a].items()},
                        **{f'conditional_domain_{a}_{k}':v for a in ARMS for k,v in conditional[a].items()})
    # Entire target cohort held outside source-only model selection and transforms.
    transfer=[]
    for source in (0,1):
        tr=np.flatnonzero(domain==source);te=np.flatnonzero(domain!=source);entries={}
        for arm in ('raw','common','gain','relative'):
            x=arrays[arm];best,trials=select(x[tr],y[tr],sites[tr],strata[tr],arm,2026110590+source)
            pred,score=train_predict(x[tr],y[tr],sites[tr],x[te],sites[te],arm,best['config'])
            entries[arm]={'best':best,'metrics':metrics(y[te],pred),'prediction':pred.tolist(),'score':score.tolist(),'trials':trials}
        selected=max(entries,key=lambda a:(entries[a]['best']['macro_f1'],entries[a]['best']['balanced_accuracy']))
        transfer.append({'source':'OpenNeuro' if source==0 else 'BrainLat','training':tr.tolist(),'test':te.tolist(),
                         'selected_using_source_only':selected,'arms':entries})
        print(f'Transfer source{source} selected{selected}: {entries[selected]["metrics"]}',flush=True)
    dump(dest/'transfer.json',transfer)
    dump(dest/'selected_arms.json',[{'repeat':r['repeat'],'fold':r['fold'],'arm':r['selected_arm']} for r in records])
    print('Nested evaluation and source-only transfer complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
