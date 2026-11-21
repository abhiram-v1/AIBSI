"""Fixed3x5 outer/3 inner participant CV; entire spectral feature search stays inside."""
import json
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from spectral_data import HERE,dump,sha
from fourier_model import ARMS,PARAMS,COUNTS,THRESHOLDS,metrics,select,train_predict

SEEDS=[2026110501,2026110502,2026110503]


def intervals(y,p,classes,other=None):
    rng=np.random.default_rng(2026110680);groups=[np.flatnonzero(y==c) for c in range(classes)];rows=[]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);r=[]
        for repeat in range(len(p)):
            a=metrics(y[ix],p[repeat,ix],classes);b=metrics(y[ix],other[repeat,ix],classes) if other is not None else None
            r.append([a[k]-(b[k] if b else 0) for k in ('macro_f1','balanced_accuracy')])
        rows.append(np.mean(r,axis=0))
    bounds=np.percentile(rows,[2.5,97.5],axis=0)
    return {k:bounds[:,i].tolist() for i,k in enumerate(('macro_f1','balanced_accuracy'))}


def main():
    out=HERE/'outputs';dest=out/'evaluation';dest.mkdir(parents=True,exist_ok=True);prep=out/'prepared'
    arrays=dict(np.load(prep/'features.npz'));ps=json.loads((prep/'plan.json').read_text())['participants'];dx=np.array([p['label'] for p in ps])
    names=json.loads((prep/'feature_names.json').read_text())
    protocol={'scope':'exploratory EEG frequency analysis, OpenNeuro only, all88 participants; no new independent cohort',
              'outer':'3repeats x5fold; stratified by original AD/FTD/HC; each participant held out once per repeat',
              'inner':'3fold; stratified original diagnosis; ANOVA ranking and StandardScaler trained within each fold',
              'seeds':SEEDS,'feature_families':ARMS,'feature_counts':COUNTS,'svm_grid':PARAMS,'binary_score_thresholds':THRESHOLDS,
              'class_weights':'balanced','pca':'none','selection':'macro F1, balanced accuracy, fewer features, smaller C; families chosen by inner scores only',
              'tasks':{'binary':['HC','AD_or_FTD'],'three_class':['HC','AD','FTD']},
              'statistics':'full-cohort descriptive associations NEVER used to choose classifier features',
              'clinical_metadata':'age,sex,MMSE excluded from classifier input',
              'features_sha256':sha(prep/'features.npz'),'plan_sha256':sha(prep/'plan.json'),
              'code_sha256':{f:sha(HERE/f) for f in ('spectral_data.py','prepare.py','fourier_model.py','evaluate.py')}}
    if (dest/'protocol.json').exists():
        if json.loads((dest/'protocol.json').read_text())!=protocol:raise RuntimeError('Do not mix changed protocols with existing folds')
    else:
        dump(dest/'protocol.json',protocol);snap=dest/'source_snapshot';snap.mkdir(exist_ok=True)
        for f in protocol['code_sha256']:shutil.copy2(HERE/f,snap/f)
    for task,labels in protocol['tasks'].items():
        classes=len(labels);y=np.array([0 if d=='HC' else 1 if classes==2 or d=='AD' else 2 for d in dx])
        pred={a:np.full((3,len(y)),-1,int) for a in (*ARMS,'selected')};records=[]
        for r,seed in enumerate(SEEDS):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(y,dx)):
                path=dest/f'{task}_{r}_{fold}.json'
                if path.exists():record=json.loads(path.read_text());assert record['train']==tr.tolist() and record['test']==te.tolist()
                else:
                    record={'task':task,'repeat':r,'fold':fold,'train':tr.tolist(),'test':te.tolist(),'arms':{}}
                    for arm in ARMS:
                        x=arrays[arm];best,trials=select(x[tr],y[tr],dx[tr],classes,seed+100+fold)
                        pp,ss,features=train_predict(x[tr],y[tr],x[te],best['config'],classes)
                        record['arms'][arm]={'best':best,'trials':trials,'prediction':pp.tolist(),'score':ss.tolist(),
                                             'selected_indices':features,'selected_feature_names':[names[arm][i] for i in features]}
                        print(f'{task} repeat{r+1}/3 fold{fold+1}/5 {arm} F1={metrics(y[te],pp,classes)["macro_f1"]:.3f}',flush=True)
                    record['selected_arm']=max(ARMS,key=lambda a:(record['arms'][a]['best']['macro_f1'],record['arms'][a]['best']['balanced_accuracy'],
                                                                                -record['arms'][a]['best']['config']['k']))
                    dump(path,record)
                for arm in ARMS:pred[arm][r,te]=record['arms'][arm]['prediction']
                pred['selected'][r,te]=record['arms'][record['selected_arm']]['prediction'];records.append(record)
        assert all(np.all(p>=0) for p in pred.values())
        summary={}
        for arm,p in pred.items():
            runs=[metrics(y,pp,classes) for pp in p]
            summary[arm]={'per_repeat':runs,'mean':{k:float(np.mean([v[k] for v in runs])) for k in ('macro_f1','balanced_accuracy','accuracy')},
                          'mean_per_class_recall':np.mean([v['per_class_recall'] for v in runs],axis=0).tolist(),
                          'bootstrap95':intervals(y,p,classes),'difference_vs_amplitude95':intervals(y,p,classes,pred['amplitude'])}
        dump(dest/f'{task}_summary.json',summary)
        np.savez_compressed(dest/f'{task}_predictions.npz',y=y,**pred)
        dump(dest/f'{task}_selection.json',[{'repeat':v['repeat'],'fold':v['fold'],'arm':v['selected_arm']} for v in records])
        print(task,'selected pipeline:',summary['selected']['mean'],flush=True)
    print('All nested Fourier classifier evaluations complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
