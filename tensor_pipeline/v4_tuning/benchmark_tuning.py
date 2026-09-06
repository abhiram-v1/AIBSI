"""Frozen nested classifier/tensor regularization comparison, same V3 inputs."""
from pathlib import Path
import json
import hashlib
import time
import joblib
import numpy as np
import xgboost
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from threadpoolctl import threadpool_limits
from tune_models import configs,feature_bank,fit_classifier,fit_supervised,scores,group_rep,weights
from run import load_inputs,metric
from prepare import ROOT,OUT as DATA,sha

OUT=ROOT/'tensor_pipeline/outputs/v4_tuning'
KINDS=['logistic','svm','xgboost','random_forest','supervised_tt']
REPS=['wave_tt','wave_tucker','spectral','fusion']


def signature():
    h=hashlib.sha256()
    for p in [Path(__file__),Path(__file__).with_name('tune_models.py'),DATA/'input_fingerprints.json',
              ROOT/'tensor_pipeline/v3_waveform/wave_models.py']:
        h.update(p.read_bytes())
    h.update(xgboost.__version__.encode())
    return h.hexdigest()[:16]


def rank_rows(rows):
    # Stronger regularization/smaller models win exact metric ties only.
    def key(r):
        c=r['config']
        complexity=(KINDS.index(c['kind']),c.get('channel_rank',0),c.get('max_depth') or 0,
                    c.get('n_estimators',0),c.get('selection') or 0,c.get('C',0))
        return (-r['metrics']['balanced_accuracy'],-r['metrics']['macro_f1'],complexity,json.dumps(c,sort_keys=True))
    return sorted(rows,key=key)


def select(rows):
    ranked=rank_rows(rows)
    result={'inner_selected_pipeline':ranked[0]}
    result.update({'clf_'+k:next(r for r in ranked if r['config']['kind']==k) for k in KINDS})
    result.update({'rep_'+k:next(r for r in ranked if group_rep(r['config'])==k) for k in REPS})
    return result


def summarize(folds):
    results={}
    if folds:
        for name in folds[0]['predictions']:
            reps=[]
            for repeat in range(3):
                fs=[f for f in folds if f['repeat']==repeat]
                if len(fs)!=5: continue
                truth=np.concatenate([f['truth'] for f in fs]); pred=np.concatenate([f['predictions'][name]['predictions'] for f in fs])
                reps.append(metric(truth,pred))
            if reps:
                results[name]=dict(repeats=reps,mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reps])),
                                   sd_balanced_accuracy=float(np.std([r['balanced_accuracy'] for r in reps],ddof=1)) if len(reps)>1 else None,
                                   mean_accuracy=float(np.mean([r['accuracy'] for r in reps])),
                                   mean_macro_f1=float(np.mean([r['macro_f1'] for r in reps])))
    r=dict(signature=signature(),completed_folds=len(folds),results=results,total_seconds=sum(f['seconds'] for f in folds))
    (OUT/'nested_report.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r


def main():
    x,controls,subjects,y=load_inputs()
    specs=configs(); sig=signature()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True); (OUT/'bases').mkdir(exist_ok=True)
    protocol=dict(signature=sig,date='2026-09-06',input_directory=str(DATA),input_signature=sha(DATA/'input_fingerprints.json'),
                  xgboost_version=xgboost.__version__,subjects=subjects.tolist(),class_counts=np.bincount(y).tolist(),
                  outer=dict(folds=5,repeats=3,seed=9060),inner=dict(folds=3,seed='90600+outer_index'),
                  candidates=specs,candidate_count=len(specs),
                  primary_endpoint='Mean subject balanced accuracy of inner-selected complete pipeline.',
                  features='Unchanged V3 waveform delay moments and spectral controls; training-only TT/Tucker, selection and scaling. Fusion is TT16 plus 418 spectral controls, training-only select32.',
                  balance='Training-subject weights N/(3*N_class), recomputed inside every fold. No oversampling.',
                  early_stopping='No early stopping using validation/test subjects; tree counts and tensor iteration caps are fixed candidates.',
                  tensor_change='Stronger core anchors (.1,1 vs .001), smaller ranks (2/4), filters (4/8), head penalties (.1/1), fixed 500 iterations.',
                  remaining_limits='Previously explored subjects and outer splits; this new frozen nested run cannot erase researcher-level adaptive reuse. No external test set.')
    p=OUT/'protocol.json'
    if p.exists(): assert json.loads(p.read_text())==protocol
    else: p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    folds=[]
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=9060)
    for split,(train,test) in enumerate(cv.split(subjects,y)):
        path=OUT/f'fold_{split:02d}.json'
        if path.exists():
            row=json.loads(path.read_text()); assert row['signature']==sig and row['train']==train.tolist() and row['test']==test.tolist()
            folds.append(row); continue
        start=time.perf_counter()
        print(f'Tuning outer {split+1}/15: {len(train)} training people, {len(test)} held out; {len(specs)} fixed candidates.',flush=True)
        oof=np.full((len(specs),len(y)),-1,dtype=int)
        inner_records=[]
        for inner,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=90600+split).split(train,y[train])):
            it,iv=train[a],train[b]
            assert not set(it)&set(iv) and not (set(it)|set(iv))&set(test)
            bank,bases=feature_bank(x,controls,y,it)
            bfile=OUT/'bases'/f'inner_{split:02d}_{inner}.joblib'
            joblib.dump(dict(signature=sig,train=it,models=bases),bfile)
            opts=[]
            for k,cfg in enumerate(specs):
                if cfg['kind']=='supervised_tt':
                    m=fit_supervised(cfg,x[it],y[it]); pred=m.predict(x[iv])
                    opts.append(dict(config=cfg,**m.optimization_))
                else:
                    z=bank[cfg['representation']]
                    m=fit_classifier(cfg,z[it],y[it]); pred=m.predict(z[iv])
                oof[k,iv]=pred
            inner_records.append(dict(train=it.tolist(),validation=iv.tolist(),basis_file=str(bfile),supervised_optimization=opts))
            print(f'  Inner {inner+1}/3 complete.',flush=True)
        assert np.all(oof[:,train]>=0)
        scored=[dict(config=cfg,metrics=metric(y[train],oof[k,train])) for k,cfg in enumerate(specs)]
        selected=select(scored)
        bank,bases=feature_bank(x,controls,y,train)
        bfile=OUT/'bases'/f'outer_{split:02d}.joblib'
        joblib.dump(dict(signature=sig,train=train,models=bases),bfile)
        models={}; predictions={}; fitted={}
        for name,choice in selected.items():
            cfg=choice['config']; key=json.dumps(cfg,sort_keys=True)
            if key not in fitted:
                fitted[key]=fit_supervised(cfg,x[train],y[train]) if cfg['kind']=='supervised_tt' else fit_classifier(cfg,bank[cfg['representation']][train],y[train])
            m=fitted[key]
            z=x if cfg['kind']=='supervised_tt' else bank[cfg['representation']]
            pred=m.predict(z[test]); score,stype=scores(m,z[test])
            predictions[name]=dict(config=cfg,predictions=pred.tolist(),scores=score.tolist(),score_type=stype,
                                   inner_metrics=choice['metrics'],training_metrics=metric(y[train],m.predict(z[train])))
            if cfg['kind']=='supervised_tt': predictions[name]['optimization']=m.optimization_
            models[name]=dict(config=cfg,model=m)
        joblib.dump(dict(signature=sig,train=train,test=test,subjects_train=subjects[train],subjects_test=subjects[test],
                         basis_file=str(bfile),models=models),OUT/'models'/f'fold_{split:02d}.joblib')
        row=dict(signature=sig,split=split,repeat=split//5,train=train.tolist(),test=test.tolist(),test_subjects=subjects[test].tolist(),
                  truth=y[test].tolist(),inner_provenance=inner_records,inner_results=scored,predictions=predictions,seconds=time.perf_counter()-start)
        path.write_text(json.dumps(row,indent=2),encoding='utf-8'); folds.append(row); summarize(folds)
        print(f'Outer {split+1} saved in {row["seconds"]:.1f}s; no changes based on held-out scores.',flush=True)
    r=summarize(folds)
    print(json.dumps({n:{k:v for k,v in s.items() if k!='repeats'} for n,s in r['results'].items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
