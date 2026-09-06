"""Nested subject CV, persisted models, and Tensor Train/Tucker comparisons."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import warnings
import joblib
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix,roc_auc_score
from threadpoolctl import threadpool_limits
from models import (TTProjection,TuckerProjection,anatomical_smoothing,apply_smoothing,
                    classifier_grid,build_classifier,predict_with_scores)

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'tensor_pipeline/outputs/v2_tensor_train'
FAMILIES=['tt_subject','tt_window_pool','graph_tt_window_pool','tucker_window_pool','tt_spectral_fusion','spectral_control']
RANKS=[8,16,32]

def signature():
    h=hashlib.sha256()
    for p in [Path(__file__),Path(__file__).with_name('models.py'),DATA/'preparation_report.json']:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]

def metrics(y,pred,scores=None):
    cm=confusion_matrix(y,pred,labels=[0,1,2])
    r=dict(balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
           accuracy=float(np.mean(y==pred)),confusion_matrix=cm.tolist(),
           sensitivities={g:float(cm[i,i]/max(cm[i].sum(),1)) for i,g in enumerate(['AD','FTD','CN'])})
    if scores is not None:
        r['macro_auc']=float(np.mean([roc_auc_score(y==i,scores[:,i]) for i in range(3)]))
    return r

class Experiment:
    def __init__(self,out):
        self.out=out
        self.sig=signature()
        self.cache=DATA/'feature_cache'/self.sig
        self.cache.mkdir(parents=True,exist_ok=True)
        self.index=np.load(DATA/'index.npz')
        self.subjects=self.index['subjects']
        self.labels=self.index['labels']
        self.windows=np.load(DATA/'window_logpsd.npy',mmap_mode='r')
        self.subject_tensor=np.load(DATA/'subject_spectral_tensor.npy')
        self.controls=np.load(DATA/'spectral_controls.npy')
        self.by_subject=[np.flatnonzero(self.index['window_subject']==i) for i in range(len(self.subjects))]
        self.smooth,self.adjacency=anatomical_smoothing(np.load(DATA/'coordinates.npy'),.5)

    def features(self,train):
        key=hashlib.sha256(np.array(sorted(train),dtype='int32').tobytes()).hexdigest()[:16]
        prefix=self.cache/key
        if prefix.with_suffix('.npz').exists():
            f=np.load(prefix.with_suffix('.npz'))
            assert np.array_equal(f['fit_subject_indices'],np.sort(train))
            return {k:f[k] for k in f.files if k!='fit_subject_indices'},prefix
        calibration=[]
        for i in sorted(train):
            ix=self.by_subject[i]
            calibration.extend(ix[np.linspace(0,len(ix)-1,24,dtype=int)])
        calibration=np.array(calibration)
        assert set(self.index['window_subject'][calibration])==set(train)
        x=np.asarray(self.windows[calibration],dtype=float)
        models={}
        arrays={}
        tt=TTProjection(32).fit(self.subject_tensor[train])
        arrays['tt_subject']=tt.transform(self.subject_tensor)
        models['tt_subject']=tt
        models['tt_window_pool']=TTProjection(32).fit(x)
        models['graph_tt_window_pool']=TTProjection(32).fit(apply_smoothing(x,self.smooth))
        models['tucker_window_pool']=TuckerProjection((6,8,3)).fit(x)
        for name in ['tt_window_pool','graph_tt_window_pool','tucker_window_pool']:
            model=models[name]
            pooled=[]
            for ix in self.by_subject:
                wx=np.asarray(self.windows[ix],dtype=float)
                if name.startswith('graph'):
                    wx=apply_smoothing(wx,self.smooth)
                z=model.transform(wx)
                pooled.append(np.stack([z.mean(0),z.std(0),np.median(z,axis=0),
                                        np.quantile(z,.75,axis=0)-np.quantile(z,.25,axis=0)],axis=1))
            arrays[name]=np.array(pooled)
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(prefix.with_suffix('.npz'),fit_subject_indices=np.sort(train),**arrays)
        # Cores, bases, means and calibration provenance are retained for replay.
        joblib.dump(dict(models=models,train_subject_indices=np.sort(train),
                         train_subjects=self.subjects[np.sort(train)],calibration_indices=calibration,
                         smooth=self.smooth,adjacency=self.adjacency,signature=self.sig),prefix.with_suffix('.joblib'))
        return arrays,prefix

    def specs(self,family):
        if family=='spectral_control':
            reps=[dict(rank=None,selection=k) for k in [16,32,64]]
        elif family=='tucker_window_pool':
            reps=[dict(rank=None,selection=k) for k in [16,32,64]]
        elif family=='tt_spectral_fusion':
            reps=[dict(rank=r,selection=32) for r in RANKS]
        else:
            reps=[dict(rank=r,selection=None) for r in RANKS]
        return [dict(family=family,**rp,**clf) for rp in reps for clf in classifier_grid()]

    def matrix(self,arrays,cfg):
        family=cfg['family']
        if family=='spectral_control':
            return self.controls
        if family=='tt_spectral_fusion':
            f=arrays['tt_window_pool'][:,:cfg['rank']].reshape(len(self.subjects),-1)
            return np.concatenate([f,self.controls],axis=1)
        f=arrays[family]
        if cfg['rank'] is not None:
            f=f[:,:cfg['rank']]
        return f.reshape(len(self.subjects),-1)

    def tune(self,outer_train,seed):
        specs=[s for family in FAMILIES for s in self.specs(family)]
        oof=np.full((len(specs),len(self.labels)),-1,dtype='int8')
        provenance=[]
        for inner,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(outer_train,self.labels[outer_train])):
            train=outer_train[it]
            val=outer_train[iv]
            assert not set(train)&set(val)
            arrays,prefix=self.features(train)
            provenance.append(dict(inner=inner,train=train.tolist(),validation=val.tolist(),decomposition_file=str(prefix.with_suffix('.joblib'))))
            for j,cfg in enumerate(specs):
                f=self.matrix(arrays,cfg)
                model=build_classifier(cfg,f.shape[1],cfg['selection'])
                model.fit(f[train],self.labels[train])
                oof[j,val]=model.predict(f[val])
        scored=[]
        for j,cfg in enumerate(specs):
            assert np.all(oof[j,outer_train]>=0)
            met=metrics(self.labels[outer_train],oof[j,outer_train])
            scored.append(dict(config=cfg,balanced_accuracy=met['balanced_accuracy'],macro_f1=met['macro_f1']))
        def key(s):
            # Reproducible ties, avoiding outer-test decisions.
            return (-s['balanced_accuracy'],-s['macro_f1'],s['config']['rank'] or s['config']['selection'],json.dumps(s['config'],sort_keys=True))
        ranked=sorted(scored,key=key)
        best={family:next(s for s in ranked if s['config']['family']==family) for family in FAMILIES}
        best['inner_selected_pipeline']=ranked[0]
        return best,scored,provenance

    def outer(self,split,train,test):
        path=self.out/f'fold_{split:02d}.json'
        if path.exists():
            old=json.loads(path.read_text())
            assert old['signature']==self.sig and old['train']==train.tolist() and old['test']==test.tolist()
            return old
        start=time.perf_counter()
        print(f'Outer {split+1}: tuning {len(train)} training subjects; {len(test)} held out',flush=True)
        best,inner_results,provenance=self.tune(train,8000+split)
        arrays,prefix=self.features(train)
        predictions={}
        saved_models={}
        for name,selection in best.items():
            cfg=selection['config']
            f=self.matrix(arrays,cfg)
            model=build_classifier(cfg,f.shape[1],cfg['selection'])
            model.fit(f[train],self.labels[train])
            pred,scores,score_type=predict_with_scores(model,f[test])
            predictions[name]=dict(config=cfg,inner_balanced_accuracy=selection['balanced_accuracy'],
                    predictions=pred.tolist(),scores=scores.tolist(),score_type=score_type)
            saved_models[name]=dict(classifier=model,config=cfg,decomposition_file=str(prefix.with_suffix('.joblib')))
        joblib.dump(saved_models,self.out/f'fold_{split:02d}_models.joblib')
        result=dict(split=split,repeat=split//5,fold=split%5,signature=self.sig,
                    train=train.tolist(),test=test.tolist(),test_subjects=self.subjects[test].tolist(),
                    true_labels=self.labels[test].tolist(),predictions=predictions,
                    inner_provenance=provenance,inner_results=inner_results,seconds=time.perf_counter()-start)
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        # Outcomes are reported only by the final aggregation, not used to alter search.
        print(f'Completed outer {split+1} in {result["seconds"]:.1f}s; saved models and predictions.',flush=True)
        return result

    def summarize(self,results):
        summaries={}
        for family in FAMILIES+['inner_selected_pipeline']:
            repeats=[]
            for repeat in sorted(set(r['repeat'] for r in results)):
                folds=[r for r in results if r['repeat']==repeat]
                if len(folds)!=5:
                    continue
                truth=np.concatenate([r['true_labels'] for r in folds])
                pred=np.concatenate([r['predictions'][family]['predictions'] for r in folds])
                met=metrics(truth,pred)
                met['repeat']=repeat
                repeats.append(met)
            if repeats:
                summaries[family]=dict(mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in repeats])),
                                      sd_balanced_accuracy=float(np.std([r['balanced_accuracy'] for r in repeats],ddof=1)) if len(repeats)>1 else None,
                                      mean_macro_f1=float(np.mean([r['macro_f1'] for r in repeats])),repeats=repeats)
        report=dict(design='Nested 3x5-fold outer / 3-fold inner stratified SUBJECT CV',
                    signature=self.sig,n_subjects=len(self.subjects),subject_names=self.subjects.tolist(),
                    label_names=['AD','FTD','CN'], completed_outer_folds=len(results),
                    families=FAMILIES,rank_candidates=RANKS,classifiers=classifier_grid(),
                    graph_smoothing_strength=.5,fit_windows_per_subject=24,
                    results=summaries,total_seconds=sum(r['seconds'] for r in results),
                    caveats=['Same cohort previously explored; nested CV cannot erase earlier researcher exposure.',
                             'Best family scores are exploratory comparisons; inner_selected_pipeline tests selection across all candidates.',
                             'Repeated folds are correlated; repeat SD is not a confidence interval.',
                             'TT and HOSVD operate on signed log power; non-negativity is not a requirement.',
                             'Graph TT uses fixed anatomical pre-smoothing, not the old NCP regularizer.'])
        (self.out/'nested_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-folds',type=int,default=15)
    args=p.parse_args()
    out=DATA/'nested_benchmark'
    out.mkdir(exist_ok=True)
    exp=Experiment(out)
    protocol=dict(signature=exp.sig,outer_folds=5,repeats=args.repeats,inner_folds=3,
                  outer_seed=5040,inner_seed='8000+outer_index',families=FAMILIES,
                  rank_candidates=RANKS,classifiers=classifier_grid(),
                  input_preparation_sha=hashlib.sha256((DATA/'preparation_report.json').read_bytes()).hexdigest(),
                  subject_ids=exp.subjects.tolist(),primary_endpoint='inner_selected_pipeline subject balanced accuracy',
                  note='Specification saved before examining new outer-test scores.')
    config=out/'protocol.json'
    if config.exists():
        assert json.loads(config.read_text())==protocol
    else:
        config.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[]
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=args.repeats,random_state=5040)
    for split,(train,test) in enumerate(cv.split(exp.subjects,exp.labels)):
        if split>=args.max_folds:
            break
        assert not set(exp.subjects[train])&set(exp.subjects[test])
        results.append(exp.outer(split,train,test))
        exp.summarize(results)
    print(json.dumps(exp.summarize(results)['results'],indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
