"""Prespecified nested subject CV and a fixed-model learning curve."""
from pathlib import Path
import json
import hashlib
import time
import joblib
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits
from wave_models import WavePipeline, candidates, weights
from prepare import ROOT, OUT, sha

FAMILIES = ['wave_tt','wave_tucker','supervised_tt','spectral_control']


def metric(y,p):
    cm = confusion_matrix(y,p,labels=[0,1,2])
    return dict(balanced_accuracy=float(balanced_accuracy_score(y,p)),
                accuracy=float(np.mean(y==p)),macro_f1=float(f1_score(y,p,average='macro')),
                confusion_matrix=cm.tolist(),recall=(cm.diagonal()/cm.sum(1)).tolist())


def sig():
    h = hashlib.sha256()
    for p in [Path(__file__),Path(__file__).with_name('wave_models.py'),OUT/'input_fingerprints.json']:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def ranked(rows):
    return sorted(rows,key=lambda s:(-s['metrics']['balanced_accuracy'],-s['metrics']['macro_f1'],json.dumps(s['config'],sort_keys=True)))


def load_inputs():
    fingerprints = json.loads((OUT/'input_fingerprints.json').read_text())
    for name,expected in fingerprints.items():
        assert sha(ROOT/name)==expected, f'Input fingerprint mismatch: {name}'
    index = np.load(OUT/'index.npz')
    return np.load(OUT/'delay_moments.npy'),np.load(OUT/'spectral_controls.npy'),index['subjects'],index['labels']


def summarize(folds):
    results = {}
    for name in FAMILIES+['inner_selected_pipeline']:
        reps = []
        for rep in range(3):
            fs = [f for f in folds if f['repeat']==rep]
            if len(fs)!=5: continue
            y = np.concatenate([f['truth'] for f in fs])
            p = np.concatenate([f['predictions'][name]['predictions'] for f in fs])
            reps.append(metric(y,p))
        if reps:
            results[name] = dict(repeats=reps,mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reps])),
                                 sd_balanced_accuracy=float(np.std([r['balanced_accuracy'] for r in reps],ddof=1)) if len(reps)>1 else None,
                                 mean_accuracy=float(np.mean([r['accuracy'] for r in reps])),
                                 mean_macro_f1=float(np.mean([r['macro_f1'] for r in reps])))
    report = dict(signature=sig(),completed_outer_folds=len(folds),results=results,
                  total_fit_seconds=sum(f['seconds'] for f in folds),
                  caution='Repeated CV on a previously explored dataset is not independent external validation. Best observed family is secondary; inner-selected pipeline is primary.')
    (OUT/'nested_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    x,controls,subjects,y = load_inputs()
    configs = candidates()
    signature = sig()
    protocol = dict(signature=signature,date='2026-09-06',subjects=subjects.tolist(),class_counts=np.bincount(y).tolist(),
                    label_order=['AD','FTD','CN'],outer_folds=5,outer_repeats=3,outer_seed=9060,
                    inner_folds=3,inner_seed='90600 + outer split',candidates=configs,
                    primary_endpoint='Mean subject balanced accuracy of inner-selected entire pipeline',
                    class_weighting='Within each training split w_i = N_train/(3*N_train_class_i); all classes equal total loss mass.',
                    subject_weighting='Each subject is one observation; within-subject window delay moments are averaged, not summed.',
                    waveform_representation='Cleaned waveform -> anti-aliased 125Hz -> 19x16x485 delay tensor -> learned filters -> mean squared response -> log -> classifier. Moments compute exact pooled filter responses without storing giant tensors.',
                    supervision='Supervised TT optimizes channel core, lag/filter core and classification head jointly with weighted softmax loss.',
                    supervised_budget=dict(maxiter=100,channel_rank=4,filters=8,anchor_penalty=.001,head_penalties=[.01,.1]),
                    spectral_control='Same 418 spectral features as V2, recomputed on all retained full-cohort windows.',
                    learning_curve=dict(outer_repeat=0,fractions=[.4,.7,1.],config=dict(family='supervised_tt',penalty=.1),
                                        note='Fixed-model secondary diagnostic, no tuning to curve outcomes. Nested stratified training subsets; outer test fixed.'),
                    no_oversampling=True,no_demographic_or_MMSE_predictors=True,
                    note='Saved before inspecting any V3 outcome; V2 comparison changes both cohort and representation.')
    path = OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol
    else: path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    modeldir = OUT/'models'; modeldir.mkdir(exist_ok=True)
    folds = []
    splits = list(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=9060).split(subjects,y))
    for split,(train,test) in enumerate(splits):
        path = OUT/f'fold_{split:02d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            assert record['signature']==signature and record['train']==train.tolist() and record['test']==test.tolist()
            folds.append(record); continue
        start = time.perf_counter()
        print(f'Outer {split+1}/15: {len(train)} training people, {len(test)} held out; tuning only inside training.',flush=True)
        oof = np.full((len(configs),len(y)),-1,dtype=int)
        inner_records = []
        for inner,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=90600+split).split(train,y[train])):
            it,iv = train[a],train[b]
            assert not set(it)&set(iv) and not (set(it)|set(iv))&set(test)
            diagnostics = []
            for k,cfg in enumerate(configs):
                model = WavePipeline(cfg).fit(x[it],controls[it],y[it])
                oof[k,iv] = model.predict(x[iv],controls[iv])
                if cfg['family']=='supervised_tt': diagnostics.append(dict(config=cfg,**model.model_.optimization_))
            inner_records.append(dict(train=it.tolist(),validation=iv.tolist(),supervised_optimization=diagnostics))
            print(f'  Inner {inner+1}/3 complete.',flush=True)
        assert np.all(oof[:,train]>=0)
        scores = [dict(config=cfg,metrics=metric(y[train],oof[k,train])) for k,cfg in enumerate(configs)]
        ranking = ranked(scores)
        selected = {f:next(s for s in ranking if s['config']['family']==f) for f in FAMILIES}
        selected['inner_selected_pipeline'] = ranking[0]
        models, predictions, fitted = {}, {}, {}
        for name,choice in selected.items():
            cfg = choice['config']; key = json.dumps(cfg,sort_keys=True)
            if key not in fitted: fitted[key] = WavePipeline(cfg).fit(x[train],controls[train],y[train])
            model = fitted[key]
            p = model.predict(x[test],controls[test]); s = model.scores(x[test],controls[test])
            predictions[name] = dict(config=cfg,predictions=p.tolist(),scores=s.tolist(),score_type='uncalibrated_decision_function',
                                     inner_metrics=choice['metrics'],training_metrics=metric(y[train],model.predict(x[train],controls[train])))
            if cfg['family']=='supervised_tt': predictions[name]['optimization'] = model.model_.optimization_
            models[name] = model
        joblib.dump(dict(signature=signature,train=train,test=test,train_subjects=subjects[train],test_subjects=subjects[test],models=models),modeldir/f'fold_{split:02d}.joblib')
        record = dict(split=split,repeat=split//5,fold=split%5,signature=signature,train=train.tolist(),test=test.tolist(),
                      test_subjects=subjects[test].tolist(),truth=y[test].tolist(),inner_provenance=inner_records,
                      inner_results=scores,predictions=predictions,seconds=time.perf_counter()-start)
        path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        folds.append(record); summarize(folds)
        print(f'Outer {split+1} saved in {record["seconds"]:.1f}s. Test outcomes reserved for final report.',flush=True)
    print('Nested fitting complete; running the predeclared fixed-model learning curve.',flush=True)
    lc_path = OUT/'learning_curve.json'
    if lc_path.exists():
        learning = json.loads(lc_path.read_text()); assert learning['signature']==signature
        points = learning['points']
    else: points=[]
    for split,(train,test) in enumerate(splits[:5]):
        rng = np.random.default_rng(906000+split)
        ordered = {c:rng.permutation(train[y[train]==c]) for c in range(3)}
        for frac in [.4,.7,1.]:
            if any(p['split']==split and p['fraction']==frac for p in points): continue
            sub = np.sort(np.concatenate([a[:max(2,int(np.floor(len(a)*frac)))] for a in ordered.values()]))
            cfg = dict(family='supervised_tt',penalty=.1)
            model = WavePipeline(cfg).fit(x[sub],controls[sub],y[sub])
            pred = model.predict(x[test],controls[test])
            name = f'curve_{split:02d}_{int(frac*100):03d}.joblib'
            joblib.dump(dict(signature=signature,train=sub,test=test,model=model),modeldir/name)
            points.append(dict(split=split,fraction=frac,train=sub.tolist(),test=test.tolist(),truth=y[test].tolist(),predictions=pred.tolist(),
                               metrics=metric(y[test],pred),training_metrics=metric(y[sub],model.predict(x[sub],controls[sub])),
                               optimization=model.model_.optimization_,model_file=name))
            lc_path.write_text(json.dumps(dict(signature=signature,points=points),indent=2),encoding='utf-8')
        print(f'Learning curve fold {split+1}/5 saved.',flush=True)
    report = summarize(folds)
    print(json.dumps({f:{k:v for k,v in r.items() if k!='repeats'} for f,r in report['results'].items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
