"""Nested participant-level binary log-mel SVM experiment (HC=0, dementia=1)."""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,log_loss,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from features import HERE,COHORT,POLICY,CHANNELS,dump,key,sha,representation,prepare_person,load_eeg,summarize_recording
from model import BinaryModel

SEED=20261105
THRESHOLDS=[-.5,0.,.5]


def candidate_plan(trials):
    rng=np.random.default_rng(SEED)
    reps=[dict(zip(['sampling_hz','segment_seconds','stft_seconds','hop_seconds','mel_bins'],v))
          for v in itertools.product([128,256],[2,4,6,8],[1,2,3],[.25,.5,1.],[16,24,32])
          if v[1]>=v[2]+2*v[3]]
    def draw(rep):
        kernel=str(rng.choice(['linear','rbf']))
        weight=[None,'balanced',1.5,2.5][int(rng.integers(4))]
        return {**rep,'summary':str(rng.choice(['mean_sd','four_stats'])),
                'site_adjustment':str(rng.choice(['none','center','zscore'])),
                'scaler':str(rng.choice(['standard','robust'])),
                'pca':[None,20,40,60][int(rng.integers(4))],
                'C':float(10**rng.uniform(-3,2)),'kernel':kernel,
                'gamma':(['scale',.001,.01,.1,1.][int(rng.integers(5))] if kernel=='rbf' else 'scale'),
                'healthy_weight':weight}
    candidates=[]
    # Matched 128/256-Hz anchors also provide the fixed baseline comparison.
    for fs in [128,256]:
        for c,pca in itertools.product([.01,.1,1.,10.,100.],[None,20,40]):
            candidates.append({'sampling_hz':fs,'segment_seconds':8,'stft_seconds':2,'hop_seconds':.5,
                               'mel_bins':24,'summary':'mean_sd','site_adjustment':'zscore','scaler':'standard',
                               'pca':pca,'C':c,'kernel':'rbf','gamma':'scale','healthy_weight':'balanced'})
    candidates.extend(draw(r) for r in reps)
    if trials<len(candidates): raise ValueError(f'At least {len(candidates)} trials required for representation coverage')
    while len(candidates)<trials: candidates.append(draw(reps[int(rng.integers(len(reps)))]))
    return candidates


def initialize(data,trials):
    data=Path(data);data.mkdir(parents=True,exist_ok=True)
    people=json.loads(COHORT.read_text())['participants']
    candidates=candidate_plan(trials)
    reps=list({key(representation(c)):representation(c) for c in candidates}.values())
    plan={'participants':people,'binary_labels':[int(p['label']!='HC') for p in people],
          'classes':['HC','Dementia'],'candidates':candidates,'representations':reps,'threshold_grid':THRESHOLDS,
          'policy':POLICY,'seed':SEED,'cohort_sha256':sha(COHORT),
          'features_code_sha256':sha(HERE/'features.py'),
          'search':'Fixed seeded finite random search plus matched anchors; not exhaustive Cartesian search',
          'evaluation':'Previously explored 174-person cohort, internal validation only'}
    path=data/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan: raise ValueError('Prepared protocol differs; use a new --data directory')
    dump(path,plan)
    return plan


def prepare(args):
    plan=initialize(args.data,args.trials)
    todo=[(i,p,plan['representations'],args.data) for i,p in enumerate(plan['participants'])]
    if args.workers==1:
        for item in todo:
            i,status=prepare_person(*item);print(f'EEG {i+1}/{len(todo)}: {status}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(prepare_person,*item) for item in todo]
            for done,future in enumerate(as_completed(jobs),1):
                i,status=future.result();print(f'EEG {done}/{len(todo)} ({plan["participants"][i]["subject_id"]}): {status}',flush=True)
    arrays={key(r):[] for r in plan['representations']}
    audit=[]
    for i in range(len(todo)):
        path=Path(args.data)/'parts'/f'{i:03d}.npz'
        part=np.load(path)
        audit.append(json.loads(path.with_suffix('.json').read_text()))
        for k in arrays: arrays[k].append(part[k])
    np.savez_compressed(Path(args.data)/'features.npz',**{k:np.stack(v) for k,v in arrays.items()})
    dump(Path(args.data)/'preparation_audit.json',audit)
    print(f'Prepared {len(todo)} people; {len(arrays)} representations; {len(plan["candidates"])} search candidates',flush=True)


def folds(plan,indices,n,seed):
    # Preserve both site and original AD/FTD/HC composition while classifying binary labels.
    strata=np.array([f'{p["site"]}:{p["label"]}' for p in plan['participants']])[indices]
    if np.unique(strata,return_counts=True)[1].min()<n: raise ValueError('Insufficient participants per site/diagnosis stratum')
    return [(indices[a],indices[b]) for a,b in StratifiedKFold(n,shuffle=True,random_state=seed).split(indices,strata)]


def metrics(y,pred,p=None):
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    result={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'healthy_recall':float(tn/max(tn+fp,1)),'dementia_recall':float(tp/max(tp+fn,1)),
            'healthy_f1':float(2*tn/max(2*tn+fp+fn,1)),'dementia_f1':float(2*tp/max(2*tp+fp+fn,1)),
            'confusion_matrix':cm.tolist()}
    if p is not None and len(np.unique(y))==2:
        result.update(log_loss=float(log_loss(y,p,labels=[0,1])),roc_auc=float(roc_auc_score(y,p[:,1])),
                      brier=float(np.mean((p[:,1]-y)**2)))
    return result


def selection_score(y,pred):
    # Binary macro-F1 and BA without sklearn overhead inside the threshold search.
    tn=np.sum((y==0)&(pred==0));fp=np.sum((y==0)&(pred==1))
    fn=np.sum((y==1)&(pred==0));tp=np.sum((y==1)&(pred==1))
    return (.5*(2*tn/max(2*tn+fp+fn,1)+2*tp/max(2*tp+fp+fn,1)),
            .5*(tn/max(tn+fp,1)+tp/max(tp+fn,1)))


class Experiment:
    def __init__(self,args):
        self.args=args;self.data=Path(args.data);self.out=Path(args.out);self.out.mkdir(parents=True,exist_ok=True)
        self.plan=json.loads((self.data/'plan.json').read_text())
        self.y=np.array(self.plan['binary_labels']);self.people=self.plan['participants']
        self.sites=np.array([p['site'] for p in self.people]);self.ids=np.array([p['subject_id'] for p in self.people])
        self.features=dict(np.load(self.data/'features.npz'));self.memo={}

    def x(self,c):
        values=self.features[key(representation(c))]
        return (values[...,:2] if c['summary']=='mean_sd' else values).reshape(len(self.y),-1)

    def fit(self,c,indices): return BinaryModel(c).fit(self.x(c)[indices],self.y[indices],self.sites[indices])

    def choose(self,development,seed,folder):
        inner=folds(self.plan,development,self.args.inner_folds,seed);rows=[];all_scores=[]
        for i,c in enumerate(self.plan['candidates']):
            score=np.full(len(self.y),np.nan)
            for fit,valid in inner:
                model=self.fit(c,fit)
                score[valid]=model.decision_function(self.x(c)[valid],self.sites[valid])
            scores=[]
            for threshold in THRESHOLDS:
                f1,ba=selection_score(self.y[development],score[development]>=threshold)
                scores.append((float(f1),float(ba),-abs(threshold),threshold))
            f1,ba,_,threshold=max(scores)
            rows.append({'candidate':i,'config':{**c,'threshold':threshold},'inner_macro_f1':f1,'inner_balanced_accuracy':ba,
                         'threshold_scores':[{'threshold':t,'macro_f1':a,'balanced_accuracy':b} for a,b,_,t in scores]})
            all_scores.append(score[development])
            if (i+1)%32==0: print(f'  search {i+1}/{len(self.plan["candidates"])}',flush=True)
        # Exact primary/secondary ranking; deterministic simplicity only breaks metric ties.
        winner=max(rows,key=lambda r:(r['inner_macro_f1'],r['inner_balanced_accuracy'],
                    r['config']['kernel']=='linear',-(r['config']['pca'] or 10000),-r['candidate']))
        dump(folder/'search.json',{'selected':winner,'candidates':rows,
             'inner_splits':[{'train_ids':self.ids[a].tolist(),'validation_ids':self.ids[b].tolist()} for a,b in inner]})
        return winner,inner

    def calibrated_fit(self,c,development,seed):
        score=np.full(len(self.y),np.nan);splits=folds(self.plan,development,self.args.inner_folds,seed)
        for fit,valid in splits:
            model=self.fit(c,fit);score[valid]=model.decision_function(self.x(c)[valid],self.sites[valid])
        calibrator=LogisticRegression(C=1e6,max_iter=1000).fit(score[development,None],self.y[development])
        if calibrator.coef_[0,0]<=0: raise RuntimeError('Calibration reverses score orientation; inspect data before deployment')
        model=self.fit(c,development);model.calibrator_=calibrator
        return model,[{'train_ids':self.ids[a].tolist(),'calibration_ids':self.ids[b].tolist()} for a,b in splits]

    def run(self):
        args=self.args
        protocol={'seed':SEED,'repeats':args.repeats,'outer_folds':args.outer_folds,'inner_folds':args.inner_folds,
                  'n_participants':len(self.y),'class_counts':np.bincount(self.y).tolist(),
                  'selection':'pooled inner OOF macro F1, then balanced accuracy; threshold chosen only on inner OOF scores',
                  'prediction':'SVM score >= inner-selected threshold; calibrated probabilities reported separately',
                  'calibration':'unweighted sigmoid on fresh 3-fold full-pipeline OOF scores; outer test excluded',
                  'source_hashes':{str(p):sha(p) for p in [self.data/'plan.json',self.data/'features.npz',*sorted(HERE.glob('*.py'))]},
                  'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__},
                  'status':'Internal validation on previously explored cohort; not independent external validation'}
        path=self.out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Output protocol differs; choose a new output directory')
        dump(path,protocol);records=[]
        for repeat in range(args.repeats):
            for fold,(fit,test) in enumerate(folds(self.plan,np.arange(len(self.y)),args.outer_folds,SEED+repeat*101)):
                folder=self.out/f'repeat_{repeat:02d}_fold_{fold:02d}';folder.mkdir(exist_ok=True)
                if (folder/'result.json').exists(): records.append(json.loads((folder/'result.json').read_text()));continue
                print(f'Outer repeat {repeat+1}/{args.repeats}, fold {fold+1}/{args.outer_folds}',flush=True)
                chosen,_=self.choose(fit,SEED+repeat*1000+fold,folder)
                c=chosen['config'];model,cal=self.calibrated_fit(c,fit,SEED+100000+repeat*1000+fold)
                x=self.x(c)[test];score=model.decision_function(x,self.sites[test]);p=model.predict_proba(x,self.sites[test])
                pred=(score>=c['threshold']).astype(int)
                controls={}
                for fs in [128,256]:
                    baseline={'sampling_hz':fs,'segment_seconds':8,'stft_seconds':2,'hop_seconds':.5,'mel_bins':24,
                              'summary':'mean_sd','site_adjustment':'zscore','scaler':'standard','pca':40,
                              'C':1.,'kernel':'rbf','gamma':'scale','healthy_weight':'balanced','threshold':0.}
                    fixed=self.fit(baseline,fit);controls[str(fs)]=fixed.predict(self.x(baseline)[test],self.sites[test]).tolist()
                result={'repeat':repeat,'fold':fold,'test_indices':test.tolist(),'train_ids':self.ids[fit].tolist(),
                        'test_ids':self.ids[test].tolist(),'true':self.y[test].tolist(),'selected':chosen,
                        'predictions':pred.tolist(),'scores':score.tolist(),'probabilities':p.tolist(),
                        'probability_threshold':model.probability_threshold,'calibration_splits':cal,
                        'fixed_sampling_controls':controls,'metrics':metrics(self.y[test],pred,p)}
                dump(folder/'result.json',result);joblib.dump(model,folder/'model.joblib');records.append(result)
                print(json.dumps(result['metrics']),flush=True)
        dump(self.out/'oof_predictions.json',records)
        from report import generate
        generate(self.out,self.data)
        if not args.no_final:
            self.finalize()

    def finalize(self):
        protocol=json.loads((self.out/'protocol.json').read_text())
        for path in [self.data/'plan.json',self.data/'features.npz']:
            if protocol['source_hashes'][str(path)]!=sha(path):raise ValueError('Validation inputs changed')
        dest=self.out/'final';dest.mkdir(exist_ok=True)
        if (dest/'model.joblib').exists() and (dest/'metadata.json').exists():return
        print('Selecting final deployment model on all participants (not test performance)',flush=True)
        dump(dest/'protocol.json',{'validation_protocol_sha256':sha(self.out/'protocol.json'),
             'source_hashes':{str(p):sha(p) for p in sorted(HERE.glob('*.py'))},
             'note':'Final refit uses the same full PCA with gesvd fallback on a gesdd convergence failure. Original validation source archived separately; validation predictions are unchanged.'})
        chosen,_=self.choose(np.arange(len(self.y)),SEED+999000,dest)
        model,cal=self.calibrated_fit(chosen['config'],np.arange(len(self.y)),SEED+999100)
        joblib.dump(model,dest/'model.joblib')
        dump(dest/'metadata.json',{'selected':chosen,'classes':['HC','Dementia'],'training_ids':self.ids.tolist(),
                                  'probability_threshold':model.probability_threshold,'calibration_splits':cal,
                                  'deployment_fit_not_test_performance':True,'policy':POLICY})
        np.save(dest/'svm_input_matrix.npy',model.transform(self.x(chosen['config']),self.sites))


def predict(args):
    model=joblib.load(args.model);c=model.config
    metadata=json.loads(Path(args.model).with_name('metadata.json').read_text())
    if args.recording:
        if not args.source or not args.site: raise ValueError('Provide --source and --site for a recording')
        person={'path':args.recording,'source':args.source,'site':args.site,'subject_id':Path(args.recording).stem}
        with contextlib.redirect_stdout(sys.stderr):
            data,intervals,_=load_eeg(person);tensor,_=summarize_recording(data,intervals,c)
        values=tensor[...,:2] if c['summary']=='mean_sd' else tensor
        x=values.reshape(1,-1);site=args.site;who=person['subject_id'];included=None
    else:
        plan=json.loads((Path(args.data)/'plan.json').read_text());ids=[p['subject_id'] for p in plan['participants']]
        i=ids.index(args.subject_id);tensor=np.load(Path(args.data)/'features.npz')[key(representation(c))][i]
        x=(tensor[...,:2] if c['summary']=='mean_sd' else tensor).reshape(1,-1)
        site=plan['participants'][i]['site'];who=args.subject_id;included=who in metadata['training_ids']
    p=model.predict_proba(x,[site])[0]
    print(json.dumps({'participant':who,'probabilities':{'HC':float(p[0]),'Dementia':float(p[1])},
                      'dementia_probability_operating_threshold':model.probability_threshold,
                      'included_in_model_training':included,'known_training_site':site in model.site_stats_,
                      'scope':'Internal-cohort research probabilities; selected operating threshold can differ from 0.5'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    default=str(HERE/'outputs/prepared')
    p=sub.add_parser('prepare');p.add_argument('--data',default=default);p.add_argument('--trials',type=int,default=256);p.add_argument('--workers',type=int,default=2)
    p=sub.add_parser('train');p.add_argument('--data',default=default);p.add_argument('--out',default=str(HERE/'outputs/nested'))
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--no-final',action='store_true')
    p=sub.add_parser('report');p.add_argument('--data',default=default);p.add_argument('--out',default=str(HERE/'outputs/nested'))
    p=sub.add_parser('finalize');p.add_argument('--data',default=default);p.add_argument('--out',default=str(HERE/'outputs/nested'))
    p.add_argument('--inner-folds',type=int,default=3);p.add_argument('--threads',type=int,default=2)
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--data',default=default)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--recording');group.add_argument('--subject-id')
    p.add_argument('--site');p.add_argument('--source',choices=['ds004504','brainlat'])
    args=parser.parse_args()
    with threadpool_limits(limits=getattr(args,'threads',1)):
        if args.command=='prepare':prepare(args)
        elif args.command=='train':Experiment(args).run()
        elif args.command=='finalize':Experiment(args).finalize()
        elif args.command=='predict':predict(args)
        else:
            from report import generate
            generate(Path(args.out),Path(args.data))


if __name__=='__main__':main()
