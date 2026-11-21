"""BrainLat-only binary montage comparison with matched nested participant CV."""
from __future__ import annotations
import argparse
import contextlib
import itertools
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import joblib
import numpy as np
import sklearn
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits

from brainlat_data import HERE,COHORT,ARMS,POLICY,dump,sha,geometry,prepare_one,extract
from brainlat_model import Transform,classifier,ProbabilityModel,MonotoneSigmoid

SEED=20261106
THRESHOLDS=[-.5,0.,.5]


def grid():
    return [dict(zip(['site_adjustment','scaler','pca','C','kernel','weight'],v))
            for v in itertools.product(['none','zscore'],['standard','robust'],[None,10,20,40],
                                       [.01,.1,1.,10.],['linear','rbf'],[None,'balanced'])]


def prepare(args):
    destination=Path(args.data);destination.mkdir(parents=True,exist_ok=True)
    mapping,meta=geometry()
    people=[p for p in json.loads(COHORT.read_text())['participants'] if p['source']=='brainlat']
    if len(people)!=86 or len({p['subject_id'] for p in people})!=86:raise ValueError('Frozen BrainLat cohort mismatch')
    plan={'participants':people,'binary_labels':[int(p['label']!='HC') for p in people],'classes':['HC','Dementia'],
          'arms':ARMS,'policy':POLICY,'geometry':meta,'candidates':grid(),'thresholds':THRESHOLDS,'seed':SEED,
          'data_code_sha256':sha(HERE/'brainlat_data.py'),'source_cohort_sha256':sha(COHORT),
          'mne_version':mne.__version__,'inference':'native channel order and512Hz required; no 19-channel harmonization before feature extraction'}
    path=destination/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Preparation protocol changed; choose a new --data')
    dump(path,plan);np.savez_compressed(destination/'operators.npz',**mapping)
    signature={'policy':POLICY,'data_code_sha256':plan['data_code_sha256'],'operators_sha256':sha(destination/'operators.npz')}
    todo=[(i,p,mapping,destination,signature) for i,p in enumerate(people)]
    if args.workers==1:
        for item in todo:
            i,status=prepare_one(*item);print(f'BrainLat {i+1}/86: {status}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(prepare_one,*item) for item in todo]
            for done,future in enumerate(as_completed(futures),1):
                i,status=future.result();print(f'BrainLat {done}/86 ({people[i]["subject_id"]}): {status}',flush=True)
    arrays={a:[] for a in ARMS};audits=[]
    for i in range(86):
        path=destination/'parts'/f'{i:03d}.npz'
        with np.load(path) as part:
            for a in ARMS:arrays[a].append(part[a].ravel())
        audits.append(json.loads(path.with_suffix('.json').read_text()))
    np.savez_compressed(destination/'features.npz',**{a:np.stack(v) for a,v in arrays.items()})
    dump(destination/'preparation_audit.json',audits)
    print('Prepared identical86-person cohort in all four arms',flush=True)


def folds(plan,indices,count,seed):
    strata=np.array([f'{p["site"]}:{p["label"]}' for p in plan['participants']])[indices]
    if np.unique(strata,return_counts=True)[1].min()<count:raise ValueError('Too few participants per original diagnosis/site stratum')
    return [(indices[a],indices[b]) for a,b in StratifiedKFold(count,shuffle=True,random_state=seed).split(indices,strata)]


def scores(y,pred):
    tn=np.sum((y==0)&(pred==0));fp=np.sum((y==0)&(pred==1));fn=np.sum((y==1)&(pred==0));tp=np.sum((y==1)&(pred==1))
    return np.array([.5*(2*tn/max(2*tn+fp+fn,1)+2*tp/max(2*tp+fp+fn,1)),
                     .5*(tn/max(tn+fp,1)+tp/max(tp+fn,1)),(tn+tp)/len(y),tn/max(tn+fp,1),tp/max(tp+fn,1)],float)


def metrics(y,pred,p=None):
    result=dict(zip(['macro_f1','balanced_accuracy','accuracy','healthy_recall','dementia_recall'],scores(y,pred).tolist()))
    result['confusion_matrix']=confusion_matrix(y,pred,labels=[0,1]).tolist()
    if p is not None:result.update(roc_auc=float(roc_auc_score(y,p[:,1])),log_loss=float(log_loss(y,p,labels=[0,1])))
    return result


class Experiment:
    def __init__(self,args):
        self.args=args;self.data=Path(args.data);self.out=Path(args.out);self.out.mkdir(parents=True,exist_ok=True)
        self.plan=json.loads((self.data/'plan.json').read_text());self.x=dict(np.load(self.data/'features.npz'))
        self.people=self.plan['participants'];self.ids=np.array([p['subject_id'] for p in self.people])
        self.sites=np.array([p['site'] for p in self.people]);self.y=np.array(self.plan['binary_labels'])

    def choose(self,arm,development,seed,folder):
        inner=folds(self.plan,development,self.args.inner_folds,seed);rows=[];candidates=self.plan['candidates']
        # Transforms are cached only within this selection call. All fitted rows
        # belong to the relevant INNER training set; SVM-only settings reuse them.
        grouped={}
        for index,c in enumerate(candidates):grouped.setdefault((c['site_adjustment'],c['scaler'],c['pca']),[]).append((index,c))
        for position,(pre,choices) in enumerate(grouped.items()):
            transformed=[]
            for fit,valid in inner:
                t=Transform(choices[0][1]).fit(self.x[arm][fit],self.sites[fit])
                transformed.append((fit,valid,t.transform(self.x[arm][fit],self.sites[fit]),t.transform(self.x[arm][valid],self.sites[valid])))
            for index,c in choices:
                decision=np.full(len(self.y),np.nan);failed=False
                for fit,valid,a,b in transformed:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');svm=classifier(c).fit(a,self.y[fit])
                    if svm.fit_status_!=0:failed=True;break
                    decision[valid]=svm.decision_function(b)
                if failed:rows.append({'candidate':index,'invalid':'SVM did not converge within200000 iterations'});continue
                options=[(scores(self.y[development],decision[development]>=t),t) for t in THRESHOLDS]
                value,threshold=max(options,key=lambda v:(v[0][0],v[0][1],-abs(v[1])))
                rows.append({'candidate':index,'config':{**c,'threshold':threshold},'inner_macro_f1':float(value[0]),'inner_balanced_accuracy':float(value[1])})
            if (position+1)%4==0:print(f'  {arm}: preprocessing {position+1}/{len(grouped)}',flush=True)
        valid=[r for r in rows if 'config' in r]
        if not valid:raise RuntimeError('All candidate SVMs failed')
        chosen=max(valid,key=lambda r:(r['inner_macro_f1'],r['inner_balanced_accuracy'],r['config']['kernel']=='linear',-r['candidate']))
        dump(folder/'search.json',{'selected':chosen,'candidates':rows,
             'inner_splits':[{'train_ids':self.ids[a].tolist(),'validation_ids':self.ids[b].tolist()} for a,b in inner]})
        return chosen

    def fit(self,arm,c,indices):
        t=Transform(c).fit(self.x[arm][indices],self.sites[indices])
        z=t.transform(self.x[arm][indices],self.sites[indices])
        for limit in [200000,2000000,10000000]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');svm=classifier(c,max_iter=limit).fit(z,self.y[indices])
            if svm.fit_status_==0:return t,svm
        raise RuntimeError('Selected SVM failed to converge after numerical retries')

    def calibrated(self,arm,c,indices,seed):
        decision=np.full(len(self.y),np.nan);inner=folds(self.plan,indices,self.args.inner_folds,seed)
        for fit,valid in inner:
            t,svm=self.fit(arm,c,fit);decision[valid]=svm.decision_function(t.transform(self.x[arm][valid],self.sites[valid]))
        cal=MonotoneSigmoid().fit(decision[indices],self.y[indices])
        t,svm=self.fit(arm,c,indices)
        return ProbabilityModel(c,t,svm,cal),[{'train_ids':self.ids[a].tolist(),'calibration_ids':self.ids[b].tolist()} for a,b in inner]

    def run(self):
        args=self.args
        files=[self.data/'plan.json',self.data/'features.npz',self.data/'operators.npz',*sorted(HERE.glob('*.py'))]
        protocol={'participants':86,'class_counts':np.bincount(self.y).tolist(),'repeats':args.repeats,
                  'outer_folds':args.outer_folds,'inner_folds':args.inner_folds,'arms':ARMS,'seed':SEED,
                  'selection':'pooled inner macro F1, balanced accuracy tie-break; threshold selected only using inner OOF',
                  'calibration':'non-negative-slope sigmoid on fresh training-only full-pipeline OOF scores; flat boundary yields training prevalence, main SVM predictions unchanged',
                  'refit_solver_retry_limits':[200000,2000000,10000000],
                  'hashes':{str(p):sha(p) for p in files},'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__,'mne':mne.__version__},
                  'caveat':'Previously explored cohort; native19 locations only approximate the interpolated19 locations. All arms share native128 reference and window-gain normalization.'}
        path=self.out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol changed; use a new --out')
        dump(path,protocol);records=[]
        for repeat in range(args.repeats):
            for fold,(fit,test) in enumerate(folds(self.plan,np.arange(len(self.y)),args.outer_folds,SEED+repeat*101)):
                for arm in ARMS:
                    folder=self.out/f'repeat_{repeat:02d}_fold_{fold:02d}'/arm;folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'result.json').exists():records.append(json.loads((folder/'result.json').read_text()));continue
                    print(f'Outer repeat {repeat+1}/{args.repeats}, fold {fold+1}/{args.outer_folds}: {arm}',flush=True)
                    chosen=self.choose(arm,fit,SEED+repeat*1000+fold,folder)
                    model,cal=self.calibrated(arm,chosen['config'],fit,SEED+100000+repeat*1000+fold)
                    p=model.predict_proba(self.x[arm][test],self.sites[test]);pred=model.predict(self.x[arm][test],self.sites[test])
                    joblib.dump(model,folder/'model.joblib')
                    row={'arm':arm,'repeat':repeat,'fold':fold,'test_indices':test.tolist(),'train_ids':self.ids[fit].tolist(),
                         'test_ids':self.ids[test].tolist(),'true':self.y[test].tolist(),'predictions':pred.tolist(),
                         'probabilities':p.tolist(),'selected':chosen,'calibration_splits':cal,
                         'calibration_diagnostics':{'slope':float(model.calibrator_.slope_),'uninformative':model.calibrator_.uninformative_},
                         'refit_solver':{'max_iter':model.svm_.max_iter,'iterations':model.svm_.n_iter_.tolist(),'converged':model.svm_.fit_status_==0},
                         'probability_threshold':model.probability_threshold,'metrics':metrics(self.y[test],pred,p)}
                    dump(folder/'result.json',row);records.append(row)
                    print(json.dumps(row['metrics']),flush=True)
        dump(self.out/'oof_predictions.json',records)
        from report import generate
        generate(self.out,self.data)
        if not args.no_final:
            for arm in ARMS:
                dest=self.out/f'final_{arm}';dest.mkdir(exist_ok=True)
                if (dest/'metadata.json').exists():continue
                print(f'Final {arm} selection on all86 (not a test score)',flush=True)
                chosen=self.choose(arm,np.arange(86),SEED+999000,dest)
                model,cal=self.calibrated(arm,chosen['config'],np.arange(86),SEED+999100)
                joblib.dump(model,dest/'model.joblib')
                dump(dest/'metadata.json',{'arm':arm,'selected':chosen,'classes':['HC','Dementia'],'training_ids':self.ids.tolist(),
                     'calibration_splits':cal,'probability_threshold':model.probability_threshold,'policy':POLICY,'not_test_performance':True})


def predict(args):
    model=joblib.load(args.model);meta=json.loads(Path(args.model).with_name('metadata.json').read_text());arm=meta['arm']
    if args.recording:
        with contextlib.redirect_stdout(sys.stderr):
            maps=dict(np.load(Path(args.data)/'operators.npz'))
            tensors,_=extract({'path':args.recording},maps)
        x=tensors[arm].reshape(1,-1);who=Path(args.recording).stem;site=args.site;included=None
        if not site:raise ValueError('Provide --site AR or CL (or a new-site name)')
    else:
        plan=json.loads((Path(args.data)/'plan.json').read_text());ids=[p['subject_id'] for p in plan['participants']]
        i=ids.index(args.subject_id);x=np.load(Path(args.data)/'features.npz')[arm][i:i+1]
        who=args.subject_id;site=plan['participants'][i]['site'];included=who in meta['training_ids']
    p=model.predict_proba(x,[site])[0]
    print(json.dumps({'participant':who,'arm':arm,'probabilities':{'HC':float(p[0]),'Dementia':float(p[1])},
         'dementia_probability_operating_threshold':model.probability_threshold,'included_in_training':included,
         'uninformative_probability_calibration':model.calibrator_.uninformative_,
         'scope':'BrainLat-only internal research model; binary AD+FTD versus HC'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    data=str(HERE/'outputs/prepared');out=str(HERE/'outputs/matched')
    p=sub.add_parser('prepare');p.add_argument('--data',default=data);p.add_argument('--workers',type=int,default=2)
    p=sub.add_parser('train');p.add_argument('--data',default=data);p.add_argument('--out',default=out)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--no-final',action='store_true')
    p=sub.add_parser('report');p.add_argument('--data',default=data);p.add_argument('--out',default=out)
    p=sub.add_parser('predict');p.add_argument('--data',default=data);p.add_argument('--model',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--recording');g.add_argument('--subject-id');p.add_argument('--site')
    args=parser.parse_args()
    with threadpool_limits(limits=getattr(args,'threads',1)):
        if args.command=='prepare':prepare(args)
        elif args.command=='train':Experiment(args).run()
        elif args.command=='predict':predict(args)
        else:
            from report import generate
            generate(Path(args.out),Path(args.data))


if __name__=='__main__':main()
