"""CLI: prepare, connectivity, train, report, predict. See README for protocols."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,log_loss,recall_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedKFold,LeaveOneOut
from threadpoolctl import threadpool_limits

from core import FeatureModel,ProbabilityModel,calibrators_from_oof,electrode_graph,make_svm
from data import HERE,CLASSES,CHANNELS,prepare,prepare_connectivity,write_json,fingerprint,raw_windows,mel_from_raw,summarize,connectivity_from_raw


def participant_folds(labels,sites,n,seed):
    strata=np.array([f'{s}:{v}' for s,v in zip(sites,labels)])
    if min(np.unique(strata,return_counts=True)[1])<n:
        raise ValueError(f'Site/diagnosis strata too small for {n} folds; change fold count explicitly')
    return list(StratifiedKFold(n,shuffle=True,random_state=seed).split(labels,strata))


def metrics(y,p):
    pred=p.argmax(1)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'ftd_recall':float(recall_score(y,pred,labels=[1],average='macro',zero_division=0)),
            'log_loss':float(log_loss(y,p,labels=[0,1,2])),
            'brier_multiclass_sum':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),
            'macro_auroc':float(roc_auc_score(y,p,multi_class='ovr',average='macro')) if len(np.unique(y))==3 else None,
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'per_class':classification_report(y,pred,labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0)}


def feature_grid(method,profile):
    ranks=[3,8] if profile=='quick' else [3,5,8,12,16]
    if method=='direct':
        return [{'method':method,'pca':v} for v in ([20] if profile=='quick' else [None,20,40,60])]
    if method=='tt':
        return [{'method':method,'rank':r} for r in ranks]
    lambdas=[0.] if method=='ntf' else ([.01,.1] if profile=='quick' else [.001,.01,.1,1.])
    couplings=([.1] if profile=='quick' else [.1,1.]) if method=='coupled' else [0.]
    return [{'method':method,'rank':r,'lam':l,'coupling':c} for r,l,c in itertools.product(ranks,lambdas,couplings)]


def svm_grid(profile):
    cs=[.1,1,10] if profile=='quick' else [.03,.1,.3,1,3,10]
    for c in cs:
        yield {'kernel':'linear','C':c,'gamma':'scale'}
        for gamma in (['scale'] if profile=='quick' else ['scale',.01,.1,1.]):
            yield {'kernel':'rbf','C':c,'gamma':gamma}


class Experiment:
    def __init__(self,dataset,out,profile,maxiter,starts,inner):
        self.directory=Path(dataset);self.out=Path(out);self.profile=profile
        self.maxiter,self.starts,self.inner=maxiter,starts,inner
        d=np.load(self.directory/'dataset.npz')
        self.x,self.direct,self.y,self.sites=d['x'],d['direct'],d['y'],d['sites']
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        self.ids=np.array([p['subject_id'] for p in self.manifest['participants']])
        self.adjacency,self.lap=electrode_graph(d['positions'])
        self.conn=np.load(self.directory/'connectivity.npy') if (self.directory/'connectivity.npy').exists() else None
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/'cache').mkdir(exist_ok=True)

    def feature_fit(self,cfg,fit):
        # Full membership/config determines seed and cache; no held-out rows enter fit.
        key=hashlib.sha256(json.dumps({'cfg':cfg,'fit':fit.tolist(),'maxiter':self.maxiter,'starts':self.starts},sort_keys=True).encode()).hexdigest()
        file=self.out/'cache'/f'{key}.joblib'
        if file.exists():return joblib.load(file)
        model=FeatureModel(cfg,self.lap,self.maxiter,self.starts,int(key[:7],16))
        model.fit(self.x[fit],self.sites[fit],self.direct[fit],None if self.conn is None else self.conn[fit])
        joblib.dump(model,file)
        return model

    def transform(self,model,ids):
        return model.transform(self.x[ids],self.sites[ids],self.direct[ids],None if self.conn is None else self.conn[ids])

    def select(self,method,development,seed):
        folds=participant_folds(self.y[development],self.sites[development],self.inner,seed)
        scores=[]
        grid=feature_grid(method,self.profile)
        for position,cfg in enumerate(grid):
            results=[{'features':cfg,'svm':s,'f1':[],'balanced':[],'ftd':[]} for s in svm_grid(self.profile)]
            for train_local,valid_local in folds:
                fit,valid=development[train_local],development[valid_local]
                model=self.feature_fit(cfg,fit)
                a,b=self.transform(model,fit),self.transform(model,valid)
                for row in results:
                    clf=make_svm(row['svm']).fit(a,self.y[fit])
                    pred=clf.predict(b)
                    row['f1'].append(float(f1_score(self.y[valid],pred,average='macro',zero_division=0)))
                    row['balanced'].append(float(balanced_accuracy_score(self.y[valid],pred)))
                    row['ftd'].append(float(recall_score(self.y[valid],pred,labels=[1],average='macro',zero_division=0)))
            for row in results:
                row['mean_f1']=float(np.mean(row['f1']))
                row['se_f1']=float(np.std(row['f1'],ddof=1)/np.sqrt(len(folds)))
            scores.extend(results)
            print(f'  {method} tensor/config {position+1}/{len(grid)}',flush=True)
        best=max(scores,key=lambda r:(r['mean_f1'],np.mean(r['balanced']),np.mean(r['ftd'])))
        threshold=best['mean_f1']-best['se_f1']
        eligible=[s for s in scores if s['mean_f1']>=threshold]
        chosen=min(eligible,key=lambda r:(r['svm']['kernel']!='linear',r['features'].get('rank',r['features'].get('pca') or 912),
                                         -r['mean_f1'],-np.mean(r['balanced']),-np.mean(r['ftd']),r['svm']['C']))
        # One-SE is a model-selection heuristic, not a significance test.
        return chosen,{'best':best,'one_se_threshold':threshold,'candidates':scores,
                       'inner_splits':[{'train':self.ids[development[a]].tolist(),'val':self.ids[development[b]].tolist()} for a,b in folds]}

    def calibrated_fit(self,chosen,development,seed):
        score=np.empty((len(development),3),float)
        folds=participant_folds(self.y[development],self.sites[development],self.inner,seed)
        audit=[]
        for a,b in folds:
            fit,valid=development[a],development[b]
            fm=self.feature_fit(chosen['features'],fit)
            svm=make_svm(chosen['svm']).fit(self.transform(fm,fit),self.y[fit])
            score[b]=svm.decision_function(self.transform(fm,valid))
            audit.append({'train':self.ids[fit].tolist(),'calibration':self.ids[valid].tolist()})
        calibrators=calibrators_from_oof(score,self.y[development])
        features=self.feature_fit(chosen['features'],development)
        svm=make_svm(chosen['svm']).fit(self.transform(features,development),self.y[development])
        return ProbabilityModel(features,svm,calibrators),audit

    def train(self,args):
        if 'coupled' in args.methods and self.conn is None:
            raise ValueError('Run connectivity preparation first for the coupled arm')
        files=[self.directory/'dataset.npz',self.directory/'manifest.json',*sorted(HERE.glob('*.py'))]
        if 'coupled' in args.methods:files.append(self.directory/'connectivity.npy')
        protocol={'profile':self.profile,'outer_folds':args.outer_folds,'repeats':args.repeats,'inner_folds':self.inner,
                  'outer':args.outer,'methods':args.methods,'maxiter':self.maxiter,'starts':self.starts,
                  'seed':args.seed,'input_hashes':{str(p):fingerprint(p) for p in files},
                  'calibration':'3 one-vs-rest sigmoids on full-pipeline OOF scores; renormalized to sum1; no outer labels used',
                  'selection':'Inner macro F1, one-SE simplicity heuristic; uncalibrated SVM.predict used for tuning, probability argmax for reported endpoints',
                  'uncertainty':'Participant bootstrap conditional on fitted OOF models, repeated predictions kept together; not independent repeat observations',
                  'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__},
                  'interpretation':'Internal validation on a previously explored cohort; no new external validation',
                  'representation':'subject x electrode x mel x statistic; no explicit time axis',
                  'graph':'4-nearest-neighbour symmetrized distance-weighted scalp graph, Laplacian spectral norm1',
                  'rank_grid':feature_grid('gntf',self.profile),'svm_grid':list(svm_grid(self.profile))}
        protocol_path=self.out/'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
            raise ValueError('Run directory protocol differs; select a new --out directory')
        write_json(protocol_path,protocol)
        records=[]
        partial_run=args.only_repeat is not None or args.only_methods is not None
        for repeat in range(args.repeats if args.outer!='loso' else 1):
            if args.only_repeat is not None and repeat!=args.only_repeat:continue
            splits=(list(LeaveOneOut().split(self.y)) if args.outer=='loso' else
                    participant_folds(self.y,self.sites,args.outer_folds,args.seed+repeat*101))
            for fold,(fit,test) in enumerate(splits):
                for method in args.methods:
                    if args.only_methods is not None and method not in args.only_methods:continue
                    if args.only_fold is not None and fold!=args.only_fold:continue
                    folder=self.out/f'repeat_{repeat:02d}_fold_{fold:03d}'/method
                    folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'result.json').exists():
                        records.append(json.loads((folder/'result.json').read_text()));continue
                    print(f'Outer repeat {repeat+1}/{args.repeats}, fold {fold+1}/{len(splits)}: {method}',flush=True)
                    chosen,search=self.select(method,fit,args.seed+repeat*1000+fold)
                    model,calibration=self.calibrated_fit(chosen,fit,args.seed+100000+repeat*1000+fold)
                    p=model.predict_proba(self.x[test],self.sites[test],self.direct[test],None if self.conn is None else self.conn[test])
                    info={'repeat':repeat,'fold':fold,'method':method,'train_ids':self.ids[fit].tolist(),
                          'test_ids':self.ids[test].tolist(),'test_indices':test.tolist(),'true':self.y[test].tolist(),
                          'probabilities':p.tolist(),'selected':chosen,'calibration_splits':calibration}
                    if len(np.unique(self.y[test]))==3:info['metrics']=metrics(self.y[test],p)
                    if model.features.decomposition is not None:
                        dec=model.features.decomposition
                        info['optimization']={'objective':dec.objective_,'relative_reconstruction_error':dec.reconstruction_error_,
                                              'graph_roughness':dec.graph_roughness_,'starts':dec.diagnostics_}
                    joblib.dump(model,folder/'model.joblib')
                    write_json(folder/'search.json',search)
                    write_json(folder/'result.json',info)
                    records.append(info)
                    print('  complete',info.get('metrics',{}).get('macro_f1','LOSO probability recorded'),flush=True)
        if args.only_fold is not None:partial_run=True
        if partial_run:
            # Parallel worker: results are on disk; a final unrestricted call merges them.
            return
        write_json(self.out/'oof_predictions.json',records)
        from report import generate
        generate(self.out,self.directory)
        if not args.no_final:
            for method in ['gntf']+(['coupled'] if 'coupled' in args.methods else []):
                if method not in args.methods:continue
                dest=self.out/f'final_{method}'
                if (dest/'model.joblib').exists():continue
                print(f'Selecting deployment {method} using inner CV on all participants',flush=True)
                all_indices=np.arange(len(self.y))
                chosen,search=self.select(method,all_indices,args.seed+999000)
                model,calibration=self.calibrated_fit(chosen,all_indices,args.seed+999100)
                dest.mkdir(exist_ok=True)
                joblib.dump(model,dest/'model.joblib')
                write_json(dest/'metadata.json',{'classes':CLASSES,'channels':CHANNELS,'training_ids':self.ids.tolist(),
                    'selected':chosen,'calibration_splits':calibration,'input_protocol':'moderate8s128Hz_logmel24',
                    'final_fit_is_not_test_performance':True})
                write_json(dest/'search.json',search)
            generate(self.out,self.directory)


def predict(args):
    # MNE and the harmonizer print progress lines; keep stdout as clean JSON for piping.
    with contextlib.redirect_stdout(sys.stderr):
        payload=_predict_payload(args)
    print(json.dumps(payload,indent=2,allow_nan=False))


def _predict_payload(args):
    path=Path(args.model)
    model=joblib.load(path)
    metadata=json.loads(path.with_name('metadata.json').read_text())
    if args.recording:
        if args.source is None or args.site is None:raise ValueError('--source and --site required for .set recordings')
        windows,_=raw_windows(args.recording,args.source,args.site)
        x,direct=summarize(mel_from_raw(windows))
        conn=connectivity_from_raw(windows)[None] if model.features.config['method']=='coupled' else None
        x,direct=x[None],direct[None]
        who=Path(args.recording).stem
    else:
        if args.subject_id is None:raise ValueError('Supply --recording or --subject-id with --data')
        manifest=json.loads((Path(args.data)/'manifest.json').read_text())
        people=manifest['participants'];ids=[p['subject_id'] for p in people]
        index=ids.index(args.subject_id);d=np.load(Path(args.data)/'dataset.npz')
        x,direct=d['x'][index:index+1],d['direct'][index:index+1]
        args.site=str(d['sites'][index]);who=args.subject_id
        conn=np.load(Path(args.data)/'connectivity.npy')[index:index+1] if model.features.config['method']=='coupled' else None
    probabilities=model.predict_proba(x,[args.site],direct,conn)[0]
    return {'participant':who,'probabilities':dict(zip(CLASSES,probabilities.tolist())),
            'included_in_model_training':args.subject_id in metadata['training_ids'] if args.subject_id else None,
            'known_training_site':args.site in model.features.site_means_,
            'scope':'Research cohort class probabilities; not clinically validated diagnostic probabilities'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    default_data=str(HERE/'outputs/prepared')
    for name in ['prepare','connectivity']:
        p=sub.add_parser(name);p.add_argument('--data',default=default_data)
    train=sub.add_parser('train')
    train.add_argument('--data',default=default_data)
    train.add_argument('--out',required=True)
    train.add_argument('--profile',choices=['quick','full'],default='full')
    train.add_argument('--methods',nargs='+',choices=['direct','ntf','gntf','shuffled','tt','coupled'],default=['direct','ntf','gntf','shuffled','tt'])
    train.add_argument('--outer',choices=['stratified','loso'],default='stratified')
    train.add_argument('--outer-folds',type=int,default=5)
    train.add_argument('--repeats',type=int,default=3)
    train.add_argument('--inner-folds',type=int,default=3)
    train.add_argument('--maxiter',type=int,default=250)
    train.add_argument('--starts',type=int,default=3)
    train.add_argument('--seed',type=int,default=20261102)
    train.add_argument('--threads',type=int,default=2)
    train.add_argument('--no-final',action='store_true')
    train.add_argument('--only-repeat',type=int,default=None,help='Parallel worker: run one repeat only')
    train.add_argument('--only-fold',type=int,default=None,help='Parallel worker: run one outer fold only')
    train.add_argument('--only-methods',nargs='+',default=None,help='Parallel worker: subset of --methods')
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--data',default=default_data)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--recording');group.add_argument('--subject-id')
    p.add_argument('--source',choices=['ds004504','brainlat']);p.add_argument('--site')
    p=sub.add_parser('report');p.add_argument('--out',required=True);p.add_argument('--data',default=default_data)
    args=parser.parse_args()
    with threadpool_limits(limits=getattr(args,'threads',2)):
        if args.command=='prepare':prepare(args.data)
        elif args.command=='connectivity':prepare_connectivity(args.data)
        elif args.command=='predict':predict(args)
        elif args.command=='report':
            from report import generate
            generate(Path(args.out),Path(args.data))
        else:Experiment(args.data,args.out,args.profile,args.maxiter,args.starts,args.inner_folds).train(args)


if __name__=='__main__':main()
