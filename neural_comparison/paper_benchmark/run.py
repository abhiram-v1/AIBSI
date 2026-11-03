"""Participant-disjoint, nested literature-adaptation benchmark on all174 people."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from prepare import OUT as DATA, HERE, ROOT, sha, write_json
from models import CP6, HierarchicalSVM, StatisticalSelector, DICENet, MicrostateNFNet, adaptive_gradient_clip, fit_templates, microstate_sequences

METHODS = ['li_cp6_gnb', 'rostamikia_hierarchical_svm', 'dice_net_3class', 'jiang_microstate_cascade']
CLASSES = ['AD', 'FTD', 'HC']


def metric(y, pred):
    return {'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'per_class':classification_report(y,pred,labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0)}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)


def folds(y,sites,n,seed):
    strata=np.array([f'{s}:{v}' for s,v in zip(sites,y)])
    assert min(np.unique(strata,return_counts=True)[1])>=n
    return list(StratifiedKFold(n,shuffle=True,random_state=seed).split(y,strata))


def heldout_selection_split(fit,y,sites,seed):
    strata=np.array([f'{sites[i]}:{y[i]}' for i in fit])
    a,b=next(StratifiedShuffleSplit(1,test_size=.2,random_state=seed).split(fit,strata))
    return fit[a],fit[b]


def tensor_arm(x,y,sites,fit,test,folder,seed):
    smooth=[1e-9,1e-7,1e-5,1e-3]
    rows={s:[] for s in smooth};audits=[]
    for j,(a,b) in enumerate(folds(y[fit],sites[fit],3,seed)):
        train,valid=fit[a],fit[b]
        model=CP6(seed=seed+j).fit(x[train])
        tr,va=model.transform(x[train]),model.transform(x[valid])
        for s in smooth:
            pred=GaussianNB(var_smoothing=s).fit(tr,y[train]).predict(va)
            rows[s].append(float(f1_score(y[valid],pred,average='macro',zero_division=0)))
        audits.append({'train':train.tolist(),'validation':valid.tolist()})
        print(f'  CP6 inner {j+1}/3 complete',flush=True)
    chosen=max(smooth,key=lambda s:(np.mean(rows[s]),-s))
    representation=CP6(seed=seed+100).fit(x[fit])
    clf=GaussianNB(var_smoothing=chosen).fit(representation.transform(x[fit]),y[fit])
    p=clf.predict_proba(representation.transform(x[test]));pred=p.argmax(1)
    joblib.dump({'representation':representation,'classifier':clf,'train_indices':fit},folder/'model.joblib')
    return pred,{'selected_var_smoothing':chosen,'inner_scores':rows,'inner_splits':audits,
                 'probabilities':p.tolist(),'optimization':representation.factorizer_.diagnostics_}


def handcrafted_arm(x,y,sites,fit,test,folder,seed):
    grid=list(itertools.product([25,100,'all'],[.03,.1,1.,10.],['linear','rbf']))
    scores={str(g):[] for g in grid};audits=[]
    for j,(a,b) in enumerate(folds(y[fit],sites[fit],3,seed)):
        train,valid=fit[a],fit[b]
        stages=[]
        for stage in range(2):
            use=train if stage==0 else train[y[train]!=2]
            target=(y[use]!=2).astype(int) if stage==0 else y[use]
            selector=StatisticalSelector('all').fit(x[use],target)
            stages.append((use,target,selector.indices_))
        for k in [25,100,'all']:
            inputs=[]
            for use,target,rank in stages:
                selected=rank if k=='all' else rank[:k]
                scale=StandardScaler().fit(x[use][:,selected])
                inputs.append((scale.transform(x[use][:,selected]),target,scale.transform(x[valid][:,selected])))
            for c,kernel in itertools.product([.03,.1,1.,10.],['linear','rbf']):
                predictions=[SVC(C=c,kernel=kernel,class_weight='balanced',gamma='scale').fit(a,t).predict(b) for a,t,b in inputs]
                pred=np.where(predictions[0]==0,2,predictions[1])
                scores[str((k,c,kernel))].append(float(f1_score(y[valid],pred,average='macro',zero_division=0)))
        audits.append({'train':train.tolist(),'validation':valid.tolist()})
        print(f'  Handcrafted inner {j+1}/3 complete',flush=True)
    chosen=max(grid,key=lambda g:(np.mean(scores[str(g)]),-(x.shape[1] if g[0]=='all' else g[0]),-g[1],g[2]=='linear'))
    model=HierarchicalSVM(k=chosen[0],C=chosen[1],kernel=chosen[2]).fit(x[fit],y[fit])
    joblib.dump({'model':model,'train_indices':fit},folder/'model.joblib')
    return model.predict(x[test]),{'selected':{'k':chosen[0],'C':chosen[1],'kernel':chosen[2]},
             'inner_scores':scores,'inner_splits':audits,
             'stage_selected_features':[p.named_steps['selection'].indices_.tolist() for p in model.models_]}


@torch.no_grad()
def evaluate_windows(model,x,owners,people,batch=64):
    model.eval();indices=np.flatnonzero(np.isin(owners,people));probs=[]
    for start in range(0,len(indices),batch):
        values=model(x[torch.as_tensor(indices[start:start+batch],device=x.device)])
        probs.append(values.softmax(-1).float().cpu().numpy())
    p=np.concatenate(probs)
    means=np.stack([p[owners[indices]==i].mean(0) for i in people])
    votes=np.stack([np.bincount(p[owners[indices]==i].argmax(1),minlength=p.shape[1])/np.sum(owners[indices]==i) for i in people])
    return means,votes


def fit_neural(model_factory,x,owners,y,train,valid,seed,max_epochs,patience,kind,stage,folder,fixed_epochs=None):
    """Only train enters gradients; only valid enters stopping. No outer-test access."""
    seed_all(seed)
    model=model_factory().to(x.device)
    use=train if stage!=2 else train[y[train]!=2]
    val=valid if stage!=2 else valid[y[valid]!=2]
    targets=y.copy() if stage in [0,2] else (y!=2).astype(int)
    classes=3 if stage==0 else 2
    indices=np.flatnonzero(np.isin(owners,use))
    counts=np.bincount(owners[indices],minlength=len(y))
    weights=1./counts[owners[indices]]
    sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(indices),replacement=True,
                                  generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(TensorDataset(torch.as_tensor(indices)),batch_size=32,sampler=sampler,drop_last=True)
    label_tensor=torch.as_tensor(targets[owners],device=x.device,dtype=torch.long)
    class_counts=np.bincount(targets[use],minlength=classes)
    cw=torch.as_tensor(len(use)/(classes*class_counts),device=x.device,dtype=torch.float32)
    criterion=torch.nn.CrossEntropyLoss(weight=cw)
    optimizer=(torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01) if kind=='dice' else
               torch.optim.Adam(model.parameters(),lr=1e-4))
    best_score=-np.inf;best_state=None;best_epoch=1;history=[];stale=0
    epochs=fixed_epochs if fixed_epochs is not None else max_epochs
    start_time=time.time()
    for epoch in range(1,epochs+1):
        model.train();total=0.
        for (batch,) in loader:
            batch=batch.to(x.device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(x[batch]);loss=criterion(logits,label_tensor[batch])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural loss')
            loss.backward()
            if kind=='micro':adaptive_gradient_clip(model,.04)
            optimizer.step();total+=float(loss.detach())
        record={'epoch':epoch,'training_loss':total/max(len(loader),1)}
        if fixed_epochs is None:
            p,votes=evaluate_windows(model,x,owners,val)
            score=float(f1_score(targets[val],votes.argmax(1) if kind=='micro' else p.argmax(1),
                                 labels=list(range(classes)),average='macro',zero_division=0))
            record['validation_macro_f1']=score
            if score>best_score+1e-8:
                best_score=score;best_epoch=epoch;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
        history.append(record)
        if epoch==1 or epoch%10==0:
            print(f'  {kind} stage{stage} {"refit" if fixed_epochs else "selection"} epoch{epoch}/{epochs}: '
                  f'loss={record["training_loss"]:.4f}, valF1={record.get("validation_macro_f1","n/a")}, elapsed={time.time()-start_time:.0f}s',flush=True)
        if fixed_epochs is None and stale>=patience:break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    # An evaluation batch must not mix participants inside the transformer axis.
    model.eval()
    with torch.no_grad():
        probe=x[indices[:3]]
        combined=model(probe)
        separate=torch.cat([model(probe[i:i+1]) for i in range(len(probe))])
        batch_error=float((combined-separate).abs().max())
        if not torch.allclose(combined,separate,atol=2e-4,rtol=2e-4):
            raise AssertionError(f'Inference depends on batch membership: {batch_error}')
    folder.mkdir(exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'training_subjects':use.tolist(),'stage':stage,
                'epochs':epochs if fixed_epochs else best_epoch,'seed':seed},folder/'weights.pt')
    write_json(folder/'history.json',{'history':history,'selected_epoch':best_epoch if fixed_epochs is None else fixed_epochs,
               'validation_subjects':val.tolist() if fixed_epochs is None else [],'training_subjects':use.tolist(),
               'batch_invariance_max_error':batch_error})
    return model,best_epoch if fixed_epochs is None else fixed_epochs


def dice_arm(y,sites,fit,test,folder,seed,args):
    raw=np.load(DATA/'dice_native.npy',mmap_mode='r');owners=np.load(DATA/'dice_native_owners.npy')
    train,valid=heldout_selection_split(fit,y,sites,seed)
    # Mean/SD fit exclusively on corresponding training windows, time axis pooled.
    def make_x(people):
        chosen=np.flatnonzero(np.isin(owners,people))
        block=np.asarray(raw[chosen],float)
        mean=block.mean(axis=(0,4),keepdims=True).astype(np.float32)
        sd=np.maximum(block.std(axis=(0,4),keepdims=True),1e-5).astype(np.float32)
        data=np.clip((np.asarray(raw,dtype=np.float32)-mean)/sd,-10,10)
        return torch.as_tensor(data,device='cuda'),mean,sd
    x,_,_=make_x(train)
    model,epoch=fit_neural(DICENet,x,owners,y,train,valid,seed,args.dice_epochs,20,'dice',0,folder/'selection')
    del model,x;torch.cuda.empty_cache()
    x,mean,sd=make_x(fit)
    model,_=fit_neural(DICENet,x,owners,y,fit,np.array([],int),seed+1,args.dice_epochs,20,'dice',0,
                       folder/'refit',fixed_epochs=epoch)
    p,votes=evaluate_windows(model,x,owners,test)
    np.savez_compressed(folder/'scaler.npz',mean=mean,sd=sd)
    del model,x;torch.cuda.empty_cache()
    return p.argmax(1),{'stopping_train':train.tolist(),'stopping_validation':valid.tolist(),'refit_train':fit.tolist(),
                       'selected_epochs':epoch,'probabilities':p.tolist(),'epoch_majority_predictions':votes.argmax(1).tolist()}


def micro_arm(y,sites,fit,test,folder,seed,args):
    owners=np.load(DATA/'owners.npy');peaks=np.load(DATA/'micro_peaks.npy',mmap_mode='r')
    maps=np.load(DATA/'micro_maps.npy',mmap_mode='r')
    train,valid=heldout_selection_split(fit,y,sites,seed)
    templates,diagnostics=fit_templates(peaks,owners,train,seed=seed)
    seq=microstate_sequences(maps,templates)
    epochs=[]
    for stage,channels in [(1,[8,9,10,11,0,1,2,3]),(2,[0,1,2,3,4,5,6,7])]:
        x=torch.as_tensor(seq[:,channels],device='cuda')
        model,epoch=fit_neural(MicrostateNFNet,x,owners,y,train,valid,seed+stage,args.micro_epochs,20,'micro',stage,
                              folder/f'selection_stage{stage}')
        epochs.append(epoch);del model,x;torch.cuda.empty_cache()
    np.savez_compressed(folder/'selection_templates.npz',templates=templates,training_subjects=train)
    templates,final_diagnostics=fit_templates(peaks,owners,fit,seed=seed+100)
    seq=microstate_sequences(maps,templates)
    outputs=[]
    for j,(stage,channels) in enumerate([(1,[8,9,10,11,0,1,2,3]),(2,[0,1,2,3,4,5,6,7])]):
        x=torch.as_tensor(seq[:,channels],device='cuda')
        model,_=fit_neural(MicrostateNFNet,x,owners,y,fit,np.array([],int),seed+100+stage,args.micro_epochs,20,
                           'micro',stage,folder/f'refit_stage{stage}',fixed_epochs=epochs[j])
        outputs.append(evaluate_windows(model,x,owners,test));del model,x;torch.cuda.empty_cache()
    np.savez_compressed(folder/'templates.npz',templates=templates,training_subjects=fit)
    pd=outputs[0][0][:,1];pf=outputs[1][0][:,1]
    p=np.column_stack([pd*(1-pf),pd*pf,1-pd])
    pred=np.where(outputs[0][1].argmax(1)==0,2,outputs[1][1].argmax(1))
    return pred,{'stopping_train':train.tolist(),'stopping_validation':valid.tolist(),'refit_train':fit.tolist(),
        'selected_epochs':epochs,'probabilities':p.tolist(),'probability_argmax_predictions':p.argmax(1).tolist(),
        'selection_template_diagnostics':diagnostics,'refit_template_diagnostics':final_diagnostics,
        'decision':'Per-participant epoch majority at each binary stage; ties select lower class index'}


def report(out,manifest,y):
    summary={};rng=np.random.default_rng(20261103)
    for method in METHODS:
        files=sorted(out.glob(f'fold_*/{method}/result.json'))
        if len(files)!=5:continue
        pred=np.full(len(y),-1,int);coverage=np.zeros(len(y),int);audits=[]
        for path in files:
            r=json.loads(path.read_text());idx=np.array(r['test_indices'])
            assert not set(r['train_indices'])&set(idx)
            pred[idx]=r['predicted'];coverage[idx]+=1
            for inner in r.get('details',{}).get('inner_splits',[]):
                assert not set(inner['train'])&set(inner['validation'])
                assert not (set(inner['train'])|set(inner['validation']))&set(idx)
            if 'stopping_train' in r['details']:
                a,b=set(r['details']['stopping_train']),set(r['details']['stopping_validation'])
                assert not a&b and not (a|b)&set(idx) and a|b==set(r['train_indices'])
            audits.append({'fold':r['fold'],'test_subjects':len(idx),'train_test_overlap':0})
        assert np.all(coverage==1) and np.all(pred>=0)
        m=metric(y,pred)
        bootstrap=[]
        for _ in range(1000):
            sample=np.concatenate([rng.choice(np.flatnonzero(y==k),np.sum(y==k),replace=True) for k in range(3)])
            bootstrap.append([accuracy_score(y[sample],pred[sample]),balanced_accuracy_score(y[sample],pred[sample]),
                              f1_score(y[sample],pred[sample],average='macro',zero_division=0)])
        m['conditional_bootstrap95']=dict(zip(['accuracy','balanced_accuracy','macro_f1'],np.quantile(bootstrap,[.025,.975],axis=0).T.tolist()))
        m['by_site']={site:metric(y[np.array([p['site']==site for p in manifest['participants']])],
                                pred[np.array([p['site']==site for p in manifest['participants']])])
                      for site in sorted({p['site'] for p in manifest['participants']})}
        summary[method]=m
        write_json(out/f'{method}_oof.json',[{'subject_id':p['subject_id'],'true':CLASSES[int(y[i])],
                    'predicted':CLASSES[int(pred[i])],'site':p['site']} for i,p in enumerate(manifest['participants'])])
    write_json(out/'summary.json',summary)
    lines=['# Literature method adaptations: full174-person, three-class benchmark','',
           'Five shared outer participant folds; each participant predicted once. Classical tuning uses three inner folds; neural stopping uses an inner participant split followed by refitting on all outer-training participants.',
           'These are documented reimplementations/adaptations, not exact reproductions of the papers. No external validation. The cohort has been explored in prior experiments.', '',
           '| Method | Accuracy | Balanced accuracy | Macro F1 | AD recall | FTD recall | HC recall |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for method,m in summary.items():
        vals=[m['accuracy'],m['balanced_accuracy'],m['macro_f1']]+[m['per_class'][k]['recall'] for k in CLASSES]
        lines.append('| '+method+' | '+' | '.join(f'{100*v:.1f}%' for v in vals)+' |')
    lines+=['','Completed methods: '+str(len(summary))+'/4.',
            'See README.md and protocol.json for source papers, implementation choices and deviations. Bootstrap intervals are conditional on these fitted predictions; overlapping training folds do not provide independent replications.',
            'Moderate preprocessing inherits missing explicit EEGLAB boundary rejection. This is a signal-quality limitation, separate from train/test participant leakage.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5),layout='constrained');positions=np.arange(len(summary))
        for j,key in enumerate(['accuracy','balanced_accuracy','macro_f1']):
            bars=ax.bar(positions+(j-1)*.25,[100*v[key] for v in summary.values()],.25,label=key.replace('_',' '))
            ax.bar_label(bars,fmt='%.1f',fontsize=8)
        ax.set(xticks=positions,xticklabels=[m.replace('_','\n') for m in summary],ylim=(0,100),ylabel='Score (%)',
               title='174 participants: three classes, shared participant-held-out folds')
        ax.legend();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'outputs/full_174')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--only-fold',type=int)
    parser.add_argument('--seed',type=int,default=20261102)
    parser.add_argument('--dice-epochs',type=int,default=80)
    parser.add_argument('--micro-epochs',type=int,default=100)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    d=np.load(DATA/'subjects.npz');y,sites=d['y'],d['sites']
    assert len(y)==174 and np.bincount(y).tolist()==[71,42,61]
    outer=folds(y,sites,5,args.seed)
    if args.report_only:
        report(args.out,manifest,y);return
    protocol={'participants':174,'classes':CLASSES,'class_counts':np.bincount(y).tolist(),'outer_folds':5,'repeats':1,
        'seed':args.seed,'classical_inner_folds':3,'neural_validation_fraction':.2,'neural_patience':20,
        'dice_epochs_ceiling':args.dice_epochs,'micro_epochs_ceiling':args.micro_epochs,'methods':METHODS,
        'data_manifest_sha256':sha(DATA/'manifest.json'),
        'native_and_micro_input_hashes':{name:sha(DATA/name) for name in
            ['dice_native.npy','dice_native_owners.npy','dice_native_protocol.json','micro_maps.npy','micro_peaks.npy','micro_maps_protocol.json']},
        'code_hashes':{p.name:sha(p) for p in HERE.glob('*.py')},
        'outer_splits':[{'train':a.tolist(),'test':b.tolist(),
            'train_ids':[manifest['participants'][i]['subject_id'] for i in a],
            'test_ids':[manifest['participants'][i]['subject_id'] for i in b]} for a,b in outer],
        'scope':'Full174-person cohort, internal validation, literature adaptations; no independent external validation.',
        'neural_stopping':'Validation participant macroF1; refit on all outer train for selected epochs. No outer-test early stopping.',
        'micro_paper_ambiguity':'Main text Adam1e-4/AGC.04 conflicts with supplement SGD/cosine/AGC.01; main text optimization frozen here, supplement layer widths used.',
        'calibration':'None added; accuracy based on defined class decisions. Softmax/posterior scores are not claimed clinically calibrated.'}
    if (args.out/'protocol.json').exists() and json.loads((args.out/'protocol.json').read_text())!=protocol:
        raise RuntimeError('Protocol/code/input changed; use a different output directory')
    if not (args.out/'protocol.json').exists():write_json(args.out/'protocol.json',protocol)
    if any(m in args.methods for m in METHODS[2:]) and not torch.cuda.is_available():raise RuntimeError('CUDA required for neural arms')
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        for fold,(fit,test) in enumerate(outer):
            if args.only_fold is not None and args.only_fold!=fold:continue
            for method in args.methods:
                folder=args.out/f'fold_{fold:02d}'/method;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'result.json').exists():continue
                print(f'FOLD {fold+1}/5: {method}; train={len(fit)}, test={len(test)}',flush=True)
                start=time.time();seed=args.seed+fold*101
                if method==METHODS[0]:pred,details=tensor_arm(d['psd_db'],y,sites,fit,test,folder,seed)
                elif method==METHODS[1]:pred,details=handcrafted_arm(d['handcrafted'],y,sites,fit,test,folder,seed)
                elif method==METHODS[2]:pred,details=dice_arm(y,sites,fit,test,folder,seed,args)
                else:pred,details=micro_arm(y,sites,fit,test,folder,seed,args)
                write_json(folder/'result.json',{'method':method,'fold':fold,'train_indices':fit.tolist(),
                    'test_indices':test.tolist(),'true':y[test].tolist(),'predicted':np.asarray(pred).tolist(),
                    'metrics':metric(y[test],pred),'details':details,'elapsed_seconds':time.time()-start})
                print(f'  completed {method}: {metric(y[test],pred)}',flush=True)
        report(args.out,manifest,y)


if __name__=='__main__':main()
