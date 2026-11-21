"""Freeze the original88-person binary method, then test all available BrainLat EEG."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits

from transfer_data import HERE,ROOT,V6,V3,V5,discover,target_one,check_source_replay,CHANNELS
from transfer_model import SourceModel

SELECTION_SEED=90660999
CALIBRATION_SEED=90660998
NAMES=['original_procedure','svm_only','spectral_svm_control']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def freeze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'source_frozen.json').exists():return json.loads((out/'source_frozen.json').read_text())
    subjects,original,banks=V6.load();y=(original!=2).astype(int)
    old_channels=np.load(V3.OUT/'index.npz')['channels'].tolist()
    aliases={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
    if [aliases.get(c,c) for c in old_channels]!=CHANNELS:raise ValueError('Old source channel order differs')
    rows=[];folds=list(StratifiedKFold(3,shuffle=True,random_state=SELECTION_SEED).split(y,y))
    for i,c in enumerate(V6.candidates()):
        pred=np.full(88,-1,int)
        for a,b in folds:
            model=V6.make_model(c).fit(banks[c['features']][a],y[a]);pred[b]=model.predict(banks[c['features']][b])
        rows.append({'candidate':i,'config':c,'source_selection_metrics':V6.metrics(y,pred)})
    result={'train_subjects':subjects.tolist(),'train_binary_counts':np.bincount(y).tolist(),
            'train_original_counts':np.bincount(original).tolist(),'source_selection_seed':SELECTION_SEED,
            'source_calibration_seed':CALIBRATION_SEED,'all_source_candidates':rows,'models':{},
            'source_inputs_sha256':V6.fingerprint([V6.V3/'index.npz',V6.V3/'spectral_controls.npy',V6.V5/'bags_1.npz']),
            'code_sha256':{str(p):sha(p) for p in [*HERE.glob('*.py'),Path(V6.__file__),Path(V3.__file__),Path(V5.__file__)]},
            'target_use':'No BrainLat features or labels enter feature selection, scaling, parameter selection, fitting, calibration or threshold setting.',
            'interpretation':'Source selection scores are tuning scores, not unbiased test accuracy. Original75% came from nested CV, not a single saved final model.'}
    rank=lambda r:(-r['source_selection_metrics']['macro_f1'],-r['source_selection_metrics']['dementia_recall'],
                   -r['source_selection_metrics']['balanced_accuracy'],r['candidate'])
    sets=[rows,[r for r in rows if r['config']['model']=='rbf_svm'],
          [r for r in rows if r['config']['model']=='rbf_svm' and r['config']['features']=='spectral']]
    for name,choices in zip(NAMES,sets):
        chosen=min(choices,key=rank);c=chosen['config'];x=banks[c['features']]
        scores=np.zeros(88);splits=list(StratifiedKFold(3,shuffle=True,random_state=CALIBRATION_SEED).split(y,y))
        for a,b in splits:
            pipeline=V6.make_model(c).fit(x[a],y[a]);scores[b]=pipeline.decision_function(x[b])
        cal=LogisticRegression(C=1e6,max_iter=2000).fit(scores[:,None],y)
        if cal.coef_[0,0]<=0:raise RuntimeError('Source calibration has no consistent score direction')
        pipeline=V6.make_model(c).fit(x,y);model=SourceModel(pipeline,cal,c)
        joblib.dump(model,out/f'{name}.joblib')
        result['models'][name]={'selected':chosen,'model_sha256':sha(out/f'{name}.joblib'),
            'source_calibration_splits':[{'train_ids':subjects[a].tolist(),'calibration_ids':subjects[b].tolist()} for a,b in splits],
            'probability_operating_threshold':model.probability_operating_threshold}
    # Persist the source-only choices and model hashes BEFORE target preparation.
    dump(out/'source_frozen.json',result)
    dump(out/'source_feature_replay.json',check_source_replay())
    print('Frozen source-only models:',json.dumps({k:v['selected']['config'] for k,v in result['models'].items()}),flush=True)
    return result


def prepare(args):
    out=Path(args.out);frozen=freeze(out)
    previous=ROOT/'neural_comparison/brainlat_native_ablation/outputs/prepared/operators.npz'
    mapping=np.load(previous)['legacy19']
    people=discover();dump(out/'target_inventory.json',people)
    if set(frozen['train_subjects'])&{p['subject_id'] for p in people}:raise ValueError('Train/test ID collision')
    # Different dataset namespaces are necessary but not sufficient for identity
    # disjointness. These datasets were independently acquired in different cohorts.
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(target_one,i,p,out,mapping) for i,p in enumerate(people)]
        for n,future in enumerate(as_completed(jobs),1):
            row=future.result();rows.append(row)
            print(f'BrainLat {n}/{len(people)} {row["subject_id"]}: {row["status"]}',flush=True)
    rows.sort(key=lambda r:r['index']);dump(out/'target_audit.json',rows)
    good=[r for r in rows if r['status']=='scored_features']
    if not good:raise RuntimeError('No target recordings have scorable EEG')
    arrays={k:[] for k in ['spectral','power','connectivity']}
    for row in good:
        with np.load(out/'target_parts'/f'{row["index"]:03d}.npz') as p:
            for k in arrays:arrays[k].append(p[k])
    np.savez_compressed(out/'target_features.npz',**{k:np.stack(v) for k,v in arrays.items()})
    dump(out/'target_manifest.json',good)
    dump(out/'transfer_protocol.json',{'source_frozen_sha256':sha(out/'source_frozen.json'),
         'target_feature_sha256':sha(out/'target_features.npz'),'mapping_sha256':sha(previous),
         'target_inventory_counts':dict(Counter(p['label'] for p in people)),
         'target_status_counts':dict(Counter(r['status'] for r in rows)),
         'source_only':True,'target_population':'All locally present AD, bvFTD and HC EEG headers; other diagnosis groups outside this binary task',
         'target_preprocessing':'Fixed legacy128-to19 spatial operator, 512-to250Hz anti-aliased resampling, microvolt units, four-second windows, original source-policy QC, all accepted windows, no test-set normalization',
         'reference_limitation':'Old spectral features use19-channel CAR. Old connectivity/power features were actually computed on temporally-demeaned original-reference windows. BrainLat cannot recreate AHEPA earlobe reference; the connectivity SVM is therefore explicitly reference-sensitive.',
         'scope':'Previously explored target dataset. Cross-dataset transfer test, not untouched external confirmation.'})


def evaluate(out):
    out=Path(out);frozen=json.loads((out/'source_frozen.json').read_text())
    people=json.loads((out/'target_manifest.json').read_text());audit=json.loads((out/'target_audit.json').read_text())
    x=dict(np.load(out/'target_features.npz'));y=np.array([int(p['label']!='HC') for p in people])
    results={};predictions={}
    for name in NAMES:
        info=frozen['models'][name];path=out/f'{name}.joblib'
        if sha(path)!=info['model_sha256']:raise ValueError('Frozen source model changed')
        model=joblib.load(path);matrix=x[model.config['features']]
        pred=model.predict(matrix);p=model.predict_proba(matrix)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        result=V6.metrics(y,pred);result.update(roc_auc=float(roc_auc_score(y,p[:,1])),log_loss=float(log_loss(y,p,labels=[0,1])))
        rng=np.random.default_rng(906600);boot=[]
        for _ in range(1500):
            idx=np.concatenate([rng.choice(np.flatnonzero(y==k),np.sum(y==k),replace=True) for k in [0,1]])
            m=V6.metrics(y[idx],pred[idx]);boot.append([m[k] for k in ['macro_f1','balanced_accuracy','accuracy']])
        result['conditional_bootstrap95']=dict(zip(['macro_f1','balanced_accuracy','accuracy'],np.quantile(boot,[.025,.975],axis=0).T.tolist()))
        result['source_config']=model.config;result['groups']={}
        for field in ['site','label']:
            result['groups'][field]={}
            for v in sorted({p[field] for p in people}):
                mask=np.array([p[field]==v for p in people])
                result['groups'][field][v]={'n':int(mask.sum()),'correct_binary_classification_rate':float(np.mean(pred[mask]==y[mask]))}
                if field=='site' and len(np.unique(y[mask]))==2:result['groups'][field][v]['metrics']=V6.metrics(y[mask],pred[mask])
        standardized=model.pipeline.named_steps['scale'].transform(model.pipeline.named_steps['select'].transform(matrix))
        result['selected_feature_shift']={'fraction_abs_source_z_above3':float(np.mean(np.abs(standardized)>3)),
                                          'median_abs_source_z':float(np.median(np.abs(standardized)))}
        results[name]=result
        predictions[name]=[{'participant':person['subject_id'],'site':person['site'],'original_label':person['label'],
            'true_binary':int(target),'predicted_binary':int(label),'predicted_class':'Dementia' if label else 'HC',
            'probabilities':{'HC':float(prob[0]),'Dementia':float(prob[1])}} for person,target,label,prob in zip(people,y,pred,p)]
    summary={'source_participants':88,'target_headers':len(audit),'target_scored':len(people),
             'target_status_counts':dict(Counter(p['status'] for p in audit)),
             'scored_class_counts':dict(Counter(p['label'] for p in people)),'models':results,
             'always_dementia':V6.metrics(y,np.ones(len(y),int)),
             'note':'Primary original_procedure; SVM-only and spectral-SVM are predeclared source-only controls, not selected using target scores. No adaptation to BrainLat.'}
    dump(out/'predictions.json',predictions);dump(out/'summary.json',summary)
    charts(out,summary,predictions,y)
    print(json.dumps(summary,indent=2),flush=True)


def charts(out,summary,predictions,y):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    figures=out/'figures';figures.mkdir(exist_ok=True)
    labels=['Original procedure','SVM-only','Spectral SVM control'];positions=np.arange(3)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for i,k in enumerate(['macro_f1','balanced_accuracy','accuracy']):
        bars=ax.bar(positions+(i-1)*.25,[100*summary['models'][n][k] for n in NAMES],width=.25,label=k.replace('_',' '));ax.bar_label(bars,fmt='%.1f')
    diagnostic=summary.get('quality_flagged',0)
    title=(f'Train on 88; all {len(y)} readable BrainLat recordings ({diagnostic} QC-flagged; diagnostic)'
           if diagnostic else f'Train on 88 original participants; test on {len(y)} source-QC-compatible BrainLat participants')
    ax.set(xticks=positions,xticklabels=labels,ylim=(0,100),ylabel='BrainLat score (%)',title=title);ax.legend()
    fig.savefig(figures/'transfer_comparison.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,name,label in zip(axes,NAMES,labels):
        pred=np.array([r['predicted_binary'] for r in predictions[name]]);cm=confusion_matrix(y,pred,labels=[0,1]);norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        for i,j in __import__('itertools').product([0,1],repeat=2):ax.text(j,i,f'{cm[i,j]}\n{norm[i,j]:.1%}',ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=['HC','Dementia'],yticks=[0,1],yticklabels=['HC','Dementia'],xlabel='Predicted',ylabel='Actual',title=label)
    fig.savefig(figures/'transfer_confusion.png',dpi=175);plt.close(fig)
    lines=['# Original88 -> BrainLat binary transfer test'+(' — all-readable coverage diagnostic' if diagnostic else ' — strict source QC'),'',
           f'Training:88 original AHEPA participants (59dementia,29HC). Target headers:{summary["target_headers"]}; scorable:{summary["target_scored"]}.',
           'No target labels/features used to fit learned transforms, select, train, calibrate or threshold the classifiers. Per-record physical preprocessing uses only that recording. BrainLat has been explored previously, so this is cross-dataset transfer rather than untouched external confirmation.',
           '', '| Frozen source model | Macro F1 | Balanced accuracy | Accuracy | Dementia recall | HC recall |','|---|---:|---:|---:|---:|---:|']
    for n in NAMES:
        m=summary['models'][n];lines.append(f'| {n} | {m["macro_f1"]:.4f} | {m["balanced_accuracy"]:.4f} | {m["accuracy"]:.4f} | {m["dementia_recall"]:.4f} | {m["cn_specificity"]:.4f} |')
    if diagnostic:
        lines+=['',f'**Coverage diagnostic:** {len(y)} readable recordings include {diagnostic} healthy recordings below the old1uV channel-SD floor. Those cases use the pre-existing0.1uV floor; the other76 retain their original strict-QC features and predictions. Models are unchanged. This is not the strict-source-QC primary endpoint. Fourteen header-only recordings remain unclassified.']
    lines+=['','## Interpretation','',
            '- The old75% result described nested cross-validation, not a single saved88-person model. The old36-candidate selection recipe was refit on all88, with choices made solely by three-fold source CV.',
            '- The primary old recipe selected a spectral logistic-regression pipeline. The SVM-only winner and a reference-compatible spectral SVM were frozen as separate source-only controls before target preparation. All results are reported; no target winner is promoted.',
            '- Source feature extraction was replayed against cached old arrays; units and channel ordering are checked. BrainLat uses the fixed pre-existing19-channel spatial mapping and250Hz feature schema.',
            '- Spectral features use19-channel CAR. The old connectivity/power bank retained the original acquisition reference; reproducing that code cannot recreate AHEPA ear references in BrainLat. Interpret the connectivity SVM accordingly.',
            '- Missing signal files and source-policy QC refusals are listed in the primary outputs/target_audit.json. Strict-QC metrics exclude refused cases; the separate all-readable diagnostic includes10 explicitly flagged low-amplitude cases. Missingness may bias the available target cohort.',
            ('- This all-readable diagnostic reports point estimates only. Bootstrap intervals in the strict-QC primary summary apply to its76 participants, not this86-person diagnostic.' if diagnostic else '- Confidence intervals bootstrap target participants within class, conditional on the frozen source model; they do not account for source training uncertainty.'),
            '- Predictions and source-calibrated probabilities are research outputs, not validated clinical probabilities. No test-set adaptation or model selection was performed.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','prepare','evaluate','all'])
    p.add_argument('--out',default=str(HERE/'outputs'));p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    with threadpool_limits(limits=2):
        if args.command in ['freeze','all']:freeze(args.out)
        if args.command in ['prepare','all']:prepare(args)
        if args.command in ['evaluate','all']:evaluate(args.out)


if __name__=='__main__':main()
