"""Matplotlib figures and participant-level metrics for the binary experiment."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix,roc_curve

from features import dump


def generate(out,data):
    from run import metrics,selection_score
    out,data=Path(out),Path(data)
    records=json.loads((out/'oof_predictions.json').read_text())
    plan=json.loads((data/'plan.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    people=plan['participants'];y=np.array(plan['binary_labels']);n=len(y)
    reps=sorted({r['repeat'] for r in records});preds=[];probabilities=[];fixed={128:[],256:[]}
    for rep in reps:
        pred=np.full(n,-1,int);p=np.full((n,2),np.nan);coverage=np.zeros(n,int)
        controls={fs:np.full(n,-1,int) for fs in fixed}
        for row in records:
            if row['repeat']!=rep:continue
            idx=np.array(row['test_indices']);coverage[idx]+=1;pred[idx]=row['predictions'];p[idx]=row['probabilities']
            fit,test=set(row['train_ids']),set(row['test_ids'])
            if fit&test or fit|test!={p['subject_id'] for p in people}:raise ValueError('Invalid participant split')
            for split in row['calibration_splits']:
                a,b=set(split['train_ids']),set(split['calibration_ids'])
                if a&b or (a|b)!=fit or (a|b)&test:raise ValueError('Invalid calibration membership')
            search=json.loads((out/f'repeat_{rep:02d}_fold_{row["fold"]:02d}'/'search.json').read_text())
            for split in search['inner_splits']:
                a,b=set(split['train_ids']),set(split['validation_ids'])
                if a&b or (a|b)!=fit or (a|b)&test:raise ValueError('Invalid inner membership')
            for fs in controls:controls[fs][idx]=row['fixed_sampling_controls'][str(fs)]
        if not np.all(coverage==1):raise ValueError('Incomplete or duplicated outer predictions')
        if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1):raise ValueError('Invalid probabilities')
        preds.append(pred);probabilities.append(p)
        for fs in fixed:fixed[fs].append(controls[fs])
    preds=np.array(preds);probabilities=np.array(probabilities)
    repeat_metrics=[metrics(y,z,p) for z,p in zip(preds,probabilities)]
    numeric=[k for k in repeat_metrics[0] if k!='confusion_matrix']
    means={k:float(np.mean([r[k] for r in repeat_metrics])) for k in numeric}
    sd={k:float(np.std([r[k] for r in repeat_metrics],ddof=1)) if len(reps)>1 else None for k in numeric}
    rng=np.random.default_rng(450);boot=[]
    for _ in range(1000):
        indices=np.concatenate([rng.choice(np.flatnonzero(y==k),np.sum(y==k),replace=True) for k in [0,1]])
        boot.append(np.mean([selection_score(y[indices],z[indices]) for z in preds],axis=0))
    ci=dict(zip(['macro_f1','balanced_accuracy'],np.quantile(boot,[.025,.975],axis=0).T.tolist()))
    grouped={}
    for field in ['site','label']:
        grouped[field]={}
        for name in sorted({p[field] for p in people}):
            use=np.array([p[field]==name for p in people])
            if field=='label':
                # Within a single original diagnosis report detection rate, not binary BA.
                correct=np.mean(preds[:,use]==y[use],axis=1)
                grouped[field][name]={'participants':int(use.sum()),'mean_correct_classification_rate':float(correct.mean()),
                                      'repeat_rates':correct.tolist()}
            else:
                rows=[metrics(y[use],z[use]) for z in preds]
                grouped[field][name]={'participants':int(use.sum()),'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='confusion_matrix'}}
    fixed_results={str(fs):{'repeat_metrics':[metrics(y,z) for z in values]} for fs,values in fixed.items()}
    for value in fixed_results.values():
        value['mean']={k:float(np.mean([r[k] for r in value['repeat_metrics']])) for k in ['macro_f1','balanced_accuracy','accuracy','healthy_recall','dementia_recall']}
    summary={'participants':n,'class_counts':{'HC':int(np.sum(y==0)),'Dementia':int(np.sum(y==1))},
             'mean':means,'repeat_sd':sd,'repeat_metrics':repeat_metrics,'conditional_participant_bootstrap95':ci,
             'groups':grouped,'fixed_sampling_controls':fixed_results,
             'always_dementia':metrics(y,np.ones(n,int)),
             'probability_argmax_secondary':{'repeat_metrics':[metrics(y,p.argmax(1),p) for p in probabilities]},
             'selected_sampling_hz_counts':dict(Counter(str(r['selected']['config']['sampling_hz']) for r in records)),
             'note':'Main labels use inner-selected decision-score thresholds. Probability argmax is a distinct secondary endpoint. Intervals condition on fitted models; repeated predictions do not add participants.'}
    dump(out/'summary.json',summary)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    cm=sum(confusion_matrix(y,z,labels=[0,1]) for z in preds)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for ax,matrix,title,percent in [(axes[0],cm,'Pooled outer prediction counts',False),
                                    (axes[1],cm/np.maximum(cm.sum(1,keepdims=True),1),'Recall by actual class',True)]:
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1 if percent else None)
        for i,j in itertools_pairs():ax.text(j,i,f'{matrix[i,j]:.1%}' if percent else str(matrix[i,j]),ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=['HC','Dementia'],yticks=[0,1],yticklabels=['HC','Dementia'],xlabel='Predicted',ylabel='Actual',title=title)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle(f'{n} participants, {len(reps)} repeats; pooled counts are repeated predictions')
    fig.savefig(figures/'confusion_matrix.png',dpi=170);plt.close(fig)
    names=['Always dementia','Fixed 128 Hz','Fixed 256 Hz','Nested search']
    values=[summary['always_dementia'],fixed_results['128']['mean'],fixed_results['256']['mean'],means]
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);x=np.arange(4)
    for i,k in enumerate(['macro_f1','balanced_accuracy','accuracy']):
        bars=ax.bar(x+(i-1)*.25,[100*v[k] for v in values],width=.25,label=k.replace('_',' '));ax.bar_label(bars,fmt='%.1f',fontsize=9)
    ax.set(xticks=x,xticklabels=names,ylim=(0,100),ylabel='Score (%)',title='Same participant folds; fixed controls differ only in sampling rate')
    ax.legend();fig.savefig(figures/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for i,(pred,p) in enumerate(zip(preds,probabilities)):
        fpr,tpr,_=roc_curve(y,p[:,1]);axes[0].plot(fpr,tpr,label=f'Repeat {i+1}')
        observed,estimated=calibration_curve(y,p[:,1],n_bins=6,strategy='quantile');axes[1].plot(estimated,observed,'o-',label=f'Repeat {i+1}')
    for ax in axes:ax.plot([0,1],[0,1],'--',color='gray');ax.legend()
    axes[0].set(xlabel='False-positive rate',ylabel='Dementia recall',title='Outer-fold probability ROC')
    axes[1].set(xlabel='Predicted dementia probability',ylabel='Observed dementia fraction',title='Calibration (small-sample descriptive)')
    fig.savefig(figures/'roc_calibration.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    fields=['healthy_recall','dementia_recall'];positions=np.arange(len(repeat_metrics))
    for i,k in enumerate(fields):
        bars=ax.bar(positions+(i-.5)*.35,[100*r[k] for r in repeat_metrics],width=.35,label=k.replace('_',' '));ax.bar_label(bars,fmt='%.1f')
    ax.set(xticks=positions,xticklabels=[f'Repeat {i+1}' for i in positions],ylim=(0,100),ylabel='Recall (%)',title='Both classes must be detected');ax.legend()
    fig.savefig(figures/'class_recall.png',dpi=170);plt.close(fig)
    dump(out/'selected_parameters.json',[{'repeat':r['repeat'],'fold':r['fold'],**r['selected']} for r in records])
    lines=['# Binary log-mel SVM: 174-participant nested experiment','',
           f'{len(reps)} repeats x {protocol["outer_folds"]} outer folds; {protocol["inner_folds"]} inner folds. HC=0; AD+FTD=1.',
           'All preprocessing choices, classifier settings, weights and thresholds selected without outer-test labels.',
           'Primary: macro F1. Secondary selection: balanced accuracy. Repeated folds do not create additional participants.','',
           '| Metric | Mean | Repeat SD |','|---|---:|---:|']
    for k in ['macro_f1','balanced_accuracy','accuracy','healthy_recall','dementia_recall','roc_auc','log_loss']:
        lines.append(f'| {k} | {means[k]:.4f} | {sd[k]:.4f} |' if sd[k] is not None else f'| {k} | {means[k]:.4f} | — |')
    lines+=['','## Sampling-rate controls','',
            'Controls use 8-second segments, 2-second STFT, 0.5-second hop, 24 mel bands, mean/SD, site z-scoring, standard scaling, PCA40, balanced RBF SVM C1/gamma scale, threshold zero.',
            '256 Hz is obtained directly from native source EEG; the 128-Hz control is anti-aliased downsampling of that signal. With 1–40 Hz filtering, higher rate does not introduce additional biological bandwidth.',
            '', '| Sampling | Macro F1 | Balanced accuracy |','|---|---:|---:|']
    for fs,v in fixed_results.items():lines.append(f'| {fs} Hz | {v["mean"]["macro_f1"]:.4f} | {v["mean"]["balanced_accuracy"]:.4f} |')
    lines+=['','## Scope and limitations','',
            '- Frozen random search with representation coverage, not an exhaustive Cartesian grid. Candidate settings are saved in prepared/plan.json.',
            '- Per-person feature extraction is deterministic and label-independent. Learned site adjustment, scaling, PCA, SVM and calibration are fitted on training participants only.',
            '- Participant summaries pool frames across accepted windows. Temporal ordering is not an SVM input.',
            '- Native source derivatives have different upstream cleaning. BrainLat interpolation is geometry-based harmonization, not additional measured electrodes.',
            '- Windows touching marked boundary/BAD/EDGE intervals (2-second guard) are rejected. Source filtering still reuses the existing full-record harmonizer.',
            '- At most 320 accepted seconds per person at every segment length; participants with less usable EEG contribute less. Missing annotation or unmarked artifact can remain.',
            '- Main predictions use inner-selected score thresholds. Calibrated probabilities may have an operating threshold different from 0.5; probability argmax metrics are saved separately.',
            '- Clinical AD/FTD labels are collapsed to dementia. This experiment does not distinguish the two conditions.',
            '- Same cohort has been explored previously. Results are internal validation, not an untouched external benchmark.',
            '- Bootstrap intervals resample participants within class, keeping repeats together, conditional on already fitted models. They do not capture all model-selection uncertainty.',
            '- A final full-cohort model is for inference and does not provide test-performance evidence.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'mean':means,'conditional95':ci,'fixed_sampling_controls':{k:v['mean'] for k,v in fixed_results.items()}},indent=2),flush=True)


def itertools_pairs():return ((i,j) for i in range(2) for j in range(2))
