"""Paired montage comparisons and Matplotlib figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from brainlat_data import ARMS,dump

NAMES={'native128':'128 native','native19':'19 native','legacy19':'19 legacy interpolation','aligned19':'19 aligned interpolation'}


def generate(out,data):
    from run import metrics,scores
    out,data=Path(out),Path(data)
    plan=json.loads((data/'plan.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    records=json.loads((out/'oof_predictions.json').read_text());y=np.array(plan['binary_labels']);n=len(y)
    ids={p['subject_id'] for p in plan['participants']};repeats=protocol['repeats']
    arrays={};summary={};figures=out/'figures';figures.mkdir(exist_ok=True)
    for arm in ARMS:
        predictions=[];probabilities=[];repeat_metrics=[]
        for repeat in range(repeats):
            pred=np.full(n,-1,int);p=np.full((n,2),np.nan);coverage=np.zeros(n,int)
            for row in records:
                if row['arm']!=arm or row['repeat']!=repeat:continue
                fit,test=set(row['train_ids']),set(row['test_ids'])
                if fit&test or fit|test!=ids:raise ValueError('Train/test membership violation')
                idx=np.array(row['test_indices']);coverage[idx]+=1;pred[idx]=row['predictions'];p[idx]=row['probabilities']
                for split in row['calibration_splits']:
                    a,b=set(split['train_ids']),set(split['calibration_ids'])
                    if a&b or a|b!=fit or (a|b)&test:raise ValueError('Calibration leakage')
                search=json.loads((out/f'repeat_{repeat:02d}_fold_{row["fold"]:02d}'/arm/'search.json').read_text())
                for split in search['inner_splits']:
                    a,b=set(split['train_ids']),set(split['validation_ids'])
                    if a&b or a|b!=fit or (a|b)&test:raise ValueError('Inner tuning leakage')
            if not np.all(coverage==1) or np.any(p<0) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1):raise ValueError('Invalid outer coverage/probabilities')
            predictions.append(pred);probabilities.append(p);repeat_metrics.append(metrics(y,pred,p))
        # Identical test IDs must be used in the corresponding folds of all arms.
        for row in [r for r in records if r['arm']==arm]:
            match=[r for r in records if r['repeat']==row['repeat'] and r['fold']==row['fold']]
            if any(r['test_ids']!=row['test_ids'] for r in match) or len(match)!=len(ARMS):raise ValueError('Unmatched arms')
        predictions=np.array(predictions);arrays[arm]=predictions
        keys=[k for k in repeat_metrics[0] if k!='confusion_matrix']
        mean={k:float(np.mean([v[k] for v in repeat_metrics])) for k in keys}
        grouped={}
        for field in ['site','label']:
            grouped[field]={}
            for name in sorted({p[field] for p in plan['participants']}):
                mask=np.array([p[field]==name for p in plan['participants']])
                if field=='label':grouped[field][name]={'n':int(mask.sum()),'mean_correct_binary_classification':float(np.mean(predictions[:,mask]==y[mask]))}
                else:
                    rows=[metrics(y[mask],p[mask]) for p in predictions]
                    grouped[field][name]={'n':int(mask.sum()),'mean':{k:float(np.mean([v[k] for v in rows])) for k in rows[0] if k!='confusion_matrix'}}
        summary[arm]={'mean':mean,'repeat_metrics':repeat_metrics,'groups':grouped,
                      'uninformative_calibration_folds':sum(r.get('calibration_diagnostics',{}).get('uninformative',False) for r in records if r['arm']==arm),
                      'repeat_sd':{k:float(np.std([v[k] for v in repeat_metrics],ddof=1)) if repeats>1 else None for k in keys}}
    rng=np.random.default_rng(771);intervals={};comparisons={}
    boot={a:[] for a in ARMS}
    for _ in range(1500):
        idx=np.concatenate([rng.choice(np.flatnonzero(y==k),np.sum(y==k),replace=True) for k in [0,1]])
        for arm in ARMS:boot[arm].append(np.mean([scores(y[idx],p[idx])[:2] for p in arrays[arm]],axis=0))
    for arm in ARMS:
        intervals[arm]=dict(zip(['macro_f1','balanced_accuracy'],np.quantile(boot[arm],[.025,.975],axis=0).T.tolist()))
        if arm=='legacy19':continue
        difference=np.array(boot[arm])-np.array(boot['legacy19'])
        comparisons[arm+'_minus_legacy19']={'mean_difference':{k:summary[arm]['mean'][k]-summary['legacy19']['mean'][k] for k in ['macro_f1','balanced_accuracy']},
               'conditional_paired_bootstrap95':dict(zip(['macro_f1','balanced_accuracy'],np.quantile(difference,[.025,.975],axis=0).T.tolist()))}
    dump(out/'summary.json',{'participants':n,'class_counts':{'HC':int(np.sum(y==0)),'Dementia':int(np.sum(y==1))},
         'arms':summary,'conditional_participant_bootstrap95':intervals,'paired_comparisons':comparisons,
         'note':'Intervals condition on fitted models and preserve repeated observations by person. Multiple descriptive comparisons, not confirmatory significance tests.'})
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True);x=np.arange(4)
    for i,k in enumerate(['macro_f1','balanced_accuracy','accuracy']):
        bars=ax.bar(x+(i-1)*.25,[100*summary[a]['mean'][k] for a in ARMS],width=.25,label=k.replace('_',' '));ax.bar_label(bars,fmt='%.1f',fontsize=9)
    ax.set(xticks=x,xticklabels=[NAMES[a] for a in ARMS],ylim=(0,100),ylabel='Score (%)',title=f'BrainLat only: same{n} people, same held-out folds');ax.legend()
    fig.savefig(figures/'montage_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,arm in zip(axes.ravel(),ARMS):
        cm=sum(np.array(v['confusion_matrix']) for v in summary[arm]['repeat_metrics'])
        normalized=cm/np.maximum(cm.sum(1,keepdims=True),1)
        ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
        for i,j in itertools.product([0,1],repeat=2):ax.text(j,i,f'{cm[i,j]}\n{normalized[i,j]:.1%}',ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=['HC','Dementia'],yticks=[0,1],yticklabels=['HC','Dementia'],xlabel='Predicted',ylabel='Actual',title=NAMES[arm])
    fig.suptitle(f'{n} participants x{repeats} repeats; counts are repeated held-out predictions')
    fig.savefig(figures/'confusion_matrices.png',dpi=170);plt.close(fig)
    geometry=plan['geometry'];src=np.array(geometry['source_head_positions_m']);dst=np.array(geometry['target_head_positions_m'])
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    ax.scatter(src[:,0],src[:,1],s=12,c='lightgray',label='128 native positions')
    names=geometry['source_channels']
    for row in geometry['native19_assignment']:
        i=names.index(row['native']);j=geometry['target_channels'].index(row['target'])
        ax.plot([src[i,0],dst[j,0]],[src[i,1],dst[j,1]],color='gray',lw=.7)
        ax.scatter(src[i,0],src[i,1],s=45,color='tab:blue');ax.scatter(dst[j,0],dst[j,1],marker='x',s=45,color='tab:red')
        ax.text(dst[j,0],dst[j,1]+.003,f'{row["target"]}/{row["native"]}',ha='center',fontsize=7)
    ax.set_aspect('equal');ax.set(xlabel='Head x (m)',ylabel='Head y (m), anterior upwards',title='19 native selections (blue) and aligned target positions (red)')
    fig.savefig(figures/'electrode_geometry.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    labels=['AR','CL'];x=np.arange(2)
    for i,arm in enumerate(ARMS):
        for ax,k in zip(axes,['macro_f1','balanced_accuracy']):
            ax.bar(x+(i-1.5)*.2,[100*summary[arm]['groups']['site'][s]['mean'][k] for s in labels],width=.2,label=NAMES[arm])
            ax.set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Score (%)',title=k.replace('_',' '))
    axes[0].legend(fontsize=7);fig.savefig(figures/'site_comparison.png',dpi=170);plt.close(fig)
    lines=['# BrainLat-only native-electrode ablation','',
           f'{n} fixed participants:35AD,19FTD,32HC. Binary HC versus AD+FTD. {repeats} repeats x{protocol["outer_folds"]} outer folds; {protocol["inner_folds"]} inner folds.',
           'All arms use identical participants, epochs, folds, native512Hz,1–40Hz filtering, common128-channel average reference and common native-window gain normalization.',
           'Fixed spectral settings:8-second segment,1-second STFT,1-second hop,32mel bands; mean/SD/median/IQR. Same256-candidate SVM/preprocessing grid in every arm.','',
           '| Arm | Macro F1 | Balanced accuracy | Accuracy | HC recall | Dementia recall |','|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        v=summary[a]['mean'];lines.append(f'| {a} | {v["macro_f1"]:.4f} | {v["balanced_accuracy"]:.4f} | {v["accuracy"]:.4f} | {v["healthy_recall"]:.4f} | {v["dementia_recall"]:.4f} |')
    lines+=['','## Interpretation limits','',
            '- Legacy19 reproduces the old spatial operator, but shares the native128 reference and normalization in this experiment. It is not an exact rerun of the earlier mixed-cohort production pipeline.',
            '- Aligned19 uses both layouts in head coordinates with the same source-sphere origin subtracted before the spline calculation. It retains the same spline regularization0.',
            '- The installed MNE legacy helper used the target montage coordinates directly, while source coordinates were head-frame and source-origin centered. Source inspection motivated the aligned diagnostic arm before seeing scores.',
            '- Native19 sensors approximate target positions; different physical locations mean native19-versus-interpolated19 is not a perfectly isolated interpolation-only intervention.',
            '- Native128 changes spatial coverage and feature dimensionality together. Every arm has the same nested tuning budget; differences compare complete selected procedures.',
            '- Native reference and per-window log-mel gain use all128 measured channels in every arm. Native19 means19 analysis channels after that common preprocessing, not a standalone19-electrode acquisition.',
            '- Common epochs pass quality checks on all128 native channels. No arm-specific epoch rejection or subject dropping is permitted.',
            '- The86-person cohort inherits previous inclusion decisions. This is not all originally available BrainLat recordings, not an untouched external test, and not clinical validation.',
            '- Site-specific results describe participants held out within AR/CL, not generalization to an unseen site.',
            '- Paired bootstrap intervals retain each person\'s repeat predictions, conditional on fitted models; they do not include all training uncertainty or adjust for multiple comparisons.',
            '- Final full-cohort probability models are inference artifacts and do not supply test-performance evidence.']
    lines+=['- Monotone sigmoid calibration is fitted inside training only. When its slope reaches zero, probabilities revert to training prevalence and no probability-space operating threshold is defined. Main classification metrics still use the selected SVM score threshold.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({a:v['mean'] for a,v in summary.items()},indent=2),flush=True)


import itertools
