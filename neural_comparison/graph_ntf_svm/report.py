from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr,false_discovery_control
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from data import CLASSES,CHANNELS,STATS,write_json


def quick_metrics(y,p):
    cm=confusion_matrix(y,p.argmax(1),labels=[0,1,2]).astype(float)
    recall=np.diag(cm)/np.maximum(cm.sum(1),1)
    f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    return np.array([np.trace(cm)/cm.sum(),f1.mean(),recall.mean()])


def interpret(model,manifest,dataset,out):
    dec=model.features.decomposition
    if dec is None:return
    out.mkdir(exist_ok=True)
    b,f,s=dec.factors_[:3];rank=b.shape[1]
    d=np.load(dataset/'dataset.npz');x,y,sites=d['x'],d['y'],d['sites']
    conn=np.load(dataset/'connectivity.npy') if model.features.config['method']=='coupled' else None
    embeddings=model.features.embedding(x,d['direct'],conn)
    np.savez_compressed(out/'factors.npz',electrodes=b,mel=f,statistics=s,activation=embeddings,
                        connectivity_bands=dec.factors_[3] if len(dec.factors_)==4 else np.empty((0,rank)))
    # Use montage positions for a schematic sensor loading plot; no invented cortex map.
    xy=d['positions'][:,:2];xy/=np.max(np.linalg.norm(xy,axis=1))
    centers=700*(10**(np.linspace(2595*np.log10(1+1/700),2595*np.log10(1+40/700),26)[1:-1]/2595)-1)
    for r in range(rank):
        fig,axes=plt.subplots(1,4,figsize=(16,3.5),constrained_layout=True)
        ax=axes[0];ax.add_patch(plt.Circle((0,0),1.07,fill=False,color='gray',lw=1))
        im=ax.scatter(xy[:,0],xy[:,1],c=b[:,r],s=125,cmap='viridis',vmin=0)
        for k,ch in enumerate(CHANNELS):ax.text(xy[k,0],xy[k,1]+.06,ch,fontsize=7,ha='center')
        ax.set_aspect('equal');ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),title=f'Component {r+1}: scalp loadings');ax.axis('off')
        fig.colorbar(im,ax=ax,shrink=.65)
        axes[1].plot(centers,f[:,r]);axes[1].set(xlabel='Mel center (Hz)',ylabel='Loading',title='Frequency factor')
        axes[2].bar(STATS,s[:,r]);axes[2].set(title='Statistic factor',ylabel='Loading')
        axes[3].boxplot([embeddings[y==k,r] for k in range(3)],tick_labels=CLASSES)
        axes[3].set(title='Full-cohort descriptive activation',ylabel='Non-negative coefficient')
        fig.savefig(out/f'component_{r+1:02d}.png',dpi=160);plt.close(fig)
    if model.svm.kernel=='linear':
        fig,ax=plt.subplots(figsize=(max(7,rank*.5),3.5),constrained_layout=True)
        im=ax.imshow(model.svm.coef_,aspect='auto',cmap='coolwarm')
        ax.set(xticks=np.arange(rank),xticklabels=np.arange(1,rank+1),yticks=np.arange(3),
               yticklabels=['AD vs FTD','AD vs HC','FTD vs HC'],xlabel='Component',title='Pairwise SVM coefficients after scaling (not probability contributions)')
        fig.colorbar(im,ax=ax);fig.savefig(out/'svm_component_coefficients.png',dpi=160);plt.close(fig)
    clinical=[]
    for instrument in ['MMSE','MoCA']:
        for group in ['all','AD','FTD','HC']:
            indices=np.array([i for i,c in enumerate(manifest['clinical']) if c['instrument']==instrument and c['score'] is not None and (group=='all' or y[i]==CLASSES.index(group))])
            if len(indices)<8:continue
            target=np.array([manifest['clinical'][i]['score'] for i in indices])
            if np.std(target)<1e-10:continue
            for r in range(rank):
                if np.std(embeddings[indices,r])<1e-10:continue
                rho,p=spearmanr(embeddings[indices,r],target)
                clinical.append({'instrument':instrument,'group':group,'n':len(indices),'component':r+1,
                                 'rho':float(rho),'p':float(p)})
    if clinical:
        for row,q in zip(clinical,false_discovery_control([r['p'] for r in clinical])):row['q_fdr']=float(q)
    write_json(out/'clinical_correlations.json',{'analysis':'Exploratory full-cohort associations; factors selected using diagnosis. Not independent clinical validation. All-group correlations can reflect diagnostic group separation. MMSE and MoCA are separate instruments.',
               'correlations':clinical,'fdr_family':'All reported components, instruments and diagnostic strata'})
    write_json(out/'optimization.json',{'relative_reconstruction_error':dec.reconstruction_error_,'objective':dec.objective_,
              'graph_roughness':dec.graph_roughness_,'starts':dec.diagnostics_})
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for row in dec.diagnostics_:ax.plot(row['history'],label=f'Start {row["start"]+1}, converged={row["converged"]}')
    ax.set(xlabel='Accepted optimizer iteration',ylabel='Objective',title='G-NTF optimization');ax.legend()
    fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
    # Match only shape factors; labels are not used for matching. Different ranks
    # receive rectangular Hungarian matching and unmatched reference scores=0.
    stability=[]
    method=model.features.config['method']
    for path in sorted(out.parent.parent.glob(f'repeat_*/{method}/model.joblib')):
        other=joblib.load(path).features.decomposition
        similarity=(b.T@other.factors_[0])*(f.T@other.factors_[1])*(s.T@other.factors_[2])
        left,right=linear_sum_assignment(-similarity)
        scores=np.zeros(rank);scores[left]=similarity[left,right]
        stability.append({'fold':str(path.parent.parent.name),'matched_similarities':scores.tolist()})
    write_json(out/'stability.json',{'comparison':'Full-cohort reference versus overlapping outer-training fits; descriptive, not independent replication.',
        'metric':'Product of electrode, frequency and statistic cosine similarities after Hungarian assignment; missing matches=0',
        'folds':stability})
    if stability:
        scores=np.array([r['matched_similarities'] for r in stability])
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        ax.bar(np.arange(rank)+1,scores.mean(0),yerr=scores.std(0));ax.axhline(.85,color='gray',ls='--')
        ax.set(ylim=(0,1.05),xlabel='Component',ylabel='Matched cosine product',title='Component stability (descriptive; threshold 0.85)')
        fig.savefig(out/'stability.png',dpi=160);plt.close(fig)


def generate(out,dataset):
    from run import metrics
    out,dataset=Path(out),Path(dataset)
    records=json.loads((out/'oof_predictions.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((dataset/'manifest.json').read_text())
    d=np.load(dataset/'dataset.npz');y=d['y'];n=len(y)
    plots=out/'figures';plots.mkdir(exist_ok=True)
    summary={}
    for method in protocol['methods']:
        rows=[r for r in records if r['method']==method]
        reps=sorted(set(r['repeat'] for r in rows));predictions=[];measurements=[]
        for rep in reps:
            proba=np.full((n,3),np.nan);coverage=np.zeros(n,int)
            for row in rows:
                if row['repeat']!=rep:continue
                idx=np.array(row['test_indices']);proba[idx]=row['probabilities'];coverage[idx]+=1
                assert not set(row['train_ids'])&set(row['test_ids'])
                for cal in row['calibration_splits']:
                    assert not set(cal['train'])&set(cal['calibration'])
                    assert not (set(cal['train'])|set(cal['calibration']))&set(row['test_ids'])
            if not np.all(coverage==1):raise ValueError('Incomplete or duplicated outer predictions')
            assert np.isfinite(proba).all() and np.allclose(proba.sum(1),1)
            predictions.append(proba);measurements.append(metrics(y,proba))
        p=np.array(predictions)
        rng=np.random.default_rng(400);boot=[]
        for _ in range(500):
            idx=np.concatenate([rng.choice(np.flatnonzero(y==k),np.sum(y==k),replace=True) for k in range(3)])
            boot.append(np.mean([quick_metrics(y[idx],r[idx]) for r in p],axis=0))
        fields=['accuracy','macro_f1','balanced_accuracy','ftd_recall','log_loss','brier_multiclass_sum','macro_auroc']
        summary[method]={'mean':{k:float(np.mean([m[k] for m in measurements])) for k in fields},
                         'repeat_metrics':measurements,'conditional_bootstrap95':dict(zip(['accuracy','macro_f1','balanced_accuracy'],np.quantile(boot,[.025,.975],axis=0).T.tolist())),
                         'by_site':{str(site):metrics(y[d['sites']==site],p.mean(0)[d['sites']==site]) for site in np.unique(d['sites'])},
                         'note':'Site metrics average each participant probability over repeats; overall metrics average per-repeat scores.'}
        cm=sum(np.array(m['confusion_matrix']) for m in measurements)
        fig,ax=plt.subplots(figsize=(5,4.5),constrained_layout=True);im=ax.imshow(cm,cmap='Blues')
        for i,j in itertools_pairs():ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        ax.set(xticks=range(3),xticklabels=CLASSES,yticks=range(3),yticklabels=CLASSES,
               xlabel='Predicted',ylabel='Actual',title=f'{method}: outer predictions, {len(reps)} repeats')
        fig.colorbar(im,ax=ax);fig.savefig(plots/f'{method}_confusion.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,4.5),constrained_layout=True)
        ax.plot([0,1],[0,1],'--',color='gray')
        for k,name in enumerate(CLASSES):
            fraction,mean=calibration_curve(np.tile(y==k,len(reps)),p[:,:,k].ravel(),n_bins=5,strategy='quantile')
            ax.plot(mean,fraction,'o-',label=name)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',ylabel='Observed proportion',title=f'{method}: outer-fold calibration')
        ax.legend();fig.savefig(plots/f'{method}_calibration.png',dpi=160);plt.close(fig)
    write_json(out/'summary.json',summary)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);xx=np.arange(len(summary))
    for i,key in enumerate(['accuracy','macro_f1','balanced_accuracy']):
        values=[100*s['mean'][key] for s in summary.values()]
        bars=ax.bar(xx+(i-1)*.25,values,width=.25,label=key.replace('_',' '));ax.bar_label(bars,fmt='%.1f',fontsize=8)
    ax.set(xticks=xx,xticklabels=list(summary),ylim=(0,100),ylabel='Score (%)',title='Matched nested participant evaluation')
    ax.legend();fig.savefig(plots/'comparison.png',dpi=170);plt.close(fig)
    lines=['# Graph-NTF experiment results','',f'Profile: **{protocol["profile"]}**; outer={protocol["outer"]}; participants={n}.',
           'All reported predictions hold participants out from tensor fitting, tuning and calibration.',
           'Quick runs are bounded development checks; full nested search is a separate command.',
           'Confidence intervals bootstrap participants within class, keeping repeat predictions together. They are conditional on fitted models.',
           '', '| Method | Accuracy | Macro F1 | Balanced accuracy | Log loss |','|---|---:|---:|---:|---:|']
    for method,value in summary.items():
        m=value['mean'];lines.append(f'| {method} | {m["accuracy"]:.3f} | {m["macro_f1"]:.3f} | {m["balanced_accuracy"]:.3f} | {m["log_loss"]:.3f} |')
    nonconverged=sum(not any(s['converged'] for s in r.get('optimization',{}).get('starts',[])) for r in records if 'optimization' in r)
    lines+=['',f'Outer tensor fits without any optimizer-converged initialization: {nonconverged}. Inspect each fold objective history before drawing decomposition conclusions.',
            'The real and shuffled graph arms also tune graph strength: their difference tests complete procedures, not one fixed graph coefficient.',
            'The TT control is rerun on the same positive tensor and current folds; historical TT scores are not mixed into this comparison.',
            'The source window preparation did not explicitly exclude EEGLAB discontinuity annotations. These results inherit that limitation.',
            'Final-model factors and cognition correlations are descriptive full-cohort analyses, not independent clinical validation.',
            'Graph regularization encourages smooth sensor maps; it does not prove anatomical causality or eliminate CP permutation ambiguity.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for method in ['gntf','coupled']:
        path=out/f'final_{method}'/'model.joblib'
        if path.exists():interpret(joblib.load(path),manifest,dataset,path.parent/'interpretation')
    print(json.dumps({m:v['mean'] for m,v in summary.items()},indent=2),flush=True)


def itertools_pairs():
    return ((i,j) for i in range(3) for j in range(3))
