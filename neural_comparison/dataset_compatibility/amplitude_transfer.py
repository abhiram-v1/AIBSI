"""Train the 97%-F1 amplitude pipeline on BrainLat; test all88 OpenNeuro people."""
import argparse
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from threadpoolctl import threadpool_limits
from audit_data import HERE,dump,sha
from compat_model import GRID,THRESHOLDS,metrics,select,train_predict
from amplitude_transfer_model import AmplitudeModel

DEST=HERE/'outputs/amplitude_transfer_brainlat_to_openneuro'


def read_amplitudes(prep,indices):
    return np.array([json.loads((prep/'parts'/f'{i:03d}.json').read_text())['common_metrics']['median_channel_rms_uv'] for i in indices])


def run():
    DEST.mkdir(parents=True,exist_ok=True);prep=HERE/'outputs/prepared'
    ps=json.loads((prep/'plan.json').read_text())['participants']
    tr=np.array([i for i,p in enumerate(ps) if p['source']=='brainlat'])
    te=np.array([i for i,p in enumerate(ps) if p['source']=='ds004504'])
    assert len(tr)==86 and len(te)==88 and not set(tr)&set(te)
    rms=read_amplitudes(prep,tr);x=np.log10(rms)[:,None]
    y=np.array([ps[i]['label']!='HC' for i in tr],int);sites=np.array([ps[i]['site'] for i in tr])
    strata=np.array([f'{ps[i]["site"]}_{ps[i]["label"]}' for i in tr])
    policy={'task':'HC0 versus AD+FTD1; one participant per sample',
            'source':'86 BrainLat;35AD19FTD32HC','target':'all88 OpenNeuro;36AD23FTD29HC',
            'feature':'log10 median19-channel RMS uV, same matched CAR/filter/windows as the earlier97.1% BrainLat CV experiment',
            'selection':'same8 linear/RBF SVM configurations and3 thresholds;3-fold source-only selection, seed2026110591',
            'grid':GRID,'thresholds':THRESHOLDS,'source_selection_seed':2026110591,
            'probabilities':'monotone sigmoid fit on source-only nested5fold OOF scores, first repeat of the previously saved BrainLat CV',
            'classification':'source-selected SVM score threshold; calibration does not change hard predictions',
            'target_adaptation':'none; no target cohort statistics or target labels used for fitting or selection',
            'caveat':'Both datasets were explored earlier; held outside this fit, not an untouched prospective test',
            'source_indices':tr.tolist(),'test_indices':te.tolist(),'source_metadata_sha256':{str(i):sha(prep/'parts'/f'{i:03d}.json') for i in tr},
            'code_sha256':{f:sha(HERE/f) for f in ('amplitude_transfer.py','amplitude_transfer_model.py','compat_model.py')}}
    if (DEST/'protocol.json').exists():
        if json.loads((DEST/'protocol.json').read_text())!=policy:raise RuntimeError('Existing frozen experiment differs; keep its outputs intact')
        print('Loading the previously frozen source model.',flush=True)
        frozen=json.loads((DEST/'frozen_source.json').read_text());assert sha(DEST/'model.joblib')==frozen['model_sha256']
        model=joblib.load(DEST/'model.joblib')
    else:
        dump(DEST/'protocol.json',policy)
        snapshot=DEST/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for f in policy['code_sha256']:shutil.copy2(HERE/f,snapshot/f)
        best,trials=select(x,y,sites,strata,'raw',policy['source_selection_seed'])
        print('Selected using BrainLat only:',best,flush=True)
        previous=json.loads((HERE/'outputs/followup/amplitude_probe.json').read_text())['BrainLat']
        source_oof=np.full(len(tr),np.nan);index_lookup={v:i for i,v in enumerate(tr)};cal_records=[]
        # Refit only the source training folds from the first previously completed nested CV repeat.
        for record in previous['records']:
            if record['repeat']!=0:continue
            a=np.array([index_lookup[i] for i in record['training']]);b=np.array([index_lookup[i] for i in record['test']])
            assert not set(a)&set(b)
            prediction,score=train_predict(x[a],y[a],sites[a],x[b],sites[b],'raw',record['best']['config'])
            np.testing.assert_array_equal(prediction,record['prediction']);source_oof[b]=score
            cal_records.append(record)
        assert np.isfinite(source_oof).all()
        model=AmplitudeModel(best['config']).fit(x,y,sites,source_oof)
        joblib.dump(model,DEST/'model.joblib')
        frozen={'best':best,'trials':trials,'source_oof_score':source_oof.tolist(),'calibration_folds':cal_records,
                'source_internal_nested_cv_mean':previous['mean'],'model_sha256':sha(DEST/'model.joblib'),
                'calibrator_slope':float(model.calibrator_.slope_),'calibrator_intercept':float(model.calibrator_.intercept_),
                'probability_threshold':model.probability_threshold,'calibration_caveat':'Source calibration may not remain valid in another cohort'}
        dump(DEST/'frozen_source.json',frozen)
        print('BrainLat model frozen and hashed BEFORE loading OpenNeuro amplitude features.',flush=True)

    # OpenNeuro enters only after the model, hyperparameters and calibrator have been frozen.
    target_rms=read_amplitudes(prep,te);xt=np.log10(target_rms)[:,None]
    before=sha(DEST/'model.joblib');prediction=model.predict(xt);score=model.decision_function(xt);prob=model.predict_proba(xt)
    yt=np.array([ps[i]['label']!='HC' for i in te],int);m=metrics(yt,prediction)
    assert before==sha(DEST/'model.joblib')==frozen['model_sha256']
    assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1));np.testing.assert_allclose(prob.sum(1),1)
    # Batch composition and site identities must not change predictions.
    for i in range(len(xt)):
        np.testing.assert_allclose(model.predict_proba(xt[i:i+1])[0],prob[i],rtol=1e-12,atol=1e-12)
    reloaded=joblib.load(DEST/'model.joblib');np.testing.assert_array_equal(reloaded.predict(xt),prediction)
    summary={'training_participants':len(tr),'test_participants':len(te),'source_internal_nested_cv':frozen['source_internal_nested_cv_mean'],
             'test_metrics':m,'test_probability_metrics':{'roc_auc':float(roc_auc_score(yt,prob[:,1])),
                'log_loss':float(log_loss(yt,prob,labels=[0,1])),'brier_score':float(brier_score_loss(yt,prob[:,1]))},
             'predicted_counts':{'HC':int(np.sum(prediction==0)),'dementia':int(np.sum(prediction==1))},
             'source_config':model.config,'model_sha256':before,
             'target_metadata_sha256':{str(i):sha(prep/'parts'/f'{i:03d}.json') for i in te},
             'heldout_amplitude_ranges_uv':{d:{'minimum':float(target_rms[np.array([ps[i]['label']==d for i in te])].min()),
                'maximum':float(target_rms[np.array([ps[i]['label']==d for i in te])].max())} for d in ('HC','AD','FTD')},
             'probability_caveat':'Calibrated on BrainLat only; probabilities are research outputs, not verified clinical confidence on OpenNeuro'}
    dump(DEST/'summary.json',summary)
    dump(DEST/'predictions.json',[{'subject_id':ps[i]['subject_id'],'original_diagnosis':ps[i]['label'],
        'actual_binary':'HC' if yt[j]==0 else 'dementia','rms_uv':float(target_rms[j]),'svm_score':float(score[j]),
        'predicted_binary':'HC' if prediction[j]==0 else 'dementia','probabilities':{'HC':float(prob[j,0]),'AD_or_FTD':float(prob[j,1])}} for j,i in enumerate(te)])
    np.savez_compressed(DEST/'predictions.npz',test_indices=te,y=yt,prediction=prediction,score=score,probabilities=prob,rms_uv=target_rms)
    dump(DEST/'verification.json',{'passed':True,'disjoint_source_test':True,'frozen_model_hash_unchanged':True,
         'probabilities_valid':True,'inference_independent_of_target_batch_statistics':True,'saved_model_reloads_identically':True,
         'source_calibration_oof_predictions_reproduced':True})
    figures(model,ps,tr,te,rms,target_rms,y,yt,prediction)
    cm=np.array(m['confusion_matrix'])
    lines=['# BrainLat amplitude-only SVM → all OpenNeuro participants','',
        'This is the exact one-feature pipeline that achieved97.1% mean macro F1 in internal BrainLat nested CV, now refitted on all86 BrainLat participants and tested on all88 OpenNeuro participants. The97.1% describes the source validation procedure, not a single previously saved fitted model.',
        '', 'One feature: log10 median19-channel RMS after the same fixed256Hz/CAR/1–40Hz/8s-window preparation. Class labels: HC versus AD+FTD. No class-specific scaling, site correction, target statistics, target tuning or target calibration.',
        '', f'Selected entirely within BrainLat: `{model.config}`. Source-only3-fold parameter/threshold selection; final scaler,1D PCA and SVM fitted on86 source participants. Probability calibration uses source-only nested5-fold OOF scores. The complete source model was saved and hashed before target features entered this script.',
        '', '| Metric | BrainLat internal nested CV mean | OpenNeuro full test set |','|---|---:|---:|']
    for key in ('macro_f1','balanced_accuracy','accuracy','healthy_recall','dementia_recall'):
        lines.append(f'|{key}|{100*frozen["source_internal_nested_cv_mean"][key]:.1f}%|{100*m[key]:.1f}%|')
    lines+=['','| Actual / predicted | HC | Dementia |','|---|---:|---:|',f'|HC|{cm[0,0]}|{cm[0,1]}|',f'|Dementia|{cm[1,0]}|{cm[1,1]}|',
        '', f'Target predictions: {summary["predicted_counts"]}. The test contains29 HC,36 AD and23 FTD; original AD/FTD diagnoses are preserved in predictions.json, but this model outputs only HC and combined AD/FTD probabilities.',
        '', 'A drop in transfer performance means the source amplitude rule did not generalize under this matched processing policy. It cannot identify whether biology, recruitment, acquisition, export/cleaning, interpolation or implementation differences caused the distribution shift, and provides no evidence of intentional author manipulation.',
        '', 'Both cohorts were explored before this experiment. OpenNeuro is outside all fitting and parameter selection in this experiment; it is not an untouched prospective cohort. Source-calibrated probabilities may be poorly calibrated on the target. Hard classifications use the source-selected SVM threshold, which need not correspond to probability0.5.',
        '', 'Each test participant is evaluated once. This test has88 distinct people, not repeated windows as independent test cases. A 50% balanced-accuracy score corresponds to chance; macro F1 can differ from50% under class imbalance.',
        '', 'Verification passed: disjoint cohorts, source OOF predictions reproduced, unchanged frozen model hash after target evaluation, valid probabilities, batch-independent inference, identical predictions after reload.',
        '', f'![Transfer results]({(DEST/"transfer_results.png").as_posix()})','']
    (DEST/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('OpenNeuro held-out result:',json.dumps(m),flush=True)
    print('Predicted class counts:',summary['predicted_counts'],flush=True)
    print('Report:',DEST/'REPORT.md',flush=True)


def figures(model,ps,tr,te,rms,target_rms,y,yt,prediction):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(17,5.4));rng=np.random.default_rng(33)
    colors=['#207766','#8c3b49'];ax=axes[0]
    for cohort,a,labels,offset in [('BrainLat',rms,y,0),('OpenNeuro',target_rms,yt,3)]:
        for c in (0,1):
            vals=a[labels==c];pos=offset+c
            ax.scatter(pos+rng.uniform(-.17,.17,len(vals)),vals,color=colors[c],s=18,alpha=.7)
            ax.plot([pos-.24,pos+.24],[np.median(vals)]*2,color='black',linewidth=2)
    ax.set_xticks([0,1,3,4],['BrainLat\nHC','BrainLat\nAD+FTD','OpenNeuro\nHC','OpenNeuro\nAD+FTD'])
    ax.set_yscale('log');ax.set_ylabel('Median-channel RMS (µV)');ax.set_title('Same feature policy; dots are people')
    ax=axes[1];grid=np.geomspace(min(rms.min(),target_rms.min())*.7,max(rms.max(),target_rms.max())*1.3,800)
    p=model.predict_proba(np.log10(grid)[:,None])[:,1];ax.plot(grid,p,color='#3166a0',linewidth=2)
    if model.probability_threshold is not None:ax.axhline(model.probability_threshold,color='gray',linestyle='--',label='Source-selected operating point')
    for c in (0,1):ax.scatter(target_rms[yt==c],np.full(np.sum(yt==c),.03+.94*c),marker='|',s=100,color=colors[c],alpha=.65,label=f'Test {"HC" if c==0 else "dementia"}')
    ax.set_xscale('log');ax.set_ylim(-.04,1.04);ax.set_xlabel('Median-channel RMS (µV)');ax.set_ylabel('Source-calibrated P(AD or FTD)')
    ax.set_title('Frozen BrainLat amplitude rule');ax.legend(fontsize=8)
    ax=axes[2];cm=np.array(metrics(yt,prediction)['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=59)
    for i in (0,1):
        for j in (0,1):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=21,color='white' if cm[i,j]>29 else 'black')
    ax.set_xticks([0,1],['HC','AD+FTD']);ax.set_yticks([0,1],['HC','AD+FTD']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title('All88 OpenNeuro participants')
    fig.suptitle('Train BrainLat86 → test OpenNeuro88: single-amplitude SVM',fontsize=16)
    fig.text(.5,.015,'Selection, scaling and probability calibration use BrainLat only. OpenNeuro is evaluated once; neither its labels nor cohort statistics enter training.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(DEST/'transfer_results.png',dpi=170,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',nargs='?',choices=['run','predict'],default='run')
    p.add_argument('--rms-uv',type=float,nargs='+');args=p.parse_args()
    if args.action=='run':run()
    else:
        if not args.rms_uv or not np.isfinite(args.rms_uv).all() or min(args.rms_uv)<=0:p.error('Supply positive finite --rms-uv values measured using the matched preparation policy')
        model=joblib.load(DEST/'model.joblib');prob=model.predict_proba(np.log10(args.rms_uv)[:,None])
        for value,row in zip(args.rms_uv,prob):print(json.dumps({'rms_uv':value,'P_HC':float(row[0]),'P_AD_or_FTD':float(row[1])}))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
