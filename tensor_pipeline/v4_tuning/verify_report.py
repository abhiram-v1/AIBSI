"""Replay fitted tuning outputs, export XGBoost boosters, and report all scores."""
from pathlib import Path
from collections import Counter
import json
import platform
import importlib.metadata
import joblib
import numpy as np
import xgboost
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits
from tune_models import weights,DelayProjection,scores,group_rep
from benchmark_tuning import OUT,DATA,ROOT,signature,select,load_inputs,metric
from prepare import sha


def main():
    x,controls,subjects,y=load_inputs()
    report=json.loads((OUT/'nested_report.json').read_text())
    assert report['completed_folds']==15 and report['signature']==signature()
    old=json.loads((DATA/'nested_report.json').read_text())
    input_report=json.loads((DATA/'preparation_report.json').read_text())
    for path,digest in input_report['source_hashes'].items(): assert sha(ROOT/path)==digest
    kinds=Counter(); reps=Counter(); configurations={}; optimization=[]; checked=0; exported=0
    matrix={n:np.full((3,88),-1,dtype=int) for n in report['results']}
    folds=[]; checked_bases=set()
    booster_dir=OUT/'xgboost_exports'; booster_dir.mkdir(exist_ok=True)
    def check_basis(path,expected_train):
        saved=joblib.load(path)
        assert saved['signature']==signature() and np.array_equal(saved['train'],expected_train)
        w=weights(y[expected_train]); w/=w.sum()
        expected=np.einsum('n,nij->ij',w,x[expected_train],optimize=True)
        for m in saved['models'].values():
            np.testing.assert_allclose(m.training_moment_,expected,atol=1e-10)
            b=m.basis_; np.testing.assert_allclose(b.T@b,np.eye(b.shape[1]),atol=1e-10)
        checked_bases.add(str(path))
        return saved
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=9060)
    for split,(train,test) in enumerate(cv.split(subjects,y)):
        fold=json.loads((OUT/f'fold_{split:02d}.json').read_text())
        artifact=joblib.load(OUT/'models'/f'fold_{split:02d}.joblib')
        assert fold['signature']==artifact['signature']==signature()
        assert fold['train']==train.tolist() and fold['test']==test.tolist()
        assert np.array_equal(artifact['train'],train) and np.array_equal(artifact['test'],test)
        assert np.array_equal(artifact['subjects_train'],subjects[train]) and np.array_equal(artifact['subjects_test'],subjects[test])
        assert not set(train)&set(test) and fold['truth']==y[test].tolist()
        seen=[]
        for inner in fold['inner_provenance']:
            it,iv=np.array(inner['train']),np.array(inner['validation'])
            assert not set(it)&set(iv) and set(it)|set(iv)==set(train) and not (set(it)|set(iv))&set(test)
            seen.extend(iv.tolist()); check_basis(inner['basis_file'],it)
            iw=weights(y[it]); np.testing.assert_allclose([iw[y[it]==c].sum() for c in range(3)],len(it)/3)
            optimization.extend(inner['supervised_optimization'])
        assert sorted(seen)==train.tolist() and len(fold['inner_results'])==139
        selected=select(fold['inner_results'])
        saved=check_basis(artifact['basis_file'],train)
        bank={'spectral':controls}
        for name,m in saved['models'].items(): bank[name]=m.transform(x)
        bank['fusion']=np.concatenate([bank['tt16'],controls],axis=1)
        for name,entry in artifact['models'].items():
            cfg=entry['config']; model=entry['model']
            assert cfg==selected[name]['config']==fold['predictions'][name]['config']
            z=x if cfg['kind']=='supervised_tt' else bank[cfg['representation']]
            pred=model.predict(z[test]); s,stype=scores(model,z[test])
            assert np.array_equal(pred,fold['predictions'][name]['predictions'])
            np.testing.assert_allclose(s,fold['predictions'][name]['scores'],atol=1e-10)
            assert stype==fold['predictions'][name]['score_type']
            assert np.all(matrix[name][split//5,test]==-1)
            matrix[name][split//5,test]=pred
            configurations.setdefault(name,Counter())[json.dumps(cfg,sort_keys=True)]+=1
            if cfg['kind']!='supervised_tt':
                fitted=z[train]
                if 'select' in model.named_steps: fitted=model.named_steps['select'].transform(fitted)
                expected=np.average(fitted,axis=0,weights=weights(y[train]))
                np.testing.assert_allclose(model.named_steps['scale'].mean_,expected,atol=1e-10)
            elif name=='clf_supervised_tt':
                init=DelayProjection('tt',rank=cfg['filters'],channel_rank=cfg['channel_rank']).fit(x[train],y[train])
                np.testing.assert_allclose(model.u0_,init.cores_[0][0],atol=1e-10)
                np.testing.assert_allclose(model.v0_,init.cores_[1],atol=1e-10)
                assert model.anchor==cfg['anchor'] and model.penalty==cfg['penalty']
                optimization.append(model.optimization_)
            if cfg['kind']=='xgboost':
                estimator=model.named_steps['clf']
                export=booster_dir/f'fold_{split:02d}_{name}.ubj'
                estimator.save_model(export)
                restored=xgboost.XGBClassifier(); restored.load_model(export)
                features=model[:-1].transform(z[test])
                np.testing.assert_allclose(restored.predict_proba(features),estimator.predict_proba(features),atol=1e-10)
                assert np.array_equal(restored.predict(features),pred)
                exported+=1
            checked+=1
        choice=fold['predictions']['inner_selected_pipeline']['config']
        kinds[choice['kind']]+=1; reps[group_rep(choice)]+=1
        folds.append(fold)
        print(f'Verified outer {split+1}/15: all selected models, scalers, bases and XGBoost exports.',flush=True)
    assert checked==150 and len(checked_bases)==60
    for name,preds in matrix.items():
        assert np.all(preds>=0)
        vals=[metric(y,p) for p in preds]
        np.testing.assert_allclose(np.mean([r['balanced_accuracy'] for r in vals]),report['results'][name]['mean_balanced_accuracy'])
    opt_summary=dict(fits=len(optimization),converged=sum(o['success'] for o in optimization),
                     statuses=dict(Counter(o['message'] for o in optimization)))
    verification=dict(status='PASS',source_arrays_unchanged=True,model_entries_replayed=checked,basis_sets_verified=len(checked_bases),
                       outer_splits=15,inner_splits=45,all_held_out_scores_reproduced=True,all_selection_choices_reproduced=True,
                       training_only_scaling_verified=True,class_weight_totals_verified=True,xgboost_exports_replayed=exported,
                       xgboost_version=xgboost.__version__,xgboost_build=xgboost.build_info(),
                       automatic_classifier_choices=dict(kinds),automatic_representation_choices=dict(reps),
                       supervised_optimization=opt_summary)
    (OUT/'verification_report.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    selected_configs={k:[dict(config=json.loads(c),outer_folds=n) for c,n in counts.most_common()] for k,counts in configurations.items()}
    (OUT/'selected_configurations.json').write_text(json.dumps(selected_configs,indent=2),encoding='utf-8')
    display=dict(inner_selected_pipeline='Inner-selected complete pipeline (PRIMARY)',clf_logistic='Tuned logistic regression',
                  clf_svm='Tuned RBF SVM',clf_xgboost='Tuned XGBoost',clf_random_forest='Tuned random forest',
                  clf_supervised_tt='Stronger-regularized supervised TT',rep_wave_tt='Waveform TT + tuned classifier',
                  rep_wave_tucker='Waveform Tucker + tuned classifier',rep_spectral='Spectral features + tuned classifier',
                  rep_fusion='TT/spectral fusion + tuned classifier')
    r=report['results']; primary=r['inner_selected_pipeline']['mean_balanced_accuracy']; previous=old['results']['inner_selected_pipeline']['mean_balanced_accuracy']
    best=max((n for n in r if n.startswith('clf_')),key=lambda n:r[n]['mean_balanced_accuracy'])
    lines=['# Tensor/classifier optimization experiment', '', 'Date: 2026-09-06. Earlier datasets, model artifacts and reports remain preserved.', '',
           '## Result', '',
           f'The complete nested-selection pipeline achieved **{100*primary:.1f}% balanced accuracy**, versus **{100*previous:.1f}%** '
           f'for V3 on the same cohort and outer subject splits ({100*(primary-previous):+.1f} percentage points). '
           'This is a comparison of the full tuning procedures, not an independently established improvement.', '',
           f'The strongest observed classifier family was **{display[best]}**, at **{100*r[best]["mean_balanced_accuracy"]:.1f}%**. '
           f'XGBoost achieved **{100*r["clf_xgboost"]["mean_balanced_accuracy"]:.1f}%**. '
           'Best-family results are secondary; they cannot replace the predeclared primary endpoint after inspection.', '',
           '## Classifier comparison', '',
           '| Method | Balanced accuracy ± repeat SD | Accuracy | Macro F1 |', '|---|---:|---:|---:|']
    order=['inner_selected_pipeline']+sorted([n for n in r if n.startswith('clf_')],key=lambda n:-r[n]['mean_balanced_accuracy'])
    for n in order:
        z=r[n]; lines.append(f'| {display[n]} | {100*z["mean_balanced_accuracy"]:.1f}% ± {100*z["sd_balanced_accuracy"]:.1f} pp | {100*z["mean_accuracy"]:.1f}% | {z["mean_macro_f1"]:.3f} |')
    lines += ['', 'Each classifier-family row tunes its feature representation as well as its settings inside inner CV. '
              'It does not necessarily use the same tensor representation in every outer fold. Results are means across '
              'three complete subject-level CV repetitions; SD is not a confidence interval.', '',
              '## Does tensor representation help?', '',
              '| Representation family | Balanced accuracy ± repeat SD | Macro F1 |','|---|---:|---:|']
    for n in sorted([n for n in r if n.startswith('rep_')],key=lambda n:-r[n]['mean_balanced_accuracy']):
        z=r[n]; lines.append(f'| {display[n]} | {100*z["mean_balanced_accuracy"]:.1f}% ± {100*z["sd_balanced_accuracy"]:.1f} pp | {z["mean_macro_f1"]:.3f} |')
    lines += ['', 'These rows select classifiers/settings inside each representation family. Fusion starts with TT16 plus '
              '418 spectral features, then selects 32 training-side features; it can favor spectral features and does not '
              'guarantee a tensor contribution. No diagnosis, MMSE, age, sex, duration or recording-count predictor is used.', '',
              '## Recall by diagnosis', '', '| Classifier family | AD recall | FTD recall | CN recall |','|---|---:|---:|---:|']
    for n in order:
        recall=np.mean([a['recall'] for a in r[n]['repeats']],axis=0)
        lines.append(f'| {display[n]} | '+' | '.join(f'{100*v:.1f}%' for v in recall)+' |')
    lines += ['', 'Class weighting is unchanged: each training diagnosis has equal total loss weight and each person '
              'contributes one pooled observation. More leftover windows do not become independent people. Baseline '
              'balanced accuracy for always predicting one class is 33.3%; always predicting AD gives 40.9% ordinary accuracy.', '',
              '## What was optimized', '',
              '- 139 configurations were fixed before new outcomes: seven feature settings paired with 19 classifier settings, plus six supervised TT settings.',
              '- XGBoost: shallow trees, learning rate, number of trees, minimum child weight, L1/L2 penalties and row subsampling; column subsampling fixed at 0.8.',
              '- SVM: C and gamma; logistic regression: stronger C regularization; random forest: depth and minimum leaf size.',
              '- Supervised TT: stronger core anchoring, smaller channel/filter ranks, head regularization, and a fixed 500-iteration budget.',
              '- All rank/feature/classifier decisions occurred inside three-fold inner subject CV; outer evaluation was the same 3x5 subject CV as V3.',
              '- No outer-test early stopping and no expansion of the grid after inspecting new outcomes.', '',
              'Automatic classifier choices across the 15 outer fits: '+json.dumps(dict(kinds))+'. '
              'Representation choices: '+json.dumps(dict(reps))+'. Full per-fold settings are in `selected_configurations.json`.', '',
              '## Optimizer behavior', '',
              f'{opt_summary["converged"]}/{opt_summary["fits"]} supervised optimization fits reported convergence. '
              'This count comprises 270 inner fits and 15 selected supervised outer fits, excluding aliases. Statuses: '+json.dumps(opt_summary['statuses'])+'.', '',
              'Any remaining iteration-budget exhaustion is a limitation, not a successful convergence claim. Even a '
              'converged nonconvex tensor fit need not be globally optimal. Stronger regularization changes the model '
              'objective; this is not simply the previous model trained longer.', '',
              '## Verification and software', '',
              f'- Replayed all {checked} saved outer model entries and scores. Selected-pipeline entries may reference other saved models.',
              '- Checked 45 inner and 15 outer subject splits, training-only decomposition moments, orthonormal bases, weighted scaler means and exact inner selection.',
              f'- Exported and reloaded {exported} XGBoost boosters in UBJ format; their predictions/probabilities match the saved complete pipelines.',
              '- Verified original primary and leftover array hashes unchanged and reused the exact V3 retained inputs.',
              '- Installed official `xgboost-cpu==3.2.0` into the existing Python 3.10 user installation (approximately 2.1 MB); no dependency upgrades, virtual environment, PyTorch or GPU runtime.',
              f'- Nested fitting time: {report["total_seconds"]:.1f} seconds, excluding installation and verification.', '',
              'Booster files alone are not the complete EEG predictor: retain the fitted feature basis, feature selector '
              'and scaler in the corresponding pipeline. Only load trusted local joblib files. Runtime versions and '
              'checksums are saved for reproducibility.', '',
              '## Interpretation', '',
              'This bounded tuning experiment tests whether classifier choice and stronger regularization help the '
              'existing representations. It does not establish the dataset\'s maximum achievable accuracy. The primary '
              'and every family score are retained regardless of whether tuning helped.', '',
              'The 88 source people and their outer splits have already been explored. Keeping tuning inside inner '
              'folds does not undo researcher-level adaptive reuse. An independent cohort is needed to confirm a '
              'performance claim. Waveform features remain pooled second-order delay statistics, not a general '
              'long-sequence neural model; no clinical-readiness claim is made.', '',
              '## Artifacts', '',
              '- [Detailed protocol](../../v4_tuning/PROTOCOL.md)',
              '- [Reproduction instructions](../../v4_tuning/README.md)',
              '- `protocol.json`: every candidate and fixed evaluation setting.',
              '- `nested_report.json`, `fold_XX.json`: all predictions, scores, selections and optimization status.',
              '- `models/`: complete outer pipelines; `bases/`: training-only inner and outer tensor bases.',
              '- `xgboost_exports/`: portable boosters, to use with matching preprocessing.',
              '- `verification_report.json`, `selected_configurations.json`, `artifact_manifest.json`.', '',
              'Official references: [XGBoost installation](https://xgboost.readthedocs.io/en/stable/install.html) and '
              '[parameter definitions](https://xgboost.readthedocs.io/en/stable/parameter.html).','']
    (OUT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json')
    paths+=sorted(p for p in Path(__file__).parent.glob('*') if p.is_file())
    manifest=dict(signature=signature(),python=platform.python_version(),
                   packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','xgboost-cpu','joblib']},
                   files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)
    print(f'Report: {OUT/"EXPERIMENT_REPORT.md"}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
