"""Verify source accounting, fitted models, learning curve, and write report."""
from collections import Counter
from pathlib import Path
import json
import platform
import importlib.metadata
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits
from prepare import ROOT,OUT,PARTS,sha,delay_moment,spectral_features
from run import FAMILIES,load_inputs,metric,ranked,sig
from wave_models import weights,balanced_moment,DelayProjection


def main():
    x,controls,subjects,y = load_inputs()
    prep = json.loads((OUT/'preparation_report.json').read_text())
    report = json.loads((OUT/'nested_report.json').read_text())
    records = json.loads((OUT/'retained_windows.json').read_text())
    assert report['completed_outer_folds']==15 and report['signature']==sig()
    for name,h in prep['source_hashes'].items(): assert sha(ROOT/name)==h
    keep_ids = {r['sample_id'] for r in records}
    exc_ids = {r['sample_id'] for r in prep['exclusions']}
    assert len(keep_ids)==len(records)==16824 and len(exc_ids)==595
    assert not keep_ids & exc_ids
    metas = [dict(np.load(p/'metadata.npz')) for p in PARTS]
    arrays = [np.load(p/'signals.npy',mmap_mode='r') for p in PARTS]
    assert keep_ids | exc_ids == set(np.concatenate([m['sample_id'] for m in metas]))
    assert np.array_equal(np.bincount(y),[36,23,29]) and len(subjects)==88
    old_v2 = set(np.load(ROOT/'tensor_pipeline/outputs/v2_tensor_train/index.npz')['sample_id'])
    assert old_v2 <= keep_ids
    source_checks = []
    for sub in ['sub-001','sub-005','sub-062','sub-086']:
        rs = [r for r in records if r['subject']==sub]
        waves = []
        for row in rs:
            part = ['primary','leftover'].index(row['origin'])
            i = row['source_index']
            assert str(metas[part]['sample_id'][i])==row['sample_id']
            wave = np.asarray(arrays[part][i],dtype=np.float32).copy()
            wave -= wave.mean(axis=0,keepdims=True)
            wave -= wave.mean(axis=1,keepdims=True)
            waves.append(wave)
        waves = np.array(waves)
        fresh = np.mean([delay_moment(w) for w in waves],axis=0)
        idx = np.flatnonzero(subjects==sub)[0]
        np.testing.assert_allclose(fresh,x[idx],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(spectral_features(waves),controls[idx],rtol=1e-10,atol=1e-10)
        source = loadmat(ROOT/f'dataset/ds004504/derivatives/{sub}/eeg/{sub}_task-eyesclosed_eeg.set',variable_names=['data'])['data'].astype('float32')
        error = 0.
        for j in [0,len(rs)-1]:
            start = int(rs[j]['start_seconds']*500)
            expected = source[:,start:start+2000:2].copy()
            expected -= expected.mean(1,keepdims=True)
            expected -= expected.mean(0,keepdims=True)
            expected -= expected.mean(1,keepdims=True)
            error = max(error,float(np.max(np.abs(expected-waves[j]))))
        assert error<.001
        source_checks.append(dict(subject=sub,windows_recomputed=len(rs),maximum_source_error_uv=error))
    assert np.isfinite(x).all() and np.isfinite(controls).all()
    np.testing.assert_allclose(x,x.transpose(0,2,1),atol=1e-10)
    assert min(np.linalg.eigvalsh(s)[0] for s in x)>-1e-6
    for row in records:
        assert row['label']==int(y[np.flatnonzero(subjects==row['subject'])[0]])
        assert not any(row['start_seconds']<t+.5 and row['start_seconds']+4>t-.5
                       for t in prep['source_headers'][row['subject']]['cuts_seconds'])
    names = FAMILIES+['inner_selected_pipeline']
    pred_matrix = {n:np.full((3,88),-1,dtype=int) for n in names}
    optimizations, family_choices, folds = [],Counter(),[]
    model_entries = 0
    splits = list(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=9060).split(subjects,y))
    for split,(train,test) in enumerate(splits):
        fold = json.loads((OUT/f'fold_{split:02d}.json').read_text())
        saved = joblib.load(OUT/'models'/f'fold_{split:02d}.joblib')
        assert fold['signature']==saved['signature']==sig()
        assert fold['train']==train.tolist() and fold['test']==test.tolist()
        assert np.array_equal(saved['train'],train) and np.array_equal(saved['test'],test)
        assert np.array_equal(saved['train_subjects'],subjects[train]) and np.array_equal(saved['test_subjects'],subjects[test])
        assert not set(train)&set(test)
        assert fold['truth']==y[test].tolist()
        val_seen = []
        for inner in fold['inner_provenance']:
            it,iv = set(inner['train']),set(inner['validation'])
            assert not it&iv and it|iv==set(train) and not (it|iv)&set(test)
            val_seen.extend(iv)
            optimizations.extend(inner['supervised_optimization'])
            iy = y[inner['train']]; iw = weights(iy)
            np.testing.assert_allclose([iw[iy==c].sum() for c in range(3)],len(iy)/3)
        assert sorted(val_seen)==train.tolist()
        ranking = ranked(fold['inner_results'])
        for name in names:
            model = saved['models'][name]
            wanted = ranking[0] if name=='inner_selected_pipeline' else next(r for r in ranking if r['config']['family']==name)
            assert model.cfg==wanted['config']==fold['predictions'][name]['config']
            assert np.array_equal(model.fit_labels_,y[train])
            pred = model.predict(x[test],controls[test])
            scores = model.scores(x[test],controls[test])
            assert np.array_equal(pred,fold['predictions'][name]['predictions'])
            np.testing.assert_allclose(scores,fold['predictions'][name]['scores'],rtol=1e-10,atol=1e-10)
            assert np.all(pred_matrix[name][split//5,test]==-1)
            pred_matrix[name][split//5,test]=pred
            if name in ['wave_tt','wave_tucker']:
                np.testing.assert_allclose(model.projector_.training_moment_,balanced_moment(x[train],y[train]),atol=1e-10)
                b = model.projector_.basis_
                np.testing.assert_allclose(b.T@b,np.eye(b.shape[1]),atol=1e-10)
            if name=='supervised_tt':
                init = DelayProjection('tt',rank=8,channel_rank=4).fit(x[train],y[train])
                np.testing.assert_allclose(model.model_.u0_,init.cores_[0][0],atol=1e-10)
                np.testing.assert_allclose(model.model_.v0_,init.cores_[1],atol=1e-10)
                optimizations.append(model.model_.optimization_)
            model_entries += 1
        family_choices[fold['predictions']['inner_selected_pipeline']['config']['family']] += 1
        folds.append(fold)
    for name,preds in pred_matrix.items():
        assert np.all(preds>=0)
        ms = [metric(y,p) for p in preds]
        np.testing.assert_allclose(np.mean([m['balanced_accuracy'] for m in ms]),report['results'][name]['mean_balanced_accuracy'])
    curve = json.loads((OUT/'learning_curve.json').read_text())
    assert curve['signature']==sig() and len(curve['points'])==15
    curve_summary = []
    for split,(train,test) in enumerate(splits[:5]):
        previous = set()
        for point in sorted([p for p in curve['points'] if p['split']==split],key=lambda p:p['fraction']):
            sub = np.array(point['train'])
            assert previous <= set(sub) <= set(train) and not set(sub)&set(test)
            previous = set(sub)
            assert point['test']==test.tolist()
            model = joblib.load(OUT/'models'/point['model_file'])
            assert model['signature']==sig() and np.array_equal(model['train'],sub) and np.array_equal(model['test'],test)
            p = model['model'].predict(x[test],controls[test])
            assert np.array_equal(p,point['predictions'])
            assert model['model'].cfg==dict(family='supervised_tt',penalty=.1)
            optimizations.append(point['optimization'])
            model_entries += 1
        assert previous==set(train)
    for fraction in [.4,.7,1.]:
        points = [p for p in curve['points'] if p['fraction']==fraction]
        truth = np.concatenate([p['truth'] for p in points])
        pred = np.concatenate([p['predictions'] for p in points])
        curve_summary.append(dict(fraction=fraction,mean_training_subjects=float(np.mean([len(p['train']) for p in points])),
                                  training_counts=[len(p['train']) for p in points],
                                  held_out=metric(truth,pred),mean_training_ba=float(np.mean([p['training_metrics']['balanced_accuracy'] for p in points]))))
    assert all(o['final_loss']<=o['initial_loss']+1e-8 for o in optimizations)
    optimization_summary = dict(fits=len(optimizations),reported_converged=sum(o['success'] for o in optimizations),
                                 status_counts=dict(Counter(o['message'] for o in optimizations)),
                                 median_gradient_inf_norm=float(np.median([o['gradient_inf_norm'] for o in optimizations])))
    verification = dict(status='PASS',source_arrays_unchanged=True,all_v2_windows_retained=True,
                         checked_model_entries=model_entries,outer_folds=15,inner_folds=45,learning_curve_models=15,
                         source_reconstruction_checks=source_checks,all_predictions_and_scores_replayed=True,
                         all_test_subjects_excluded_from_training=True,all_inner_choices_reproduced=True,
                         equal_class_weight_mass=True,subject_window_averaging=True,
                         automatic_family_choices=dict(family_choices),optimization=optimization_summary,
                         learning_curve=curve_summary,
                         caveat='Numerical/provenance checks do not imply diagnostic validity or optimizer convergence.')
    (OUT/'verification_report.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    display = dict(wave_tt='Waveform TT + LR/SVM',wave_tucker='Waveform Tucker + LR/SVM',
                   supervised_tt='Supervised waveform TT',spectral_control='Spectral control, same 88 people',
                   inner_selected_pipeline='Inner-selected entire pipeline (PRIMARY)')
    order = ['inner_selected_pipeline']+sorted(FAMILIES,key=lambda n:-report['results'][n]['mean_balanced_accuracy'])
    lines = ['# Full-cohort waveform tensor experiment: results', '',
             'Date: 2026-09-06. All previous data partitions, models and reports are preserved.', '',
             '## Outcome', '',
             'Using the leftovers and class/subject weighting did not produce the hoped-for waveform accuracy improvement. '
             'The primary nested method-selection endpoint is **52.0% balanced accuracy**. Waveform methods range from '
             '**49.9% to 51.8%**, while the same-cohort spectral control achieves **55.8%**. '
             'This experiment does not establish a maximum attainable accuracy for this dataset.', '',
             '| Pipeline | Balanced accuracy ± repeat SD | Accuracy | Macro F1 |', '|---|---:|---:|---:|']
    for name in order:
        r = report['results'][name]
        lines.append(f"| {display[name]} | {100*r['mean_balanced_accuracy']:.1f}% ± {100*r['sd_balanced_accuracy']:.1f} pp | {100*r['mean_accuracy']:.1f}% | {r['mean_macro_f1']:.3f} |")
    lines += ['', 'All scores are subject-level, from three repeats of five-fold outer CV with three-fold inner tuning. '
              'Repeat SD is descriptive, not a confidence interval. Balanced accuracy gives each diagnosis equal weight. '
              'With 36/23/29 people, ordinary accuracy is no longer identical to balanced accuracy. Always predicting AD '
              'would give 40.9% accuracy but only 33.3% balanced accuracy.', '',
              'The primary row tests selecting a complete pipeline using only training-side validation. Per-family rows '
              'are secondary comparisons; selecting the best after inspecting outer scores is not an independently '
              'confirmed performance estimate.', '',
              '## Data actually used', '',
              '| Diagnosis | People | Four-second windows | Retained duration |', '|---|---:|---:|---:|']
    for i,g in enumerate(['AD','FTD','CN']):
        count = prep['retained_windows_per_class'][g]; seconds = count*4
        lines.append(f'| {g} | {int(np.sum(y==i))} | {count:,} | {seconds//3600} h {(seconds%3600)//60} min {seconds%60} s |')
    lines += ['', '- Total: 88 people and 16,824 windows (18 h 41 min 36 s).',
              '- Included 4,828 windows from the old leftover partition and recovered 54 valid balance-only trims.',
              '- All 11,942 V2 windows remained; no former primary artifact exclusion was reinstated.',
              '- Applied the same reference, amplitude/step/flatness rules and boundary guards to both source partitions.',
              '- Excluded 595 windows with IDs/reasons retained. No signals were deleted; partial tails remain separate.', '',
              'FTD imbalance is handled by training-fold class weights, not duplication. Each diagnosis has one third '
              'of training loss weight, and each subject contributes one pooled observation. More windows improve that '
              'person\'s pooled estimate but do not make them count as more independent people.', '',
              '## What the waveform methods tested', '',
              'Cleaned EEG is resampled to 125 Hz and represented as channel × lag × local-position tensors. '
              'TT/Tucker filters measure short-lag channel/time relationships; squared responses are pooled across '
              'positions and windows. An exact covariance identity makes this computationally small enough for CPU. '
              'The waveform branches do not use Fourier/PSD features.', '',
              'The supervised TT branch jointly learns its channel and lag/filter cores and classification head from '
              'weighted diagnosis loss. It is a compact, second-order waveform model—not a general long-sequence neural '
              'network. Fixed pooling still discards window ordering and higher-order waveform detail. '
              'Its failure to beat the spectral control does not rule out all raw-signal models.', '',
              'Waveform branches retain approximately 0.5–45 Hz source content, while the spectral control uses the '
              'earlier 1–30 Hz convention. This is a pipeline comparison, not a perfectly bandwidth-matched ablation.', '',
              '## Recall by diagnosis', '',
              '| Pipeline | AD recall | FTD recall | CN recall |', '|---|---:|---:|---:|']
    for name in order:
        recall = np.mean([r['recall'] for r in report['results'][name]['repeats']],axis=0)
        lines.append(f'| {display[name]} | '+' | '.join(f'{100*v:.1f}%' for v in recall)+' |')
    lines += ['', '## Does adding more people help?', '',
              'A prespecified secondary learning curve used one fixed supervised TT configuration (head penalty 0.1), '
              'the first five outer folds, and nested stratified subsets of each training set. No curve outcome was '
              'used to choose model settings. Each point below pools held-out predictions for all 88 people once.', '',
              '| Training fraction | Mean training people per fold | Training balanced accuracy | Held-out balanced accuracy |', '|---|---:|---:|---:|']
    for r in curve_summary:
        lines.append(f"| {100*r['fraction']:.0f}% | {r['mean_training_subjects']:.1f} | {100*r['mean_training_ba']:.1f}% | {100*r['held_out']['balanced_accuracy']:.1f}% |")
    curve_values = [r['held_out']['balanced_accuracy'] for r in curve_summary]
    lines += ['', ('Held-out performance increased at every tested size. This is compatible with additional people helping this model, '
                   'but the single five-fold curve is too small to extrapolate a required sample size or target accuracy.'
                   if all(b>a for a,b in zip(curve_values,curve_values[1:])) else
                   'Held-out performance did not increase steadily with training size. This small, noisy curve does not demonstrate '
                   'a reliable data-scaling trend or establish how many additional people are needed.'), '',
              'The original cohort still contains only 23 independent FTD people. Leftover windows do not add new FTD '
              'participants. Broader independent participants and a separate external test cohort remain distinct needs; '
              'oversampling cannot create that independence.', '',
              '## Optimization and verification', '',
              f'- Replayed all {model_entries} saved model entries: 75 outer pipeline entries (including selected-pipeline aliases) and 15 learning-curve models.',
              '- Verified all 15 outer and 45 inner subject splits, training-only choices, class loss weights, bases and scores.',
              '- Recomputed waveform moments and spectral controls from every retained window of four selected subjects, '
              'including new AD/CN people and an FTD recording with many cuts; checked representative windows against source EEGLAB signals.',
              '- Confirmed source array SHA-256 hashes unchanged, unique-window accounting, no test/training overlap, '
              'finite positive-semidefinite moments, and retention of every V2 window.',
              '- Numerical tests verify raw squared-response/covariance equivalence, TT-SVD reconstruction against TensorLy, '
              'orthonormality, batch-invariant projection, analytic supervised gradients and class weight totals.',
              f'- Nested fitting time: {report["total_fit_seconds"]:.1f} seconds on CPU, excluding preparation, curve and verification. No new packages, CUDA or virtual environment were used.', '',
              f'Supervised optimizer status: {optimization_summary["reported_converged"]}/{optimization_summary["fits"]} fits reported convergence under the fixed 100-iteration budget. '
              'All fits reduced their objective from initialization. Status counts: '+json.dumps(optimization_summary['status_counts'])+'.', '',
              '**Budget-limited optimization is a remaining limitation**, not proof of a globally optimal fit. '
              'L-BFGS solves a nonconvex factorized problem; even its convergence flag would not certify a global optimum. '
              'No iteration budget was changed in response to outer-test accuracy.', '',
              'A separate [fixed-budget convergence sensitivity](convergence_sensitivity/REPORT.md) compares 100 versus 2000 iterations '
              'for the already specified penalty-0.1 model. It was added after the original run because of the optimizer status flags; '
              'it is post-hoc, reports both outcomes, and does not replace this primary result.', '',
              '## Interpretation and course reporting', '',
              'The tested waveform representations did not beat the spectral control on this expanded cohort. '
              'This is useful negative evidence about these specific tensor filters and pooling choices; it is not '
              'evidence that tensor methods or raw EEG cannot work.', '',
              'Do not call this a controlled improvement over V2: cohort, window coverage, representation and fold seed '
              'changed. These same source people have been explored repeatedly, so nested fitting does not create an '
              'untouched external evaluation cohort. Class weighting addresses training imbalance but cannot manufacture '
              'diagnostic signal, remove every acquisition confound or guarantee higher FTD recall.', '',
              'No further model search was performed after inspecting these outcomes. Preserve the full result table '
              'for the course, including the spectral control and the primary nested-selection endpoint.', '',
              '## Files', '',
              '- [Detailed protocol](../../v3_waveform/PROTOCOL.md)',
              '- [Reproduction instructions](../../v3_waveform/README.md)',
              '- `preparation_report.json`: exclusions, cohort accounting and source fingerprints.',
              '- `retained_windows.json`: every included sample ID and its original partition/index.',
              '- `protocol.json`, `fold_XX.json`, `nested_report.json`: fixed search, splits and all outcomes.',
              '- `models/fold_XX.joblib`: complete outer models, factors, scalers and heads with subject IDs.',
              '- `learning_curve.json` and `models/curve_XX_YYY.joblib`: fixed-model size diagnostic.',
              '- `verification_report.json` and `artifact_manifest.json`: checks and replay fingerprints.', '']
    (OUT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    paths = list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.md'))+list(OUT.glob('*'))+list((OUT/'models').glob('*.joblib'))
    paths = sorted(set(p for p in paths if p.is_file() and p.name!='artifact_manifest.json'))
    manifest = dict(signature=sig(),python=platform.python_version(),
                    packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','tensorly','joblib']},
                    files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)
    print(f'Report: {OUT/"EXPERIMENT_REPORT.md"}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
