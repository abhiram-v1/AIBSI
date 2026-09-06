"""Read-only model/data verification, then write V2 reports and fingerprints.

No fitting, hyperparameter changes, or additional model selection occur here.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import platform
import importlib.metadata
import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits
from benchmark import Experiment, DATA, FAMILIES, signature, metrics
from models import apply_smoothing, predict_with_scores


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pooled(z):
    return np.stack([z.mean(0), z.std(0), np.median(z, axis=0),
                     np.quantile(z, .75, axis=0)-np.quantile(z, .25, axis=0)], axis=1)


def main():
    out = DATA / 'nested_benchmark'
    exp = Experiment(out)
    prep = json.loads((DATA / 'preparation_report.json').read_text())
    report = json.loads((out / 'nested_report.json').read_text())
    assert report['completed_outer_folds'] == 15 and report['signature'] == signature()
    for name, expected in prep['source_hashes'].items():
        assert sha(Path(prep['source']) / name) == expected, f'Source changed: {name}'
    assert np.array_equal(np.bincount(exp.labels), [23, 23, 23])
    assert len(set(exp.index['sample_id'])) == len(exp.windows) == 11942
    assert len(set(exp.index['source_indices'])) == len(exp.windows)
    excluded = {e['source_index'] for e in prep['excluded']}
    assert not excluded & set(exp.index['source_indices'])
    assert len(excluded) + len(exp.windows) == prep['original_windows']
    assert np.isfinite(exp.subject_tensor).all() and np.isfinite(exp.controls).all()
    assert np.isfinite(exp.windows).all()
    assert all(len(ix) >= 24 for ix in exp.by_subject)
    forbidden = ['MMSE', 'diagnosis', 'duration', 'age/', 'sex/', 'count/']
    assert not any(s.lower() in n.lower() for s in forbidden for n in exp.index['control_names'])

    provenance_files = set()
    family_names = FAMILIES + ['inner_selected_pipeline']
    all_pred = {n: np.full((3, 69), -1, dtype=int) for n in family_names}
    selection_counts = {n: Counter() for n in family_names}
    family_counts = Counter()
    train_ba = {n: [] for n in family_names}
    inner_ba = {n: [] for n in family_names}
    replay_count = 0
    max_pool_error = 0.
    folds = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=5040)
    for split, (expected_train, expected_test) in enumerate(cv.split(exp.subjects, exp.labels)):
        fold = json.loads((out / f'fold_{split:02d}.json').read_text())
        folds.append(fold)
        train = np.array(fold['train'])
        test = np.array(fold['test'])
        assert np.array_equal(train, expected_train) and np.array_equal(test, expected_test)
        assert not set(train) & set(test)
        assert np.array_equal(fold['true_labels'], exp.labels[test])
        assert fold['test_subjects'] == exp.subjects[test].tolist()
        assert fold['signature'] == exp.sig
        val_seen = []
        for inner in fold['inner_provenance']:
            it, iv = set(inner['train']), set(inner['validation'])
            assert not it & iv and it | iv == set(train)
            assert not (it | iv) & set(test)
            val_seen.extend(iv)
            p = Path(inner['decomposition_file'])
            provenance_files.add(p)
            saved = joblib.load(p)
            assert set(saved['train_subject_indices']) == it
            assert saved['signature'] == exp.sig
        assert sorted(val_seen) == sorted(train)
        assert len(fold['inner_results']) == 144
        def selection_key(s):
            c = s['config']
            return (-s['balanced_accuracy'], -s['macro_f1'],
                    c['rank'] or c['selection'], json.dumps(c, sort_keys=True))
        ranked = sorted(fold['inner_results'], key=selection_key)
        saved_models = joblib.load(out / f'fold_{split:02d}_models.joblib')
        assert set(saved_models) == set(family_names)
        p = Path(saved_models['tt_subject']['decomposition_file'])
        provenance_files.add(p)
        basis = joblib.load(p)
        assert set(basis['train_subject_indices']) == set(train)
        arrays_file = np.load(p.with_suffix('.npz'))
        assert set(arrays_file['fit_subject_indices']) == set(train)
        arrays = {k: arrays_file[k].copy() for k in arrays_file.files if k != 'fit_subject_indices'}

        # Rebuild ALL held-out tensor features directly from persisted bases and
        # window spectra, rather than relying on the saved feature cache.
        fresh_subject = basis['models']['tt_subject'].transform(exp.subject_tensor[test])
        np.testing.assert_allclose(fresh_subject, arrays['tt_subject'][test], rtol=1e-8, atol=1e-8)
        arrays['tt_subject'][test] = fresh_subject
        for name in ['tt_window_pool', 'graph_tt_window_pool', 'tucker_window_pool']:
            for i in test:
                wx = np.asarray(exp.windows[exp.by_subject[i]], dtype=float)
                if name.startswith('graph'):
                    wx = apply_smoothing(wx, basis['smooth'])
                fresh = pooled(basis['models'][name].transform(wx))
                max_pool_error = max(max_pool_error, float(np.max(np.abs(fresh - arrays[name][i]))))
                np.testing.assert_allclose(fresh, arrays[name][i], rtol=1e-8, atol=1e-8)
                arrays[name][i] = fresh
        for name, saved in saved_models.items():
            cfg = saved['config']
            expected = ranked[0] if name == 'inner_selected_pipeline' else next(
                r for r in ranked if r['config']['family'] == name)
            assert cfg == expected['config'] == fold['predictions'][name]['config']
            matrix = exp.matrix(arrays, cfg)
            pred, scores, score_type = predict_with_scores(saved['classifier'], matrix[test])
            assert np.array_equal(pred, fold['predictions'][name]['predictions'])
            np.testing.assert_allclose(scores, fold['predictions'][name]['scores'], rtol=1e-7, atol=1e-7)
            assert score_type == fold['predictions'][name]['score_type']
            assert np.all(all_pred[name][split // 5, test] == -1)
            all_pred[name][split // 5, test] = pred
            selection_counts[name][cfg['kind']] += 1
            train_ba[name].append(float(balanced_accuracy_score(exp.labels[train], saved['classifier'].predict(matrix[train]))))
            inner_ba[name].append(expected['balanced_accuracy'])
            replay_count += 1
        family_counts[saved_models['inner_selected_pipeline']['config']['family']] += 1
        print(f'Verified fold {split+1}/15: disjoint subjects, inner choices, fresh tensor projection, saved predictions.', flush=True)

    max_orth_error = 0.
    for p in sorted(provenance_files):
        saved = joblib.load(p)
        ix = saved['calibration_indices']
        train = saved['train_subject_indices']
        counts = np.bincount(exp.index['window_subject'][ix], minlength=69)
        assert np.all(counts[train] == 24) and counts.sum() == 24*len(train)
        assert len(set(ix)) == len(ix)
        assert np.array_equal(saved['train_subjects'], exp.subjects[train])
        x = np.asarray(exp.windows[ix], dtype=float)
        for name, model in saved['models'].items():
            if name == 'tt_subject':
                expected_mean = exp.subject_tensor[train].astype(float).mean(0)
            elif name.startswith('graph'):
                expected_mean = apply_smoothing(x, saved['smooth']).mean(0)
            else:
                expected_mean = x.mean(0)
            np.testing.assert_allclose(model.mean_, expected_mean, rtol=1e-8, atol=1e-8)
            err = float(np.max(np.abs(model.basis_.T @ model.basis_ - np.eye(model.basis_.shape[1]))))
            max_orth_error = max(max_orth_error, err)
            assert err < 1e-9
    for name, preds in all_pred.items():
        assert np.all(preds >= 0)
        per_repeat = [metrics(exp.labels, p) for p in preds]
        np.testing.assert_allclose(np.mean([r['balanced_accuracy'] for r in per_repeat]),
                                   report['results'][name]['mean_balanced_accuracy'])

    verification = dict(status='PASS', fitted_model_replays=replay_count,
        decomposition_sets_checked=len(provenance_files), inner_folds_checked=45, outer_folds_checked=15,
        unique_subjects=69, subject_counts={'AD':23,'FTD':23,'CN':23}, retained_windows=len(exp.windows),
        source_hashes_unchanged=True, all_test_features_recomputed=True,
        all_test_predictions_and_scores_match=True, all_split_and_selection_checks_pass=True,
        no_test_subjects_in_fitted_decomposition_calibration=True,
        calibration_windows_per_training_subject=24, max_projection_replay_error=max_pool_error,
        max_basis_orthogonality_error=max_orth_error,
        automatic_family_selection_counts=dict(family_counts),
        classifier_selection_counts={k:dict(v) for k,v in selection_counts.items()},
        training_resubstitution_ba={k:float(np.mean(v)) for k,v in train_ba.items()},
        selected_inner_validation_ba={k:float(np.mean(v)) for k,v in inner_ba.items()},
        note='Checks establish implementation/provenance consistency, not diagnostic validity or an independent test cohort.')
    (DATA / 'verification_report.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')

    labels = dict(tt_subject='Subject Tensor Train', tt_window_pool='Window Tensor Train',
                  graph_tt_window_pool='Graph-smoothed Tensor Train', tucker_window_pool='Window Tucker / HOSVD',
                  tt_spectral_fusion='Tensor Train + spectral features', spectral_control='Spectral control (no decomposition)',
                  inner_selected_pipeline='Automatic inner-CV method selection (PRIMARY)')
    order = ['inner_selected_pipeline'] + sorted(FAMILIES, key=lambda n:-report['results'][n]['mean_balanced_accuracy'])
    lines = ['# Second tensor experiment: results and verification', '',
             'Date: 2026-09-05. CPU / Python 3.10. Previous datasets, experiments and model artifacts were preserved.', '',
             '## Outcome', '',
             'The requested Tensor Train run completed, but it did **not** deliver a large accuracy improvement. '
             'The best observed individual method family was Tucker/HOSVD at **56.0%** mean subject-level balanced accuracy. '
             'The predeclared primary endpoint, selecting the entire method in inner CV, achieved **47.3%**. '
             'These are different questions: the best family after inspecting outer results must not replace the primary result.', '',
             'The earlier exploratory best was 54.1% (PCA/NMF + logistic regression). The nominal 1.9 percentage-point '
             'difference is not evidence of a reliable improvement: preprocessing and model selection changed, and the same '
             '69 subjects have already been explored. No claimed 70–80% range was reached.', '',
             '## Scores', '',
             'All numbers use held-out **subjects**, not windows. Mean and SD are over three complete five-fold CV repetitions; '
             'SD is descriptive split variation, not a confidence interval. Each repetition has 23 subjects per class, so '
             'its accuracy and balanced accuracy coincide. Uniform three-class guessing has expected accuracy 33.3%.', '',
             '| Pipeline | Balanced accuracy, mean ± SD | Macro F1 |', '|---|---:|---:|']
    for name in order:
        r = report['results'][name]
        lines.append(f"| {labels[name]} | {100*r['mean_balanced_accuracy']:.1f}% ± {100*r['sd_balanced_accuracy']:.1f} pp | {r['mean_macro_f1']:.3f} |")
    lines += ['', 'Ranks and classifiers are selected independently within each outer training fold. '
              'A family result is not a single fixed classifier evaluated 15 times. Candidates include logistic regression, '
              'linear/RBF SVM and histogram gradient boosting. The boosting implementation is **not XGBoost**.', '',
              '## Class-level performance', '',
              'Recall averaged across the three held-out repetitions:', '',
              '| Pipeline | AD recall | FTD recall | CN recall |', '|---|---:|---:|---:|']
    for name in order:
        means = [np.mean([r['sensitivities'][g] for r in report['results'][name]['repeats']]) for g in ['AD','FTD','CN']]
        lines.append(f"| {labels[name]} | " + ' | '.join(f'{100*v:.1f}%' for v in means) + ' |')
    lines += ['', 'For the strongest observed family, Tucker/HOSVD, FTD recall is 50.7%, AD recall is 59.4%, and CN recall is 58.0%. '
              'That is more even than several alternatives, but still too weak for a clinical-use claim.', '',
              '## What was corrected', '',
              '- Verified the actual electrode row order in all 69 source recordings and rebuilt the anatomical graph.',
              '- Excluded 328 windows near annotated discontinuities from this run only; no stored EEG was deleted.',
              '- Preserved absolute spectral power and variability, and used common training-only TT/Tucker bases.',
              '- Kept all windows of a person within one subject unit; each training subject supplies exactly 24 windows to basis fitting.',
              '- Nested rank, feature selection, scaling, classifier and method-family choices within training subjects.',
              '- Used SVM predictions directly; persisted decomposition cores/bases, classifiers, fold membership and scores.',
              '- Replaced the questionable old graph-CP optimizer with explicitly named graph pre-smoothing followed by TT. '
              'The old optimizer itself was not edited or presented as repaired.', '',
              'See [the detailed audit and mathematical protocol](../../v2/AUDIT_AND_PROTOCOL.md) for qualifications and references.', '',
              '## Dataset used', '',
              '| Diagnosis | Subjects | Retained four-second windows | Duration |', '|---|---:|---:|---:|',
              '| AD | 23 | 3,985 | 4 h 25 min 40 s |', '| FTD | 23 | 4,000 | 4 h 26 min 40 s |',
              '| CN | 23 | 3,957 | 4 h 23 min 48 s |', '| Total | 69 | 11,942 | 13 h 16 min 8 s |', '',
              'The subject-level classes remain exactly balanced. All retained windows contribute to pooled features. '
              'Only fitting the window basis uses the 24-per-training-subject calibration subset; the remaining windows '
              'were not thrown away. Previously separate leftover/excluded datasets were not merged or reprocessed.', '',
              '## Selection stability diagnostic', '',
              'These training/inner scores are descriptive diagnostics, not additional test-set performance estimates.', '',
              '| Pipeline | Training resubstitution BA | Selected inner validation BA | Outer held-out BA |', '|---|---:|---:|---:|']
    for name in order:
        lines.append(f"| {labels[name]} | {100*np.mean(train_ba[name]):.1f}% | {100*np.mean(inner_ba[name]):.1f}% | {100*report['results'][name]['mean_balanced_accuracy']:.1f}% |")
    lines += ['', 'Automatic inner-CV family selections across the 15 outer fits: ' +
              '; '.join(f'{labels[n]}: {v}' for n,v in family_counts.items()) + '.', '',
              'The gap between selected inner validation scores and held-out performance is consistent with unstable '
              'model selection on a small cohort. It does not prove a single causal explanation for the low score. '
              'Changing the decomposition alone has not solved class separation in these representations.', '',
              '## Verification and saved artifacts', '',
              f'- Replayed all {replay_count} saved classifiers after freshly reconstructing every held-out subject tensor feature from saved bases.',
              f'- Checked all {len(provenance_files)} decomposition sets, 45 inner splits and 15 outer splits; no test subject entered basis calibration.',
              '- All replayed predictions and decision scores matched; input clean-signal and metadata SHA-256 fingerprints were unchanged.',
              '- Confirmed finite inputs, unique sample IDs, boundary-exclusion accounting, balanced subjects, training-only centering and orthonormal bases.',
              '- Numerical unit tests compare the TT reconstruction against TensorLy TT-SVD and test full-rank recovery and batch-invariant projection.',
              f'- Benchmark fitting time: {report["total_seconds"]:.1f} seconds on CPU (excludes preparation and verification). No packages or GPU runtime installed.', '',
              'Saved classifiers: `nested_benchmark/fold_00_models.joblib` through `fold_14_models.joblib` (seven fitted pipelines each). '
              'Their records point to the corresponding decomposition files under `feature_cache/47758609d9ed833b/`. '
              'These are fold-trained models, not a separately validated all-subject deployment model. '
              'Only load trusted local joblib/pickle files.', '',
              'Machine-readable outputs: `nested_benchmark/nested_report.json`, all `fold_XX.json` prediction/configuration records, '
              '`verification_report.json`, `artifact_manifest.json`, and `preparation_report.json` with exclusion IDs.', '',
              '## Remaining limits and next decision', '',
              'This run corrects the identified implementation and evaluation issues; it cannot claim that all scientific '
              'limitations are fixed. There are only 69 independent subjects, previous researcher exposure to this cohort, '
              'an earlier globally selected/matched subset, and retained preprocessing choices such as common-average reference. '
              'Neither nested CV nor more epochs creates an untouched external test cohort.', '',
              'For the course, retain the original CP/NMF/PCA experiments, document the graph caveat, and compare them with this '
              'TT/HOSVD experiment without inventing a performance win. Reconstruction quality and diagnostic separation are distinct objectives.', '',
              'A subsequent, separately approved experiment could use the full original subject cohort with subject/class weighting '
              'and fold-safe access to extra windows, plus a small prespecified representation comparison. Leftover windows from '
              'an existing person must not be called an independent test set. That would change the agreed primary/leftover usage '
              'policy, so it has not been done here. Keep any external validation data untouched; avoid repeatedly tuning to these same outer scores.', '']
    (DATA / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

    files = sorted(set(list(DATA.glob('*.npy')) + list(DATA.glob('*.npz')) +
                       list(DATA.glob('*.json')) + list(out.glob('*.json')) + list(out.glob('*.joblib')) +
                       list(exp.cache.glob('*.npz')) + list(exp.cache.glob('*.joblib')) +
                       list(Path(__file__).parent.glob('*.py'))))
    files = [p for p in files if p.name != 'artifact_manifest.json']
    manifest = dict(signature=exp.sig, python=platform.python_version(),
                    packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','joblib','tensorly','threadpoolctl']},
                    files={str(p.relative_to(DATA.parents[2])):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
                    note='Post-run provenance snapshot. Verify fingerprints before replay; code/inputs/results must stay together.')
    (DATA / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2), flush=True)
    print(f'Report: {DATA / "EXPERIMENT_REPORT.md"}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
