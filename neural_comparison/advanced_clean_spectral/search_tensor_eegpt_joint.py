"""Jointly select TT/SVM and EEGPT blend settings using historical validation.

EEGPT is never retrained. Training-only inner CV supplies a tie-break/stability
score; the final combination is chosen by validation ENSEMBLE F1, rather than
the standalone SVM F1. The old test split is evaluated only after selection.
"""
from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import evaluate_tensor_eegpt_blend as common

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/tensor_eegpt_joint_search_2026-11-05'
SVMT = [.5, .75, 1., 1.5, 2.]
EEGPTT = [.75, 1., 1.25, 1.5]
WEIGHTS = np.linspace(0, 1, 21)
BLENDS = [dict(svm_temperature=st, eegpt_temperature=et, svm_weight=float(w))
          for st, et, w in itertools.product(SVMT, EEGPTT, WEIGHTS)]


def classifier(config):
    return Pipeline([('scale', StandardScaler()),
                     ('svm', SVC(C=config['C'], kernel=config['kernel'], gamma=config['gamma'],
                                 class_weight='balanced', decision_function_shape='ovr',
                                 random_state=common.tt.eeg.SEED))])


def tensor_features(tensors, participants, fit, ranks):
    normalized, normalization = common.tt.site_normalize(tensors, participants, fit)
    cache = {}
    for rank, channels, spectral in ranks:
        projection = common.tt.TTProjection(rank=rank, channel_rank=channels, spectral_rank=spectral).fit(normalized[fit])
        cache[(rank, channels, spectral)] = projection.transform(normalized)
    return cache


def vector_scores(prediction, true):
    """Metrics for (candidate, participant) predictions, in fixed 3-class order."""
    counts = np.asarray([(true == c).sum() for c in range(3)])
    assert np.all(counts > 0)
    tp = np.stack([np.sum((prediction == c) & (true[None, :] == c), axis=1) for c in range(3)], axis=1)
    predicted = np.stack([np.sum(prediction == c, axis=1) for c in range(3)], axis=1)
    f1 = np.mean(2 * tp / np.maximum(counts[None, :] + predicted, 1), axis=1)
    ba = np.mean(tp / counts[None, :], axis=1)
    accuracy = np.mean(prediction == true[None, :], axis=1)
    return f1, ba, accuracy


def choose_key(f1, ba, acc, inner_f1, config, blend):
    # Predeclared selection, before obtaining any new test predictions.
    return (float(f1), float(ba), float(acc), float(inner_f1),
            -abs(blend['svm_weight']-.5), -config['rank'],
            -config['spectral_rank'], -config['channel_rank'],
            config['kernel'] == 'linear', -config['C'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'metrics.json').exists():
        raise RuntimeError('Completed experiment exists; preserve it.')
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        prepared = common.tt.PREPARED
        participants = common.load_json(prepared/'manifest.json')['participants']
        raw_participants = common.load_json(HERE/'outputs/eegpt/prepared/manifest.json')['participants']
        assert [p['subject_id'] for p in participants] == [p['subject_id'] for p in raw_participants]
        splits = common.tt.eeg.make_subject_splits(participants)
        train, val, test = [np.asarray(splits[k], dtype=int) for k in ('train', 'val', 'test')]
        assert tuple(map(len, (train, val, test))) == (121, 26, 27)
        assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
        attention = common.load_json(common.EEGPT/'attention.json')
        eegpt = np.load(common.EEGPT/'probabilities.npz')
        for name, att in [('val', 'validation'), ('test', 'test')]:
            assert [r['subject'] for r in attention[att]] == splits[name]
        windows = np.load(prepared/'log_mel.npy', mmap_mode='r')
        window_y = np.load(prepared/'labels.npy')
        owners = np.load(prepared/'subject_index.npy')
        tensors, labels = common.tt.participant_tensors(windows, window_y, owners, participants)
        assert tensors.shape == (174, 19, 24, 4)
        assert np.array_equal(eegpt['val_true'], labels[val])
        assert np.array_equal(eegpt['test_true'], labels[test])
        ranks = list(itertools.product(common.tt.RANKS, common.tt.CHANNEL_RANKS, common.tt.SPECTRAL_RANKS))
        configs = []
        for (r, ch, sp), cost in itertools.product(ranks, common.tt.SVM_C):
            for kernel, gamma in [('linear', 'scale'), ('rbf', 'scale'), ('rbf', .01), ('rbf', .1), ('rbf', 1.)]:
                configs.append(dict(rank=r, channel_rank=ch, spectral_rank=sp, C=cost, kernel=kernel, gamma=gamma))
        protocol = {'classes': common.CLASSES, 'tensor_shape': list(tensors.shape),
                    'participant_split': splits, 'svm_candidates': len(configs),
                    'blend_candidates_per_svm': len(BLENDS), 'joint_candidates': len(configs)*len(BLENDS),
                    'rank_grid': ranks, 'C': common.tt.SVM_C, 'kernels': ['linear', 'rbf'],
                    'rbf_gamma': ['scale', .01, .1, 1.], 'class_weight': 'balanced',
                    'svm_temperatures': SVMT, 'eegpt_temperatures': EEGPTT, 'svm_weights': WEIGHTS.tolist(),
                    'selection': 'Validation ensemble macro F1, then balanced accuracy, accuracy, training-only inner-CV standalone F1, then weight closeness to .5 and simpler TT/SVM settings.',
                    'inner': '5-fold source/diagnosis-stratified SVM CV on 121 training participants; refit normalization/TT/scaler each fold. These scores are a tie-break, not an ensemble CV estimate.',
                    'validation': 'Joint rank/SVM/blend selection on the 26 validation participants. All tensor/SVM candidates fitted on 121 training participants.',
                    'final': 'Refit only selected tensor/SVM on 147 development participants, reuse original EEGPT, then evaluate 27-person test.',
                    'eegpt_retrained': False,
                    'limitations': 'Large search on 26 validation participants can overfit validation. Historical test split inspected previously; this is exploratory and not a full nested ensemble evaluation.'}
        common.dump(OUT/'protocol.json', protocol)
        common.dump(OUT/'svm_configurations.json', configs)
        print(f'Joint search: {len(configs)} TT/SVM settings x {len(BLENDS)} blends = {len(configs)*len(BLENDS):,} combinations.', flush=True)
        strata = np.asarray([f'{participants[i]["source"]}:{labels[i]}' for i in train])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=common.tt.eeg.SEED)
        oof = np.empty((len(configs), len(train)), dtype=np.int64)
        folds = []
        for fi, (fit_local, valid_local) in enumerate(cv.split(train, strata)):
            fit, valid = train[fit_local], train[valid_local]
            features = tensor_features(tensors, participants, fit, ranks)
            for ci, config in enumerate(configs):
                x = features[(config['rank'], config['channel_rank'], config['spectral_rank'])]
                m = classifier(config).fit(x[fit], labels[fit])
                oof[ci, valid_local] = m.predict(x[valid])
            folds.append(dict(fold=fi, fit=fit.tolist(), validation=valid.tolist()))
            print(f'Training-only CV {fi+1}/5 complete; {time.perf_counter()-start:.1f}s', flush=True)
        inner_f1, inner_ba, inner_acc = vector_scores(oof, labels[train])
        common.dump(OUT/'inner_cv.json', {'folds': folds, 'candidates': [
            {'candidate': i, 'config': c, 'inner_macro_f1': float(inner_f1[i]),
             'inner_balanced_accuracy': float(inner_ba[i]), 'inner_accuracy': float(inner_acc[i])}
            for i, c in enumerate(configs)]})
        np.savez_compressed(OUT/'inner_oof.npz', subjects=train, true=labels[train], predictions=oof)
        features = tensor_features(tensors, participants, train, ranks)
        eegpt_p = np.stack([common.eegpt_probability(eegpt['val'], t) for t in EEGPTT])
        winning = None
        svm_rows = []
        parameter_best = {}
        total = 0
        winning_validation_scores = None
        with gzip.open(OUT/'all_joint_candidates.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as output:
            for ci, config in enumerate(configs):
                x = features[(config['rank'], config['channel_rank'], config['spectral_rank'])]
                model = classifier(config).fit(x[train], labels[train])
                scores = model.decision_function(x[val])
                assert np.array_equal(model.classes_, np.arange(3))
                svm_p = np.stack([common.probability(scores, t) for t in SVMT])
                probabilities = (svm_p[:, None, None, :, :]*WEIGHTS[None, None, :, None, None]
                                 + eegpt_p[None, :, None, :, :]*(1-WEIGHTS)[None, None, :, None, None])
                predictions = probabilities.reshape(-1, len(val), 3).argmax(axis=2)
                f1, ba, acc = vector_scores(predictions, labels[val])
                # Check vectorized metrics against the established sklearn path.
                if ci == 0:
                    for bi in [0, 209, 419]:
                        reference = common.metrics(labels[val], predictions[bi])
                        assert np.isclose(f1[bi], reference['macro_f1'])
                        assert np.isclose(ba[bi], reference['balanced_accuracy'])
                        assert np.isclose(acc[bi], reference['accuracy'])
                local = None
                for bi, blend in enumerate(BLENDS):
                    row = {'svm_candidate': ci, 'blend_candidate': bi, 'config': config, **blend,
                           'val_macro_f1': float(f1[bi]), 'val_balanced_accuracy': float(ba[bi]),
                           'val_accuracy': float(acc[bi]), 'inner_svm_macro_f1': float(inner_f1[ci])}
                    output.write(json.dumps(row, separators=(',', ':'))+'\n')
                    total += 1
                    ranking = choose_key(f1[bi], ba[bi], acc[bi], inner_f1[ci], config, blend)
                    if local is None or ranking > local[0]:
                        local = (ranking, row)
                    if winning is None or ranking > winning[0]:
                        winning = (ranking, row)
                        winning_validation_scores = scores.copy()
                    for param, value in {**config, **blend}.items():
                        if param == 'gamma' and config['kernel'] != 'rbf':
                            continue
                        group = (param, json.dumps(value))
                        old = parameter_best.get(group)
                        if old is None or ranking > old[0]:
                            parameter_best[group] = (ranking, row)
                svm_rows.append({'candidate': ci, 'config': config,
                                 'inner_macro_f1': float(inner_f1[ci]), 'inner_balanced_accuracy': float(inner_ba[ci]),
                                 'inner_accuracy': float(inner_acc[ci]), 'best_validation_blend': local[1],
                                 'standalone_validation': common.metrics(labels[val], model.predict(x[val]))})
                if (ci+1) % 60 == 0:
                    print(f'Joint search {ci+1}/{len(configs)} SVMs ({100*(ci+1)/len(configs):.1f}%), {total:,} blends checked; best validation F1={winning[1]["val_macro_f1"]:.4f}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
        assert total == protocol['joint_candidates']
        selected = winning[1]
        common.dump(OUT/'selection.json', selected)
        common.dump(OUT/'svm_search.json', {'selected': selected, 'candidates': svm_rows})
        common.dump(OUT/'parameter_best_results.json', [dict(parameter=p, value=json.loads(v), winner=r[1]) for (p,v),r in parameter_best.items()])
        blend_choices = []
        for bi, blend in enumerate(BLENDS):
            p = common.combined(winning_validation_scores, eegpt['val'], blend)
            m = common.metrics(labels[val], p.argmax(axis=1))
            blend_choices.append(dict(blend_candidate=bi, **blend, val_macro_f1=m['macro_f1'],
                                      val_accuracy=m['accuracy'], val_balanced_accuracy=m['balanced_accuracy']))
        common.dump(OUT/'blend_search.json', {'selected': {k: selected[k] for k in BLENDS[0]}, 'validation_candidates': blend_choices})
        print('Jointly selected configuration:', json.dumps(selected), flush=True)
        # Selection is persisted before the test scores/probabilities are formed.
        dev = np.concatenate([train, val])
        normalized, normalization = common.tt.site_normalize(tensors, participants, dev)
        config = selected['config']
        projection = common.tt.TTProjection(rank=config['rank'], channel_rank=config['channel_rank'],
                                            spectral_rank=config['spectral_rank']).fit(normalized[dev])
        x = projection.transform(normalized)
        model = classifier(config).fit(x[dev], labels[dev])
        assert np.array_equal(model.classes_, np.arange(3))
        test_scores = model.decision_function(x[test])
        p = common.combined(test_scores, eegpt['test'], selected)
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
        test_metrics = common.metrics(labels[test], p.argmax(axis=1))
        standalone = common.metrics(labels[test], model.predict(x[test]))
        old = common.load_json(common.DEFAULT_OUT/'metrics.json')
        historical = common.load_json(common.DEFAULT_OUT/'historical_tt_control/metrics.json')
        results = {'No-tensor SVM + EEGPT': old['test']['Spectral SVM + EEGPT'],
                   'Previous sequential TT search + EEGPT': old['test']['TT-SVM + EEGPT tuned'],
                   'Historical TT settings + EEGPT': historical['test']['Historical TT-SVM + EEGPT tuned'],
                   'Joint-selected TT-SVM alone': standalone, 'Joint-selected TT-SVM + EEGPT': test_metrics}
        joblib.dump(dict(normalization=normalization, decomposition=projection, classifier=model,
                         blend={k: selected[k] for k in BLENDS[0]}, settings=config,
                         classes=common.CLASSES, fit_participants=dev, participant_tensor_shape=[19,24,4]), OUT/'model.joblib')
        np.savez_compressed(OUT/'probabilities.npz', val_probabilities=common.combined(winning_validation_scores, eegpt['val'], selected),
                            test_probabilities=p, test_subjects=test, val_subjects=val,
                            val_true=labels[val], test_true=labels[test], test_svm_scores=test_scores)
        rows = [dict(subject_id=participants[i]['subject_id'], subject_index=int(i), actual=common.CLASSES[labels[i]],
                     probabilities={c: float(row[j]) for j,c in enumerate(common.CLASSES)}) for i,row in zip(test,p)]
        common.dump(OUT/'test_participant_probabilities.json', rows)
        common.dump(OUT/'metrics.json', dict(protocol=protocol, selected=selected, test=results,
                                            runtime_seconds=time.perf_counter()-start,
                                            verification=dict(vector_metrics_match_sklearn=True, eegpt_ids_and_labels_aligned=True,
                                                              test_excluded_from_selection=True, all_joint_trials_saved=True)))
        paths = [Path(__file__), Path(common.__file__), common.tt.PREPARED/'manifest.json',
                 common.tt.PREPARED/'log_mel.npy', common.tt.PREPARED/'subject_index.npy', common.tt.PREPARED/'labels.npy',
                 HERE/'outputs/eegpt/prepared/manifest.json', common.EEGPT/'probabilities.npz', common.EEGPT/'attention.json',
                 common.EEGPT/'best_model.pt', common.DEFAULT_OUT/'metrics.json',
                 common.DEFAULT_OUT/'historical_tt_control/metrics.json']
        common.dump(OUT/'source_hashes.json', {str(path.resolve()): common.digest(path) for path in paths})
        # Parameter/value rows contain all metrics from one F1-winning combination.
        lines = ['# Joint tensor-SVM + EEGPT parameter search', '',
                 'The ensemble, rather than the standalone SVM, is the selection target. EEGPT was fixed. TT/SVM candidates were fitted on121 training participants; all joint selections use26 validation participants. Five-fold training-only SVM CV is a tie-break/stability diagnostic. Final TT/SVM refitting uses147 development participants; test results use the historical27-person split.', '',
                 f'**{len(configs)} TT/SVM candidates × {len(BLENDS)} blend choices = {total:,} joint validation comparisons.**', '',
                 f'Selected settings: `{json.dumps(selected)}`.', '',
                 '## Actual test results', '',
                 '| Model | Accuracy (%) | Macro F1 (%) | Balanced accuracy (%) | FTD recall (%) |',
                 '|---|---:|---:|---:|---:|']
        for name,m in results.items():
            lines.append(f'| {name} | {100*m["accuracy"]:.2f} | {100*m["macro_f1"]:.2f} | {100*m["balanced_accuracy"]:.2f} | {100*m["per_class"]["FTD"]["recall"]:.2f} |')
        lines += ['', '**Validation maxima below are selection scores, not test results or isolated parameter effects.** Other settings vary between rows; the best ensemble F1 determines the whole row. Gamma applies only to RBF kernels. A larger search on26 validation participants can overfit validation. The previously examined test split remains exploratory.', '',
                  '## Best saved ensemble result for each parameter value', '',
                  '| Parameter | Value | Validation macro F1 (%) | Validation BA (%) | Validation accuracy (%) | SVM candidate | Blend candidate |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for (param,value),(_,r) in sorted(parameter_best.items()):
            lines.append(f'| {param} | {value} | {100*r["val_macro_f1"]:.2f} | {100*r["val_balanced_accuracy"]:.2f} | {100*r["val_accuracy"]:.2f} | {r["svm_candidate"]} | {r["blend_candidate"]} |')
        lines += ['', '![Model comparison](comparison.png)', '', '![Confusion matrices](confusion_matrices.png)', '',
                  'All joint combinations and metrics are in `all_joint_candidates.jsonl.gz`; `svm_configurations.json` defines candidate IDs. Each SVM’s best blend is in `svm_search.json`; all420 blends for the selected SVM are in `blend_search.json`. Source hashes, model, predictions, probabilities and inner-CV diagnostics are saved alongside this report.', '']
        (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        fig,ax = plt.subplots(figsize=(12,5.5),constrained_layout=True)
        for k,(metric,label,color) in enumerate([('accuracy','Accuracy','#3979ae'),('macro_f1','Macro F1','#168d89'),('balanced_accuracy','Balanced accuracy','#dc8533')]):
            bars=ax.bar(np.arange(len(results))+(k-1)*.25,[100*m[metric] for m in results.values()],.25,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
        ax.set_xticks(range(len(results)), ['No tensor\nSVM + EEGPT','Previous sequential\nTT search + EEGPT','Historical TT\n+ EEGPT','Joint-selected\nTT-SVM alone','Joint-selected\nTT-SVM + EEGPT'])
        ax.set(ylabel='Score (%)',ylim=(0,100),title='Joint tensor/SVM/blend search: historical 27-person test set')
        ax.legend(loc='upper center',ncols=3);ax.grid(axis='y',alpha=.2)
        fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
        for ax,name in zip(axes,['No-tensor SVM + EEGPT','Joint-selected TT-SVM + EEGPT']):
            matrix=np.array(results[name]['confusion_matrix']);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=11)
            for i,j in itertools.product(range(3),repeat=2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
            ax.set(xticks=range(3),yticks=range(3),xticklabels=['AD','FTD','CN'],yticklabels=['AD','FTD','CN'],xlabel='Predicted',ylabel='Actual',title=name)
        fig.savefig(OUT/'confusion_matrices.png',dpi=180);plt.close(fig)
        print('\nTEST RESULTS (not used for selection):', flush=True)
        print(json.dumps({name:{k:m[k] for k in ['accuracy','macro_f1','balanced_accuracy']} for name,m in results.items()},indent=2),flush=True)
        print('\nPer-participant probabilities:',flush=True)
        for r in rows:print(r['subject_id'],' '.join(f'{c}={100*v:.2f}%' for c,v in r['probabilities'].items()),flush=True)
        print(f'Completed in {time.perf_counter()-start:.1f}s; output {OUT}',flush=True)


if __name__=='__main__':
    main()
