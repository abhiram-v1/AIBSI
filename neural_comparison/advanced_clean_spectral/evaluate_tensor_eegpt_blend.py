"""Train a TT-SVM and blend with existing fine-tuned EEGPT predictions.

No EEGPT training occurs. Keep the historical 121/26/27 subject split: rank/SVM
selection is training-only CV; blend selection uses validation; test is scored
only after both choices are fixed. All candidate scores are persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import train_tensor_train_svm as tt
import evaluate_eegpt_svm_blend as spectral_blend

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'outputs' / 'tensor_eegpt_blend_2026-11-05'
EEGPT = HERE / 'outputs/deep_models/eegpt_attention'
CLASSES = ['AD', 'FTD', 'HC']


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fit_branch(tensors, labels, participants, fit, config):
    normalized, normalization = tt.site_normalize(tensors, participants, fit)
    projection = tt.TTProjection(rank=config['rank'], channel_rank=config['channel_rank'],
                                 spectral_rank=config['spectral_rank']).fit(normalized[fit])
    features = projection.transform(normalized)
    model = tt.classifier(config['C'], config['kernel']).fit(features[fit], labels[fit])
    assert np.array_equal(model.classes_, np.arange(3))
    return features, model, projection, normalization


def probability(scores, temperature):
    return softmax(np.asarray(scores, dtype=np.float64) / temperature, axis=1)


def eegpt_probability(p, temperature):
    # Match the historical implementation, including its clipping constant.
    return softmax(np.log(np.maximum(p, 1e-7)) / temperature, axis=1)


def combined(scores, eegpt_p, choice):
    weight = choice['svm_weight']
    return (weight * probability(scores, choice['svm_temperature'])
            + (1 - weight) * eegpt_probability(eegpt_p, choice['eegpt_temperature']))


def metrics(y, predictions):
    return tt.eeg.metrics(y, predictions)


def make_plots(out, results):
    names = list(results)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    positions = np.arange(len(names))
    for index, (key, label) in enumerate([('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'),
                                         ('balanced_accuracy', 'Balanced accuracy')]):
        values = [100 * results[name][key] for name in names]
        bars = ax.bar(positions + (index - 1) * .24, values, .24, label=label)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    ax.set_xticks(positions, ['Spectral SVM', 'EEGPT', 'Spectral SVM\n+ EEGPT',
                             'TT-SVM', 'TT-SVM + EEGPT\nfixed 50/50', 'TT-SVM + EEGPT\nvalidation tuned'])
    ax.set(ylabel='Score (%)', ylim=(0, 100), title='Historical 27-participant test set: tensor ensemble comparison')
    ax.legend(loc='upper center', ncols=3)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, ['Spectral SVM + EEGPT', 'TT-SVM', 'TT-SVM + EEGPT tuned']):
        matrix = np.asarray(results[name]['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=11)
        for i, j in itertools.product(range(3), repeat=2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted', ylabel='Actual', title=name)
    fig.savefig(out / 'confusion_matrices.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        raise RuntimeError('Completed output already exists; choose another --output to preserve it.')
    start = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        prepared = tt.PREPARED
        participants = load_json(prepared / 'manifest.json')['participants']
        raw_participants = load_json(HERE / 'outputs/eegpt/prepared/manifest.json')['participants']
        assert [p['subject_id'] for p in participants] == [p['subject_id'] for p in raw_participants]
        splits = tt.eeg.make_subject_splits(participants)
        train, val, test = [np.asarray(splits[k], dtype=int) for k in ('train', 'val', 'test')]
        assert (len(train), len(val), len(test)) == (121, 26, 27)
        assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        cached = np.load(EEGPT / 'probabilities.npz')
        attention = load_json(EEGPT / 'attention.json')
        for split, att_key in [('val', 'validation'), ('test', 'test')]:
            assert [r['subject'] for r in attention[att_key]] == splits[split], 'EEGPT subject order differs'
            p = cached[split]
            assert p.shape == (len(splits[split]), 3) and np.isfinite(p).all() and np.all(p >= 0)
            # These cached outputs came from float16 autocast softmax. Row sums
            # can differ from one by a few 1e-4; blend softmax renormalizes them.
            assert np.allclose(p.sum(axis=1), 1, atol=1e-3)
        windows = np.load(prepared / 'log_mel.npy', mmap_mode='r')
        owners = np.load(prepared / 'subject_index.npy')
        window_y = np.load(prepared / 'labels.npy')
        tensors, labels = tt.participant_tensors(windows, window_y, owners, participants)
        assert tensors.shape == (174, 19, 24, 4)
        assert np.array_equal(cached['val_true'], labels[val])
        assert np.array_equal(cached['test_true'], labels[test])
        np.save(out / 'participant_tensors.npy', tensors)
        dump(out / 'splits.json', {k: [{'index': i, 'id': participants[i]['subject_id'],
                                      'label': participants[i]['label']} for i in values]
                                  for k, values in splits.items()})
        rank_space = list(itertools.product(tt.RANKS, tt.CHANNEL_RANKS, tt.SPECTRAL_RANKS))
        configs = [dict(rank=r, channel_rank=c, spectral_rank=s, C=cost, kernel=k)
                   for (r, c, s), cost, k in itertools.product(rank_space, tt.SVM_C, tt.SVM_KERNELS)]
        protocol = {'task': 'AD / FTD / HC', 'classes': CLASSES, 'tensor_shape': list(tensors.shape),
                    'split_counts': {k: len(v) for k, v in splits.items()},
                    'tensor': 'Participant electrode x mel band x mean/SD/median/IQR of windows and STFT frames',
                    'branch_search': '5-fold diagnosis/source-stratified CV on the 121 training participants only',
                    'fold_fits': 'Refit site normalization, common TT basis and SVM scaler on each training fold',
                    'svm_candidates': len(configs), 'rank_grid': rank_space, 'C': tt.SVM_C, 'kernels': tt.SVM_KERNELS,
                    'gamma': 'scale', 'class_weight': 'balanced', 'seed': tt.eeg.SEED,
                    'blend_search': '420 combinations on 26 validation participants, selected by macro F1 then accuracy then closeness to equal weight',
                    'final_refit': 'TT and SVM refit on 147 development participants after selecting all settings; EEGPT stays unchanged',
                    'evaluation': 'Previously examined 27-participant test set; exploratory, not a new independent estimate or full nested EEGPT CV',
                    'eegpt_retrained': False, 'threads': args.threads}
        dump(out / 'protocol.json', protocol)
        print(f'Inputs aligned. Tensor {tensors.shape}; {len(configs)} rank/SVM candidates, 5 inner folds.', flush=True)
        predictions = np.empty((len(configs), len(train)), dtype=np.int64)
        strata = np.asarray([f'{participants[i]["source"]}:{labels[i]}' for i in train])
        fold_log = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=tt.eeg.SEED)
        for fold, (fit_local, valid_local) in enumerate(cv.split(train, strata)):
            fit, valid = train[fit_local], train[valid_local]
            normalized, _ = tt.site_normalize(tensors, participants, fit)
            features = {}
            for r, c, s in rank_space:
                projection = tt.TTProjection(rank=r, channel_rank=c, spectral_rank=s).fit(normalized[fit])
                features[(r, c, s)] = projection.transform(normalized)
            for ci, config in enumerate(configs):
                x = features[(config['rank'], config['channel_rank'], config['spectral_rank'])]
                model = tt.classifier(config['C'], config['kernel']).fit(x[fit], labels[fit])
                predictions[ci, valid_local] = model.predict(x[valid])
            fold_log.append({'fold': fold, 'fit': fit.tolist(), 'validation': valid.tolist()})
            print(f'Inner CV {fold+1}/5 complete ({20*(fold+1)}%); elapsed {time.perf_counter()-start:.1f}s', flush=True)
        candidates = []
        for ci, config in enumerate(configs):
            m = metrics(labels[train], predictions[ci])
            candidates.append({'candidate': ci, 'config': config, 'inner_metrics': m})
        candidates.sort(key=lambda r: (r['inner_metrics']['macro_f1'], r['inner_metrics']['balanced_accuracy'],
                                      r['inner_metrics']['accuracy'], -r['config']['rank'],
                                      -r['config']['spectral_rank'], -r['config']['channel_rank'],
                                      -r['config']['C'], r['config']['kernel'] == 'linear'), reverse=True)
        selected = candidates[0]
        dump(out / 'svm_search.json', {'selected': selected, 'candidates': candidates, 'folds': fold_log})
        np.savez_compressed(out / 'inner_oof_predictions.npz', predictions=predictions, subjects=train,
                            labels=labels[train], candidate_ids=np.arange(len(configs)))
        print('Selected training-CV configuration:', selected['config'], flush=True)
        x, model, _, _ = fit_branch(tensors, labels, participants, train, selected['config'])
        val_scores = model.decision_function(x[val])
        blend_candidates = []
        for st, et, w in itertools.product([.5, .75, 1., 1.5, 2.], [.75, 1., 1.25, 1.5], np.linspace(0, 1, 21)):
            choice = {'svm_temperature': st, 'eegpt_temperature': et, 'svm_weight': float(w)}
            m = metrics(labels[val], np.argmax(combined(val_scores, cached['val'], choice), axis=1))
            blend_candidates.append({**choice, 'val_macro_f1': m['macro_f1'],
                                     'val_balanced_accuracy': m['balanced_accuracy'], 'val_accuracy': m['accuracy']})
        blend_candidates.sort(key=lambda r: (r['val_macro_f1'], r['val_accuracy'], -abs(r['svm_weight']-.5)), reverse=True)
        choice = blend_candidates[0]
        fixed = {'svm_temperature': .5, 'eegpt_temperature': .75, 'svm_weight': .5}
        dump(out / 'blend_search.json', {'selected': choice, 'fixed_historical_blend_control': fixed,
                                        'validation_candidates': blend_candidates})
        print('Blend selected on validation:', choice, flush=True)
        # Both selections are now frozen. Only now construct test predictions.
        dev = np.concatenate([train, val])
        x, model, projection, normalization = fit_branch(tensors, labels, participants, dev, selected['config'])
        test_scores = model.decision_function(x[test])
        tuned_p = combined(test_scores, cached['test'], choice)
        fixed_p = combined(test_scores, cached['test'], fixed)
        spectral = load_json(HERE / 'outputs/results/site_normalized_metrics.json')
        old_blend = load_json(EEGPT / 'svm_blend_metrics.json')
        plain_features, _ = tt.eeg.subject_features(windows, window_y, owners, participants)
        plain_scores = spectral_blend.svm_scores(plain_features.reshape(174, -1), labels, participants,
                                                dev, test, spectral['selected_on_validation']['config'])
        old_probability = combined(plain_scores, cached['test'], old_blend['selection'])
        reproduced = metrics(labels[test], old_probability.argmax(axis=1))
        for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            assert np.isclose(reproduced[metric], old_blend['test'][metric]), 'Historical baseline did not reproduce'
        results = {'Spectral SVM': spectral['test'], 'EEGPT': metrics(labels[test], cached['test'].argmax(axis=1)),
                   'Spectral SVM + EEGPT': reproduced, 'TT-SVM': metrics(labels[test], model.predict(x[test])),
                   'TT-SVM + EEGPT fixed': metrics(labels[test], fixed_p.argmax(axis=1)),
                   'TT-SVM + EEGPT tuned': metrics(labels[test], tuned_p.argmax(axis=1))}
        joblib.dump({'normalization': normalization, 'decomposition': projection, 'classifier': model,
                     'blend': choice, 'classes': CLASSES, 'fit_participants': dev,
                     'participant_tensor_shape': [19, 24, 4], 'settings': selected['config']}, out / 'tt_svm_blend.joblib')
        np.savez_compressed(out / 'probabilities.npz', val_tuned=combined(val_scores, cached['val'], choice),
                            test_tuned=tuned_p, test_fixed=fixed_p, test_spectral_blend=old_probability,
                            test_svm_scores=test_scores, test_eegpt=cached['test'],
                            val_subjects=val, test_subjects=test, val_true=labels[val], test_true=labels[test])
        rows = [{'subject_id': participants[i]['subject_id'], 'subject_index': int(i), 'actual': CLASSES[labels[i]],
                 'probabilities': {c: float(p[j]) for j, c in enumerate(CLASSES)}} for i, p in zip(test, tuned_p)]
        dump(out / 'test_participant_probabilities.json', rows)
        sources = [Path(__file__), Path(tt.__file__), HERE / '../../tensor_pipeline/v2/models.py',
                   prepared / 'manifest.json', prepared / 'log_mel.npy', prepared / 'labels.npy',
                   prepared / 'subject_index.npy', HERE / 'outputs/eegpt/prepared/manifest.json',
                   EEGPT / 'probabilities.npz', EEGPT / 'attention.json', EEGPT / 'best_model.pt',
                   HERE / 'outputs/results/site_normalized_metrics.json', EEGPT / 'svm_blend_metrics.json']
        dump(out / 'source_hashes.json', {str(p.resolve()): digest(p) for p in sources})
        payload = {'protocol': protocol, 'selected_svm': selected, 'selected_blend': choice,
                   'validation_tensor_svm': metrics(labels[val], np.argmax(val_scores, axis=1)),
                   'test': results, 'runtime_seconds': time.perf_counter() - start,
                   'tensor_centered_reconstruction_error': projection.relative_centered_reconstruction_error_,
                   'verification': {'participant_ids_and_labels_aligned': True, 'classes_aligned': True,
                                    'test_not_used_for_selection': True, 'historical_blend_reproduced': True,
                                    'all_svm_and_blend_candidates_saved': True, 'eegpt_retrained': False}}
        dump(out / 'metrics.json', payload)
        make_plots(out, results)
        lines = ['# Tensor Train SVM + fine-tuned EEGPT: fixed-split experiment', '',
                 'This run adds the previously unverified TT-SVM + EEGPT ensemble. EEGPT was reused, not retrained. All models predict AD/FTD/HC per participant.', '',
                 'Pipeline: saved log-mel windows → participant tensor (174 × 19 × 24 × 4) → training-only site normalization → shared Tensor Train projection → scaled balanced SVM → temperature-adjusted blend with raw-EEG EEGPT outputs.', '',
                 'The 121/26/27 participant split is unchanged. Rank/SVM choice used 5-fold CV within the 121 training participants. Blend weights/temperatures used only the 26 validation participants. The final TT-SVM was refitted on 147 development participants before test evaluation. This differs from the older standalone TT search, which selected settings by CV over all 147 development participants.', '',
                 'The test set was examined during previous project experiments. These are exploratory fixed-split results; this reuse of a fine-tuned EEGPT does not constitute a new nested cross-validation of the ensemble.', '',
                 f'SVM settings: `{json.dumps(selected["config"])}`.', '',
                 f'Validation-selected blend: `{json.dumps(choice)}`.', '',
                 '| Model | Test accuracy (%) | Test macro F1 (%) | Test balanced accuracy (%) | FTD recall (%) |',
                 '|---|---:|---:|---:|---:|']
        for name, m in results.items():
            lines.append(f'| {name} | {100*m["accuracy"]:.2f} | {100*m["macro_f1"]:.2f} | {100*m["balanced_accuracy"]:.2f} | {100*m["per_class"]["FTD"]["recall"]:.2f} |')
        lines += ['', 'The fixed 50/50 control uses SVM temperature 0.5 and EEGPT temperature 0.75, matching the old blend. It was specified before testing. The tuned blend may select zero weight for a branch; that would mean it selected a single-model endpoint rather than evidence for complementarity.', '',
                  'All 288 SVM candidates (1,440 inner SVM fits) and all 420 blend combinations were saved. Rankings used macro F1. Temperature-adjusted SVM scores are softmax scores, not clinically calibrated risk estimates.', '',
                  '![Held-out comparison](comparison.png)', '', '![Confusion matrices](confusion_matrices.png)', '',
                  'See `svm_search.json`, `blend_search.json`, `metrics.json`, `source_hashes.json`, `splits.json`, `tt_svm_blend.joblib`, `probabilities.npz` and `test_participant_probabilities.json` for complete reproducible records.', '']
        (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        print('\nTEST RESULTS (settings frozen before scoring):', flush=True)
        print(json.dumps({name: {k: m[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy')} for name, m in results.items()}, indent=2), flush=True)
        print('\nTuned ensemble class probabilities per test participant:', flush=True)
        for row in rows:
            print(row['subject_id'], ' '.join(f'{c}={100*p:.2f}%' for c, p in row['probabilities'].items()), flush=True)
        print(f'Completed in {time.perf_counter()-start:.1f}s. Outputs: {out}', flush=True)


if __name__ == '__main__':
    main()
