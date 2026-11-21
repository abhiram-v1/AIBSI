"""Follow-up control using the historical TT/SVM settings, not retuned on test."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import evaluate_tensor_eegpt_blend as run

HERE = Path(__file__).resolve().parent
PARENT = run.DEFAULT_OUT
OUT = PARENT / 'historical_tt_control'


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'metrics.json').exists():
        raise RuntimeError('Completed historical control already exists.')
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        prior_path = HERE / 'outputs/tensor_train_svm/metrics.json'
        prior = run.load_json(prior_path)
        config = {k: prior['selected'][k] for k in ('rank', 'channel_rank', 'spectral_rank', 'C', 'kernel')}
        participants = run.load_json(run.tt.PREPARED / 'manifest.json')['participants']
        splits = run.tt.eeg.make_subject_splits(participants)
        train, val, test = [np.asarray(splits[k], dtype=int) for k in ('train', 'val', 'test')]
        tensors = np.load(PARENT / 'participant_tensors.npy')
        windows_y = np.load(run.tt.PREPARED / 'labels.npy')
        owners = np.load(run.tt.PREPARED / 'subject_index.npy')
        labels = np.asarray([windows_y[np.flatnonzero(owners == i)[0]] for i in range(len(participants))])
        eegpt = np.load(run.EEGPT / 'probabilities.npz')
        attention = run.load_json(run.EEGPT / 'attention.json')
        for split, name in [('val', 'validation'), ('test', 'test')]:
            assert np.array_equal(eegpt[split + '_true'], labels[splits[split]])
            assert [r['subject'] for r in attention[name]] == splits[split]
        protocol = {'config': config, 'source_config': str(prior_path.resolve()),
                    'source_sha256': run.digest(prior_path),
                    'timing': 'Historical-configuration control added after the fresh training-CV-selected TT ensemble was evaluated. Both results are retained, not selected by test score.',
                    'selection_history': 'TT/SVM settings inherited from the earlier 147-person development-CV experiment. They are fixed for this control, not a new training-only hyperparameter search.',
                    'validation': 'Fit TT/SVM on 121 participants, tune 420 blend settings on 26 validation participants.',
                    'test': 'Refit TT/SVM on 147 development participants; evaluate the original 27-person test set.',
                    'eegpt_retrained': False, 'caveat': 'Exploratory historical split; validation also influenced the inherited TT/SVM settings in the earlier experiment.'}
        run.dump(OUT / 'protocol.json', protocol)
        print('Historical fixed TT/SVM configuration:', config, flush=True)
        x, model, _, _ = run.fit_branch(tensors, labels, participants, train, config)
        val_scores = model.decision_function(x[val])
        candidates = []
        for st, et, w in itertools.product([.5, .75, 1., 1.5, 2.], [.75, 1., 1.25, 1.5], np.linspace(0, 1, 21)):
            choice = {'svm_temperature': st, 'eegpt_temperature': et, 'svm_weight': float(w)}
            m = run.metrics(labels[val], run.combined(val_scores, eegpt['val'], choice).argmax(axis=1))
            candidates.append({**choice, 'val_macro_f1': m['macro_f1'], 'val_accuracy': m['accuracy'],
                               'val_balanced_accuracy': m['balanced_accuracy']})
        candidates.sort(key=lambda r: (r['val_macro_f1'], r['val_accuracy'], -abs(r['svm_weight']-.5)), reverse=True)
        choice = candidates[0]
        run.dump(OUT / 'blend_search.json', {'selected': choice, 'validation_candidates': candidates})
        print('Historical TT blend selected on validation:', choice, flush=True)
        dev = np.concatenate([train, val])
        x, model, projection, normalization = run.fit_branch(tensors, labels, participants, dev, config)
        native_prediction = model.predict(x[test])
        standalone = run.metrics(labels[test], native_prediction)
        for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            assert np.isclose(standalone[metric], prior['test'][metric]), 'Historical TT did not reproduce'
        test_scores = model.decision_function(x[test])
        tuned = run.combined(test_scores, eegpt['test'], choice)
        fixed_choice = {'svm_temperature': .5, 'eegpt_temperature': .75, 'svm_weight': .5}
        fixed = run.combined(test_scores, eegpt['test'], fixed_choice)
        results = {'Historical TT-SVM': standalone,
                   'Historical TT-SVM + EEGPT fixed': run.metrics(labels[test], fixed.argmax(axis=1)),
                   'Historical TT-SVM + EEGPT tuned': run.metrics(labels[test], tuned.argmax(axis=1))}
        run.dump(OUT / 'metrics.json', {'protocol': protocol, 'selected_blend': choice, 'test': results,
                                      'historical_standalone_reproduced': True, 'runtime_seconds': time.perf_counter()-start})
        rows = [{'subject_id': participants[i]['subject_id'], 'actual': run.CLASSES[labels[i]],
                 'probabilities': {c: float(p[j]) for j, c in enumerate(run.CLASSES)}} for i, p in zip(test, tuned)]
        run.dump(OUT / 'test_participant_probabilities.json', rows)
        np.savez_compressed(OUT / 'probabilities.npz', val_tuned=run.combined(val_scores, eegpt['val'], choice),
                            test_tuned=tuned, test_fixed=fixed, test_svm_scores=test_scores,
                            val_subjects=val, test_subjects=test, val_true=labels[val], test_true=labels[test])
        joblib.dump({'normalization': normalization, 'decomposition': projection, 'classifier': model,
                     'blend': choice, 'settings': config, 'classes': run.CLASSES,
                     'fit_participants': dev, 'participant_tensor_shape': [19, 24, 4]}, OUT / 'tt_svm_blend.joblib')
        parent = run.load_json(PARENT / 'metrics.json')
        lines = ['# Tensor-SVM + EEGPT: historical-settings control', '',
                 'EEGPT weights and cached predictions were reused. TT/SVM parameters were inherited unchanged from the old Tensor Train experiment. This control was added after the fresh training-only CV run; both outcomes are reported. Blend tuning used 26 validation participants, followed by TT/SVM refitting on 147 development participants and evaluation on the historical 27-person test set.', '',
                 f'Fixed historical tensor/SVM settings: `{json.dumps(config)}`.', '',
                 f'Validation-selected blend: `{json.dumps(choice)}`.', '',
                 '| Model | Test accuracy (%) | Test macro F1 (%) | Test balanced accuracy (%) | FTD recall (%) |',
                 '|---|---:|---:|---:|---:|']
        comparison = {**{k: parent['test'][k] for k in ['Spectral SVM + EEGPT', 'TT-SVM + EEGPT tuned']}, **results}
        for name, m in comparison.items():
            lines.append(f'| {name} | {100*m["accuracy"]:.2f} | {100*m["macro_f1"]:.2f} | {100*m["balanced_accuracy"]:.2f} | {100*m["per_class"]["FTD"]["recall"]:.2f} |')
        lines += ['', 'The historical standalone TT scores were reproduced exactly. The fresh run selects TT/SVM settings only within 121 training participants; the historical control inherits settings selected by the previous 147-person development CV. Their selection protocols differ. Neither is a new independent estimate of generalization.', '',
                  'All 420 blend candidates, model parameters and per-participant probabilities are saved here. The parent directory contains the fresh 288-configuration TT/SVM search and the no-tensor baseline reproduction.', '']
        (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        run.dump(OUT / 'source_hashes.json', {str(p.resolve()): run.digest(p) for p in [Path(__file__), Path(run.__file__), prior_path, run.EEGPT/'probabilities.npz', PARENT/'participant_tensors.npy']})
        print(json.dumps({'selected_blend': choice, 'test': results}, indent=2), flush=True)
        for row in rows:
            print(row['subject_id'], ' '.join(f'{c}={100*p:.2f}%' for c, p in row['probabilities'].items()), flush=True)


if __name__ == '__main__':
    main()
