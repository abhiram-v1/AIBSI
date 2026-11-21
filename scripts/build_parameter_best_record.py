"""Summarize existing searches by parameter value, choosing each row by F1."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import shutil
from statistics import mean

from build_experiment_search_record import ROOT, OUT, table, link, read, sha, literal_constants

MAIN = ROOT / 'EXPERIMENT_SEARCH_RECORD.md'
DETAIL = ROOT / 'EXPERIMENT_SEARCH_DETAILS.md'
WINNERS = OUT / 'parameter_winning_configurations.md'
LABELS = {
    'sampling_hz': 'Sampling frequency (Hz)', 'segment_seconds': 'Main window (s)',
    'stft_seconds': 'STFT window (s)', 'hop_seconds': 'STFT hop (s)',
    'mel_bins': 'Mel bands', 'summary': 'Feature statistics',
    'site_adjustment': 'Site adjustment', 'normalization': 'Normalization',
    'scaler': 'Scaler', 'pca': 'PCA components', 'C': 'SVM C',
    'kernel': 'SVM kernel function', 'gamma': 'RBF gamma',
    'healthy_weight': 'Healthy-control/class weight', 'weight': 'Class weight',
    'threshold': 'Decision-score threshold', 'ftd_boost': 'FTD class-weight multiplier',
    'rank': 'Tensor rank', 'channel_rank': 'Channel rank', 'spectral_rank': 'Spectral rank',
    'ranks': 'Tucker ranks', 'covariance': 'Covariance normalization',
    'band_set': 'Frequency-band set', 'representation': 'Representation',
    'model': 'Classifier structure', 'svm_temperature': 'SVM temperature',
    'eegpt_temperature': 'EEGPT temperature', 'svm_weight': 'SVM blend weight',
    'features.rank': 'Tensor rank', 'features.lam': 'Graph regularization strength',
    'features.coupling': 'Connectivity coupling', 'svm.C': 'SVM C',
    'svm.kernel': 'SVM kernel function', 'svm.gamma': 'RBF gamma',
}


def flatten(d, prefix=''):
    result = {}
    for k, v in d.items():
        name = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            result.update(flatten(v, name))
        else:
            result[name] = v
    return result


def key(v):
    # An integer and its exact floating representation are the same setting.
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def sort_value(v):
    if isinstance(v, (float, int)):
        return (0, v)
    return (1, str(v))


def score(v):
    return 'Not saved' if v is None else f'{100 * v:.2f}'


def record(i, params, f1, ba=None, accuracy=None, **other):
    assert f1 is not None and math.isfinite(f1)
    return {'id': str(i), 'parameters': params, 'f1': f1, 'ba': ba,
            'accuracy': accuracy, **other}


def read_records(data, kind):
    rows = data['candidates'] if kind in ('binary', 'graph', 'inner_svm') else data.get('top_candidates', data.get('validation_candidates', []))
    out = []
    for i, row in enumerate(rows):
        if row.get('invalid') or row.get('error'):
            continue
        if kind == 'inner_svm':
            m = row['inner_metrics']
            out.append(record(row['candidate'], row['config'], m['macro_f1'], m['balanced_accuracy'], m['accuracy']))
        elif kind == 'binary':
            out.append(record(row['candidate'], row['config'], row['inner_macro_f1'], row['inner_balanced_accuracy']))
        elif kind == 'graph':
            out.append(record(i, flatten({'features': row['features'], 'svm': row['svm']}),
                              row['mean_f1'], mean(row['balanced']), ftd_recall=mean(row['ftd'])))
        else:
            params = {k: v for k, v in row.items()
                      if not any(w in k for w in ('accuracy', 'f1', 'recall', 'precision'))
                      and k != 'feature_count'}
            config = params.pop('config', {}) or {}
            params.update(flatten(config))
            f1 = next(row[k] for k in ('cv_macro_f1', 'val_macro_f1', 'macro_f1') if k in row)
            ba = next((row[k] for k in ('cv_macro_recall', 'val_balanced_accuracy', 'balanced_accuracy') if k in row), None)
            acc = next((row[k] for k in ('cv_accuracy', 'val_accuracy', 'accuracy') if k in row), None)
            out.append(record(i, params, f1, ba, acc))
    return out


def rank(row):
    # One actual winning configuration supplies ALL metrics in a displayed row.
    return (row['f1'], row['ba'] if row['ba'] is not None else -1,
            row['accuracy'] if row['accuracy'] is not None else -1)


def main():
    if not DETAIL.exists():
        assert MAIN.read_text(encoding='utf-8').startswith('# EEG experiment and parameter-search record')
        shutil.copy2(MAIN, DETAIL)
    nc = ROOT / 'neural_comparison'
    adv = nc / 'advanced_clean_spectral'
    specs = [
        ('binary', 'Binary log-mel SVM: 174 participants', nc / 'binary_logmel_search/outputs/nested/final/search.json', 'binary',
         'All 256 final-refit candidates. Inner 3-fold CV selection scores; not outer-test scores. These rows address the window/STFT/mel/SVM search.', None),
        ('svm3', 'Original three-class spectral SVM', adv / 'outputs/results/site_normalized_metrics.json', 'other',
         'All 66 saved validation candidates; pooled cohort, 26-person validation split. Selection scores, not 27-person test scores.', None),
        ('extended', 'Extended three-class SVM', adv / 'outputs/site_svm_extended_tuning/metrics.json', 'other',
         'Only the top 50 of 1,568 candidates survive. Values absent from that subset have unknown scores. Development CV: 2 repeats × 5 folds.',
         ('tune_site_svm_extended.py', {'pca': 'PCA_COMPONENTS', 'C': 'C_VALUES', 'gamma': 'RBF_GAMMAS', 'ftd_boost': 'FTD_BOOSTS', 'kernel': ['linear', 'rbf']})),
        ('tt', 'Tensor Train + SVM', adv / 'outputs/tensor_train_svm/metrics.json', 'other',
         'Only the top 20 of 288 candidates survive. Development CV: 2 repeats × 5 folds.',
         ('train_tensor_train_svm.py', {'rank': 'RANKS', 'channel_rank': 'CHANNEL_RANKS', 'spectral_rank': 'SPECTRAL_RANKS', 'C': 'SVM_C', 'kernel': 'SVM_KERNELS'})),
        ('tucker', 'Tucker decomposition before pooling + SVM', adv / 'outputs/prepool_tucker_svm/metrics.json', 'other',
         'Only the top 20 of 288 candidates survive. Scores are from the 26-person validation split.',
         ('train_prepool_tucker_svm.py', {'ranks': 'RANKS', 'summary': 'SUMMARIES', 'normalization': 'NORMALIZATIONS', 'pca': 'PCA_COMPONENTS', 'C': 'C_VALUES', 'kernel': 'KERNELS'})),
        ('riemann', 'Filter-bank Riemannian SVM', adv / 'outputs/riemannian_svm/metrics.json', 'other',
         'Only the top 30 of the declared 1,440 candidates survive. Development-CV scores; balanced accuracy was not saved per retained candidate.',
         ('train_riemannian_svm.py', {'covariance': ['raw', 'trace'], 'pca': [10, 20, 30, 40, None], 'C': [.01, .1, 1, 10], 'kernel': ['linear', 'rbf'], 'normalization': ['site_zscore', 'site_center', 'global']})),
        ('blend', 'SVM + EEGPT probability blend', adv / 'outputs/deep_models/eegpt_attention/svm_blend_metrics.json', 'other',
         'All 420 validation combinations; pooled cohort, 26-person validation split. Balanced accuracy was not saved per combination.', None),
        ('gntf', 'Graph-regularized NTF + SVM', nc / 'graph_ntf_svm/outputs/full/final_gntf/search.json', 'graph',
         'Full final-refit inner search. The table maximizes mean inner F1; the deployed refit used a one-standard-error simplicity rule and can therefore select a different configuration. Coupling was fixed at zero.', None),
    ]
    for arm in ('native128', 'native19', 'legacy19', 'aligned19'):
        specs.append((arm, f'BrainLat-only binary SVM: {arm}', nc / f'brainlat_native_ablation/outputs/matched/final_{arm}/search.json', 'binary',
                      'All 256 final-refit inner-search candidates; 86 BrainLat participants. These are source-CV selection scores.', None))
    tensor_blend = adv / 'outputs/tensor_eegpt_blend_2026-11-05'
    if (tensor_blend / 'metrics.json').exists():
        specs += [
            ('new_tt_svm', 'New Tensor Train SVM for EEGPT blend: training-only rank/SVM search', tensor_blend / 'svm_search.json', 'inner_svm',
             'All 288 candidates. Pooled out-of-fold metrics from 5-fold CV within 121 training participants. Each fold refits normalization and TT. The 26 validation participants tune the subsequent blend; 27 test participants are excluded from both selections.', None),
            ('new_tt_blend', 'New Tensor Train SVM + EEGPT: blend search', tensor_blend / 'blend_search.json', 'other',
             'All 420 combinations, evaluated on the 26-person validation split. EEGPT is reused without retraining.', None),
            ('historical_tt_blend', 'Historical TT/SVM settings + EEGPT: follow-up control blend search', tensor_blend / 'historical_tt_control/blend_search.json', 'other',
             'All 420 validation combinations. This follow-up fixes the old TT/SVM settings; those settings were originally selected by 147-person development CV. This control was added after the fresh-search test result; both outcomes are retained.', None)]

    lines = ['# Best saved results for each parameter value', '',
             'Each row fixes the parameter/value on the left, then selects the **highest macro F1 among the saved configurations using that value**. Other settings may change between rows. Accuracy and balanced accuracy come from that same winning configuration; they are not independently maximized.', '',
             '**All scores in these parameter tables are validation/development/inner-CV selection scores (%).** They are not test performance or isolated effects of changing one parameter. A value tried in more configurations has more opportunities to achieve a high score. No new training was run.', '',
             '`BA / macro recall` is balanced accuracy, equivalent to unweighted mean class recall. `Not saved` means the per-candidate metric is absent. In truncated searches, “No retained score” does not mean the value performed badly.', '',
             f'The earlier full experiment record, including held-out results and all source tables, is preserved in {link(DETAIL, "the detailed experiment record")}. Exact winning configurations are in {link(WINNERS, "the configuration lookup")}.', '',
             '## How to read a row', '',
             '“Main window = 4 s” means: among all saved candidates using a 4-second window, this is the one with the highest inner macro F1. The STFT window, mel bands, scaler and SVM settings are allowed to differ from the winning 2-second or 8-second candidate.', '']
    lookup = ['# Winning configurations for the parameter-result tables', '',
              'Candidate IDs are local to each experiment. Unless the source stores an explicit candidate ID, the number is the zero-based position in the saved candidate list. Scores retain full precision here.', '']
    audit = {'selection': 'maximum F1; ties broken by balanced accuracy, then accuracy, then source order',
             'no_models_trained': True, 'sources': [], 'rows': []}
    for tag, title, path, kind, note, declared_spec in specs:
        data = read(path)
        records = read_records(data, kind)
        declared = {}
        if declared_spec:
            script, fields = declared_spec
            constants = literal_constants(adv / script)
            declared = {p: (constants[v] if isinstance(v, str) else v) for p, v in fields.items()}
        if tag == 'riemann':
            declared['band_set'] = list(literal_constants(adv / 'train_riemannian_svm.py')['BAND_SETS'])
        params = list(dict.fromkeys([*declared, *(p for r in records for p in r['parameters'])]))
        params = [p for p in params if p in LABELS]
        threshold_rows = []
        if kind == 'binary':
            for r in data['candidates']:
                for t in r.get('threshold_scores', []):
                    threshold_rows.append(record(f'{r["candidate"]}@{t["threshold"]}',
                                                 {**r['config'], 'threshold': t['threshold']},
                                                 t['macro_f1'], t['balanced_accuracy']))
        failed = [r for r in data.get('candidates', []) if r.get('invalid') or r.get('error')]
        lines += [f'## {title}', '', note, '',
                  f'Source: {link(path)}. Retained base configurations: **{len(records)}**.', '']
        if failed:
            lines += [f'The source also contains **{len(failed)} failed/invalid candidates**, excluded from performance comparisons; their errors remain in the original log. The retained count above includes scored candidates only.', '']
        output = []
        winners = {}
        for param in params:
            groups = defaultdict(list)
            pool = threshold_rows if param == 'threshold' and threshold_rows else records
            for r in pool:
                if param not in r['parameters']:
                    continue
                if param.endswith('gamma') and r['parameters'].get(param.replace('gamma', 'kernel')) != 'rbf':
                    continue  # gamma is inactive in a linear SVM
                groups[key(r['parameters'][param])].append(r)
            values = {key(v): v for v in declared.get(param, [])}
            values.update({k: json.loads(k) for k in groups})
            for k, value in sorted(values.items(), key=lambda item: sort_value(item[1])):
                group = groups.get(k, [])
                if not group:
                    if tag == 'tt' and param == 'rank' and str(value) in data.get('rank_summary', {}):
                        f1 = data['rank_summary'][str(value)]
                        output.append([LABELS[param], value, score(f1), 'Not saved', 'Not saved', 'Summary only', 'Not saved'])
                        audit['rows'].append({'experiment': tag, 'parameter': param, 'value': value,
                                              'summary_macro_f1': f1, 'source_field': f'rank_summary.{value}'})
                        continue
                    output.append([LABELS[param], value, 'No retained score', 'Not saved', 'Not saved', 0, '—'])
                    continue
                winner = max(group, key=rank)
                assert winner['f1'] == max(r['f1'] for r in group)
                winners[winner['id']] = winner
                output.append([LABELS[param], value, score(winner['f1']), score(winner['ba']),
                               score(winner['accuracy']), len(group), winner['id']])
                audit['rows'].append({'experiment': tag, 'parameter': param, 'value': value,
                                      'retained_matching_records': len(group), 'winner': winner})
        lines += table(['Parameter', 'Value', 'Best macro F1 (%)', 'BA / macro recall (%)',
                        'Accuracy (%)', 'Matching saved configs', 'Winning candidate'], output) + ['']
        lines += ['RBF gamma rows include only RBF configurations. All other metrics in a row belong to its F1-winning candidate.', '']
        if threshold_rows:
            lines += ['Threshold rows use all saved per-threshold scores. Other parameter rows use each base candidate’s selected threshold.', '']
        if tag == 'tt':
            lines += ['The terminal-rank summary also retained the best F1 for ranks absent from the top-20 candidate list. Those F1 values are included; their winning configurations and other metrics were not saved.', '']
        lookup += [f'## {title}', '', f'Source: {link(path)}.', '']
        lookup += table(['Candidate', 'Complete settings', 'Macro F1 (0–1)', 'BA (0–1)', 'Accuracy (0–1)'],
                        [(i, r['parameters'], r['f1'], r['ba'], r['accuracy']) for i, r in winners.items()]) + ['']
        audit['sources'].append({'source': path.as_posix(), 'sha256': sha(path), 'records': len(records),
                                 'parameter_rows': len(output), 'threshold_records': len(threshold_rows),
                                 'excluded_invalid_candidates': failed})
    if (tensor_blend / 'metrics.json').exists():
        fresh = read(tensor_blend / 'metrics.json')
        historical = read(tensor_blend / 'historical_tt_control/metrics.json')
        rows = []
        for label, m in [('No-tensor SVM + EEGPT', fresh['test']['Spectral SVM + EEGPT']),
                         ('Historical TT-SVM alone', historical['test']['Historical TT-SVM']),
                         ('Historical TT-SVM + EEGPT', historical['test']['Historical TT-SVM + EEGPT tuned']),
                         ('New training-CV-selected TT-SVM + EEGPT', fresh['test']['TT-SVM + EEGPT tuned'])]:
            rows.append((label, score(m['accuracy']), score(m['macro_f1']), score(m['balanced_accuracy'])))
        lines += ['## Newly completed tensor-SVM + EEGPT ensemble: actual test results', '',
                  'This table is explicitly held-out performance, separate from the parameter-selection scores above. All rows use the historical 27-person test set. EEGPT was not retrained. The no-tensor blend and historical standalone TT scores were reproduced exactly.', '']
        lines += table(['Model', 'Test accuracy (%)', 'Test macro F1 (%)', 'Test BA / macro recall (%)'], rows) + ['',
                  f'Full reports: {link(tensor_blend / "REPORT.md", "fresh training-only search")}, {link(tensor_blend / "historical_tt_control/REPORT.md", "historical-settings control")}. These exploratory results do not confirm a 70% tensor-ensemble result.', '']
    joint = adv / 'outputs/tensor_eegpt_joint_search_2026-11-05'
    if (joint / 'metrics.json').exists():
        summary = read(joint / 'metrics.json')
        lines += ['## Joint tensor/SVM/EEGPT-blend search: 302,400 combinations', '',
                  'This search chooses tensor/SVM settings for the ENSEMBLE rather than choosing the standalone SVM first. It evaluates 720 tensor/SVM configurations, including RBF gamma scale/0.01/0.1/1, with 420 blend settings each. EEGPT remains fixed. All final choices use validation ensemble F1; training-only five-fold SVM CV is a tie-break. The 26-person validation split and previously examined 27-person test split limit generalization claims.', '',
                  '### Actual test results (separate from parameter-selection scores)', '']
        lines += table(['Model', 'Test accuracy (%)', 'Test macro F1 (%)', 'Test BA / macro recall (%)'],
                       [(name, score(m['accuracy']), score(m['macro_f1']), score(m['balanced_accuracy']))
                        for name, m in summary['test'].items()]) + ['',
                  '### Best joint validation result for each parameter value', '',
                  'Each row maximizes ENSEMBLE validation macro F1 among matching configurations; all displayed scores belong to that same winner. The selected ensemble scored 80.11% validation F1 but 58.73% test F1. Do not report a validation maximum as test performance.', '']
        best = read(joint / 'parameter_best_results.json')
        lines += table(['Parameter', 'Value', 'Best validation macro F1 (%)', 'Validation BA (%)', 'Validation accuracy (%)', 'SVM candidate', 'Blend candidate'],
                       [(LABELS.get(r['parameter'], r['parameter']), r['value'], score(r['winner']['val_macro_f1']),
                         score(r['winner']['val_balanced_accuracy']), score(r['winner']['val_accuracy']),
                         r['winner']['svm_candidate'], r['winner']['blend_candidate']) for r in best]) + ['',
                  f'Full report: {link(joint/"REPORT.md")}. All 302,400 scored combinations: {link(joint/"all_joint_candidates.jsonl.gz")}. Full winning settings per parameter: {link(joint/"parameter_best_results.json")}. Verification: {link(joint/"verification.json")}.', '']
    lines += ['## Held-out results and searches with different records', '',
              'Use the detailed record for the CNN/EEGPT training settings, frozen-EEGPT outer folds, Fourier nested evaluation, compatibility experiments and historical benchmarks. Their artifacts do not all contain a common complete candidate table suitable for pooling here. We do not mix inner scores from different outer folds to create an apparent universal winner.', '',
              f'Complete context and held-out metrics: {link(DETAIL)}. Full saved candidate tables: {link(OUT / "01_binary_logmel.md", "binary searches")}, {link(OUT / "02_threeclass_spectral_neural.md", "SVM / CNN / EEGPT searches")}, {link(OUT / "03_graph_ntf.md", "tensor searches")}, {link(OUT / "04_openneuro_fourier.md", "Fourier searches")}.', '',
              'More search may improve results, but these conditional maxima cannot establish a performance ceiling or a causal advantage for a particular parameter. Confirm a new choice using the predefined outer evaluation.', '',
              f'Generation script: {link(Path(__file__))}.', '']
    WINNERS.write_text('\n'.join(lookup), encoding='utf-8')
    MAIN.write_text('\n'.join(lines), encoding='utf-8')
    for path in (MAIN, WINNERS):
        width = None
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('|'):
                size = len(re.split(r'(?<!\\)\|', line)) - 2
                width = size if width is None else width
                assert width == size, (path, line)
            else:
                width = None
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            assert Path(target).exists(), target
    for source in audit['sources']:
        assert sha(source['source']) == source['sha256']
    audit['verification_passed'] = True
    (OUT / 'parameter_best_results_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'Updated {MAIN}: {len(audit["rows"])} scored parameter/value rows from {len(specs)} searches. All checks passed.')


if __name__ == '__main__':
    main()
