"""Plot the completed fixed-split tensor ensembles without fitting models."""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / 'outputs/tensor_eegpt_blend_2026-11-05'


def main():
    fresh = json.loads((OUT / 'metrics.json').read_text())['test']
    old = json.loads((OUT / 'historical_tt_control/metrics.json').read_text())['test']
    methods = [('No tensor:\nSVM + EEGPT', fresh['Spectral SVM + EEGPT']),
               ('Historical\nTT-SVM alone', old['Historical TT-SVM']),
               ('Historical\nTT-SVM + EEGPT', old['Historical TT-SVM + EEGPT tuned']),
               ('Fresh CV search:\nTT-SVM + EEGPT', fresh['TT-SVM + EEGPT tuned'])]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for i, (metric, label, color) in enumerate([('accuracy', 'Accuracy', '#3979ae'),
                                              ('macro_f1', 'Macro F1', '#168d89'),
                                              ('balanced_accuracy', 'Balanced accuracy / macro recall', '#dc8533')]):
        bars = ax.bar(np.arange(4) + (i-1)*.25, [100*m[metric] for _, m in methods], .25,
                      label=label, color=color)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    ax.set_xticks(range(4), [name for name, _ in methods])
    ax.set(ylabel='Score (%)', ylim=(0, 100), title='Tensor-SVM + EEGPT: historical 27-participant test set')
    ax.legend(loc='upper center', ncols=3, fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT / 'combined_comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for ax, (name, m) in zip(axes, [methods[0], methods[2], methods[3]]):
        matrix = np.asarray(m['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=11)
        for i, j in itertools.product(range(3), repeat=2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=12)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=['AD', 'FTD', 'CN'],
               yticklabels=['AD', 'FTD', 'CN'], title=name, xlabel='Predicted', ylabel='Actual')
    fig.suptitle('Three-class participant predictions: confusion matrices')
    fig.savefig(OUT / 'combined_confusion_matrices.png', dpi=180)
    plt.close(fig)
    report = OUT / 'REPORT.md'
    text = report.read_text(encoding='utf-8')
    addition = ('\n## Historical-settings control\n\n'
                'The separate historical TT-settings control reproduced the old standalone TT-SVM exactly (51.85% accuracy). Its validation-tuned EEGPT blend reached 59.26% accuracy, 52.12% macro F1 and 53.43% balanced accuracy. The fresh training-CV-selected ensemble above reached 59.26%, 52.08% and 53.43%, respectively. Both are below the reproduced no-tensor ensemble.\n\n'
                '![Combined historical and fresh comparison](combined_comparison.png)\n\n'
                '![Combined confusion matrices](combined_confusion_matrices.png)\n\n'
                'See [the control report](historical_tt_control/REPORT.md) for its settings, validation-selection history and probabilities.\n')
    if '## Historical-settings control' not in text:
        report.write_text(text + addition, encoding='utf-8')
    print('Saved combined comparison and confusion matrices; no model fitted.')


if __name__ == '__main__':
    main()
