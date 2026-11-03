# Literature method adaptations: full174-person, three-class benchmark

Five shared outer participant folds; each participant predicted once. Classical tuning uses three inner folds; neural stopping uses an inner participant split followed by refitting on all outer-training participants.
These are documented reimplementations/adaptations, not exact reproductions of the papers. No external validation. The cohort has been explored in prior experiments.

| Method | Accuracy | Balanced accuracy | Macro F1 | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| li_cp6_gnb | 60.3% | 56.5% | 56.5% | 74.6% | 31.0% | 63.9% |
| rostamikia_hierarchical_svm | 55.2% | 52.5% | 51.8% | 53.5% | 28.6% | 75.4% |
| dice_net_3class | 51.1% | 48.3% | 47.9% | 56.3% | 26.2% | 62.3% |
| jiang_microstate_cascade | 41.4% | 42.7% | 40.4% | 25.4% | 45.2% | 57.4% |

Completed methods: 4/4.
See README.md and protocol.json for source papers, implementation choices and deviations. Bootstrap intervals are conditional on these fitted predictions; overlapping training folds do not provide independent replications.
Moderate preprocessing inherits missing explicit EEGLAB boundary rejection. This is a signal-quality limitation, separate from train/test participant leakage.
