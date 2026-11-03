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

## Matched existing controls (repeat0 only)

These use the same174 people and the exact same five outer test memberships. Their representations and training procedures differ from the literature adaptations.

| Existing control | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| direct | 53.4% | 48.5% | 46.3% |
| gntf | 53.4% | 47.1% | 40.5% |
| tt | 54.6% | 49.1% | 46.3% |

## Verification and interpretation

All20 saved fold models were reloaded and reproduced their outer-test class decisions. Each method has exactly one held-out prediction for every participant; inner and outer participant memberships are disjoint. Prepared input and code hashes were checked.

CP6 optimization: 3/5 final outer fits had no optimizer-converged initialization within the frozen500-iteration/three-start budget (zero-based folds: [1, 2, 4]). Scores are bounded-optimization results, not proof of an optimal CP solution.

The best observed accuracy is 60.3% (105/174 people), but its FTD recall is only 31.0%. Do not describe it as reliable recognition of all three diagnoses.

The paper methods were adapted, not exactly replicated. In particular SVM band extraction/nonlinear estimators and the microstate model differ from source details, DICE has corrected batch semantics and a three-class head, and the entire evaluation uses the merged cohort and common harmonization. These scores do not establish what the original papers would achieve under their original full protocols, or that their published results arose from leakage.

[Implementation, sources and departures](../../README.md) | [Machine-readable verification](verification.json) | [Conditional bootstrap intervals and site results](summary.json)
