# Compact-feature Random Forest

## Repeated participant-level evaluation

The feature family, site normalization, minimum leaf size, and feature-subsampling rate were selected using four-fold training-only cross-validation inside every outer fold.

| Accuracy | Macro F1 | Macro recall | FTD recall |
|---:|---:|---:|---:|
| 56.2% +/- 6.8% | 51.7% +/- 7.8% | 52.8% +/- 6.9% | 26.7% +/- 16.4% |

Compared with the compact SVM fusion on the identical folds, Random Forest changed macro F1 by +0.35 percentage points.

## Original fixed split

| Model | Accuracy | Macro F1 | Macro recall |
|---|---:|---:|---:|
| SVM | 66.7% | 64.3% | 64.2% |
| EEGPT | 63.0% | 55.0% | 56.5% |
| SVM + EEGPT | 70.4% | 67.8% | 67.6% |
| Random Forest | 55.6% | 48.9% | 50.4% |

Validation selected `relative_band` features, `global` normalization, minimum leaf size 2, and `sqrt` feature subsampling.

## Decision

The Random Forest is retained as a classical-model comparison. The SVM + EEGPT blend remains the final exploratory benchmark because it has the strongest fixed-split accuracy and macro F1. Its 70.4% score must still be described as fixed-split performance rather than expected external performance.
