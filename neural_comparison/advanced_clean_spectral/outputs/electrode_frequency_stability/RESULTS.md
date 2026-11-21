# Electrode and frequency stability experiment

Run date: 2026-10-13

## Protocol

- Data: 174 participants ({'AD': 71, 'FTD': 42, 'HC': 61}).
- Representation: participant-level mean and standard deviation of 19-channel, 24-bin log-mel spectra.
- Evaluation: 10 repeats × 5 participant-level folds, stratified by site and diagnosis.
- Site normalization, grouped permutation ranking, electrode selection, PCA, regularization selection, and SVM fitting were repeated inside the applicable training fold.
- Primary metric: participant-level macro F1.

## Repeated cross-validation

| Electrodes | Macro F1 mean ± fold SD | Accuracy mean ± fold SD | Macro recall mean ± fold SD |
|---:|---:|---:|---:|
| 5 | 47.4% ± 7.6 | 50.1% ± 7.8 | 47.9% ± 7.8 |
| 8 | 49.1% ± 7.0 | 51.8% ± 7.4 | 49.3% ± 7.0 |
| 12 | 47.4% ± 5.9 | 50.7% ± 6.6 | 47.9% ± 6.1 |
| 19 | 49.0% ± 6.6 | 51.9% ± 6.7 | 49.4% ± 6.8 |

Control with the original fixed `C=0.1` and all 19 electrodes: 48.1% ± 7.4 macro F1. This confirms that the lower repeated-CV estimate is caused by participant-split instability, not by the new C search.

## Fixed exploratory held-out split

Inner development folds selected 5 electrodes: F8, T8, C3, P3, P8.

| Model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| 5 electrodes | 40.7% | 37.7% | 37.4% | 45.5% | 16.7% | 50.0% |
| 8 electrodes | 55.6% | 52.8% | 52.6% | 54.5% | 33.3% | 70.0% |
| 12 electrodes | 51.9% | 42.0% | 44.5% | 63.6% | 0.0% | 70.0% |
| 19 electrodes | 66.7% | 64.3% | 64.2% | 72.7% | 50.0% | 70.0% |

## Most stable electrodes

| Electrode | Top-8 selection frequency | Permutation F1 drop | Leave-one-out F1 drop |
|---|---:|---:|---:|
| T8 | 62% | 0.93 pp | 0.76 pp |
| O2 | 60% | 0.97 pp | 2.38 pp |
| O1 | 56% | 0.55 pp | 0.02 pp |
| C3 | 54% | 0.58 pp | 0.56 pp |
| F7 | 52% | 0.37 pp | 0.53 pp |
| P7 | 50% | 0.49 pp | -0.81 pp |
| F3 | 48% | 0.46 pp | 0.71 pp |
| T7 | 46% | 0.47 pp | -0.41 pp |

## Frequency bands

| Band | Permutation F1 drop |
|---|---:|
| Theta 4-8 Hz | 6.11 pp |
| Beta 13-30 Hz | 4.01 pp |
| Gamma 30-40 Hz | 3.11 pp |
| Alpha 8-13 Hz | 0.87 pp |
| Delta 1-4 Hz | 0.10 pp |

## Generalization weaknesses

Across repeated folds, the 19-electrode model's pooled FTD recall was 32.1% and FTD F1 was 29.9%.

| Site | Repeated-prediction macro F1 |
|---|---:|
| AHEPA | 56.0% |
| AR | 44.1% |
| CL | 39.7% |

## Interpretation

- Positive permutation or leave-one-out values mean performance fell when that information was destroyed or removed.
- Negative values mean the feature group was unstable or sometimes harmful; they are not evidence that the electrode is biologically irrelevant.
- The electrode-frequency heatmap is descriptive. It shows class-associated spectral differences after site centering, not causal biomarkers.
- The fixed 27-participant test set has been inspected in earlier experiments, so repeated nested cross-validation is the more credible estimate.
- Electrode selection can reduce input dimension and noise, but it does not create additional independent FTD patients.
- The data-driven 8-electrode model matched the 19-electrode mean, but its selected channels changed substantially between folds. There is no defensible fixed reduced montage yet.
- Theta carried the largest reproducible grouped contribution. Beta and 30–40 Hz also contributed, although the latter must be checked for residual muscle or acquisition noise.
- No compact montage improved generalization. The fixed split's 64.3% macro F1 is optimistic relative to the repeated-fold estimate.
