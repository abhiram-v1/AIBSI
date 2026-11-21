# Periodic/aperiodic biomarkers and EEG microstates

## Protocol

- Raw input: 6,913 four-second, 19-electrode windows at 256 Hz from 174
  participants.
- Spectral branch: robust log-log aperiodic background exponent, offset and fit
  quality; relative delta/theta/alpha/beta power; slowing ratios; and
  theta/alpha/beta peak frequency, height and width. Channel, regional and
  global summaries produced 468 features.
- Microstate branch: four polarity-invariant templates fitted to training-only
  GFP peak maps. Coverage, occurrence, duration, explained variance and
  off-diagonal transition probabilities produced 28 features.
- Supervised training-only ANOVA selection retained 20, 40, 80, 160 or all
  features. Global and site-centred representations, linear/RBF SVM kernels and
  four C values were compared.
- Existing participant split: 121 train, 26 validation and 27 fixed test.
- After validation selection, all feature-learning steps were refit on the 147
  development participants before test evaluation.

The optional `specparam` package was not installed, so the periodic/aperiodic
branch uses a transparent Huber-regression log-log background fit inspired by
that parameterization rather than claiming an exact `specparam` reproduction.

## Selected pipeline

- Feature family: spectral biomarkers only.
- Site-centred features.
- Top 80 ANOVA features.
- Class-balanced linear SVM, `C=10`.
- Validation: 84.6% accuracy, 83.8% macro F1 and 83.8% balanced accuracy.

The same validation score was obtained by the combined branch because its top
80 selected features were spectral; microstate features added no selected
information.

## Fixed-test result

| Measure | Result |
|---|---:|
| Accuracy | 59.3% |
| Macro F1 | 58.3% |
| Balanced accuracy / macro recall | 57.9% |
| AD recall | 63.6% |
| FTD recall | 50.0% |
| Healthy-control recall | 60.0% |

Confusion matrix, rows true and columns predicted in AD/FTD/HC order:

```text
[[7, 2, 2],
 [3, 3, 0],
 [3, 1, 6]]
```

## Feature-family ablation

| Validation-selected branch | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Spectral biomarkers | 59.3% | 58.3% | 57.9% |
| Microstates | 48.1% | 42.3% | 44.3% |
| Combined | 59.3% | 58.3% | 57.9% |

Microstates did not improve the spectral representation. The combined branch
made the same test predictions as the spectral-only branch.

## Comparison with existing models

| Pipeline | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Log-mel SVM | 66.7% | 64.3% | 64.2% |
| Pre-pooling Tucker + SVM | 59.3% | 57.6% | 58.2% |
| Biomarker + microstate experiment | 59.3% | 58.3% | 57.9% |

The targeted features did not beat the existing log-mel SVM. The validation to
test macro-F1 drop from 83.8% to 58.3% is large and confirms development-split
overfitting. The highest ANOVA scores were dominated by aperiodic offsets and
the test accuracy varied markedly by acquisition site (AHEPA 42.9%, AR 80.0%,
CL 66.7%; small per-site test counts). This is consistent with remaining
acquisition-domain sensitivity, not evidence of a stable new dementia marker.

This is exploratory fixed-split evidence. The test set has been inspected in
prior project iterations; repeated nested participant-level evaluation or a new
locked external test set is required for a publication-grade estimate.
