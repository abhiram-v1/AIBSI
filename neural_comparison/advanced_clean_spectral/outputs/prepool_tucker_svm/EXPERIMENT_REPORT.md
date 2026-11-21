# Pre-pooling Tucker/HOSVD + SVM

## Question

Does decomposing the full electrode x mel-band x STFT-frame tensor before
participant aggregation preserve more diagnostic information than the earlier
summary-first Tensor Train pipeline?

## Protocol

- Input per EEG window: `19 electrodes x 24 mel bands x 13 STFT frames`.
- The shared Tucker/HOSVD basis was fitted using training-participant windows
  only.
- The existing participant-disjoint split was retained: 121 train, 26
  validation, and 27 test participants.
- Validation macro F1 selected ranks, participant summary, site normalization,
  PCA size, SVM C, and kernel from 288 candidates.
- After selection, the decomposition and classifier were refitted on the 147
  development participants and evaluated once on the 27 fixed test participants.

## Selected pipeline

- Tucker ranks: `(8, 16, 5)` for electrode, mel-frequency, and STFT-time modes.
- Per-window core size: `8 x 16 x 5 = 640` coefficients.
- Participant pooling: mean and standard deviation across windows, producing
  1,280 features.
- PCA: 20 components.
- Class-balanced RBF SVM with `C=10`.
- Validation: 73.1% accuracy, 69.9% macro F1, 70.2% balanced accuracy.

## Fixed-test result

| Measure | Result |
|---|---:|
| Accuracy | 59.3% |
| Macro F1 | 57.6% |
| Balanced accuracy / macro recall | 58.2% |
| AD recall | 54.5% |
| FTD recall | 50.0% |
| Healthy-control recall | 70.0% |

Confusion matrix, rows true and columns predicted in AD/FTD/HC order:

```text
[[6, 2, 3],
 [2, 3, 1],
 [0, 3, 7]]
```

## Comparison

| Pipeline | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Log-mel SVM | 66.7% | 64.3% | 64.2% |
| Summary-first Tensor Train + SVM | 51.9% | 46.4% | 46.8% |
| **Pre-pooling Tucker/HOSVD + SVM** | **59.3%** | **57.6%** | **58.2%** |

Pre-pooling decomposition improved substantially over summary-first Tensor
Train, but it did not surpass the undecomposed log-mel SVM. The 69.9% validation
macro F1 fell to 57.6% on the test participants, showing that rank and classifier
selection still overfit the small development cohort.

This is an exploratory fixed-split result. The test set was inspected in prior
project iterations, so it is not an unbiased publication-grade performance
estimate. Repeated nested participant-level evaluation is required for that
claim.
