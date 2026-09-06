# 1. Project and source dataset

## Project objective

The proposal describes a dementia EEG classification project using a
subject-by-channel-by-frequency-by-time representation and graph-regularized
non-negative tensor factorization. The intended downstream work includes
interpretable tensor components, classification of Alzheimer's disease (AD),
frontotemporal dementia (FTD), and cognitively normal controls (CN), and clinical
association analysis involving MMSE.

The attached proposal was treated as project context, not as an instruction to
execute unreviewed steps. Dataset design and leakage prevention were considered
before tensor construction.

## Source dataset

The local source is OpenNeuro dataset `ds004504`, version `1.0.9`. It contains
eyes-closed resting-state EEG from 88 participants:

| Group | Subjects | Age range | Mean age | Sex, M/F |
|---|---:|---:|---:|---:|
| AD | 36 | 49–79 | 66.39 | 12/24 |
| FTD | 23 | 44–78 | 63.65 | 14/9 |
| CN | 29 | 57–78 | 67.90 | 18/11 |
| **Total** | **88** | **44–79** | **66.17** | **44/44** |

Recordings use 19 scalp electrodes from the international 10–20 system:
`Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`, `T3`, `C3`, `Cz`, `C4`, `T4`,
`T5`, `P3`, `Pz`, `P4`, `T6`, `O1`, and `O2`.

The acquisition sampling rate is 500 Hz. The BIDS metadata identifies 50 Hz as
the local power-line frequency.

## Recording coverage

Durations below were obtained by scanning the BIDS recording metadata:

| Group | Total duration | Samples at 500 Hz |
|---|---:|---:|
| AD | 29,403.3 s | 14,701,650 |
| FTD | 16,753.4 s | 8,376,700 |
| CN | 24,433.6 s | 12,216,800 |
| **Total** | **70,590.3 s** | **35,295,150** |

FTD is the smallest group by subject count, recording duration, and available
complete windows. It therefore became the reference group for construction of a
balanced primary dataset.

## Preprocessing already supplied by the dataset authors

The analysis uses the artifact-cleaned derivative EEGLAB `.set` files rather
than the raw recordings. According to the dataset documentation, those files
already received:

1. Butterworth 0.5–45 Hz band-pass filtering.
2. Re-referencing to the A1–A2 mastoid reference.
3. Artifact Subspace Reconstruction (ASR), using a conservative threshold.
4. RunICA decomposition.
5. Automatic ICLabel rejection of eye and jaw artifact components.

Consequently, the project did not repeat ASR, ICA, or broad band-pass filtering.
Repeating those operations without a specific failure could remove useful
disease-related EEG structure.

## Important methodological risks identified

- Windows from the same participant are statistically dependent. Dataset splits
  must be made by subject, never by window.
- Recording duration differs between groups and could become a shortcut for a
  model if sampling is not controlled.
- Sex and age are not perfectly matched in the original diagnostic groups.
- A shared mastoid reference can dominate channel covariance and corrupt a
  graph based on apparent channel connectivity.
- Tensor decomposition, learned graph construction, normalization, feature
  selection, and classifiers must be fitted using training subjects only.
- CN MMSE values are at ceiling, so MMSE association should be interpreted
  within patient groups or with diagnosis controlled.
- Raw EEG is signed and cannot be used directly in a non-negative tensor model;
  the tensor must use a non-negative representation such as spectral power.

