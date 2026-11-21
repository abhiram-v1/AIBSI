# OpenNeuro-only native 500 Hz raw 1D CNN

This experiment uses only the 88 OpenNeuro ds004504 participants and the supplied artifact-cleaned derivative EEG. The native 19-channel montage and 500 Hz sampling rate are preserved. No BrainLat signals, electrode interpolation, resampling, new filtering, or rereferencing are used.

## Participant split

- Train: 60
- Validation: 14
- Test: 14

## Validation-selected model

- Accuracy: 0.4286
- Balanced accuracy: 0.4722
- Macro F1: 0.4268
- Recall AD / FTD / HC: 0.1667 / 0.5000 / 0.7500
- Confusion matrix: [[1, 4, 1], [1, 2, 1], [0, 1, 3]]

The held-out set has only 14 people, so class-specific percentages have high uncertainty and should not be interpreted as external clinical performance.
