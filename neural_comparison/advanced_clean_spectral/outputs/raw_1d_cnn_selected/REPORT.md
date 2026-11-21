# Validation-selected raw 1D CNN

A compact 75,463-parameter 1D CNN received raw 19-channel, eight-second EEG windows at 128 Hz. No STFT, mel filterbank or MFCC transform was used. The checkpoint was selected only by participant-level validation macro F1 among three predetermined random seeds.

## Selection

- Selected seed: 20261117
- Selected epoch: 4
- Validation macro F1: 0.6143
- Validation balanced accuracy: 0.6476

## Held-out participant result

- Accuracy: 0.7037
- Balanced accuracy: 0.7010
- Macro F1: 0.6963
- Recall AD / FTD / HC: 0.6364 / 0.6667 / 0.8000
- Confusion matrix: [[7, 4, 0], [2, 4, 0], [0, 2, 8]]

## Controls on the same fixed split

- Three-seed probability average: accuracy 0.5926, macro F1 0.5754
- Log-mel SVM: accuracy 0.6667, macro F1 0.6429
- MFCC SVM: accuracy 0.5926, macro F1 0.5847

This remains an exploratory fixed-split result. Repeated nested participant-level validation is required before treating it as the expected generalization score.
