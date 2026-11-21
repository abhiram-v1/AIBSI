# Raw 1D CNN experiment

The model received raw 19-channel, eight-second EEG windows at 128 Hz. It used learned temporal filters and no STFT, mel filterbank or MFCC transform. Three predetermined seeds were early-stopped using participant-level validation macro F1, then averaged before one evaluation on the held-out participants.

## Protocol

- Input per window: 19 electrodes x 1024 samples
- Model parameters: 75,463
- Train / validation / test participants: 121 / 26 / 27
- Selected epochs by seed: 6, 4, 9

## Held-out participant result

- Accuracy: 0.5926
- Balanced accuracy: 0.5848
- Macro F1: 0.5754
- Recall AD / FTD / HC: 0.4545 / 0.5000 / 0.8000
- Confusion matrix: [[5, 4, 2], [3, 3, 0], [0, 2, 8]]

## Reference results on the same fixed test participants

- Log-mel SVM: accuracy 0.6667, balanced accuracy 0.6424, macro F1 0.6429
- MFCC SVM: accuracy 0.5926, balanced accuracy 0.5818, macro F1 0.5847
