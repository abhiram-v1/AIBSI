# MFCC SVM experiment

MFCCs were obtained by an orthonormal DCT-II across the existing 24 log-mel bands. All representation and SVM settings were chosen on the 26-person validation split; the 27-person test split was evaluated once after selection.

## Selected configuration

- MFCC coefficients: 8
- Include coefficient 0: True
- Summaries: mean_std
- Feature columns: 304
- PCA: 40
- SVM kernel: linear
- SVM C: 0.01
- Validation macro F1: 0.6441

## Held-out participant result

- Accuracy: 0.5926
- Balanced accuracy: 0.5818
- Macro F1: 0.5847
- Recall AD / FTD / HC: 0.5455 / 0.5000 / 0.7000
- Confusion matrix: [[6, 1, 4], [1, 3, 2], [2, 1, 7]]

## Existing log-mel SVM reference

- Accuracy: 0.6667
- Balanced accuracy: 0.6424
- Macro F1: 0.6429
