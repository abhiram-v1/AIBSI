# Validation-selected raw CNN and log-mel SVM fusion

Temperatures and weights were selected solely on the 26 validation participants. The frozen configuration was then applied to the 27 held-out participants.

## Selected validation configuration

- SVM / window CNN / MIL CNN weights: 0.30 / 0.60 / 0.10
- Validation accuracy: 0.8077
- Validation balanced accuracy: 0.7762
- Validation macro F1: 0.7760

## Held-out participant result

- Accuracy: 0.7037
- Balanced accuracy: 0.7010
- Macro F1: 0.6934
- Recall AD / FTD / HC: 0.6364 / 0.6667 / 0.8000
- Confusion matrix: [[7, 3, 1], [2, 4, 0], [1, 1, 8]]

The fixed test cohort has been inspected repeatedly, so this remains exploratory even though the fusion rule never used test labels.
