# Raw CNN and SVM refit

After architecture, epoch counts, temperatures and fusion weights were fixed from validation, the two CNNs and the SVM were refitted using all 147 development participants. The 27 test participants did not influence these choices.

## Held-out fusion result

- Accuracy: 0.6667
- Balanced accuracy: 0.6646
- Macro F1: 0.6555
- Recall AD / FTD / HC: 0.7273 / 0.6667 / 0.6000
- Confusion matrix: [[8, 2, 1], [2, 4, 0], [1, 3, 6]]

Window CNN alone: 0.7037 accuracy, 0.7098 macro F1.
MIL CNN alone: 0.7037 accuracy, 0.6705 macro F1.

The test cohort has been inspected in earlier experiments, so this remains exploratory rather than a fresh blind estimate.
