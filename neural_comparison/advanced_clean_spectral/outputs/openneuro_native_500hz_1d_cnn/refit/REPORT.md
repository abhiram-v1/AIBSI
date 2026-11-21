# Native OpenNeuro 500 Hz CNN refit

The validation-selected configuration was refitted on all 74 development participants. Only the 88-person OpenNeuro cohort was used; montage interpolation and resampling were disabled.

- Accuracy: 0.3571
- Balanced accuracy: 0.3889
- Macro F1: 0.3397
- Recall AD / FTD / HC: 0.1667 / 0.7500 / 0.2500
- Confusion matrix: [[1, 5, 0], [0, 3, 1], [0, 3, 1]]

The test set contains only 14 participants (6 AD, 4 FTD, 4 HC), so uncertainty is high.
