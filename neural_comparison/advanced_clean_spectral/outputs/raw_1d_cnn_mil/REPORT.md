# Multiscale participant-level raw 1D CNN

The CNN learned temporal filters at four scales directly from raw EEG. Training used eight-window participant bags and attention pooling, so the loss optimized participant classification rather than treating every window as an independent diagnosis.

## Selection

- Selected seed: 20261121
- Selected epoch: 14
- Validation accuracy: 0.7308
- Validation balanced accuracy: 0.7381
- Validation macro F1: 0.7202

## Held-out participant result

- Accuracy: 0.6667
- Balanced accuracy: 0.6707
- Macro F1: 0.6637
- Recall AD / FTD / HC: 0.5455 / 0.6667 / 0.8000
- Confusion matrix: [[6, 5, 0], [2, 4, 0], [0, 2, 8]]

The fixed test set has already been inspected in earlier experiments. Treat this as an exploratory architecture comparison, not a new blind test estimate.
