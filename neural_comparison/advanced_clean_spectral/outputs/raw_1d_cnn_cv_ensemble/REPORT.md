# Five-fold raw 1D CNN ensemble

Five CNNs were trained on participant-stratified folds of the 147-person development set. The epoch count and architecture were frozen before this run. Out-of-fold predictions cover each development participant once; the untouched 27-person test probabilities are averaged.

## Results

- OOF development: accuracy 0.6599, balanced accuracy 0.6534, macro F1 0.6431
- Held-out test: accuracy 0.6667, balanced accuracy 0.6929, macro F1 0.6667
- Test recall AD / FTD / HC: 0.5455 / 0.8333 / 0.7000

The test set has been inspected in prior experiments, so it is an exploratory held-out result, not a pristine external validation estimate.
