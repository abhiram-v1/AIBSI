# OpenNeuro-only native CNN + EEGPT blend

EEGPT was trained only on the same 60 OpenNeuro training participants. Blend temperature and weight were selected only on the same 14 validation participants, then frozen for the 14-person test set.

- CNN weight: 0.50
- EEGPT weight: 0.50
- Test accuracy: 0.5000
- Test balanced accuracy: 0.5556
- Test macro F1: 0.4841
- Recall AD / FTD / HC: 0.1667 / 0.7500 / 0.7500

The validation and test sets each contain only 14 participants, so this remains exploratory.
