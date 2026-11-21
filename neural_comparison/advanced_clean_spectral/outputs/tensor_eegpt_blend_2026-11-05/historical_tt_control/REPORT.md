# Tensor-SVM + EEGPT: historical-settings control

EEGPT weights and cached predictions were reused. TT/SVM parameters were inherited unchanged from the old Tensor Train experiment. This control was added after the fresh training-only CV run; both outcomes are reported. Blend tuning used 26 validation participants, followed by TT/SVM refitting on 147 development participants and evaluation on the historical 27-person test set.

Fixed historical tensor/SVM settings: `{"rank": 4, "channel_rank": 4, "spectral_rank": 24, "C": 0.1, "kernel": "linear"}`.

Validation-selected blend: `{"svm_temperature": 0.75, "eegpt_temperature": 1.0, "svm_weight": 0.5, "val_macro_f1": 0.7000000000000001, "val_accuracy": 0.7307692307692307, "val_balanced_accuracy": 0.7095238095238096}`.

| Model | Test accuracy (%) | Test macro F1 (%) | Test balanced accuracy (%) | FTD recall (%) |
|---|---:|---:|---:|---:|
| Spectral SVM + EEGPT | 70.37 | 67.82 | 67.58 | 50.00 |
| TT-SVM + EEGPT tuned | 59.26 | 52.08 | 53.43 | 16.67 |
| Historical TT-SVM | 51.85 | 46.35 | 46.77 | 16.67 |
| Historical TT-SVM + EEGPT fixed | 59.26 | 52.12 | 53.43 | 16.67 |
| Historical TT-SVM + EEGPT tuned | 59.26 | 52.12 | 53.43 | 16.67 |

The historical standalone TT scores were reproduced exactly. The fresh run selects TT/SVM settings only within 121 training participants; the historical control inherits settings selected by the previous 147-person development CV. Their selection protocols differ. Neither is a new independent estimate of generalization.

All 420 blend candidates, model parameters and per-participant probabilities are saved here. The parent directory contains the fresh 288-configuration TT/SVM search and the no-tensor baseline reproduction.
