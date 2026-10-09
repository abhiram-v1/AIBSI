# Advanced cleaned EEG spectral experiment

This experiment keeps participant-level splits and compares two representations generated
from the same cleaned 19-electrode windows:

- log-STFT power, 1--40 Hz
- 24-bin log-mel power, 1--40 Hz

Cleaning includes robust bad-channel detection and spherical interpolation, conservative
ocular ICA, average referencing, strict channel-level window rejection, and train-only
electrode ranking. Electrode count is selected using validation macro F1. The held-out test
participants are evaluated once after selection.

```powershell
py -3.10 neural_comparison\advanced_clean_spectral\pipeline.py all
py -3.10 neural_comparison\advanced_clean_spectral\train_deep_models.py --epochs 30
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_deep_ensembles.py
py -3.10 neural_comparison\advanced_clean_spectral\calibrate_log_mel_cnn.py
py -3.10 neural_comparison\advanced_clean_spectral\train_logmel_ensemble.py --epochs 30
py -3.10 neural_comparison\advanced_clean_spectral\train_moderate_logmel.py --epochs 30
py -3.10 neural_comparison\advanced_clean_spectral\train_long_logmel.py --epochs 120 --patience 18
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_site_normalized.py
py -3.10 neural_comparison\advanced_clean_spectral\refine_site_svm.py
py -3.10 neural_comparison\advanced_clean_spectral\calibrate_site_svm.py
py -3.10 neural_comparison\advanced_clean_spectral\train_riemannian_svm.py all
py -3.10 neural_comparison\advanced_clean_spectral\diagnose_best_svm.py
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_site_shift.py
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_binary_ensemble.py
py -3.10 neural_comparison\advanced_clean_spectral\train_attention_pooling.py --epochs 80 --patience 15 --bag-size 12
py -3.10 neural_comparison\advanced_clean_spectral\prepare_eegpt_raw.py
py -3.10 neural_comparison\advanced_clean_spectral\train_eegpt_attention.py --warmup-epochs 6 --finetune-epochs 24 --patience 8 --bag-size 8
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_eegpt_svm_blend.py
```

`pipeline.py` performs cleaning, spectral preparation, electrode analysis, and the
hierarchical SVM comparison. `train_deep_models.py` compares CNN and CNN-BiLSTM models
on both cleaned representations using CUDA. See `RUN_SUMMARY.md` for the final results.
`train_long_logmel.py` tests a 120-epoch ceiling with validation macro-F1 early stopping
and restores the best checkpoint for each seed. Longer training reduced held-out
performance. The current best exploratory result is the validation-selected blend of the
site-normalized log-mel SVM and the fine-tuned EEGPT participant model: 70.4% accuracy,
67.8% macro F1, and 67.6% macro recall on the fixed 27-participant held-out split.
