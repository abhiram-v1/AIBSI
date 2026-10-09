# Combined EEG STFT baseline

Run date: 2026-10-07

## Data

- Task: participant-level AD vs FTD/bvFTD vs healthy control classification.
- Sources: OpenNeuro `ds004504` and BrainLat EEG.
- Usable participants: 174.
- Prepared examples: 6,721 non-overlapping 8-second windows.
- Input per window: 19 electrodes x 79 frequency bins (1--40 Hz) x 13 STFT frames.
- Split: 121 train, 26 validation, and 27 held-out test participants.
- The split is stratified by source and diagnosis. A participant's windows never cross splits.

Fourteen BrainLat healthy-control `.set` headers were excluded because their referenced
`.fdt` signal files are absent. Details are recorded in `outputs/prepared/audit.json`.

## Held-out participant results

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro recall |
|---|---:|---:|---:|---:|
| Original GPU residual CNN | 0.481 | 0.443 | 0.431 | 0.443 |
| Flat PCA + SVM | 0.556 | 0.476 | 0.435 | 0.476 |
| Enhanced participant-bag CNN | 0.444 | 0.432 | 0.431 | 0.432 |
| Enhanced CNN-BiLSTM | 0.444 | 0.388 | 0.354 | 0.388 |
| Hierarchical PCA + SVM | **0.556** | **0.520** | **0.525** | **0.520** |

The original CNN checkpoint was selected using validation participant macro F1. It reached
0.609 validation macro F1 at epoch 1 and then overfit. Participant-bag training, spectral
shape normalization, augmentation, compact models, three random seeds, and a CNN-BiLSTM
were then tested on the same split. CNN-BiLSTM reached 0.725 validation macro F1 in its best
seed but did not generalize to the test participants. A validation-selected hierarchical SVM
(dementia vs healthy, then AD vs FTD) produced the best held-out macro F1 and recovered two
of six FTD participants.

These are single-split baseline results. They should not be presented as final research
performance; repeated participant-level cross-validation and a stronger strategy for FTD
separation are required.

## Reproduce

```powershell
py -3.10 neural_comparison\stft_gpu_3class\run_pipeline.py prepare
py -3.10 neural_comparison\stft_gpu_3class\run_pipeline.py train --epochs 40 --batch-size 96 --learning-rate 3e-4
py -3.10 neural_comparison\stft_gpu_3class\train_svm.py
py -3.10 neural_comparison\stft_gpu_3class\train_deep_compare.py --epochs 50 --seeds 3
py -3.10 neural_comparison\stft_gpu_3class\calibrate_deep_compare.py
py -3.10 neural_comparison\stft_gpu_3class\train_hierarchical_svm.py
py -3.10 neural_comparison\stft_gpu_3class\summarize_experiments.py
```
