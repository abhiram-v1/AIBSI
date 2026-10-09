# Three-class STFT GPU baseline

This experiment classifies a participant as Alzheimer's disease (AD),
frontotemporal dementia (FTD/bvFTD), or healthy control (HC). It combines the
ds004504 derivatives with the BrainLat AD, bvFTD, and HC EEG folders.

The data path is deliberately leakage-safe:

- BrainLat BioSemi-128 signals are spline-interpolated to the same 19 standard
  10-20 positions used by ds004504.
- Each recording is average-referenced, filtered to 1-40 Hz, and resampled to
  128 Hz.
- Non-overlapping 8-second windows are converted to log-power STFT arrays with
  2-second Hann frames and a 0.5-second hop.
- At most 40 accepted windows are retained per participant.
- Train, validation, and test partitions are made by participant. Scaling is
  fitted on training participants only.
- The CNN is trained with CUDA mixed precision and predictions are averaged
  over each participant before metrics are computed.

Run both stages from the repository root:

```powershell
py -3.10 neural_comparison\stft_gpu_3class\run_pipeline.py all
```

Run the subject-level SVM benchmark on the same STFT cache and participant split:

```powershell
py -3.10 neural_comparison\stft_gpu_3class\train_svm.py
```

Run the enhanced participant-bag CNN versus CNN-BiLSTM comparison on CUDA:

```powershell
py -3.10 neural_comparison\stft_gpu_3class\train_deep_compare.py --epochs 50 --seeds 3
py -3.10 neural_comparison\stft_gpu_3class\calibrate_deep_compare.py
py -3.10 neural_comparison\stft_gpu_3class\train_hierarchical_svm.py
```

Generated tensors, checkpoints, metrics, predictions, and ordinary Matplotlib
figures are written below `outputs/` and ignored by Git.
