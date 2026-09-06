# NeuroTensor tensorization and decomposition benchmark

This folder contains the first end-to-end subject-level tensor experiment built
from `preprocessing/04_clean_primary/primary`.

## Completed pipeline

1. Welch power estimation for every clean four-second EEG window.
2. Retention of 1–30 Hz at 0.5 Hz resolution.
3. Window-level spatial-spectral relative-power normalization.
4. Non-negative square-root power transform.
5. Chronological division of every subject into ten rank-based time bins.
6. Median aggregation within each subject-time bin.
7. Fold-local decomposition using PCA, NMF, non-negative CP,
   graph-regularized non-negative CP, and non-negative Tucker.
8. Classification using Logistic Regression, RBF SVM, and scikit-learn
   histogram gradient boosting.
9. Repeated subject-level cross-validation at ranks 3, 5, and 8.

## Final tensor

`outputs/tensorization/tensor_relative_sqrt_1_30hz.npy`

Shape: `(69, 19, 59, 10)`

Axes:

1. Subject.
2. EEG channel.
3. Frequency from 1 to 30 Hz.
4. Relative recording-time decile.

The tensor is finite and non-negative. Every time bin contains 7–26 original
four-second windows.

## Evaluation design

- Three repeats of stratified five-fold cross-validation.
- Splits operate on subjects because each tensor row represents one participant.
- Every decomposition is refitted using only the training subjects of its fold.
- Held-out subjects are projected onto the fixed training basis.
- Rank and model settings are fixed for each reported run; test folds are not
  used to fit decomposition components or classifiers.

The rank sweep is exploratory. Selecting the best rank from these results can
introduce selection optimism; a future confirmatory analysis should use nested
rank selection or an untouched external/test cohort.

## Results

See [the full experiment report](outputs/summary/EXPERIMENT_REPORT.md) and the
machine-readable files under `outputs/benchmark_rank_sweep` and
`outputs/summary`.

## CPU environment

The tensor is small enough that GPU acceleration was unnecessary. Python 3.10
was used with NumPy, SciPy, scikit-learn, TensorLy, pandas, and matplotlib from
the persistent user installation.

The external XGBoost wheel was not used: its first download failed package hash
validation and a fresh download stalled. Scikit-learn histogram gradient
boosting was substituted and is clearly labeled in every output.

## Main scripts

- `tensorize_subjects.py`: deterministic subject-tensor construction.
- `run_decomposition_benchmark.py`: decomposition and evaluation engine.
- `run_decomposition_benchmark_cpu.py`: CPU classifier launcher.
- `run_ranked_benchmark_cpu.py`: rank-specific output launcher.
- `run_nmf_convergence_check.py`: extended-iteration NMF verification.
- `summarize_benchmark.py`: cross-rank tables and figures.

