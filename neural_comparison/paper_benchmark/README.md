# Three-class literature-method adaptations on the full merged EEG cohort

This benchmark evaluates all 174 usable participants (71 AD, 42 FTD, 61 HC).
Five identical outer participant folds cover each person exactly once. The
outer membership and seed match repeat 0 of the existing graph-NTF comparison.
This is one five-fold run, not three repeats or LOSO. The cohort has been
explored previously; this is internal validation, not a fresh external test.

## Methods and source papers

1. **Li CP6 + Gaussian Naive Bayes**:
   https://doi.org/10.1093/braincomms/fcae227
   Channel-by-frequency PSD, nonnegative CP rank six, participant component
   activations, Gaussian NB. We refit both min/max normalization and CP using
   only inner/outer training participants. Held-out activations use NNLS with
   fixed shape factors. The original full-population exploratory factorization
   is not reused across test folds. NB variance smoothing is selected in three
   inner participant folds.
   Each CP fit is bounded at500 L-BFGS-B iterations with three initializations;
   saved optimizer flags distinguish convergence from reaching this cap.
2. **Rostamikia-style hierarchical handcrafted-feature SVM**:
   https://doi.org/10.1007/s11571-024-10152-7
   Mean, variance, band powers, maximum lagged cross-correlation, magnitude
   squared coherence, Katz fractal dimension, approximate entropy and a bounded
   Rosenstein-style largest Lyapunov estimate. Two stages first separate
   dementia from HC, then AD from FTD, giving one three-class decision. Within
   each training stage, KS normality guides Welch t versus Mann-Whitney ranking.
   The number of features, C and linear/RBF kernel are selected in three inner
   participant folds. All feature selection and scaling are refit per fold.
3. **DICE-Net three-class adaptation**:
   https://doi.org/10.1109/ACCESS.2023.3294618
   Author source: https://github.com/souvlasvegas/EEG-DICE-net
   The author's `Model_cls_late_concat` topology has two 19-channel depthwise
   5x5 convolutions, sinusoidal encodings, separate 20-dimensional two-head
   transformer branches, feature-channel tokens, late concatenation, and a
   24-unit head. Here the head produces three logits. The transformer's
   `batch_first=True` is explicit: the author source feeds B,T,F tensors to
   a default sequence-first transformer, which can mix records along the wrong
   axis. We correct this and test inference against single-record batches.
   Both inputs retain 30 frames: relative band power and channel-averaged
   Morlet coherence, using 31-second continuous windows and 2-second/1-second
   analysis steps. At most ten windows/person, evenly spaced over valid windows.
   Scaling is trained only on training windows. AdamW lr .001, wd .01,
   batch32, 80-epoch ceiling, inner participant early stopping (patience20).
4. **Jiang-style microstate NFNet cascade**:
   https://doi.org/10.3390/bios16050258
   The linked author repository returned 404 when checked on 2026-11-03.
   Reimplementation from the main article and downloaded Supplement S2:
   four polarity-invariant modified-k-means templates per band, GFP peaks,
   Hungarian alignment against broadband anchors, continuous absolute spatial
   correlations, Alpha+Delta stage1 and Delta+Theta stage2. Initial conv15/s2,
   residual conv7 blocks with 128/256 channels, pool4, projection to64,
   four-head attention, dropout .5, two binary heads. Stage decisions use
   participant majority voting. Main text says Adam1e-4/AGC.04; supplement
   instead says SGD/cosine/AGC.01. We freeze the main-text optimizer choice,
   use the supplement widths, and document this unresolved source conflict.
   Maximum100 epochs, inner participant stopping patience20 on macroF1.

## Shared inputs and explicit departures from the papers

- All174 participants, common verified 19-channel montage; BrainLat interpolated
  from128 channels, average reference, 1-40Hz, 128Hz. No age/MMSE/MoCA/site dummy
  or duration/window-count predictor is appended.
- The original moderate eight-second windows are reconstructed from source EEG.
  Membership/counts match exactly: 6,721 windows, max40/person. Duplicate EEG
  windows across different participants are explicitly checked.
- Legacy window selection does not explicitly reject EEGLAB boundary
  annotations. This inherited signal-quality limitation is retained and stated;
  it is not itself participant train/test leakage.
- Li uses the 19x40 log-PSD summary at integer1-40Hz rather than1-45Hz, L-BFGS-B
  rather than the source's tensor-library solver, and no age residualization
  because harmonized ages are not available for the full cohort. Rank6 is
  fixed from the paper, not chosen after seeing outer predictions.
- SVM uses the five shared Butterworth subbands1-4/4-8/8-13/13-25/25-40Hz
  instead of the source's db4 DWT subbands. Feature estimates are first computed
  within eight-second windows, then averaged within person. Nonlinear measures
  use four evenly spaced windows/person at64Hz for a fixed compute budget;
  all other windows still enter spectral/time/connectivity summaries. LE uses
  dimension6, delay4, Theiler32, horizon12 at64Hz; ApEn m2,r.2SD. These choices
  are explicit adaptations, not an exact replication of the paper's estimates.
- The common `dice.npy` generated during preparation is NOT the benchmark's
  DICE input; `dice_native.npy` supplies the longer native30-frame input.
  DICE shares the common 1-40Hz/128Hz harmonization, with terminal gamma clipped
  to40Hz. Coherence is Morlet-based channel averaging with within-frame temporal
  averaging, not a claim of bitwise replication of old mne-connectivity0.4.
- Microstates use8s/128Hz instead of10s/256Hz; common1-40Hz rather than.5-45Hz.
  At most16 GFP peak maps/window, at most8 windows/person for template fitting,
  max24000 maps/band, five starts and40 clustering iterations. Templates for
  neural stopping use only the stopping-training subjects; final templates are
  relearned from all outer-training subjects. Residual branches use 1/sqrt2
  scaling. These are documented implementation choices where source code was
  unavailable. No source template is learned from the entire174-person cohort.

These results must be called **literature-method adaptations**, not the
papers' exact reproduced accuracies or evidence that the originals had leakage.

## Evaluation and reproducibility

- Frozen seed20261102; source+diagnosis stratified five-fold participant splits.
- Classical hyperparameter selection uses three inner participant folds.
- Neural models use an inner20% participant validation split, choose epochs
  using participant macroF1, then initialize afresh and refit on all outer
  training participants for the selected epochs. No outer-test early stopping.
- Training samples people uniformly despite variable numbers of windows;
  cross-entropy weights derive from training participant class counts only.
- No post-hoc outer-test threshold tuning or model blending. No added
  probability calibration. Softmax/posterior outputs are not claimed clinically
  calibrated. Microstate hard cascade and DICE mean-probability rules are fixed.
- Saved protocol pins code/input hashes, participant IDs, splits and budgets.
  Each fold saves its model, training membership, selections, predictions and
  diagnostics. `verify.py` replays saved held-out predictions and audits fold
  membership. A separate matched-control table uses only repeat0 of the old
  baseline evaluation, never the three-repeat average against these five folds.
- PyTorch warns that the CUDA memory-efficient attention backward kernel can
  remain nondeterministic despite fixed seeds (`warn_only=True`). Checkpoint
  prediction replay is verified; bitwise-identical fresh retraining is not
  promised.
- Conditional participant-bootstrap intervals describe these fitted OOF
  predictions, not external generalization uncertainty.

## Commands (PowerShell, project root, Python3.10)

```powershell
py -3.10 -u neural_comparison/paper_benchmark/prepare.py
py -3.10 -u neural_comparison/paper_benchmark/prepare_native.py dice
py -3.10 -u neural_comparison/paper_benchmark/prepare_native.py micro
py -3.10 -u neural_comparison/paper_benchmark/test_pipeline.py
py -3.10 -u neural_comparison/paper_benchmark/run.py
py -3.10 -u neural_comparison/paper_benchmark/verify.py
```

Preparation accepts `--shards 4 --shard 0` through3 for independent workers.
Initialize the preparation protocol once before concurrent workers on Windows;
otherwise simultaneous file replacement may fail and the affected shard must
be resumed. Training accepts `--methods ...` and `--only-fold ...`; use identical
protocol arguments for all workers. Serialize GPU methods on this12GB GPU.
Run `--report-only` after all workers finish to merge the final report.

Generated inputs, author-source snapshots, papers, models and reports remain
local under ignored `sources/` and `outputs/`. Existing experiment outputs are
preserved.
