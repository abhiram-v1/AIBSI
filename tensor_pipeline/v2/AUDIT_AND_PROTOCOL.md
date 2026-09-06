# Second experiment: audit corrections and frozen protocol

This experiment addresses the user's request for Tensor Train and an improved,
course-focused decomposition-to-classifier pipeline. Prior experiments and
datasets are preserved. Its code and outputs are separate from the first run.

## Confirmed issues in the first experiment

1. **Incorrect electrode labels on signal rows.** The source EEGLAB `chanlocs`
   order is Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5,
   T6, Fz, Cz, Pz. The previous tensorizer exported a different hard-coded order
   while leaving the rows unchanged. Consequently its anatomical graph was
   attached to the wrong electrodes. Electrode-specific plots and interpretations
   based on that old list need correction. Global amplitude statistics, common
   average referencing, and ordinary PCA are invariant to that row relabeling;
   the issue does not invalidate every earlier result.
2. **Discontinuity events were ignored.** The source derivatives contain EEGLAB
   `boundary` annotations where previous cleaning removed time periods. Some
   four-second windows cross those cuts. Such windows can contain nonphysiological
   jumps and should not enter spectral/dynamic analyses. The new run uses a
   fixed half-second guard around every cut, applied to all diagnoses equally.
3. **Rank/model selection was not nested.** Selecting the highest score among
   many outer-CV configurations is optimistic. The new run selects rank,
   classifier, and feature-selection count in inner subject folds. The primary
   reported endpoint also selects between method families inside those folds.
4. **SVM prediction convention.** The old benchmark scored argmax of SVC's
   calibrated probabilities rather than `predict()`. Those can disagree. The new
   implementation uses the classifier's own predictions and stores decision
   scores without labeling them probabilities.
5. **Graph-CP optimization caveat.** The old graph update used raw reconstruction
   terms, while the selection/stopping objective normalized the reconstruction
   term and divided the graph penalty by rank. Rescaling channel factors also
   changes an unconstrained graph penalty. Those old results are preserved as
   preliminary implementation results, not evidence validating a particular
   graph-regularized objective. This run uses an explicitly defined fixed
   anatomical pre-smoothing transform followed by TT-SVD instead.
6. **Compression and information loss were hypotheses, not proven causes.**
   Whole-window relative normalization removes global amplitude. Median summaries
   omit some variability; low subject rank can omit discriminative directions.
   This run tests log absolute PSD and richer variability explicitly. It does not
   assume that a more complex method must score better.
7. **Reference interpretation was too strong.** A large across-channel common
   signal and high correlation do not by themselves prove mastoid artifacts.
   Common-average reference is retained here for continuity with the approved
   cleaned dataset; no claim is made that it is uniquely optimal.
8. **Persistence/replay was incomplete.** Earlier outputs mainly saved features
   and predictions, not all decomposition factors and classifiers. The second
   run saves those fitted objects with fold membership and input fingerprints.

## Inputs and retained observations

The approved clean primary array contained 12,270 windows from 69 subjects.
The boundary rule excludes 328 additional windows from this experiment's index,
leaving 11,942 windows: AD 3,985, FTD 4,000, CN 3,957. All 69 subjects remain,
23 per diagnosis. Exclusion IDs are retained in the preparation report; source
arrays are not overwritten. The small window-count differences do not change
subject-level class balance. Each subject contributes exactly 24 chronologically
distributed windows to fitting a window-decomposition basis, and all their
retained windows contribute to pooled features.

Electrode order is checked against `chanlocs` for all 69 subjects. A source-to-
stored-signal transformation check is additionally performed for three subjects.

## Tensor representations

The per-window tensor has axes channel x frequency x local segment time:
19 x 59 x 3. Three overlapping two-second Hann segments lie inside each four-
second window. Frequency bins are 1 to 30 Hz at 0.5 Hz spacing. Entries are
log10 absolute power spectral density, with a fixed numerical floor 1e-12.

TT and HOSVD accept signed real input, so no artificial non-negative shift is
required. Local segment time is not treated as an event-aligned clock across
participants. Subject predictors summarize the distribution of projected window
scores (mean, standard deviation, median, interquartile range).

A second subject tensor uses mean, median and standard deviation of log PSD as
its third feature mode. These are statistical summaries, not a time axis.

The spectral control includes absolute and relative band power, variability,
spectral entropy, alpha peak and slow/fast power ratios. These are computed from
the same retained windows. Demographics, MMSE, diagnosis, recording duration and
sample counts are never input predictors.

## Decomposition families

- Subject Tensor Train projection.
- Window Tensor Train with subject-level pooling.
- Graph-smoothed window Tensor Train with subject-level pooling.
- Window HOSVD/Tucker with subject-level pooling.
- Window Tensor Train fused with spectral controls.
- Spectral control without decomposition (ablation).

TT uses feature modes first and observations last. Left-orthogonal cores form a
shared training basis. Test data are only contracted against that fixed basis.
This avoids treating independent subjects' arbitrarily rotated/sign-flipped TT
cores as comparable features. Candidate terminal ranks are 8, 16, 32; preceding
rank caps are 8 (channel) and 24 (channel-frequency). TT-SVD is a classical tensor
network decomposition, not a claim of a newly invented algorithm.

Tucker uses HOSVD with ranks (6, 8, 3). Feature-selection counts (16, 32, 64) are
chosen inside inner folds. The graph variant uses a symmetric four-nearest-
neighbor graph derived from verified electrode coordinates, normalized graph
Laplacian L and transform (I + 0.5 L)^(-1/2). It is called graph pre-smoothing,
not graph-regularized CP.

## Validation and model selection

Outer evaluation: three repeated stratified five-fold splits of subjects, seed
5040. Inner evaluation: three stratified folds of the outer training subjects,
seed 8000 + outer split index. Folds have no subject overlap. Each inner training
fold refits its own TT/Tucker basis, feature selector, scaler, and classifier.

Classifier candidates are Logistic Regression (C=0.1,1,10), linear SVM
(C=0.1,1), RBF SVM (C=1,10), and histogram gradient boosting (fixed settings).
Class balancing operates at the subject level. No probability calibration using
randomly mixed windows is performed. Final predictions use `predict()`.

The primary endpoint is the outer held-out balanced accuracy of the complete
pipeline selected in inner CV, including the choice of method family. Per-family
outer scores are also reported transparently. The protocol is saved before new
outer scores are inspected. Code/data fingerprints invalidate incompatible
feature caches. Fold-specific models and exact membership are persisted.

## Remaining limits

The cohort has already been explored in earlier experiments. Nested CV protects
the current fitting/selection procedure but cannot remove prior researcher
exposure. An external or genuinely untouched cohort is still needed for an
independent performance estimate. No clinical readiness or guaranteed target
accuracy is implied. Repeated-CV standard deviation is not a confidence interval.
The earlier globally matched subject cohort and approved QC thresholds are held
fixed for this comparison. Generalization to all 88 subjects remains untested.

## References

- Oseledets, Tensor-Train Decomposition (2011):
  https://epubs.siam.org/doi/10.1137/090752286
- TensorLy tensor decomposition documentation:
  https://tensorly.org/dev/user_guide/tensor_decomposition.html
- scikit-learn SVC documentation (predict/probability inconsistency):
  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
