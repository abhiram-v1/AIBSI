# 4. Data-use protocol and next steps

## Which dataset to use now

The canonical dataset for the first controlled experiment is:

`preprocessing/04_clean_primary/primary`

The following remain separate:

- `preprocessing/04_clean_primary/excluded`: signal-QC exclusions and
  balance-only trims from the primary cohort.
- `preprocessing/02_balanced_primary/leftover`: original subjects/windows not
  selected for the first balanced cohort.
- `preprocessing/02_balanced_primary/leftover/partial_segments`: incomplete
  terminal fragments shorter than four seconds.

These directories must not be physically merged into the controlled primary
dataset.

## Mandatory split rule

Split by subject before any learned transformation. Windows from one participant
must never occur in both training and validation/test data.

This applies to:

- Cross-validation.
- Hyperparameter selection.
- Scaling or normalization.
- Artifact thresholds learned from data.
- Graph estimation from functional connectivity.
- Tensor rank selection and tensor decomposition.
- Feature selection and classifier fitting.

The proposal's leave-one-subject-out approach is valid if every step is refitted
inside each training fold. Nested grouped cross-validation may be used when
hyperparameters are tuned.

## Recommended first experiment

1. Use the 69-subject cleaned primary cohort.
2. Construct a non-negative time-frequency power representation over 1–30 Hz.
3. Give subjects equal influence. Do not allow a longer recording or a subject
   with more retained windows to dominate the tensor objective.
4. Build or select the channel graph within the training workflow.
5. Fit tensor decomposition using training subjects only.
6. Project the held-out subject using the learned training factors without
   refitting on the test subject.
7. Train the classifier on training-subject features.
8. Aggregate window predictions into one subject-level prediction.
9. Report subject-level balanced accuracy, macro F1, confusion matrix, and
   class-wise sensitivity/specificity.

## Tensor representation decision

Raw EEG contains positive and negative voltages and is unsuitable as direct
input to non-negative tensor factorization. Use non-negative spectral power,
for example a wavelet or short-time Fourier power representation.

The planned conceptual tensor is:

`subject × channel × frequency × time`

To maintain equal subject influence, either:

- summarize all clean windows for each subject using robust statistics; or
- sample the same number of windows per subject and rotate the selected windows
  across repeated training runs.

Any log-power conversion must be followed by a non-negative shift or another
factorization-compatible transformation whose parameters are fitted on training
subjects only.

## Frequency policy

- Primary analysis: 1–30 Hz.
- Sensitivity analysis: 1–45 Hz.
- Compare results with and without `sub-067`, `sub-077`, and `sub-085` in a
  subject-level sensitivity analysis; do not silently remove them.
- Do not apply a 50 Hz notch to the current derivative data unless a later audit
  demonstrates new contamination.

## Baselines

Tensor results should be compared with simpler subject-safe baselines:

- Relative/absolute band-power features with a conventional classifier.
- Riemannian covariance features.
- A tensor model without graph regularization.
- If practical, a graph model with an anatomical fixed graph versus a
  training-derived functional graph.

These comparisons are necessary to show whether graph regularization and tensor
decomposition add value beyond standard EEG features.

## Later full-data experiment

The clean balanced primary dataset is the controlled benchmark. A second
experiment may use all QC-approved windows from the earlier primary and leftover
collections without physically merging or duplicating the files.

Within each training fold, use a virtual sampler:

1. Select AD, FTD, or CN with equal probability.
2. Select a training subject uniformly inside that class.
3. Select one of that subject's approved windows uniformly.

This revisits FTD windows as needed while rotating through the additional AD and
CN windows. It avoids static duplication and prevents subjects with long
recordings from dominating training.

The subject split must be created before the virtual index is sampled. Leftover
windows from a held-out subject must never enter training.

## Work not yet completed

- Time-frequency feature generation.
- Graph definition and validation.
- Tensor rank selection.
- Graph-regularized tensor decomposition.
- Classifier training.
- LOSO or grouped evaluation.
- MMSE/component association.
- Comparison with the full-data virtual-sampling experiment.

