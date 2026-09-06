# Full-cohort waveform tensor experiment

This third experiment implements the approved use of the earlier leftover
dataset, with class/subject weighting for FTD imbalance. It preserves both
original partitions and every previous model/result. New artifacts live under
`tensor_pipeline/outputs/v3_waveform_full_cohort/`.

## Dataset and exclusions

The starting arrays are the 12,396 original primary windows and 5,023 leftover
windows from stage 02. Together they contain distinct four-second windows from
all 88 original people (36 AD, 23 FTD, 29 CN). These arrays are read-only.

The same common-average reference and per-channel mean removal are applied to
both partitions. Exclude nonfinite signals, original non-OK QC, post-reference
absolute amplitude >=400 microvolts, steps >=150 microvolts per four milliseconds,
channel standard deviation <1 microvolt, and windows overlapping a 0.5-second
guard around an EEGLAB boundary. Thresholds are inherited, not fitted to labels.
Electrode ordering is verified against all 88 source recording headers.

This retains 16,824 windows: AD 7,030, FTD 4,000, CN 5,794. Of these, 4,828 came
from the old leftover partition; 54 are valid stage-04 balance-only trims now
recovered from stage 02. All 11,942 windows in V2 remain available. No formerly
artifact-excluded primary window is reinstated. All 595 exclusions retain IDs
and reasons; short partial tails remain outside fixed-length modeling.

## What “waveform training” means here

Each cleaned 19 x 1,000 waveform at 250 Hz is anti-alias resampled to 125 Hz.
Overlapping 16-sample delay vectors produce a tensor with modes:

`19 channels x 16 lags x 485 local positions`.

Lag spacing is 8 ms and the first-to-last lag spans 120 ms. Each channel/lag
row is centered over local positions. A learned channel-lag filter contracts
against this raw tensor. Its squared response is averaged over positions and
then equally over all retained windows from the person. Log energy becomes
the feature. This preserves short-lag temporal and cross-channel relationships
without requiring arbitrary waveform phase alignment between people.

For efficiency, the code stores each person's average delay covariance S.
For any filter vector b, the pooled squared response is exactly b^T S b.
This identity is tested directly against raw tensor contraction. It avoids
storing or repeatedly processing huge delay tensors; it is not a Fourier/PSD
transform. Filter norm normalization removes arbitrary filter scale.

This is a **second-order, pooled waveform model**, not a claim to model every
detail of raw EEG or long-range sequence order. The supervised version learns
its filters from labels, but pooling remains fixed. Waveform arms retain the
source derivative passband (approximately 0.5–45 Hz); the spectral control retains
the earlier 1–30 Hz feature convention. Thus their comparison is between these
complete pipelines, not a perfectly isolated frequency-matched ablation.
Resampling is performed independently within each four-second window using
SciPy's default boundary padding.

## Models

1. Waveform TT-SVD: channel rank 6, terminal filter rank 8 or 16. The weighted
   delay covariance gives the same left factors as TT-SVD of the corresponding
   weighted stack of raw observations. This equivalence is numerically checked
   against TensorLy on synthetic waveforms.
2. Waveform Tucker/HOSVD: channel/lag ranks (4,4) or (6,6), with log pooled filter
   energies as classifier features.
3. Supervised TT: channel core 19 x 4, lag/filter core 4 x 16 x 8, and an 8 x 3
   softmax head. Cores and head are jointly optimized against class-weighted
   cross-entropy. Training-only unsupervised TT initializes the cores and feature
   scaling. Head L2 penalty is selected from 0.01 or 0.1; a fixed 0.001 quadratic
   anchor keeps cores near initialization. L-BFGS has a fixed 100-iteration
   budget. Objective, gradient, status and budget exhaustion are recorded;
   hitting that budget is not reported as mathematical convergence.
4. Spectral control: the same 418 feature definitions as V2, recomputed on the
   same full-cohort retained windows. Training-only ANOVA selects 16 or 32.

The first, second and fourth families use logistic regression (C=0.1 or 1) or
RBF SVM (C=1 or 10). There are 26 candidate configurations total. SVM uses
`predict()` for scoring; its decision values are not called probabilities.

## FTD imbalance and subject dependence

Each training person has weight `N_train / (3 * N_train_class)`; each diagnosis
has exactly one third of total training loss mass. This rule is recomputed in
each training fold. It is also used for pooled decomposition fitting and feature
scaling. ANOVA feature selection uses training subjects only.

Within a person, windows are averaged, not summed. Thus a person with a long
recording does not count as hundreds of independent training people. Equivalently,
each window contributes its person's class weight divided by their window count
to the pooled signal statistic. No oversampling, SMOTE, duplicated people or
synthetic FTD cases are used. Demographics, MMSE and duration are not predictors.

## Evaluation

The full procedure uses three repetitions of five-fold stratified outer
**subject** cross-validation (seed 9060), with three-fold inner subject CV
(seed 90600 + outer split index). Every stage fitted from data is refitted
inside its training split. Existing and leftover windows of the same person
never go to different folds.

Ranks, regularization and classifiers are selected inside inner CV. The primary
endpoint selects the entire family there too. Per-family outer scores remain
secondary exploratory comparisons. The machine-readable protocol and code/input
signature are saved before outcomes are examined. All outer fitted models,
membership, predictions, optimization status and inner candidate scores persist.

The learning curve is a separately prespecified diagnostic: fixed supervised
TT with head penalty 0.1; training fractions 0.4, 0.7 and 1.0 on nested stratified
subsets of each training set in the first five outer folds. The held-out people
stay fixed. It is not used to select the benchmark winner, and a single noisy
curve cannot establish a required sample size.

## Remaining limitations

These same source subjects have already been explored. No outer score here is
an independent external validation result. Changes versus V2 include cohort,
window coverage, representation and fold seed, so differences cannot be assigned
to one factor. Even the learning curve changes both participant count and total
recording exposure. Small samples and the fixed optimization budget limit claims.

This remains course-focused tensor representation/decomposition/classification
work. No claim of a newly invented TT method, guaranteed high accuracy, clinical
readiness or complete capture of raw waveform information is made.

## References

- [Oseledets: Tensor-Train Decomposition](https://epubs.siam.org/doi/10.1137/090752286)
- [Novikov et al.: Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569)
- [scikit-learn: learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)

The compact log-energy supervised architecture here is a project-specific
implementation inspired by tensor-factorized trainable layers, not a claimed
reproduction of a published dementia classifier.
