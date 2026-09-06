# NeuroTensor dementia EEG project: complete record through 2026-09-06

## Purpose and status

This is the consolidated record of the dataset preparation, tensorization,
classification experiments, corrections, and verification completed in this
workspace. It supplements rather than replaces the versioned reports and
artifacts listed below. No original BIDS files, earlier derived datasets, or
earlier fitted models were deleted or overwritten.

The work explores subject-level classification of Alzheimer's disease (AD),
frontotemporal dementia (FTD), and cognitively normal controls (CN) from
eyes-closed resting EEG. It is course/research work, not a clinical diagnostic
system. The current evidence does **not** establish an accuracy ceiling, a
65–70% performance level, diagnostic validity, or external generalization.

## Executive snapshot

| Item | Current documented result |
|---|---|
| Original source | OpenNeuro `ds004504`, local v1.0.9; 88 people: 36 AD, 23 FTD, 29 CN |
| Controlled primary cohort | 69 people, exactly 23 per diagnosis; 12,270 clean four-second windows |
| Full-cohort experiment | 88 people; 16,824 eligible four-second windows after consistent re-QC |
| First exploratory tensor benchmark | PCA/NMF + logistic regression: 54.1% balanced accuracy; exploratory comparison |
| Corrected nested TT/Tucker benchmark | Best observed Tucker/HOSVD: 56.0%; primary automatic selection: 47.3% |
| Full-cohort waveform benchmark | Spectral control: 55.8%; waveform methods: 49.9–51.8%; primary selection: 52.0% |
| Bounded tuning benchmark | Best observed RBF SVM: 59.7%; XGBoost: 57.6%; primary complete selection: 54.5% |
| Main weakness | FTD recall remains substantially lower than AD/CN in stronger comparisons |
| Independent validation | Not available; all source people have now been explored in several analyses |

Balanced accuracy is the main metric because classes differ in the 88-person
analysis. It averages AD, FTD, and CN recall; always predicting one diagnosis
has 33.3% balanced accuracy.

## Source dataset and supplied preprocessing

The original data are preserved at `dataset/ds004504`: 19 scalp electrodes,
500 Hz sampling, eyes-closed resting EEG. Source recording totals are 29,403.3
seconds for AD, 16,753.4 for FTD, and 24,433.6 for CN; total 70,590.3 seconds /
35,295,150 samples.

We started from the dataset authors' artifact-cleaned EEGLAB derivatives. They
already have approximately 0.5–45 Hz filtering, A1–A2 mastoid reference, ASR,
ICA, and automatic ICLabel artifact-component rejection. We did not repeat ASR,
ICA, broad filtering, or add a 50 Hz notch: residual 50 Hz was negligible after
the supplied 45 Hz low-pass. The source raw EEG is untouched.

The project proposal PDF informed the objective and tensor workflow. It was
treated as context, not as an instruction to bypass quality control,
subject-level splitting, or reproducibility safeguards.

## Dataset construction chronology

### 1. Duration inventory and first balanced dataset

All 88 recordings were ranked longest to shortest using non-destructive views
under `preprocessing/01_duration_sorted`. Longest: AD `sub-010` (1,291.1 s).
Shortest: AD `sub-003` (307.1 s). FTD was the limiting diagnosis by participant
count and usable coverage.

Complete non-overlapping four-second windows were downsampled from 500 to 250 Hz
by retaining alternate samples, producing 19 × 1,000 arrays. This is compatible
with the supplied 45 Hz low-pass. Per-channel temporal means were removed;
incomplete end fragments were separately preserved.

Initial conservative QC found 7,252 approved AD windows, 4,132 FTD, and 6,007
CN. All 23 FTD people became the reference. Matching selected 23 AD and 23 CN
people with broadly comparable duration, age and sex coverage. FTD duration
buckets were short (<=647.6 s, 8 people), medium (647.6–806.45 s, 7), and long
(>806.45 s, 8). AD/CN selection spread windows over the recording.

The first balanced dataset is 12,396 windows: 4,132/class, 69 people, 49,584 s,
at `preprocessing/02_balanced_primary/primary`. The non-overlapping leftover
set has 5,023 complete windows (AD 3,135, FTD 5, CN 1,883); 4,995 passed initial
conservative QC. Eighty-eight incomplete tails totaling 155.12 seconds are
separately preserved. Nothing was discarded.

### 2. Signal audit and clean controlled primary cohort

The first balanced set had no material nonfinite/repeated-value or residual 50
Hz issue. Across-channel common variance was high (median about 95.3%), but this
is not proof of artifact/bridging; it is consistent with mastoid reference and
volume conduction. We common-average re-referenced 19 scalp channels at each
sample, then removed each channel's window mean.

Fixed label-blind post-reference exclusions were amplitude >=400 µV, adjacent
4 ms step >=150 µV, or any channel SD <1 µV. This removed 65 windows (AD 9,
FTD 42, CN 14), concentrated in FTD `sub-086`; that person was retained.
Persistent 30–45 Hz activity in FTD `sub-067`, `sub-077`, and `sub-085` was
recorded but not used for removal, avoiding label-dependent filtering.

FTD then limited clean windows to 4,090. Thirty-three AD and 28 CN medium-bucket
windows were moved as balance-only trims, not artifacts. The clean controlled
primary is 12,270 windows / 69 people / 4,090 per diagnosis at
`preprocessing/04_clean_primary/primary`. Its 126-window excluded partition
contains 65 genuine QC exclusions and 61 balance-only trims, exactly 42/class,
at `preprocessing/04_clean_primary/excluded`. Every first-stage primary window
is in exactly one partition.

### 3. Later input corrections

The second tensor experiment verified the actual source `chanlocs` row order:
`Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz`.
The first tensorizer attached a different hard-coded label order to unchanged
rows. Therefore its anatomical graph and electrode-specific interpretation are
wrong. Channel-permutation-invariant operations, such as ordinary PCA/global
statistics/CAR, are not thereby invalidated.

EEGLAB `boundary` annotations were also found. V2 applies a fixed 0.5 s guard
around each boundary, excluding 328 from its analysis index and leaving 11,942
windows: AD 3,985, FTD 4,000, CN 3,957. All 69 people remain. Stored primary
arrays were not edited.

## Safeguards common to all current experiments

- Split by **participant**, never by window. Primary and leftover windows of a
  person cannot enter different train/test or inner-validation groups.
- Fit decompositions, selection, scaling, classifiers and class weights only on
  training people.
- Pool within person. Long recordings do not represent additional independent
  patients; full-cohort training gives each class equal total weight.
- No SMOTE/duplication/synthetic FTD data, MMSE, age, sex, diagnosis, duration,
  or window count are predictors.
- Report subject-level balanced accuracy, ordinary accuracy, macro F1, confusion
  matrices and class recall. Repeated-CV SD is descriptive, not a CI.
- Saved `joblib` artifacts are trusted-local only.

## Modeling chronology and outcomes

### Experiment 1 — exploratory non-negative spectral tensor benchmark

Input: controlled 69-person clean primary cohort, before boundary guarding.
Representation: `69 × 19 × 59 × 10`, subject × channel × 1–30 Hz (0.5 Hz) ×
rank-based recording-time decile. Four-second windows used two-second Hann/Welch
with 50% overlap. Relative power was square-root transformed for non-negativity;
each decile used median maps.

PCA, NMF, non-negative CP, graph NCP, and non-negative Tucker at ranks 3/5/8
were combined with logistic regression, RBF SVM, and sklearn histogram gradient
boosting under 3×5 subject CV. Best exploratory PCA rank 8 + logistic and NMF
rank 8 + logistic both scored 54.1% balanced accuracy. NMF was rerun to 5,000
iterations with the same result. Best reported method figures: Tucker 53.6%, CP
52.2%, graph NCP 51.7%. FTD recall was around 30% for the cited PCA result.

This rank/model comparison is not nested: its maximum is exploratory. A
feature-bottleneck diagnostic with less-compressed channel/band/time features
reached 57.0%, motivating representation work but not proving a causal loss.
The old graph-NCP update/selection objective had inconsistent scaling of
reconstruction and graph penalties under factor rescaling. Its results remain
preliminary; its code was not silently recast as validated graph regularization.

### Experiment 2 — corrected nested TT/Tucker benchmark (V2)

Input: corrected channels plus boundary-safe 69-person index (11,942 windows).
Each window is a signed log10 absolute-PSD `19 × 59 × 3` tensor: channel ×
1–30 Hz × three overlapping local two-second segments. Subject predictors pool
projected windows by mean/SD/median/IQR. A second subject tensor uses mean,
median and SD log power. No demographic/clinical predictors are included.

Methods: subject TT, window TT, graph-pre-smoothed TT, window Tucker/HOSVD,
TT + spectral controls, and spectral controls. TT ranks 8/16/32; Tucker (6,8,3).
Logistic, linear/RBF SVM and histogram gradient boosting were candidates. The
protocol uses nested 3-repeat 5-fold outer / 3-fold inner subject CV to select
rank, features, classifier, and method family. SVM uses `predict()`, not
probability argmax. Fixed anatomical graph pre-smoothing is clearly distinct
from old graph-CP regularization.

| V2 result | Mean balanced accuracy |
|---|---:|
| Primary automatic full-pipeline selection | 47.3% |
| Strongest observed family: Tucker/HOSVD | 56.0% |
| TT + spectral features | 54.1% |
| Graph-smoothed TT | 53.1% |
| Spectral control | 51.7% |
| Window TT | 51.2% |
| Subject TT | 50.2% |

V2 numerical tests passed TT reconstruction against TensorLy, orthogonality,
full-rank recovery and batch-invariant projection. Post-run verification replayed
105 classifier pipelines, 60 decomposition sets, 45 inner and 15 outer splits;
held-out features were recomputed from saved bases with matching predictions and
scores. Source hashes were unchanged; no test person entered basis calibration.

### Experiment 3 — full-cohort waveform tensor analysis (V3)

User-approved scope: use primary plus leftover windows without modifying either
source partition. A new 88-person analysis index re-applies CAR/QC/boundary
rules consistently. It retains 16,824 windows: AD 7,030, FTD 4,000, CN 5,794.
It includes 4,828 old-leftover windows and 54 valid prior balance-only trims,
retains all V2 windows, and does not reintroduce prior artifact exclusions.
There are 595 recorded V3 exclusions.

Waveform representation: clean 19 × 1,000 data are anti-alias resampled to
125 Hz then represented as `19 × 16 × 485` channel × lag × local-position delay
tensors (120 ms lag span). TT/Tucker filters produce squared response energy
pooled per person. Per-person delay covariance is an exact sufficient statistic
for mean squared linear-filter response. This is raw-waveform second-order
information, not PSD conversion or a general long-sequence network.

V3 compared waveform TT, waveform Tucker, supervised TT (joint cores/head),
and same-cohort spectral controls with nested 3×5 outer / 3-fold inner subject
CV and class/subject weighting.

| V3 result | Mean balanced accuracy |
|---|---:|
| Primary automatic full-pipeline selection | 52.0% |
| Spectral control, same 88 people | 55.8% |
| Waveform Tucker | 51.8% |
| Waveform TT | 50.9% |
| Supervised waveform TT | 49.9% |

Original supervised TT hit its 100-iteration cap in all 120 uses. A clearly
labeled post-hoc fixed-model sensitivity used identical splits: 100 iterations
averaged 55.0%, 2,000 averaged 53.7%, with only 3/25 long runs reporting
convergence. It is a diagnostic, not a selected winner. One small noisy learning
curve with approximately 27/48/70 training people gave 42.0%/40.9%/55.2%
held-out balanced accuracy; it cannot estimate required sample size.

V3 checks passed source reconstructions, all saved prediction replays, split
membership, class-weight totals, V2-window preservation and unchanged arrays.

### Experiment 4 — bounded classifier/optimization tuning (V4)

V4 used exactly V3's 88-person inputs and outer splits. This makes V3/V4 a
cleaner procedure comparison, but still reuses people already explored.
`xgboost-cpu==3.2.0` was installed into the existing Python 3.10 user install;
no virtual environment, PyTorch, CUDA runtime or dependency upgrade was used.

Before outcomes were inspected, 139 configurations were frozen: wave TT ranks
8/16, waveform Tucker ranks 4/6, spectral 16/32 selected features, TT16+
spectral fusion; each paired with 19 logistic/RBF SVM/eight conservative XGBoost/
two random-forest settings; plus six stronger-regularized supervised-TT models.
XGBoost uses shallow CPU histogram trees and fixed conservative depth, rate,
tree-count, minimum-child-weight, L1/L2, and subsampling combinations. There is
no test-driven early stopping. All choices are inner-CV only.

| V4 classifier-family result | Mean BA ± repeat SD | FTD recall |
|---|---:|---:|
| Primary automatic full-pipeline selection | 54.5% ± 3.2 pp | 42.0% |
| Strongest observed: tuned RBF SVM | 59.7% ± 0.9 pp | 43.5% |
| Tuned random forest | 58.3% ± 2.9 pp | 39.1% |
| Tuned XGBoost | 57.6% ± 1.9 pp | 39.1% |
| Tuned logistic regression | 54.6% ± 1.1 pp | 39.1% |
| Stronger-regularized supervised TT | 50.9% ± 2.4 pp | 30.4% |

Representation families, with inner classifier selection: TT/spectral fusion
59.1% ± 3.6 pp, spectral 56.8% ± 1.2 pp, waveform Tucker 52.0% ± 2.3 pp, and
waveform TT 46.8% ± 11.7 pp. Fusion begins with TT + spectral features but
training-only selection can favor spectra, so it is not proof of a tensor gain.

V4's primary automatic selection rose nominally +2.5 percentage points versus
V3 (52.0% to 54.5%) on the same input/splits; this is not independently
confirmed improvement. The 59.7% SVM is a secondary strongest-family result,
not the primary automatic-selection estimate. Classifier selections across 15
outer fits: SVM 7, logistic 5, random forest/XGBoost/supervised TT 1 each.
Fusion and spectral were selected 7 times each; supervised TT once.

Stronger TT regularization improved optimization status: 200/285 supervised
fits reported convergence; 85 reached the 500-iteration cap. It remains
nonconvex; convergence is not proof of a global optimum. XGBoost was useful to
test, but it was not the strongest candidate in this bounded comparison.

V4 verification passed: 150 outer model entries replayed, 60 bases checked,
45 inner and 15 outer split memberships verified, training-only scaling/weights
reproduced, and 23 XGBoost boosters exported/reloaded with matching outputs.
Original primary and leftover array hashes remained unchanged.

## Cross-experiment interpretation

1. Tensor decomposition is not automatically superior to spectral features.
   V3 spectral controls beat waveform tensor branches; V4 fusion/SVM is
   encouraging but does not establish that tensor factors caused the gain.
2. FTD remains the hardest diagnosis. More windows do not add independent FTD
   people: the full source still has only 23 FTD participants.
3. More independent people may help, but the small noisy V3 curve cannot give a
   required sample size. New FTD participants and an external cohort are more
   valuable than duplicating existing FTD windows.
4. More iterations alone did not improve the V3 supervised TT fixed model.
   Stronger regularization improved optimizer status but not that family's rank.
5. Experiments differ in cohort/boundary rule/representation/features and some
   fold settings. Do not claim one algorithm alone caused cross-version changes.
6. Nested CV protects each run's fitting/tuning but cannot erase researcher-level
   adaptive exposure to the same source participants. Do not search repeatedly
   until a desired outer score appears.

## Report and artifact map

| Topic | Main report / artifact |
|---|---|
| Source, duration inventory, matching | [preprocessing index](preprocessing/reports/README.md) |
| Controlled cleaning/exclusions | [signal audit](preprocessing/reports/03_signal_quality_audit_and_cleaning.md) |
| First exploratory benchmark | [report](tensor_pipeline/outputs/summary/EXPERIMENT_REPORT.md) |
| Old graph/mapping caveats | [V2 audit note](tensor_pipeline/outputs/summary/V2_AUDIT_NOTE.md) |
| Corrected nested TT/Tucker | [V2 result](tensor_pipeline/outputs/v2_tensor_train/EXPERIMENT_REPORT.md) · [protocol](tensor_pipeline/v2/AUDIT_AND_PROTOCOL.md) |
| Full-cohort waveform | [V3 result](tensor_pipeline/outputs/v3_waveform_full_cohort/EXPERIMENT_REPORT.md) · [protocol](tensor_pipeline/v3_waveform/PROTOCOL.md) |
| V3 optimizer sensitivity | [report](tensor_pipeline/outputs/v3_waveform_full_cohort/convergence_sensitivity/REPORT.md) |
| Bounded SVM/XGBoost/RF/tensor tuning | [V4 result](tensor_pipeline/outputs/v4_tuning/EXPERIMENT_REPORT.md) · [protocol](tensor_pipeline/v4_tuning/PROTOCOL.md) |
| V4 configuration choices/checks | [choices](tensor_pipeline/outputs/v4_tuning/selected_configurations.json) · [verification](tensor_pipeline/outputs/v4_tuning/verification_report.json) |

Each versioned output folder has machine-readable protocol, held-out membership,
predictions, saved models/bases, and fingerprints. The 88-person V3/V4 results
must not silently replace the original balanced 69-person controlled benchmark.

## Current stopping point

Preprocessing, tensor decomposition, waveform modeling, leftover/full-cohort
use, class weighting, optimization sensitivity, and bounded SVM/XGBoost/random
forest/tensor tuning are complete and documented. No unstarted experiment is
implied here. Any future work needs a new explicitly approved protocol, ideally
with external people or an untouched external test cohort.
