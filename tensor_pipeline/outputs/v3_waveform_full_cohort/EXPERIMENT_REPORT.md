# Full-cohort waveform tensor experiment: results

Date: 2026-09-06. All previous data partitions, models and reports are preserved.

## Outcome

Using the leftovers and class/subject weighting did not produce the hoped-for waveform accuracy improvement. The primary nested method-selection endpoint is **52.0% balanced accuracy**. Waveform methods range from **49.9% to 51.8%**, while the same-cohort spectral control achieves **55.8%**. This experiment does not establish a maximum attainable accuracy for this dataset.

| Pipeline | Balanced accuracy ± repeat SD | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Inner-selected entire pipeline (PRIMARY) | 52.0% ± 3.2 pp | 53.4% | 0.516 |
| Spectral control, same 88 people | 55.8% ± 2.7 pp | 56.8% | 0.553 |
| Waveform Tucker + LR/SVM | 51.8% ± 3.4 pp | 52.3% | 0.513 |
| Waveform TT + LR/SVM | 50.9% ± 6.2 pp | 51.5% | 0.504 |
| Supervised waveform TT | 49.9% ± 2.9 pp | 52.3% | 0.499 |

All scores are subject-level, from three repeats of five-fold outer CV with three-fold inner tuning. Repeat SD is descriptive, not a confidence interval. Balanced accuracy gives each diagnosis equal weight. With 36/23/29 people, ordinary accuracy is no longer identical to balanced accuracy. Always predicting AD would give 40.9% accuracy but only 33.3% balanced accuracy.

The primary row tests selecting a complete pipeline using only training-side validation. Per-family rows are secondary comparisons; selecting the best after inspecting outer scores is not an independently confirmed performance estimate.

## Data actually used

| Diagnosis | People | Four-second windows | Retained duration |
|---|---:|---:|---:|
| AD | 36 | 7,030 | 7 h 48 min 40 s |
| FTD | 23 | 4,000 | 4 h 26 min 40 s |
| CN | 29 | 5,794 | 6 h 26 min 16 s |

- Total: 88 people and 16,824 windows (18 h 41 min 36 s).
- Included 4,828 windows from the old leftover partition and recovered 54 valid balance-only trims.
- All 11,942 V2 windows remained; no former primary artifact exclusion was reinstated.
- Applied the same reference, amplitude/step/flatness rules and boundary guards to both source partitions.
- Excluded 595 windows with IDs/reasons retained. No signals were deleted; partial tails remain separate.

FTD imbalance is handled by training-fold class weights, not duplication. Each diagnosis has one third of training loss weight, and each subject contributes one pooled observation. More windows improve that person's pooled estimate but do not make them count as more independent people.

## What the waveform methods tested

Cleaned EEG is resampled to 125 Hz and represented as channel × lag × local-position tensors. TT/Tucker filters measure short-lag channel/time relationships; squared responses are pooled across positions and windows. An exact covariance identity makes this computationally small enough for CPU. The waveform branches do not use Fourier/PSD features.

The supervised TT branch jointly learns its channel and lag/filter cores and classification head from weighted diagnosis loss. It is a compact, second-order waveform model—not a general long-sequence neural network. Fixed pooling still discards window ordering and higher-order waveform detail. Its failure to beat the spectral control does not rule out all raw-signal models.

Waveform branches retain approximately 0.5–45 Hz source content, while the spectral control uses the earlier 1–30 Hz convention. This is a pipeline comparison, not a perfectly bandwidth-matched ablation.

## Recall by diagnosis

| Pipeline | AD recall | FTD recall | CN recall |
|---|---:|---:|---:|
| Inner-selected entire pipeline (PRIMARY) | 54.6% | 34.8% | 66.7% |
| Spectral control, same 88 people | 56.5% | 42.0% | 69.0% |
| Waveform Tucker + LR/SVM | 48.1% | 40.6% | 66.7% |
| Waveform TT + LR/SVM | 48.1% | 39.1% | 65.5% |
| Supervised waveform TT | 59.3% | 26.1% | 64.4% |

## Does adding more people help?

A prespecified secondary learning curve used one fixed supervised TT configuration (head penalty 0.1), the first five outer folds, and nested stratified subsets of each training set. No curve outcome was used to choose model settings. Each point below pools held-out predictions for all 88 people once.

| Training fraction | Mean training people per fold | Training balanced accuracy | Held-out balanced accuracy |
|---|---:|---:|---:|
| 40% | 27.0 | 100.0% | 42.0% |
| 70% | 48.2 | 98.7% | 40.9% |
| 100% | 70.4 | 92.9% | 55.2% |

Held-out performance did not increase steadily with training size. This small, noisy curve does not demonstrate a reliable data-scaling trend or establish how many additional people are needed.

The original cohort still contains only 23 independent FTD people. Leftover windows do not add new FTD participants. Broader independent participants and a separate external test cohort remain distinct needs; oversampling cannot create that independence.

## Optimization and verification

- Replayed all 90 saved model entries: 75 outer pipeline entries (including selected-pipeline aliases) and 15 learning-curve models.
- Verified all 15 outer and 45 inner subject splits, training-only choices, class loss weights, bases and scores.
- Recomputed waveform moments and spectral controls from every retained window of four selected subjects, including new AD/CN people and an FTD recording with many cuts; checked representative windows against source EEGLAB signals.
- Confirmed source array SHA-256 hashes unchanged, unique-window accounting, no test/training overlap, finite positive-semidefinite moments, and retention of every V2 window.
- Numerical tests verify raw squared-response/covariance equivalence, TT-SVD reconstruction against TensorLy, orthonormality, batch-invariant projection, analytic supervised gradients and class weight totals.
- Nested fitting time: 68.5 seconds on CPU, excluding preparation, curve and verification. No new packages, CUDA or virtual environment were used.

Supervised optimizer status: 0/120 fits reported convergence under the fixed 100-iteration budget. All fits reduced their objective from initialization. Status counts: {"STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT": 120}.

**Budget-limited optimization is a remaining limitation**, not proof of a globally optimal fit. L-BFGS solves a nonconvex factorized problem; even its convergence flag would not certify a global optimum. No iteration budget was changed in response to outer-test accuracy.

A separate [fixed-budget convergence sensitivity](convergence_sensitivity/REPORT.md) compares 100 versus 2000 iterations for the already specified penalty-0.1 model. It was added after the original run because of the optimizer status flags; it is post-hoc, reports both outcomes, and does not replace this primary result.

## Interpretation and course reporting

The tested waveform representations did not beat the spectral control on this expanded cohort. This is useful negative evidence about these specific tensor filters and pooling choices; it is not evidence that tensor methods or raw EEG cannot work.

Do not call this a controlled improvement over V2: cohort, window coverage, representation and fold seed changed. These same source people have been explored repeatedly, so nested fitting does not create an untouched external evaluation cohort. Class weighting addresses training imbalance but cannot manufacture diagnostic signal, remove every acquisition confound or guarantee higher FTD recall.

No further model search was performed after inspecting these outcomes. Preserve the full result table for the course, including the spectral control and the primary nested-selection endpoint.

## Files

- [Detailed protocol](../../v3_waveform/PROTOCOL.md)
- [Reproduction instructions](../../v3_waveform/README.md)
- `preparation_report.json`: exclusions, cohort accounting and source fingerprints.
- `retained_windows.json`: every included sample ID and its original partition/index.
- `protocol.json`, `fold_XX.json`, `nested_report.json`: fixed search, splits and all outcomes.
- `models/fold_XX.joblib`: complete outer models, factors, scalers and heads with subject IDs.
- `learning_curve.json` and `models/curve_XX_YYY.joblib`: fixed-model size diagnostic.
- `verification_report.json` and `artifact_manifest.json`: checks and replay fingerprints.
