# Tensorization and decomposition benchmark report

**Date:** 2026-09-05  
**Input cohort:** 69 subjects, 23 AD / 23 FTD / 23 CN  
**Input windows:** 12,270 clean four-second windows

## Executive result

The full exploratory pipeline ran successfully on CPU. Forty-five combinations
were evaluated: five decomposition methods, three classifiers, and three ranks.

The strongest mean balanced accuracy was `0.541`, obtained by both rank-8 PCA
and rank-8 NMF with Logistic Regression. Chance balanced accuracy for three
balanced classes is `0.333`.

These results demonstrate that the complete pipeline works without subject
leakage, but the current classification performance is modest. The
graph-regularized model produced smoother channel factors but did not yet
outperform simpler PCA or NMF features.

## Tensor construction

The cleaned EEG was converted into a tensor with shape:

`69 subjects × 19 channels × 59 frequencies × 10 time bins`

For each original four-second window:

- Welch PSD used two-second Hann segments with 50% overlap.
- Frequency resolution was 0.5 Hz.
- Frequencies from 1 through 30 Hz were retained.
- Power was divided by total channel-frequency power within that window.
- The square root compressed the heavy-tailed distribution while maintaining
  non-negativity.

Windows were sorted chronologically within each subject and split into ten
rank-based time bins. The median power map in each bin became the tensor entry.
This ensured that every subject and time bin was populated despite different
recording durations.

The resulting tensor value range was approximately `0.00171–0.55796`. All
values were finite and non-negative.

## Decomposition methods

| Method | Description |
|---|---|
| PCA | Flattened linear decomposition baseline |
| NMF | Flattened non-negative matrix factorization baseline |
| Non-negative CP | Four-way non-negative canonical polyadic decomposition |
| Graph NCP | Non-negative CP with anatomical graph regularization on the channel factor |
| Non-negative Tucker | Four-way non-negative Tucker decomposition |

The graph used a fixed, symmetric neighborhood structure over the standard
10–20 electrodes. A normalized anatomical adjacency was used, with graph
regularization strength `lambda = 0.1`. Because this graph is fixed in advance,
it does not inspect diagnosis labels or held-out EEG.

CP methods used two initializations and up to 60 multiplicative-update
iterations per fold. Tucker used up to 40 iterations. Subject features from CP
and Tucker were obtained by non-negative least-squares projection onto the
training-fold basis.

## Classifiers

- Class-balanced Logistic Regression.
- Class-balanced RBF SVM.
- Class-balanced histogram gradient boosting.

An external XGBoost installation was attempted. The first wheel failed package
hash validation and the clean retry stalled. Rather than hold the experiment or
mislabel another model as XGBoost, scikit-learn histogram gradient boosting was
used and reported by its actual name.

## Validation design

Three repeats of stratified five-fold cross-validation were used. All splitting
occurred at subject level. Within every outer fold:

1. The decomposition was fitted on training subjects only.
2. Test subjects were projected onto the fixed training basis.
3. Feature scaling and classification were fitted on training features only.
4. Predictions were produced for the held-out subjects.

No window from a subject appears as an independent sample, because tensor rows
are already aggregated at subject level.

Ranks `3`, `5`, and `8` were evaluated. The comparison is an exploratory rank
sweep, not nested rank selection. Therefore, the best observed combination
should not be presented as an independently confirmed generalization estimate.

## Best result for each decomposition

| Decomposition | Best rank | Classifier | Mean balanced accuracy | Mean macro F1 |
|---|---:|---|---:|---:|
| PCA | 8 | Logistic Regression | 0.541 ± 0.036 | 0.525 ± 0.036 |
| NMF | 8 | Logistic Regression | 0.541 ± 0.055 | 0.525 ± 0.048 |
| Non-negative Tucker | 3 | Histogram gradient boosting | 0.536 ± 0.038 | 0.509 ± 0.030 |
| Non-negative CP | 5 | Logistic Regression | 0.522 ± 0.052 | 0.510 ± 0.052 |
| Graph NCP | 5 | Logistic Regression | 0.517 ± 0.047 | 0.507 ± 0.049 |

The rank-8 NMF run initially reached its 1,000-iteration budget in some folds.
It was independently repeated with a 5,000-iteration budget. The convergence
warnings disappeared and the Logistic Regression result remained unchanged at
`0.541 ± 0.055`, confirming that the reported NMF result was not caused by the
earlier iteration cap.

## Class-level behavior

For the rank-8 PCA plus Logistic Regression model, probabilities were averaged
across the three repeated held-out predictions for each subject. The resulting
confusion matrix was:

| True / predicted | AD | FTD | CN |
|---|---:|---:|---:|
| AD | 12 | 7 | 4 |
| FTD | 8 | 7 | 8 |
| CN | 2 | 4 | 17 |

Corresponding sensitivities were:

- AD: `0.522`.
- FTD: `0.304`.
- CN: `0.739`.

FTD is the main failure mode. It is confused with both AD and CN. This is more
important than the small differences among decomposition methods and should
guide the next iteration.

Rank-8 NMF plus Logistic Regression produced a similar confusion pattern:

| True / predicted | AD | FTD | CN |
|---|---:|---:|---:|
| AD | 12 | 7 | 4 |
| FTD | 7 | 8 | 8 |
| CN | 2 | 5 | 16 |

## Reconstruction and graph behavior

Mean relative squared reconstruction error decreased with rank, as expected:

| Rank | Non-negative CP | Graph NCP | Non-negative Tucker |
|---:|---:|---:|---:|
| 3 | 0.11157 | 0.11157 | 0.11999 |
| 5 | 0.08586 | 0.08589 | 0.10222 |
| 8 | 0.07230 | 0.07246 | 0.09548 |

At rank 8, graph regularization reduced mean channel-factor graph roughness from
`0.8865` to `0.8403`, approximately a 5% reduction, with almost no reconstruction
penalty. The regularizer is doing what it was designed to do geometrically, but
that additional smoothness did not improve classification with `lambda = 0.1`.

## Interpretation

The current representation separates CN more reliably than FTD. This suggests
that broad relative spectral structure contains useful healthy-versus-dementia
information, while AD-versus-FTD separation requires more specific spatial,
connectivity, temporal-variability, or absolute-power information.

The result does **not** show that tensor decomposition is useless. It shows that:

- the present relative-power summary is strongly compressed;
- a single fixed graph strength is not sufficient evidence for graph benefit;
- rank-8 linear/non-negative flattened baselines remain competitive;
- the FTD boundary is the limiting modeling problem;
- classifier optimization should remain secondary to improving the EEG
  representation and validation design.

## Recommended next experiments

1. Perform nested rank and graph-lambda selection inside subject-level folds.
2. Add an absolute-power sensitivity tensor using training-fold robust scaling.
3. Compare median-only time bins with median-plus-variability features.
4. Run the planned 1–45 Hz sensitivity tensor and explicitly audit the three
   persistent high-frequency FTD subjects.
5. Compare the anatomical graph with a training-only phase-connectivity graph.
6. Run a binary hierarchy: dementia versus CN, followed by AD versus FTD.
7. After the balanced benchmark is stable, run the virtual subject-balanced
   sampler using all QC-approved leftover data.
8. Use nested validation or an untouched evaluation cohort before presenting a
   single selected model as final performance.

## Output files

- `all_rank_results.csv`: all 45 rank/method/classifier combinations.
- `summary.json`: consolidated numerical results and decomposition diagnostics.
- `balanced_accuracy_heatmap.png`: full comparison heatmap.
- `rank_sensitivity_logistic.png`: rank effect for Logistic Regression.
- `selected_confusion_matrices.png`: selected confusion matrices.
- `../tensorization/tensorization_report.json`: tensor construction details.
- `../benchmark_rank_sweep/rank_*/benchmark_report.json`: complete fold results.
- `../nmf_rank8_convergence_check/benchmark_report.json`: extended NMF check.

