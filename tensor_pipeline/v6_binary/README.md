# Binary dementia screening experiment

Run from the project root with `py -3.10 -u tensor_pipeline/v6_binary/run.py`.
The script reads the existing V3 subject index and spectral features plus V5
one-bag power/connectivity features. It writes only to
`tensor_pipeline/outputs/v6_binary/`.

The primary task matches the label definition in Mootoo et al. (2023):
**AD + FTD (59 people) versus cognitively normal controls (29 people)**.
Their preprint reports 85% accuracy for its best SVM binary experiment:
<https://doi.org/10.1101/2023.11.01.23297940>. Our features and validation
procedure differ, so the figures are context, not a head-to-head replication.

There are 36 fixed candidate pipelines using spectral, band-power, or
connectivity subject summaries, training-only ANOVA feature selection,
standardization, and class-balanced logistic regression or RBF SVM. Three-fold
inner participant CV selects the whole pipeline by **macro F1**, then dementia
recall, then balanced accuracy. The untouched outer partitions use three
repetitions of five-fold stratified participant CV. Each person appears once per
repeat in the outer test set. No threshold tuning uses test labels.

## Results

Means across three complete 88-person outer CV repetitions:

| Measure | Mean | Repeat SD |
|---|---:|---:|
| Macro F1 (selection priority) | 0.718 | 0.021 |
| Dementia F1 | 0.813 | 0.004 |
| Dementia recall | 80.8% | 2.0 pp |
| Dementia precision | 81.8% | 2.5 pp |
| CN specificity | 63.2% | 7.2 pp |
| Balanced accuracy | 72.0% | 2.6 pp |
| Ordinary accuracy | 75.0% | 1.1 pp |

The always-dementia baseline has 67.0% accuracy, 0.401 macro F1, 0.803
dementia F1, and 100% dementia recall, but zero CN specificity. This is why
positive-class F1 or recall alone would be misleading on the 59:29 class split.
Our macro F1 and specificity show a real gain over that trivial rule, while the
85% accuracy reference was not reached.

The three repeat confusion matrices, in `TN, FP, FN, TP` order, are
`[16,13,10,49]`, `[20,9,12,47]`, and `[19,10,12,47]`. Per-fold predictions,
inner candidate scores, selected settings, input hashes, and protocol are saved
in the output directory. An independent check recomputed each repeat's metrics
from saved predictions and confirmed disjoint training/test people and complete
88-person test coverage per repeat.

This is internal nested validation on a cohort explored in V1–V5. The new
binary task was motivated by published and prior project results; it is not
external confirmation or a clinical performance estimate.
