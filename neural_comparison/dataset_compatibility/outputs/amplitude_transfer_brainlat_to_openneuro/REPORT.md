# BrainLat amplitude-only SVM → all OpenNeuro participants

This is the exact one-feature pipeline that achieved97.1% mean macro F1 in internal BrainLat nested CV, now refitted on all86 BrainLat participants and tested on all88 OpenNeuro participants. The97.1% describes the source validation procedure, not a single previously saved fitted model.

One feature: log10 median19-channel RMS after the same fixed256Hz/CAR/1–40Hz/8s-window preparation. Class labels: HC versus AD+FTD. No class-specific scaling, site correction, target statistics, target tuning or target calibration.

Selected entirely within BrainLat: `{'kernel': 'linear', 'C': 0.01, 'threshold': 0.0}`. Source-only3-fold parameter/threshold selection; final scaler,1D PCA and SVM fitted on86 source participants. Probability calibration uses source-only nested5-fold OOF scores. The complete source model was saved and hashed before target features entered this script.

| Metric | BrainLat internal nested CV mean | OpenNeuro full test set |
|---|---:|---:|
|macro_f1|97.1%|40.1%|
|balanced_accuracy|97.2%|50.0%|
|accuracy|97.3%|67.0%|
|healthy_recall|96.9%|0.0%|
|dementia_recall|97.5%|100.0%|

| Actual / predicted | HC | Dementia |
|---|---:|---:|
|HC|0|29|
|Dementia|0|59|

Target predictions: {'HC': 0, 'dementia': 88}. The test contains29 HC,36 AD and23 FTD; original AD/FTD diagnoses are preserved in predictions.json, but this model outputs only HC and combined AD/FTD probabilities.

A drop in transfer performance means the source amplitude rule did not generalize under this matched processing policy. It cannot identify whether biology, recruitment, acquisition, export/cleaning, interpolation or implementation differences caused the distribution shift, and provides no evidence of intentional author manipulation.

Both cohorts were explored before this experiment. OpenNeuro is outside all fitting and parameter selection in this experiment; it is not an untouched prospective cohort. Source-calibrated probabilities may be poorly calibrated on the target. Hard classifications use the source-selected SVM threshold, which need not correspond to probability0.5.

Each test participant is evaluated once. This test has88 distinct people, not repeated windows as independent test cases. A 50% balanced-accuracy score corresponds to chance; macro F1 can differ from50% under class imbalance.

Verification passed: disjoint cohorts, source OOF predictions reproduced, unchanged frozen model hash after target evaluation, valid probabilities, batch-independent inference, identical predictions after reload.

![Transfer results](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/amplitude_transfer_brainlat_to_openneuro/transfer_results.png)
