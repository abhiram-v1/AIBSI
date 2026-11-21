# BrainLat-only native-electrode ablation

86 fixed participants:35AD,19FTD,32HC. Binary HC versus AD+FTD. 3 repeats x5 outer folds; 3 inner folds.
All arms use identical participants, epochs, folds, native512Hz,1–40Hz filtering, common128-channel average reference and common native-window gain normalization.
Fixed spectral settings:8-second segment,1-second STFT,1-second hop,32mel bands; mean/SD/median/IQR. Same256-candidate SVM/preprocessing grid in every arm.

| Arm | Macro F1 | Balanced accuracy | Accuracy | HC recall | Dementia recall |
|---|---:|---:|---:|---:|---:|
| native128 | 0.6143 | 0.6165 | 0.6357 | 0.5417 | 0.6914 |
| native19 | 0.5655 | 0.5677 | 0.5930 | 0.4688 | 0.6667 |
| legacy19 | 0.5987 | 0.6020 | 0.6202 | 0.5312 | 0.6728 |
| aligned19 | 0.5538 | 0.5554 | 0.5775 | 0.4688 | 0.6420 |

## Interpretation limits

- Legacy19 reproduces the old spatial operator, but shares the native128 reference and normalization in this experiment. It is not an exact rerun of the earlier mixed-cohort production pipeline.
- Aligned19 uses both layouts in head coordinates with the same source-sphere origin subtracted before the spline calculation. It retains the same spline regularization0.
- The installed MNE legacy helper used the target montage coordinates directly, while source coordinates were head-frame and source-origin centered. Source inspection motivated the aligned diagnostic arm before seeing scores.
- Native19 sensors approximate target positions; different physical locations mean native19-versus-interpolated19 is not a perfectly isolated interpolation-only intervention.
- Native128 changes spatial coverage and feature dimensionality together. Every arm has the same nested tuning budget; differences compare complete selected procedures.
- Native reference and per-window log-mel gain use all128 measured channels in every arm. Native19 means19 analysis channels after that common preprocessing, not a standalone19-electrode acquisition.
- Common epochs pass quality checks on all128 native channels. No arm-specific epoch rejection or subject dropping is permitted.
- The86-person cohort inherits previous inclusion decisions. This is not all originally available BrainLat recordings, not an untouched external test, and not clinical validation.
- Site-specific results describe participants held out within AR/CL, not generalization to an unseen site.
- Paired bootstrap intervals retain each person's repeat predictions, conditional on fitted models; they do not include all training uncertainty or adjust for multiple comparisons.
- Final full-cohort probability models are inference artifacts and do not supply test-performance evidence.
- Monotone sigmoid calibration is fitted inside training only. When its slope reaches zero, probabilities revert to training prevalence and no probability-space operating threshold is defined. Main classification metrics still use the selected SVM score threshold.
