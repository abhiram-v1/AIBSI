# Graph-NTF experiment results

Profile: **quick**; outer=stratified; participants=174.
All reported predictions hold participants out from tensor fitting, tuning and calibration.
Quick runs are bounded development checks; full nested search is a separate command.
Confidence intervals bootstrap participants within class, keeping repeat predictions together. They are conditional on fitted models.

| Method | Accuracy | Macro F1 | Balanced accuracy | Log loss |
|---|---:|---:|---:|---:|
| direct | 0.575 | 0.484 | 0.517 | 0.967 |
| ntf | 0.523 | 0.404 | 0.459 | 1.020 |
| gntf | 0.540 | 0.456 | 0.486 | 1.009 |
| shuffled | 0.534 | 0.406 | 0.470 | 1.026 |
| tt | 0.552 | 0.457 | 0.493 | 0.974 |

Outer tensor fits without any optimizer-converged initialization: 14. Inspect each fold objective history before drawing decomposition conclusions.
The real and shuffled graph arms also tune graph strength: their difference tests complete procedures, not one fixed graph coefficient.
The TT control is rerun on the same positive tensor and current folds; historical TT scores are not mixed into this comparison.
The source window preparation did not explicitly exclude EEGLAB discontinuity annotations. These results inherit that limitation.
Final-model factors and cognition correlations are descriptive full-cohort analyses, not independent clinical validation.
Graph regularization encourages smooth sensor maps; it does not prove anatomical causality or eliminate CP permutation ambiguity.
