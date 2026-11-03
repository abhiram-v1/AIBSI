# Graph-NTF experiment results

Profile: **quick**; outer=stratified; participants=174.
All reported predictions hold participants out from tensor fitting, tuning and calibration.
Quick runs are bounded development checks; full nested search is a separate command.
Confidence intervals bootstrap participants within class, keeping repeat predictions together. They are conditional on fitted models.

| Method | Accuracy | Macro F1 | Balanced accuracy | Log loss |
|---|---:|---:|---:|---:|
| coupled | 0.483 | 0.414 | 0.436 | 1.038 |

Outer tensor fits without any optimizer-converged initialization: 5. Inspect each fold objective history before drawing decomposition conclusions.
The real and shuffled graph arms also tune graph strength: their difference tests complete procedures, not one fixed graph coefficient.
The TT control is rerun on the same positive tensor and current folds; historical TT scores are not mixed into this comparison.
The source window preparation did not explicitly exclude EEGLAB discontinuity annotations. These results inherit that limitation.
Final-model factors and cognition correlations are descriptive full-cohort analyses, not independent clinical validation.
Graph regularization encourages smooth sensor maps; it does not prove anatomical causality or eliminate CP permutation ambiguity.
