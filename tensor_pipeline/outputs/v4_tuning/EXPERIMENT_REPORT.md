# Tensor/classifier optimization experiment

Date: 2026-09-06. Earlier datasets, model artifacts and reports remain preserved.

## Result

The complete nested-selection pipeline achieved **54.5% balanced accuracy**, versus **52.0%** for V3 on the same cohort and outer subject splits (+2.5 percentage points). This is a comparison of the full tuning procedures, not an independently established improvement.

The strongest observed classifier family was **Tuned RBF SVM**, at **59.7%**. XGBoost achieved **57.6%**. Best-family results are secondary; they cannot replace the predeclared primary endpoint after inspection.

## Classifier comparison

| Method | Balanced accuracy ± repeat SD | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Inner-selected complete pipeline (PRIMARY) | 54.5% ± 3.2 pp | 55.3% | 0.540 |
| Tuned RBF SVM | 59.7% ± 0.9 pp | 60.6% | 0.589 |
| Tuned random forest | 58.3% ± 2.9 pp | 59.8% | 0.578 |
| Tuned XGBoost | 57.6% ± 1.9 pp | 59.5% | 0.574 |
| Tuned logistic regression | 54.6% ± 1.1 pp | 55.7% | 0.540 |
| Stronger-regularized supervised TT | 50.9% ± 2.4 pp | 52.7% | 0.503 |

Each classifier-family row tunes its feature representation as well as its settings inside inner CV. It does not necessarily use the same tensor representation in every outer fold. Results are means across three complete subject-level CV repetitions; SD is not a confidence interval.

## Does tensor representation help?

| Representation family | Balanced accuracy ± repeat SD | Macro F1 |
|---|---:|---:|
| TT/spectral fusion + tuned classifier | 59.1% ± 3.6 pp | 0.585 |
| Spectral features + tuned classifier | 56.8% ± 1.2 pp | 0.561 |
| Waveform Tucker + tuned classifier | 52.0% ± 2.3 pp | 0.515 |
| Waveform TT + tuned classifier | 46.8% ± 11.7 pp | 0.463 |

These rows select classifiers/settings inside each representation family. Fusion starts with TT16 plus 418 spectral features, then selects 32 training-side features; it can favor spectral features and does not guarantee a tensor contribution. No diagnosis, MMSE, age, sex, duration or recording-count predictor is used.

## Recall by diagnosis

| Classifier family | AD recall | FTD recall | CN recall |
|---|---:|---:|---:|
| Inner-selected complete pipeline (PRIMARY) | 53.7% | 42.0% | 67.8% |
| Tuned RBF SVM | 57.4% | 43.5% | 78.2% |
| Tuned random forest | 61.1% | 39.1% | 74.7% |
| Tuned XGBoost | 64.8% | 39.1% | 69.0% |
| Tuned logistic regression | 55.6% | 39.1% | 69.0% |
| Stronger-regularized supervised TT | 55.6% | 30.4% | 66.7% |

Class weighting is unchanged: each training diagnosis has equal total loss weight and each person contributes one pooled observation. More leftover windows do not become independent people. Baseline balanced accuracy for always predicting one class is 33.3%; always predicting AD gives 40.9% ordinary accuracy.

## What was optimized

- 139 configurations were fixed before new outcomes: seven feature settings paired with 19 classifier settings, plus six supervised TT settings.
- XGBoost: shallow trees, learning rate, number of trees, minimum child weight, L1/L2 penalties and row subsampling; column subsampling fixed at 0.8.
- SVM: C and gamma; logistic regression: stronger C regularization; random forest: depth and minimum leaf size.
- Supervised TT: stronger core anchoring, smaller channel/filter ranks, head regularization, and a fixed 500-iteration budget.
- All rank/feature/classifier decisions occurred inside three-fold inner subject CV; outer evaluation was the same 3x5 subject CV as V3.
- No outer-test early stopping and no expansion of the grid after inspecting new outcomes.

Automatic classifier choices across the 15 outer fits: {"logistic": 5, "svm": 7, "random_forest": 1, "xgboost": 1, "supervised_tt": 1}. Representation choices: {"fusion": 7, "spectral": 7, "supervised_tt": 1}. Full per-fold settings are in `selected_configurations.json`.

## Optimizer behavior

200/285 supervised optimization fits reported convergence. This count comprises 270 inner fits and 15 selected supervised outer fits, excluding aliases. Statuses: {"STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT": 85, "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH": 192, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL": 8}.

Any remaining iteration-budget exhaustion is a limitation, not a successful convergence claim. Even a converged nonconvex tensor fit need not be globally optimal. Stronger regularization changes the model objective; this is not simply the previous model trained longer.

## Verification and software

- Replayed all 150 saved outer model entries and scores. Selected-pipeline entries may reference other saved models.
- Checked 45 inner and 15 outer subject splits, training-only decomposition moments, orthonormal bases, weighted scaler means and exact inner selection.
- Exported and reloaded 23 XGBoost boosters in UBJ format; their predictions/probabilities match the saved complete pipelines.
- Verified original primary and leftover array hashes unchanged and reused the exact V3 retained inputs.
- Installed official `xgboost-cpu==3.2.0` into the existing Python 3.10 user installation (approximately 2.1 MB); no dependency upgrades, virtual environment, PyTorch or GPU runtime.
- Nested fitting time: 316.3 seconds, excluding installation and verification.

Booster files alone are not the complete EEG predictor: retain the fitted feature basis, feature selector and scaler in the corresponding pipeline. Only load trusted local joblib files. Runtime versions and checksums are saved for reproducibility.

## Interpretation

This bounded tuning experiment tests whether classifier choice and stronger regularization help the existing representations. It does not establish the dataset's maximum achievable accuracy. The primary and every family score are retained regardless of whether tuning helped.

The 88 source people and their outer splits have already been explored. Keeping tuning inside inner folds does not undo researcher-level adaptive reuse. An independent cohort is needed to confirm a performance claim. Waveform features remain pooled second-order delay statistics, not a general long-sequence neural model; no clinical-readiness claim is made.

## Artifacts

- [Detailed protocol](../../v4_tuning/PROTOCOL.md)
- [Reproduction instructions](../../v4_tuning/README.md)
- `protocol.json`: every candidate and fixed evaluation setting.
- `nested_report.json`, `fold_XX.json`: all predictions, scores, selections and optimization status.
- `models/`: complete outer pipelines; `bases/`: training-only inner and outer tensor bases.
- `xgboost_exports/`: portable boosters, to use with matching preprocessing.
- `verification_report.json`, `selected_configurations.json`, `artifact_manifest.json`.

Official references: [XGBoost installation](https://xgboost.readthedocs.io/en/stable/install.html) and [parameter definitions](https://xgboost.readthedocs.io/en/stable/parameter.html).
