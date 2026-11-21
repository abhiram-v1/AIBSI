# Original88 -> BrainLat binary transfer test — strict source QC

Training:88 original AHEPA participants (59dementia,29HC). Target headers:100; scorable:76.
No target labels/features used to fit learned transforms, select, train, calibrate or threshold the classifiers. Per-record physical preprocessing uses only that recording. BrainLat has been explored previously, so this is cross-dataset transfer rather than untouched external confirmation.

| Frozen source model | Macro F1 | Balanced accuracy | Accuracy | Dementia recall | HC recall |
|---|---:|---:|---:|---:|---:|
| original_procedure | 0.5476 | 0.5690 | 0.5789 | 0.5926 | 0.5455 |
| svm_only | 0.4154 | 0.5000 | 0.7105 | 1.0000 | 0.0000 |
| spectral_svm_control | 0.3812 | 0.4024 | 0.5526 | 0.7593 | 0.0455 |

## Interpretation

- The old75% result described nested cross-validation, not a single saved88-person model. The old36-candidate selection recipe was refit on all88, with choices made solely by three-fold source CV.
- The primary old recipe selected a spectral logistic-regression pipeline. The SVM-only winner and a reference-compatible spectral SVM were frozen as separate source-only controls before target preparation. All results are reported; no target winner is promoted.
- Source feature extraction was replayed against cached old arrays; units and channel ordering are checked. BrainLat uses the fixed pre-existing19-channel spatial mapping and250Hz feature schema.
- Spectral features use19-channel CAR. The old connectivity/power bank retained the original acquisition reference; reproducing that code cannot recreate AHEPA ear references in BrainLat. Interpret the connectivity SVM accordingly.
- Missing signal files and source-policy QC refusals are listed in the primary outputs/target_audit.json. Strict-QC metrics exclude refused cases; the separate all-readable diagnostic includes10 explicitly flagged low-amplitude cases. Missingness may bias the available target cohort.
- Confidence intervals bootstrap target participants within class, conditional on the frozen source model; they do not account for source training uncertainty.
- Predictions and source-calibrated probabilities are research outputs, not validated clinical probabilities. No test-set adaptation or model selection was performed.
