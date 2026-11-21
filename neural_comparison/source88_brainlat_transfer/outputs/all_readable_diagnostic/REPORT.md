# Original88 -> BrainLat binary transfer test — all-readable coverage diagnostic

Training:88 original AHEPA participants (59dementia,29HC). Target headers:100; scorable:86.
No target labels/features used to fit learned transforms, select, train, calibrate or threshold the classifiers. Per-record physical preprocessing uses only that recording. BrainLat has been explored previously, so this is cross-dataset transfer rather than untouched external confirmation.

| Frozen source model | Macro F1 | Balanced accuracy | Accuracy | Dementia recall | HC recall |
|---|---:|---:|---:|---:|---:|
| original_procedure | 0.5700 | 0.5775 | 0.5814 | 0.5926 | 0.5625 |
| svm_only | 0.3857 | 0.5000 | 0.6279 | 1.0000 | 0.0000 |
| spectral_svm_control | 0.3471 | 0.3953 | 0.4884 | 0.7593 | 0.0312 |

**Coverage diagnostic:** 86 readable recordings include 10 healthy recordings below the old1uV channel-SD floor. Those cases use the pre-existing0.1uV floor; the other76 retain their original strict-QC features and predictions. Models are unchanged. This is not the strict-source-QC primary endpoint. Fourteen header-only recordings remain unclassified.

## Interpretation

- The old75% result described nested cross-validation, not a single saved88-person model. The old36-candidate selection recipe was refit on all88, with choices made solely by three-fold source CV.
- The primary old recipe selected a spectral logistic-regression pipeline. The SVM-only winner and a reference-compatible spectral SVM were frozen as separate source-only controls before target preparation. All results are reported; no target winner is promoted.
- Source feature extraction was replayed against cached old arrays; units and channel ordering are checked. BrainLat uses the fixed pre-existing19-channel spatial mapping and250Hz feature schema.
- Spectral features use19-channel CAR. The old connectivity/power bank retained the original acquisition reference; reproducing that code cannot recreate AHEPA ear references in BrainLat. Interpret the connectivity SVM accordingly.
- Missing signal files and source-policy QC refusals are listed in the primary outputs/target_audit.json. Strict-QC metrics exclude refused cases; the separate all-readable diagnostic includes10 explicitly flagged low-amplitude cases. Missingness may bias the available target cohort.
- This all-readable diagnostic reports point estimates only. Bootstrap intervals in the strict-QC primary summary apply to its76 participants, not this86-person diagnostic.
- Predictions and source-calibrated probabilities are research outputs, not validated clinical probabilities. No test-set adaptation or model selection was performed.
