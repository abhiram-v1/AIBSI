# Binary log-mel SVM: 174-participant nested experiment

1 repeats x 2 outer folds; 2 inner folds. HC=0; AD+FTD=1.
All preprocessing choices, classifier settings, weights and thresholds selected without outer-test labels.
Primary: macro F1. Secondary selection: balanced accuracy. Repeated folds do not create additional participants.

| Metric | Mean | Repeat SD |
|---|---:|---:|
| macro_f1 | 1.0000 | — |
| balanced_accuracy | 1.0000 | — |
| accuracy | 1.0000 | — |
| healthy_recall | 1.0000 | — |
| dementia_recall | 1.0000 | — |
| roc_auc | 1.0000 | — |
| log_loss | 0.0001 | — |

## Sampling-rate controls

Controls use 8-second segments, 2-second STFT, 0.5-second hop, 24 mel bands, mean/SD, site z-scoring, standard scaling, PCA40, balanced RBF SVM C1/gamma scale, threshold zero.
256 Hz is obtained directly from native source EEG; the 128-Hz control is anti-aliased downsampling of that signal. With 1–40 Hz filtering, higher rate does not introduce additional biological bandwidth.

| Sampling | Macro F1 | Balanced accuracy |
|---|---:|---:|
| 128 Hz | 1.0000 | 1.0000 |
| 256 Hz | 1.0000 | 1.0000 |

## Scope and limitations

- Frozen random search with representation coverage, not an exhaustive Cartesian grid. Candidate settings are saved in prepared/plan.json.
- Per-person feature extraction is deterministic and label-independent. Learned site adjustment, scaling, PCA, SVM and calibration are fitted on training participants only.
- Participant summaries pool frames across accepted windows. Temporal ordering is not an SVM input.
- Native source derivatives have different upstream cleaning. BrainLat interpolation is geometry-based harmonization, not additional measured electrodes.
- Windows touching marked boundary/BAD/EDGE intervals (2-second guard) are rejected. Source filtering still reuses the existing full-record harmonizer.
- At most 320 accepted seconds per person at every segment length; participants with less usable EEG contribute less. Missing annotation or unmarked artifact can remain.
- Main predictions use inner-selected score thresholds. Calibrated probabilities may have an operating threshold different from 0.5; probability argmax metrics are saved separately.
- Clinical AD/FTD labels are collapsed to dementia. This experiment does not distinguish the two conditions.
- Same cohort has been explored previously. Results are internal validation, not an untouched external benchmark.
- Bootstrap intervals resample participants within class, keeping repeats together, conditional on already fitted models. They do not capture all model-selection uncertainty.
- A final full-cohort model is for inference and does not provide test-performance evidence.
