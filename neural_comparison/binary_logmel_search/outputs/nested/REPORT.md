# Binary log-mel SVM: 174-participant nested experiment

3 repeats x 5 outer folds; 3 inner folds. HC=0; AD+FTD=1.
All preprocessing choices, classifier settings, weights and thresholds selected without outer-test labels.
Primary: macro F1. Secondary selection: balanced accuracy. Repeated folds do not create additional participants.

| Metric | Mean | Repeat SD |
|---|---:|---:|
| macro_f1 | 0.6985 | 0.0033 |
| balanced_accuracy | 0.7029 | 0.0019 |
| accuracy | 0.7203 | 0.0066 |
| healthy_recall | 0.6448 | 0.0250 |
| dementia_recall | 0.7611 | 0.0234 |
| roc_auc | 0.7718 | 0.0220 |
| log_loss | 0.5498 | 0.0375 |

## Sampling-rate controls

Controls use 8-second segments, 2-second STFT, 0.5-second hop, 24 mel bands, mean/SD, site z-scoring, standard scaling, PCA40, balanced RBF SVM C1/gamma scale, threshold zero.
256 Hz is obtained directly from native source EEG; the 128-Hz control is anti-aliased downsampling of that signal. With 1–40 Hz filtering, higher rate does not introduce additional biological bandwidth.

| Sampling | Macro F1 | Balanced accuracy |
|---|---:|---:|
| 128 Hz | 0.6931 | 0.7018 |
| 256 Hz | 0.6931 | 0.7018 |

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
