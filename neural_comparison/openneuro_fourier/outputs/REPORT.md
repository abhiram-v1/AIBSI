# OpenNeuro Fourier analysis: finding diagnosis-associated patterns

**There are measurable frequency differences in this cohort.** Overall amplitude is a weak discriminator, while relative alpha, theta/alpha ratio and slowing are associated with HC versus AD/FTD. These groups still overlap, and AD-versus-FTD differences are much weaker.

## What the Fourier analysis does

An EEG channel contains a mixture of activity over frequencies and sources. FFT/Welch estimates its frequency power. It does not uniquely unmix brain sources, identify a particular neuron population, or prove that a frequency component is caused by disease. STFT in earlier experiments already used Fourier transforms; this experiment explicitly measures label associations and performs fold-internal frequency feature selection.

All88 OpenNeuro people are included:36 AD,23 FTD,29 HC. We use supplied cleaned derivatives, the same256Hz/19-channel CAR/1–40Hz FIR and accepted8s windows as the compatibility audit. At most40 windows/person. Welch:4-second Hann periodograms,2-second hop,1024-point FFT;157 frequency bins at0.25Hz spacing from1–40Hz. The delta band here is1–4Hz because this processing policy excludes frequencies below1Hz.

The averaged PSD has shape88×19×157. Flattened Fourier features give88×2983 matrices. Spectral summaries give88×361, relative/shape summaries88×247, and the amplitude control88×1. Window spectra are aggregated within each person; windows are never counted as independent participants.

Summary features: absolute and relative power in five bands; slow/fast and theta/alpha ratios;7–13Hz peak location; spectral centroid, entropy and95% edge; total power; a rough background slope and alpha prominence. The background fit is a log-log straight-line proxy, not a validated oscillation/aperiodic decomposition. A7–13Hz peak location is the largest bin in that range, not proof of a distinct alpha oscillation.

## Which patterns correlate with labels?

Global spectra average channels within each participant. The following are descriptive full-cohort comparisons; they were not used to choose classifier features.

| Global marker | HC median | AD median | FTD median | Dementia–HC r | FDR q | Age/sex-adjusted q |
|---|---:|---:|---:|---:|---:|---:|
|theta_alpha_ratio (power ratio)|0.25|1.51|0.708|0.566|1.64e-06|1.23e-06|
|relative_alpha_8_13 (%)|43.3|13.4|21.8|-0.522|4.83e-06|6.6e-06|
|relative_theta_4_8 (%)|8.49|17.7|13|0.442|5.39e-05|0.000235|
|slow_fast_ratio (power ratio)|0.47|2.1|1.1|0.479|3.22e-05|5.63e-05|
|alpha_peak_7_13_hz|9.75|7.25|9.25|-0.413|0.000247|0.000285|
|alpha_above_background_db|12.6|4.9|6.61|-0.518|4.83e-06|7.01e-06|

Power-ratio medians are displayed on the original scale; ratio correlations/tests use their log10 features. Positive r means higher values in the patient group; negative r means lower. Categorical diagnoses were not correlated as arbitrary0/1/2 codes. We tested HC versus pooled AD+FTD and each explicit pairwise contrast. Mann–Whitney tests and age/sex-adjusted OLS results are both saved. Age/sex adjustment is exploratory, assumes the specified linear model, and does not eliminate unmeasured confounding.

FDR is corrected separately within each declared feature family and contrast: global19, regional76, channel-summary361, and combined absolute+relative electrode-frequency5966 tests. These are related exploratory views, not independent replications. Adjacent frequencies and electrodes are correlated; a large number of significant bins is not a large number of independent biomarkers.

| Contrast | Global markers q<.05 /19 | Channel summaries q<.05 /361 | Frequency values q<.05 /5966 |
|---|---:|---:|---:|
|dementia_vs_HC|10|200|1768|
|AD_vs_HC|13|224|2135|
|FTD_vs_HC|5|99|210|
|AD_vs_FTD|0|0|0|

No AD-versus-FTD feature survived these FDR thresholds. This is insufficient evidence for a difference under this analysis; it does not establish that AD and FTD have identical EEG physiology.

## Do these features classify held-out participants?

3 repeated5-fold outer splits, stratified by original diagnosis;3-fold inner selection. ANOVA feature ranking, number of features (5/15/40/all), scaling, linear/RBF SVM C(.01/.1/1/10), and binary score threshold(−.5/0/.5) are fitted/selected inside training folds. Balanced class weights are used. There is no PCA. Age,sex,MMSE,participant IDs, and recording length are not classifier features.

“Inner-selected” chooses the feature family using inner-fold scores. It is the estimate for the complete selection procedure. Picking the highest outer-CV family afterward gives a descriptive ablation result, not a new independently validated optimum.

| Method | Binary F1 | Binary balanced accuracy | Three-class F1 | Three-class balanced accuracy |
|---|---:|---:|---:|---:|
|Amplitude alone|42.5%|44.2%|32.5%|34.5%|
|Spectral summaries|72.1%|72.6%|50.9%|52.6%|
|Relative/shape summaries|68.1%|68.3%|51.4%|52.4%|
|Absolute Fourier spectrum|70.3%|70.6%|54.8%|55.4%|
|Relative Fourier spectrum|74.7%|75.2%|54.4%|54.7%|
|Inner-selected pipeline|71.9%|72.0%|54.2%|54.6%|

The amplitude-only binary score exactly reproduces the earlier OpenNeuro control. A Fourier spectrum therefore provides useful information beyond overall amplitude in this cohort, while statistical association remains compatible with substantial participant-level overlap.

Inner-selected feature-family counts across15 outer folds: binary{'summary': 1, 'relative_spectrum': 5, 'absolute_spectrum': 8, 'relative_summary': 1}, three-class{'absolute_spectrum': 3, 'relative_spectrum': 7, 'summary': 2, 'relative_summary': 3}.

Intervals are participant bootstraps that keep all repeated predictions for each person together. They are conditional on fitted models, not full retraining intervals. Repeats reuse the same88 people. This cohort has informed many earlier project experiments, so results are exploratory rather than an untouched external evaluation.

## Clinical labels and MMSE

The original descriptor reports that initial AD/FTD diagnoses used DSM-IIIR/DSM-IV/ICD-10 and NINCDS-ADRDA criteria, with MMSE to characterize cognitive status. These are clinical labels; the descriptor does not define them by an EEG amplitude or Fourier threshold. We cannot infer the availability of every imaging/CSF/pathology test for every individual from this release. Source: [Miltiadous et al., Data2023,8,95](https://doi.org/10.3390/data8060095), section2.1/page3; retrieval/hash recorded in clinical_context.json.

MMSE is excluded from classification. Controls all score30, so all-cohort MMSE correlation would partly repeat the diagnosis contrast. We instead analyzed only59 AD/FTD patients, with age/sex/AD-versus-FTD adjustment of rank correlations. No severity association survived FDR correction in the380 global/channel-summary tests; the minimum q is0.784.

## What this establishes

OpenNeuro contains label-associated spectral structure, including relatively weaker alpha and greater theta/alpha ratio in the patient groups. Those patterns provide moderate held-out binary classification, with weaker AD/FTD discrimination. The analysis does not show a universally separable disease signature, establish biological causation, or contradict clinical diagnoses when an EEG-only model makes errors.

Earlier conventional spectral features and log-mel/STFT inputs already represented related information. This is a clearer analysis of which features correlate with labels, not evidence that Fourier transformation is a previously unused source-separation method.

## Outputs

- `prepared/features.npz`, `feature_names.json`: participant matrices and exact feature names.
- `associations.json`: per-feature effects, correlations, raw/adjusted p and FDR q.
- `severity_associations.json`: within-patient severity analysis.
- `evaluation/protocol.json`, source snapshots, folds and predictions: reproducible nested CV.
- `selected_feature_stability.json`: feature selection counts across outer training folds; selection frequency is not causal importance.
- `figures/*.png`: seven Matplotlib figures; no PDF/CSV.

![fourier_group_spectra](C:/Projects/AIBSI/neural_comparison/openneuro_fourier/outputs/figures/fourier_group_spectra.png)

![interpretable_frequency_patterns](C:/Projects/AIBSI/neural_comparison/openneuro_fourier/outputs/figures/interpretable_frequency_patterns.png)

![electrode_frequency_associations](C:/Projects/AIBSI/neural_comparison/openneuro_fourier/outputs/figures/electrode_frequency_associations.png)

![electrode_band_associations](C:/Projects/AIBSI/neural_comparison/openneuro_fourier/outputs/figures/electrode_band_associations.png)

![nested_svm_comparison](C:/Projects/AIBSI/neural_comparison/openneuro_fourier/outputs/figures/nested_svm_comparison.png)

![heldout_confusion_matrices](C:/Projects/AIBSI/neural_comparison/openneuro_fourier/outputs/figures/heldout_confusion_matrices.png)

![patient_severity_check](C:/Projects/AIBSI/neural_comparison/openneuro_fourier/outputs/figures/patient_severity_check.png)
