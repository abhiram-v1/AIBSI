# OpenNeuro–BrainLat compatibility audit

This audit compares existing EEG derivatives, before and after a matched processing stage. It cannot reconstruct original unprocessed signals or undo earlier ICA/ASR/interpolation.

Fixed cohort: **174 participants** (88 OpenNeuro,86 BrainLat);71 AD,42 FTD,61 HC. BrainLat comprises Argentina48 and Chile38.14 additional HC headers lack their external EEG data files and cannot be analyzed.

Common rules: same19 channels (existing legacy spline operator for BrainLat), anti-aliased256Hz resampling,19-channel average reference (CAR),1–40Hz FIR,8-second windows, edge/boundary guards,0.1–200µV SD and500µV amplitude limits. All174 pass;6698 retained windows, at most40/person. Existing and matched signals use the exact same accepted windows.

STFT:2-second Hann,0.5-second hop,24 mel bands. Per-person frame mean/SD gives174×912 matrices. These are participant samples, not independent window samples.

**Main finding:** signal matching is possible, but the current derivatives are not demonstrably interchangeable. A classifier still identifies the dataset with90.6% balanced accuracy after matched CAR/filtering. Stronger site alignment reduces detectable dataset differences but also lowers diagnosis performance.

## Measured signal differences

| Metric (median across people) | OpenNeuro existing | BrainLat existing | OpenNeuro matched | BrainLat matched |
|---|---:|---:|---:|---:|
|median_channel_rms_uv|33.189|3.9048|6.3191|3.8499|
|common_mode_variance_fraction|0.9418|0.023079|3.378e-17|3.3744e-17|
|median_pair_correlation|0.94547|-0.07113|-0.092942|-0.09978|
|median_excess_kurtosis|-0.23486|0.11356|0.19249|0.10093|
|high_frequency_30_40_fraction|0.0052828|0.041139|0.033453|0.041358|
|residual_line50_ratio|2.0146e-09|2.8582e-05|3.3109e-10|9.2357e-09|
|residual_line60_ratio|1.4995e-11|7.9879e-06|1.5196e-12|7.1773e-12|

Native sample rates are500Hz for OpenNeuro and512Hz for BrainLat, now both resampled to256Hz. File metadata confirm reference `A1 A2` for all88 OpenNeuro files and `average` for all86 BrainLat files. Matching the reference is a substantive change. BrainLat channel values are interpolated estimates; the original19 OpenNeuro electrode measurements cannot be made equivalent to them by schema conversion alone.

The common-mode fraction measures variance shared in the channel average. It is consistent with a major reference/shared-signal difference; it does **not** prove that shared activity is artifact. CAR mathematically removes the channel mean, so near-zero common-mode variance afterward is expected, not validation of biological equivalence. All voltage figures use the EEGLAB/MNE microvolt convention; independent acquisition gain/calibration has not been verified from these headers.

Median age by diagnosis (OpenNeuro→BrainLat): AD 67→76, FTD 64→70, HC 67→72 years.6 BrainLat participants lack unambiguous matched age metadata. Age, severity, population and acquisition differences can coexist. MMSE and MoCA are not treated as interchangeable scores.

Channel/band tests are stratified by AD/FTD/HC, with FDR correction across855 tests. Significant differences do not by themselves identify bad electrodes or justify deleting channels. The age/diagnosis plots and effect maps are descriptive, not feature-selection input.

## Does matching preserve diagnosis performance?

Binary task: HC versus AD+FTD.3 repeated5-fold outer participant splits, stratified by site×original diagnosis;3-fold inner selection. All six transformations are refitted inside each inner/outer training fold. SVM: linear/RBF, C=.01/.1/1/10, balanced class weights, decision thresholds−.5/0/.5. Global standardization and PCA20 are shared controls. Hyperparameter selection maximizes pooled inner macro F1, then balanced accuracy.

| Method | Macro F1 | Balanced accuracy | Accuracy | HC recall | Dementia recall | Dataset probe BA (linear/RBF) |
|---|---:|---:|---:|---:|---:|---:|
|Existing derivatives|82.4%|82.2%|84.1%|76.0%|88.5%|99.8/99.8|
|Matched CAR + filter|80.3%|79.9%|82.4%|71.6%|88.2%|88.5/90.6|
|Matched + gain norm.|68.9%|69.0%|71.6%|60.1%|77.9%|87.9/89.7|
|Matched + relative power|67.6%|68.4%|69.5%|64.5%|72.3%|86.4/87.2|
|Matched + site z-score|63.9%|63.6%|68.6%|47.0%|80.2%|49.6/62.5|
|Matched + site CORAL|66.8%|66.6%|70.1%|54.6%|78.5%|49.9/46.2|
|Inner-selected method|81.2%|81.0%|83.0%|74.3%|87.6%|—|

These are **binary** internal results and are not comparable to earlier three-class results as a claim of improvement. Existing-derivative pooled macro F1 is82.4%, but the same held-out predictions give 63.9% within OpenNeuro and100.0% within BrainLat. See subgroup_metrics.json.

“Inner-selected” chooses the processing method as well as SVM settings using only the inner folds. It is the fair estimate of the method-selection procedure; do not select the best outer result and call that a new validated optimum.

Site z-score uses training-site feature means/SD. CORAL starts from the gain-normalized features and aligns training-site PCA-space means/covariances to the pooled training covariance, with30% covariance shrinkage. These are distinct arms, not sequential corrections. No test labels, means or covariances enter a fit. Known-site corrections assume site metadata at inference; they cannot estimate an unseen site from zero observations.

Dataset probes predict OpenNeuro versus BrainLat on held-out people.50% balanced accuracy is chance. Diagnosis-conditioned probes are also in summary.json. They fit probes separately within each diagnosis as a descriptive check; diagnosis is never given to a deployed harmonizer. A probe near chance only establishes reduced detectable site information for this feature/PCA/probe setup, not that the cohorts are biologically identical. Some diagnosis-conditioned CORAL probes score below50%; reversed associations or overcorrection can produce this, so those scores do not demonstrate that dataset information is absent.

## Source-only cross-dataset transfer

| Training cohort | Method selected in source CV | Target macro F1 | Target balanced accuracy | HC recall | Dementia recall |
|---|---|---:|---:|---:|---:|
|OpenNeuro|Matched + gain norm.|60.1%|60.5%|56.2%|64.8%|
|BrainLat|Existing derivatives|40.1%|50.0%|0.0%|100.0%|

Only deterministic per-record corrections are eligible for transfer; no target-cohort normalization or parameter selection. All other method results are saved as diagnostic ablations and plotted; choosing one using target performance would invalidate the source-only selection claim.

## Exploratory amplitude/provenance follow-up

This follow-up was added after observing the diagnosis-specific amplitude separation in the main audit; it is not an independent confirmatory endpoint.

**All54 BrainLat AD/FTD files record EEGLAB2020.0, and all32 usable BrainLat HC files record EEGLAB2022.1.** OpenNeuro records2022.1 for every diagnosis. This is a perfect association between the recorded BrainLat processing/export version and the binary label. A version field alone does not establish that algorithms, settings or gains changed, and does not prove label leakage or incorrect diagnoses. It is a concrete provenance issue that needs explanation.

The lower HC amplitude also appears in the original128 BrainLat channels using a fixed56-second native-CAR/1–40Hz segment, **before interpolation**. Thus interpolation is not necessary to create this amplitude gap. These native-segment measurements are descriptive checks with a different fixed-window policy from the main feature experiment.

| Nested SVM using only log10 median-channel RMS | Macro F1 | Balanced accuracy |
|---|---:|---:|
|OpenNeuro|42.5%|44.2%|
|BrainLat|97.1%|97.2%|
|Pooled|77.5%|75.2%|

The single-amplitude feature is almost sufficient for BrainLat diagnosis but does not transfer into a general cohort-independent disease rule. This can reflect biology, cohort composition, gain or prior processing; these data alone do not identify the cause. **Do not multiply HC signals by a diagnosis-specific factor:** that would require the answer at preprocessing time and distort the evaluation.

If different processing batches caused the amplitude difference, their perfect alignment with BrainLat labels prevents statistical correction from reliably separating batch effects from diagnosis. Obtain acquisition gain/unit information, original preprocessing scripts/settings and ideally raw recordings that can be processed consistently across diagnostic groups. Site z-scoring or CORAL cannot establish that this issue has been resolved.

## Interpretation and limits

Matched processing substantially reduces some acquisition differences, but compatibility and prediction performance are separate outcomes. Removing global amplitude or shared activity can also remove disease information. Do not export a single cohort-normalized feature matrix and then cross-validate it: learned site correction must remain inside the training pipeline.

The bootstrap intervals resample participants together with all their repeated held-out predictions. They are conditional on these fitted models and do not include all variability from retraining or recruitment. Repeated folds are not independent new participants. These cohorts have already informed earlier project experiments, so results remain exploratory rather than untouched external validation.

A defensible next analysis should predefine the correction and evaluate on new participants/sites. Keeping source/site metadata, matched referencing/filtering, separate cohort results and the uncorrected comparator is essential. Additional aggressive cleaning or channel deletion is not justified by cohort differences alone.

## Files

- `signal_audit.json`: cohort summaries, age, within-diagnosis tests and channel/band effect sizes/FDR.
- `prepared/parts/*.json`: per-record source hashes, signal metrics, accepted/rejected-window counts.
- `evaluation/protocol.json` and `source_snapshot/`: frozen model protocol/code.
- `evaluation/fold_*.json`: exact participant indices, inner searches and held-out predictions.
- `evaluation/summary.json`, `predictions.npz`, `transfer.json`: machine-readable results.
- `followup/native_metadata.json` and `amplitude_probe.json`: exploratory native-montage/provenance and nested one-feature checks.
- `figures/*.png`: Matplotlib figures, no PDF.

![signal_differences](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/figures/signal_differences.png)

![spectral_comparison](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/figures/spectral_comparison.png)

![channel_band_differences](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/figures/channel_band_differences.png)

![compatibility_vs_diagnosis](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/figures/compatibility_vs_diagnosis.png)

![confusion_matrices](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/figures/confusion_matrices.png)

![cross_dataset_transfer](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/figures/cross_dataset_transfer.png)

![amplitude_and_version_warning](C:/Projects/AIBSI/neural_comparison/dataset_compatibility/outputs/figures/amplitude_and_version_warning.png)
