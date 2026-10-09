# Neural network and SVM EEG comparison

Status: the pooled experiment is specified, but BrainLat file access is still needed; it has not been trained or scored.

## Data

- Current local source: OpenNeuro [ds004504](https://openneuro.org/datasets/ds004504), 88 participants (36 AD, 23 FTD, 29 healthy controls), eyes-closed resting EEG. Local files: `dataset/ds004504`.
- Independent candidate: [BrainLat](https://www.synapse.org/Synapse:syn51549340), with 35 AD, 19 behavioral-variant FTD, and 42 healthy-control **EEG recordings** (96 relevant participants) in Table 4 of the [dataset paper](https://doi.org/10.1038/s41597-023-02806-8). Its EEG is 128-channel eyes-closed resting state, about 10 minutes per recording. The full BrainLat cohort has 780 participants, but most do not have EEG; do not use 780 as the EEG sample size. BrainLat is not yet downloaded locally.
- BrainLat's public folder listing contains separate AD (`syn53222482`), bvFTD (`syn53222483`), and control (`syn53222486`) EEG folders, each with recordings and group CSV metadata. Anonymous file download returned HTTP 403 in this workspace; a Synapse login with file access is needed before ingestion. No Synapse credentials are configured here.
- Complementary candidate: OpenNeuro [ds006036](https://openneuro.org/datasets/ds006036), eyes-open photic-stimulation EEG. Its [README](https://github.com/OpenNeuroDatasets/ds006036) states that its participant IDs match ds004504. It adds a recording condition, **not independent patients**. It is not yet downloaded locally.

For the BrainLat download, place the AD (`1_AD`), bvFTD (`2_bvFTD`), and healthy-control (`5_HC`) folders from Synapse's `EEG data` folder under `dataset/brainlat/`. Retain the participant subfolders and EEG `.set` files. Any source metadata CSVs are only for checking participant identity, site, and labels; **no CSV columns or hand-built tabular feature files enter the neural network**. A Synapse account is required for file downloads in this workspace. Never put an access token in this repository or chat.

## Pooled experiment

Continue the current binary task: AD or FTD versus healthy control. Also report AD and FTD recall separately within the dementia-positive class. If the scientific question changes to three classes, use a separate protocol and do not compare raw accuracy to binary results.

| Input representation | Neural network | Classical model |
| --- | --- | --- |
| Log-power spectrogram, no tensor decomposition | 2D CNN on per-channel time-frequency maps (target roughly 0.2–0.5 million trainable parameters), with person-level probability aggregation | Optional class-weighted RBF SVM on fixed spectral summaries as a classical baseline |
| Log-power spectrogram with tensor decomposition | The same spectrogram input followed by training-fold-fitted TT or Tucker compression and a neural classifier | Optional SVM on the fold-fitted compressed representation |
| Filtered waveform | 1D temporal/spatial CNN on the matched EEG windows | No classical model required |

Use identical participant folds for all arms, stratified by dataset and diagnosis. Keep all windows from one participant in the same fold. Fit decomposition, channel scaling, feature selection, and all learned preprocessing on training participants only. Select ranks, network settings, and SVM hyperparameters inside training folds. Stop neural training using a participant-level validation set from the training fold, and weight or sample participants equally so long recordings do not dominate. A CNN is the first sequence model; an RNN can be a later comparison if there is enough signal and training stability to justify it. Spectrograms are the primary representation because they expose time-varying frequency power; the raw-waveform CNN tests whether that choice actually helps. Do not claim spectrograms must outperform raw EEG.

Primary outcome: participant-level macro F1. Secondary outcomes: dementia recall, AD recall, FTD recall, healthy-control specificity, balanced accuracy, and accuracy. Save out-of-fold predictions, conventional Matplotlib confusion matrices, and uncertainty across repeated participant-level folds. Report both pooled and per-dataset scores. Do not treat correlated windows as independent test cases.

Perform two evaluations: (1) pooled training with repeated participant-disjoint cross-validation stratified by source and label, and (2) cross-dataset transfer (train on ds004504, test on BrainLat; then reverse). Cross-dataset testing assesses site shift, while the pooled model uses both cohorts for training. Do not call BrainLat an external test set for a model that was trained on BrainLat. Do not use ds006036 as an external patient test set.

## Common EEG preprocessing

The local ds004504 derivatives are already cleaned with approximately 0.5–45 Hz filtering, ASR, and ICA; the [BrainLat paper](https://doi.org/10.1038/s41597-023-02806-8) describes 0.5–40 Hz filtering, ICA, and bad-channel interpolation. The two pipelines cannot be made identical after the fact. Record these differences and test their impact.

1. Inspect BrainLat EEG headers and channel locations before choosing a common scalp montage. Align the same 10–20 channels available in ds004504, excluding reference and non-EEG channels. Do not guess BioSemi channel-to-position mapping.
2. Verify units and polarity, re-reference both to the mean of the verified common scalp channels, low-pass the ds004504 derivatives to a common 40 Hz ceiling, then anti-alias resample both to 128 Hz. Avoid re-running ICA blindly on already cleaned files.
3. Divide recordings into nonoverlapping eight-second windows. Exclude boundaries, flat channels, gross artifacts, missing channels that cannot be consistently mapped, and nonfinite data using the same documented QC rules across sources. Keep a per-person QC ledger and inspect representative traces and spectra.
4. For each retained window, compute a fixed short-time Fourier transform (2-second Hann sections, 0.5-second hop) and retain approximately 1–40 Hz. Use log power, yielding a channel × frequency × time array; preserve per-channel maps instead of collapsing everything into one image. Save numeric tensors as compressed arrays, not as CSV feature tables or rendered PNG model inputs. Make conventional Matplotlib spectrograms only for QC and explanation.
5. Compute any amplitude scaling from training participants only. Preserve physical power differences by default rather than normalizing every window independently. Keep diagnosis, filenames, site, and clinical test scores out of the model input. Audit for residual dataset/site effects.

## Label and data audit before modeling

The BrainLat labels are pre-existing clinical diagnoses, not EEG-derived ground truth. The [dataset paper](https://doi.org/10.1038/s41597-023-02806-8) describes multidisciplinary diagnosis using probable AD/bvFTD criteria, clinical interviews, standardized cognitive and functional assessments, and neuroimaging when needed. These are clinical reference labels; an EEG correlation cannot prove an individual diagnosis.

Once files are accessible, verify that every EEG participant ID maps to exactly one group and the matching records/demographics row. Check recording durations, channels, sampling rates, missing or corrupt files, near-duplicate recordings, and whether site, age, sex, recording quality, or preprocessing batch tracks the diagnostic label. Compare cognitive/functional scores with the documented inclusion criteria as an audit only; exclude these fields and folder names from EEG model inputs. Keep all data from one person together, and report site-stratified or leave-one-site-out results where class counts allow. Flag contradictory records for manual review rather than changing diagnoses from EEG features alone.
