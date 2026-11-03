# Advanced EEG cleaning, electrode analysis, log-STFT, and log-mel

## Experiment design

- Task: participant-level three-class classification of AD, FTD, and healthy controls.
- Usable participants: 174 (AD 71, FTD 42, HC 61).
- Participant-disjoint split: 121 train, 26 validation, 27 held-out test.
- Both dataset source and diagnosis were used for split stratification.
- The held-out test set was not used to rank electrodes or choose the electrode count.

Fourteen BrainLat healthy-control `.set` headers were excluded because their required `.fdt` signal files are absent.

## Cleaning pipeline

1. Map both datasets to the same 19 standard 10-20 electrodes.
2. Band-pass filter from 1 to 40 Hz and resample to 128 Hz.
3. Detect recording-level channel outliers using robust deviation of amplitude, peak-to-peak range, 30-40 Hz noise ratio, and agreement with the other electrodes.
4. Interpolate at most four bad channels per recording with spherical splines.
5. Apply average reference.
6. Fit conservative ICA and remove at most two components only when temporal correlation, frontal loading, and kurtosis jointly indicate a strong ocular component.
7. Remove 10 seconds from each recording edge.
8. Form non-overlapping 8-second windows and reject windows containing a flat, extreme-amplitude, high-derivative, impulsive, or nonfinite channel.

Cleaning audit:

- 114 recordings required at least one bad-channel interpolation.
- 76 recordings had one or more strong ocular ICA components removed (87 components total).
- 277 artifact windows were rejected.
- 6,719 clean windows were retained.

## Spectral representations

- Log-STFT: 2-second Hann window, 0.5-second hop, 1-40 Hz, resulting tensor `(6719, 19, 79, 13)`.
- Log-mel: the same STFT power compressed into 24 mel-spaced filters between 1 and 40 Hz, resulting tensor `(6719, 19, 24, 13)`.
- Each window is centered by its median log power to reduce recording-gain differences.

## Electrode analysis

The ranking combines training-only mutual information, single-electrode validation macro F1, and an explicit penalty for recording/window noise. It is a predictive ranking, not evidence that an electrode or brain region causes dementia.

- Log-STFT selected 8 electrodes: C3, P7, P8, Fp1, P4, Fz, O1, T8.
- Log-mel selected 6 electrodes: C4, P4, C3, P8, Fz, F4.
- The log-mel subset is mainly central, parietal, and frontal. It avoided the noisiest temporal/frontal channels such as F7, T8, T7, F8, and Fp2.

## Held-out classical-model results

| Representation | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| Cleaned log-STFT | 40.7% | 40.2% | 39.9% | 36.4% | 33.3% | 50.0% |
| Cleaned log-mel | **48.1%** | **46.1%** | **46.3%** | 45.5% | 33.3% | 60.0% |
| Earlier STFT hierarchical SVM | **55.6%** | **52.5%** | **52.0%** | 72.7% | 33.3% | 50.0% |

Log-mel was better than the cleaned full-resolution STFT, but advanced cleaning and electrode removal did not improve the earlier SVM result. An expanded direct SVM search reached 71.3% macro F1 on validation but only 43.0% on the held-out participants, so that validation result must not be reported as model performance.

## Held-out deep-model results

All four models used the same participant split, class-balanced participant bag sampling, spectral augmentation, early stopping on validation macro F1, and one fixed seed. Training ran on the CUDA GPU.

| Representation and model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| Log-STFT CNN | 48.1% | 39.0% | 41.8% | 45.5% | 0.0% | 80.0% |
| Log-STFT CNN-BiLSTM | 44.4% | 40.9% | 40.7% | 45.5% | 16.7% | 60.0% |
| **Log-mel CNN** | **59.3%** | **52.1%** | **53.4%** | 63.6% | 16.7% | 80.0% |
| Log-mel CNN-BiLSTM | 55.6% | 48.9% | 50.1% | 63.6% | 16.7% | 70.0% |

The log-mel CNN is the best cleaned model. Relative to the earlier best hierarchical SVM, it improves accuracy from 55.6% to 59.3% and macro recall from 52.0% to 53.4%. Macro F1 is essentially tied but slightly lower, 52.1% versus 52.5%. A CNN/BiLSTM probability ensemble and validation-only class calibration did not improve on the single log-mel CNN.

## Longer CNN training with early stopping

The moderate-cleaning log-mel CNN was retrained with three seeds, a ceiling of 120 epochs, and early-stopping patience of 18 epochs. Checkpoints were selected by participant-level validation macro F1 and the best checkpoint was restored before evaluation. The runs selected epochs 29, 42, and 23 and stopped at epochs 47, 60, and 41, respectively, so no run needed the full ceiling.

| Model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| 30-epoch CNN ensemble | 59.3% | 52.6% | 53.4% | 63.6% | 16.7% | 80.0% |
| **Long CNN ensemble with early stopping** | **51.9%** | **40.7%** | **44.5%** | 63.6% | 0.0% | 70.0% |

The long ensemble reached 72.4% validation macro F1 but only 40.7% held-out macro F1. Training loss continued to fall after the validation peaks, while validation F1 varied substantially. Early stopping prevented each run from continuing to 120 epochs, but it could not fix the validation-to-test distribution shift. More epochs therefore did not improve generalization on this split.

## Site-normalized result

The large validation-to-test gap suggested acquisition-site shift. The participants come from three sites: AHEPA, AR, and CL. A final experiment therefore standardized every participant's log-mel features using the mean and standard deviation calculated from training participants at the same site. Validation participants and held-out participants never contributed to their normalization statistics.

Validation selected moderate cleaning, all 19 electrodes, 20 PCA components, and a class-balanced linear SVM with `C=0.1`.

| Model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| **Site-normalized log-mel SVM** | **66.7%** | **64.3%** | **64.2%** | 72.7% | 50.0% | 70.0% |

This is 18 correct predictions among 27 held-out participants. It improves on the earlier hierarchical SVM by 11.1 percentage points in accuracy, 11.8 points in macro F1, and 12.2 points in macro recall. The result also shows that moderate cleaning generalized better than interpolating and applying ICA to a large fraction of recordings.

This 66.7% result is exploratory because several model families were investigated after inspecting performance on the same held-out split. It should be confirmed with repeated nested participant-level cross-validation or a fresh locked test set before being presented as an unbiased publication result.

### Failure analysis of the best SVM

The nine held-out errors are concentrated in acquisition-site and diagnostic overlap rather than low-confidence boundary cases:

- AHEPA: 10/14 correct (71.4%).
- AR: 5/10 correct (50.0%). Five of the nine total errors are from AR.
- CL: 3/3 correct, which is too small for a stable site estimate.
- Recall is 72.7% for AD, 50.0% for FTD, and 70.0% for healthy controls.
- Eight of the nine incorrectly classified participants are closest to the training centroid of the same incorrect class chosen by the SVM.
- Incorrect predictions have a higher mean decision margin than correct predictions (1.31 versus 1.16), and six of nine errors are above the median confidence. The problem is therefore feature overlap/site shift, not merely an untuned decision threshold.
- Only one of nine errors has fewer than 40 retained windows, so the error pattern is not explained by recordings having too few windows.

One corrected prediction would produce 70.4% accuracy but only 69.4% macro F1. At least two appropriate corrections are needed to exceed both thresholds; the best two-error counterfactual is 74.1% accuracy and 73.8% macro F1.

### Site-shift audit and harmonization

The acquisition protocols differ materially by site. AHEPA recordings are native 19-channel, 500 Hz recordings with a median duration of about 805 seconds. AR and CL recordings begin as 128-channel, 512 Hz data and require spatial interpolation to the common 19-channel montage. AR is also shorter (about 360 seconds median) than CL (about 564 seconds).

PCA-space site centering, z-scoring, and CORAL covariance alignment were compared with global normalization using repeated development-only cross-validation. Cross-validation selected global normalization with 40 PCA components and a linear SVM; it reached only 59.3% accuracy, 52.7% macro F1, and 53.1% macro recall on held-out participants. The existing training-only site-normalized SVM remains better.

| Site | Within-site CV accuracy | Leave-site-out accuracy |
|---|---:|---:|
| AHEPA | 63.6% | 44.3% |
| AR | 47.9% | 56.2% |
| CL | 52.6% | 52.6% |

The low leave-site-out scores confirm poor transfer between acquisition domains. However, AR also performs poorly in within-site cross-validation, so domain shift is not the only issue; diagnostic separation is weak within AR itself. A fixed site-specific SVM produced 59.3% combined held-out accuracy and failed to recognize any of the three held-out AR FTD participants.

The nine errors were older on average (70.1 versus 65.3 years) and had shorter recordings (535 versus 674 seconds), but neither difference was statistically reliable in this small held-out set (Welch-test `p=0.21` and `p=0.16`). These are hypotheses for a larger cohort rather than established causes.

## Binary-specialist SVM ensemble

An ensemble combined an AD-versus-HC SVM, an FTD-versus-HC SVM, and the existing three-class site-normalized SVM. Binary hyperparameters were selected with five-fold cross-validation inside the training split. The fusion weights were selected on validation macro F1.

The AD-versus-HC specialist was substantially stronger than the FTD-versus-HC specialist during training cross-validation (79.2% versus 59.2% macro F1). On the held-out binary subsets, they reached 57.1% and 61.9% macro F1, respectively.

| Fusion | Accuracy | Macro F1 | Macro recall |
|---|---:|---:|---:|
| **Original three-class SVM** | **66.7%** | **64.3%** | **64.2%** |
| Three-model majority vote | 63.0% | 61.2% | 61.2% |
| Binary hard gate with three-class tie-break | 51.9% | 51.0% | 51.5% |
| Validation-tuned probability fusion | 44.4% | 34.2% | 38.2% |

The ensemble did not help. The two specialists often disagreed on healthy participants and overturned correct predictions from the three-class model. Probability fusion looked better on validation (73.1% accuracy and 68.7% macro F1) but missed every held-out FTD participant, another example of validation overfitting.

### SVM refinement attempts

Two follow-up refinements were selected without using held-out labels:

1. A repeated five-fold development-set search compared mean/standard-deviation, median/IQR, relative-power, hemispheric-asymmetry, and inter-electrode correlation features under four site-normalization strategies. It selected robust median/IQR features with 20 PCA components, linear SVM `C=0.03`, and an FTD weight of 1.5. Held-out performance was 55.6% accuracy, 52.6% macro F1, and 52.3% macro recall.
2. Class decision offsets for the 66.7% site-normalized SVM were tuned only on validation macro F1. Validation macro F1 rose to 67.7%, but the calibrated model made exactly the same held-out predictions and therefore remained at 66.7% accuracy and 64.3% macro F1.
3. An extended search evaluated 1,568 configurations using two repeated five-fold participant-level development splits. Every fold refitted site normalization, scaling, and PCA. The search covered 10-80 or no PCA components, linear/RBF kernels, seven `C` values, six RBF gamma settings, and four FTD class weights. It selected 60 PCA components, a linear SVM with `C=0.01`, and an FTD multiplier of 1.25. Held-out accuracy remained 66.7%, while macro F1 decreased to 61.0% and macro recall to 61.4%.

None of these refinements improved the original configuration. Robust and connectivity features generalized worse, threshold calibration was neutral, and the broader hyperparameter search shifted errors toward AD recognition at the expense of FTD and healthy-control recall.

## Filter-bank Riemannian SVM

A filter-bank covariance experiment was run directly from the moderately cleaned raw EEG. For every participant, OAS-regularized 19-by-19 electrode covariance matrices were calculated in delta, theta, alpha, beta, and gamma bands and averaged across accepted 8-second windows. The matrices were mapped into Riemannian tangent space before SVM training.

Repeated five-fold development-only cross-validation selected raw covariance, theta/alpha/beta bands, site centering, 10 PCA components, and a class-balanced linear SVM with `C=0.1`. It reached 69.3% cross-validated macro F1, but this did not transfer to the held-out participants.

| Model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| Site-normalized log-mel SVM | **66.7%** | **64.3%** | **64.2%** | 72.7% | 50.0% | 70.0% |
| Filter-bank Riemannian SVM | 48.1% | 42.3% | 44.6% | 27.3% | 16.7% | 90.0% |

The Riemannian model overemphasized the healthy-control covariance pattern and generalized poorly for both dementia classes. It should not replace the log-mel SVM on this dataset.

## Attention pooling and EEGPT transfer learning

The next CNN used all 19 log-mel electrodes, learned electrode weights, learned attention over participant windows, and included a site-adversarial head. Three independently trained seeds were averaged. It underperformed the site-normalized SVM, with the largest loss again occurring in FTD.

EEGPT was then evaluated as a pretrained EEG foundation model. The official 58-channel, four-second checkpoint was verified by SHA256 before use. Its named channel embedding directly supports all 19 electrodes in this project. The input pipeline retained 6,913 accepted four-second windows from all 174 participants, converted signals to microvolts, and resampled them to EEGPT's native 256 Hz. Training first fit a frozen participant attention head and then fine-tuned the last two transformer blocks with a smaller learning rate. Early stopping selected epoch 9.

| Model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| Site-normalized log-mel SVM | 66.7% | 64.3% | 64.2% | 72.7% | 50.0% | 70.0% |
| CNN with electrode/window attention | 55.6% | 49.7% | 50.1% | 63.6% | 16.7% | 70.0% |
| Fine-tuned EEGPT with window attention | 63.0% | 55.0% | 56.5% | 72.7% | 16.7% | 80.0% |
| EEGPT with two new transformer blocks | 51.9% | 46.9% | 46.8% | 63.6% | 16.7% | 60.0% |
| **Validation-selected SVM + EEGPT blend** | **70.4%** | **67.8%** | **67.6%** | **72.7%** | **50.0%** | **80.0%** |

The validation-selected blend assigned equal weight to SVM and EEGPT probabilities after temperature scaling. Validation performance was 73.1% accuracy and 71.0% macro F1. On the held-out set it made 19 of 27 predictions correctly, one more than the SVM, and raised healthy-control recall from 70% to 80% while preserving FTD recall at 50%.

The depth follow-up appended two newly initialized 512-dimensional, eight-head transformer blocks after EEGPT's 64 summary tokens and added learned token positions. This increased the model from 25.9 million to 31.8 million parameters. The new blocks were trained during the frozen-encoder warmup; the final two pretrained EEGPT blocks were then unfrozen, giving 12.9 million trainable parameters. Validation macro F1 exactly matched the original model at 64.1%, but held-out macro F1 fell from 55.0% to 46.9%. Additional transformer depth therefore increased split-specific overfitting and was rejected.

This ensemble is the strongest exploratory result, but it is not yet a publication-grade estimate. The same small held-out set has been inspected during several model iterations, and blend settings were selected using only 26 validation participants. Repeated nested participant-level cross-validation or a new locked external test set is required before claiming 70.4% as expected performance.

## Tensor Train decomposition with SVM

For the tensor component, each participant's log-mel windows were summarized as a four-way tensor with shape `19 electrodes x 24 mel bands x 4 statistics`. The statistics are mean, standard deviation, median, and interquartile range across resting-state windows and STFT frames. This produced an input array of shape `(174, 19, 24, 4)`.

A common Tensor Train basis was fitted from training participants, with observations placed after the feature modes during TT-SVD. Validation participants were contracted against that fixed basis. This avoids comparing independently decomposed participant cores, whose signs and rotations are not aligned. Site normalization and the TT basis were both refitted inside every development fold.

Two repeated five-fold development splits compared terminal TT ranks 4, 8, 16, and 32; channel ranks 4 and 8; spectral ranks 8, 16, and 24; and class-balanced linear/RBF SVM settings. Development selection chose terminal rank 4, channel rank 4, spectral rank 24, and a linear SVM with `C=0.1`. Its development macro F1 was 55.9%.

| Model | Accuracy | Macro F1 | Macro recall | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| Site-normalized log-mel SVM | 66.7% | 64.3% | 64.2% | 72.7% | 50.0% | 70.0% |
| **Tensor Train + SVM** | **51.9%** | **46.4%** | **46.8%** | 63.6% | 16.7% | 60.0% |
| SVM + EEGPT blend | 70.4% | 67.8% | 67.6% | 72.7% | 50.0% | 80.0% |

The selected rank-4 representation retained four TT features and had a centered reconstruction error of 63.7%. Higher TT ranks retained more variance but had lower development macro F1, so the reduced held-out performance cannot be fixed by reporting a larger rank after seeing test results. TT is useful here as a compact tensor-course experiment and ablation; it did not improve diagnostic separation.

## Electrode and frequency stability analysis

A leakage-safe stability experiment tested whether removing weak electrodes could improve the spectral SVM. It used 10 repeated five-fold participant splits stratified by site and diagnosis. Site normalization, electrode ranking, feature scaling, and SVM tuning were refitted using only each training fold. Grouped permutation importance and leave-one-electrode-out ablation were also calculated inside every fold.

| Electrode count | Accuracy, mean +/- fold SD | Macro F1, mean +/- fold SD | Macro recall, mean +/- fold SD |
|---:|---:|---:|---:|
| 5 | 50.1% +/- 7.8% | 47.4% +/- 7.6% | 47.9% +/- 7.8% |
| 8 | 51.8% +/- 7.4% | **49.1% +/- 7.0%** | 49.3% +/- 7.0% |
| 12 | 50.7% +/- 6.6% | 47.4% +/- 5.9% | 47.9% +/- 6.1% |
| 19 | **51.9% +/- 6.7%** | 49.0% +/- 6.6% | **49.4% +/- 6.8%** |

The eight-electrode model matched the full montage on average but did not improve it. The exact eight channels changed substantially between folds: even the most frequently selected electrodes, T8 and O2, appeared in only 62% and 60% of top-eight sets. The next channels were O1, C3, F7, P7, F3, and P8. This is weak evidence for a fixed reduced montage and does not justify permanently discarding electrodes.

Theta activity from 4-8 Hz caused the largest grouped permutation decrease in macro F1 at 6.11 percentage points, followed by beta at 4.01 and gamma at 3.11. Alpha caused a 0.87-point decrease and delta only 0.10. The gamma association needs cautious interpretation because residual muscle and acquisition noise can occupy the same range.

Replaying the original held-out split showed why the earlier result looked stronger. The 19-electrode model again reached 66.7% accuracy and 64.3% macro F1, while the development-selected five-electrode subset fell to 40.7% accuracy and 37.7% macro F1. Across the 50 repeated test folds, the original fixed `C=0.1` model averaged 48.1% macro F1 and ranged from 33.3% to 66.0%. The old held-out score is therefore near the optimistic edge of the observed split distribution rather than a stable estimate of expected performance.

## Preprocessing and compact-feature SVM audit

A paired experiment compared the moderate preprocessing stream with the advanced stream on identical participant-level folds. It extracted relative spectral power, theta/alpha ratios, individual alpha peak, hemispheric asymmetry, regional spectral summaries, and regional band-envelope correlations. Feature family, site normalization, PCA, SVM `C`, and FTD class weight were selected only inside each outer training fold. A fixed equal-probability fusion combined the two independently selected models without tuning on outer-fold labels.

| Model | Accuracy | Macro F1 | Macro recall | FTD recall |
|---|---:|---:|---:|---:|
| Moderate preprocessing | 54.3% +/- 7.0% | 50.9% +/- 7.6% | 51.6% +/- 7.2% | 31.3% +/- 16.4% |
| Advanced preprocessing | 51.6% +/- 7.7% | 48.9% +/- 8.0% | 49.6% +/- 7.8% | 34.8% +/- 18.7% |
| **Fixed 50/50 fusion** | **54.3% +/- 7.8%** | **51.4% +/- 8.9%** | **52.4% +/- 8.3%** | **36.4% +/- 21.3%** |

Advanced cleaning interpolated 238 channels in 114 of 174 participants and removed 87 ICA components from 76 participants. Relative-band features were selected in 16 of 25 moderate-cleaning folds and 13 of 25 advanced-cleaning folds; regional connectivity was selected only four and seven times. Advanced cleaning reduced paired-fold macro F1 by 1.98 percentage points while increasing FTD recall by 3.56 points. The fusion recovered some complementary decisions, but its 51.4% repeated macro F1 remains far below the exploratory fixed-split result.

This audit rejects the idea that more aggressive automatic cleaning, broad connectivity features, or a simple preprocessing ensemble can reliably produce 70% macro F1. The most credible next performance gain requires additional well-matched FTD participants or external pretraining data, followed by a locked external evaluation.

## Compact-feature Random Forest

A Random Forest baseline used the same moderate-preprocessing compact feature families and the same 25 repeated participant folds as the compact SVM audit. Four-fold training-only selection compared feature family, global versus shrunk site normalization, minimum leaf sizes of 2 and 5, and square-root versus 50% feature subsampling. Each final forest used 800 trees and class weights with a modest FTD boost.

| Evaluation | Accuracy | Macro F1 | Macro recall | FTD recall |
|---|---:|---:|---:|---:|
| 5 x 5 repeated participant folds | 56.2% +/- 6.8% | 51.7% +/- 7.8% | 52.8% +/- 6.9% | 26.7% +/- 16.4% |
| Original fixed test split | 55.6% | 48.9% | 50.4% | 16.7% |

Random Forest exceeded the compact SVM fusion by only 0.35 percentage points of repeated macro F1 while reducing FTD recall from 36.4% to 26.7%. On the original fixed test split it identified only one of six FTD participants. The fixed-split SVM + EEGPT blend remains the final exploratory benchmark at 70.4% accuracy, 67.8% macro F1, and 67.6% macro recall. Random Forest is retained as a classical baseline rather than the final model.

## Interpretation

- The current bottleneck is generalization across participants and acquisition sites, especially for FTD, which has only six held-out participants. The best cleaned CNN identified only one of those six FTD participants.
- Small electrode subsets look strong on validation but are unstable on the held-out participants.
- The high rate of channel interpolation suggests that dataset-specific references and montage conversion affect the automatic noise detector. Interpolation may remove useful disease signal along with noise.
- The CNN-BiLSTM added parameters without improving held-out performance. The next credible experiment is repeated nested participant-level cross-validation with site-aware normalization and cleaning thresholds selected inside each training fold.
- Training-only site normalization produced the largest gain and increased FTD recall to 50%, showing that acquisition harmonization matters more here than additional neural-network depth.
- Longer CNN training increased the validation score but reduced held-out performance. The small validation set is too noisy to use late validation peaks as evidence of a real gain.
- EEGPT transferred useful complementary information but was not stronger than the spectral SVM by itself. Combining both representations corrected one additional held-out participant.
- Appending two new transformer blocks matched validation performance but reduced held-out accuracy by 11.1 percentage points. The original partially fine-tuned EEGPT should be retained.
- Tensor Train compression reduced the participant tensor to four selected features but removed discriminative information, especially for FTD. The undecomposed spectral SVM remains the stronger classical model.
- Repeated participant-level evaluation showed that the 64.3% held-out macro F1 is split-sensitive. The all-electrode spectral SVM averaged 49.0% macro F1 across 50 test folds, with FTD remaining the weakest class.
- Electrode removal did not improve generalization. T8 and O2 were the most consistent candidates, but their selection frequency was too low to claim a stable disease-specific montage.
- Theta was the strongest frequency group in permutation analysis. A useful next experiment is to test training-fold-only regional theta features and stronger site harmonization rather than adding model depth.
- Compact regional and connectivity features improved the repeated estimate slightly, and a fixed fusion reached 51.4% macro F1, but advanced cleaning was counterproductive overall. Automatic interpolation affected 65.5% of participants and should be made less aggressive before reuse.
- Random Forest matched the compact SVM on average but weakened FTD recognition. It does not challenge the SVM + EEGPT blend as the final fixed-split benchmark.

## Reproduction

```powershell
py -3.10 neural_comparison\advanced_clean_spectral\pipeline.py prepare
py -3.10 neural_comparison\advanced_clean_spectral\pipeline.py analyze
py -3.10 neural_comparison\advanced_clean_spectral\train_deep_models.py --epochs 30
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_deep_ensembles.py
py -3.10 neural_comparison\advanced_clean_spectral\calibrate_log_mel_cnn.py
py -3.10 neural_comparison\advanced_clean_spectral\train_logmel_ensemble.py --epochs 30
py -3.10 neural_comparison\advanced_clean_spectral\train_moderate_logmel.py --epochs 30
py -3.10 neural_comparison\advanced_clean_spectral\train_long_logmel.py --epochs 120 --patience 18
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_site_normalized.py
py -3.10 neural_comparison\advanced_clean_spectral\refine_site_svm.py
py -3.10 neural_comparison\advanced_clean_spectral\calibrate_site_svm.py
py -3.10 neural_comparison\advanced_clean_spectral\tune_site_svm_extended.py
py -3.10 neural_comparison\advanced_clean_spectral\train_riemannian_svm.py all
py -3.10 neural_comparison\advanced_clean_spectral\diagnose_best_svm.py
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_site_shift.py
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_binary_ensemble.py
py -3.10 neural_comparison\advanced_clean_spectral\train_attention_pooling.py --epochs 80 --patience 15 --bag-size 12
py -3.10 neural_comparison\advanced_clean_spectral\prepare_eegpt_raw.py
py -3.10 neural_comparison\advanced_clean_spectral\train_eegpt_attention.py --warmup-epochs 6 --finetune-epochs 24 --patience 8 --bag-size 8
py -3.10 neural_comparison\advanced_clean_spectral\train_eegpt_attention.py --warmup-epochs 6 --finetune-epochs 24 --patience 8 --bag-size 8 --extra-blocks 2
py -3.10 neural_comparison\advanced_clean_spectral\compare_eegpt_depth.py
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_eegpt_svm_blend.py
py -3.10 neural_comparison\advanced_clean_spectral\train_tensor_train_svm.py
py -3.10 neural_comparison\advanced_clean_spectral\analyze_electrode_frequency_stability.py
py -3.10 neural_comparison\advanced_clean_spectral\audit_preprocessing_feature_svm.py --repeats 5
py -3.10 neural_comparison\advanced_clean_spectral\evaluate_compact_random_forest.py --repeats 5
```

Prepared arrays and generated figures are under `outputs/`, which is ignored by Git because the tensors are large and reproducible.
