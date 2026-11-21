# Literature method adaptations: full174-person, three-class benchmark

Five shared outer participant folds; each participant predicted once. Classical tuning uses three inner folds; neural stopping uses an inner participant split followed by refitting on all outer-training participants.
These are documented reimplementations/adaptations, not exact reproductions of the papers. No external validation. The cohort has been explored in prior experiments.

| Method | Accuracy | Balanced accuracy | Macro F1 | AD recall | FTD recall | HC recall |
|---|---:|---:|---:|---:|---:|---:|
| li_cp6_gnb | 60.3% | 56.5% | 56.5% | 74.6% | 31.0% | 63.9% |
| rostamikia_hierarchical_svm | 55.2% | 52.5% | 51.8% | 53.5% | 28.6% | 75.4% |
| dice_net_3class | 51.1% | 48.3% | 47.9% | 56.3% | 26.2% | 62.3% |
| jiang_microstate_cascade | 41.4% | 42.7% | 40.4% | 25.4% | 45.2% | 57.4% |

Completed methods: 4/4.
See README.md and protocol.json for source papers, implementation choices and deviations. Bootstrap intervals are conditional on these fitted predictions; overlapping training folds do not provide independent replications.
Moderate preprocessing inherits missing explicit EEGLAB boundary rejection. This is a signal-quality limitation, separate from train/test participant leakage.

## Matched existing controls (repeat0 only)

These use the same174 people and the exact same five outer test memberships. Their representations and training procedures differ from the literature adaptations.

| Existing control | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| direct | 53.4% | 48.5% | 46.3% |
| gntf | 53.4% | 47.1% | 40.5% |
| tt | 54.6% | 49.1% | 46.3% |

## Verification and interpretation

All20 saved fold models were reloaded and reproduced their outer-test class decisions. Each method has exactly one held-out prediction for every participant; inner and outer participant memberships are disjoint. Prepared input and code hashes were checked.

CP6 optimization: 3/5 final outer fits had no optimizer-converged initialization within the frozen500-iteration/three-start budget (zero-based folds: [1, 2, 4]). Scores are bounded-optimization results, not proof of an optimal CP solution.

The best observed accuracy is 60.3% (105/174 people), but its FTD recall is only 31.0%. Do not describe it as reliable recognition of all three diagnoses.

The paper methods were adapted, not exactly replicated. In particular SVM band extraction/nonlinear estimators and the microstate model differ from source details, DICE has corrected batch semantics and a three-class head, and the entire evaluation uses the merged cohort and common harmonization. These scores do not establish what the original papers would achieve under their original full protocols, or that their published results arose from leakage.

## Why the published performance did not transfer

The papers' headline figures are not four measurements of our task. Our endpoint is one held-out AD/FTD/HC decision for each of 174 participants: 88 AHEPA/OpenNeuro participants and 86 BrainLat participants from Argentina and Chile. Every outer fold contains held-out people from these sources; feature selection, scaling, factor fitting, and model training use outer-training people only. The numbers below separate **observed failure patterns** from **plausible mechanisms**. This experiment did not isolate the effect of each protocol change.

| Method | Original paper's headline endpoint | Our held-out endpoint |
|---|---|---|
| Li CP6 + Gaussian NB | Several binary AUCs of 0.59-0.91; AD dementia vs. Lewy body dementia AUC 0.61. No AD/FTD/HC accuracy. | Three-class accuracy 60.3%; balanced accuracy 56.5%. |
| Rostamikia feature SVM | Separate binary ten-fold CV: dementia vs. HC 93.5% accuracy; AD vs. FTD 87.8%. | One hierarchical three-class decision: 55.2% accuracy; 52.5% balanced accuracy. |
| DICE-Net | Separate binary LOSO: AD vs. control 83.28%; FTD vs. control 74.96%. | One three-class model: 51.1% accuracy; 48.3% balanced accuracy. |
| Jiang microstate cascade | Three-class balanced accuracy 63.9% under subject-level LOSO on 88 AHEPA subjects. | Three-class balanced accuracy 42.7% under five participant folds on 174 people. |

### Li: the factors were learned from far less EEG

[Li et al.](https://doi.org/10.1093/braincomms/fcae227) discovered six nonnegative factors using **12,176 routine EEG recordings from 11,001 patients**, then assessed several clinical contrasts. Our leakage-safe benchmark must relearn the CP factors within each training fold, using roughly 139 participants instead of the paper's large external pool. With fewer and more diagnosis-focused recordings, the rank-six factors may be less stable and may capture different spectral structure. We also used 1-40 Hz instead of 1-45 Hz and a bounded L-BFGS-B implementation. In **three of five** final outer fits, none of three starts converged before the 500-iteration cap, so our score reflects a finite-budget fit, not a proven optimum. These are concrete limitations of this adaptation, although their individual effects have not been measured.

Our confusion matrix identifies an additional problem for this task: **22 of 42 FTD people were classified as AD**, while only 13 FTD people were correctly identified. On the 88 AHEPA people our accuracy was 47.7%, compared with 60.3% over the merged cohort. This is the AHEPA subgroup of a model trained in mixed-site folds, not a rerun trained only on the original cohort. The paper's AUCs cannot be subtracted from our accuracy because the metric, labels, and evaluation population differ.

### Rostamikia: two binary successes do not equal one successful cascade

[Rostamikia et al.](https://doi.org/10.1007/s11571-024-10152-7) reported **93.5%** for dementia versus HC and **87.8%** for AD versus FTD, each evaluated as its own binary ten-fold CV problem on 88 subjects. Our cascade must get **both** decisions right for a dementia participant to receive the correct three-class label. Stage 1 sent **25 of 113 dementia participants to HC**. It also sent 15 of 61 controls to a dementia label. Among the 88 dementia participants that received a dementia label, only 50 were given the correct AD/FTD subtype (**56.8% conditional accuracy**). That conditional figure is calculated from our final predictions and is not equivalent to the paper's separately tested AD/FTD classifier. It shows exactly where our three-class decision loses people.

Our feature representation is also different: Butterworth frequency bands replace the paper's db4 wavelet subbands; nonlinear measures use a bounded, sampled-window computation; and all feature ranking and scaling are redone inside each training fold. These choices can change the Fp1, delta/alpha connectivity, and nonlinear signatures used by the SVM. Their effects cannot be separated without one-change-at-a-time ablations. The original 88-person study and our 174-person mixed-site benchmark also differ in acquisition and population. Within our mixed-fold evaluation, AHEPA subgroup accuracy was 46.6%, so the gap is not explained solely by adding the BrainLat test participants.

### DICE-Net: we changed the output problem and the signal pipeline

[DICE-Net](https://doi.org/10.1109/ACCESS.2023.3294618) reported **83.28% AD/control** and **74.96% FTD/control** accuracy using **two binary models** and leave-one-subject-out (LOSO) validation. The authors' split was participant-disjoint; we have no basis to label that result as leakage. Our three-logit head must distinguish AD from FTD at the same time it distinguishes both from HC. The observed FTD breakdown is **11/42 correct, 14/42 called AD, 17/42 called HC**, producing only 26.2% FTD recall. This shows why good binary disease/control scores do not imply good joint subtyping.

The source study used ASR/ICA cleaning and approximately 30-second power/coherence inputs. We used common 1-40 Hz/128 Hz harmonization, 31-second windows with 30 frames, our Morlet coherence estimator, and a corrected `batch_first=True` transformer. The latter avoids cross-record attention under batched inference but means our implementation is not bitwise identical to the released model. Our AHEPA subgroup reached **54.5%** three-class accuracy, versus 51.1% overall. Thus the added sites contribute some difficulty for DICE, but the same-site subgroup still does not approach the paper's easier binary endpoints. We have not measured how much is due to task, cleaning, feature calculation, or model implementation.

### Jiang: a closer endpoint match, but substantial signal and site differences

[Jiang et al.](https://doi.org/10.3390/bios16050258) reported **63.9% three-class balanced accuracy** using a two-stage microstate cascade and subject-level LOSO on the 88-person AHEPA dataset. Our adaptation reached **42.7%** on the merged cohort. The site breakdown is informative: **48.8% balanced accuracy for AHEPA, 35.7% for Argentina, and 39.1% for Chile**. Transfer to the BrainLat sites is harder, but the same-site subgroup also falls below the published figure; site difference alone is insufficient as an explanation. Our main error is AD: **18/71** AD people correct, **30/71** called FTD, and **23/71** called HC. FTD recall was 45.2%, close to the paper's 43.5%, while the overall balanced accuracy fell because AD and HC recognition were lower.

The original analyzed overlapping **10-second, 256 Hz** epochs with **0.5-45 Hz** filtering. We used **8-second, 128 Hz**, **1-40 Hz** windows, caps on maps/windows used for microstate template learning, and an architecture reimplementation because the linked author code was unavailable. Fewer samples and shorter sequences could remove brief state transitions; different filtering and template fits could shift the four spatial maps and their continuous correlation trajectories. These are mechanistic hypotheses consistent with the protocol differences, not measured causal attributions. Original LOSO and our five-fold CV are both participant-disjoint, but they train on different mixtures of subjects and cannot be treated as interchangeable.

### What we can conclude and what would test the causes

The binary/three-class and AUC/accuracy mismatches are **known comparability problems**. The site breakdown, class confusion, CP convergence flags, and signal-processing differences supply **specific evidence** for why these adaptations underperform, but they do not tell us how many percentage points each change caused. The moderate preprocessing path also lacks explicit EEGLAB boundary rejection, a signal-quality limitation rather than evidence of train/test leakage. A causal follow-up would first hold the AHEPA people, three-class labels and participant splits fixed, then change one of cleaning, feature estimator, input duration, or model implementation at a time. A separate held-out-site test would measure transfer to BrainLat. Until those experiments are done, these are internal literature-method adaptations, not exact replications or evidence that the original papers' results were invalid.

[Implementation, sources and departures](../../README.md) | [Machine-readable verification](verification.json) | [Conditional bootstrap intervals and site results](summary.json)
