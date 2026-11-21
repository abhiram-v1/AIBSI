# AIBSI: Spectral, Tensor and Raw-Waveform EEG Classification of AD and FTD

Course project for *Tensor Methods for Multimodal Data Analysis and AI* (IIIT Dharwad;
instructor Dr. Sibasankar Padhy). Authors: Neeruganti Jeevan Sai, Kothapalli Tharun Chowdary,
Vatipalli Abhiram, Palyam Maruthi and Shaik Moiz.

The research report is [`final_report/report.pdf`](final_report/report.pdf) (LaTeX source
[`final_report/report.tex`](final_report/report.tex)). This README explains how the code in this
repository corresponds to that report, how to run it, and how much of it has been checked.

> **Branch.** The report's code, configurations and result summaries are on the
> **`research-validated-experiments`** branch:
>
> ```
> git clone https://github.com/abhiram-v1/AIBSI.git
> cd AIBSI
> git checkout research-validated-experiments
> ```
>
> "Validated" has a limited meaning here, set out in [Validated results](#7-validated-results).
> Twelve headline numbers were recomputed from saved per-participant predictions and matched
> the report exactly. The other reported numbers were checked against the metric files that the
> original runs wrote. **No model was retrained for this check, and no result has been reproduced
> from raw EEG on a second machine.**

---

## 1. Project overview

**Objective.** Classify Alzheimer's disease (AD), frontotemporal dementia (FTD) and cognitively
normal controls (HC/CN) from eyes-closed resting-state scalp EEG, at the level of the
*participant*, and measure how honest the resulting estimates are.

**Problem.** Public EEG cohorts with FTD are small (42 FTD people in total here), the two
cohorts used were recorded with different equipment and processing, and fixed train/test splits
on a few dozen people give unstable, optimistic numbers.

**Approach.**

1. Study I: 88 OpenNeuro participants, spectral and tensor features (PCA/NMF, CP, Tucker,
   Tensor Train) with SVM, random forest, XGBoost and logistic regression, under nested
   participant-level cross-validation (CV).
2. Add BrainLat (Argentina and Chile) by spherical-spline interpolation from 128 to the common
   19 electrodes, giving 174 participants (71 AD, 42 FTD, 61 HC).
3. Log-mel spectrogram features with site-normalised SVMs for the binary (HC vs. dementia) and
   three-class tasks.
4. Tensor decompositions (CP / graph-regularised non-negative CP, Tensor Train, Tucker/HOSVD)
   as a dedicated study, plus an interpretability analysis of their factors.
5. A compact 1D CNN applied directly to the raw waveform, with a per-site audit and an
   OpenNeuro-only control.
6. Reference points: adaptations of four published methods and a pretrained EEG model (EEGPT).

**Main findings** (details in the report): honest repeated-CV three-class macro F1 is about
0.49–0.52; binary HC-vs-dementia macro F1 is 0.699; tensor methods did not beat the
undecomposed spectral baseline; the raw-waveform CNN looks strong on the fixed split
(70.4% accuracy) but its advantage sits at the BrainLat sites and was not confirmed on
OpenNeuro alone.

---

## 2. Repository structure

| Path | Purpose | In the report |
|---|---|---|
| `final_report/` | Report source and PDF, figures, `make_figures.py`, `tensor_interpretability.py` | whole report |
| `verification/` | `recompute_metrics.py`, its output, and the experiment verification summary | Sec. 7 of this README |
| `preprocessing/` | Study I windowing, label-blind QC and balanced 69-person cohort | Report §3 |
| `tensor_pipeline/` | Study I tensor and classifier benchmarks V1–V10 (V1 = top-level scripts) | Report §4 |
| `conventional_ml/` | Early binary feature/ML baseline (pre-dates the tensor study) | not reported |
| `neural_comparison/stft_gpu_3class/` | Merged 174-person harmonisation (BrainLat 128→19 spline interpolation), the fixed 121/26/27 split, first STFT SVM/CNN | Report §5, §8 |
| `neural_comparison/advanced_clean_spectral/` | Main merged-cohort experiments: cleaning streams, log-mel SVM, repeated CV, 2D CNNs, Riemannian/MFCC baselines, EEGPT, TT and Tucker, raw-waveform CNNs | Report §5–§8, §10–§12 |
| `neural_comparison/binary_logmel_search/` | Binary HC-vs-dementia nested search | Report §7 |
| `neural_comparison/dataset_compatibility/` | OpenNeuro–BrainLat compatibility audit, site z-score/CORAL | Report §5.5 |
| `neural_comparison/brainlat_native_ablation/` | BrainLat-only native-128 vs interpolated-19 ablation | Report §5.2 |
| `neural_comparison/graph_ntf_svm/` | CP: NTF and graph-regularised NTF, shuffled-graph and TT controls | Report §11.2, §11.5 |
| `neural_comparison/paper_benchmark/` | Adaptations of Li CP6, Rostamikia, DICE-Net, Jiang microstate cascade | Report §9 |
| `neural_comparison/openneuro_fourier/`, `source88_brainlat_transfer/`, `wav2vec_joint_eeg/` | Further exploratory experiments | not reported |
| `technical_report/` | Earlier course technical report (superseded by `final_report/`) | — |
| `presentation_progress/`, `output/`, `demo_handoff_2026-10-26/` | Slides, exported PDFs/CSVs and an earlier hand-off package | — |
| `docs/PREVIOUS_README.md` | The repository README before this branch | — |

Every experiment folder has its own `README.md` with the exact commands it was run with.
Generated outputs (`outputs/` folders) are git-ignored because they hold multi-GB caches,
checkpoints and per-participant records. This branch adds only **aggregate** evidence from them:
`REPORT.md`/`RESULTS.md`, `metrics.json`, `summary.json`, `protocol.json`, `verification.json`
and figures. This branch does **not add** files containing participant IDs, predictions,
probabilities, splits, feature matrices or model weights. Note that two folders committed
earlier on `master` already contain such files and were left unchanged:
`neural_comparison/graph_ntf_svm/outputs/` (per-fold caches, models and predictions, about
500 MB) and `demo_handoff_2026-10-26/` (participant-level log-mel arrays).

---

## 3. Dataset information

Raw EEG is **not** in this repository and must be downloaded from the original sources.

| Dataset | Content | Where to put it |
|---|---|---|
| [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) v1.0.9 (Miltiadous et al., 2023) | 88 people: 36 AD, 23 FTD, 29 CN; 19 electrodes (10–20), 500 Hz, A1–A2 reference; authors' cleaned EEGLAB derivatives | `dataset/ds004504/` (keep the BIDS layout, including `participants.tsv` and `derivatives/sub-XXX/eeg/*_task-eyesclosed_eeg.set`) |
| [BrainLat](https://www.synapse.org/Synapse:syn51549340) (Prado et al., 2023), EEG folders `1_AD`, `2_bvFTD`, `5_HC` | 128-channel BioSemi, 512 Hz, Argentina (AR) and Chile (CL) | `dataset/brainlat/EEG/1_AD/{AR,CL}/…`, `dataset/brainlat/EEG/2_bvFTD/{AR,CL}/…`, `dataset/brainlat/EEG/5_HC/{AR,CL}/…` |

BrainLat file downloads need a Synapse account with access approval; anonymous download returned
HTTP 403. Of the 100 relevant BrainLat headers, **14 HC recordings have no `.fdt` signal file**
on Synapse and cannot be used, leaving 86 BrainLat people (AR 48, CL 38) and 174 in total.

**Configuring the path.** All current scripts locate data relative to the repository root
(`<repo>/dataset/...`); there is no separate configuration variable. If the data live elsewhere,
create a link instead of editing code, for example on Windows
`mklink /J dataset D:\eeg\dataset` or on Linux/macOS `ln -s /data/eeg dataset`.

**Preprocessing used for the reported results** (report §3 and §5):

* *Study I (88 people):* non-overlapping 4 s windows decimated 500→250 Hz, common-average
  reference (CAR), label-blind QC (|amplitude| ≥ 400 µV, step ≥ 150 µV, channel SD < 1 µV), 0.5 s
  guard around EEGLAB boundary events; scripts in `preprocessing/` and `tensor_pipeline/`.
* *Merged cohort (174 people):* BrainLat 128→19 electrodes with MNE `interpolate_to(..., method="spline")`
  from the `biosemi128` to the `standard_1020` montage; both cohorts band-passed 1–40 Hz,
  resampled to 128 Hz, CAR, 10 s edge trim, non-overlapping 8 s windows, artefact-window
  rejection, at most 40 windows per person. An "advanced" stream adds bad-channel spline repair
  and conservative ocular ICA. STFT 2 s Hann / 0.5 s hop, 24 mel bands 1–40 Hz, window-median
  centring, participant mean/SD → 912 features. Code: `stft_gpu_3class/run_pipeline.py`,
  `advanced_clean_spectral/pipeline.py`, `advanced_clean_spectral/train_moderate_logmel.py`.

---

## 4. Environment setup

The experiments were run with **Python 3.10.11** on Windows 11 with an NVIDIA RTX 5070 Ti Laptop
GPU (12 GB) and PyTorch 2.14.1 built for CUDA 13.0. SVM, tensor and preprocessing steps run on a
CPU; the CNN and EEGPT experiments need a CUDA GPU to finish in reasonable time.

```powershell
py -3.10 -m venv .venv310
.venv310\Scripts\activate
python -m pip install --upgrade pip
# install the PyTorch build that matches your CUDA version first, see pytorch.org
python -m pip install -r requirements.txt
```

The commands in this README and in the experiment READMEs are written as `py -3.10 ...`
(Windows launcher). On Linux/macOS use `python` from the activated environment instead.

**EEGPT (only for report §10 and the OpenNeuro EEGPT blend in §12.6).** Clone the official
repository into `external/EEGPT` (`git clone https://github.com/BINE022/EEGPT.git external/EEGPT`)
and place the checkpoint `eegpt_mcae_58chs_4s_large4E.ckpt`, published by the EEGPT authors, at
`external/EEGPT/checkpoint/eegpt/`. `train_eegpt_attention.py` refuses to run unless the file's
SHA-256 is `9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0`.

---

## 5. Methodology

* **Evaluation unit.** Every metric is computed per participant (window predictions are averaged
  within a person). All splits are by participant; scalers, PCA, decompositions, feature and
  hyperparameter selection are fitted on training participants only. Site normalisation uses
  training-participant statistics per site (AHEPA, AR, CL).
* **Two protocols.** (i) a fixed 121/26/27 train/validation/test split, stratified by
  site × diagnosis (`stft_gpu_3class/run_pipeline.py`, `StratifiedShuffleSplit`, seed 20261007);
  this test set was inspected repeatedly, so its numbers are exploratory. (ii) repeated nested
  participant-level CV (3×5 or 5×5 outer folds with inner selection), the primary evidence.
* **Models.** Linear/RBF SVMs on log-mel features; CP (non-negative, graph-regularised with an
  electrode-graph Laplacian, solved by L-BFGS-B), Tensor Train (TT-SVD with the participant mode
  last), Tucker/HOSVD on windows; 2D CNNs on log-mel/STFT; a 75,463-parameter depth-wise/separable
  1D CNN on raw 19×1024 windows and a 148,524-parameter attention-MIL variant; EEGPT frozen
  embeddings and partial fine-tuning; Riemannian, MFCC, microstate and random-forest baselines.
* **Metrics.** Accuracy, balanced accuracy (mean class recall), macro F1, per-class recall,
  ROC AUC for the binary task. Macro F1 is primary for the binary task because dementia is the
  majority class.

---

## 6. Experiments

Paths below are relative to the repository root. "Evidence" is the committed aggregate file
that holds the run's recorded result. Commands are the ones documented by each experiment;
most scripts take no arguments and write to their own `outputs/` folder.

### Order of execution for the merged cohort

Several scripts read caches written by earlier ones. Run in this order:

```powershell
py -3.10 neural_comparison\stft_gpu_3class\run_pipeline.py all           # harmonise 174 people, fixed split
py -3.10 neural_comparison\advanced_clean_spectral\pipeline.py all        # advanced cleaning stream
py -3.10 neural_comparison\advanced_clean_spectral\train_moderate_logmel.py --epochs 30   # moderate log-mel cache + 2D CNN
py -3.10 neural_comparison\advanced_clean_spectral\experiment_waveform_tucker_before_stft.py --stage prepare   # raw-waveform cache
```

### Experiment index

| # | Experiment (report section) | Code | Command(s) | Evidence |
|---|---|---|---|---|
| 1 | Study I V1–V4 three-class benchmarks (§4) | `tensor_pipeline/run_decomposition_benchmark*.py`, `tensor_pipeline/v2`, `v3_waveform`, `v4_tuning` | see `tensor_pipeline/README.md`; V4: `py -3.10 -u tensor_pipeline\v4_tuning\benchmark_tuning.py` | `tensor_pipeline/outputs/summary/EXPERIMENT_REPORT.md`, `outputs/v4_tuning/EXPERIMENT_REPORT.md` |
| 2 | Study I V6 binary (§4) | `tensor_pipeline/v6_binary/run.py` | `py -3.10 -u tensor_pipeline\v6_binary\run.py` | `tensor_pipeline/outputs/v6_binary/summary.json` |
| 3 | BrainLat-only interpolation ablation (§5.2) | `neural_comparison/brainlat_native_ablation/` | `run.py prepare --workers 2`, `run.py train` | `.../outputs/matched/REPORT.md` |
| 4 | Advanced vs moderate cleaning (§5.3) | `advanced_clean_spectral/audit_preprocessing_feature_svm.py` | `... --repeats 5` | `.../outputs/preprocessing_feature_audit/RESULTS.md` |
| 5 | Cross-cohort compatibility audit (§5.5) | `neural_comparison/dataset_compatibility/` | `prepare.py --workers 2`, `evaluate.py`, `report.py` | `.../outputs/REPORT.md` |
| 6 | Binary HC vs dementia, 174 (§7) | `neural_comparison/binary_logmel_search/` | `run.py prepare --workers 2`, `run.py train`, `run.py finalize` (a fresh run must use a new `--out`, see its README) | `.../outputs/nested/REPORT.md` |
| 7 | Three-class fixed split, site-normalised log-mel SVM (§8.1) | `advanced_clean_spectral/evaluate_site_normalized.py` | `py -3.10 ...\evaluate_site_normalized.py` | `.../outputs/results/site_normalized_metrics.json` |
| 8 | 2D log-mel / log-STFT CNNs (§8.1) | `advanced_clean_spectral/train_deep_models.py`, `train_long_logmel.py` | `--epochs 30`; `--epochs 120 --patience 18` | `.../outputs/deep_models/metrics.json` |
| 9 | Riemannian SVM baseline (§8.1) | `advanced_clean_spectral/train_riemannian_svm.py` | `... all` | `.../outputs/riemannian_svm/metrics.json` |
| 10 | Repeated CV, 50 folds and electrode subsets (§8.2) | `advanced_clean_spectral/analyze_electrode_frequency_stability.py` | no arguments | `.../outputs/electrode_frequency_stability/RESULTS.md` |
| 11 | Compact features, fusion, random forest (§8.2) | `advanced_clean_spectral/evaluate_compact_random_forest.py` | `--repeats 5` | `.../outputs/compact_random_forest/RESULTS.md` |
| 12 | Published-method adaptations (§9) | `neural_comparison/paper_benchmark/` | `prepare.py`, `prepare_native.py dice`, `prepare_native.py micro`, `run.py`, `verify.py` | `.../outputs/full_174/REPORT.md`, `FINAL_REPORT.md`, `summary.json` |
| 13 | EEGPT fine-tuning and SVM blend, fixed split (§10) | `advanced_clean_spectral/prepare_eegpt_raw.py`, `train_eegpt_attention.py`, `compare_eegpt_depth.py`, `evaluate_eegpt_svm_blend.py` | see `advanced_clean_spectral/README.md` | `.../outputs/compact_random_forest/RESULTS.md` (EEGPT rows), `RUN_SUMMARY.md` |
| 14 | Frozen EEGPT repeated CV, three input normalisations (§10) | `extract_frozen_eegpt_embeddings.py`, `evaluate_frozen_eegpt_cv.py` | `extract_... --normalize {none,recording_gain,window_channel}`; `evaluate_... --repeats 5 [--embeddings … --tag …]` | `.../outputs/frozen_eegpt_cv*/frozen_eegpt_cv.json` |
| 15 | CP: NTF, G-NTF, shuffled graph, TT control, repeated CV (§11.2, §11.5) | `neural_comparison/graph_ntf_svm/` | `run.py prepare`, `run.py train --profile full --out neural_comparison/graph_ntf_svm/outputs/full` | `.../outputs/full/REPORT.md` |
| 16 | Tensor Train + SVM, fixed split (§11.3) | `advanced_clean_spectral/train_tensor_train_svm.py` | no arguments | `.../outputs/tensor_train_svm/metrics.json` |
| 17 | Pre-pooling Tucker/HOSVD + SVM (§11.4) | `advanced_clean_spectral/train_prepool_tucker_svm.py` | no arguments | `.../outputs/prepool_tucker_svm/metrics.json` |
| 18 | Waveform Tucker-2 before STFT (§11.4) | `advanced_clean_spectral/experiment_waveform_tucker_before_stft.py` | `--stage prepare`, then `--stage run`, then `--stage report` | `.../outputs/waveform_tucker_before_stft_2026-11-05/REPORT.md` |
| 19 | Tensor interpretability (§11.6) | `final_report/tensor_interpretability.py` | `py -3.10 final_report\tensor_interpretability.py` | `final_report/interp_summary.json`, `final_report/figures/fig_interp_*.pdf` |
| 20 | Raw-waveform 1D CNN, MIL, fusion, refit (§12) | `train_raw_1d_cnn.py`, `evaluate_raw_1d_cnn.py`, `train_raw_1d_cnn_mil.py`, `evaluate_raw_cnn_fusion.py`, `refit_raw_cnn_fusion.py` | no arguments, in this order | `.../outputs/raw_1d_cnn*/metrics.json`, `raw_cnn_svm_fusion*/metrics.json` |
| 21 | Raw CNN five-fold ensemble and per-site audit (§12.4–12.5) | `evaluate_raw_1d_cnn_cv_ensemble.py`, `final_report/make_figures.py` | no arguments | `.../outputs/raw_1d_cnn_cv_ensemble/metrics.json` |
| 22 | OpenNeuro-only native 500 Hz CNN and EEGPT blend (§12.6) | `train_openneuro_native_1d_cnn.py`, `refit_openneuro_native_1d_cnn.py`, `train_openneuro_native_cnn_eegpt_blend.py` | no arguments | `.../outputs/openneuro_native_500hz_1d_cnn/metrics.json`, `refit/metrics.json`, `eegpt_blend/metrics.json` |
| 23 | MFCC SVM baseline (§12.4) | `advanced_clean_spectral/evaluate_mfcc_svm.py` | no arguments | `.../outputs/mfcc_svm/metrics.json` |

Random seeds are fixed in the scripts (for example `SEED = 20261007` for the split; CNN seeds
20261124–20261126 for the OpenNeuro CNN). GPU training is not bit-for-bit deterministic, so CNN
and EEGPT re-runs can differ slightly from the recorded numbers.

---

## 7. Validated results

Status categories used here and in [`verification/VERIFICATION.md`](verification/VERIFICATION.md):

* **Recomputed** — recomputed in this audit (2026-11-21) from the saved per-participant
  predictions with `verification/recompute_metrics.py`; matches the report.
* **Recorded** — the report value equals the value in the metric file written by the original
  run (checked by hand in this audit); not recomputed and not re-run.
* **Not re-run** — no experiment was retrained or re-executed from raw EEG for this branch.

| Result (report section) | Report value | Status |
|---|---|---|
| Li CP6 + naive Bayes, 5-fold (§9) | 60.3% acc., 56.5% BA | Recomputed |
| Rostamikia / DICE-Net / Jiang adaptations (§9) | 55.2/52.5, 51.1/48.3, 41.4/42.7 (acc./BA %) | Recomputed |
| Binary HC vs dementia, 3×5 nested CV (§7) | macro F1 0.699, BA 0.703, acc. 0.720 | Recomputed |
| CP/NTF, G-NTF, shuffled graph, TT, direct; 3×5 CV (§11.5) | BA 46.8, 47.2, 48.5, 50.5, 48.7 % | Recomputed |
| Raw CNN 5-fold ensemble, OOF (147) / test (27) (§12.4) | macro F1 0.643 / 0.667 | Recomputed |
| Site-normalised log-mel SVM, fixed split (§8.1) | 66.7% acc., macro F1 64.3% | Recorded |
| Raw CNN selected seed, test (§12.4) | 70.4% acc., macro F1 69.6% | Recorded |
| Attention-MIL / fusion / refit fusion, test (§12.4) | macro F1 66.4 / 69.3 / 65.5 % | Recorded |
| OpenNeuro-only CNN; refit; EEGPT blend (§12.6) | 42.9%, 35.7%, 50.0% acc. | Recorded |
| Frozen EEGPT CV: unnormalised / gain / window z (§10) | macro F1 60.1, 49.7, 48.9 % | Recorded |
| EEGPT alone and SVM+EEGPT blend, fixed split (§10) | macro F1 55.0 / 67.8 % | Recorded |
| Repeated CV 50 folds; electrode subsets; band importance (§8.2) | 49.0% F1; 8 el. 49.1%; theta 6.1 pp | Recorded |
| Compact features / advanced / fusion / RF (§8.2) | 50.9, 48.9, 51.4, 51.7 % F1 | Recorded |
| TT, pre-pool Tucker, waveform Tucker-2, fixed split (§11) | F1 46.4, 57.6; 51.9 (SVM) / 54.8 (blend) | Recorded |
| Riemannian SVM, MFCC SVM, 2D CNNs (§8.1, §12.4) | 42.3, 58.5, 52.1 % F1 | Recorded |
| BrainLat ablation native128 / native19 / legacy19 / aligned19 (§5.2) | 0.614, 0.566, 0.599, 0.554 | Recorded |
| Compatibility audit (§5.5) | dataset probe 90.6% BA | Recorded |
| Study I V4 tuned SVM / auto; V6 binary (§4) | 59.7 / 54.5 % BA; 75.0% acc., F1 0.718 | Recorded |

The full per-experiment mapping, including evidence paths, is in
[`verification/VERIFICATION.md`](verification/VERIFICATION.md). One report inconsistency was
found and corrected: the waveform-Tucker table row is the SVM alone (51.9%), while the 54.8%
quoted in the text is the SVM+EEGPT blend that was its primary protocol; both are now labelled.

---

## 8. Reproducibility guide

1. **Clone and check out** the branch (box at the top), create the environment (§4).
2. **Run the checks that need no EEG data:**
   ```powershell
   py -3.10 -m unittest discover -s neural_comparison\graph_ntf_svm -p test_pipeline.py
   py -3.10 -m unittest discover -s neural_comparison\binary_logmel_search -p test_pipeline.py
   py -3.10 -m unittest discover -s neural_comparison\brainlat_native_ablation -p test_pipeline.py
   py -3.10 -m unittest discover -s neural_comparison\dataset_compatibility -p test_compatibility.py
   py -3.10 tensor_pipeline\v4_tuning\test_tuning.py
   ```
   All five passed on the authors' machine on 2026-11-21 (4, 5, 5 and 5 unit tests; the V4
   check reports `PASS`).
3. **Get the data** (§3) and place it under `dataset/`.
4. **Rebuild caches** in the order in §6, then run the experiment you want to check. Each script
   writes `metrics.json` / `REPORT.md` to its `outputs/` folder; compare with the committed copy
   of the same file (e.g. `git diff --no-index`).
5. **Recompute metrics from predictions:** `py -3.10 verification\recompute_metrics.py`
   (needs the prediction files the runs write; reports any missing input as `missing input`).
6. **Rebuild the report:** in `final_report/`, run `pdflatex report.tex` twice (MiKTeX or
   TeX Live). Figures: `py -3.10 final_report\make_figures.py` and
   `py -3.10 final_report\tensor_interpretability.py`.

Run times were not recorded systematically. SVM, tensor and preprocessing steps run on a CPU;
CNN and EEGPT training should be run on a CUDA GPU.

---

## 9. Limitations

* **Not re-run end to end.** For this branch nothing was retrained; validation is limited to
  recomputation from saved predictions and comparison with recorded metric files (§7).
* **Data not included.** Raw EEG must be obtained from OpenNeuro and Synapse; 14 BrainLat HC
  recordings are unusable because their signal files are absent upstream.
* **Committed evidence is aggregate only.** Per-participant predictions, splits and caches stay
  local, so `verification/recompute_metrics.py` only runs after the experiments have been run.
* **Known script problem.** `neural_comparison/paper_benchmark/test_pipeline.py` fails on a
  fresh run because it expects `graph_ntf_svm/outputs/full/repeat_00_fold_000/gntf/result.json`,
  a file that the current graph-NTF output layout does not produce. The benchmark itself
  (`run.py`, `verify.py`) does not depend on it.
* **Hard-coded paths fixed.** Eleven Study I scripts in `preprocessing/` and `tensor_pipeline/`
  used `C:\Projects\AIBSI`; on this branch they resolve the repository root from their own
  location, which gives the same paths on the original machine. They were not re-run after the
  change.
* **Small test sets and repeated exposure.** The 27-person fixed test set was inspected many
  times; fixed-split numbers are exploratory. No external validation cohort exists.
* **Cohort confounds.** The cohorts differ in site, amplifier, reference, upstream cleaning and
  age; a classifier still detects the dataset with 90.6% balanced accuracy after harmonisation.
* **GPU non-determinism.** CNN and EEGPT re-runs may differ slightly from recorded values.
* **Earlier materials.** `technical_report/`, `presentation_progress/` and
  `demo_handoff_2026-10-26/` pre-date the final report and are kept for history.
* **History size.** Earlier commits on `master` contain large generated files
  (`graph_ntf_svm/outputs`, presentation build caches), so a full clone is several hundred MB.
  Use `git clone --depth 1 --branch research-validated-experiments …` for a lighter checkout.

---

## 10. Research report reference

* Report: [`final_report/report.pdf`](final_report/report.pdf); source:
  [`final_report/report.tex`](final_report/report.tex).
* The table in §6 maps every report section to its code and evidence; the report's own
  Reproducibility statement points back to this branch.
* Report sections that do not correspond to a single script: the literature review (§2), the
  discussion (§13), limitations (§14) and conclusion (§15).
* Experiments in this repository that the report does not discuss:
  `conventional_ml/`, `neural_comparison/openneuro_fourier/`,
  `neural_comparison/source88_brainlat_transfer/`, `neural_comparison/wav2vec_joint_eeg/`,
  `tensor_pipeline/v5_connectivity` and `v7`–`v10`, and several scripts in
  `advanced_clean_spectral/` (e.g. ComBat, microstate/biomarker SVM, tensor–EEGPT blends).
  Their code and recorded results are included for completeness but were not audited.
