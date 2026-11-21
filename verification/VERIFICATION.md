# Experiment verification summary

Audit date: 2026-11-21. Branch: `research-validated-experiments`.
Report: `final_report/report.pdf`.

## What was checked, and what was not

* **Recomputed:** metric recomputed from the saved per-participant predictions with
  `verification/recompute_metrics.py` (output: `verification/recomputed_metrics.json`).
  It uses each experiment's own aggregation (for repeated CV: pool the out-of-fold predictions
  of each repeat, score, then average repeats).
* **Recorded:** the report value equals the value written by the original run to the evidence
  file listed. Checked by reading the file; not recomputed.
* **Not re-run:** no model was retrained and no experiment was re-executed from raw EEG
  for this audit. Reproduction from scratch on another machine has not been attempted.

Code checks performed: every Python file in the project folders parses; `compileall` succeeds;
the unit tests of `graph_ntf_svm` (4), `binary_logmel_search` (5), `brainlat_native_ablation` (5)
and `dataset_compatibility` (5) pass; `tensor_pipeline/v4_tuning/test_tuning.py` reports PASS.
`paper_benchmark/test_pipeline.py` fails (missing G-NTF fold file, see Outstanding issues).

## Mapping

| Report § | Experiment | Code | Evidence | Report value | Status |
|---|---|---|---|---|---|
| 4 | V1 exploratory tensor benchmark (69) | `tensor_pipeline/run_decomposition_benchmark*.py`, `tensorize_subjects.py` | `tensor_pipeline/outputs/summary/EXPERIMENT_REPORT.md` | 54.1% BA | Recorded |
| 4 | V2 nested TT/Tucker (69) | `tensor_pipeline/v2/` | `tensor_pipeline/outputs/v2_tensor_train/EXPERIMENT_REPORT.md` | auto 47.3%, Tucker 56.0% BA | Recorded |
| 4 | V3 waveform tensors (88) | `tensor_pipeline/v3_waveform/` | `tensor_pipeline/outputs/v3_waveform_full_cohort/EXPERIMENT_REPORT.md` | auto 52.0%, control 55.8% BA | Recorded |
| 4 | V4 bounded tuning (88) | `tensor_pipeline/v4_tuning/` | `tensor_pipeline/outputs/v4_tuning/EXPERIMENT_REPORT.md` | SVM 59.7%, auto 54.5% BA | Recorded |
| 4 | V6 binary (88) | `tensor_pipeline/v6_binary/run.py` | `tensor_pipeline/outputs/v6_binary/summary.json` | 75.0% acc., F1 0.718 | Recorded |
| 5.2 | BrainLat-only interpolation ablation | `neural_comparison/brainlat_native_ablation/` | `outputs/matched/REPORT.md` | 0.614 / 0.566 / 0.599 / 0.554 | Recorded |
| 5.3 | Moderate vs advanced cleaning | `advanced_clean_spectral/audit_preprocessing_feature_svm.py` | `outputs/preprocessing_feature_audit/RESULTS.md` | −1.98 pp macro F1 | Recorded |
| 5.5 | Compatibility audit | `neural_comparison/dataset_compatibility/` | `outputs/REPORT.md` | probe 90.6% BA; F1 82.4% → 63.9–66.8% | Recorded |
| 7 | Binary HC vs dementia, 3×5 nested CV | `neural_comparison/binary_logmel_search/` | `outputs/nested/REPORT.md`, `oof_predictions.json` (local) | F1 0.699, BA 0.703, acc. 0.720, AUC 0.772 | **Recomputed** (F1, BA, acc.); AUC Recorded |
| 8.1 | Site-normalised log-mel SVM, fixed split | `advanced_clean_spectral/evaluate_site_normalized.py` | `outputs/results/site_normalized_metrics.json` | 66.7% acc., F1 64.3%, BA 64.2% | Recorded |
| 8.1 | 2D log-mel / log-STFT CNNs | `advanced_clean_spectral/train_deep_models.py`, `train_long_logmel.py` | `outputs/deep_models/metrics.json`, `RUN_SUMMARY.md` | 59.3/52.1; long 51.9/40.7 | Recorded |
| 8.1 | Filter-bank Riemannian SVM | `advanced_clean_spectral/train_riemannian_svm.py` | `outputs/riemannian_svm/metrics.json` | 48.1% acc., F1 42.3% | Recorded |
| 8.2 | 50-fold repeated CV; electrode subsets; band importance | `advanced_clean_spectral/analyze_electrode_frequency_stability.py` | `outputs/electrode_frequency_stability/RESULTS.md` | 49.0 ± 6.6; 8 el. 49.1; T8 62%, O2 60%; theta 6.11 pp | Recorded |
| 8.2 | Compact features, fusion, random forest | `advanced_clean_spectral/evaluate_compact_random_forest.py`, `audit_preprocessing_feature_svm.py` | `outputs/compact_random_forest/RESULTS.md`, `outputs/preprocessing_feature_audit/RESULTS.md` | 50.9, 48.9, 51.4, 51.7 | Recorded |
| 9 | Li CP6 + naive Bayes | `neural_comparison/paper_benchmark/` | `outputs/full_174/REPORT.md`, `li_cp6_gnb_oof.json` (local) | 60.3% acc., 56.5% BA | **Recomputed** |
| 9 | Rostamikia hierarchical SVM | same | `rostamikia_hierarchical_svm_oof.json` (local) | 55.2% / 52.5% | **Recomputed** |
| 9 | DICE-Net three-class | same | `dice_net_3class_oof.json` (local) | 51.1% / 48.3% | **Recomputed** |
| 9 | Jiang microstate cascade | same | `jiang_microstate_cascade_oof.json` (local) | 41.4% / 42.7% | **Recomputed** |
| 10 | EEGPT fine-tune; SVM+EEGPT blend (fixed split) | `advanced_clean_spectral/train_eegpt_attention.py`, `evaluate_eegpt_svm_blend.py` | `outputs/compact_random_forest/RESULTS.md` (EEGPT rows), `RUN_SUMMARY.md` | 63.0/55.0; blend 70.4/67.8 | Recorded |
| 10 | EEGPT depth (+2 blocks) | `advanced_clean_spectral/compare_eegpt_depth.py` | `RUN_SUMMARY.md` | val 64.1, test 46.9 | Recorded |
| 10 | Frozen EEGPT 5×5 CV, three input variants | `extract_frozen_eegpt_embeddings.py`, `evaluate_frozen_eegpt_cv.py` | `outputs/frozen_eegpt_cv*/frozen_eegpt_cv.json` | 60.1 ± 7.9, 49.7 ± 7.6, 48.9 ± 8.9; SVM 51.3 ± 9.1 | Recorded |
| 11.5 | Direct / NTF / G-NTF / shuffled / TT, 3×5 CV | `neural_comparison/graph_ntf_svm/` | `outputs/full/REPORT.md`, `oof_predictions.json` | BA 48.7/46.8/47.2/48.5/50.5; F1 45.1/42.3/41.1/–/47.2 | **Recomputed** |
| 11.3 | TT + SVM, fixed split | `advanced_clean_spectral/train_tensor_train_svm.py` | `outputs/tensor_train_svm/metrics.json` | 51.9% acc., F1 46.4% | Recorded |
| 11.4 | Pre-pooling Tucker/HOSVD + SVM | `advanced_clean_spectral/train_prepool_tucker_svm.py` | `outputs/prepool_tucker_svm/metrics.json` | 59.3% acc., F1 57.6% | Recorded |
| 11.4 | Waveform Tucker-2 before STFT | `advanced_clean_spectral/experiment_waveform_tucker_before_stft.py` | `outputs/waveform_tucker_before_stft_2026-11-05/REPORT.md` | SVM 51.9; blend val 81.2 / test 54.8 | Recorded (report wording corrected) |
| 11.6 | Tensor interpretability | `final_report/tensor_interpretability.py` | `final_report/interp_summary.json` | e.g. CP comp. 2 MMSE ρ 0.53 | Regenerated by the script during report writing (descriptive) |
| 12.4 | Raw 1D CNN selected seed / 3-seed mean | `train_raw_1d_cnn.py`, `evaluate_raw_1d_cnn.py` | `outputs/raw_1d_cnn_selected/metrics.json`, `outputs/raw_1d_cnn/metrics.json` | 70.4% acc., F1 69.6; mean 57.5 | Recorded |
| 12.4 | Attention-MIL CNN | `train_raw_1d_cnn_mil.py` | `outputs/raw_1d_cnn_mil/metrics.json` | 66.7% acc., F1 66.4 | Recorded |
| 12.4 | CNN+MIL+SVM fusion; refit | `evaluate_raw_cnn_fusion.py`, `refit_raw_cnn_fusion.py` | `outputs/raw_cnn_svm_fusion*/metrics.json` | 70.4/69.3; refit 66.7/65.5 | Recorded |
| 12.4 | Five-fold CNN ensemble OOF / test | `evaluate_raw_1d_cnn_cv_ensemble.py` | `outputs/raw_1d_cnn_cv_ensemble/metrics.json`, `probabilities.npz` (local) | F1 0.643 / 0.667 | **Recomputed** |
| 12.4 | MFCC SVM | `evaluate_mfcc_svm.py` | `outputs/mfcc_svm/metrics.json` | 59.3% acc., F1 58.5 | Recorded |
| 12.5 | Per-site breakdown of the CNN ensemble | `final_report/make_figures.py` | `final_report/figures/fig_site_audit.pdf` | BrainLat HC 32/32; 96.5 vs 77.3% | Computed from saved probabilities during report writing |
| 12.6 | OpenNeuro-only native CNN; refit; EEGPT blend | `train_openneuro_native_1d_cnn.py`, `refit_...`, `train_openneuro_native_cnn_eegpt_blend.py` | `outputs/openneuro_native_500hz_1d_cnn/{metrics,refit/metrics,eegpt_blend/metrics}.json` | 42.9%, 35.7%, 50.0% acc. | Recorded |

(`outputs/...` paths in the code column's experiment folder; `advanced_clean_spectral` means
`neural_comparison/advanced_clean_spectral`. Files marked "(local)" hold per-participant
predictions and are intentionally not committed.)

## Mismatches between report and code/evidence

1. **Waveform Tucker-2 (§11.4–11.5).** The tensor table listed 51.9% macro F1 (SVM alone) while
   the text quoted 54.8% (the SVM+EEGPT blend that was the experiment's primary protocol) without
   saying so. Fixed in the report: the table row is labelled "SVM only (no EEGPT)" and the text
   names the blend.
2. No other numerical mismatch was found among the values checked above.

## Outstanding issues

* `neural_comparison/paper_benchmark/test_pipeline.py` expects
  `neural_comparison/graph_ntf_svm/outputs/full/repeat_00_fold_000/gntf/result.json`, which the
  saved graph-NTF output does not contain, so the script stops with `FileNotFoundError`.
* Eleven Study I scripts were changed from a hard-coded `C:\Projects\AIBSI` to a repository-relative
  root on this branch; they were compiled but not re-run.
* Reproducing from raw EEG requires BrainLat access via Synapse and the EEGPT checkpoint.
* Several scripts read caches produced by earlier scripts (order in README §6); running one
  script in isolation on a fresh clone will fail until those caches exist.
* Fixed-split results depend on a 27-person test set that was inspected repeatedly.
