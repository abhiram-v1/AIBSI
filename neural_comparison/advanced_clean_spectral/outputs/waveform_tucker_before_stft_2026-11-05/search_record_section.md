
## Waveform Tucker-2 reconstruction BEFORE STFT + fine-tuned EEGPT: 174 participants

All 126,000 saved candidates, including the no-decomposition arm; the rows below use only the 113,400 Tucker-reconstruction candidates (9 rank pairs x 30 SVM settings x 420 blends). Scores are **validation (26 participants) ensemble selection scores, not test performance**. Training-only factors/SVM; EEGPT unchanged. Each row's macro F1, BA and accuracy come from the same winning configuration; other parameters vary. Actual test results are kept separately in [the experiment report](C:/Projects/AIBSI/neural_comparison/advanced_clean_spectral/outputs/waveform_tucker_before_stft_2026-11-05/REPORT.md).

Source: [all_validation_candidates.jsonl.gz](C:/Projects/AIBSI/neural_comparison/advanced_clean_spectral/outputs/waveform_tucker_before_stft_2026-11-05/all_validation_candidates.jsonl.gz).

| Parameter | Value | Best macro F1 (%) | BA / macro recall (%) | Accuracy (%) | Matching saved configs | Winning candidate |
|---|---|---|---|---|---|---|
| SVM C | 0.03 | 81.24 | 80.11 | 80.77 | 22,680 | c19_d8/svm 10/blend 136 |
| SVM C | 0.1 | 79.64 | 79.05 | 80.77 | 22,680 | c19_d8/svm 12/blend 115 |
| SVM C | 0.3 | 79.64 | 79.05 | 80.77 | 22,680 | c19_d8/svm 14/blend 115 |
| SVM C | 1.0 | 79.64 | 79.05 | 80.77 | 22,680 | c19_d8/svm 16/blend 115 |
| SVM C | 3.0 | 79.64 | 79.05 | 80.77 | 22,680 | c19_d8/svm 18/blend 115 |
| Waveform Tucker electrode rank | 12 | 79.64 | 79.05 | 80.77 | 37,800 | c12_d32/svm 14/blend 10 |
| Waveform Tucker electrode rank | 19 | 81.24 | 80.11 | 80.77 | 37,800 | c19_d8/svm 10/blend 136 |
| Waveform Tucker electrode rank | 8 | 79.20 | 79.05 | 80.77 | 37,800 | c8_d8/svm 12/blend 116 |
| Waveform Tucker delay rank (L=32) | 16 | 76.56 | 75.34 | 76.92 | 37,800 | c8_d16/svm 10/blend 31 |
| Waveform Tucker delay rank (L=32) | 32 | 79.64 | 79.05 | 80.77 | 37,800 | c12_d32/svm 14/blend 10 |
| Waveform Tucker delay rank (L=32) | 8 | 81.24 | 80.11 | 80.77 | 37,800 | c19_d8/svm 10/blend 136 |
| EEGPT temperature | 0.75 | 79.64 | 79.05 | 80.77 | 28,350 | c19_d8/svm 12/blend 95 |
| EEGPT temperature | 1.0 | 81.24 | 80.11 | 80.77 | 28,350 | c19_d8/svm 10/blend 116 |
| EEGPT temperature | 1.25 | 81.24 | 80.11 | 80.77 | 28,350 | c19_d8/svm 10/blend 136 |
| EEGPT temperature | 1.5 | 81.24 | 80.11 | 80.77 | 28,350 | c19_d8/svm 10/blend 241 |
| SVM kernel | linear | 81.24 | 80.11 | 80.77 | 56,700 | c19_d8/svm 10/blend 136 |
| SVM kernel | rbf | 76.09 | 75.34 | 76.92 | 56,700 | c19_d16/svm 7/blend 31 |
| SVM PCA components | 20 | 76.56 | 75.34 | 76.92 | 37,800 | c12_d8/svm 8/blend 10 |
| SVM PCA components | 40 | 81.24 | 80.11 | 80.77 | 37,800 | c19_d8/svm 10/blend 136 |
| SVM PCA components | None | 72.93 | 73.44 | 73.08 | 37,800 | c8_d32/svm 25/blend 115 |
| SVM temperature | 0.5 | 81.24 | 80.11 | 80.77 | 22,680 | c19_d8/svm 10/blend 71 |
| SVM temperature | 0.75 | 81.24 | 80.11 | 80.77 | 22,680 | c19_d8/svm 10/blend 136 |
| SVM temperature | 1.0 | 81.24 | 80.11 | 80.77 | 22,680 | c19_d8/svm 10/blend 241 |
| SVM temperature | 1.5 | 81.24 | 80.11 | 80.77 | 22,680 | c19_d8/svm 10/blend 327 |
| SVM temperature | 2.0 | 81.24 | 80.11 | 80.77 | 22,680 | c19_d8/svm 10/blend 413 |
| SVM blend weight | 0.0 | 64.13 | 66.19 | 69.23 | 5,400 | c19_d16/svm 19/blend 0 |
| SVM blend weight | 0.05 | 64.13 | 66.19 | 69.23 | 5,400 | c19_d16/svm 19/blend 1 |
| SVM blend weight | 0.1 | 64.13 | 66.19 | 69.23 | 5,400 | c19_d16/svm 19/blend 2 |
| SVM blend weight | 0.15000000000000002 | 64.31 | 66.19 | 69.23 | 5,400 | c12_d16/svm 19/blend 66 |
| SVM blend weight | 0.2 | 67.63 | 69.52 | 73.08 | 5,400 | c19_d8/svm 10/blend 46 |
| SVM blend weight | 0.25 | 67.63 | 69.52 | 73.08 | 5,400 | c19_d8/svm 10/blend 26 |
| SVM blend weight | 0.30000000000000004 | 73.97 | 74.29 | 76.92 | 5,400 | c12_d16/svm 12/blend 69 |
| SVM blend weight | 0.35000000000000003 | 76.15 | 75.71 | 76.92 | 5,400 | c19_d8/svm 12/blend 70 |
| SVM blend weight | 0.4 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 71 |
| SVM blend weight | 0.45 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 156 |
| SVM blend weight | 0.5 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 136 |
| SVM blend weight | 0.55 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 116 |
| SVM blend weight | 0.6000000000000001 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 201 |
| SVM blend weight | 0.65 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 307 |
| SVM blend weight | 0.7000000000000001 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 287 |
| SVM blend weight | 0.75 | 81.24 | 80.11 | 80.77 | 5,400 | c19_d8/svm 10/blend 372 |
| SVM blend weight | 0.8 | 76.56 | 75.34 | 76.92 | 5,400 | c12_d8/svm 8/blend 352 |
| SVM blend weight | 0.8500000000000001 | 74.00 | 74.29 | 76.92 | 5,400 | c8_d8/svm 0/blend 269 |
| SVM blend weight | 0.9 | 74.00 | 74.29 | 76.92 | 5,400 | c8_d8/svm 0/blend 354 |
| SVM blend weight | 0.9500000000000001 | 72.39 | 71.64 | 73.08 | 5,400 | c12_d8/svm 8/blend 19 |
| SVM blend weight | 1.0 | 72.39 | 71.64 | 73.08 | 5,400 | c12_d8/svm 8/blend 20 |
