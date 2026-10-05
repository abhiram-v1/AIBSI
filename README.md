# AIBSI — AI-Based Brain-Signal Identification

**Subject-level dementia classification from resting EEG using tensor decomposition.**

This project explores distinguishing Alzheimer''s disease (AD), frontotemporal dementia (FTD), and cognitively normal controls (CN) from eyes-closed resting EEG at the subject level. It is course/research work — not a clinical diagnostic system.

---

## Results at a glance

| Experiment | Cohort | Primary auto-selection | Best observed |
|---|---|---:|---:|
| V1 — Spectral tensor benchmark | 69 subjects | 54.1% BA | 54.1% (PCA + logistic) |
| V2 — Corrected nested TT/Tucker | 69 subjects | 47.3% BA | 56.0% (Tucker/HOSVD) |
| V3 — Full-cohort waveform tensors | 88 subjects | 52.0% BA | 55.8% (spectral control) |
| V4 — Bounded classifier/optimizer tuning | 88 subjects | 54.5% BA | 59.7% (tuned RBF SVM) |

> **BA** = balanced accuracy (averages AD, FTD, CN recall). Chance = 33.3%.
> FTD recall is the persistent weak point across all experiments (~30–43%).
> No independent external validation has been performed.

---

## Dataset

- **Source:** OpenNeuro `ds004504` v1.0.9
- **Participants:** 88 people — 36 AD, 23 FTD, 29 CN
- **Signals:** 19 scalp electrodes, 500 Hz, eyes-closed resting EEG
- **Pre-processing supplied by dataset authors:** ~0.5–45 Hz bandpass, A1–A2 mastoid reference, ASR, ICA, ICLabel artifact rejection

The raw dataset lives in `dataset/ds004504/` locally and is not tracked in Git
(about 5.8 GB). Obtain it from [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504)
before running data-dependent scripts. Generated experiment arrays and model
checkpoints are also excluded; reports and source code are included.

---

## Repository structure

```
AIBSI/
├── preprocessing/              # Dataset construction and quality control
│   ├── build_balanced_dataset.py
│   ├── build_clean_primary_dataset.py
│   ├── audit_primary_signal_quality.py
│   ├── simulate_car_quality_audit.py
│   ├── plot_primary_qc_examples.py
│   └── reports/                # Markdown reports for each preprocessing step
│       ├── 00_CURRENT_INDEX.md
│       ├── 01_project_and_source_dataset.md
│       ├── 02_duration_organization_and_balancing.md
│       ├── 03_signal_quality_audit_and_cleaning.md
│       ├── 04_data_usage_protocol_and_next_steps.md
│       ├── 05_reproducibility_and_file_map.md
│       ├── 06_tensor_train_followup.md
│       └── 07_full_cohort_waveform_experiment.md
│
├── tensor_pipeline/            # Tensor decomposition and classification experiments
│   ├── tensorize_subjects.py
│   ├── run_decomposition_benchmark.py
│   ├── summarize_benchmark.py
│   ├── diagnose_feature_bottleneck.py
│   │
│   ├── v2/                     # Experiment 2 — corrected nested TT/Tucker benchmark
│   │   ├── prepare.py
│   │   ├── benchmark.py
│   │   ├── models.py
│   │   ├── verify_and_report.py
│   │   ├── test_models.py
│   │   └── AUDIT_AND_PROTOCOL.md
│   │
│   ├── v3_waveform/            # Experiment 3 — full-cohort waveform tensors
│   │   ├── prepare.py
│   │   ├── run.py
│   │   ├── wave_models.py
│   │   ├── convergence_check.py
│   │   ├── finalize.py
│   │   ├── verify_report.py
│   │   ├── test_wave_models.py
│   │   └── PROTOCOL.md
│   │
│   ├── v4_tuning/              # Experiment 4 — bounded classifier/optimizer tuning
│   │   ├── tune_models.py
│   │   ├── benchmark_tuning.py
│   │   ├── verify_report.py
│   │   ├── test_tuning.py
│   │   └── PROTOCOL.md
│   ├── v5_connectivity/        # Additive connectivity experiment code and protocol
│   └── v6_binary/              # Experimental binary classification script
│
├── presentation_progress/     # Slides, literature review, illustrations
├── technical_report/          # LaTeX source and compiled technical report PDF
└── PROJECT_MASTER_REPORT.md    # Consolidated record through V4
```

---

## Preprocessing pipeline

Preprocessing is non-destructive — original BIDS files are never modified.

1. **Duration inventory** — all 88 recordings ranked longest to shortest; four-second windows at 250 Hz extracted.
2. **Balanced primary cohort (69 subjects)** — all 23 FTD subjects kept as the reference; 23 AD and 23 CN matched by duration. Result: 12,396 windows (4,132/class).
3. **Signal QC audit** — common-average re-reference, fixed amplitude/step/flatness thresholds. 65 windows excluded; clean cohort = **12,270 windows / 69 subjects**.
4. **Boundary guarding (V2+)** — 0.5 s guard around EEGLAB `boundary` annotations; 328 windows excluded, leaving 11,942 windows for V2.
5. **Full-cohort extension (V3+)** — primary + leftover windows re-QC''d consistently; **16,824 windows / 88 subjects**.

Scripts: [`preprocessing/`](preprocessing/)
Reports: [`preprocessing/reports/`](preprocessing/reports/)

---

## Experiments

### V1 — Exploratory spectral tensor benchmark

**Input:** 69-subject clean primary cohort
**Tensor shape:** `(69, 19, 59, 10)` — subject × channel × frequency × time-decile
**Methods:** PCA, NMF, non-negative CP, graph-NCP, non-negative Tucker at ranks 3/5/8 × logistic / RBF SVM / gradient boosting under 3×5 subject CV
**Best:** PCA rank-8 + logistic = **54.1% balanced accuracy**

Scripts: [`tensor_pipeline/`](tensor_pipeline/)

---

### V2 — Corrected nested TT/Tucker benchmark

**Input:** 69-subject boundary-safe index (11,942 windows)
**Representation:** signed log10 absolute PSD tensor `19 × 59 × 3` per window (channel × frequency × segment)
**Methods:** subject TT, window TT, graph-smoothed TT, window Tucker/HOSVD, TT+spectral fusion — ranks 8/16/32 and (6,8,3); fully nested 3×5 outer / 3-fold inner subject CV
**Primary automatic selection:** 47.3% BA | **Best family observed:** Tucker/HOSVD **56.0% BA**

Scripts: [`tensor_pipeline/v2/`](tensor_pipeline/v2/)
Protocol: [`AUDIT_AND_PROTOCOL.md`](tensor_pipeline/v2/AUDIT_AND_PROTOCOL.md)

---

### V3 — Full-cohort waveform tensor analysis

**Input:** 88-subject re-QC''d index (16,824 windows)
**Representation:** `19 × 16 × 485` delay-covariance tensors (channel × lag × local position) — raw waveform second-order statistics at 125 Hz
**Methods:** waveform TT, waveform Tucker, supervised TT (joint), same-cohort spectral controls — 3×5 outer / 3-fold inner subject CV
**Primary automatic selection:** 52.0% BA | **Spectral control:** 55.8% BA

Scripts: [`tensor_pipeline/v3_waveform/`](tensor_pipeline/v3_waveform/)
Protocol: [`PROTOCOL.md`](tensor_pipeline/v3_waveform/PROTOCOL.md)

---

### V4 — Bounded classifier/optimizer tuning

**Input:** Same 88-subject inputs and outer splits as V3
**New additions:** XGBoost (CPU, shallow trees), random forest, stronger-regularized supervised TT; 139 configurations pre-frozen before results were inspected
**Primary automatic selection:** 54.5% ± 3.2 pp BA | **Best family observed:** tuned RBF SVM **59.7% ± 0.9 pp BA**

Scripts: [`tensor_pipeline/v4_tuning/`](tensor_pipeline/v4_tuning/)
Protocol: [`PROTOCOL.md`](tensor_pipeline/v4_tuning/PROTOCOL.md)

---

## Key design safeguards

- **Subject-level splits** — no window from a person appears in both train and test.
- **Nested CV** — decompositions, feature selection, scaling, and classifiers are all fit only on training subjects.
- **No data leakage** — no SMOTE, no demographic predictors, no test-set early stopping.
- **Reproducibility** — input fingerprints, saved model artifacts, and prediction replays are verified after every experiment.
- **No overwriting** — each experiment version writes to its own output directory; earlier results are preserved.

---

## Running the code

Requires **Python 3.10** with `numpy`, `scipy`, `scikit-learn`, `tensorly`, `pandas`, `matplotlib`. V4 additionally needs `xgboost>=3.2.0`.

```powershell
# Verify V2 experiment (no refitting)
py -3.10 tensor_pipeline\v2\test_models.py
py -3.10 tensor_pipeline\v2\verify_and_report.py

# Verify V3 experiment (no refitting)
py -3.10 tensor_pipeline\v3_waveform\test_wave_models.py
py -3.10 tensor_pipeline\v3_waveform\verify_report.py

# Verify V4 experiment (no refitting)
py -3.10 tensor_pipeline\v4_tuning\test_tuning.py
py -3.10 tensor_pipeline\v4_tuning\verify_report.py
```

> **Note:** To reproduce from scratch, use a separate project copy with an empty output directory. Do not overwrite completed experiment artifacts.

---

## Limitations

- All 88 source subjects have been explored across multiple experiments. No independent external cohort exists; researcher-level adaptive exposure limits generalization claims even with nested CV.
- FTD is the hardest class: only 23 FTD participants exist in the entire source dataset. More windows do not add independent FTD patients.
- Tensor decomposition was not consistently superior to spectral baselines.
- Results should not be interpreted as establishing a diagnostic accuracy ceiling or clinical usability.

---

## Full documentation

| Topic | Document |
|---|---|
| Complete experiment record | [`PROJECT_MASTER_REPORT.md`](PROJECT_MASTER_REPORT.md) |
| Preprocessing reports | [`preprocessing/reports/`](preprocessing/reports/) |
| V2 corrections and methodology | [`tensor_pipeline/v2/AUDIT_AND_PROTOCOL.md`](tensor_pipeline/v2/AUDIT_AND_PROTOCOL.md) |
| V3 waveform protocol | [`tensor_pipeline/v3_waveform/PROTOCOL.md`](tensor_pipeline/v3_waveform/PROTOCOL.md) |
| V4 tuning protocol | [`tensor_pipeline/v4_tuning/PROTOCOL.md`](tensor_pipeline/v4_tuning/PROTOCOL.md) |
| V5 connectivity protocol | [`tensor_pipeline/v5_connectivity/PROTOCOL.md`](tensor_pipeline/v5_connectivity/PROTOCOL.md) |
| V6 binary experiment | [`tensor_pipeline/v6_binary/README.md`](tensor_pipeline/v6_binary/README.md) |
| Latest status presentation | [`presentation_progress/output/NeuroTensor_Status_Presentation_10_Slides_EEG_Pipeline.pptx`](presentation_progress/output/NeuroTensor_Status_Presentation_10_Slides_EEG_Pipeline.pptx) |
| Literature review | [`presentation_progress/output/Literature_Review_EEG_Dementia.pdf`](presentation_progress/output/Literature_Review_EEG_Dementia.pdf) |
| Technical report | [`technical_report/main.pdf`](technical_report/main.pdf) |

---

## License

This is academic/course research. The source EEG dataset (`ds004504`) is governed by its [OpenNeuro license](https://openneuro.org/datasets/ds004504). Code in this repository is released for research reference only.
