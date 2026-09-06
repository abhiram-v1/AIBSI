# Current preprocessing and modeling index

The older [README](README.md) is an original stage record and contains a stale
statement that modeling had not started. It is intentionally preserved. This is
the current index through 2026-09-06.

- [Complete project record](../../PROJECT_MASTER_REPORT.md)
- [Source dataset and design](01_project_and_source_dataset.md)
- [Duration matching and first primary/leftover split](02_duration_organization_and_balancing.md)
- [Signal audit, CAR and controlled primary cohort](03_signal_quality_audit_and_cleaning.md)
- [Original data-use protocol](04_data_usage_protocol_and_next_steps.md)
- [File map and reproducibility](05_reproducibility_and_file_map.md)
- [Corrected TT/Tucker audit](06_tensor_train_followup.md)
- [Full-cohort leftover waveform experiment](07_full_cohort_waveform_experiment.md)

Model reports:

- [First exploratory spectral tensor benchmark](../../tensor_pipeline/outputs/summary/EXPERIMENT_REPORT.md)
- [Corrected nested TT/Tucker benchmark](../../tensor_pipeline/outputs/v2_tensor_train/EXPERIMENT_REPORT.md)
- [Full-cohort waveform benchmark](../../tensor_pipeline/outputs/v3_waveform_full_cohort/EXPERIMENT_REPORT.md)
- [V3 optimization-budget sensitivity](../../tensor_pipeline/outputs/v3_waveform_full_cohort/convergence_sensitivity/REPORT.md)
- [Bounded SVM/XGBoost/random-forest/tensor tuning](../../tensor_pipeline/outputs/v4_tuning/EXPERIMENT_REPORT.md)

Canonical controlled data are `preprocessing/04_clean_primary/primary`.
The later V3/V4 88-person experiments use a separate analysis index over
primary plus leftovers and do not modify the canonical folders.
