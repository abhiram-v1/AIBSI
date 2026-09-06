# NeuroTensor EEG preprocessing reports

**Project:** Graph-Regularized Tensor Decomposition for Dementia EEG  
**Dataset:** OpenNeuro `ds004504`, local version `1.0.9`  
**Report date:** 2026-09-05

This folder records the dataset decisions and processing completed so far. It is
intended to make the work reproducible and to prevent later experiments from
quietly changing the cohort, preprocessing, or validation rules.

## Current status

The signal-cleaned balanced primary dataset is complete and verified:

- 69 subjects: 23 AD, 23 FTD, and 23 CN.
- 12,270 non-overlapping four-second windows.
- 4,090 windows per diagnostic group.
- 19 EEG channels, 250 Hz, 1,000 samples per channel/window.
- Common-average reference across the 19 scalp channels.
- No additional filtering after the supplied derivative preprocessing.
- Every excluded or balance-trimmed window is preserved separately.

The time-frequency representation, graph construction, tensor decomposition,
classification, and cross-validation experiments have **not** yet been run.

## Report index

1. [Project and source dataset](01_project_and_source_dataset.md)
2. [Duration organization and balanced-primary construction](02_duration_organization_and_balancing.md)
3. [Signal-quality audit and cleaned-primary construction](03_signal_quality_audit_and_cleaning.md)
4. [Data-use protocol and next steps](04_data_usage_protocol_and_next_steps.md)
5. [Reproducibility and file map](05_reproducibility_and_file_map.md)

## Canonical data locations

| Purpose | Location |
|---|---|
| Original BIDS dataset | `dataset/ds004504` |
| Duration-sorted, non-copying views | `preprocessing/01_duration_sorted` |
| First balanced window dataset | `preprocessing/02_balanced_primary` |
| Read-only signal-quality audit | `preprocessing/03_signal_qc_audit` |
| Current cleaned balanced dataset | `preprocessing/04_clean_primary/primary` |
| Windows excluded from the cleaned dataset | `preprocessing/04_clean_primary/excluded` |

The current input for the next modeling stage is
`preprocessing/04_clean_primary/primary`. Earlier directories remain provenance
records and must not be overwritten.

