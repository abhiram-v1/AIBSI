# 5. Reproducibility and file map

## Source and derived data

| Path | Description |
|---|---|
| `dataset/ds004504` | Original BIDS dataset; unchanged |
| `preprocessing/01_duration_sorted/raw_desc` | Duration-ranked links to raw subject EEG folders |
| `preprocessing/01_duration_sorted/preprocessed_desc` | Duration-ranked links to derivative subject EEG folders |
| `preprocessing/01_duration_sorted/by_duration_desc` | Combined pointer index |
| `preprocessing/01_duration_sorted/duration_sorted_manifest.xlsx` | Duration ranking workbook |
| `preprocessing/02_balanced_primary/primary` | First 4,132-per-group balanced window dataset |
| `preprocessing/02_balanced_primary/leftover` | Complete windows not used in that primary set |
| `preprocessing/03_signal_qc_audit` | Read-only audit tables and reports |
| `preprocessing/04_clean_primary/primary` | Current cleaned, 4,090-per-group dataset |
| `preprocessing/04_clean_primary/excluded` | All QC-excluded and balance-trimmed primary windows |

## Main scripts

| Script | Purpose |
|---|---|
| `preprocessing/build_balanced_dataset.py` | Build the first duration-, age-, and sex-aware balanced dataset |
| `preprocessing/audit_primary_signal_quality.py` | Calculate window-, subject-, and channel-level QC metrics |
| `preprocessing/simulate_car_quality_audit.py` | Test common-average reference and final QC rules without changing stored signals |
| `preprocessing/plot_primary_qc_examples.py` | Produce example traces for suspicious windows |
| `preprocessing/build_clean_primary_dataset.py` | Create the final common-average-referenced balanced primary and excluded sets |

## Machine-readable reports

| Report | Contents |
|---|---|
| `preprocessing/02_balanced_primary/selection_plan.json` | Subject matching, duration buckets, and expected primary composition |
| `preprocessing/02_balanced_primary/processing_report.json` | Initial balancing results and partition checks |
| `preprocessing/03_signal_qc_audit/audit_report.json` | Pre-reference signal metric distributions |
| `preprocessing/03_signal_qc_audit/car_diagnostic_report.json` | In-memory common-average reference findings |
| `preprocessing/04_clean_primary/processing_report.json` | Final transform, exclusions, balance trims, and integrity checks |

## Array formats

### Final primary

- `signals.npy`: `float32`, shape `(12270, 19, 1000)`.
- `labels.npy`: aligned class labels, with `A=0`, `F=1`, and `C=2`.
- `metadata.npz`: aligned NumPy metadata fields.
- `manifest.jsonl`: one readable JSON object per window.

Important metadata fields include:

- `sample_id` and `subject`.
- `group`, `label`, and `bucket`.
- `age`, `sex`, and `mmse`.
- original recording/window position.
- `original_array_index` linking back to the first balanced primary array.
- post-reference maximum amplitude, maximum step, and minimum channel standard
  deviation.
- cleaning status and exclusion reason.

### Final excluded set

The excluded directory uses the same structure and signal transform. It contains
126 windows with shape `(126, 19, 1000)`. Exclusion reasons distinguish genuine
signal-QC exclusions from balance-only trims.

## Fixed random seeds

| Operation | Seed |
|---|---:|
| First balanced-primary shuffle | 4504 |
| First leftover shuffle | 4505 |
| Final cleaned-primary shuffle | 4506 |
| Final excluded shuffle | 4507 |

## Reproducibility guarantees already checked

- Original BIDS files were not modified.
- Duration-ranked directories are links/pointers, not independent altered data.
- First primary and first leftover sets do not overlap.
- First-stage complete-window partition is exhaustive.
- Final primary and final excluded sample IDs do not overlap.
- Final primary plus final excluded equals all first-primary windows.
- The stored final signal transformation matches an independent recomputation.
- All final primary samples are finite and below the fixed QC limits.
- All 69 selected subjects remain in the final primary dataset.

## Re-running caution

The final builder intentionally refuses to overwrite an existing
`preprocessing/04_clean_primary` directory. This protects the documented result.
If a future preprocessing policy changes, create a new versioned directory and
report rather than modifying this result in place.

