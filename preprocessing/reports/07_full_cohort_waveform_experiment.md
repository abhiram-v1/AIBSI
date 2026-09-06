# Full-cohort waveform experiment with leftovers

Date: 2026-09-06. The user authorized using the earlier leftover partition
alongside primary data, with class/subject weighting for FTD imbalance.
Original arrays and all older model results remain unchanged.

| Diagnosis | People | Usable four-second windows |
|---|---:|---:|
| AD | 36 | 7,030 |
| FTD | 23 | 4,000 |
| CN | 29 | 5,794 |
| Total | 88 | 16,824 |

Included: 4,828 windows from the old leftover partition, all 11,942 V2 windows,
and 54 valid windows removed only for exact balancing. The 595 rejected windows
retain IDs/reasons and remain in their source arrays. No formerly artifact-
excluded primary window was restored. Partial tails remain separate.

Both partitions receive the same reference and signal-quality rules. Every
person remains entirely within training or test. Training-fold class weights
give each diagnosis equal total loss mass; within-person window averaging
prevents longer recordings from multiplying a person's training weight.
No FTD cases are duplicated or synthesized.

The original nested-selection endpoint was 52.0% balanced accuracy. Waveform
families scored 49.9–51.8%; the same-cohort spectral control scored 55.8%.
Supervised fits reached their 100-iteration budget, so a separate fixed-budget
sensitivity check was added. It does not replace the original primary result.

- [Full results and learning curve](../../tensor_pipeline/outputs/v3_waveform_full_cohort/EXPERIMENT_REPORT.md)
- [Waveform and weighting protocol](../../tensor_pipeline/v3_waveform/PROTOCOL.md)
- [Optimization-budget sensitivity](../../tensor_pipeline/outputs/v3_waveform_full_cohort/convergence_sensitivity/REPORT.md)
- [Verification](../../tensor_pipeline/outputs/v3_waveform_full_cohort/verification_report.json)

This experiment does not establish a performance ceiling. Changes versus V2
include participant count, window coverage, representation and fold seed, so
score differences cannot be attributed solely to tensorization.
