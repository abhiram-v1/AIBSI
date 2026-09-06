# Tensor Train follow-up and boundary audit

The second model experiment is complete and preserved separately. The original
BIDS dataset, approved cleaned primary arrays, earlier exclusions, leftover
dataset, and first model experiment were not replaced.

The modeling audit found EEGLAB `boundary` annotations in the source derivatives.
For the new experiment only, a fixed 0.5-second guard around each annotation
excluded 328 of the 12,270 primary windows. Their IDs and reasons are recorded;
their original EEG remains available in the unchanged primary array.

| Diagnosis | Subjects | Windows used in V2 |
|---|---:|---:|
| AD | 23 | 3,985 |
| FTD | 23 | 4,000 |
| CN | 23 | 3,957 |
| Total | 69 | 11,942 |

All subjects remain, with equal class counts. Window counts were not forcibly
rebalanced after this additional quality rule because the classifiers operate
on subject-level features. Each training subject contributes 24 distributed
windows to fitting a window basis, and all their retained windows contribute
to their pooled representation. There is no duplication or oversampling.

Actual electrode row order was verified against all source files and used to
correct anatomical graph alignment. The earlier interpretation of strong common
signals as necessarily reference artifacts was also qualified: correlation
alone does not establish artifact origin. Common-average reference was retained
for this controlled comparison, not proved uniquely optimal.

- [Full results and remaining limitations](../../tensor_pipeline/outputs/v2_tensor_train/EXPERIMENT_REPORT.md)
- [Audit and methodology](../../tensor_pipeline/v2/AUDIT_AND_PROTOCOL.md)
- [Verification and data accounting](../../tensor_pipeline/outputs/v2_tensor_train/verification_report.json)

Tucker/HOSVD was the strongest observed new family at 56.0% balanced accuracy;
the nested automatic-selection endpoint was 47.3%. This was a methodological
improvement, not an established large classification improvement. Previously
separated leftover samples have not been merged or used for this run.
