# 3. Signal-quality audit and cleaned-primary construction

## Purpose

The first balanced primary dataset was audited before time-frequency or tensor
construction. The audit was read-only and checked amplitude, channel variance,
repeated values, abrupt changes, inter-channel correlation, slow drift,
frequency-band power, and residual 50 Hz energy.

## Audit findings

### Operations that did not need repeating

- No non-finite windows were present.
- No meaningful flat-line or repeated-value problem was found.
- Residual 50 Hz power was negligible because the supplied derivatives were
  already low-pass filtered at 45 Hz. No additional notch filter was added.
- ASR and ICA were not repeated.

### Shared-reference problem

The median fraction of window variance carried by the signal common to all 19
channels was approximately `0.953`. The 95th percentile was approximately
`0.986`. Correspondingly, the median absolute inter-channel correlation was
very high.

A naive channel-correlation or “bridging” rule would therefore have rejected
most of the dataset. That rule was rejected as inappropriate: the pattern was
consistent with the common A1–A2 reference and volume conduction, not thousands
of independent channel failures.

Because channel relationships will be used for graph construction, the signal
was re-referenced to the common average across the 19 scalp channels at every
time sample. Per-channel temporal means were then removed again within each
window.

### Abrupt transients

Before common-average re-referencing, many of the largest jumps appeared across
several electrodes at the same instant. Re-referencing removed much of this
shared component. Remaining abrupt changes were then assessed on the signal that
will actually be modeled.

`sub-086` was the main concentration of these events. The entire subject was not
removed: 39 contaminated windows were excluded while 105 clean windows were
retained, preserving the FTD subject count.

### Persistent 30–45 Hz activity

Subjects `sub-067`, `sub-077`, and `sub-085`, all FTD, showed persistently high
30–45 Hz power after common-average re-referencing. This may represent residual
muscle/electrode contamination, genuine subject variation, or both. Removing
windows purely on this frequency ratio would disproportionately remove FTD data
and could create label-dependent bias.

No sample was therefore rejected from 30–45 Hz power alone. The agreed analysis
policy is:

- Primary tensor analysis: 1–30 Hz.
- Sensitivity analysis: include 30–45 Hz and check whether conclusions change.
- Report subject-level sensitivity for the three high-frequency FTD subjects.

## Final signal-QC rules

Rules were fixed, label-blind, and applied after common-average referencing:

| Check | Threshold | Action |
|---|---:|---|
| Maximum absolute amplitude | `>= 400 µV` | Exclude window |
| Abrupt step over one 4 ms interval | `>= 150 µV` | Exclude window |
| Extreme abrupt step | `>= 300 µV` | Record stronger reason |
| Minimum channel standard deviation | `< 1 µV` | Exclude window |

Some windows triggered more than one reason, so reason counts are not unique
window counts.

## Signal-QC outcome

Sixty-five unique windows were excluded by the signal rules:

| Group | Before QC | Signal-QC exclusions | Remaining |
|---|---:|---:|---:|
| AD | 4,132 | 9 | 4,123 |
| FTD | 4,132 | 42 | 4,090 |
| CN | 4,132 | 14 | 4,118 |

Recorded reasons across these windows were:

- 52 abrupt-step flags between 150 and 300 µV/4 ms.
- 9 extreme abrupt-step flags at or above 300 µV/4 ms.
- 8 extreme-amplitude flags.
- 1 near-flat-channel flag after re-referencing.

## Restoring exact group balance

FTD remained the limiting group at 4,090 clean windows. To restore exact balance,
33 AD and 28 CN windows were moved to the excluded dataset. These 61 windows were
not artifacts; they were balance-only trims.

The FTD artifact loss was concentrated in the medium-duration bucket. Therefore,
balance trims were taken from the corresponding medium-duration AD and CN
buckets. They were distributed nearly equally across all seven medium-bucket
subjects in each group. Within a subject, the highest remaining label-blind
amplitude/step scores were trimmed first.

## Final cleaned primary dataset

| Group | Windows | Duration | Subjects | Short | Medium | Long |
|---|---:|---:|---:|---:|---:|---:|
| AD | 4,090 | 16,360 s | 23 | 1,156 | 1,188 | 1,746 |
| FTD | 4,090 | 16,360 s | 23 | 1,159 | 1,183 | 1,748 |
| CN | 4,090 | 16,360 s | 23 | 1,155 | 1,193 | 1,742 |
| **Total** | **12,270** | **49,080 s** | **69** | **3,470** | **3,564** | **5,236** |

The final array has shape `(12270, 19, 1000)`. It was shuffled with seed `4506`.
All 69 selected subjects remain represented.

## Final excluded dataset

The 65 signal-QC exclusions and 61 balance-only trims were stored after the same
common-average transformation. The excluded dataset contains 126 windows,
exactly 42 from each diagnostic group, and was shuffled with seed `4507`.

Nothing was deleted. Every source-primary window occurs exactly once in either
the final primary or final excluded dataset.

## Verification results

- Primary labels: 4,090 AD, 4,090 FTD, and 4,090 CN.
- Primary and excluded sample IDs do not overlap.
- Their union equals all 12,396 source-primary windows.
- All values are finite.
- Independent recomputation produced zero numerical difference from the stored
  common-average transform for tested windows.
- Maximum residual temporal channel mean was about `7.8e-6 µV`.
- Maximum residual across-channel mean was about `6.1e-5 µV`.
- No retained window crosses the final amplitude, step, or variance thresholds.

