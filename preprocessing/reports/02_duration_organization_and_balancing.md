# 2. Duration organization and balanced-primary construction

## Duration-sorted organization

The 88 subjects were organized from longest to shortest recording duration.
The original BIDS dataset was not moved or copied. Directory junctions and path
pointers were created as non-destructive views under
`preprocessing/01_duration_sorted`.

The duration workbook contains overall and per-group rankings. Folder names use
the format:

`<rank>__<subject>__<group>__<duration-seconds>s`

The overall longest recording is `sub-010` (AD), 1,291.1 seconds. The overall
shortest is `sub-003` (AD), 307.1 seconds.

| Group | Longest | Duration | Shortest | Duration |
|---|---|---:|---|---:|
| AD | `sub-010` | 1,291.1 s | `sub-003` | 307.1 s |
| FTD | `sub-074` | 1,026.9 s | `sub-070` | 479.1 s |
| CN | `sub-040` | 1,017.1 s | `sub-060` | 751.5 s |

## Window construction

The artifact-cleaned derivative recordings were converted into modeling windows
using the following fixed procedure:

- Downsample from 500 Hz to 250 Hz by retaining every second sample. This is
  acceptable because the supplied derivatives are already low-pass filtered at
  45 Hz, well below the new 125 Hz Nyquist frequency.
- Divide each recording into non-overlapping four-second windows.
- Store each window with shape `(19, 1000)`.
- Remove the temporal mean independently from each channel/window.
- Do not normalize across subjects.
- Preserve incomplete terminal fragments separately.

The initial conservative scan flagged non-finite data, channels with standard
deviation below 0.1 µV, and absolute amplitudes above 500 µV. Flagged windows
were retained outside the primary dataset.

## Available complete windows before balancing

| Group | Subjects | Complete windows | Initially QC-approved | Initially flagged |
|---|---:|---:|---:|---:|
| AD | 36 | 7,267 | 7,252 | 15 |
| FTD | 23 | 4,137 | 4,132 | 5 |
| CN | 29 | 6,015 | 6,007 | 8 |

FTD supplied 4,132 approved windows and was the limiting reference. It was not
oversampled or synthetically expanded.

## Duration buckets

FTD subjects were sorted by recording duration and divided by rank into three
approximately equal subject groups:

| Bucket | Duration rule | FTD subjects | Target windows/group |
|---|---|---:|---:|
| Short | `<= 647.6 s` | 8 | 1,160 |
| Medium | `647.6–806.45 s` | 7 | 1,222 |
| Long | `> 806.45 s` | 8 | 1,750 |

These thresholds preserve the natural FTD coverage rather than imposing
arbitrary equal-width duration intervals.

## Subject matching and window selection

Twenty-three AD subjects and 23 CN subjects were matched to the 23 FTD subjects.
A mixed-integer optimization favored sufficient window capacity, similar age and
sex, and similar recording duration. It enforced 12 male AD and 13 male CN
subjects. The FTD group naturally contained 14 male subjects.

For selected AD and CN recordings, windows were chosen evenly across the full
recording instead of taking only the beginning or a single contiguous segment.
All initially approved FTD windows were retained.

The resulting subject composition was:

| Group | Subjects | Windows | Sex, M/F | Age, mean ± SD | Age range |
|---|---:|---:|---:|---:|---:|
| AD | 23 | 4,132 | 12/11 | 65.74 ± 7.35 | 49–79 |
| FTD | 23 | 4,132 | 14/9 | 63.65 ± 8.22 | 44–78 |
| CN | 23 | 4,132 | 13/10 | 66.87 ± 5.24 | 57–78 |

The first balanced primary dataset therefore contained 12,396 windows and
49,584 seconds of EEG. Its shape was `(12396, 19, 1000)`. It was shuffled with
seed `4504` so group and bucket order were removed.

## Leftovers from the first balancing stage

No complete window was discarded. The separate leftover array contains 5,023
windows:

| Group | Leftover windows | Main causes |
|---|---:|---|
| AD | 3,135 | Unselected subjects, excess matched windows, initial QC flags |
| FTD | 5 | Initial QC flags |
| CN | 1,883 | Unselected subjects, excess matched windows, initial QC flags |

Of these, 4,995 passed the initial conservative QC and 28 were amplitude-flagged.
Eighty-eight incomplete terminal fragments totaling 155.12 seconds were also
preserved separately.

The first balanced primary and leftover arrays form a complete, non-overlapping
partition of all complete four-second windows produced at that stage.

