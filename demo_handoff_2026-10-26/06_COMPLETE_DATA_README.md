# Complete prepared dataset for the combined-cohort demo

The `06_logmel_full_part_01.npz` through `part_10.npz` files contain **all 6,721 prepared log-mel EEG windows from 174 participants** in the moderate-cleaning branch of the OpenNeuro + BrainLat analysis. The files are split into roughly 8 MB chunks for easier Drive transfer. This is the complete derived model input for that branch; it is **not raw EEG** and does not include the separate EEGPT raw-window input.

- `log_mel`: `float16`, shape `(6721, 19, 24, 13)` after combining; axes are window, electrode, mel band, time frame.
- `labels`: diagnosis code `0=AD`, `1=FTD`, `2=HC`.
- `subject_index`: integer pointer to a row in `06_complete_data_manifest.json` → `participants`. The manifest has source, site, pseudonymous ID, diagnosis, and window count. Local file paths were removed.
- Participant counts: 71 AD, 42 FTD, 61 HC. All 174 participants are represented.

For a quick inspection, load any chunk with `numpy.load(..., allow_pickle=False)`. To restore the three `.npy` arrays used by the project, download all 10 chunks, the JSON manifest, and `06_reconstruct_full_data.py` into one folder, then run `py -3.10 06_reconstruct_full_data.py` with NumPy installed. The script checks each chunk's SHA-256 before joining it.

**Evaluation rule:** split by participant, never by window. A window-level random split would leak the same person's EEG into training and testing. These files contain both development and previously evaluated participants; do not use them to claim a fresh independent test score. The complete raw source recordings are available from their original providers under their respective access terms.
