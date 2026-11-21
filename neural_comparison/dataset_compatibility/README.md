# OpenNeuro–BrainLat compatibility audit

From `C:\Projects\AIBSI`, using the existing global Python3.10 installation:

```powershell
py -3.10 -m unittest discover -s neural_comparison/dataset_compatibility -p test_compatibility.py -v
py -3.10 -u neural_comparison/dataset_compatibility/prepare.py --workers 2
py -3.10 -u neural_comparison/dataset_compatibility/evaluate.py
py -3.10 -u neural_comparison/dataset_compatibility/followup.py
py -3.10 -u neural_comparison/dataset_compatibility/report.py
```

Read [the generated report](outputs/REPORT.md). Outputs are NumPy/JSON and Matplotlib PNG; no PDF or spreadsheet export is required.

The audit uses the fixed174 usable participants from the prior experiment, the existing BrainLat128→19 interpolation operator, and the original derivative recordings. It adds a matched reference/filter stage, per-record normalization ablations, training-only site harmonization, repeated nested binary classification, dataset probes and source-only cross-cohort transfer. It does not overwrite the source recordings or previous experiments.

Preparation is cached by code/operator/source-file hashes. Evaluation freezes its code and protocol and refuses to mix changed protocols with cached folds. Remove nothing from previous experiments: use a fresh output location for a changed model protocol. `followup.py` is explicitly exploratory and was motivated by the amplitude separation seen in the main audit.

Every learned transform, PCA and SVM is fitted inside the appropriate training fold. A normalized full-cohort feature file must not be substituted for these training-only transformations. No harmonized matrix is claimed to remove all biological or acquisition differences.

The exact amplitude-only BrainLat-to-OpenNeuro transfer is a separate experiment:

```powershell
py -3.10 -u neural_comparison/dataset_compatibility/amplitude_transfer.py run
py -3.10 neural_comparison/dataset_compatibility/amplitude_transfer.py predict --rms-uv 1.3 6.5
```

The prediction command accepts positive RMS values already measured with the same19-channel matched preparation policy and prints both class probabilities. It does not load a raw recording. The frozen source scaler/SVM and source-only probability calibrator are reused on subsequent runs. OpenNeuro cohort statistics are never fitted.

Read [the amplitude-only transfer report](outputs/amplitude_transfer_brainlat_to_openneuro/REPORT.md). It includes all88 test-person predictions and source-calibrated probabilities, the frozen model, source selection/calibration folds, a Matplotlib figure and verification records. `interpretation.json` records the fitted linear decision boundary (approximately2.70µV), below every OpenNeuro participant's measured amplitude. All88 are consequently predicted as dementia: macro F1 40.1%, balanced accuracy50%, healthy recall0%. This is a transfer failure of the amplitude rule; it does not identify the origin of the BrainLat separation or establish author intent.
