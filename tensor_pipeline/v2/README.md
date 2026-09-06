# NeuroTensor: second, Tensor Train experiment

The run is complete. Everything for this experiment is separate from the first
benchmark. No original dataset, earlier features, or earlier model outputs were
deleted or replaced.

## Start here

- [Results, class recalls, limitations and artifact locations](../outputs/v2_tensor_train/EXPERIMENT_REPORT.md)
- [Audit corrections and complete methodology](AUDIT_AND_PROTOCOL.md)
- [Machine-readable verification](../outputs/v2_tensor_train/verification_report.json)
- [Complete nested scores](../outputs/v2_tensor_train/nested_benchmark/nested_report.json)
- [Code, array and model fingerprints](../outputs/v2_tensor_train/artifact_manifest.json)

The best observed family was window Tucker/HOSVD (56.0% balanced accuracy).
Tensor Train variants scored 50.2–54.1%. Automatic method selection inside
nested CV scored 47.3%, the predeclared primary endpoint. The run did not
establish a large performance gain or clinical usability.

## Recheck the completed experiment

From `C:\Projects\AIBSI`, using the existing Python 3.10 installation:

```powershell
py -3.10 tensor_pipeline\v2\test_models.py
py -3.10 tensor_pipeline\v2\verify_and_report.py
```

The first command checks TT numerics against TensorLy and basic inductive
projection properties. The second verifies 15 outer folds, 45 inner folds,
60 fitted decomposition sets, and all 105 saved classifier pipelines. It
rebuilds every held-out tensor feature from saved bases, reproduces predictions,
checks input fingerprints and writes the report. It does not fit new models.

## Pipeline sequence used

```powershell
py -3.10 -u tensor_pipeline\v2\prepare.py
py -3.10 -u tensor_pipeline\v2\benchmark.py --repeats 3 --max-folds 15
py -3.10 -u tensor_pipeline\v2\verify_and_report.py
```

Do not rerun preparation over completed artifacts just to view the results.
The benchmark resumes compatible saved folds and caches rather than refitting
them. For a genuinely fresh reproduction, use a separate project copy with an
empty V2 output directory. Keep this completed experiment intact. A scientific
change should receive a new version/output location, not silently reuse V2.

Runtime package versions are recorded in the artifact manifest. No virtual
environment, PyTorch, CUDA, or new package installation was used. The tensor
decompositions use NumPy/SciPy; classifiers use scikit-learn; TensorLy provides
an independent TT-SVD numerical test. Histogram gradient boosting is not
misrepresented as XGBoost.

## Saved model structure

`outputs/v2_tensor_train/nested_benchmark/fold_XX_models.joblib` contains a
dictionary of the six evaluated method families plus the inner-selected
pipeline. Each entry stores the fitted classifier pipeline, selected settings,
and path to the matching decomposition file.

The decomposition files in `feature_cache/47758609d9ed833b/` contain TT cores,
orthonormal bases, training means, Tucker factors, anatomical graph/smoother,
training subject IDs and calibration window indices. The neighboring `.npz`
files hold projected features. Do not move classifiers without their bases and
input provenance. Load trusted joblib files with this directory on Python's
import path so the `models` module can be found.

These are evaluation-fold models, not a single separately validated deployment
model. Their outer predictions correspond to subjects excluded from fitting.
Prior exposure to this same cohort still limits independent generalization
claims, even though the new fitted stages and tuning are nested correctly.
