# V5 connectivity, Riemannian and participant-bag experiment

This version is a separate, additive experiment. It does not alter V1–V4 data,
models or reports.

From `C:\Projects\AIBSI` using the existing Python 3.10 installation:

```powershell
py -3.10 tensor_pipeline\v5_connectivity\test_models.py
py -3.10 -u tensor_pipeline\v5_connectivity\prepare_features.py
py -3.10 -u tensor_pipeline\v5_connectivity\benchmark.py
```

The feature builder uses all 16,824 V3-approved windows from all 88 people. It
creates whole-subject, four-bag and eight-bag representations. The benchmark
uses repeated nested participant-level validation, training-only Riemannian
references, training-only feature selection and equal participant/class weight.

Do not interpret a window or bag as an independent person. The same cohort and
outer split seed were previously explored, so a new external cohort remains
necessary to confirm any performance improvement.
