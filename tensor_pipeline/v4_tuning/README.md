# Bounded tensor/classifier tuning

- [Results](../outputs/v4_tuning/EXPERIMENT_REPORT.md)
- [Protocol](PROTOCOL.md)
- [Verification](../outputs/v4_tuning/verification_report.json)

This experiment uses exactly the V3 88-person cohort and inputs, preserving
all earlier data/models. XGBoost CPU 3.2.0 is installed in the existing Python
3.10 user environment; no virtual environment is needed.

From `C:\Projects\AIBSI`:

```powershell
py -3.10 tensor_pipeline\v4_tuning\test_tuning.py
py -3.10 -u tensor_pipeline\v4_tuning\benchmark_tuning.py
py -3.10 tensor_pipeline\v4_tuning\verify_report.py
```

The benchmark resumes only signature-compatible saved folds. The verification
script rebuilds held-out features from saved bases, replays model predictions,
checks training provenance/scaling and exports/reloads XGBoost boosters. It
does not fit additional classifiers or select another winner.

For a fresh reproduction, use a separate project copy with an empty V4 output
directory. Keep completed experiments intact. Scientific changes should receive
their own version/output location. Package versions and code/data fingerprints
are recorded in the artifact manifest.

Models in `outputs/v4_tuning/models/fold_XX.joblib` include ten selected pipeline
entries and subject membership; some entries reference the same fitted model.
They point to their corresponding tensor bases in `bases/`. XGBoost UBJ files
are portable booster exports, not complete predictors without the original
feature projection, selection and scaling. Only load trusted local joblib files
with this directory and the V3 directory importable.

The primary endpoint selects the complete pipeline inside inner CV. Classifier-
family and representation-family rows are secondary comparisons. Reusing a
previously explored cohort remains an independent-validation limitation even
when the new run's fitting/tuning is correctly nested.
