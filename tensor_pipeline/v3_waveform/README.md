# Waveform tensors with the full cohort

This is a separate experiment using both original primary and leftover windows,
plus recovery of valid balance-only trims. All earlier datasets/models remain.

- [Results](../outputs/v3_waveform_full_cohort/EXPERIMENT_REPORT.md)
- [Detailed protocol and limitations](PROTOCOL.md)
- [Verification](../outputs/v3_waveform_full_cohort/verification_report.json)
- [Input provenance](../outputs/v3_waveform_full_cohort/input_fingerprints.json)
- [Separate fixed-budget optimization check](../outputs/v3_waveform_full_cohort/convergence_sensitivity/REPORT.md)

## Verify completed results

Run from `C:\Projects\AIBSI` with the existing Python 3.10 installation:

```powershell
py -3.10 tensor_pipeline\v3_waveform\test_wave_models.py
py -3.10 tensor_pipeline\v3_waveform\verify_report.py
```

The first checks the tensor mathematics and analytic gradients. The second
replays saved models, validates splits/provenance and regenerates reports;
it does not fit new models. Joblib files must be trusted local files. Keep
`wave_models.py` importable when loading them.

## Original run sequence

```powershell
py -3.10 -u tensor_pipeline\v3_waveform\prepare.py
py -3.10 tensor_pipeline\v3_waveform\test_wave_models.py
py -3.10 -u tensor_pipeline\v3_waveform\run.py
py -3.10 tensor_pipeline\v3_waveform\verify_report.py
```

Preparation refuses to overwrite its output directory. The benchmark verifies
input fingerprints and resumes only compatible completed folds. For a fresh
reproduction use a separate project copy with no V3 output directory. A changed
scientific protocol should receive a new version/output location; do not delete
or silently replace this completed experiment.

The waveform arms use exact sufficient statistics of pooled raw delay-filter
energy, not PSD inputs. This is a specific second-order waveform model, not a
claim to preserve all raw time information. The supervised TT learns its two
input-mode cores and classifier head jointly. SciPy handles optimization, so
no PyTorch/CUDA installation or virtual environment is required.

`convergence_check.py` is a separately labeled post-hoc sensitivity run for
the already specified penalty-0.1 supervised model, comparing 100 versus 2000
iterations on identical subject splits. It preserves all original outcomes
and saves its own protocol/models/report under `convergence_sensitivity/`.

Each outer model file contains four method families plus a reference to the
inner-selected pipeline, with training/test subject membership. These are
fold-trained evaluation models, not a single externally validated deployment
model. The 15 learning-curve model files use one fixed configuration.
