# OpenNeuro Fourier/label association audit

Uses the existing global Python3.10 environment, original supplied OpenNeuro derivatives and the fixed88-person cohort. No BrainLat signals enter this experiment.

Run from `C:\Projects\AIBSI`:

```powershell
py -3.10 -X utf8 -m unittest discover -s neural_comparison/openneuro_fourier -p test_fourier.py -v
py -3.10 -X utf8 -u neural_comparison/openneuro_fourier/prepare.py
py -3.10 -X utf8 -u neural_comparison/openneuro_fourier/analyze.py
py -3.10 -X utf8 -u neural_comparison/openneuro_fourier/evaluate.py
py -3.10 -X utf8 neural_comparison/openneuro_fourier/verify_results.py
py -3.10 -X utf8 neural_comparison/openneuro_fourier/report.py
```

Read [the report](outputs/REPORT.md). Outputs are NumPy/JSON, exact feature names, source snapshots and seven Matplotlib PNG figures. No PDF or CSV is generated.

Preparation reuses the compatibility audit's fixed signal policy. Welch/FFT gives frequency power rather than uniquely separating overlapping brain sources. Absolute spectra, relative spectra and spectral descriptors are tested. Full-cohort descriptive association tests are separate from the classifiers; ANOVA feature selection, scaling and SVM settings are fitted inside training folds. MMSE/age/sex are excluded from classifier inputs. Main task is binary HC versus AD+FTD; three-class classification is secondary.

The cohort has been explored in previous project work; these are exploratory repeated nested-CV results, not a new independent clinical validation. Diagnosis-associated features are not necessarily causal disease biomarkers. A low AD/FTD univariate significance count does not prove no difference exists.

Cached evaluation refuses to mix code/protocol changes with frozen folds. Use a new output directory for a changed experiment. Existing source recordings and previous experiments remain unchanged.
