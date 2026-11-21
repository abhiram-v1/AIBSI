# Binary log-mel SVM, 174 participants

HC=0 (61 people); dementia=1 (71 AD + 42 FTD =113 people). This is a new standalone
binary experiment, preserving the earlier 88-person binary and 174-person three-class results.

From `C:\Projects\AIBSI`, using global Python3.10:

```powershell
py -3.10 neural_comparison/binary_logmel_search/test_pipeline.py
py -3.10 -u neural_comparison/binary_logmel_search/run.py prepare --workers 2
py -3.10 -u neural_comparison/binary_logmel_search/run.py train
py -3.10 neural_comparison/binary_logmel_search/run.py predict --model neural_comparison/binary_logmel_search/outputs/nested/final/model.joblib --subject-id ds004504:sub-001
py -3.10 neural_comparison/binary_logmel_search/run.py predict --model neural_comparison/binary_logmel_search/outputs/nested/final/model.joblib --recording C:/path/new_person.set --source ds004504 --site AHEPA
```

The completed `outputs/nested` validation retains its original source snapshot in
`validation_source_snapshot/`. A later numerical fix adds a mathematically
equivalent `gesvd` fallback when full PCA's default `gesdd` SVD fails to converge.
This occurred during final-model selection after all15 validation folds had
completed. Saved validation predictions are unchanged. Final refit has its own
source hashes in `final/protocol.json`. The fallback is tested against ordinary
full PCA. To finish only deployment fitting, use:

```powershell
py -3.10 -u neural_comparison/binary_logmel_search/run.py finalize
```

For a fresh full validation with the updated source, use a new output directory
(`train --out neural_comparison/binary_logmel_search/outputs/nested_v2`), since
resuming a run with different source hashes is intentionally rejected.

The native recordings are spatially harmonized to19 channels, average-referenced,
filtered1–40Hz and resampled directly to256Hz. A matched128Hz control is downsampled
from this256Hz signal. Resampling does not create independent participants or
additional biological bandwidth. Windows crossing marked bad/discontinuous
intervals with a2-second guard are rejected. The existing harmonizer filters the
full recording; upstream artifact cleaning differs between datasets. At most320
accepted seconds are used per person, at every segment length, with no segment
overlap. All174 participants must have at least3 accepted windows for every
representation; preparation raises instead of silently changing the cohort.

The frozen256-candidate seeded search covers:

| Parameter | Choices |
|---|---|
| Sampling rate |128,256Hz|
| Recording segment |2,4,6,8seconds|
| STFT analysis window |1,2,3seconds|
| STFT hop |.25,.5,1second; require at least3 frames|
| Mel filters |16,24,32 spanning1–40Hz|
| Participant summaries |mean/SD or mean/SD/median/IQR|
| Site adjustment |none, training-site centering, training-site z-score|
| Global scaler |standard or robust|
| PCA |none,20,40,60 components (training-size capped)|
| SVM kernel |linear or RBF|
| C |log-uniform .001–100 plus fixed anchors|
| RBF gamma |scale,.001,.01,.1,1|
| Class weights |none, balanced, HC:Dementia1.5:1 or2.5:1|
| Decision-score threshold |−.5,0,.5 selected on inner OOF predictions|

This covers every valid sampling/segment/STFT/hop/mel combination at least once,
with random classifier settings and extra matched anchors. It is not exhaustive
over the full Cartesian product. All candidates are frozen before outer testing.

Validation:3repeats ×5outer folds;3inner folds, stratified by site and original
AD/FTD/HC diagnosis. Every person has exactly one test prediction per repeat.
Candidate selection maximizes pooled inner out-of-fold macroF1, then balanced
accuracy. Site statistics, scaling, PCA and SVM are refit inside every inner fold.
Deterministic, label-independent within-person spectral summaries may be cached
before splitting. No SMOTE or synthetic subjects are used.

The selected score threshold is applied to outer predictions. Sigmoid probability
calibration uses fresh full-pipeline cross-fitted scores inside outer training.
Probability0.5 need not equal the selected decision threshold. Main metrics use
the selected operating point; probability-argmax metrics are a separate endpoint.
Inference prints both class probabilities and the saved operating threshold.

Fixed128Hz and256Hz controls differ only in sampling rate and share outer folds,
with8s segment,2sSTFT,.5shop,24mel,mean/SD,training-site zscore,standard scale,
PCA40,balanced RBFSVM C1/gamma scale,threshold0. Controls are not tuned using test
results. Per-site and original-diagnosis detection rates accompany aggregate scores.

Outputs: `prepared/plan.json`, per-person preparation audits and cached features;
`nested/protocol.json`, per-fold searches/models/predictions, `summary.json`,
`REPORT.md`, Matplotlib figures, `final/model.joblib` and final SVM input matrix.
Data/code hashes prevent accidental resume with a different protocol. Full-cohort
final fitting is for deployment, not independent performance measurement.

The cohort has already been explored in earlier experiments. This is internal
validation, not an untouched external or clinical benchmark. Bootstrap intervals
condition on fitted models and keep each participant's repeated predictions together.
