# BrainLat native-electrode ablation

86 fixed BrainLat participants:35AD,19FTD,32HC. Binary HC versus AD+FTD.
Four matched arms: all128 native analysis channels,19 geometry-selected native
channels,19 legacy-interpolated channels,19 head-aligned-interpolated channels.
The fourth control investigates coordinate/origin handling discovered by inspecting
the installed MNE spline implementation before training.

All arms retain native512Hz and share native128 average reference,1–40Hz filter,
the same accepted8-second epochs (max320seconds/person), common native-window
log-mel gain,1-second Hann STFT with1-second hop,32mel filters, and four
participant summaries (mean, SD, median, IQR). Channel-location selection uses
geometry only. Bad/discontinuity intervals have a2-second exclusion guard.
No arm may change participants or accepted epochs.

```powershell
py -3.10 neural_comparison/brainlat_native_ablation/test_pipeline.py
py -3.10 -u neural_comparison/brainlat_native_ablation/run.py prepare --workers 2
py -3.10 -u neural_comparison/brainlat_native_ablation/run.py train
py -3.10 neural_comparison/brainlat_native_ablation/run.py predict --model neural_comparison/brainlat_native_ablation/outputs/matched/final_native128/model.joblib --subject-id brainlat:AR:sub-30001
py -3.10 neural_comparison/brainlat_native_ablation/run.py predict --model neural_comparison/brainlat_native_ablation/outputs/matched/final_native128/model.joblib --recording C:/path/native128_person.set --site AR
```

Three repeats x five outer folds with three inner folds, matched by participant
across all arms, stratified by site and original AD/FTD/HC diagnosis. Each arm
uses the same256-candidate full grid: training-site z-scoring or none; standard
or robust global scaling; PCA none/10/20/40; C .01/.1/1/10; linear/RBF gamma
scale; unweighted/balanced SVM. Threshold -.5/0/.5 is selected only from inner
OOF scores. Select macroF1, then balanced accuracy. Inner search caps SVM at
200000iterations; nonconverged inner candidates are marked invalid. Refitting
selected models retries with2million and10million iterations if necessary,
without changing the objective, hyperparameters or convergence tolerance.
One selected full-training fit required425661iterations. The initial source
and protocol are archived in `outputs/matched/initial_source_snapshot/` and
`protocol_initial.json`; `solver_retry_amendment.json` records the numerical
retry extension and hashes of already-converged results, which remain unchanged.
Dual PCA uses only the
training Gram matrix and is tested against standard full PCA.

After selection, full-pipeline cross-fitting within outer training fits an
unweighted non-negative-slope sigmoid calibrator. If scores are unreliable and
the fitted slope reaches zero, probabilities revert to training prevalence;
the probability-space operating threshold is then null. Main class predictions
remain based on the inner-selected SVM score threshold. Outer test labels are excluded from all fitting
and selection. Inference prints HC/Dementia probabilities and the operating
threshold (which need not be0.5), not a hard diagnosis. Full-cohort final models
are separate from held-out performance evidence.

Cached operators, coordinates, nearest-sensor assignments and source hashes are
in `outputs/prepared/`. Per-fold selected settings, search scores, participant
IDs, probabilities and models are in `outputs/matched/`. The earlier partial
`outputs/nested/` run stopped during calibration and is not part of these results.
Reports include paired
participant-bootstrap differences, site results and Matplotlib figures. Repeats
are kept together during bootstrap; intervals condition on fitted models.

Interpretation: native128 changes spatial coverage and dimensionality together;
native19 locations only approximate the interpolated targets. Legacy19 reproduces
the prior spatial operator but uses shared native128 reference/normalization here,
so it is not an exact replication of the old mixed-cohort pipeline. Common
reference and window gain use128 channels for every arm. Aligned19 uses source
and target head coordinates minus the same sphere origin, with spline alpha0.
Montages are standard cap coordinates, not individually digitized head shapes.
The86-person cohort inherits prior inclusion decisions; this is exploratory
internal validation, not an independent external or clinical benchmark.
