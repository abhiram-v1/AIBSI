# Original88 -> BrainLat transfer test

The earlier75% accuracy /71.8% macroF1 result was nested validation on88
original participants. No all88 final model had been saved. Here the old
36-candidate recipe is selected using source-only3-fold CV and refitted on
all88. Three predeclared models are frozen before target preparation:

- Primary original recipe: spectral ANOVA32 + standard scaling + balanced
  logistic regression C1 (chosen by source-only CV).
- Best source-only SVM: connectivity ANOVA128 + standard scaling + balanced
  RBFSVM C10/gamma scale.
- Spectral SVM control: spectral ANOVA64 + standard scaling + balanced
  RBFSVM C10/gamma scale.

All feature selection, scaling, hyperparameter selection and probability
calibration use the88 source people only. BrainLat never supplies training,
normalization, calibration or threshold estimates. Old default classifier
decision boundaries are preserved. Calibration uses full-pipeline source-only
OOF scores and does not change the hard predictions used for metrics.

The old spectral, power and connectivity definitions are imported from their
original implementations. Replaying the first original participant reproduces
cached feature vectors to floating-point precision. Target conversion uses
the fixed pre-existing legacy128-to19 operator, microvolts,250Hz,4-second
windows and original source QC. The source spectral branch uses19-channel
CAR. The old connectivity/power implementation actually retained the original
acquisition reference; BrainLat cannot recreate AHEPA earlobe references, so
the connectivity SVM has an explicit reference-mismatch limitation.

```powershell
py -3.10 -u neural_comparison/source88_brainlat_transfer/run.py all --workers 2
py -3.10 -u neural_comparison/source88_brainlat_transfer/coverage_diagnostic.py
```

`outputs/source_frozen.json` contains source-only selection, model hashes and
calibration membership. `target_audit.json` lists every local AD/bvFTD/HC header.
Of100 headers,14HC files have missingFDT signal data; authenticated checks of
their corresponding Synapse folders found onlySET headers. Those cannot be
evaluated. Other diagnostic groups are outside this binary task.

Strict original QC permits76 recordings (35AD,19FTD,22HC);10additional readable
HC recordings do not meet the old1uV minimum channel-SD requirement. The
primary result is in `outputs/summary.json`, `REPORT.md`, and `predictions.json`.

For visibility of all86 readable recordings, a separate coverage diagnostic
keeps those76 predictions and adds the10flagged HC files using the previously
established0.1uV floor. Models remain byte-for-byte unchanged. It is explicitly
post-primary, not a new tuned result, and lives under
`outputs/all_readable_diagnostic/`. Every prediction carries a quality flag.
The14missing signals remain unclassified. Original extraction source is
archived in `outputs/initial_source_snapshot/`.

BrainLat has been explored extensively in this project; call this a cross-dataset
transfer test, not an untouched external confirmation. Missingness is entirely
among controls and may bias target representativeness. Probability outputs are
source-calibrated research scores, not clinically validated probabilities.
