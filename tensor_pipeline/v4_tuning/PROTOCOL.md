# Bounded optimization and classifier comparison

Date: 2026-09-06. This fourth experiment responds to the request to try parameter
optimization and other classifiers, especially XGBoost. All earlier datasets,
fitted models and outcomes are preserved in their existing directories.

## Unchanged data and evaluation units

Use exactly the V3 full cohort: 88 people (36 AD, 23 FTD, 29 CN), 16,824 eligible
four-second windows, and the same waveform delay moments and spectral controls.
No additional exclusions, resampling changes or transformations are applied to
the source EEG. The original input fingerprints are checked before fitting.

Outer evaluation uses the same three repeated five-fold subject splits as V3,
seed 9060. Inner evaluation uses three stratified subject folds with seed
90600 + outer index. Every person is wholly in training or test. Training-only
class weights N/(3*N_class) equalize total class loss; within-person window
averaging prevents long recordings from increasing the person's weight.
No oversampling, duplication, SMOTE, MMSE or demographic predictors are used.

## What changes

Seven candidate feature settings:

- Waveform TT log-energy features, terminal ranks 8 or 16, channel rank 6.
- Waveform Tucker log-energy features, channel/lag ranks (4,4) or (6,6).
- Spectral controls with training-only ANOVA selection of 16 or 32 features.
- TT16 plus 418 spectral controls, with training-only ANOVA selection of 32
  combined features. Selection can favor mostly or entirely spectral features;
  it does not guarantee that the final classifier uses the tensor features.

Each is paired with 19 classifier settings:

- Logistic regression: C = 0.01, 0.1, 1.
- RBF SVM: C = 0.1, 1, 10; gamma = scale or 0.01.
- XGBoost: eight fixed conservative combinations of depth 1–3, 100–250 trees,
  learning rate 0.03–0.1, minimum child weight 1–5, L2 penalty 1–10, L1 penalty
  0–0.5, row subsampling 0.8–1 and column subsampling 0.8. Exact combinations
  are recorded in `protocol.json`; they are not an expanded Cartesian sweep.
- Random forest: 200 trees, depth 3/minimum leaf 3 or unrestricted depth/minimum
  leaf 5, square-root feature subsampling and fixed seed 9064.

There are six supervised TT settings, changing the previously weakly anchored
core optimization: channel ranks 2 or 4, filters 4 or 8, head penalty 0.1 or 1,
core anchor 0.1 or 1 (previously 0.001), and a fixed 500-iteration budget. These
are six specified combinations, not every Cartesian combination. Gradients for
the altered anchor objective are checked numerically before benchmarking.

Total: **139 prespecified configurations**. No additional settings are added
after examining new outer results. There is no validation/test-driven early
stopping; tree counts and iteration budgets are predetermined settings. Budget
exhaustion is reported rather than silently called convergence.

## Software

XGBoost was not previously installed in the working Python 3.10 interpreter.
The official `xgboost-cpu==3.2.0` package was installed in the persistent user
installation, without dependency upgrades, PyTorch, CUDA, or a virtual
environment. The wheel was approximately 2.1 MB. Earlier histogram gradient
boosting results are not relabeled as XGBoost results.

XGBoost uses CPU histogram trees and multiclass soft probabilities, but metrics
use the model's class predictions. Multiclass sample weights handle imbalance;
binary `scale_pos_weight` is not used. SVM decision values are not described as
probabilities. Package versions and model fingerprints are retained.

## Selection and reporting

Decomposition bases, ANOVA selection, scaling and classifier fitting occur only
on the appropriate training subjects. Inner folds refit bases from scratch.
Feature banks are reused across classifiers only when training membership is
identical. Basis files include their training IDs and code/data signature.

The primary endpoint is the held-out balanced accuracy of the entire pipeline
chosen by inner validation. Selection maximizes inner out-of-fold balanced
accuracy, then macro F1, with deterministic lower-complexity exact tie breaking.

Secondary comparisons independently select the best configuration inside each
classifier family and each representation family. All are evaluated on outer
held-out subjects. Per-family best outer scores must not replace the primary
endpoint after inspection. Ordinary accuracy, balanced accuracy, macro F1 and
per-class recall are reported separately because class counts differ.

All 15 outer fits are saved, with ten pipeline entries each, plus 45 inner and
15 outer decomposition sets. Inner-selected entries can reference the same
fitted model; 150 saved entries do not imply 150 independent models. XGBoost
booster exports supplement, not replace, their preprocessing/basis pipelines.

## Limits

Reusing the same splits helps compare this tuning procedure with V3 but does
not provide a new untouched test cohort. The researcher has already seen results
on these people. Nested fitting protects this run's internal tuning; it cannot
erase that adaptive exposure. A small apparent gain is not automatically a
statistically reliable improvement or evidence of clinical readiness.

The waveform models remain pooled second-order delay-filter models with a
120-ms lag span, not unrestricted long-sequence raw EEG networks. Spectral
controls retain the earlier 1–30 Hz convention, while waveform moments retain
approximately 0.5–45 Hz source content. More tuning cannot by itself resolve
these representation limits or add independent FTD participants.

## References

- [Official XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html)
- [Official XGBoost parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [V3 waveform methodology](../v3_waveform/PROTOCOL.md)
