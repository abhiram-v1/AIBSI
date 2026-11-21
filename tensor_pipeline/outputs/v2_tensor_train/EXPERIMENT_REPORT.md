# Second tensor experiment: results and verification

Date: 2026-09-05. CPU / Python 3.10. Previous datasets, experiments and model artifacts were preserved.

## Outcome

The requested Tensor Train run completed, but it did **not** deliver a large accuracy improvement. The best observed individual method family was Tucker/HOSVD at **56.0%** mean subject-level balanced accuracy. The predeclared primary endpoint, selecting the entire method in inner CV, achieved **47.3%**. These are different questions: the best family after inspecting outer results must not replace the primary result.

The earlier exploratory best was 54.1% (PCA/NMF + logistic regression). The nominal 1.9 percentage-point difference is not evidence of a reliable improvement: preprocessing and model selection changed, and the same 69 subjects have already been explored. No claimed 70–80% range was reached.

## Scores

All numbers use held-out **subjects**, not windows. Mean and SD are over three complete five-fold CV repetitions; SD is descriptive split variation, not a confidence interval. Each repetition has 23 subjects per class, so its accuracy and balanced accuracy coincide. Uniform three-class guessing has expected accuracy 33.3%.

| Pipeline | Balanced accuracy, mean ± SD | Macro F1 |
|---|---:|---:|
| Automatic inner-CV method selection (PRIMARY) | 47.3% ± 3.0 pp | 0.470 |
| Window Tucker / HOSVD | 56.0% ± 3.3 pp | 0.560 |
| Tensor Train + spectral features | 54.1% ± 3.3 pp | 0.536 |
| Graph-smoothed Tensor Train | 53.1% ± 1.7 pp | 0.533 |
| Spectral control (no decomposition) | 51.7% ± 3.6 pp | 0.510 |
| Window Tensor Train | 51.2% ± 3.6 pp | 0.509 |
| Subject Tensor Train | 50.2% ± 3.0 pp | 0.495 |

Ranks and classifiers are selected independently within each outer training fold. A family result is not a single fixed classifier evaluated 15 times. Candidates include logistic regression, linear/RBF SVM and histogram gradient boosting. The boosting implementation is **not XGBoost**.

## Class-level performance

Recall averaged across the three held-out repetitions:

| Pipeline | AD recall | FTD recall | CN recall |
|---|---:|---:|---:|
| Automatic inner-CV method selection (PRIMARY) | 46.4% | 33.3% | 62.3% |
| Window Tucker / HOSVD | 59.4% | 50.7% | 58.0% |
| Tensor Train + spectral features | 56.5% | 37.7% | 68.1% |
| Graph-smoothed Tensor Train | 49.3% | 47.8% | 62.3% |
| Spectral control (no decomposition) | 53.6% | 36.2% | 65.2% |
| Window Tensor Train | 46.4% | 43.5% | 63.8% |
| Subject Tensor Train | 49.3% | 34.8% | 66.7% |

For the strongest observed family, Tucker/HOSVD, FTD recall is 50.7%, AD recall is 59.4%, and CN recall is 58.0%. That is more even than several alternatives, but still too weak for a clinical-use claim.

## What was corrected

- Verified the actual electrode row order in all 69 source recordings and rebuilt the anatomical graph.
- Excluded 328 windows near annotated discontinuities from this run only; no stored EEG was deleted.
- Preserved absolute spectral power and variability, and used common training-only TT/Tucker bases.
- Kept all windows of a person within one subject unit; each training subject supplies exactly 24 windows to basis fitting.
- Nested rank, feature selection, scaling, classifier and method-family choices within training subjects.
- Used SVM predictions directly; persisted decomposition cores/bases, classifiers, fold membership and scores.
- Replaced the questionable old graph-CP optimizer with explicitly named graph pre-smoothing followed by TT. The old optimizer itself was not edited or presented as repaired.

See [the detailed audit and mathematical protocol](../../v2/AUDIT_AND_PROTOCOL.md) for qualifications and references.

## Dataset used

| Diagnosis | Subjects | Retained four-second windows | Duration |
|---|---:|---:|---:|
| AD | 23 | 3,985 | 4 h 25 min 40 s |
| FTD | 23 | 4,000 | 4 h 26 min 40 s |
| CN | 23 | 3,957 | 4 h 23 min 48 s |
| Total | 69 | 11,942 | 13 h 16 min 8 s |

The subject-level classes remain exactly balanced. All retained windows contribute to pooled features. Only fitting the window basis uses the 24-per-training-subject calibration subset; the remaining windows were not thrown away. Previously separate leftover/excluded datasets were not merged or reprocessed.

## Selection stability diagnostic

These training/inner scores are descriptive diagnostics, not additional test-set performance estimates.

| Pipeline | Training resubstitution BA | Selected inner validation BA | Outer held-out BA |
|---|---:|---:|---:|
| Automatic inner-CV method selection (PRIMARY) | 95.1% | 65.0% | 47.3% |
| Window Tucker / HOSVD | 96.6% | 57.1% | 56.0% |
| Tensor Train + spectral features | 84.8% | 56.4% | 54.1% |
| Graph-smoothed Tensor Train | 95.9% | 61.7% | 53.1% |
| Spectral control (no decomposition) | 87.2% | 58.3% | 51.7% |
| Window Tensor Train | 96.5% | 62.0% | 51.2% |
| Subject Tensor Train | 91.4% | 60.4% | 50.2% |

Automatic inner-CV family selections across the 15 outer fits: Graph-smoothed Tensor Train: 5; Spectral control (no decomposition): 2; Window Tensor Train: 5; Subject Tensor Train: 2; Window Tucker / HOSVD: 1.

The gap between selected inner validation scores and held-out performance is consistent with unstable model selection on a small cohort. It does not prove a single causal explanation for the low score. Changing the decomposition alone has not solved class separation in these representations.

## Verification and saved artifacts

- Replayed all 105 saved classifiers after freshly reconstructing every held-out subject tensor feature from saved bases.
- Checked all 60 decomposition sets, 45 inner splits and 15 outer splits; no test subject entered basis calibration.
- All replayed predictions and decision scores matched; input clean-signal and metadata SHA-256 fingerprints were unchanged.
- Confirmed finite inputs, unique sample IDs, boundary-exclusion accounting, balanced subjects, training-only centering and orthonormal bases.
- Numerical unit tests compare the TT reconstruction against TensorLy TT-SVD and test full-rank recovery and batch-invariant projection.
- Benchmark fitting time: 161.4 seconds on CPU (excludes preparation and verification). No packages or GPU runtime installed.

Saved classifiers: `nested_benchmark/fold_00_models.joblib` through `fold_14_models.joblib` (seven fitted pipelines each). Their records point to the corresponding decomposition files under `feature_cache/47758609d9ed833b/`. These are fold-trained models, not a separately validated all-subject deployment model. Only load trusted local joblib/pickle files.

Machine-readable outputs: `nested_benchmark/nested_report.json`, all `fold_XX.json` prediction/configuration records, `verification_report.json`, `artifact_manifest.json`, and `preparation_report.json` with exclusion IDs.

## Remaining limits and next decision

This run corrects the identified implementation and evaluation issues; it cannot claim that all scientific limitations are fixed. There are only 69 independent subjects, previous researcher exposure to this cohort, an earlier globally selected/matched subset, and retained preprocessing choices such as common-average reference. Neither nested CV nor more epochs creates an untouched external test cohort.

For the course, retain the original CP/NMF/PCA experiments, document the graph caveat, and compare them with this TT/HOSVD experiment without inventing a performance win. Reconstruction quality and diagnostic separation are distinct objectives.

A subsequent, separately approved experiment could use the full original subject cohort with subject/class weighting and fold-safe access to extra windows, plus a small prespecified representation comparison. Leftover windows from an existing person must not be called an independent test set. That would change the agreed primary/leftover usage policy, so it has not been done here. Keep any external validation data untouched; avoid repeatedly tuning to these same outer scores.
