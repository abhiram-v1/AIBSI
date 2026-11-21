# Supervised TT: optimization-budget sensitivity

This is a post-hoc diagnostic motivated by all original supervised fits exhausting 100 iterations. It preserves the original experiment and does not replace its primary score. The head penalty is fixed at 0.1, already specified for the original learning curve; no model setting is selected using these outcomes.

| Iteration cap | Mean balanced accuracy ± repeat SD | Ordinary accuracy | Optimizer convergence across 25 fits |
|---|---:|---:|---:|
| 100 | 55.0% ± 0.7 pp | 57.2% | 0/25 |
| 2000 | 53.7% ± 2.3 pp | 55.3% | 3/25 |

The accuracy rows use all 88 held-out subjects once per repetition, across the same three five-fold repetitions. The 25 optimization counts also include ten smaller training subsets. SD is not a confidence interval.

| Training fraction | 100-iteration held-out BA | 2000-iteration held-out BA |
|---|---:|---:|
| 40% | 42.0% | 54.5% |
| 70% | 40.9% | 40.5% |
| 100% | 55.2% | 51.1% |

Learning-curve rows use only the first CV repetition. They cannot establish a required sample size. The fixed-model 100-iteration mean differs from the original nested supervised-family mean because that original family could select penalty 0.01 or 0.1 inside each training fold.

All 50 fitted models and their scores were saved and replayed. Subject memberships and class-weight totals were checked. A convergence flag, when present, does not establish a global optimum for this nonconvex model. Remaining budget exhaustion is explicitly reported rather than hidden.

No further settings were changed after examining this sensitivity result. See [the original full experiment](../EXPERIMENT_REPORT.md) for the primary endpoint and all other methods.
