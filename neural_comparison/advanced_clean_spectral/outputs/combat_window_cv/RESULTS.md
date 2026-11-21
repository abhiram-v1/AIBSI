| Model | Accuracy | Macro F1 (25 folds) | Pooled F1 (5 repeats) | FTD recall |
|---|---:|---:|---:|---:|
| spectral|site_zscore | 0.510 | 0.482 +/- 0.057 | 0.488 | 0.320 |
| eegpt2048|site_zscore | 0.639 | 0.614 +/- 0.085 | 0.619 | 0.443 |
| early_fusion|site_zscore | 0.614 | 0.587 +/- 0.073 | 0.592 | 0.411 |
| spectral|combat_dx | 0.522 | 0.494 +/- 0.063 | 0.499 | 0.324 |
| eegpt2048|combat_dx | 0.642 | 0.616 +/- 0.078 | 0.620 | 0.442 |
| early_fusion|combat_dx | 0.640 | 0.612 +/- 0.067 | 0.616 | 0.421 |
| spectral|combat_dx_age_sex | 0.519 | 0.492 +/- 0.070 | 0.497 | 0.334 |
| eegpt2048|combat_dx_age_sex | 0.632 | 0.611 +/- 0.087 | 0.614 | 0.447 |
| early_fusion|combat_dx_age_sex | 0.635 | 0.601 +/- 0.073 | 0.606 | 0.380 |
| demographics_only | 0.405 | 0.368 +/- 0.061 | 0.373 | 0.548 |
| late_fusion|site_zscore | 0.616 | 0.597 +/- 0.089 | 0.601 | 0.480 |
| late_fusion|combat_dx | 0.622 | 0.600 +/- 0.067 | 0.604 | 0.471 |
| late_fusion|combat_dx_age_sex | 0.631 | 0.607 +/- 0.099 | 0.612 | 0.462 |
| win_spectral|site_zscore | 0.516 | 0.490 +/- 0.071 | 0.496 | 0.327 |
| win_eegpt|site_zscore | 0.613 | 0.584 +/- 0.069 | 0.589 | 0.396 |
| win_spectral|combat_dx | 0.508 | 0.479 +/- 0.071 | 0.484 | 0.296 |
| win_eegpt|combat_dx | 0.630 | 0.601 +/- 0.074 | 0.605 | 0.412 |
| win_late_fusion|site_zscore | 0.634 | 0.603 +/- 0.070 | 0.608 | 0.417 |
| win_late_fusion|combat_dx | 0.639 | 0.609 +/- 0.077 | 0.614 | 0.421 |
| previous_selected|spectral | 0.543 | 0.513 +/- 0.091 | 0.518 | 0.348 |
| previous_selected|eegpt2048 | 0.624 | 0.601 +/- 0.079 | 0.603 | 0.417 |
| previous_selected|early_fusion | 0.652 | 0.629 +/- 0.068 | 0.633 | 0.480 |

| Paired comparison (pooled macro F1) | Difference | 95% CI | P(first better) |
|---|---:|---|---:|
| spectral|combat_dx - spectral|site_zscore | +0.011 | [-0.013, +0.035] | 0.81 |
| spectral|combat_dx_age_sex - spectral|combat_dx | -0.001 | [-0.018, +0.017] | 0.43 |
| eegpt2048|combat_dx - eegpt2048|site_zscore | +0.001 | [-0.029, +0.033] | 0.52 |
| eegpt2048|combat_dx_age_sex - eegpt2048|combat_dx | -0.006 | [-0.026, +0.013] | 0.25 |
| early_fusion|combat_dx - early_fusion|site_zscore | +0.023 | [-0.001, +0.048] | 0.97 |
| early_fusion|combat_dx_age_sex - early_fusion|combat_dx | -0.010 | [-0.032, +0.013] | 0.19 |
| late_fusion|combat_dx - late_fusion|site_zscore | +0.003 | [-0.025, +0.032] | 0.58 |
| late_fusion|combat_dx_age_sex - late_fusion|combat_dx | +0.007 | [-0.014, +0.029] | 0.75 |
| win_spectral|site_zscore - spectral|site_zscore | +0.008 | [-0.030, +0.050] | 0.67 |
| win_spectral|combat_dx - spectral|combat_dx | -0.014 | [-0.054, +0.024] | 0.24 |
| win_spectral|combat_dx - win_spectral|site_zscore | -0.012 | [-0.030, +0.008] | 0.12 |
| win_eegpt|site_zscore - eegpt2048|site_zscore | -0.031 | [-0.079, +0.014] | 0.10 |
| win_eegpt|combat_dx - eegpt2048|combat_dx | -0.015 | [-0.057, +0.027] | 0.23 |
| win_eegpt|combat_dx - win_eegpt|site_zscore | +0.017 | [-0.000, +0.033] | 0.97 |
| win_late_fusion|site_zscore - late_fusion|site_zscore | +0.007 | [-0.040, +0.053] | 0.60 |
| win_late_fusion|combat_dx - late_fusion|combat_dx | +0.010 | [-0.036, +0.052] | 0.67 |
| win_late_fusion|combat_dx - win_late_fusion|site_zscore | +0.006 | [-0.012, +0.024] | 0.75 |
| eegpt2048|site_zscore - spectral|site_zscore | +0.132 | [+0.070, +0.194] | 1.00 |
| early_fusion|site_zscore - spectral|site_zscore | +0.105 | [+0.049, +0.162] | 1.00 |
| eegpt2048|combat_dx - spectral|site_zscore | +0.133 | [+0.072, +0.190] | 1.00 |
| early_fusion|combat_dx - spectral|site_zscore | +0.128 | [+0.073, +0.184] | 1.00 |
| spectral|combat_dx_age_sex - spectral|site_zscore | +0.010 | [-0.016, +0.035] | 0.78 |
| eegpt2048|combat_dx_age_sex - spectral|site_zscore | +0.126 | [+0.065, +0.186] | 1.00 |
| early_fusion|combat_dx_age_sex - spectral|site_zscore | +0.118 | [+0.064, +0.170] | 1.00 |
| late_fusion|site_zscore - spectral|site_zscore | +0.113 | [+0.078, +0.149] | 1.00 |
| late_fusion|combat_dx - spectral|site_zscore | +0.117 | [+0.079, +0.156] | 1.00 |
| late_fusion|combat_dx_age_sex - spectral|site_zscore | +0.124 | [+0.082, +0.168] | 1.00 |
| win_eegpt|site_zscore - spectral|site_zscore | +0.101 | [+0.031, +0.169] | 1.00 |
| win_spectral|combat_dx - spectral|site_zscore | -0.003 | [-0.041, +0.036] | 0.44 |
| win_eegpt|combat_dx - spectral|site_zscore | +0.118 | [+0.048, +0.185] | 1.00 |
| win_late_fusion|site_zscore - spectral|site_zscore | +0.120 | [+0.062, +0.177] | 1.00 |
| win_late_fusion|combat_dx - spectral|site_zscore | +0.126 | [+0.072, +0.181] | 1.00 |
| previous_selected|spectral - spectral|site_zscore | +0.030 | [+0.002, +0.059] | 0.98 |
| previous_selected|eegpt2048 - spectral|site_zscore | +0.116 | [+0.053, +0.178] | 1.00 |
| previous_selected|early_fusion - spectral|site_zscore | +0.146 | [+0.089, +0.201] | 1.00 |