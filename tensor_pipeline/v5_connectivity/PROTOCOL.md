# Frozen V5 protocol

The primary endpoint is mean participant-level balanced accuracy of the complete
pipeline chosen inside three-fold inner validation, evaluated over three repeated
five-fold outer participant splits. The outer seed remains 9060 for direct V3/V4
comparison. All bags belonging to one person remain together.

Signal representations are computed from the unchanged 16,824 approved four-
second windows: four-band trace-normalized shrinkage covariance, correlation,
amplitude-envelope correlation, imaginary phase-locking component, weighted
phase-lag index, and log/relative power. Whole-recording, four contiguous-bag and
eight contiguous-bag summaries are compared.

Riemannian reference matrices, tangent mappings, ANOVA selection, scaling and
classifiers are fitted exclusively on the current training participants. Training
weights give every diagnosis equal total weight and every person equal weight
within diagnosis. No oversampling, synthetic EEG, MMSE, age or sex predictor is
used.

There are 150 configurations fixed in `models.py`: power, Riemannian covariance,
connectivity and hybrid representations; logistic regression, RBF SVM and two
conservative XGBoost controls; multiclass and hierarchical CN-v-dementia then
AD-v-FTD formulations; and one/four/eight participant bags. Earlier artifacts are
preserved. The cohort and split seed have already been explored, so this remains
internal comparative evidence rather than independent confirmation.
