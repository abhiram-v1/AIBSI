# Graph-regularized non-negative tensor factorization + calibrated SVM

Run commands from `C:\Projects\AIBSI` using Python 3.10. This experiment preserves
the existing experiments and implements a new optimizer instead of reusing the
old graph-CP update whose reported objective and update rule disagreed.

## Commands

```powershell
py -3.10 neural_comparison/graph_ntf_svm/test_pipeline.py
py -3.10 neural_comparison/graph_ntf_svm/run.py prepare

# Complete bounded nested run, all five core comparisons, final probability model:
py -3.10 -u neural_comparison/graph_ntf_svm/run.py train --profile quick --repeats 1 --maxiter 100 --starts 1 --out neural_comparison/graph_ntf_svm/outputs/quick

# Full protocol: 3 repeats x 5 outer folds, 3 inner folds, 3 initializations:
py -3.10 -u neural_comparison/graph_ntf_svm/run.py train --profile full --out neural_comparison/graph_ntf_svm/outputs/full

# Optional coupled spectral/connectivity arm, same cohort and windows:
py -3.10 -u neural_comparison/graph_ntf_svm/run.py connectivity
py -3.10 -u neural_comparison/graph_ntf_svm/run.py train --profile full --methods direct ntf gntf shuffled tt coupled --out neural_comparison/graph_ntf_svm/outputs/full_coupled

# Fully nested LOSO is implemented but computationally much more expensive:
py -3.10 -u neural_comparison/graph_ntf_svm/run.py train --outer loso --methods gntf --out neural_comparison/graph_ntf_svm/outputs/loso

# Probability-only terminal prediction on a new original 19-channel .set file:
py -3.10 neural_comparison/graph_ntf_svm/run.py predict --model neural_comparison/graph_ntf_svm/outputs/quick/final_gntf/model.joblib --recording C:/path/person_task-eyesclosed_eeg.set --source ds004504 --site AHEPA

# Replay a known prepared participant (explicitly marked as a training subject):
py -3.10 neural_comparison/graph_ntf_svm/run.py predict --model neural_comparison/graph_ntf_svm/outputs/quick/final_gntf/model.joblib --subject-id ds004504:sub-001

py -3.10 neural_comparison/graph_ntf_svm/run.py report --out neural_comparison/graph_ntf_svm/outputs/quick
```

Connectivity preparation reads the source `.set`/`.fdt` recordings and caches
participant results with source hashes. BrainLat inference uses `--source brainlat`
and `--site AR` or `CL`; source channel order is checked by the original harmonizer.
An unseen site's embedding uses global training centering and is identified in
the probability output. No test-time population estimates are fitted.

## Data and objective

174 people (71 AD, 42 FTD, 61 HC); 6,721 eight-second moderate-cleaning windows.
At each STFT frame convert centered log-mel power back to positive power, divide
by total electrode/mel power, and take sqrt. Centering cancels in this relative
normalization; it does not recover absolute power. Pool mean, SD, median, IQR
to `[174,19,24,4]`. Mode4 is statistics, not synchronized recording time.
The original 912 mean/SD log-mel features are kept for the direct-SVM comparison.
Inputs, channel order, statistics, participant IDs and source hashes are saved.
The legacy moderate pipeline lacks explicit rejection of EEGLAB discontinuity
annotations; the report states this inherited limitation instead of silently
claiming a corrected preprocessing replication.

G-NTF here means non-negative CP with a graph penalty on its electrode factor:

`Xhat[n,c,f,s] = sum_r A[n,r] B[c,r] F[f,r] S[s,r]`

`loss = .5 ||X-Xhat||² / ||X||² + lambda/(2R) trace(B.T L B)`

All factors are non-negative. Columns of B,F,S have unit L2 norm INSIDE the
objective and gradient. This fixes the scale ambiguity that otherwise allows
shrinking B to evade the graph penalty while increasing A. L-BFGS-B optimizes
bounded raw positive shape factors through this normalization. The exact
analytic gradients are checked by central finite differences. Multi-start
selection uses training reconstruction/regularized objective, never test scores.
Convergence flags and objective trajectories are retained; reaching maxiter is
not called convergence. Scale each statistic by training RMS (positive) before
factorization, applying the same saved scale to validation/test participants.
The graph is a symmetrized four-neighbour scalp-distance graph with Laplacian
spectral norm1. Electrode labels use the verified canonical 19-channel order.

Coupled mode adds `Y[n,i,j,b] = sum_r A[n,r] B[i,r] B[j,r] H[b,r]` using wPLI in
1-4, 4-8, 8-13, 13-30, 30-40 Hz bands. The participant and electrode factors are
shared; diagonal connectivity is masked. Add `coupling*.5||Y-Yhat||²/||Y||²`
to the loss. H also has unit-norm non-negative columns. Window phases are never
joined before the Hilbert transform. This arm requires raw EEG, since log-mel
power alone does not retain the phase needed for wPLI. BrainLat interpolation
can distort sensor connectivity and is an explicit interpretation limitation.

## Inductive evaluation and search

Every inner fold refits statistic scaling, G-NTF/TT/PCA, site centering, feature
standardization, and SVM. Held-out participant embeddings use NNLS with the
training factors fixed. Per-site means are computed AFTER decomposition from
training embeddings only. This preserves tensor non-negativity. It addresses
site shifts at the classifier input but cannot guarantee removal of site factors.
No diagnosis, cognition, participant ID, site dummy, or window-count predictor
is appended to the feature vector.

Full: ranks 3/5/8/12/16, graph lambda .001/.01/.1/1, C .03/.1/.3/1/3/10,
linear/RBF; RBF gamma scale/.01/.1/1; balanced class weights. Coupling .1/1.
Quick: ranks3/8, lambda .01/.1, C .1/1/10, linear/RBF gamma scale; coupling .1.
NTF uses lambda0. Direct baseline tunes PCA None/20/40/60 (quick20). TT is a
common training basis on the same positive tensor, with the sample mode last.
Shuffled-graph control deterministically permutes graph nodes, preserving graph
structure. All arms use identical outer participants and inner splits.

Candidate selection uses inner-fold macro F1. Within best mean minus its fold
standard error, prefer linear kernel then smaller rank/PCA; break remaining ties
by macroF1, balanced accuracy, FTD recall and smaller C. The one-SE rule is a
practical simplicity heuristic; correlated CV folds do not establish significance.
The tuning endpoint uses SVM.predict; final reported endpoints use argmax of
calibrated probabilities. Log loss is evaluated on outer folds, not used as a
pretend inner tuning score without inner calibration.

Calibration obtains three-class decision scores by cross-fitting the ENTIRE
selected pipeline on outer-training subjects, including decomposition. Three
unweighted sigmoid logistic calibrators fit those OOF scores; their outputs
are renormalized. Then refit the base pipeline on all outer-training subjects.
An outer test person never enters tuning, representation fitting or calibration.
Hyperparameters were chosen using outer-training data; calibration is not a new
independent performance estimate. Outer CV estimates the complete procedure.
The final model repeats selection/calibration on all174 and is for deployment,
not a source of test scores. Report conditional participant-bootstrap intervals
with all repeat observations resampled together, not 522 independent subjects.

## Outputs and interpretation

Each fold saves selection candidates, train/test IDs, calibration membership,
probabilities and a complete model. Resume only with identical protocol and
code/data hashes. Use another output directory after changing code or settings.
`summary.json`, `REPORT.md`, `figures/` contain matched performance, per-site
results, confusion matrices and reliability diagrams. `final_gntf/` contains a
complete replayable probability model and metadata. Its `interpretation/`
contains 19-electrode loading maps, mel curves in Hz, statistic loadings,
class activation distributions, pairwise linear-SVM coefficients, objective
history, and Hungarian-matched factor stability across overlapping folds.
Nonlinear SVM arms do not get misleading linear coefficient explanations.

MMSE (88 OpenNeuro) and MoCA (66 available BrainLat) are correlated separately
with component activation using Spearman and BH FDR across all reported tests.
Within-diagnosis analyses require at least8 people and a varying score. These
are explicitly exploratory full-cohort associations: pooled correlations can
be driven by diagnosis, age and site. They are not evidence of causal sources
or independent severity prediction. CP remains permutation ambiguous; factor
normalization and label-blind Hungarian matching support comparison, not a
uniqueness guarantee. Final-reference stability is descriptive because folds
overlap and the reference includes the cohort.

The compact tensors fit comfortably in CPU memory. Thread limits and fold-local
decomposition caches control runtime; no GPU is required for the SVM or SciPy
optimizer. Run the full profile for the planned search. A quick run does not
substitute for the full search and does not justify a LOSO claim.

Calibration rationale: https://scikit-learn.org/stable/modules/calibration.html
