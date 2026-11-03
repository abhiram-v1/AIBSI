# Professor progress chart pack

These figures summarize the saved experiments without treating incompatible
tasks or validation protocols as interchangeable.

1. `00_dataset_growth.png` — original 88-person dataset versus the later usable
   174-person combined cohort.
2. `01_original_tensor_progress.png` — early tensor-course work on the original
   OpenNeuro cohort, reported using balanced accuracy.
3. `02_binary_replication.png` — binary dementia-vs-control comparison and why
   the reference paper's 85% accuracy is misleading.
4. `03_combined_model_progress.png` — fixed-split model progression after the
   BrainLat cohort was combined with the original dataset.
5. `04_main_model_class_recall.png` — AD, FTD, and healthy-control recall for
   the principal models.
6. `05_tensor_compression_tradeoff.png` — dimensionality reduction versus lost
   classification performance.
7. `06_validation_robustness.png` — the optimistic fixed split compared with
   repeated participant-level evaluation.
8. `07_preprocessing_ablation.png` — evidence that more aggressive cleaning did
   not improve generalization.

Presentation wording:

- Call 70.4% accuracy / 67.8% macro F1 the **strongest exploratory fixed-split
  benchmark**, not the expected clinical performance.
- Call 75.0% accuracy / 71.8% macro F1 the **nested binary dementia-screening
  result** on the original 88-person dataset.
- Describe Tensor Train as a **structured compression ablation**. It compressed
  1,824 participant values to four coefficients but reduced three-class macro F1
  from 64.3% for the direct log-mel SVM to 46.4%.
- Emphasize that all reported model outputs are participant-level predictions.
