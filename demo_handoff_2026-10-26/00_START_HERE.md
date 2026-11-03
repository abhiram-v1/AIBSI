# AIBSI demo handoff — 26 October 2026

## Five-minute demo path

1. Open `NeuroTensor_Status_Presentation_10_Slides_Focused.pptx` for the visual explanation of EEG windowing, tensor decomposition, and subject-level evaluation. This deck and `technical_report_main.pdf` describe the earlier **88-person OpenNeuro study**; do not cite their numbers as the newer combined-cohort results.
2. Open `advanced_clean_spectral_RUN_SUMMARY.md` and `03_results_and_proof.zip` for the newer **174-person OpenNeuro + BrainLat analysis**. Its 121/26/27 participant-disjoint train/validation/test split is the basis for the fixed-split comparison.
3. Show `neural_comparison/advanced_clean_spectral/outputs/deep_models/eegpt_attention/svm_eegpt_blend_comparison.png` and `svm_blend_metrics.json` inside the proof ZIP. The validation-selected SVM + EEGPT blend was correct for **19 of 27** held-out participants: **70.4% accuracy, 67.8% macro F1, 67.6% balanced accuracy**.
4. State the limitation with the same emphasis: the small test split was inspected during multiple exploratory iterations. Repeated participant-level evaluation of the fixed spectral SVM averaged about **49.0% macro F1**. This is research evidence, not clinical validation or a stable expected 70.4% performance claim.
5. Use `01_data_*.npz` as quick examples of actual prepared log-mel EEG tensors. Use `02_models_compact.zip` and `eegpt_attention_best_model.pt` to show saved trained artifacts. The model files require the project code, dependencies, and compatible preprocessing for inference.

## Data samples

Each of the three class chunks contains 48 windows from 6 pseudonymous participants for AD, FTD, or HC, drawn from the moderate log-mel preprocessing stream. `log_mel` has axes `[window, 19 electrodes, 24 mel bands, 13 time frames]` and dtype `float16`; labels are 0=AD, 1=FTD, 2=HC. `01_data_manifest.csv` maps each chunk row to source, site, subject, and original prepared-array window index. The dataset is a **tiny demonstration sample**, not the training set and not suitable for re-estimating accuracy. No local source paths are included.

The full source datasets were not uploaded. OpenNeuro ds004504 is available at https://openneuro.org/datasets/ds004504 . The project also uses BrainLat EEG; consult the code and study documentation for source access and usage terms.

## File guide

- `advanced_clean_spectral_RUN_SUMMARY.md`: latest combined-cohort narrative and limitations.
- `03_results_and_proof.zip`: metric JSON, fold evidence, figures, current compact-feature audits, and the older V4 tensor report. Internal paths mirror the project.
- `02_models_compact.zip`: TT-SVM model, extended-search SVM model, one CNN checkpoint, scaler, and corresponding metrics. These artifacts represent different experiments; the extended-search SVM is not the original SVM used in the best blend.
- `eegpt_attention_best_model.pt`: trained EEGPT + attention checkpoint used in the exploratory blend. Its exact architecture is in `train_eegpt_attention.py`; this is a PyTorch state dictionary and needs the EEGPT source code. The large upstream pretraining checkpoint is not in this handoff.
- `04_project_code_and_protocols.zip`: project Python, Markdown, and LaTeX sources, including current scripts and protocol notes. Raw datasets and generated arrays are excluded.
- `technical_report_main.pdf`: complete earlier 88-person tensor study.
- `NeuroTensor_Status_Presentation_10_Slides_Focused.pptx`: earlier visual deck.

## Presenter wording

“Our research pipeline cleans resting EEG, converts windows into spectral and tensor representations, and evaluates diagnosis at the participant level. The older 88-person nested tensor study reached 54.5% balanced accuracy for its primary V4 selection. In a later 174-person combined-cohort exploratory split, an SVM plus EEGPT blend reached 70.4% accuracy on 27 held-out people, but repeated evaluation shows substantial split sensitivity. FTD remains hardest. This is a prototype, not a diagnostic tool.”
