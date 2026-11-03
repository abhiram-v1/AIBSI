"""Build compact, traceable files for the 2026-10-26 team demonstration."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ADV = ROOT / "neural_comparison/advanced_clean_spectral"
PREP = ADV / "outputs/moderate_log_mel/prepared"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def archive(name: str, files: list[Path], extras: dict[str, str] | None = None) -> None:
    with zipfile.ZipFile(OUT / name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(set(files)):
            if path.is_file():
                z.write(path, rel(path))
        for entry, content in (extras or {}).items():
            z.writestr(entry, content)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((PREP / "manifest.json").read_text(encoding="utf-8"))
    arrays = {key: np.load(PREP / f"{key}.npy", mmap_mode="r") for key in ("log_mel", "labels", "subject_index")}
    participants = manifest["participants"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in participants:
        groups[(p["label"], p["source"])].append(p)

    rows = []
    sample_counts = {}
    for label in ("AD", "FTD", "HC"):
        selected = []
        for source in sorted({p["source"] for p in participants}):
            selected.extend(groups[(label, source)][:3])
        window_indices = []
        for p in selected:
            sid = p["subject_index"]
            candidates = np.flatnonzero(arrays["subject_index"] == sid)
            for idx in candidates[:8]:
                window_indices.append(int(idx))
                rows.append({"chunk": f"01_data_{label}.npz", "row_in_chunk": len(window_indices) - 1,
                             "source": p["source"], "site": p["site"], "subject_id": p["subject_id"],
                             "label": label, "original_window_index": int(idx)})
        ix = np.asarray(window_indices, dtype=np.int64)
        np.savez_compressed(OUT / f"01_data_{label}.npz", log_mel=np.asarray(arrays["log_mel"][ix]),
                            labels=np.asarray(arrays["labels"][ix]),
                            subject_index=np.asarray(arrays["subject_index"][ix]))
        sample_counts[label] = {"participants": len(selected), "windows": len(ix)}

    with (OUT / "01_data_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    classical = [
        ADV / "outputs/tensor_train_svm/tt_svm_model.joblib",
        ADV / "outputs/tensor_train_svm/metrics.json",
        ADV / "outputs/site_svm_extended_tuning/model.joblib",
        ADV / "outputs/site_svm_extended_tuning/metrics.json",
        ADV / "outputs/deep_models/log_mel/cnn/seed_20261009/best_model.pt",
        ADV / "outputs/deep_models/log_mel/train_scaler.npz",
    ]
    archive("02_models_compact.zip", classical, {
        "MODEL_NOTES.txt": "These are saved research artifacts. The extended-tuning SVM and TT-SVM are not the 70.4% blend model. The CNN checkpoint is one seed, not an ensemble. Use the corresponding scripts and metrics before inference. PyTorch/joblib files must only be loaded from trusted sources.\n"
    })

    evidence_dirs = [
        ADV / "outputs/results", ADV / "outputs/deep_models/eegpt_attention",
        ADV / "outputs/tensor_train_svm", ADV / "outputs/electrode_frequency_stability",
        ADV / "outputs/preprocessing_feature_audit", ADV / "outputs/compact_random_forest",
        ROOT / "tensor_pipeline/outputs/v4_tuning",
    ]
    evidence = []
    for directory in evidence_dirs:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".png", ".md"}:
                evidence.append(path)
    evidence.extend([ADV / "RUN_SUMMARY.md", ROOT / "PROJECT_MASTER_REPORT.md"])
    archive("03_results_and_proof.zip", evidence)

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    code = [ROOT / p for p in tracked if p and Path(p).suffix.lower() in {".py", ".md", ".tex", ".bib"}]
    code += [ADV / "audit_preprocessing_feature_svm.py", ADV / "evaluate_compact_random_forest.py"]
    archive("04_project_code_and_protocols.zip", code, {
        "EXTERNAL_EEGPT_VERSION.txt": "EEGPT source: https://github.com/BINE022/EEGPT.git\nLocal commit: ea7282f698c406cd423ccbedbe9960853761144c\nThe large upstream pretraining checkpoint is not included. The project-trained EEGPT checkpoint is uploaded separately.\n"
    })

    readme = f"""# AIBSI demo handoff — 26 October 2026

## Five-minute demo path

1. Open `NeuroTensor_Status_Presentation_10_Slides_Focused.pptx` for the visual explanation of EEG windowing, tensor decomposition, and subject-level evaluation. This deck and `technical_report_main.pdf` describe the earlier **88-person OpenNeuro study**; do not cite their numbers as the newer combined-cohort results.
2. Open `advanced_clean_spectral_RUN_SUMMARY.md` and `03_results_and_proof.zip` for the newer **174-person OpenNeuro + BrainLat analysis**. Its 121/26/27 participant-disjoint train/validation/test split is the basis for the fixed-split comparison.
3. Show `neural_comparison/advanced_clean_spectral/outputs/deep_models/eegpt_attention/svm_eegpt_blend_comparison.png` and `svm_blend_metrics.json` inside the proof ZIP. The validation-selected SVM + EEGPT blend was correct for **19 of 27** held-out participants: **70.4% accuracy, 67.8% macro F1, 67.6% balanced accuracy**.
4. State the limitation with the same emphasis: the small test split was inspected during multiple exploratory iterations. Repeated participant-level evaluation of the fixed spectral SVM averaged about **49.0% macro F1**. This is research evidence, not clinical validation or a stable expected 70.4% performance claim.
5. Use `01_data_*.npz` as quick examples of actual prepared log-mel EEG tensors. Use `02_models_compact.zip` and `eegpt_attention_best_model.pt` to show saved trained artifacts. The model files require the project code, dependencies, and compatible preprocessing for inference.

## Data samples

Each of the three class chunks contains {sample_counts['AD']['windows']} windows from {sample_counts['AD']['participants']} pseudonymous participants for AD, FTD, or HC, drawn from the moderate log-mel preprocessing stream. `log_mel` has axes `[window, 19 electrodes, 24 mel bands, 13 time frames]` and dtype `float16`; labels are 0=AD, 1=FTD, 2=HC. `01_data_manifest.csv` maps each chunk row to source, site, subject, and original prepared-array window index. The dataset is a **tiny demonstration sample**, not the training set and not suitable for re-estimating accuracy. No local source paths are included.

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
"""
    (OUT / "00_START_HERE.md").write_text(readme, encoding="utf-8")
    (OUT / "advanced_clean_spectral_RUN_SUMMARY.md").write_bytes((ADV / "RUN_SUMMARY.md").read_bytes())

    manifest_rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "build_handoff.py":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "00_SHA256_MANIFEST.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    print(json.dumps({"sample_counts": sample_counts, "files": manifest_rows}, indent=2))


if __name__ == "__main__":
    main()
