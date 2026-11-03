from ml_app import node


@node(
    name="System EEG STFT files",
    category="EEG",
    description="Reference the prepared EEG STFT and participant files in the AIBSI project without importing them into Mielin storage.",
    color="#4472a8",
    inputs=[],
    outputs=[{"name": "source", "type": "any"}],
)
def system_stft_files():
    base = r"C:\Projects\AIBSI\neural_comparison\stft_gpu_3class\outputs\prepared"
    paths = {
        "stft": base + r"\spectrograms.npy",
        "labels": base + r"\labels.npy",
        "subject_index": base + r"\subject_index.npy",
        "manifest": base + r"\manifest.json",
    }
    stft = np.load(paths["stft"], mmap_mode="r")
    labels = np.load(paths["labels"], mmap_mode="r")
    subjects = np.load(paths["subject_index"], mmap_mode="r")
    participants = pd.read_json(paths["manifest"], typ="series")["participants"]
    if stft.shape != (6721, 19, 79, 13):
        raise ValueError("Unexpected source STFT shape: " + str(stft.shape))
    if len(labels) != len(stft) or len(subjects) != len(stft) or len(participants) != 174:
        raise ValueError("Source STFT, labels, subject indices, and manifest do not match")
    return {"source": paths}
