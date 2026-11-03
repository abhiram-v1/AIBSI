from ml_app import node


@node(
    name="Log-mel STFT + participant tensors",
    category="EEG",
    description="Convert the external log-STFT cache to 24 mel bands and summarize each participant for both model branches.",
    color="#568d62",
    inputs=[{"name": "source", "type": "any"}],
    outputs=[{"name": "spectral", "type": "any"}, {"name": "tensor", "type": "any"}],
)
def log_mel_participants(source):
    from sklearn.model_selection import StratifiedShuffleSplit

    stft = np.load(source["stft"], mmap_mode="r")
    labels = np.load(source["labels"], mmap_mode="r")
    subject_index = np.load(source["subject_index"], mmap_mode="r")
    participants = pd.read_json(source["manifest"], typ="series")["participants"]

    frequencies = np.linspace(1.0, 40.0, stft.shape[2])
    mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
    inverse_mel = lambda m: 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)
    points = inverse_mel(np.linspace(mel(1.0), mel(40.0), 26))
    bank = np.zeros((24, len(frequencies)), dtype=np.float32)
    for band in range(24):
        left, center, right = points[band:band + 3]
        bank[band] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-6),
                (right - frequencies) / max(right - center, 1e-6),
            ),
        )
        bank[band] /= max(bank[band].sum(), 1e-8)

    flat_features = []
    tt_tensors = []
    targets = []
    for subject in range(len(participants)):
        indexes = np.flatnonzero(subject_index == subject)
        if len(indexes) == 0:
            raise ValueError("Participant has no STFT windows: " + str(subject))
        block = stft[indexes].astype(np.float32)
        relative_power = np.power(10.0, block)
        mel_power = np.einsum("mf,ncft->ncmt", bank, relative_power, optimize=True)
        log_mel = np.log10(mel_power + 1e-20)
        log_mel -= np.median(log_mel, axis=(1, 2, 3), keepdims=True)
        # The original experiment stores this representation as float16.
        log_mel = log_mel.astype(np.float16).astype(np.float32)
        repeated = log_mel.transpose(0, 3, 1, 2).reshape(-1, 19, 24)
        mean = repeated.mean(axis=0)
        std = repeated.std(axis=0)
        median = np.median(repeated, axis=0)
        iqr = np.quantile(repeated, 0.75, axis=0) - np.quantile(repeated, 0.25, axis=0)
        flat_features.append(np.concatenate([mean, std], axis=1).reshape(-1))
        tt_tensors.append(np.stack([mean, std, median, iqr], axis=-1))
        targets.append(int(labels[indexes[0]]))

    ids = np.arange(len(participants))
    strata = np.asarray([p["source"] + ":" + p["label"] for p in participants])
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=20261007)
    train, remaining = next(outer.split(ids, strata))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=20261008)
    val_local, test_local = next(inner.split(remaining, strata[remaining]))
    splits = {"train": train, "val": remaining[val_local], "test": remaining[test_local]}

    shared = {"participants": participants, "labels": np.asarray(targets, dtype=np.int64), "splits": splits}
    return {
        "spectral": {**shared, "features": np.asarray(flat_features, dtype=np.float32)},
        "tensor": {**shared, "tensors": np.asarray(tt_tensors, dtype=np.float32)},
    }
