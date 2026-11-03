from ml_app import node, Chart


@node(
    name="Log-mel STFT preview",
    category="EEG",
    description="Plot one external EEG STFT window after the same 24-band log-mel conversion used for training.",
    color="#4e90a6",
    inputs=[{"name": "source", "type": "any"}],
    outputs=[{"name": "graph_output", "type": "plot_spec"}],
)
def log_mel_preview(source):
    stft = np.load(source["stft"], mmap_mode="r")
    # One representative window, first electrode. Data stays in the source file.
    window = np.asarray(stft[0, 0], dtype=np.float32)
    frequencies = np.linspace(1.0, 40.0, window.shape[0])
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

    log_mel = np.log10(bank @ np.power(10.0, window) + 1e-20)
    log_mel -= np.median(log_mel)
    log_mel = log_mel.astype(np.float16).astype(np.float32)

    chart = Chart(rows=1, cols=2, title="Example EEG window: 24-band log-mel STFT")
    chart.subplot(0, 0, title="Electrode 1 · 13 STFT frames", x_label="STFT frame", y_label="Mel band", legend=False).heatmap(
        log_mel, color_low="#153f78", color_high="#e9a44b"
    )
    chart.subplot(0, 1, title="Mean across STFT frames", x_label="Mel band", y_label="Relative log power", legend=False).line(
        np.arange(1, 25), log_mel.mean(axis=1), color="#4e90a6", width=2.5
    )
    return chart
