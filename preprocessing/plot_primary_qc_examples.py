"""Create compact visual examples for the primary EEG quality audit."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"C:\Projects\AIBSI\preprocessing\02_balanced_primary\primary")
AUDIT = Path(r"C:\Projects\AIBSI\preprocessing\03_signal_qc_audit")
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]


def main() -> None:
    signals = np.load(ROOT / "signals.npy", mmap_mode="r")
    with (AUDIT / "window_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    ranked = sorted(rows, key=lambda r: float(r["max_step_uv"]), reverse=True)
    chosen = ranked[:4]
    high = max(rows, key=lambda r: float(r["high_30_45_ratio"]))
    typical = min(rows, key=lambda r: abs(float(r["max_step_uv"]) - 15.77))
    if high not in chosen:
        chosen.append(high)
    chosen.append(typical)

    fig, axes = plt.subplots(len(chosen), 1, figsize=(15, 2.7 * len(chosen)), constrained_layout=True)
    time = np.arange(signals.shape[-1]) / 250
    for ax, row in zip(axes, chosen):
        idx = int(row["array_index"])
        data = np.asarray(signals[idx])
        scale = max(float(np.quantile(np.std(data, axis=1), 0.5)) * 4, 35)
        offsets = np.arange(len(CHANNELS))[::-1] * scale
        ax.plot(time, data.T + offsets, linewidth=0.45)
        ax.set_yticks(offsets)
        ax.set_yticklabels(CHANNELS, fontsize=7)
        ax.set_xlim(0, 4)
        ax.set_title(
            f"{row['sample_id']} | {row['group']} | max step {float(row['max_step_uv']):.1f} µV | "
            f"max abs {float(row['max_abs_uv']):.1f} µV | 30–45 Hz {100*float(row['high_30_45_ratio']):.1f}%",
            fontsize=10,
        )
        ax.set_xlabel("seconds")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Primary dataset: highest-risk windows plus a typical comparison", fontsize=14)
    fig.savefig(AUDIT / "qc_examples.png", dpi=160)


if __name__ == "__main__":
    main()
