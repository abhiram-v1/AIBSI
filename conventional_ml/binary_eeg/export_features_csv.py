"""Export the conventional EEG subject feature archive as a labelled CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "conventional_ml/outputs/binary_eeg/subject_features.npz"
DESTINATION = ROOT / "conventional_ml/outputs/binary_eeg/subject_features.csv"

FEATURES = (
    "log10_power_delta_1_4hz",
    "log10_power_theta_4_8hz",
    "log10_power_alpha_8_13hz",
    "log10_power_beta_13_30hz",
    "log10_power_gamma_30_45hz",
    "relative_power_delta_1_4hz",
    "relative_power_theta_4_8hz",
    "relative_power_alpha_8_13hz",
    "relative_power_beta_13_30hz",
    "relative_power_gamma_30_45hz",
    "spectral_entropy",
    "alpha_peak_hz",
    "log10_theta_alpha_ratio",
    "log10_delta_alpha_ratio",
    "log10_rms",
    "hjorth_mobility_hz",
    "hjorth_complexity",
    "log10_mean_absolute_step",
    "zero_crossing_rate",
)

REGIONS = ("frontal", "temporal", "central", "parietal", "occipital")


def matrix_columns(prefixes: tuple[str, ...]) -> list[str]:
    return [
        f"{prefix}__{stat}__{feature}"
        for stat in ("window_mean", "window_std")
        for prefix in prefixes
        for feature in FEATURES
    ]


def global_columns() -> list[str]:
    return [
        f"global__{summary}__{feature}"
        for summary in (
            "channel_mean_of_window_mean",
            "channel_std_of_window_mean",
            "channel_mean_of_window_std",
        )
        for feature in FEATURES
    ]


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as archive:
        subjects = archive["subjects"]
        groups = archive["groups"]
        counts = archive["counts"]
        channels = tuple(str(value) for value in archive["channels"])
        channel_features = archive["channel_features"]
        regional_features = archive["regional_features"]
        global_features = archive["global_features"]

    channel_columns = matrix_columns(channels)
    regional_columns = matrix_columns(REGIONS)
    whole_head_columns = global_columns()
    feature_columns = channel_columns + regional_columns + whole_head_columns
    matrix = np.concatenate(
        [channel_features, regional_features, global_features], axis=1
    )

    assert matrix.shape == (88, 969)
    assert len(feature_columns) == matrix.shape[1]
    assert len(set(feature_columns)) == len(feature_columns)
    assert np.isfinite(matrix).all()

    with DESTINATION.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "group", "retained_windows", *feature_columns])
        for subject, group, count, features in zip(
            subjects, groups, counts, matrix, strict=True
        ):
            writer.writerow([subject, group, int(count), *features])

    print(
        f"Exported {matrix.shape[0]} participants and {matrix.shape[1]} features "
        f"to {DESTINATION}"
    )


if __name__ == "__main__":
    main()
