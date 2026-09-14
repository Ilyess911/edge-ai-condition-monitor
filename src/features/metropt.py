"""Window features for the MetroPT-3 air production unit.

An air leak makes the compressor work harder to hold reservoir pressure: it
runs under load for a larger share of time, cycles more often, and pressure
decays faster between cycles. The features describe that duty cycle and the
pressure/temperature/current levels over a 30-minute window. They were chosen
from the sensor descriptions before any detector was evaluated.
"""

from __future__ import annotations

import numpy as np

from src.data.metropt import CHANNELS

C = {name: i for i, name in enumerate(CHANNELS)}

FEATURE_NAMES = (
    "tp2_mean", "tp2_std", "tp3_mean", "tp3_min", "reservoirs_min", "h1_mean",
    "dv_pressure_mean", "oil_temp_mean", "oil_temp_range", "motor_current_mean",
    "load_duty", "load_cycles", "lps_fraction", "oil_level_fraction",
)


class MetroPTFeatureExtractor:
    def __init__(self, min_coverage: float = 0.8):
        self.min_coverage = min_coverage

    def __call__(self, w: np.ndarray) -> np.ndarray | None:
        # Too many missing samples (logger gap longer than the hold time): no decision.
        if np.mean(~np.isnan(w[:, C["TP3"]])) < self.min_coverage:
            return None
        col = lambda name: w[:, C[name]]  # noqa: E731
        load = col("DV_eletric")
        valid = ~np.isnan(load)
        on = load[valid] > 0.5
        cycles = float(np.sum(on[1:] & ~on[:-1]))
        return np.array([
            np.nanmean(col("TP2")), np.nanstd(col("TP2")),
            np.nanmean(col("TP3")), np.nanmin(col("TP3")),
            np.nanmin(col("Reservoirs")), np.nanmean(col("H1")),
            np.nanmean(col("DV_pressure")),
            np.nanmean(col("Oil_temperature")),
            np.nanmax(col("Oil_temperature")) - np.nanmin(col("Oil_temperature")),
            np.nanmean(col("Motor_current")),
            float(np.mean(on)) if len(on) else 0.0, cycles,
            np.nanmean(col("LPS")), np.nanmean(col("Oil_level")),
        ])
