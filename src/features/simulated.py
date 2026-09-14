"""Window features for the simulated rotating machine.

Twelve features, all standard in vibration-based condition monitoring and all
computable in O(window) with one FFT. No learned feature extraction: on an
edge device every feature must be cheap and explainable to a maintenance
technician.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "vib_rms",
    "vib_peak",
    "vib_crest",
    "vib_kurtosis",
    "vib_band_low",  # energy fraction 0-60 Hz   (shaft 1x/2x: imbalance, misalignment)
    "vib_band_mid",  # energy fraction 60-200 Hz (bearing defect frequencies)
    "vib_band_high",  # energy fraction 200-500 Hz (structural resonance excited by impacts)
    "temp_mean",
    "current_mean",
    "current_std",
    "speed_mean",
    "pressure_mean",
)

BANDS_HZ = ((0.0, 60.0), (60.0, 200.0), (200.0, 500.0))


class SimulatedFeatureExtractor:
    names = FEATURE_NAMES

    def __init__(self, fs: float, window: int):
        self.fs = fs
        self.window = window
        self._hann = np.hanning(window)
        freqs = np.fft.rfftfreq(window, 1.0 / fs)
        # A band above Nyquist is empty: its feature is 0 at every window, which
        # is exactly the information lost by sampling too slowly.
        self._bands = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS_HZ]

    def __call__(self, w: np.ndarray) -> np.ndarray:
        vib = w[:, 0]
        vib = vib - vib.mean()
        rms = np.sqrt(np.mean(vib * vib))
        peak = np.max(np.abs(vib))
        crest = peak / rms if rms > 0 else 0.0
        var = rms * rms
        kurt = np.mean(vib**4) / (var * var) if var > 0 else 0.0

        power = np.abs(np.fft.rfft(vib * self._hann)) ** 2
        total = power.sum()
        bands = [power[m].sum() / total if total > 0 else 0.0 for m in self._bands]

        slow = w[:, 1:5]
        means = np.nanmean(slow, axis=0)
        current_std = np.nanstd(slow[:, 1])
        return np.array(
            [rms, peak, crest, kurt, *bands, means[0], means[1], current_std, means[2], means[3]]
        )


def extract_offline(signals: np.ndarray, fs: float, window: int, hop: int) -> np.ndarray:
    """Batch reference used by tests to check the streaming path gives the same features."""
    ext = SimulatedFeatureExtractor(fs, window)
    starts = range(0, len(signals) - window + 1, hop)
    return np.array([ext(signals[s : s + window]) for s in starts])
