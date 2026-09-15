"""Window features for real bearing recordings (Paderborn track).

Input window columns: vibration, phase current, speed [rpm], torque [Nm],
radial force [N]. Speed, torque and force are per-recording means held
constant over the recording (the 4 kHz channels are not replayed sample by
sample; see src/data/paderborn.py).

Fifteen features, three families:
- time-domain vibration statistics (4);
- vibration spectral energy fractions in four fixed bands (4);
- envelope analysis (2): a localised bearing defect produces impacts at a
  characteristic frequency (BPFO for the outer race, BPFI for the inner race)
  that modulate high-frequency resonances. Demodulating the band above
  `envelope_low_hz` and measuring envelope-spectrum energy at the first three
  harmonics of that frequency is the classical detection method;
- motor current (2) and operating context (3).
"""

from __future__ import annotations

import numpy as np

from src.data.paderborn import defect_orders

FEATURE_NAMES = (
    "vib_rms", "vib_peak", "vib_crest", "vib_kurtosis",
    "band_0_1k", "band_1k_4k", "band_4k_10k", "band_10k_up",
    "env_bpfo", "env_bpfi",
    "current_rms", "current_residual",
    "speed", "torque", "force",
)
BAND_EDGES_HZ = (0.0, 1_000.0, 4_000.0, 10_000.0, np.inf)


class BearingFeatureExtractor:
    def __init__(self, fs: float, window: int, envelope_low_hz: float = 1_000.0):
        self.fs = fs
        self.window = window
        self._hann = np.hanning(window)
        freqs = np.fft.rfftfreq(window, 1.0 / fs)
        self._freqs = freqs
        self._bands = [(freqs >= lo) & (freqs < hi) for lo, hi in zip(BAND_EDGES_HZ, BAND_EDGES_HZ[1:])]
        # At low sampling rates the demodulation band must stay below Nyquist.
        self._env_band = (freqs >= min(envelope_low_hz, fs / 4)) & (freqs < fs / 2)
        self._orders = defect_orders()
        self._resolution = fs / window

    def _harmonic_energy(self, env_power: np.ndarray, f0: float) -> float:
        """Share of envelope power (5 Hz to 5 x f0) within +-3 % (>= 1 bin) of f0, 2 f0, 3 f0."""
        f = self._freqs
        span = (f >= 5.0) & (f <= 5.0 * f0)
        total = env_power[span].sum()
        if total <= 0 or f0 <= 0:
            return 0.0
        hit = np.zeros_like(span)
        for k in (1, 2, 3):
            half = max(0.03 * k * f0, self._resolution)
            hit |= np.abs(f - k * f0) <= half
        return float(env_power[hit & span].sum() / total)

    def __call__(self, w: np.ndarray) -> np.ndarray:
        vib = w[:, 0] - w[:, 0].mean()
        var = float(np.mean(vib * vib))
        rms = np.sqrt(var)
        peak = float(np.max(np.abs(vib)))
        crest = peak / rms if rms > 0 else 0.0
        kurt = float(np.mean(vib**4) / (var * var)) if var > 0 else 0.0

        spec = np.fft.rfft(vib * self._hann)
        power = np.abs(spec) ** 2
        total = power.sum()
        bands = [float(power[m].sum() / total) if total > 0 else 0.0 for m in self._bands]

        # Analytic signal of the demodulation band -> envelope -> envelope spectrum.
        analytic = np.zeros(self.window, dtype=complex)
        n_pos = len(spec)
        analytic[:n_pos] = np.where(self._env_band, 2.0 * np.fft.rfft(vib), 0.0)
        envelope = np.abs(np.fft.ifft(analytic))
        envelope -= envelope.mean()
        env_power = np.abs(np.fft.rfft(envelope * self._hann)) ** 2
        shaft_hz = float(np.mean(w[:, 2])) / 60.0
        env_bpfo = self._harmonic_energy(env_power, self._orders["bpfo"] * shaft_hz)
        env_bpfi = self._harmonic_energy(env_power, self._orders["bpfi"] * shaft_hz)

        cur = w[:, 1] - w[:, 1].mean()
        cur_power = np.abs(np.fft.rfft(cur * self._hann)) ** 2
        cur_total = cur_power.sum()
        k = int(np.argmax(cur_power))
        fundamental = cur_power[max(0, k - 2):k + 3].sum()
        residual = float(1.0 - fundamental / cur_total) if cur_total > 0 else 0.0

        return np.array([
            rms, peak, crest, kurt, *bands, env_bpfo, env_bpfi,
            float(np.sqrt(np.mean(cur * cur))), residual,
            float(np.mean(w[:, 2])), float(np.mean(w[:, 3])), float(np.mean(w[:, 4])),
        ])
