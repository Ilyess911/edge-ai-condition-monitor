"""Simulated rotating machine (motor + bearing + pump) with injected faults.

THIS DATA IS SYNTHETIC. It exists so the streaming engine can be exercised at
kHz sampling rates with exact ground-truth labels, which no public dataset with
real-time replay offers at the same time. Every physical constant below is a
modelling choice, documented in docs/data.md, not a measurement.

Channels (one row per sample):
    0 vibration   [g]      accelerometer, band-limited by an anti-aliasing filter
    1 temperature [degC]   bearing housing, first-order thermal model
    2 current     [A]      motor phase current
    3 speed       [rpm]    shaft speed
    4 pressure    [bar]    pump discharge pressure

Design rules:
- Normal operation is NOT stationary: speed and load change every few minutes.
  A detector that flags every regime change is useless, so this is the main
  source of false alarms in the benchmark.
- Faults have random severity, so some are genuinely hard to detect.
- The analog vibration signal is generated at BASE_FS and decimated with an
  anti-aliasing filter to the requested sampling rate, as a real ADC chain does.
- Sensor artefacts (short dropouts, single-sample glitches) are injected and
  labelled NORMAL: handling them is the job of edge preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter, resample_poly

CHANNELS = ("vibration", "temperature", "current", "speed", "pressure")
BASE_FS = 4000  # Hz, "analog" simulation rate before the ADC decimation stage

FAULT_TYPES = (
    "bearing_outer_race",
    "imbalance",
    "cooling_failure",
    "cavitation",
    "electrical_overload",
)

# Scaled-down bearing geometry. Real bearing resonances sit in the kHz range,
# which is why industrial vibration monitoring samples at 10-25 kHz. It is
# lowered to 350 Hz here to keep the simulation cheap; see docs/data.md.
BPFO_ORDER = 3.58  # outer-race defect frequency as a multiple of shaft speed
BEARING_RESONANCE_HZ = 350.0


@dataclass(frozen=True)
class FaultEvent:
    kind: str
    start_s: float
    duration_s: float
    severity: float  # 0..1

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass
class SimulatedRun:
    fs: float
    signals: np.ndarray  # (n_samples, 5) float64, may contain NaN (dropouts)
    labels: np.ndarray  # (n_samples,) uint8, 1 inside a fault event
    events: list[FaultEvent] = field(default_factory=list)
    seed: int = 0

    @property
    def duration_s(self) -> float:
        return len(self.labels) / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.labels)) / self.fs


def _regime_profile(n: int, fs: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant speed (rpm) and load (0..1) with 5 s linear ramps."""
    speed_levels = np.array([1200.0, 1500.0, 1800.0])
    speed = np.empty(n)
    load = np.empty(n)
    i = 0
    cur_speed, cur_load = 1500.0, 0.6
    ramp = int(5 * fs)
    while i < n:
        seg = int(rng.uniform(60, 240) * fs)
        new_speed = float(rng.choice(speed_levels))
        new_load = float(rng.uniform(0.35, 0.9))
        j = min(n, i + seg)
        k = min(j, i + ramp)
        w = np.linspace(0.0, 1.0, k - i, endpoint=False)
        speed[i:k] = cur_speed + (new_speed - cur_speed) * w
        load[i:k] = cur_load + (new_load - cur_load) * w
        speed[k:j] = new_speed
        load[k:j] = new_load
        cur_speed, cur_load = new_speed, new_load
        i = j
    return speed, load


def _envelope(t: np.ndarray, ev: FaultEvent, progressive: bool) -> np.ndarray:
    """Fault intensity over time: 0 outside the event, severity inside.

    Progressive faults (bearing wear, cooling loss) grow linearly from 10 % to
    100 % of their severity, so their first seconds are hard to detect.
    """
    inside = (t >= ev.start_s) & (t < ev.end_s)
    env = np.zeros_like(t)
    if progressive:
        frac = (t[inside] - ev.start_s) / ev.duration_s
        env[inside] = ev.severity * (0.1 + 0.9 * frac)
    else:
        env[inside] = ev.severity
    return env


def sample_fault_schedule(
    duration_s: float,
    n_faults: int,
    rng: np.random.Generator,
    min_gap_s: float = 60.0,
    warmup_s: float = 60.0,
) -> list[FaultEvent]:
    """Non-overlapping faults, cycling through every fault type."""
    ranges = {
        "bearing_outer_race": (120, 300),
        "imbalance": (60, 180),
        "cooling_failure": (180, 400),
        "cavitation": (60, 180),
        "electrical_overload": (20, 60),
    }
    kinds = [FAULT_TYPES[i % len(FAULT_TYPES)] for i in range(n_faults)]
    rng.shuffle(kinds)
    durations = [rng.uniform(*ranges[k]) for k in kinds]
    free = duration_s - warmup_s - sum(durations) - min_gap_s * (n_faults + 1)
    if free <= 0:
        raise ValueError("run too short for the requested number of faults")
    # Random split of the free time into n_faults + 1 gaps.
    cuts = np.sort(rng.uniform(0, free, n_faults))
    gaps = np.diff(np.concatenate([[0.0], cuts]))
    events, t = [], warmup_s
    for kind, dur, gap in zip(kinds, durations, gaps):
        t += min_gap_s + gap
        events.append(FaultEvent(kind, float(t), float(dur), float(rng.uniform(0.3, 1.0))))
        t += dur
    return events


def simulate(
    duration_s: float,
    fs: float = 1000.0,
    events: list[FaultEvent] | None = None,
    seed: int = 0,
    artefacts: bool = True,
) -> SimulatedRun:
    """Generate one run. Same seed and arguments give bit-identical output."""
    events = events or []
    for ev in events:
        if ev.kind not in FAULT_TYPES:
            raise ValueError(f"unknown fault type {ev.kind!r}")
    if BASE_FS % fs:
        raise ValueError(f"fs must divide {BASE_FS} Hz, got {fs}")
    rng = np.random.default_rng(seed)

    n_base = int(round(duration_s * BASE_FS))
    tb = np.arange(n_base) / BASE_FS
    speed_b, load_b = _regime_profile(n_base, BASE_FS, rng)

    env = {k: np.zeros(n_base) for k in FAULT_TYPES}
    for ev in events:
        progressive = ev.kind in ("bearing_outer_race", "cooling_failure")
        env[ev.kind] = np.maximum(env[ev.kind], _envelope(tb, ev, progressive))

    # --- vibration (analog, BASE_FS) -------------------------------------
    f_shaft = speed_b / 60.0
    phase = 2 * np.pi * np.cumsum(f_shaft) / BASE_FS
    speed_ratio = speed_b / 1500.0
    one_x = 0.20 * speed_ratio**2 * (1.0 + 2.0 * env["imbalance"])
    vib = one_x * np.sin(phase) + 0.06 * speed_ratio * np.sin(2 * phase + 0.7)
    noise_scale = 0.05 * (0.6 + 0.6 * load_b) * (1.0 + 1.5 * env["cavitation"])
    vib += noise_scale * rng.standard_normal(n_base)

    bearing = env["bearing_outer_race"]
    if bearing.any():
        # Impulse train at BPFO, each impulse ringing the bearing resonance.
        bpfo_phase = BPFO_ORDER * phase / (2 * np.pi)
        hits = np.flatnonzero((np.diff(np.floor(bpfo_phase)) > 0) & (bearing[1:] > 0)) + 1
        ring_len = int(0.02 * BASE_FS)
        kernel_t = np.arange(ring_len) / BASE_FS
        kernel = np.exp(-kernel_t / 0.003) * np.sin(2 * np.pi * BEARING_RESONANCE_HZ * kernel_t)
        impulses = np.zeros(n_base)
        impulses[hits] = 0.9 * bearing[hits] * (1 + 0.2 * rng.standard_normal(len(hits)))
        vib += np.convolve(impulses, kernel, mode="full")[:n_base]

    # --- ADC stage: anti-aliasing filter + decimation --------------------
    q = int(BASE_FS // fs)
    vib_s = resample_poly(vib, 1, q) if q > 1 else vib
    idx = np.arange(len(vib_s)) * q
    speed = speed_b[idx]
    load = load_b[idx]
    n = len(vib_s)
    t = idx / BASE_FS
    e = {k: v[idx] for k, v in env.items()}

    # --- motor current ----------------------------------------------------
    current = 4.0 + 8.0 * load * speed / 1500.0
    current *= 1.0 + 0.35 * e["electrical_overload"] - 0.08 * e["cavitation"]
    current += 0.08 * rng.standard_normal(n)

    # --- temperature: dT/dt = (T_eq - T) / tau ---------------------------
    t_eq = 35.0 + 2.2 * current + 2.0 * bearing[idx] + 18.0 * e["cooling_failure"]
    tau = 120.0  # s, thermal time constant of a small motor housing
    alpha = 1.0 - np.exp(-1.0 / (tau * fs))
    # Discrete first-order lag: T[i] = T[i-1] + alpha * (T_eq[i] - T[i-1]).
    temp, _ = lfilter([alpha], [1.0, alpha - 1.0], t_eq, zi=[(1.0 - alpha) * t_eq[0]])
    temp += 0.05 * rng.standard_normal(n)

    # --- pump pressure (affinity law: p ~ speed^2) ------------------------
    pressure = 4.0 * (speed / 1500.0) ** 2 * (1.0 - 0.15 * e["cavitation"])
    pressure += 0.03 * rng.standard_normal(n)
    speed_meas = speed + 3.0 * rng.standard_normal(n)

    signals = np.column_stack([vib_s, temp, current, speed_meas, pressure])

    labels = np.zeros(n, dtype=np.uint8)
    for ev in events:
        labels[(t >= ev.start_s) & (t < ev.end_s)] = 1

    if artefacts:
        # Short dropouts (NaN on a slow channel) and single-sample glitches.
        for _ in range(int(duration_s / 60)):
            ch = int(rng.integers(1, 5))
            start = int(rng.integers(0, n - int(0.2 * fs)))
            signals[start : start + int(0.2 * fs), ch] = np.nan
        for _ in range(int(duration_s / 30)):
            ch = int(rng.integers(1, 5))
            pos = int(rng.integers(0, n))
            signals[pos, ch] *= 25.0

    return SimulatedRun(fs=fs, signals=signals, labels=labels, events=events, seed=seed)


def simulate_with_faults(
    duration_s: float,
    fs: float,
    n_faults: int,
    seed: int,
    min_gap_s: float = 60.0,
    artefacts: bool = True,
) -> SimulatedRun:
    rng = np.random.default_rng(seed + 10_000)  # schedule stream independent of signal noise
    events = sample_fault_schedule(duration_s, n_faults, rng, min_gap_s=min_gap_s)
    return simulate(duration_s, fs, events, seed=seed, artefacts=artefacts)
