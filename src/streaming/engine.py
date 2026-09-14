"""Streaming engine: online, window-by-window inference over a replayed sensor stream.

    chunk of raw samples
      -> SampleCleaner        (plausibility check, sample-and-hold)
      -> SlidingWindow        (circular buffer, emits every `hop` samples)
      -> feature extractor    (one feature vector per window, or None to skip it)
      -> detector.score       (one anomaly score per window)
      -> AlertEngine          (persistence + hysteresis)

Samples arrive in chunks, as they do from a DAQ driver that hands over its DMA
buffer: chunk_size = 1 is the sample-by-sample limit. Every stage is timed
with perf_counter_ns around the call, on the same thread, one window at a time
(no batching across windows: an online monitor cannot wait for future data).

This is soft real-time at best: CPython, a garbage collector and a general-purpose
OS scheduler give no latency guarantee. The engine measures lateness behind the
sensor clock (paced mode) instead of assuming it is zero.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.alerts.engine import AlertEngine, Health
from src.preprocessing.cleaning import SampleCleaner
from src.preprocessing.windowing import SlidingWindow


class NullDetector:
    """Stands in for a model while collecting training features: scores are ignored."""

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


@dataclass
class WindowRecord:
    t_end: float
    score: float
    health: Health
    feature_ns: int
    inference_ns: int
    alert_ns: int

    @property
    def pipeline_ns(self) -> int:
        return self.feature_ns + self.inference_ns + self.alert_ns


@dataclass
class StreamReport:
    fs: float
    samples: int
    windows: int
    wall_s: float
    cpu_s: float
    ingest_ns: np.ndarray  # per chunk: cleaning + buffering, excluding window work
    records: list[WindowRecord] = field(default_factory=list)
    paced: bool = False
    max_lateness_s: float = 0.0  # paced mode only: worst delay behind the sample clock

    @property
    def throughput_sps(self) -> float:
        return self.samples / self.wall_s if self.wall_s > 0 else float("nan")

    @property
    def cpu_utilisation(self) -> float:
        """CPU seconds per wall second for this process (1.0 = one full core)."""
        return self.cpu_s / self.wall_s if self.wall_s > 0 else float("nan")

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in self.records])


class StreamingEngine:
    def __init__(
        self,
        fs: float,
        window: int,
        hop: int,
        cleaner: SampleCleaner,
        extractor: Callable[[np.ndarray], np.ndarray],
        detector,
        alerts: AlertEngine | None = None,
        keep_features: bool = False,
    ):
        self.fs = fs
        self.cleaner = cleaner
        self.windower = SlidingWindow(window, hop, len(cleaner.low))
        self.extractor = extractor
        self.detector = detector
        self.alerts = alerts
        self.keep_features = keep_features
        self.features: list[np.ndarray] = []
        self.skipped_windows = 0

    def reset(self) -> None:
        self.skipped_windows = 0
        self.cleaner.reset()
        self.windower.reset()
        self.features.clear()
        if self.alerts is not None:
            self.alerts.reset()

    def process_chunk(self, chunk: np.ndarray) -> tuple[int, list[WindowRecord]]:
        clock = time.perf_counter_ns
        t0 = clock()
        cleaned = self.cleaner.process(chunk)
        emitted = self.windower.push(cleaned)
        ingest_ns = clock() - t0

        out = []
        for end_idx, w in emitted:
            a = clock()
            f = self.extractor(w)
            if f is None:  # extractor refused the window (e.g. too much missing data)
                self.skipped_windows += 1
                continue
            b = clock()
            s = float(self.detector.score(f[None, :])[0])
            c = clock()
            t_end = end_idx / self.fs
            health = self.alerts.update(t_end, s) if self.alerts is not None else Health.OK
            d = clock()
            if self.keep_features:
                self.features.append(f)
            out.append(WindowRecord(t_end, s, health, b - a, c - b, d - c))
        return ingest_ns, out

    def run(
        self,
        signals: np.ndarray,
        chunk_size: int,
        paced: bool = False,
        on_window: Callable[[WindowRecord], None] | None = None,
        speed: float = 1.0,
    ) -> StreamReport:
        """Replay `signals` through the pipeline.

        paced=False: as fast as possible, measures maximum throughput.
        paced=True:  each chunk is released when the sensor would have produced
                     it, measures CPU utilisation and lateness at the real rate.
        speed:       paced replay clock multiplier (demo only; benchmarks use 1).
        """
        self.reset()
        records: list[WindowRecord] = []
        ingest = []
        max_late = 0.0
        n = len(signals)
        cpu0 = time.process_time()
        wall0 = time.perf_counter()
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            if paced:
                due = wall0 + stop / (self.fs * speed)  # last sample of the chunk exists now
                now = time.perf_counter()
                if due > now:
                    time.sleep(due - now)
                else:
                    max_late = max(max_late, now - due)
            ing, recs = self.process_chunk(signals[start:stop])
            ingest.append(ing)
            records.extend(recs)
            if on_window is not None:
                for r in recs:
                    on_window(r)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        if self.alerts is not None:
            self.alerts.close(n / self.fs)
        return StreamReport(
            fs=self.fs,
            samples=n,
            windows=len(records),
            wall_s=wall,
            cpu_s=cpu,
            ingest_ns=np.array(ingest, dtype=np.int64),
            records=records,
            paced=paced,
            max_lateness_s=max_late,
        )
