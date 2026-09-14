"""Glue for the simulated track: train, calibrate and evaluate through the
streaming engine, so training features and test features come from exactly the
code path that runs online."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.alerts.engine import AlertEngine
from src.benchmarking.metrics import (
    aftermath_mask,
    alarm_time_fraction,
    chance_detections,
    event_metrics,
    window_labels,
    window_metrics,
)
from src.config import detector_kwargs
from src.data.simulator import SimulatedRun, simulate, simulate_with_faults
from src.features.simulated import SimulatedFeatureExtractor
from src.inference.threshold import calibrate_persistence, calibrate_threshold
from src.models.detectors import build
from src.preprocessing.cleaning import SampleCleaner
from src.streaming.engine import NullDetector, StreamingEngine, StreamReport


def geometry(cfg: dict, fs: float | None = None) -> tuple[float, int, int, int]:
    s = cfg["stream"]
    fs = fs or s["fs"]
    return fs, int(s["window_s"] * fs), int(s["hop_s"] * fs), max(1, int(s["chunk_s"] * fs))


@dataclass
class Calibration:
    threshold: float
    raise_after: int


def make_engine(cfg: dict, detector, calib: "Calibration | None", fs: float | None = None,
                keep_features: bool = False) -> StreamingEngine:
    fs, window, hop, _ = geometry(cfg, fs)
    c = cfg["cleaning"]
    cleaner = SampleCleaner(c["low"], c["high"], max_hold=int(c["max_hold_s"] * fs))
    alerts = None
    if calib is not None:
        alerts = AlertEngine(calib.threshold, calib.raise_after, cfg["alerts"]["clear_after"])
    return StreamingEngine(fs, window, hop, cleaner, SimulatedFeatureExtractor(fs, window),
                           detector, alerts, keep_features=keep_features)


def healthy_runs(cfg: dict, fs: float | None = None) -> tuple[SimulatedRun, SimulatedRun]:
    d = cfg["data"]
    fs = fs or cfg["stream"]["fs"]
    train = simulate(d["train_duration_s"], fs, seed=d["train_seed"])
    calib = simulate(d["calib_duration_s"], fs, seed=d["calib_seed"])
    return train, calib


def test_run(cfg: dict, seed: int, fs: float | None = None) -> SimulatedRun:
    d = cfg["data"]
    return simulate_with_faults(d["test_duration_s"], fs or cfg["stream"]["fs"],
                                d["faults_per_run"], seed, min_gap_s=d["min_gap_s"])


def stream_features(cfg: dict, run: SimulatedRun) -> np.ndarray:
    eng = make_engine(cfg, NullDetector(), None, fs=run.fs, keep_features=True)
    _, _, _, chunk = geometry(cfg, run.fs)
    eng.run(run.signals, chunk)
    return np.array(eng.features)


def fit_and_calibrate(cfg: dict, model_name: str, X_train: np.ndarray, X_calib: np.ndarray,
                      **overrides):
    name, kwargs = detector_kwargs(cfg["models"][model_name])
    kwargs.update(overrides)
    det = build(name, **kwargs).fit(X_train)
    return det, calibrate(cfg, det, X_calib)


def calibrate(cfg: dict, detector, X_calib: np.ndarray, raise_after=None) -> Calibration:
    """Threshold and alert persistence from healthy calibration windows only."""
    scores = detector.score(X_calib)
    thr = calibrate_threshold(scores, cfg["threshold"]["quantile"])
    k = raise_after if raise_after is not None else cfg["alerts"]["raise_after"]
    if k == "auto":
        k = calibrate_persistence(scores, thr, cfg["alerts"]["max_raise_after"])
    return Calibration(thr, int(k))


def evaluate(cfg: dict, detector, calib: Calibration, run: SimulatedRun,
             paced: bool = False) -> tuple[StreamReport, dict]:
    eng = make_engine(cfg, detector, calib, fs=run.fs)
    fs, window, _, chunk = geometry(cfg, run.fs)
    report = eng.run(run.signals, chunk, paced=paced)
    end_idx = np.round(report.column("t_end") * fs).astype(np.int64)
    y = window_labels(run.labels, end_idx, window)
    scores = report.column("score")
    ev_cfg = cfg["evaluation"]
    ignore = aftermath_mask(report.column("t_end"), y, run.events, ev_cfg["recovery_s"])
    wm = window_metrics(y[~ignore], scores[~ignore], calib.threshold)
    em = event_metrics(run.events, eng.alerts.history, run.duration_s, ev_cfg["grace_s"],
                       ev_cfg["recovery_s"])
    event = em.to_dict()
    event["alarm_time_fraction"] = alarm_time_fraction(run.events, eng.alerts.history,
                                                       run.duration_s, recovery_s=ev_cfg["recovery_s"])
    event["chance"] = chance_detections(run.events, eng.alerts.history, run.duration_s,
                                        ev_cfg["grace_s"], n_shifts=500, seed=run.seed)
    return report, {"window": wm.__dict__, "event": event, "y": y, "ignore": ignore,
                    "scores": scores, "alerts": list(eng.alerts.history)}
