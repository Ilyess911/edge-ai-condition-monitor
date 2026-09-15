"""Evaluation glue for the Paderborn real-bearing track (shared by the benchmark and the dashboard).

Features come out of the StreamingEngine (one engine run per 4 s recording,
reset in between because recordings are not continuous). Scoring and alerting
are then replayed per recording with the same detector.score and AlertEngine
the engine uses, so that twelve model/fold combinations do not repeat the FFTs.
tests/test_paderborn_track.py checks that replay and full engine agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, roc_auc_score

from src.alerts.engine import AlertEngine
from src.config import ROOT, detector_kwargs
from src.data.paderborn import FS, Recording
from src.features.bearing import BearingFeatureExtractor
from src.preprocessing.cleaning import SampleCleaner
from src.streaming.engine import NullDetector, StreamingEngine


def geometry(cfg: dict, fs: float) -> tuple[int, int, int]:
    s = cfg["stream"]
    return int(s["window_s"] * fs), int(s["hop_s"] * fs), max(1, int(s["chunk_s"] * fs))


def recording_signals(rec: Recording, fs: float) -> np.ndarray:
    """Columns: vibration, current, speed, torque, force at `fs` (anti-aliased)."""
    q = int(FS // fs)
    vib = rec.vibration.astype(np.float64)
    cur = rec.current.astype(np.float64)
    if q > 1:
        vib, cur = resample_poly(vib, 1, q), resample_poly(cur, 1, q)
    n = len(vib)
    ones = np.ones(n)
    return np.column_stack([vib, cur, rec.speed_rpm * ones, rec.torque_nm * ones, rec.force_n * ones])


def make_engine(cfg: dict, fs: float, detector=None, alerts: AlertEngine | None = None,
                keep_features: bool = False, feature_idx: list[int] | None = None) -> StreamingEngine:
    window, hop, _ = geometry(cfg, fs)
    c = cfg["cleaning"]
    cleaner = SampleCleaner(c["low"], c["high"], max_hold=max(1, int(c["max_hold_s"] * fs)))
    extractor = BearingFeatureExtractor(fs, window)
    if feature_idx is not None:
        full = extractor
        extractor = lambda w: full(w)[feature_idx]  # noqa: E731
    return StreamingEngine(fs, window, hop, cleaner, extractor,
                           detector or NullDetector(), alerts, keep_features=keep_features)


@dataclass
class BearingFeatures:
    code: str
    features: list[np.ndarray] = field(default_factory=list)  # one (n_windows, 15) per recording
    conditions: list[str] = field(default_factory=list)
    feature_ns: list[np.ndarray] = field(default_factory=list)
    ingest_ns_per_sample: list[float] = field(default_factory=list)
    throughput_sps: list[float] = field(default_factory=list)

    @property
    def stacked(self) -> np.ndarray:
        return np.vstack(self.features)


def extract_bearing(cfg: dict, recordings: list[Recording], fs: float) -> BearingFeatures:
    _, _, chunk = geometry(cfg, fs)
    out = BearingFeatures(recordings[0].bearing)
    eng = make_engine(cfg, fs, keep_features=True)
    for rec in recordings:
        rep = eng.run(recording_signals(rec, fs), chunk)
        out.features.append(np.array(eng.features))
        out.conditions.append(rec.condition)
        out.feature_ns.append(rep.column("feature_ns"))
        out.ingest_ns_per_sample.append(float(rep.ingest_ns.sum() / rep.samples))
        out.throughput_sps.append(rep.throughput_sps)
    return out


FEATURE_CACHE = ROOT / "data" / "cache" / "paderborn_features"


def load_features(cfg: dict, fs: float, codes, verbose: bool = False) -> dict[str, BearingFeatures]:
    """Features per bearing at `fs`, extracted once through the engine and cached."""
    from src.data.paderborn import load_bearing

    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for code in codes:
        path = FEATURE_CACHE / f"{code}_{int(fs)}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            bounds = np.cumsum(np.concatenate([[0], z["n_windows"]]))
            pairs = list(zip(bounds, bounds[1:]))
            bf = BearingFeatures(code, [z["features"][a:b] for a, b in pairs], list(z["conditions"]),
                                 [z["feature_ns"][a:b] for a, b in pairs],
                                 list(z["ingest_ns_per_sample"]), list(z["throughput_sps"]))
        else:
            bf = extract_bearing(cfg, load_bearing(code), fs)
            np.savez(path, features=np.vstack(bf.features),
                     n_windows=np.array([len(f) for f in bf.features]),
                     conditions=np.array(bf.conditions), feature_ns=np.concatenate(bf.feature_ns),
                     ingest_ns_per_sample=np.array(bf.ingest_ns_per_sample),
                     throughput_sps=np.array(bf.throughput_sps))
        out[code] = bf
        if verbose:
            print(f"  features {code} @ {fs / 1000:.0f} kHz: {sum(len(f) for f in bf.features)} windows",
                  flush=True)
    return out


def subset(feats: dict[str, BearingFeatures], idx: list[int] | None) -> dict[str, BearingFeatures]:
    if idx is None:
        return feats
    return {c: BearingFeatures(c, [f[:, idx] for f in bf.features], bf.conditions, bf.feature_ns,
                               bf.ingest_ns_per_sample, bf.throughput_sps)
            for c, bf in feats.items()}


def replay_alerts(scores: np.ndarray, hop_s: float, threshold: float, raise_after: int,
                  clear_after: int) -> bool:
    """True if the alert engine raises at least one alert within this recording."""
    eng = AlertEngine(threshold, raise_after, clear_after)
    for i, s in enumerate(scores):
        eng.update((i + 1) * hop_s, float(s))
    return len(eng.history) > 0


def score_recordings(detector, bf: BearingFeatures) -> tuple[list[np.ndarray], np.ndarray]:
    """Score window by window, timing each call exactly as the engine does."""
    import time

    clock = time.perf_counter_ns
    scores, lat = [], []
    for F in bf.features:
        s = np.empty(len(F))
        for i, row in enumerate(F):
            a = clock()
            s[i] = float(detector.score(row[None, :])[0])
            lat.append(clock() - a)
        scores.append(s)
    return scores, np.array(lat)


def longest_run_within(scores_per_rec: list[np.ndarray], threshold: float) -> int:
    best = 0
    for s in scores_per_rec:
        cur = 0
        for v in s:
            cur = cur + 1 if v > threshold else 0
            best = max(best, cur)
    return best


@dataclass
class FittedMonitor:
    scorer: object  # detector used for scoring (packed runtime for Isolation Forest)
    detector: object  # fitted detector as trained
    threshold: float
    raise_after: int
    fit_s: float
    train_windows: int


def fit_fold(cfg: dict, feats: dict[str, BearingFeatures], fold: dict, model_cfg: dict) -> FittedMonitor:
    """Fit on the fold's training bearings, calibrate threshold and persistence on its
    calibration bearing. No damaged bearing and no held-out healthy bearing is touched."""
    import time

    from src.inference.packed_forest import PackedIsolationForestDetector
    from src.inference.threshold import calibrate_threshold
    from src.models.detectors import build

    kind, kwargs = detector_kwargs(model_cfg)
    X_train = np.vstack([feats[c].stacked for c in fold["train"]])
    t0 = time.perf_counter()
    det = build(kind, **kwargs).fit(X_train)
    fit_s = time.perf_counter() - t0
    scorer = PackedIsolationForestDetector(det) if kind == "iforest" else det
    calib_scores: list[np.ndarray] = []
    for code in fold["calib"]:
        s, _ = score_recordings(scorer, feats[code])
        calib_scores += s
    thr = calibrate_threshold(np.concatenate(calib_scores), cfg["threshold"]["quantile"])
    k = min(longest_run_within(calib_scores, thr) + 1, cfg["alerts"]["max_raise_after"])
    return FittedMonitor(scorer, det, thr, k, fit_s, int(len(X_train)))


def evaluate_fold(test: dict[str, tuple[list[np.ndarray], bool]], threshold: float,
                  raise_after: int, clear_after: int, hop_s: float) -> dict:
    """test: bearing -> (scores per recording, is_damaged)."""
    per_bearing = {}
    y, s = [], []
    for code, (scores, damaged) in test.items():
        flags = [replay_alerts(sc, hop_s, threshold, raise_after, clear_after) for sc in scores]
        per_bearing[code] = {"damaged": damaged, "recordings": len(flags),
                             "flagged": int(sum(flags))}
        for sc in scores:
            y.append(np.full(len(sc), int(damaged)))
            s.append(sc)
    y, s = np.concatenate(y), np.concatenate(s)
    dmg = [v for v in per_bearing.values() if v["damaged"]]
    hlt = [v for v in per_bearing.values() if not v["damaged"]]
    rec_detect = sum(v["flagged"] for v in dmg) / sum(v["recordings"] for v in dmg)
    rec_false = sum(v["flagged"] for v in hlt) / sum(v["recordings"] for v in hlt)
    return {
        "window_roc_auc": float(roc_auc_score(y, s)),
        "window_pr_auc": float(average_precision_score(y, s)),
        "window_damaged_over_threshold": float(np.mean(s[y == 1] > threshold)),
        "window_healthy_over_threshold": float(np.mean(s[y == 0] > threshold)),
        "recording_detection_rate": rec_detect,
        "recording_false_alarm_rate": rec_false,
        "bearings_detected_majority": int(sum(v["flagged"] > v["recordings"] / 2 for v in dmg)),
        "healthy_bearings_flagged_majority": int(sum(v["flagged"] > v["recordings"] / 2 for v in hlt)),
        "per_bearing": per_bearing,
    }
