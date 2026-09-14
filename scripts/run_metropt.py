"""MetroPT-3 benchmark: the same pipeline on real compressor data.

    uv run python scripts/download_metropt.py
    uv run python scripts/run_metropt.py

Writes results/metropt_benchmark.json and results/traces/metropt_*.npz.
"""

from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy)

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts.engine import AlertEngine  # noqa: E402
from src.benchmarking.metrics import (  # noqa: E402
    aftermath_mask, event_metrics, window_labels, window_metrics,
)
from src.benchmarking.profiling import environment, latency_summary  # noqa: E402
from src.config import ROOT, detector_kwargs, load_config  # noqa: E402
from src.data.metropt import load_grid, report_events  # noqa: E402
from src.features.metropt import MetroPTFeatureExtractor  # noqa: E402
from src.inference.packed_forest import PackedIsolationForestDetector  # noqa: E402
from src.inference.threshold import calibrate_persistence, calibrate_threshold  # noqa: E402
from src.models.detectors import build  # noqa: E402
from src.preprocessing.cleaning import SampleCleaner  # noqa: E402
from src.streaming.engine import StreamingEngine  # noqa: E402


class _Null:
    def score(self, X):
        return np.zeros(len(X))


def engine(cfg, detector, alerts=None, keep=False):
    s, c = cfg["stream"], cfg["cleaning"]
    return StreamingEngine(
        fs=0.1, window=s["window"], hop=s["hop"],
        cleaner=SampleCleaner(c["low"], c["high"], max_hold=s["max_hold"]),
        extractor=MetroPTFeatureExtractor(s["min_coverage"]),
        detector=detector, alerts=alerts, keep_features=keep,
    )


def main() -> None:
    cfg = load_config("configs/metropt.toml")
    t_start = time.time()
    env_start = environment()
    grid = load_grid()
    sp = cfg["split"]
    part = lambda a, b: grid.signals[grid.index_of(sp[a]) : grid.index_of(sp[b])]  # noqa: E731
    train, calib, test = part("train_start", "train_end"), part("calib_start", "calib_end"), \
        part("test_start", "test_end")
    chunk = cfg["stream"]["chunk"]

    feats = {}
    for name, sig in (("train", train), ("calib", calib)):
        eng = engine(cfg, _Null(), keep=True)
        eng.run(sig, chunk)
        feats[name] = np.array(eng.features)
        print(f"{name}: {len(sig)} grid samples, {len(feats[name])} windows, "
              f"{eng.skipped_windows} skipped for missing data")

    offset = grid.seconds(sp["test_start"])
    events = report_events(grid, offset_s=offset)
    duration = len(test) * 10.0
    ev_cfg = cfg["evaluation"]

    detectors = {}
    for name, mcfg in cfg["models"].items():
        kind, kwargs = detector_kwargs(mcfg)
        detectors[name] = build(kind, **kwargs).fit(feats["train"])
    detectors["iforest-packed"] = PackedIsolationForestDetector(detectors["iforest"])

    results = {}
    for name, det in detectors.items():
        cs = det.score(feats["calib"])
        thr = calibrate_threshold(cs, cfg["threshold"]["quantile"])
        k = calibrate_persistence(cs, thr, cfg["alerts"]["max_raise_after"])
        alerts = AlertEngine(thr, k, cfg["alerts"]["clear_after"])
        eng = engine(cfg, det, alerts)
        rep = eng.run(test, chunk)
        t = rep.column("t_end")
        end_idx = np.round(t * 0.1).astype(np.int64)
        labels = np.zeros(len(test), dtype=np.uint8)
        for ev in events:
            labels[int(ev.start_s / 10) : int(ev.end_s / 10) + 1] = 1
        y = window_labels(labels, end_idx, cfg["stream"]["window"])
        scores = rep.column("score")
        ignore = aftermath_mask(t, y, events, 0.0)
        wm = window_metrics(y[~ignore], scores[~ignore], thr)
        strict = event_metrics(events, alerts.history, duration, ev_cfg["grace_s"])
        early = event_metrics(events, alerts.history, duration, ev_cfg["grace_s"],
                              lead_s=ev_cfg["early_warning_s"])
        per_event = []
        for ev in events:
            hit = [a for a in alerts.history
                   if a.start_t <= ev.end_s and (a.end_t or duration) >= ev.start_s]
            per_event.append({"start_h": ev.start_s / 3600, "detected": bool(hit),
                              "delay_min": (max(0.0, min(a.start_t for a in hit) - ev.start_s) / 60
                                            if hit else None)})
        results[name] = {
            "threshold": thr, "raise_after": k,
            "window": wm.__dict__, "event_strict": strict.to_dict(),
            "event_early_2h": early.to_dict(), "per_event": per_event,
            "false_alarms_per_day": strict.false_alarms_per_hour * 24,
            "inference_latency": latency_summary(rep.column("inference_ns")),
            "pipeline_latency": latency_summary(rep.column("pipeline_ns")),
            "throughput_sps": rep.throughput_sps, "windows": rep.windows,
            "skipped_windows": eng.skipped_windows,
        }
        np.savez_compressed(ROOT / "results" / "traces" / f"metropt_{name}.npz",
                            t=t, score=scores, y=y, threshold=thr,
                            alerts=np.array([[a.start_t, a.end_t] for a in alerts.history]).reshape(-1, 2),
                            events=np.array([[e.start_s, e.end_s] for e in events]))
        r = results[name]
        print(f"{name:<15} k={k:<2} PR-AUC {wm.pr_auc:.3f} ROC-AUC {wm.roc_auc:.3f} "
              f"detected {strict.n_detected}/4 (2h early view {early.n_detected}/4) "
              f"alerts {strict.n_alerts} false/day {r['false_alarms_per_day']:.2f} "
              f"inf p50 {r['inference_latency']['p50_us']:.1f} us")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "config": "configs/metropt.toml",
        "train_windows": int(len(feats["train"])), "calib_windows": int(len(feats["calib"])),
        "test_hours": duration / 3600, "environment_start": env_start,
        "environment": environment(), "results": results, "wall_clock_s": time.time() - t_start,
    }
    (ROOT / "results" / "metropt_benchmark.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote results/metropt_benchmark.json in {payload['wall_clock_s']:.0f} s")


if __name__ == "__main__":
    main()
