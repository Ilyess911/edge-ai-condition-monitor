"""Real-time streaming demo in the terminal.

    uv run python scripts/stream_demo.py                      # PCA, 1 kHz, real time
    uv run python scripts/stream_demo.py --model iforest --speed 4

Trains on a healthy simulated run, calibrates on another, then replays a
SIMULATED 3-minute run with two injected faults at the sensor rate (or `speed`
times faster) and prints one status line per second.
"""

from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy)

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarking import simulated_track as st  # noqa: E402
from src.benchmarking.profiling import latency_summary  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.simulator import FaultEvent, simulate  # noqa: E402
from src.inference.packed_forest import PackedIsolationForestDetector  # noqa: E402

DEMO_EVENTS = [
    FaultEvent("imbalance", start_s=50.0, duration_s=35.0, severity=0.8),
    FaultEvent("bearing_outer_race", start_s=120.0, duration_s=50.0, severity=0.9),
]


def bar(ratio: float, width: int = 20) -> str:
    filled = int(min(ratio, 2.0) / 2.0 * width)
    return "#" * filled + "." * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pca", choices=["zscore", "pca", "iforest", "autoencoder"])
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 1 = real time")
    ap.add_argument("--seconds", type=float, default=180.0)
    args = ap.parse_args()

    cfg = load_config("configs/simulated.toml")
    print("fitting on 60 min of simulated healthy operation ...")
    train, calib = st.healthy_runs(cfg)
    det, calib_ = st.fit_and_calibrate(cfg, args.model, st.stream_features(cfg, train),
                                       st.stream_features(cfg, calib))
    if args.model == "iforest":
        det = PackedIsolationForestDetector(det)
    fs, window, hop, chunk = st.geometry(cfg)
    run = simulate(args.seconds, fs, DEMO_EVENTS, seed=4242)
    print(f"model={args.model} threshold={calib_.threshold:.3f} raise_after={calib_.raise_after} "
          f"fs={fs:.0f} Hz window={window} hop={hop} chunk={chunk}")
    print("SIMULATED stream, injected faults: "
          + ", ".join(f"{e.kind} @ {e.start_s:.0f}-{e.end_s:.0f} s" for e in DEMO_EVENTS))
    print(f"{'t [s]':>6} {'vib rms':>8} {'temp':>6} {'curr':>6}  {'score/threshold':<22} "
          f"{'state':<8} {'infer':>8}")

    eng = st.make_engine(cfg, det, calib_, fs=fs)
    last_print = [-1]
    sig = run.signals

    def on_window(rec):
        t_stream = rec.t_end
        if int(t_stream) == last_print[0]:
            return
        last_print[0] = int(t_stream)
        i = int(t_stream * fs)
        seg = sig[max(0, i - window) : i]
        truth = "  <- fault active" if run.labels[min(i, len(run.labels) - 1)] else ""
        print(f"{t_stream:6.0f} {np.nanstd(seg[:, 0]):8.3f} {np.nanmean(seg[:, 1]):6.1f} "
              f"{np.nanmean(seg[:, 2]):6.2f}  {bar(rec.score / calib_.threshold)} "
              f"{rec.health.value:<8} {rec.inference_ns / 1e3:6.1f}us{truth}")

    report = eng.run(sig, chunk, paced=True, on_window=on_window, speed=args.speed)
    print("\nalert history")
    for a in eng.alerts.history:
        print(f"  ALARM {a.start_t:6.1f} s -> {a.end_t:6.1f} s  peak score {a.peak_score:.2f}")
    lat = latency_summary(report.column("inference_ns"))
    pipe = latency_summary(report.column("pipeline_ns"))
    print(f"\nwindows {report.windows}, inference p50 {lat['p50_us']:.1f} us p99 {lat['p99_us']:.1f} us, "
          f"pipeline p50 {pipe['p50_us']:.1f} us, CPU {100 * report.cpu_utilisation:.1f} % of one core "
          f"(includes terminal printing), max lateness {report.max_lateness_s * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
