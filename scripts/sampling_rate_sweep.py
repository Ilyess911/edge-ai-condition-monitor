"""How does the sampling rate trade detection against computation?

    uv run python scripts/sampling_rate_sweep.py

For each ADC rate the whole chain is rebuilt: simulation with anti-aliasing,
windows of the same duration (so more samples per window at higher rates),
detectors refitted and recalibrated. Writes results/sampling_rate_sweep.json.
"""

from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy)

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarking import simulated_track as st  # noqa: E402
from src.benchmarking.profiling import environment, latency_summary  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.inference.packed_forest import PackedIsolationForestDetector  # noqa: E402

RATES = (250, 500, 1000, 2000)
MODELS = ("zscore", "pca", "iforest", "autoencoder")


def main() -> None:
    cfg = load_config("configs/simulated.toml")
    seeds = cfg["data"]["test_seeds"]
    t_start = time.time()
    env_start = environment()
    out = {}
    for fs in RATES:
        print(f"fs = {fs} Hz")
        train, calib = st.healthy_runs(cfg, fs=fs)
        X_train, X_calib = st.stream_features(cfg, train), st.stream_features(cfg, calib)
        runs = [st.test_run(cfg, s, fs=fs) for s in seeds]
        out[fs] = {}
        for name in MODELS:
            det, calib_ = st.fit_and_calibrate(cfg, name, X_train, X_calib)
            if name == "iforest":
                det = PackedIsolationForestDetector(det)  # same scores, affordable runtime
            detected = events = false = 0
            pr, feat_ns, pipe_ns, ingest, tput = [], [], [], [], []
            by_kind: dict = {}
            for run in runs:
                rep, res = st.evaluate(cfg, det, calib_, run)
                e = res["event"]
                detected += e["n_detected"]
                events += e["n_events"]
                false += e["n_false_alarms"]
                for kind, v in e["detected_by_kind"].items():
                    d = by_kind.setdefault(kind, [0, 0])
                    d[0] += v["detected"]
                    d[1] += v["total"]
                pr.append(res["window"]["pr_auc"])
                feat_ns.append(rep.column("feature_ns")[20:])
                pipe_ns.append(rep.column("pipeline_ns")[20:])
                ingest.append(rep.ingest_ns.sum() / rep.samples)
                tput.append(rep.throughput_sps)
            out[fs][name] = {
                "event_recall": detected / events, "detected": detected, "events": events,
                "false_alarms": false, "pr_auc_mean": float(np.mean(pr)),
                "pr_auc_std": float(np.std(pr)), "detected_by_kind": by_kind,
                "raise_after": calib_.raise_after,
                "feature_latency": latency_summary(np.concatenate(feat_ns), warmup=0),
                "pipeline_latency": latency_summary(np.concatenate(pipe_ns), warmup=0),
                "ingest_ns_per_sample": float(np.mean(ingest)),
                "throughput_sps_mean": float(np.mean(tput)),
                "realtime_factor_mean": float(np.mean(tput) / fs),
            }
            r = out[fs][name]
            print(f"  {name:<12} recall {r['event_recall']:.2f} ({detected}/{events}) "
                  f"false {false} PR-AUC {r['pr_auc_mean']:.3f} "
                  f"feature p50 {r['feature_latency']['p50_us']:.1f} us "
                  f"RTF {r['realtime_factor_mean']:.0f}x")
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "rates_hz": RATES,
               "test_seeds": seeds, "environment_start": env_start, "environment": environment(),
               "results": out, "wall_clock_s": time.time() - t_start}
    (ROOT / "results" / "sampling_rate_sweep.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote results/sampling_rate_sweep.json in {payload['wall_clock_s']:.0f} s")


if __name__ == "__main__":
    main()
