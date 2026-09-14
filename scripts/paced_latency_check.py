"""Does latency measured in a hot loop hold when the stream is paced?

    uv run python scripts/paced_latency_check.py

The main benchmark times inference while replaying as fast as possible. A real
monitor mostly waits for samples, and an idle CPU can downclock or migrate the
thread between calls. This replays the same 60 s of simulated data (dev seed)
alternately unpaced and paced, three times each, and writes
results/paced_vs_unpaced.json.
"""

from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy)

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarking import simulated_track as st  # noqa: E402
from src.benchmarking.profiling import environment, latency_summary  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402


def main() -> None:
    cfg = load_config("configs/simulated.toml")
    env_start = environment()
    train, calib = st.healthy_runs(cfg)
    X_train, X_calib = st.stream_features(cfg, train), st.stream_features(cfg, calib)
    run = st.test_run(cfg, cfg["data"]["dev_seed"])
    fs, _, _, chunk = st.geometry(cfg)
    signals = run.signals[: int(60 * fs)]
    out = {}
    for name in ("zscore", "pca", "autoencoder"):
        det, cal = st.fit_and_calibrate(cfg, name, X_train, X_calib)
        out[name] = []
        for rep_i in range(3):
            for paced in (False, True):
                eng = st.make_engine(cfg, det, cal)
                rep = eng.run(signals, chunk, paced=paced)
                row = {"repeat": rep_i, "paced": paced,
                       "inference": latency_summary(rep.column("inference_ns")),
                       "pipeline": latency_summary(rep.column("pipeline_ns"))}
                out[name].append(row)
                print(f"{name:<12} repeat {rep_i} paced={paced!s:<5} inference p50 "
                      f"{row['inference']['p50_us']:6.1f} µs  pipeline p50 {row['pipeline']['p50_us']:6.1f} µs")
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds_per_replay": 60,
               "environment_start": env_start, "environment": environment(), "results": out}
    (ROOT / "results" / "paced_vs_unpaced.json").write_text(json.dumps(payload, indent=2))
    print("wrote results/paced_vs_unpaced.json")


if __name__ == "__main__":
    main()
