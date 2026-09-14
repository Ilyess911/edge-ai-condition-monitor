"""Lightweight monitoring dashboard (optional, not the contribution).

    uv run streamlit run src/dashboard/app.py

Replays a SIMULATED run with injected faults through the same StreamingEngine
the benchmarks use, and shows what an operator at the machine would see.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmarking import simulated_track as trk  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.simulator import simulate_with_faults  # noqa: E402
from src.inference.packed_forest import PackedIsolationForestDetector  # noqa: E402

st.set_page_config(page_title="Edge AI Condition Monitor", layout="wide")
CFG = load_config("configs/simulated.toml")


@st.cache_resource(show_spinner="Fitting detectors on simulated healthy data ...")
def fitted_models():
    train, calib = trk.healthy_runs(CFG)
    X_train, X_calib = trk.stream_features(CFG, train), trk.stream_features(CFG, calib)
    out = {}
    for name in ("zscore", "pca", "iforest", "autoencoder"):
        det, cal = trk.fit_and_calibrate(CFG, name, X_train, X_calib)
        if name == "iforest":
            det = PackedIsolationForestDetector(det)
        out[name] = (det, cal)
    return out


@st.cache_data
def demo_run(seed: int, minutes: int):
    run = simulate_with_faults(minutes * 60, CFG["stream"]["fs"], 5, seed, min_gap_s=60)
    return run.signals, run.labels, [(e.kind, e.start_s, e.end_s) for e in run.events]


st.title("Edge AI Condition Monitor")
st.caption("SIMULATED rotating machine, 1 kHz, 5 channels. Same engine as the benchmarks.")

with st.sidebar:
    model = st.selectbox("Detector", ["pca", "zscore", "iforest", "autoencoder"])
    speed = st.slider("Replay speed (x real time)", 1, 60, 20)
    seed = st.number_input("Scenario seed", value=7, step=1)
    start = st.button("Start stream", type="primary")

models = fitted_models()
det, cal = models[model]
signals, labels, events = demo_run(int(seed), 25)
fs, window, hop, chunk = trk.geometry(CFG)
st.sidebar.markdown(f"threshold `{cal.threshold:.3f}` · persistence `{cal.raise_after}` windows")
st.sidebar.markdown("**Injected faults (ground truth)**")
st.sidebar.dataframe(pd.DataFrame(events, columns=["fault", "start s", "end s"]).round(0),
                     hide_index=True)

row = st.columns(6)
tiles = [c.empty() for c in row]
chart = st.empty()
lower = st.columns([2, 1])
alerts_box, perf_box = lower[0].empty(), lower[1].empty()

if start:
    engine = trk.make_engine(CFG, det, cal)
    engine.reset()
    hist_t, hist_s = [], []
    inf_ns, samples = [], 0
    wall0 = time.perf_counter()
    last_draw = 0.0
    for i in range(0, len(signals), chunk):
        due = wall0 + (i + chunk) / (fs * speed)
        delay = due - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        block = signals[i : i + chunk]
        _, recs = engine.process_chunk(block)
        samples += len(block)
        for r in recs:
            hist_t.append(r.t_end)
            hist_s.append(r.score / cal.threshold)
            inf_ns.append(r.inference_ns)
        if not recs or time.perf_counter() - last_draw < 0.25:
            continue
        last_draw = time.perf_counter()
        seg = signals[max(0, i + chunk - window) : i + chunk]
        state = engine.alerts.state.value
        tiles[0].metric("Vibration RMS [g]", f"{np.nanstd(seg[:, 0]):.3f}")
        tiles[1].metric("Temperature [°C]", f"{np.nanmean(seg[:, 1]):.1f}")
        tiles[2].metric("Current [A]", f"{np.nanmean(seg[:, 2]):.2f}")
        tiles[3].metric("Speed [rpm]", f"{np.nanmean(seg[:, 3]):.0f}")
        tiles[4].metric("Score / threshold", f"{hist_s[-1]:.2f}")
        icon = {"OK": "🟢", "WARNING": "🟠", "ALARM": "🔴"}[state]
        tiles[5].metric("System health", f"{icon} {state}")
        view = pd.DataFrame({"t [s]": hist_t[-240:], "score / threshold": np.clip(hist_s[-240:], 0, 8),
                             "threshold": 1.0}).set_index("t [s]")
        chart.line_chart(view, height=260)
        alerts_box.dataframe(pd.DataFrame(
            [{"start s": a.start_t, "end s": a.end_t, "peak score": round(a.peak_score, 2),
              "windows": a.n_windows} for a in engine.alerts.history]
            or [{"start s": None, "end s": None, "peak score": None, "windows": None}]),
            hide_index=True, width="stretch")
        lat = np.array(inf_ns[-400:]) / 1e3
        perf_box.markdown(
            f"**Inference latency** p50 `{np.median(lat):.1f} µs` · p99 `{np.percentile(lat, 99):.1f} µs`  \n"
            f"**Stream time** `{samples / fs:.0f} s` · **processed** `{samples / (time.perf_counter() - wall0):,.0f}` samples/s "
            f"(paced at {speed}x of {fs:.0f} Hz)  \n"
            f"Drawing shares the thread between chunks; inference timing excludes it.")
    engine.alerts.close(len(signals) / fs)
    st.success(f"Stream finished: {len(engine.alerts.history)} alerts, {len(events)} injected faults.")
