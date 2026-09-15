"""Monitoring dashboard on REAL Paderborn bearing recordings (optional, not the contribution).

    uv run streamlit run src/dashboard/app.py

Needs the Paderborn data (`make paderborn`). The detector is fitted and
calibrated exactly as in the benchmark fold you pick; the bearing you replay
is one the detector never saw. Recordings go through the same StreamingEngine
and AlertEngine as scripts/run_paderborn.py, paced at the sensor clock (or
faster), one 4 s recording at a time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alerts.engine import AlertEngine  # noqa: E402
from src.benchmarking import paderborn_track as pt  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.data.paderborn import CACHE, CONDITIONS, HEALTHY, RAW, REAL_DAMAGE, Recording, load_bearing  # noqa: E402
from src.features.bearing import FEATURE_NAMES, BearingFeatureExtractor, group_indices  # noqa: E402

st.set_page_config(page_title="Edge AI Condition Monitor", layout="wide")
CFG = load_config("configs/paderborn.toml")
FOLDS = {f["name"]: f for f in CFG["folds"]}
MODELS = {"PCA (SPE + T²)": "pca", "Z-score": "zscore", "Isolation Forest (packed)": "iforest",
          "Autoencoder": "autoencoder"}
GROUPS = {"All 15 features": "all", "Envelope + context (5)": "envelope_context"}
RATES = {"64 kHz (native)": 64_000, "16 kHz": 16_000, "8 kHz": 8_000}
CONDITION_LABELS = {
    "N15_M07_F10": "1500 rpm, 0.7 Nm, 1000 N",
    "N09_M07_F10": "900 rpm, 0.7 Nm, 1000 N",
    "N15_M01_F10": "1500 rpm, 0.1 Nm, 1000 N",
    "N15_M07_F04": "1500 rpm, 0.7 Nm, 400 N",
}


def damage_profiles() -> dict:
    path = ROOT / "results" / "paderborn_benchmark.json"
    return json.loads(path.read_text())["damage_profiles"] if path.exists() else {}


@st.cache_resource(show_spinner="Fitting and calibrating on the fold's healthy bearings ...")
def monitor(fold_name: str, model: str, group: str, fs: int):
    fold = FOLDS[fold_name]
    feats = pt.subset(pt.load_features(CFG, fs, fold["train"] + fold["calib"]), group_indices(group))
    return pt.fit_fold(CFG, feats, fold, CFG["models"][model])


@st.cache_data(show_spinner="Loading recordings ...")
def recordings(code: str, condition: str, n: int) -> list[tuple]:
    recs = [r for r in load_bearing(code) if r.condition == condition][:n]
    return [(r.index, r.vibration, r.current, r.speed_rpm, r.torque_nm, r.force_n) for r in recs]


st.title("Edge AI Condition Monitor")
st.caption("REAL bearing recordings from the Paderborn University test bench, replayed through the "
           "same streaming engine as the benchmark. Laptop timings, not an edge device.")

if not (CACHE / "K001.npz").exists() and not any(RAW.glob("*.rar")):
    st.error("Paderborn data not found. Run `make paderborn` first.")
    st.stop()

profiles = damage_profiles()


def describe(code: str) -> str:
    if code in HEALTHY:
        return f"{code} · healthy, never seen in training"
    p = profiles.get(code, {})
    return f"{code} · damaged {p.get('component', '')} ({p.get('combination', '')})"


with st.sidebar:
    st.header("Monitor")
    fold_name = st.selectbox("Fold", list(FOLDS),
                             help="Which healthy bearings train, calibrate and are held out")
    fold = FOLDS[fold_name]
    st.caption(f"Train: {', '.join(fold['train'])} · Calibrate: {', '.join(fold['calib'])}")
    model_label = st.selectbox("Detector", list(MODELS))
    group_label = st.selectbox("Features", list(GROUPS))
    rate_label = st.selectbox("Sampling rate", list(RATES))
    st.header("Stream")
    bearing = st.selectbox("Bearing to replay", fold["test_healthy"] + list(REAL_DAMAGE),
                           format_func=describe)
    condition = st.selectbox("Operating condition", CONDITIONS, format_func=lambda c: CONDITION_LABELS[c])
    n_recs = st.slider("Recordings (4 s each)", 1, 20, 5)
    speed = st.slider("Replay speed (x real time)", 1, 20, 4)
    start = st.button("Start stream", type="primary")

model, group, fs = MODELS[model_label], GROUPS[group_label], RATES[rate_label]
mon = monitor(fold_name, model, group, fs)
idx = group_indices(group)
window, hop, chunk = pt.geometry(CFG, fs)

truth = st.container(border=True)
if bearing in HEALTHY:
    truth.markdown(f"**Ground truth (hidden from the detector):** `{bearing}` is **healthy** and was not "
                   f"used for training or calibration in fold {fold_name}. Any alert is a false alarm.")
else:
    p = profiles.get(bearing, {})
    truth.markdown(f"**Ground truth (hidden from the detector):** `{bearing}` has **real damage** from an "
                   f"accelerated lifetime test: component {p.get('component')}, combination "
                   f"{p.get('combination')}, extent {p.get('extent')}, {p.get('characteristic')}.")
st.caption(f"Calibrated threshold {mon.threshold:.4g} · alert after {mon.raise_after} consecutive windows "
           f"({mon.raise_after * CFG['stream']['hop_s']:.2f} s) · window {window} samples, hop {hop}, "
           f"chunk {chunk} · fitted on {mon.train_windows} healthy windows")

tiles = [c.empty() for c in st.columns(6)]
left, right = st.columns(2)
wave_box, score_box = left.empty(), right.empty()
lower = st.columns([3, 2])
table_box, perf_box = lower[0].empty(), lower[1].empty()

if start:
    display_features = BearingFeatureExtractor(fs, window)  # for the BPFO tile only
    env_i = FEATURE_NAMES.index("env_bpfo")
    results: list[dict] = []
    inf_ns: list[int] = []
    pipe_ns: list[int] = []
    lateness = 0.0
    recs = recordings(bearing, condition, n_recs)
    for r_i, (rec_index, vib, cur, rpm, nm, force) in enumerate(recs):
        signals = pt.recording_signals(Recording(bearing, condition, rec_index, vib, cur, rpm, nm, force), fs)
        alerts = AlertEngine(mon.threshold, mon.raise_after, CFG["alerts"]["clear_after"])
        eng = pt.make_engine(CFG, fs, mon.scorer, alerts, feature_idx=idx)
        eng.reset()
        t_hist: list[float] = []
        s_hist: list[float] = []
        wall0 = time.perf_counter()
        last_draw = 0.0
        for start_i in range(0, len(signals), chunk):
            block = signals[start_i:start_i + chunk]
            due = wall0 + (start_i + len(block)) / (fs * speed)
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                lateness = max(lateness, -delay)
            _, windows = eng.process_chunk(block)
            for w in windows:
                t_hist.append(w.t_end)
                s_hist.append(w.score / mon.threshold)
                inf_ns.append(w.inference_ns)
                pipe_ns.append(w.pipeline_ns)
            if not windows or time.perf_counter() - last_draw < 0.25:
                continue
            last_draw = time.perf_counter()
            end = start_i + len(block)
            seg = signals[end - window:end]
            state = alerts.state.value
            icon = {"OK": "🟢", "WARNING": "🟠", "ALARM": "🔴"}[state]
            tiles[0].metric("Recording", f"{r_i + 1} / {len(recs)}")
            tiles[1].metric("Vibration RMS", f"{np.std(seg[:, 0]):.3f}")
            tiles[2].metric("Envelope energy at BPFO", f"{display_features(seg)[env_i]:.2f}",
                            help="Share of envelope-spectrum power at 1-3 x the outer-race defect frequency")
            tiles[3].metric("Score / threshold", f"{s_hist[-1]:.2f}")
            tiles[4].metric("Health", f"{icon} {state}")
            tiles[5].metric("Recordings flagged", f"{sum(r['alert raised'] for r in results)} / {r_i}")
            wave_box.line_chart(pd.DataFrame({"vibration, last 50 ms": seg[-int(0.05 * fs):, 0]}), height=220)
            score_box.line_chart(pd.DataFrame({"t [s]": t_hist, "score / threshold": np.clip(s_hist, 0, 10),
                                               "threshold": 1.0}).set_index("t [s]"), height=220)
            lat_inf = np.array(inf_ns[-400:]) / 1e3
            lat_pipe = np.array(pipe_ns[-400:]) / 1e3
            perf_box.markdown(
                f"**Model inference** p50 `{np.median(lat_inf):.1f} µs`  \n"
                f"**Window pipeline** (features + model + alert) p50 `{np.median(lat_pipe):.0f} µs`, "
                f"p99 `{np.percentile(lat_pipe, 99):.0f} µs`  \n"
                f"Sensor rate `{fs / 1000:.0f} kHz`, replay `{speed}x` real time, "
                f"max lateness `{lateness * 1e3:.1f} ms`  \n"
                "Drawing shares the thread between chunks; stage timers exclude it.")
        alerts.close(len(signals) / fs)
        results.append({"recording": rec_index,
                        "max score / threshold": round(max(s_hist) if s_hist else 0.0, 2),
                        "alert raised": len(alerts.history) > 0})
        table_box.dataframe(pd.DataFrame(results), hide_index=True, width="stretch")
    flagged = sum(r["alert raised"] for r in results)
    kind = "false alarms" if bearing in HEALTHY else "detections"
    st.success(f"Finished: {flagged} of {len(results)} recordings flagged ({kind}).")
