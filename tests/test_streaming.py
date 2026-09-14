import numpy as np

from src.benchmarking import simulated_track as st
from src.data.simulator import simulate
from src.features.simulated import extract_offline


def test_streaming_features_equal_offline_features(cfg):
    run = simulate(30, 1000, seed=5, artefacts=False)
    online = st.stream_features(cfg, run)
    fs, window, hop, _ = st.geometry(cfg)
    offline = extract_offline(run.signals, fs, window, hop)
    np.testing.assert_allclose(online, offline, rtol=1e-12, atol=1e-12)


def test_engine_report_counts_and_timings(cfg):
    run = simulate(20, 1000, seed=6)
    det = st._NullDetector()
    calib = st.Calibration(threshold=1.0, raise_after=3)
    eng = st.make_engine(cfg, det, calib)
    report = eng.run(run.signals, chunk_size=50)
    fs, window, hop, _ = st.geometry(cfg)
    assert report.samples == len(run.signals)
    assert report.windows == (len(run.signals) - window) // hop + 1
    assert (report.column("inference_ns") > 0).all()
    assert report.throughput_sps > 0


def test_paced_replay_keeps_up_with_sensor_clock(cfg):
    run = simulate(2, 1000, seed=7)
    eng = st.make_engine(cfg, st._NullDetector(), None)
    report = eng.run(run.signals, chunk_size=50, paced=True)
    assert report.wall_s >= 1.9  # really waited for the samples
    assert report.cpu_utilisation < 1.0
