import numpy as np

from src.alerts.engine import AlertEngine
from src.benchmarking import paderborn_track as pt
from src.config import load_config
from src.data.paderborn import Recording
from src.models.detectors import build


def _recording(seed, damaged=False, fs=64_000, seconds=2.0):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    vib = rng.standard_normal(n).astype(np.float32) * (2.0 if damaged else 1.0)
    cur = np.sin(2 * np.pi * 50 * np.arange(n) / fs).astype(np.float32)
    return Recording("KX01" if damaged else "K0X1", "N15_M07_F10", seed, vib, cur, 1500.0, 0.7, 1000.0)


def test_replayed_alerts_match_the_full_streaming_engine():
    cfg = load_config("configs/paderborn.toml")
    fs = 16_000
    train = pt.extract_bearing(cfg, [_recording(i, fs=64_000) for i in range(3)], fs)
    det = build("pca").fit(train.stacked)
    test_recs = [_recording(10), _recording(11, damaged=True)]
    bf = pt.extract_bearing(cfg, test_recs, fs)
    scores, _ = pt.score_recordings(det, bf)
    threshold = float(np.quantile(np.concatenate(scores[:1]), 0.5))
    _, hop, chunk = pt.geometry(cfg, fs)
    for rec, sc in zip(test_recs, scores):
        alerts = AlertEngine(threshold, 2, 4)
        eng = pt.make_engine(cfg, fs, det, alerts)
        report = eng.run(pt.recording_signals(rec, fs), chunk)
        np.testing.assert_allclose(report.column("score"), sc, rtol=1e-12)
        replay = pt.replay_alerts(sc, hop / fs, threshold, 2, 4)
        assert replay == (len(alerts.history) > 0)


def test_fold_splits_never_reuse_a_bearing():
    cfg = load_config("configs/paderborn.toml")
    for fold in cfg["folds"]:
        roles = fold["train"] + fold["calib"] + fold["test_healthy"]
        assert len(roles) == len(set(roles)) == 6
