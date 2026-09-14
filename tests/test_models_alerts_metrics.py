from types import SimpleNamespace

import numpy as np
import pytest

from src.alerts.engine import AlertEngine, Health
from src.benchmarking.metrics import aftermath_mask, event_metrics, window_labels
from src.inference.threshold import calibrate_persistence, calibrate_threshold
from src.models.detectors import build


@pytest.mark.parametrize(
    "name,kwargs",
    [("zscore", {}), ("pca", {}), ("iforest", {"n_estimators": 20}), ("autoencoder", {})],
)
def test_detectors_rank_shifted_data_higher(rng, name, kwargs):
    cov = np.eye(6) + 0.8 * (np.ones((6, 6)) - np.eye(6))
    healthy = rng.multivariate_normal(np.zeros(6), cov, size=2000)
    test_ok = rng.multivariate_normal(np.zeros(6), cov, size=300)
    # Break the correlation structure without changing the marginals much.
    test_bad = rng.normal(size=(300, 6)) * 1.5
    det = build(name, **kwargs).fit(healthy)
    s_ok, s_bad = det.score(test_ok), det.score(test_bad)
    assert s_ok.shape == (300,)
    assert np.median(s_bad) > np.percentile(s_ok, 75)
    assert det.n_parameters() > 0 and det.serialized_bytes() > 0


def test_unknown_detector():
    with pytest.raises(KeyError):
        build("transformer")


def test_threshold_quantile_and_minimum_size():
    s = np.arange(1000, dtype=float)
    assert calibrate_threshold(s, 0.99) == pytest.approx(np.quantile(s, 0.99))
    with pytest.raises(ValueError):
        calibrate_threshold(np.arange(50.0), 0.995)


def test_persistence_is_longest_healthy_excursion_plus_one():
    s = np.array([0, 5, 5, 0, 5, 5, 5, 5, 0, 5], dtype=float)
    assert calibrate_persistence(s, 1.0, max_windows=20) == 5
    assert calibrate_persistence(s, 1.0, max_windows=3) == 3


def test_alert_engine_persistence_and_hysteresis():
    eng = AlertEngine(threshold=1.0, raise_after=3, clear_after=2)
    states = [eng.update(t, s) for t, s in enumerate([2, 2, 0, 2, 2, 2, 0, 2, 0, 0, 0])]
    assert states[:3] == [Health.WARNING, Health.WARNING, Health.OK]
    assert states[5] == Health.ALARM
    assert states[7] == Health.ALARM  # one window under does not clear
    assert states[9] == Health.OK
    assert len(eng.history) == 1 and eng.history[0].start_t == 5 and eng.history[0].end_t == 9


def test_window_labels_majority_rule():
    labels = np.array([0] * 10 + [1] * 10)
    assert window_labels(labels, np.array([10, 15, 16, 20]), 10).tolist() == [0, 1, 1, 1]
    assert window_labels(labels, np.array([14]), 10).tolist() == [0]


def test_event_metrics_hand_example():
    events = [SimpleNamespace(kind="a", start_s=100, end_s=200),
              SimpleNamespace(kind="b", start_s=500, end_s=600)]
    alerts = [SimpleNamespace(start_t=130, end_t=210),   # detects a, delay 30
              SimpleNamespace(start_t=250, end_t=260),   # aftermath of a: excused
              SimpleNamespace(start_t=900, end_t=910)]   # false alarm
    m = event_metrics(events, alerts, duration_s=3600, grace_s=5, recovery_s=100)
    assert (m.n_detected, m.n_false_alarms, m.n_alerts) == (1, 1, 2)
    assert m.mean_delay_s == 30
    assert m.alert_precision == 0.5 and m.event_recall == 0.5
    healthy_h = (3600 - 200 - 200) / 3600
    assert m.false_alarms_per_hour == pytest.approx(1 / healthy_h)


def test_aftermath_mask_only_marks_healthy_windows():
    ev = [SimpleNamespace(start_s=10, end_s=20)]
    t = np.array([15, 21, 25, 40])
    y = np.array([1, 1, 0, 0])
    assert aftermath_mask(t, y, ev, recovery_s=10).tolist() == [False, False, True, False]
