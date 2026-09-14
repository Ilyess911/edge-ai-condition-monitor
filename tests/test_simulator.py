import numpy as np
import pytest

from src.data.simulator import FAULT_TYPES, FaultEvent, sample_fault_schedule, simulate


def test_same_seed_is_bit_identical():
    a = simulate(20, 1000, seed=3)
    b = simulate(20, 1000, seed=3)
    np.testing.assert_array_equal(a.signals, b.signals)


def test_labels_cover_exactly_the_event():
    ev = FaultEvent("imbalance", start_s=5.0, duration_s=3.0, severity=0.8)
    run = simulate(12, 1000, [ev], seed=0, artefacts=False)
    t = run.t
    assert run.labels[(t >= 5.0) & (t < 8.0)].all()
    assert not run.labels[(t < 5.0) | (t >= 8.0)].any()


def test_imbalance_raises_vibration_rms():
    base = simulate(20, 1000, seed=0, artefacts=False)
    ev = FaultEvent("imbalance", 0.0, 20.0, 1.0)
    faulty = simulate(20, 1000, [ev], seed=0, artefacts=False)
    assert np.std(faulty.signals[:, 0]) > 1.5 * np.std(base.signals[:, 0])


def test_artefacts_only_touch_slow_channels():
    run = simulate(300, 1000, seed=1, artefacts=True)
    assert not np.isnan(run.signals[:, 0]).any()
    assert np.isnan(run.signals[:, 1:]).any()


def test_rejects_rate_that_does_not_divide_base_rate():
    with pytest.raises(ValueError):
        simulate(1, 3000)


def test_schedule_is_non_overlapping_and_uses_every_type():
    events = sample_fault_schedule(3600, 5, np.random.default_rng(7), min_gap_s=240)
    assert sorted(e.kind for e in events) == sorted(FAULT_TYPES)
    for a, b in zip(events, events[1:]):
        assert b.start_s - a.end_s >= 240 - 1e-9
    assert events[-1].end_s <= 3600
