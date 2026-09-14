import numpy as np

from src.preprocessing.cleaning import SampleCleaner
from src.preprocessing.windowing import SlidingWindow


def _cleaner(max_hold=None):
    return SampleCleaner(low=[-10, -10], high=[10, 10], max_hold=max_hold)


def test_out_of_range_is_replaced_by_last_valid_value():
    x = np.array([[1.0, 2.0], [99.0, 3.0], [np.nan, 4.0], [5.0, 5.0]])
    out = _cleaner().process(x)
    np.testing.assert_array_equal(out[:, 0], [1.0, 1.0, 1.0, 5.0])
    np.testing.assert_array_equal(out[:, 1], [2.0, 3.0, 4.0, 5.0])


def test_hold_carries_across_chunks_and_respects_max_hold():
    c = _cleaner(max_hold=2)
    c.process(np.array([[1.0, 1.0]]))
    out = c.process(np.array([[np.nan, 1.0], [np.nan, 1.0], [np.nan, 1.0]]))
    assert out[0, 0] == 1.0 and out[1, 0] == 1.0
    assert np.isnan(out[2, 0])  # third missing sample exceeds max_hold


def test_cleaning_is_independent_of_chunk_size(rng):
    x = rng.normal(size=(1000, 2)) * 6
    x[rng.random((1000, 2)) < 0.05] = np.nan
    full = _cleaner(max_hold=3).process(x)
    c = _cleaner(max_hold=3)
    pieces, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 40))
        pieces.append(c.process(x[i : i + n]))
        i += n
    np.testing.assert_array_equal(np.vstack(pieces), full)


def test_window_emission_matches_offline_slicing(rng):
    x = rng.normal(size=(1037, 3))
    win, hop = 100, 25
    sw = SlidingWindow(win, hop, 3)
    got, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 60))
        got.extend(sw.push(x[i : i + n]))
        i += n
    expected_ends = list(range(win, len(x) + 1, hop))
    assert [e for e, _ in got] == expected_ends
    for end, w in got:
        np.testing.assert_array_equal(w, x[end - win : end])


def test_chunk_larger_than_window(rng):
    x = rng.normal(size=(500, 1))
    got = SlidingWindow(50, 50, 1).push(x)
    assert len(got) == 10
    np.testing.assert_array_equal(got[-1][1], x[450:500])
