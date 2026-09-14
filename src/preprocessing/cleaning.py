"""Stateful sample cleaning, applied chunk by chunk as data arrives.

Two edge-realistic problems are handled, nothing more:
1. Physically implausible values (sensor glitch, bus corruption) are rejected
   with a per-channel plausibility range.
2. Missing values (dropout, rejected sample) are replaced by the last valid
   value of that channel (sample-and-hold), up to `max_hold` samples. Beyond
   that the value stays NaN and the feature stage decides what to do.

State is carried across chunks, so cleaning a stream in chunks of any size
gives exactly the same output as cleaning it in one piece (tested).
"""

from __future__ import annotations

import numpy as np


class SampleCleaner:
    def __init__(self, low: np.ndarray, high: np.ndarray, max_hold: int | None = None):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.max_hold = max_hold
        n_ch = len(self.low)
        self._last = np.full(n_ch, np.nan)
        self._age = np.zeros(n_ch, dtype=np.int64)  # samples since last valid value
        self.n_rejected = 0
        self.n_filled = 0

    def reset(self) -> None:
        self._last[:] = np.nan
        self._age[:] = 0
        self.n_rejected = 0
        self.n_filled = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = np.array(chunk, dtype=float, copy=True)
        if x.ndim == 1:
            x = x[None, :]
        out_of_range = (x < self.low) | (x > self.high)
        self.n_rejected += int(out_of_range.sum())
        x[out_of_range] = np.nan

        n = len(x)
        rows = np.arange(n)
        for ch in range(x.shape[1]):
            col = x[:, ch]
            valid = ~np.isnan(col)
            if valid.all():
                self._last[ch] = col[-1]
                self._age[ch] = 0
                continue
            # Index of the most recent valid sample at or before each row (-1: none in chunk).
            last_idx = np.maximum.accumulate(np.where(valid, rows, -1))
            from_chunk = last_idx >= 0
            fill = np.where(from_chunk, col[np.maximum(last_idx, 0)], self._last[ch])
            age = np.where(from_chunk, rows - last_idx, rows + 1 + self._age[ch])
            missing = ~valid
            if self.max_hold is not None:
                missing &= age <= self.max_hold
            self.n_filled += int((missing & ~np.isnan(fill)).sum())
            col[missing] = fill[missing]
            if valid.any():
                self._last[ch] = col[np.flatnonzero(valid)[-1]]
                self._age[ch] = n - 1 - np.flatnonzero(valid)[-1]
            else:
                self._age[ch] += n
        return x
