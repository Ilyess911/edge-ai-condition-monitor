"""Sliding windows over a stream, backed by a fixed-size circular buffer.

Memory is O(window) whatever the stream length, and nothing is allocated per
sample: the property that matters on a microcontroller-class device. A window
is emitted every `hop` samples once `window` samples have been seen.
"""

from __future__ import annotations

import numpy as np


class SlidingWindow:
    def __init__(self, window: int, hop: int, n_channels: int):
        if not 0 < hop <= window:
            raise ValueError("need 0 < hop <= window")
        self.window = window
        self.hop = hop
        self._buf = np.zeros((window, n_channels))
        self._pos = 0  # next write index
        self.seen = 0  # total samples ingested

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pos = 0
        self.seen = 0

    def _write(self, block: np.ndarray) -> None:
        n = len(block)
        if n >= self.window:
            self._buf[:] = block[-self.window :]
            self._pos = 0
        else:
            first = min(n, self.window - self._pos)
            self._buf[self._pos : self._pos + first] = block[:first]
            self._buf[: n - first] = block[first:]
            self._pos = (self._pos + n) % self.window
        self.seen += n

    def _snapshot(self) -> np.ndarray:
        return np.concatenate([self._buf[self._pos :], self._buf[: self._pos]])

    def push(self, chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Ingest a chunk; return (index of last sample + 1, window) for each emission."""
        out = []
        i = 0
        n = len(chunk)
        while i < n:
            if self.seen < self.window:
                to_next = self.window - self.seen
            else:
                to_next = self.hop - (self.seen - self.window) % self.hop
            take = min(to_next, n - i)
            self._write(chunk[i : i + take])
            i += take
            if take == to_next:
                out.append((self.seen, self._snapshot()))
        return out
