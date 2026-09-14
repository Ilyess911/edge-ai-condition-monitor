"""Threshold calibration without fault labels.

The threshold is the q-quantile of anomaly scores on a HELD-OUT healthy run,
never on the training run (scores on training data are optimistically low)
and never on the test run (that would use the labels we evaluate against).

Choosing q fixes the expected rate of windows over threshold during healthy
operation to (1 - q). With hop = 0.5 s and q = 0.995 that is about 36 windows
per hour; the alert engine's persistence rule then filters most of them.
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(healthy_scores: np.ndarray, quantile: float) -> float:
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1)")
    s = np.asarray(healthy_scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < int(np.ceil(1.0 / (1.0 - quantile))):
        raise ValueError(
            f"{len(s)} healthy windows cannot resolve a {quantile} quantile; "
            "collect a longer calibration run"
        )
    return float(np.quantile(s, quantile))


def longest_run_over(scores: np.ndarray, threshold: float) -> int:
    over = np.asarray(scores) > threshold
    best = cur = 0
    for o in over:
        cur = cur + 1 if o else 0
        best = max(best, cur)
    return best


def calibrate_persistence(healthy_scores: np.ndarray, threshold: float, max_windows: int) -> int:
    """Smallest alert persistence that raises no alert on the healthy calibration run.

    Normal transients (a load step, a speed ramp) push the score over threshold
    for a few consecutive windows. Requiring one more consecutive window than
    the longest healthy excursion suppresses them without looking at a single
    fault label. The price is detection delay: raise_after * hop seconds.
    Capped at max_windows so a pathological calibration run cannot make the
    monitor arbitrarily slow.
    """
    return int(min(longest_run_over(healthy_scores, threshold) + 1, max_windows))
