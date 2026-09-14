"""Detection metrics at two levels.

Window level: every decision (one per hop) is a sample. Threshold-free metrics
(ROC-AUC, PR-AUC) measure how well the score ranks faulty above healthy
windows; precision/recall/F1 use the calibrated threshold on the raw score.

Event level: what a maintenance team experiences. A fault counts as detected
if an alert overlaps it; an alert that overlaps no fault is a false alarm.
Detection delay is measured from fault onset to the alert.

Aftermath: a fault's physical effects outlive its cause (a hot bearing takes
minutes to cool). For `recovery_s` after each fault, windows are excluded from
window metrics and alerts are counted neither as detections nor as false
alarms. That time is also removed from the healthy-hours denominator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def window_labels(labels: np.ndarray, end_idx: np.ndarray, window: int) -> np.ndarray:
    """A window is faulty if at least half of its samples are inside a fault."""
    csum = np.concatenate([[0], np.cumsum(labels, dtype=np.int64)])
    frac = (csum[end_idx] - csum[end_idx - window]) / window
    return (frac >= 0.5).astype(np.uint8)


def aftermath_mask(t_end: np.ndarray, y: np.ndarray, events, recovery_s: float) -> np.ndarray:
    """True for healthy-labelled windows ending within recovery_s after a fault."""
    mask = np.zeros(len(t_end), dtype=bool)
    for ev in events:
        rec = getattr(ev, "recovery_s", recovery_s)
        mask |= (t_end > ev.end_s) & (t_end <= ev.end_s + rec)
    return mask & (y == 0)


@dataclass
class WindowMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    positives: int
    negatives: int


def window_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> WindowMetrics:
    pred = scores > threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return WindowMetrics(
        roc_auc=float(roc_auc_score(y, scores)),
        pr_auc=float(average_precision_score(y, scores)),
        precision=precision,
        recall=recall,
        f1=f1,
        positives=int(y.sum()),
        negatives=int(len(y) - y.sum()),
    )


@dataclass
class EventMetrics:
    n_events: int
    n_detected: int
    n_alerts: int
    n_false_alarms: int
    event_recall: float
    alert_precision: float
    event_f1: float
    false_alarms_per_hour: float
    mean_delay_s: float  # over detected events only
    detected_by_kind: dict

    def to_dict(self) -> dict:
        return asdict(self)


def event_metrics(
    events, alerts, duration_s: float, grace_s: float, recovery_s: float = 0.0,
    lead_s: float = 0.0,
) -> EventMetrics:
    """events: objects with kind/start_s/end_s (and optionally their own recovery_s).
    alerts: objects with start_t/end_t.

    lead_s > 0 also credits an alert raised up to lead_s BEFORE the reported
    start (early warning). The simulated track uses 0: its onset is exact.
    """
    spans = [(a.start_t, a.end_t if a.end_t is not None else duration_s) for a in alerts]
    matched_alert = np.zeros(len(spans), dtype=bool)
    excused = np.zeros(len(spans), dtype=bool)  # starts inside an aftermath
    for ev in events:
        rec = getattr(ev, "recovery_s", recovery_s)
        for i, (a, _) in enumerate(spans):
            if ev.end_s < a <= ev.end_s + rec:
                excused[i] = True
    delays = []
    by_kind: dict[str, list[int]] = {}
    for ev in events:
        lo, hi = ev.start_s - lead_s, ev.end_s + grace_s
        hit = [i for i, (a, b) in enumerate(spans) if a <= hi and b >= lo]
        by_kind.setdefault(ev.kind, [0, 0])
        by_kind[ev.kind][1] += 1
        if hit:
            matched_alert[hit] = True
            by_kind[ev.kind][0] += 1
            delays.append(max(0.0, min(spans[i][0] for i in hit) - ev.start_s))
    n_events = len(events)
    n_detected = len(delays)
    excused &= ~matched_alert
    n_false = int((~matched_alert & ~excused).sum())
    counted = matched_alert | ~excused
    busy = sum(min(ev.end_s + getattr(ev, "recovery_s", recovery_s), duration_s)
               - max(ev.start_s - lead_s, 0.0) for ev in events)
    healthy_h = max(duration_s - busy, 1e-9) / 3600
    recall = n_detected / n_events if n_events else float("nan")
    precision = float(matched_alert[counted].mean()) if counted.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventMetrics(
        n_events=n_events,
        n_detected=n_detected,
        n_alerts=int(counted.sum()),
        n_false_alarms=n_false,
        event_recall=recall,
        alert_precision=precision,
        event_f1=f1,
        false_alarms_per_hour=n_false / healthy_h,
        mean_delay_s=float(np.mean(delays)) if delays else float("nan"),
        detected_by_kind={k: {"detected": v[0], "total": v[1]} for k, v in by_kind.items()},
    )
