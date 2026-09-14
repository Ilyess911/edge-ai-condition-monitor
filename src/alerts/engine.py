"""Local alert engine: turns a noisy per-window score into operator alerts.

A raw threshold on the score fires on single noisy windows. Maintenance teams
stop reading an alarm that flickers, so the alert logic adds:
- persistence: raise only after `raise_after` consecutive windows over threshold
- hysteresis:  clear only after `clear_after` consecutive windows under it
Both are counted in windows, so their duration in seconds is `count * hop`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Health(str, Enum):
    OK = "OK"
    WARNING = "WARNING"  # score over threshold, not yet persistent
    ALARM = "ALARM"


@dataclass
class Alert:
    start_t: float
    end_t: float | None = None
    peak_score: float = 0.0
    n_windows: int = 0


@dataclass
class AlertEngine:
    threshold: float
    raise_after: int = 3
    clear_after: int = 5
    state: Health = Health.OK
    history: list[Alert] = field(default_factory=list)
    _over: int = 0
    _under: int = 0

    def reset(self) -> None:
        self.state = Health.OK
        self.history.clear()
        self._over = self._under = 0

    @property
    def active(self) -> Alert | None:
        if self.history and self.history[-1].end_t is None:
            return self.history[-1]
        return None

    def update(self, t: float, score: float) -> Health:
        over = score > self.threshold
        alert = self.active
        if over:
            self._over += 1
            self._under = 0
        else:
            self._under += 1
            self._over = 0

        if alert is None:
            if self._over >= self.raise_after:
                self.history.append(Alert(start_t=t, peak_score=score, n_windows=self._over))
                self.state = Health.ALARM
            else:
                self.state = Health.WARNING if over else Health.OK
        else:
            alert.n_windows += 1
            alert.peak_score = max(alert.peak_score, score)
            if self._under >= self.clear_after:
                alert.end_t = t
                self.state = Health.OK
            else:
                self.state = Health.ALARM
        return self.state

    def close(self, t: float) -> None:
        if self.active is not None:
            self.active.end_t = t
