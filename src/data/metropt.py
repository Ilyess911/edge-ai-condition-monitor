"""MetroPT-3: real air-compressor data from a metro train (Porto, 2020).

Facts checked against the files on 2026-09-14:
- 1,516,948 rows, 15 signals, 2020-02-01 to 2020-09-01.
- The UCI description says both "logged at 1Hz" and "collected at 0.1Hz". The
  CSV settles it: the median interval is 10 s (0.1 Hz), with 331 gaps longer
  than 60 s totalling about 910 hours.
- The data is unlabelled. The company's failure reports (below) are the only
  ground truth, transcribed from "Data Description_Metro.pdf" in the archive.

Transcription notes, kept visible because they affect evaluation:
- The PDF numbers the reports #1, #1, #3, #4. They are renumbered 1-4 here.
- Report 2 (29-30 May) says "Maintenance on 30Apr at 12:00", which predates
  the failure. It is read as 30 May 12:00. Assumption, not a fact.
- Timestamps are taken as local logger time, the same clock as the CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT

CSV = ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
CACHE = ROOT / "data" / "cache" / "metropt3_grid.npz"

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
           "Caudal_impulses"]
CHANNELS = ANALOG + DIGITAL
PERIOD_S = 10.0


@dataclass(frozen=True)
class FailureReport:
    number: int
    kind: str
    start: str
    end: str
    maintenance: str | None


FAILURES = (
    FailureReport(1, "air_leak", "2020-04-18 00:00", "2020-04-18 23:59", None),
    FailureReport(2, "air_leak", "2020-05-29 23:30", "2020-05-30 06:00", "2020-05-30 12:00"),
    FailureReport(3, "air_leak", "2020-06-05 10:00", "2020-06-07 14:30", "2020-06-08 16:00"),
    FailureReport(4, "air_leak", "2020-07-15 14:30", "2020-07-15 19:00", "2020-07-16 00:00"),
)


@dataclass
class ReportEvent:
    """Failure report in stream seconds, shaped like simulator.FaultEvent."""

    kind: str
    start_s: float
    end_s: float
    recovery_s: float  # until the reported maintenance: state in between is unknown


@dataclass
class GridData:
    t0: pd.Timestamp
    signals: np.ndarray  # (n, 15) on a 10 s grid, NaN where no sample arrived
    fs: float = 1.0 / PERIOD_S

    def index_of(self, when: str) -> int:
        return int((pd.Timestamp(when) - self.t0).total_seconds() // PERIOD_S)

    def seconds(self, when: str) -> float:
        return (pd.Timestamp(when) - self.t0).total_seconds()


def align_to_grid(ts: pd.Series, values: np.ndarray, period_s: float) -> tuple[pd.Timestamp, np.ndarray]:
    """Place irregular samples on a fixed grid, nearest-earlier slot, NaN if none.

    A gateway does the same when it hands a fixed-rate stream to a model: the
    logger interval is mostly 9 to 13 s, the model expects a sample every 10 s.
    """
    t0 = ts.iloc[0].floor(f"{int(period_s)}s")
    slot = ((ts - t0).dt.total_seconds() // period_s).to_numpy().astype(np.int64)
    n = int(slot[-1]) + 1
    grid = np.full((n, values.shape[1]), np.nan)
    grid[slot] = values  # duplicates in a slot: the later sample wins
    return t0, grid


def load_grid(csv: Path = CSV, use_cache: bool = True) -> GridData:
    if use_cache and CACHE.exists():
        z = np.load(CACHE)
        return GridData(pd.Timestamp(str(z["t0"])), z["signals"])
    if not csv.exists():
        raise FileNotFoundError(f"{csv} missing: run scripts/download_metropt.py")
    df = pd.read_csv(csv, usecols=["timestamp", *CHANNELS], parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    t0, grid = align_to_grid(df["timestamp"], df[CHANNELS].to_numpy(float), PERIOD_S)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, t0=str(t0), signals=grid)
    return GridData(t0, grid)


def report_events(grid: GridData, offset_s: float = 0.0) -> list[ReportEvent]:
    events = []
    for f in FAILURES:
        start, end = grid.seconds(f.start) - offset_s, grid.seconds(f.end) - offset_s
        maint = grid.seconds(f.maintenance) - offset_s if f.maintenance else end
        events.append(ReportEvent(f.kind, start, end, max(0.0, maint - end)))
    return events
