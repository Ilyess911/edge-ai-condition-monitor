"""Paderborn University (KAt) bearing data: real test-bench measurements.

Facts checked in the archives on 2026-09-15 (measuring logs and damage profiles):
- Deep groove ball bearings 6203 on a test bench driven by a 425 W motor.
- Vibration (PCB 336C04 accelerometer) and two phase currents at 64 kHz;
  speed, torque and radial force at 4 kHz; bearing-module temperature at 1 Hz.
- Four operating conditions, 20 recordings of about 4 s each:
      N15_M07_F10  1500 rpm, 0.7 Nm, 1000 N
      N09_M07_F10   900 rpm, 0.7 Nm, 1000 N
      N15_M01_F10  1500 rpm, 0.1 Nm, 1000 N
      N15_M07_F04  1500 rpm, 0.7 Nm,  400 N
- Recordings are NOT continuous: a stream here is recordings replayed one after
  another, and no window ever spans two recordings.

Each archive is converted once into a compact float32 cache (vibration, one
phase current, mean speed/torque/force per recording). The .mat files are read
straight from the RAR archive and never written to disk.
"""

from __future__ import annotations

import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from src.config import ROOT

RAW = ROOT / "data" / "raw" / "paderborn"
HEALTHY = ("K001", "K002", "K003", "K004", "K005", "K006")
# Real damage from accelerated lifetime tests (per the archive damage profiles).
REAL_DAMAGE = ("KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24", "KB27",
               "KI04", "KI14", "KI16", "KI17", "KI18", "KI21")
CACHE = ROOT / "data" / "cache" / "paderborn"
FS = 64_000
CONDITIONS = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
NAME = re.compile(r"(N\d\d_M\d\d_F\d\d)_(K[A-Z]?\d\d\d?|K[AIB]\d\d)_(\d+)\.mat$")

# Bearing 6203 geometry from the damage profiles (pitch diameter 28.55 mm,
# 8 balls of 6.75 mm, contact angle 0). K001's profile lists 29.05 mm; the
# difference moves the defect frequencies by under 2 %.
N_BALLS = 8
BALL_D = 6.75
PITCH_D = 28.55


def defect_orders() -> dict[str, float]:
    """Characteristic defect frequencies as multiples of shaft speed."""
    r = BALL_D / PITCH_D
    return {
        "bpfo": N_BALLS / 2 * (1 - r),  # outer race
        "bpfi": N_BALLS / 2 * (1 + r),  # inner race
    }


@dataclass
class Recording:
    bearing: str
    condition: str
    index: int
    vibration: np.ndarray  # float32, 64 kHz
    current: np.ndarray  # float32, 64 kHz, phase current 1
    speed_rpm: float
    torque_nm: float
    force_n: float


def _channels(mat_bytes: bytes) -> dict[str, np.ndarray]:
    m = sio.loadmat(io.BytesIO(mat_bytes), squeeze_me=True, struct_as_record=False)
    s = m[[k for k in m if not k.startswith("__")][0]]
    return {c.Name: np.asarray(c.Data, dtype=np.float64).ravel() for c in np.atleast_1d(s.Y)}


def convert_archive(code: str) -> Path:
    """RAR archive -> data/cache/paderborn/<code>.npz (idempotent)."""
    out = CACHE / f"{code}.npz"
    if out.exists():
        return out
    archive = RAW / f"{code}.rar"
    if not archive.exists():
        raise FileNotFoundError(f"{archive} missing: run scripts/download_paderborn.py")
    names = subprocess.run(["bsdtar", "-tf", str(archive)], check=True, capture_output=True,
                           text=True).stdout.split()
    mats = sorted(n for n in names if n.endswith(".mat"))
    vib, cur, meta = [], [], []
    for name in mats:
        match = NAME.search(name)
        if not match:
            continue
        blob = subprocess.run(["bsdtar", "-xOf", str(archive), name], check=True,
                              capture_output=True).stdout
        ch = _channels(blob)
        vib.append(ch["vibration_1"].astype(np.float32))
        cur.append(ch["phase_current_1"].astype(np.float32))
        meta.append((match.group(1), int(match.group(3)), ch["speed"].mean(),
                     ch["torque"].mean(), ch["force"].mean()))
    lengths = np.array([len(v) for v in vib])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, vibration=np.concatenate(vib), current=np.concatenate(cur), lengths=lengths,
             condition=np.array([m[0] for m in meta]), index=np.array([m[1] for m in meta]),
             speed=np.array([m[2] for m in meta]), torque=np.array([m[3] for m in meta]),
             force=np.array([m[4] for m in meta]))
    return out


def load_bearing(code: str) -> list[Recording]:
    z = np.load(convert_archive(code))
    ends = np.cumsum(z["lengths"])
    starts = ends - z["lengths"]
    recs = [
        Recording(code, str(z["condition"][i]), int(z["index"][i]),
                  z["vibration"][starts[i]:ends[i]], z["current"][starts[i]:ends[i]],
                  float(z["speed"][i]), float(z["torque"][i]), float(z["force"][i]))
        for i in range(len(ends))
    ]
    return sorted(recs, key=lambda r: (CONDITIONS.index(r.condition), r.index))


def parse_damage_profile(text: str) -> dict:
    """Fields of interest from the text of a KAt damage-profile PDF."""

    def grab(pattern: str) -> str | None:
        m = re.search(pattern, text, flags=re.S)
        return " ".join(m.group(1).split()) if m else None

    return {
        "n_damages": grab(r"Number of damages\s+(\d+)"),
        "component": grab(r"Component\s+([A-Z ]+?)\s*\n"),
        "mode": grab(r"\nMode\s+([^\n]+)"),
        "combination": grab(r"Damage combination\s+([A-Z ]+?)\s*\n"),
        "extent": grab(r"Extent of damage\s+((?:\d+|n/a)(?: +(?:\d+|n/a))*)\s*\n"),
        "characteristic": grab(r"Characteristic of\s*\ndamage\s+([^\n]+)"),
        "method": grab(r"Damage method\s+([^\n]+)"),
    }


def damage_profile(code: str) -> dict:
    """Damage description parsed from the profile PDF shipped inside the archive."""
    from pypdf import PdfReader

    archive = RAW / f"{code}.rar"
    blob = subprocess.run(["bsdtar", "-xOf", str(archive), f"{code}/{code}.pdf"], check=True,
                          capture_output=True).stdout
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(blob)).pages)
    return {"bearing": code, **parse_damage_profile(text)}
