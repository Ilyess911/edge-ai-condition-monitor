"""TOML configuration loading (tomllib is in the standard library, no dependency)."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("rb") as fh:
        return tomllib.load(fh)


def detector_kwargs(model_cfg: dict) -> tuple[str, dict]:
    kwargs = {k: v for k, v in model_cfg.items() if k != "detector"}
    if "hidden" in kwargs:
        kwargs["hidden"] = tuple(kwargs["hidden"])
    return model_cfg["detector"], kwargs
