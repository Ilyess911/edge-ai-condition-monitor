"""Download MetroPT-3 from the UCI repository and verify its checksum.

    uv run python scripts/download_metropt.py

Source: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset (CC BY 4.0)
Veloso, Ribeiro, Pereira, Gama. The MetroPT dataset for predictive maintenance.
Scientific Data 9, 764 (2022). doi:10.1038/s41597-022-01877-3
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
SHA256 = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"  # verified 2026-09-14
RAW = ROOT / "data" / "raw"
ZIP = RAW / "metropt3.zip"
CSV = RAW / "MetroPT3(AirCompressor).csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        print(f"downloading {URL} (218 MB)")
        urllib.request.urlretrieve(URL, ZIP)
    digest = sha256(ZIP)
    if digest != SHA256:
        sys.exit(f"checksum mismatch: {digest}. The upstream archive changed; results may differ.")
    if not CSV.exists():
        with zipfile.ZipFile(ZIP) as zf:
            zf.extractall(RAW)
    print(f"ok: {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
