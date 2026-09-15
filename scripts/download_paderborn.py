"""Download the Paderborn University (KAt) bearing archives used here.

    uv run python scripts/download_paderborn.py            # download + verify
    uv run python scripts/download_paderborn.py --record   # (re)write checksums

Source: Paderborn University, Chair of Design and Drive Technology (KAt),
Bearing DataCenter, https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter
Lessmeier, Kimotho, Zimmer, Sextro. Condition Monitoring of Bearing Damage in
Electromechanical Drive Systems by Using Motor Current Signals of Electric
Motors: A Benchmark Data Set for Data-Driven Classification. PHM Society
European Conference, 2016.

Archives are fetched from the Zenodo mirror (record 15845309), which serves the
official files faster; each archive is checked against the mirror's MD5 and was
verified byte-identical to the university copy for K001, K002, K004, K006, KA04
and KA16 on 2026-09-15. The university server is the fallback.

Only healthy bearings and bearings with REAL damage from accelerated lifetime
tests are downloaded. Artificially damaged bearings (drilled, EDM, engraved) are
left out on purpose: they are not what an edge monitor meets in service.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
ZENODO = "https://zenodo.org/api/records/15845309"
RAW = ROOT / "data" / "raw" / "paderborn"
CHECKSUMS = ROOT / "configs" / "paderborn_sha256.txt"

sys.path.insert(0, str(ROOT))
from src.data.paderborn import HEALTHY, REAL_DAMAGE  # noqa: E402

BEARINGS = list(HEALTHY + REAL_DAMAGE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def zenodo_md5() -> dict[str, str]:
    import json
    import urllib.request

    with urllib.request.urlopen(ZENODO, timeout=60) as resp:
        record = json.load(resp)
    return {f["key"]: f["checksum"].removeprefix("md5:") for f in record["files"]}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(code: str, expected_md5: str | None) -> Path:
    """curl with resume; the university server throttles long transfers."""
    target = RAW / f"{code}.rar"
    if target.exists():
        return target
    part = target.with_suffix(".part")
    for url in (f"{ZENODO}/files/{code}.rar/content", BASE + f"{code}.rar"):
        done = subprocess.run(["curl", "-s", "-f", "-L", "--retry", "5", "-C", "-", "-o", str(part),
                               url]).returncode == 0
        if done and (expected_md5 is None or md5(part) == expected_md5):
            part.rename(target)
            print(f"downloaded {code}.rar from {url.split('/')[2]}", flush=True)
            return target
        part.unlink(missing_ok=True)
    raise RuntimeError(f"could not download a verified copy of {code}.rar")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write checksums of downloaded files")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    known = {}
    if CHECKSUMS.exists():
        for line in CHECKSUMS.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                digest, name = line.split()
                known[name] = digest

    expected = zenodo_md5()
    with ThreadPoolExecutor(args.jobs) as pool:
        paths = list(pool.map(lambda c: fetch(c, expected.get(f"{c}.rar")), BEARINGS))

    digests = {}
    for target in paths:
        digests[target.name] = sha256(target)
        if not args.record and target.name in known and known[target.name] != digests[target.name]:
            sys.exit(f"checksum mismatch for {target.name}")
        print(f"ok {target.name} {digests[target.name][:12]}", flush=True)

    if args.record:
        lines = ["# SHA-256 of the Paderborn archives as downloaded on 2026-09-15"]
        lines += [f"{d} {n}" for n, d in digests.items()]
        CHECKSUMS.write_text("\n".join(lines) + "\n")
        print(f"wrote {CHECKSUMS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
