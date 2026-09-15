"""Paderborn benchmark: real bearings, three folds, sampling-rate sweep.

    uv run python scripts/download_paderborn.py
    uv run python scripts/run_paderborn.py

Writes results/paderborn_benchmark.json. Feature matrices are cached per rate
in data/cache/paderborn_features/ (git-ignored).
"""

from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy)

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts.engine import AlertEngine  # noqa: E402
from src.benchmarking import paderborn_track as pt  # noqa: E402
from src.benchmarking.profiling import environment, latency_summary  # noqa: E402
from src.config import ROOT, detector_kwargs, load_config  # noqa: E402
from src.data.paderborn import HEALTHY, REAL_DAMAGE, damage_profile, load_bearing  # noqa: E402
from src.features.bearing import FEATURE_GROUPS, group_indices  # noqa: E402
from src.inference.packed_forest import PackedIsolationForestDetector  # noqa: E402
from src.models.detectors import build  # noqa: E402

BEARINGS = HEALTHY + REAL_DAMAGE


def features_at_rate(cfg: dict, fs: float) -> dict[str, pt.BearingFeatures]:
    return pt.load_features(cfg, fs, BEARINGS, verbose=True)


def run_rate(cfg: dict, fs: float, feats: dict[str, pt.BearingFeatures], timing: bool,
             models: dict | None = None) -> dict:
    _, hop, _ = pt.geometry(cfg, fs)
    hop_s = hop / fs
    a = cfg["alerts"]
    result = {"folds": {}, "cost": {}}
    all_feature_ns = np.concatenate([np.concatenate(bf.feature_ns) for bf in feats.values()])
    result["cost"]["feature_latency"] = latency_summary(all_feature_ns, warmup=0)
    result["cost"]["ingest_ns_per_sample_mean"] = float(np.mean(
        [x for bf in feats.values() for x in bf.ingest_ns_per_sample]))
    result["cost"]["extraction_throughput_sps_mean"] = float(np.mean(
        [x for bf in feats.values() for x in bf.throughput_sps]))
    inference = {}
    for fold in cfg["folds"]:
        fold_res = {}
        for name, mcfg in (models or cfg["models"]).items():
            mon = pt.fit_fold(cfg, feats, fold, mcfg)
            scorer, thr, k = mon.scorer, mon.threshold, mon.raise_after
            test = {}
            lat = []
            for c in fold["test_healthy"] + list(REAL_DAMAGE):
                s, ns = pt.score_recordings(scorer, feats[c])
                test[c] = (s, c in REAL_DAMAGE)
                lat.append(ns)
            ev = pt.evaluate_fold(test, thr, k, a["clear_after"], hop_s)
            ev.update({"threshold": thr, "raise_after": k, "fit_s": mon.fit_s,
                       "train_windows": mon.train_windows})
            fold_res[name] = ev
            inference.setdefault(name, []).append(np.concatenate(lat))
            print(f"  fold {fold['name']} {name:<12} k={k} rec-detect {ev['recording_detection_rate']:.3f} "
                  f"rec-false {ev['recording_false_alarm_rate']:.3f} ROC-AUC {ev['window_roc_auc']:.3f}",
                  flush=True)
        result["folds"][fold["name"]] = fold_res
    if timing:
        result["cost"]["inference_latency"] = {
            ("iforest-packed" if n == "iforest" else n): latency_summary(np.concatenate(v), warmup=0)
            for n, v in inference.items()}
    return result


def pipeline_timing(cfg: dict, fs: float, paced_seconds: float) -> dict:
    """Full engine (features + model + alerts) on real recordings, unpaced and paced."""
    fold = cfg["folds"][0]
    feats = features_at_rate(cfg, fs)
    X_train = np.vstack([feats[c].stacked for c in fold["train"]])
    recs = load_bearing("K001")[:20] + load_bearing("KA04")[:20]
    _, _, chunk = pt.geometry(cfg, fs)
    out = {}
    for name, mcfg in cfg["models"].items():
        kind, kwargs = detector_kwargs(mcfg)
        det = build(kind, **kwargs).fit(X_train)
        variants = {name: det}
        if kind == "iforest":
            variants["iforest-packed"] = PackedIsolationForestDetector(det)
        for vname, v in variants.items():
            pipe, inf, tput = [], [], []
            for rec in recs:
                eng = pt.make_engine(cfg, fs, v, AlertEngine(1e9, 3, 4))
                rep = eng.run(pt.recording_signals(rec, fs), chunk)
                pipe.append(rep.column("pipeline_ns")[2:])
                inf.append(rep.column("inference_ns")[2:])
                tput.append(rep.throughput_sps)
            # Paced: replay recordings at the 64 kHz sensor clock for `paced_seconds`.
            played = 0.0
            cpu0, wall0 = time.process_time(), time.perf_counter()
            max_late = 0.0
            for rec in recs:
                if played >= paced_seconds:
                    break
                eng = pt.make_engine(cfg, fs, v, AlertEngine(1e9, 3, 4))
                rep = eng.run(pt.recording_signals(rec, fs), chunk, paced=True)
                played += rep.samples / fs
                max_late = max(max_late, rep.max_lateness_s)
            cpu_util = (time.process_time() - cpu0) / (time.perf_counter() - wall0)
            out[vname] = {
                "pipeline_latency": latency_summary(np.concatenate(pipe), warmup=0),
                "inference_latency": latency_summary(np.concatenate(inf), warmup=0),
                "throughput_sps_mean": float(np.mean(tput)),
                "realtime_factor_mean": float(np.mean(tput) / fs),
                "paced_seconds": played, "paced_cpu_utilisation": cpu_util,
                "paced_max_lateness_ms": max_late * 1e3,
                "serialized_kib": v.serialized_bytes() / 1024,
                "n_parameters": int(v.n_parameters()),
            }
            r = out[vname]
            print(f"  timing {vname:<15} pipeline p50 {r['pipeline_latency']['p50_us']:.0f} us "
                  f"inference p50 {r['inference_latency']['p50_us']:.1f} us RTF {r['realtime_factor_mean']:.1f}x "
                  f"paced CPU {100 * cpu_util:.1f} %", flush=True)
    return out


def main() -> None:
    cfg = load_config("configs/paderborn.toml")
    env_start = environment()
    t_start = time.time()
    profiles = {c: damage_profile(c) for c in REAL_DAMAGE}
    rates = cfg["sweep"]["rates_hz"]
    results = {}
    for fs in rates:
        print(f"rate {fs} Hz", flush=True)
        feats = features_at_rate(cfg, fs)
        results[str(fs)] = run_rate(cfg, fs, feats, timing=(fs == cfg["stream"]["fs"]))
    ablation = {}
    feats64 = features_at_rate(cfg, cfg["stream"]["fs"])
    for group, names in FEATURE_GROUPS.items():
        if names is None:
            continue
        print(f"ablation {group}", flush=True)
        ablation[group] = {"features": names,
                           **run_rate(cfg, cfg["stream"]["fs"], pt.subset(feats64, group_indices(group)), timing=False,
                                      models={k: cfg["models"][k] for k in ("pca", "iforest")})}
    print("full-engine timing at 64 kHz", flush=True)
    timing = pipeline_timing(cfg, cfg["stream"]["fs"], paced_seconds=60.0)
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "config": "configs/paderborn.toml",
        "damage_profiles": profiles, "rates": results, "feature_group_ablation_64k": ablation,
        "pipeline_timing_64k": timing,
        "environment_start": env_start, "environment": environment(),
        "wall_clock_s": time.time() - t_start,
    }
    (ROOT / "results" / "paderborn_benchmark.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote results/paderborn_benchmark.json in {payload['wall_clock_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
