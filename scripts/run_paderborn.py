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
from src.features.bearing import FEATURE_NAMES  # noqa: E402
from src.inference.packed_forest import PackedIsolationForestDetector  # noqa: E402
from src.inference.threshold import calibrate_threshold  # noqa: E402
from src.models.detectors import build  # noqa: E402

FEAT_CACHE = ROOT / "data" / "cache" / "paderborn_features"
# Feature-group ablation at 64 kHz. Envelope energy at BPFO/BPFI is locked to
# shaft speed and bearing geometry; broadband statistics can also pick up
# differences between recording sessions or mountings. Comparing the groups
# shows which kind of evidence the detectors rely on.
GROUPS = {
    "all": None,
    "no_envelope": [n for n in FEATURE_NAMES if not n.startswith("env_")],
    "envelope_context": ["env_bpfo", "env_bpfi", "speed", "torque", "force"],
}
BEARINGS = HEALTHY + REAL_DAMAGE


def features_at_rate(cfg: dict, fs: float) -> dict[str, pt.BearingFeatures]:
    out = {}
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    for code in BEARINGS:
        path = FEAT_CACHE / f"{code}_{int(fs)}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            n = z["n_windows"]
            bounds = np.cumsum(np.concatenate([[0], n]))
            bf = pt.BearingFeatures(code)
            bf.features = [z["features"][a:b] for a, b in zip(bounds, bounds[1:])]
            bf.conditions = list(z["conditions"])
            bf.feature_ns = [z["feature_ns"][a:b] for a, b in zip(bounds, bounds[1:])]
            bf.ingest_ns_per_sample = list(z["ingest_ns_per_sample"])
            bf.throughput_sps = list(z["throughput_sps"])
        else:
            bf = pt.extract_bearing(cfg, load_bearing(code), fs)
            np.savez(path, features=np.vstack(bf.features),
                     n_windows=np.array([len(f) for f in bf.features]),
                     conditions=np.array(bf.conditions), feature_ns=np.concatenate(bf.feature_ns),
                     ingest_ns_per_sample=np.array(bf.ingest_ns_per_sample),
                     throughput_sps=np.array(bf.throughput_sps))
        out[code] = bf
        print(f"  features {code} @ {fs / 1000:.0f} kHz: {sum(len(f) for f in bf.features)} windows",
              flush=True)
    return out


def subset(feats: dict[str, pt.BearingFeatures], names: list[str] | None) -> dict[str, pt.BearingFeatures]:
    if names is None:
        return feats
    idx = [FEATURE_NAMES.index(n) for n in names]
    out = {}
    for code, bf in feats.items():
        sub = pt.BearingFeatures(code, [f[:, idx] for f in bf.features], bf.conditions,
                                 bf.feature_ns, bf.ingest_ns_per_sample, bf.throughput_sps)
        out[code] = sub
    return out


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
        X_train = np.vstack([feats[c].stacked for c in fold["train"]])
        fold_res = {}
        for name, mcfg in (models or cfg["models"]).items():
            kind, kwargs = detector_kwargs(mcfg)
            t0 = time.perf_counter()
            det = build(kind, **kwargs).fit(X_train)
            fit_s = time.perf_counter() - t0
            scorer = PackedIsolationForestDetector(det) if kind == "iforest" else det
            calib_scores = []
            for c in fold["calib"]:
                s, _ = pt.score_recordings(scorer, feats[c])
                calib_scores += s
            thr = calibrate_threshold(np.concatenate(calib_scores), cfg["threshold"]["quantile"])
            k = min(pt.longest_run_within(calib_scores, thr) + 1, a["max_raise_after"])
            test = {}
            lat = []
            for c in fold["test_healthy"] + list(REAL_DAMAGE):
                s, ns = pt.score_recordings(scorer, feats[c])
                test[c] = (s, c in REAL_DAMAGE)
                lat.append(ns)
            ev = pt.evaluate_fold(test, thr, k, a["clear_after"], hop_s)
            ev.update({"threshold": thr, "raise_after": k, "fit_s": fit_s,
                       "train_windows": int(len(X_train))})
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
    for group, names in GROUPS.items():
        if names is None:
            continue
        print(f"ablation {group}", flush=True)
        ablation[group] = {"features": names,
                           **run_rate(cfg, cfg["stream"]["fs"], subset(feats64, names), timing=False,
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
