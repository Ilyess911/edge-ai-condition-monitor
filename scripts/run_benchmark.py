"""Simulated-track benchmark: detection quality vs computational cost.

    uv run python scripts/run_benchmark.py            # full run, ~15 min on a laptop
    uv run python scripts/run_benchmark.py --quick    # development seed only, short paced runs

Writes results/simulated_benchmark.json (every number) and
results/traces/*.npz (per-window scores used by the figures).
"""

from __future__ import annotations

import _threads  # noqa: F401  (must precede numpy)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarking import simulated_track as st  # noqa: E402
from src.benchmarking.profiling import (  # noqa: E402
    environment,
    latency_summary,
    model_load_rss,
    scoring_memory,
)
from src.config import ROOT, load_config  # noqa: E402
from src.inference.onnx_export import OnnxDetector, to_onnx  # noqa: E402
from src.inference.packed_forest import PackedIsolationForestDetector  # noqa: E402

IFOREST_SWEEP = (10, 25, 50, 100, 200)


def aggregate(per_seed: list[dict]) -> dict:
    def ms(key, level):
        vals = np.array([r[level][key] for r in per_seed], dtype=float)
        return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    window = {k: ms(k, "window") for k in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
    event = {k: ms(k, "event") for k in
             ("event_recall", "alert_precision", "event_f1", "false_alarms_per_hour")}
    n_events = sum(r["event"]["n_events"] for r in per_seed)
    n_detected = sum(r["event"]["n_detected"] for r in per_seed)
    n_false = sum(r["event"]["n_false_alarms"] for r in per_seed)
    n_alerts = sum(r["event"]["n_alerts"] for r in per_seed)
    delays = [r["event"]["mean_delay_s"] * r["event"]["n_detected"] for r in per_seed
              if r["event"]["n_detected"]]
    by_kind: dict = {}
    for r in per_seed:
        for kind, v in r["event"]["detected_by_kind"].items():
            d = by_kind.setdefault(kind, {"detected": 0, "total": 0})
            d["detected"] += v["detected"]
            d["total"] += v["total"]
    return {
        "window": window,
        "event": event,
        "pooled": {
            "events": n_events,
            "detected": n_detected,
            "alerts": n_alerts,
            "false_alarms": n_false,
            "event_recall": n_detected / n_events,
            "alert_precision": (n_alerts - n_false) / n_alerts if n_alerts else 0.0,
            "mean_delay_s": float(sum(delays) / n_detected) if n_detected else float("nan"),
            "chance_detected_mean": float(sum(r["event"]["chance"]["chance_mean"] for r in per_seed)),
            "alarm_time_fraction_mean": float(np.mean([r["event"]["alarm_time_fraction"]
                                                       for r in per_seed])),
            "detected_by_kind": by_kind,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/simulated.toml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paced-seconds", type=float, default=60.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # --quick runs on the development seed so the test seeds stay unseen.
    seeds = [cfg["data"]["dev_seed"]] if args.quick else cfg["data"]["test_seeds"]
    paced_s = 10.0 if args.quick else args.paced_seconds
    out_dir = ROOT / "results"
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    env_start = environment()

    print("generating healthy training and calibration runs")
    train, calib = st.healthy_runs(cfg)
    X_train = st.stream_features(cfg, train)
    X_calib = st.stream_features(cfg, calib)
    fs, window, hop, chunk = st.geometry(cfg)
    n_feat = X_train.shape[1]

    # ---- variants ----------------------------------------------------------
    variants: dict[str, dict] = {}
    for name in ("zscore", "pca", "iforest", "autoencoder"):
        t0 = time.perf_counter()
        det, calib_ = st.fit_and_calibrate(cfg, name, X_train, X_calib)
        fit_s = time.perf_counter() - t0
        variants[name] = {"family": name, "runtime": "native", "detector": det,
                          "calib": calib_, "fit_s": fit_s}
    for n in IFOREST_SWEEP:
        if n == cfg["models"]["iforest"]["n_estimators"]:
            continue
        t0 = time.perf_counter()
        det, calib_ = st.fit_and_calibrate(cfg, "iforest", X_train, X_calib, n_estimators=n)
        variants[f"iforest{n}"] = {"family": "iforest", "runtime": "native", "detector": det,
                                   "calib": calib_, "fit_s": time.perf_counter() - t0}
    # Same trees, exported to flat arrays: identical scores, so identical calibration.
    for vname in [k for k in variants if k.startswith("iforest")]:
        base = variants[vname]
        variants[f"{vname}-packed"] = {"family": "iforest", "runtime": "numpy-packed",
                                       "detector": PackedIsolationForestDetector(base["detector"]),
                                       "calib": base["calib"], "fit_s": base["fit_s"]}
    for name in ("zscore", "pca", "iforest", "autoencoder"):
        base = variants[name]
        onx = OnnxDetector(to_onnx(base["detector"], n_feat), name)
        # The float32 graph gets its own threshold, calibrated the same way.
        variants[f"{name}-onnx"] = {"family": name, "runtime": "onnxruntime", "detector": onx,
                                    "calib": st.calibrate(cfg, onx, X_calib), "fit_s": base["fit_s"]}
    # Ablation: fixed persistence of 3 windows instead of the calibrated value.
    for name in ("zscore", "pca", "iforest", "autoencoder"):
        base = variants[name]
        variants[f"{name}-k3"] = {"family": name, "runtime": "native", "detector": base["detector"],
                                  "calib": st.calibrate(cfg, base["detector"], X_calib, raise_after=3),
                                  "fit_s": base["fit_s"], "ablation": True}

    # ---- evaluation on unseen test seeds ---------------------------------------
    results: dict[str, dict] = {}
    runs = {}
    for seed in seeds:
        print(f"test seed {seed}")
        runs[seed] = st.test_run(cfg, seed)
    for vname, v in variants.items():
        per_seed, inf_ns, pipe_ns, feat_ns, ingest_ns, tput = [], [], [], [], [], []
        for seed, run in runs.items():
            report, res = st.evaluate(cfg, v["detector"], v["calib"], run)
            per_seed.append({"seed": seed, "window": res["window"], "event": res["event"]})
            inf_ns.append(report.column("inference_ns")[20:])
            pipe_ns.append(report.column("pipeline_ns")[20:])
            feat_ns.append(report.column("feature_ns")[20:])
            ingest_ns.append(report.ingest_ns / chunk)
            tput.append(report.throughput_sps)
            if seed == seeds[0] and not v.get("ablation"):
                np.savez_compressed(
                    out_dir / "traces" / f"{vname}_seed{seed}.npz",
                    t=report.column("t_end"), score=res["scores"], y=res["y"],
                    ignore=res["ignore"], threshold=v["calib"].threshold,
                    alerts=np.array([[a.start_t, a.end_t] for a in res["alerts"]]).reshape(-1, 2),
                    events=np.array([[e.start_s, e.end_s] for e in run.events]),
                    kinds=np.array([e.kind for e in run.events]),
                )
        det = v["detector"]
        cost = {
            "inference_latency": latency_summary(np.concatenate(inf_ns), warmup=0),
            "pipeline_latency": latency_summary(np.concatenate(pipe_ns), warmup=0),
            "feature_latency": latency_summary(np.concatenate(feat_ns), warmup=0),
            "ingest_ns_per_sample_mean": float(np.mean(np.concatenate(ingest_ns))),
            "throughput_sps_mean": float(np.mean(tput)),
            "realtime_factor_mean": float(np.mean(tput) / fs),
            "serialized_kib": det.serialized_bytes() / 1024,
        }
        if hasattr(det, "n_parameters"):
            cost["n_parameters"] = int(det.n_parameters())
        if v["runtime"] == "native":
            cost.update(model_load_rss(det))
        cost.update(scoring_memory(det, X_calib))
        results[vname] = {
            "family": v["family"], "runtime": v["runtime"], "ablation": bool(v.get("ablation")),
            "threshold": v["calib"].threshold, "raise_after": v["calib"].raise_after,
            "fit_s": v["fit_s"], "per_seed": per_seed, **aggregate(per_seed), "cost": cost,
        }
        r = results[vname]
        print(f"  {vname:<18} eventF1 {r['event']['event_f1']['mean']:.3f} "
              f"PR-AUC {r['window']['pr_auc']['mean']:.3f} "
              f"inf p50 {cost['inference_latency']['p50_us']:.1f} us "
              f"tput {cost['throughput_sps_mean']:.0f} sps")

    # ---- paced CPU utilisation at the real sensor rate -------------------------
    paced_run = runs[seeds[0]]
    n_paced = int(paced_s * fs)
    for vname in ("zscore", "pca", "iforest", "iforest-packed", "autoencoder",
                  "zscore-onnx", "pca-onnx", "iforest-onnx", "autoencoder-onnx"):
        v = variants[vname]
        eng = st.make_engine(cfg, v["detector"], v["calib"])
        rep = eng.run(paced_run.signals[:n_paced], chunk, paced=True)
        results[vname]["cost"]["paced"] = {
            "seconds": paced_s, "cpu_utilisation": rep.cpu_utilisation,
            "max_lateness_ms": rep.max_lateness_s * 1e3,
        }
        print(f"  paced {vname:<18} cpu {100 * rep.cpu_utilisation:.2f} % of one core, "
              f"max lateness {rep.max_lateness_s * 1e3:.2f} ms")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config, "quick": args.quick, "test_seeds": seeds,
        "stream": {"fs": fs, "window": window, "hop": hop, "chunk": chunk},
        "train_windows": int(len(X_train)), "calib_windows": int(len(X_calib)),
        "environment_start": env_start, "environment": environment(), "results": results,
        "wall_clock_s": time.time() - t_start,
    }
    name = "simulated_benchmark_quick.json" if args.quick else "simulated_benchmark.json"
    (out_dir / name).write_text(json.dumps(payload, indent=2))
    print(f"wrote results/{name} in {payload['wall_clock_s']:.0f} s")


if __name__ == "__main__":
    main()
