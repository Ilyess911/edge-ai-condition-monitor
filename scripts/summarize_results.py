"""Print the Markdown tables used in README.md and docs/experiments.md.

    uv run python scripts/summarize_results.py > results/summary.md

Every number in those documents is copied from this output, which is computed
from results/*.json. Nothing is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

LABEL = {"zscore": "Z-score", "pca": "PCA (SPE+T²)", "iforest": "Isolation Forest",
         "autoencoder": "Autoencoder"}
RUNTIME = {"native": "NumPy / scikit-learn", "numpy-packed": "packed NumPy", "onnxruntime": "ONNX Runtime"}
MCU_FLASH_KIB = 256  # a mid-range Cortex-M4F has 0.5-1 MiB of flash; half for the model


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def suitability(r, hop_ms):
    c = r["cost"]
    p99_ms = c["pipeline_latency"]["p99_us"] / 1e3
    size = c["serialized_kib"]
    budget = f"p99 pipeline {100 * p99_ms / hop_ms:.2f} % of the {hop_ms:.0f} ms hop"
    if r["runtime"] == "native" and r["family"] == "iforest":
        tier = "gateway only (framework overhead)"
    elif r["family"] in ("zscore", "pca", "autoencoder") and size < MCU_FLASH_KIB:
        tier = "MCU-portable (matrix ops, small)"
    elif size < MCU_FLASH_KIB:
        tier = "MCU-portable (tables, small)"
    else:
        tier = "SBC / gateway"
    return f"{tier}; {budget}"


def simulated(b):
    res = b["results"]
    hop_ms = b["stream"]["hop"] / b["stream"]["fs"] * 1e3
    env = b["environment"]
    print(f"### Simulated track: {len(b['test_seeds'])} unseen test runs "
          f"(seeds {b['test_seeds'][0]}-{b['test_seeds'][-1]}), "
          f"{res['pca']['pooled']['events']} injected faults, {b['stream']['fs']:.0f} Hz\n")
    print(f"Machine: {env['cpu']}, {env['logical_cpus']} logical CPUs, Python {env['python']}, "
          f"one thread. Load average at start {b['environment_start']['load_average_1_5_15']}, "
          f"at end {env['load_average_1_5_15']} (other workloads were running).\n")

    print("| Model | Runtime | Faults detected | Event F1 | False alarms / h | Mean delay | "
          "Window PR-AUC | Inference p50 / p99 | Throughput | Size | Deployment class (rule, untested) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    order = ["zscore", "pca", "autoencoder", "iforest", "iforest-packed",
             "zscore-onnx", "pca-onnx", "autoencoder-onnx", "iforest-onnx"]
    for name in order:
        r = res[name]
        p, e, w, c = r["pooled"], r["event"], r["window"], r["cost"]
        lat = c["inference_latency"]
        print(f"| {LABEL[r['family']]} | {RUNTIME[r['runtime']]} | {p['detected']}/{p['events']} | "
              f"{e['event_f1']['mean']:.2f} ± {e['event_f1']['std']:.2f} | "
              f"{e['false_alarms_per_hour']['mean']:.2f} | {p['mean_delay_s']:.0f} s | "
              f"{w['pr_auc']['mean']:.2f} ± {w['pr_auc']['std']:.2f} | "
              f"{lat['p50_us']:.0f} / {lat['p99_us']:.0f} µs | "
              f"{c['throughput_sps_mean'] / 1e3:.0f} k samples/s | {c['serialized_kib']:.1f} KiB | "
              f"{suitability(r, hop_ms)} |")

    print("\nDeployment class is a stated rule, not a hardware result: matrix-operation or "
          f"table models under {MCU_FLASH_KIB} KiB serialized are marked MCU-portable, larger "
          "ones SBC / gateway, scikit-learn Isolation Forest gateway only. The budget column "
          "compares laptop p99 window pipeline latency (features + inference + alert) with "
          "the 500 ms hop.")
    print("\n#### Chance level and alarm time\n")
    print("| Model | Detected | Same alerts at random offsets | Healthy time in alarm | "
          "Persistence (windows) |")
    print("|---|---|---|---|---|")
    for name in ("zscore", "pca", "autoencoder", "iforest"):
        r = res[name]
        print(f"| {LABEL[name]} | {r['pooled']['detected']} | "
              f"{r['pooled']['chance_detected_mean']:.1f} | "
              f"{100 * r['pooled']['alarm_time_fraction_mean']:.2f} % | {r['raise_after']} |")

    print("\n#### Detection by fault type (pooled over test runs)\n")
    kinds = ["bearing_outer_race", "imbalance", "cooling_failure", "cavitation", "electrical_overload"]
    print("| Model | " + " | ".join(kinds) + " |")
    print("|---|" + "---|" * len(kinds))
    for name in ("zscore", "pca", "autoencoder", "iforest"):
        d = res[name]["pooled"]["detected_by_kind"]
        print(f"| {LABEL[name]} | " + " | ".join(f"{d[k]['detected']}/{d[k]['total']}" for k in kinds) + " |")

    print("\n#### Isolation Forest size sweep (packed runtime)\n")
    print("| Trees | Event F1 | Faults detected | PR-AUC | Inference p50 packed | p50 scikit-learn | Size packed |")
    print("|---|---|---|---|---|---|---|")
    for n in (10, 25, 50, 100, 200):
        key = "iforest" if n == 100 else f"iforest{n}"
        rp, rs = res[f"{key}-packed"], res[key]
        print(f"| {n} | {rp['event']['event_f1']['mean']:.2f} | {rp['pooled']['detected']}/{rp['pooled']['events']} | "
              f"{rp['window']['pr_auc']['mean']:.2f} | {rp['cost']['inference_latency']['p50_us']:.0f} µs | "
              f"{rs['cost']['inference_latency']['p50_us']:.0f} µs | {rp['cost']['serialized_kib']:.0f} KiB |")

    print("\n#### Ablation: fixed persistence (3 windows) vs calibrated\n")
    print("| Model | Calibrated k | Event F1 calibrated | False alarms / h | Delay | "
          "Event F1 k=3 | False alarms / h k=3 | Delay k=3 |")
    print("|---|---|---|---|---|---|---|---|")
    for name in ("zscore", "pca", "autoencoder", "iforest"):
        a, b3 = res[name], res[f"{name}-k3"]
        print(f"| {LABEL[name]} | {a['raise_after']} | {a['event']['event_f1']['mean']:.2f} | "
              f"{a['event']['false_alarms_per_hour']['mean']:.2f} | {a['pooled']['mean_delay_s']:.0f} s | "
              f"{b3['event']['event_f1']['mean']:.2f} | {b3['event']['false_alarms_per_hour']['mean']:.2f} | "
              f"{b3['pooled']['mean_delay_s']:.0f} s |")

    print("\n#### Resource use at the real sensor rate (paced 60 s, 1 kHz)\n")
    print("| Variant | CPU (% of one core) | Max lateness | Scoring heap peak | Parameters |")
    print("|---|---|---|---|---|")
    for name in ("zscore", "pca", "autoencoder", "iforest", "iforest-packed",
                 "zscore-onnx", "pca-onnx", "autoencoder-onnx", "iforest-onnx"):
        c = res[name]["cost"]
        paced = c.get("paced", {})
        print(f"| {name} | {100 * paced.get('cpu_utilisation', float('nan')):.2f} | "
              f"{paced.get('max_lateness_ms', float('nan')):.1f} ms | {c['scoring_peak_kib']:.1f} KiB | "
              f"{c.get('n_parameters', '')} |")


def sweep(s):
    print("\n### Sampling-rate sweep (SIMULATED, same 5 test seeds)\n")
    print("| Rate | Model | Faults detected | False alarms | PR-AUC | Feature p50 | Real-time factor |")
    print("|---|---|---|---|---|---|---|")
    for rate in s["rates_hz"]:
        for name in ("zscore", "pca", "autoencoder", "iforest"):
            r = s["results"][str(rate)][name]
            print(f"| {rate} Hz | {LABEL[name]} | {r['detected']}/{r['events']} | {r['false_alarms']} | "
                  f"{r['pr_auc_mean']:.2f} | {r['feature_latency']['p50_us']:.0f} µs | "
                  f"{r['realtime_factor_mean']:.0f}x |")
    print("\nBearing faults detected per rate:\n")
    print("| Model | " + " | ".join(f"{r} Hz" for r in s["rates_hz"]) + " |")
    print("|---|" + "---|" * len(s["rates_hz"]))
    for name in ("zscore", "pca", "autoencoder", "iforest"):
        cells = []
        for rate in s["rates_hz"]:
            d = s["results"][str(rate)][name]["detected_by_kind"]["bearing_outer_race"]
            cells.append(f"{d[0]}/{d[1]}")
        print(f"| {LABEL[name]} | " + " | ".join(cells) + " |")


def metropt(m, title):
    print(f"\n### {title}\n")
    print(f"Test period {m['test_hours'] / 24:.0f} days, {m['train_windows']} training windows, "
          f"{m['calib_windows']} calibration windows.\n")
    print("| Model | Reports detected | Chance level (p) | Detected with 2 h early window | "
          "False alarms / day | Healthy time in alarm | ROC-AUC | PR-AUC | Inference p50 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in ("zscore", "pca", "autoencoder", "iforest", "iforest-packed"):
        r = m["results"][name]
        label = LABEL.get(name, "Isolation Forest (packed)")
        print(f"| {label} | {r['event_strict']['n_detected']}/4 | {r['chance']['chance_mean']:.2f} "
              f"(p = {r['chance']['p_value']:.3f}) | {r['event_early_2h']['n_detected']}/4 | "
              f"{r['false_alarms_per_day']:.2f} | {100 * r['alarm_time_fraction_healthy']:.1f} % | "
              f"{r['window']['roc_auc']:.3f} | {r['window']['pr_auc']:.3f} | "
              f"{r['inference_latency']['p50_us']:.0f} µs |")
    print("\nDelay per report (minutes from reported start to first alert):\n")
    print("| Model | #1 | #2 | #3 | #4 |")
    print("|---|---|---|---|---|")
    for name in ("zscore", "pca", "autoencoder", "iforest"):
        cells = [("missed" if not e["detected"] else f"{e['delay_min']:.0f}")
                 for e in m["results"][name]["per_event"]]
        print(f"| {LABEL[name]} | " + " | ".join(cells) + " |")


def main():
    b = load("simulated_benchmark.json")
    if b:
        simulated(b)
    s = load("sampling_rate_sweep.json")
    if s:
        sweep(s)
    m = load("metropt_benchmark.json")
    if m:
        metropt(m, "MetroPT-3 (REAL data), pre-registered features")
    mp = load("metropt_benchmark_posthoc_drop_oil_level_fraction.json")
    if mp:
        metropt(mp, "MetroPT-3 POST-HOC ablation: without oil_level_fraction")


if __name__ == "__main__":
    main()
