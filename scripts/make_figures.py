"""Regenerate every figure in assets/ from results/ (no number is typed by hand).

    uv run python scripts/make_figures.py

Needs results/simulated_benchmark.json, results/sampling_rate_sweep.json,
results/metropt_benchmark.json and results/traces/ from the benchmark scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarking import simulated_track as st  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.data.simulator import CHANNELS, FaultEvent, simulate  # noqa: E402

ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"

# Categorical slots in fixed order (validated reference palette), one per model family.
FAMILY = {"zscore": "#2a78d6", "pca": "#eb6834", "iforest": "#1baf7a", "autoencoder": "#eda100"}
LABEL = {"zscore": "Z-score", "pca": "PCA (SPE+T²)", "iforest": "Isolation Forest",
         "autoencoder": "Autoencoder"}
RUNTIME_MARKER = {"native": "o", "numpy-packed": "s", "onnxruntime": "^"}
RUNTIME_LABEL = {"native": "native (NumPy / scikit-learn)", "numpy-packed": "packed NumPy trees",
                 "onnxruntime": "ONNX Runtime"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAULT_BG = "#d9d8d3"
ALERT = "#e34948"  # status: critical

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "text.color": INK,
})


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(ASSETS / name, facecolor="white")
    plt.close(fig)
    print(f"assets/{name}")


def shade_faults(ax, events, kinds=None, label_y=None, scale=1.0):
    for i, (a, b) in enumerate(events):
        ax.axvspan(a / scale, b / scale, color=FAULT_BG, alpha=0.6, lw=0, zorder=0)
        if kinds is not None and label_y is not None:
            ax.text((a + b) / 2 / scale, label_y, str(kinds[i]).replace("_", " "),
                    ha="center", va="bottom", fontsize=7, color=INK2,
                    transform=ax.get_xaxis_transform())


# ----------------------------------------------------------------------------- architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.set_axis_off()
    ax.grid(False)
    stages = [
        ("Sensor\nstream", "vibration, temp,\ncurrent, speed,\npressure\n1 kHz, 50 ms chunks"),
        ("Edge\npreprocessing", "plausibility check\nsample-and-hold\ncircular buffer"),
        ("Feature\nextraction", "RMS, kurtosis,\ncrest factor\nFFT band energy\nslow-channel means"),
        ("Lightweight\nmodel", "z-score, PCA\nIsolation Forest\nautoencoder"),
        ("Anomaly\nscore", "one per window\nevery 0.5 s\nlatency timed"),
        ("Local alert\nengine", "calibrated threshold\npersistence\nhysteresis"),
        ("Dashboard\n(optional)", "Streamlit\nhealth, alerts\nlatency"),
    ]
    w, h, gap = 1.5, 1.9, 0.3
    for i, (title, body) in enumerate(stages):
        x = i * (w + gap)
        dashed = i == len(stages) - 1
        box = FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             fc="#f4f3f0" if not dashed else "white", ec=INK2 if not dashed else GRID,
                             lw=1.0, ls="--" if dashed else "-")
        ax.add_patch(box)
        ax.text(x + w / 2, h - 0.33, title, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=INK, linespacing=1.2)
        ax.text(x + w / 2, h / 2 - 0.3, body, ha="center", va="center", fontsize=7, color=INK2,
                linespacing=1.45)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, h / 2), xytext=(x + w + 0.02, h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))
    ax.set_xlim(-0.05, len(stages) * (w + gap))
    ax.set_ylim(-0.1, h + 0.1)
    save(fig, "architecture.png")


# ----------------------------------------------------------------------------- sensor stream
def fig_sensor_stream(cfg, seed):
    run = st.test_run(cfg, seed)
    fs = run.fs
    n_sec = int(run.duration_s)
    per_s = run.signals[: n_sec * int(fs)].reshape(n_sec, int(fs), -1)
    t = np.arange(n_sec) / 60
    rows = [("vibration RMS [g]", np.sqrt(np.nanmean(per_s[:, :, 0] ** 2, axis=1))),
            ("temperature [°C]", np.nanmean(per_s[:, :, 1], axis=1)),
            ("current [A]", np.nanmean(per_s[:, :, 2], axis=1)),
            ("speed [rpm]", np.nanmean(per_s[:, :, 3], axis=1)),
            ("pressure [bar]", np.nanmean(per_s[:, :, 4], axis=1))]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 7), sharex=True)
    events = [(e.start_s, e.end_s) for e in run.events]
    for k, (ax, (name, y)) in enumerate(zip(axes, rows)):
        shade_faults(ax, events, [e.kind for e in run.events] if k == 0 else None,
                     label_y=1.02, scale=60)
        ax.plot(t, y, color=INK, lw=0.8)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[0].set_title(f"SIMULATED sensor stream, test run seed {seed} (1 s means of 1 kHz samples; "
                      "grey = injected fault)", loc="left", pad=16)
    axes[-1].set_xlabel("time [min]")
    save(fig, "sensor_stream.png")


def fig_vibration_signature():
    fs = 1000
    healthy = simulate(4, fs, seed=11, artefacts=False)
    faulty = simulate(4, fs, [FaultEvent("bearing_outer_race", 0, 4, 1.0)], seed=11, artefacts=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))
    t = np.arange(300) / fs * 1000
    for run, color, name in ((healthy, "#8c8b87", "healthy"), (faulty, ALERT, "outer-race defect")):
        axes[0].plot(t, run.signals[1000:1300, 0], color=color, lw=0.9, label=name)
        x = run.signals[: 4 * fs, 0] - run.signals[: 4 * fs, 0].mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / fs)
        axes[1].semilogy(f, spec + 1e-6, color=color, lw=0.7, label=name)
    axes[0].set(title="SIMULATED raw vibration, 300 ms", xlabel="time [ms]",
                ylabel="acceleration [g]")
    axes[1].set(title="Power spectrum: defect impacts add BPFO harmonics (band edges in grey)",
                xlabel="frequency [Hz]", ylabel="power")
    for edge in (60, 200):
        axes[1].axvline(edge, color="#b5b4ae", lw=1)
    axes[0].legend(loc="upper right")
    save(fig, "vibration_signature.png")


# ----------------------------------------------------------------------------- score timeline
def fig_score_timeline(seed):
    names = ["zscore", "pca", "iforest", "autoencoder"]
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 7), sharex=True)
    for ax, name in zip(axes, names):
        z = np.load(RESULTS / "traces" / f"{name}_seed{seed}.npz")
        t = z["t"] / 60
        ratio = z["score"] / z["threshold"]
        kinds = z["kinds"] if name == names[0] else None
        shade_faults(ax, z["events"], kinds, label_y=1.02, scale=60)
        ax.plot(t, np.clip(ratio, 0, 6), color=FAMILY[name], lw=0.7)
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        for a, b in z["alerts"]:
            ax.plot([a / 60, b / 60], [5.6, 5.6], color=ALERT, lw=4, solid_capstyle="butt")
        ax.set_ylim(0, 6)
        ax.set_ylabel(LABEL[name], rotation=0, ha="right", va="center")
    axes[0].set_title("Anomaly score / calibrated threshold (clipped at 6). Red bars = raised "
                      "alerts, grey = injected fault", loc="left", pad=16)
    axes[-1].set_xlabel("time [min]  ·  SIMULATED test run seed %d" % seed)
    save(fig, "anomaly_score_timeline.png")


# ----------------------------------------------------------------------------- model comparison
def fig_model_comparison(bench):
    res = bench["results"]
    names = ["zscore", "pca", "iforest", "autoencoder"]
    metrics = [("event recall", lambda r: r["pooled"]["event_recall"]),
               ("alert precision", lambda r: r["pooled"]["alert_precision"]),
               ("window PR-AUC", lambda r: r["window"]["pr_auc"]["mean"]),
               ("window ROC-AUC", lambda r: r["window"]["roc_auc"]["mean"])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    width = 0.19
    x = np.arange(len(metrics))
    for i, name in enumerate(names):
        vals = [f(res[name]) for _, f in metrics]
        bars = ax.bar(x + (i - 1.5) * width, vals, width - 0.02, color=FAMILY[name],
                      label=LABEL[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [m for m, _ in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Detection over {len(bench['test_seeds'])} unseen test runs "
                 f"({res['pca']['pooled']['events']} faults)", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    ax.grid(axis="x", visible=False)

    kinds = ["bearing_outer_race", "imbalance", "cooling_failure", "cavitation",
             "electrical_overload"]
    mat = np.array([[res[n]["pooled"]["detected_by_kind"][k]["detected"] /
                     res[n]["pooled"]["detected_by_kind"][k]["total"] for k in kinds] for n in names])
    ax2.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(kinds)):
            d = res[names[i]]["pooled"]["detected_by_kind"][kinds[j]]
            ax2.text(j, i, f"{d['detected']}/{d['total']}", ha="center", va="center", fontsize=8,
                     color="white" if mat[i, j] > 0.6 else INK)
    ax2.set_xticks(range(len(kinds)), [k.replace("_", "\n") for k in kinds], fontsize=7)
    ax2.set_yticks(range(len(names)), [LABEL[n] for n in names])
    ax2.grid(False)
    ax2.set_title("Faults detected, by type", loc="left")
    save(fig, "model_comparison.png")


# ----------------------------------------------------------------------------- latency
def fig_latency(bench):
    res = bench["results"]
    order = ["zscore", "zscore-onnx", "pca", "pca-onnx", "autoencoder", "autoencoder-onnx",
             "iforest-packed", "iforest-onnx", "iforest"]
    order = [o for o in order if o in res]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, name in enumerate(order):
        lat = res[name]["cost"]["inference_latency"]
        fam, rt = res[name]["family"], res[name]["runtime"]
        y = len(order) - 1 - i
        ax.plot([lat["p50_us"], lat["p99_us"]], [y, y], color=FAMILY[fam], lw=2)
        ax.plot(lat["p50_us"], y, RUNTIME_MARKER[rt], color=FAMILY[fam], ms=8, mec="white", mew=1)
        ax.plot(lat["p99_us"], y, "|", color=FAMILY[fam], ms=12, mew=2)
        ax.text(lat["p99_us"] * 1.15, y, f"p50 {lat['p50_us']:.0f} µs · p99 {lat['p99_us']:.0f} µs",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), [f"{LABEL[res[n]['family']]} · {RUNTIME_LABEL[res[n]['runtime']]}"
                                      for n in reversed(order)], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 6)
    ax.set_xlabel("inference latency per window [µs], log scale  (marker = p50, bar end = p99)")
    ax.grid(axis="y", visible=False)
    env = bench["environment"]
    ax.set_title(f"Single-window inference latency, one thread · {env['cpu']} (laptop, not an edge "
                 "device)", loc="left")
    save(fig, "latency_distribution.png")


# ----------------------------------------------------------------------------- trade-off
def fig_tradeoff(bench):
    res = bench["results"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    pts = []
    for name, r in res.items():
        if r["ablation"]:
            continue
        x = r["cost"]["inference_latency"]["p50_us"]
        y = r["event"]["event_f1"]["mean"]
        pts.append((x, y, name))
        size = 30 + 60 * np.log10(1 + r["cost"]["serialized_kib"])
        ax.scatter(x, y, s=size, marker=RUNTIME_MARKER[r["runtime"]], color=FAMILY[r["family"]],
                   edgecolor="white", linewidth=1.2, zorder=3)
    # Isolation Forest size sweep on the packed runtime, joined in tree-count order.
    sweep = sorted([(int(n.split("-")[0].replace("iforest", "") or 100), res[n])
                    for n in res if n.startswith("iforest") and n.endswith("-packed")])
    ax.plot([r["cost"]["inference_latency"]["p50_us"] for _, r in sweep],
            [r["event"]["event_f1"]["mean"] for _, r in sweep], color=FAMILY["iforest"], lw=1,
            ls=":", zorder=2)
    for n, r in sweep:
        ax.annotate(f"{n} trees", (r["cost"]["inference_latency"]["p50_us"],
                                   r["event"]["event_f1"]["mean"]),
                    textcoords="offset points", xytext=(4, -11), fontsize=7, color=INK2)
    # Pareto frontier: nothing both faster and better.
    front = []
    for x, y, n in sorted(pts):
        if not front or y > front[-1][1]:
            front.append((x, y, n))
    ax.step([p[0] for p in front], [p[1] for p in front], where="post", color=INK2, lw=0.8,
            ls="--", zorder=1)
    for x, y, n in front:
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("inference latency per window, p50 [µs], log scale")
    ax.set_ylabel("event F1 (mean over test runs)")
    ax.set_title("Detection quality vs inference cost. Colour = model, marker = runtime, size = "
                 "serialized size. Dashed = Pareto frontier", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=LABEL[f])
               for f, c in FAMILY.items()]
    handles += [plt.Line2D([], [], marker=m, ls="", color=INK2, ms=7, label=RUNTIME_LABEL[rt])
                for rt, m in RUNTIME_MARKER.items()]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)
    save(fig, "tradeoff_performance_latency.png")


# ----------------------------------------------------------------------------- sampling rate
def fig_sampling(sweep):
    rates = [str(r) for r in sweep["rates_hz"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for name, color in FAMILY.items():
        rec = [sweep["results"][r][name]["event_recall"] for r in rates]
        a1.plot(rates, rec, "-o", color=color, lw=2, ms=6, label=LABEL[name])
    a1.set(title="Event recall vs ADC sampling rate", xlabel="sampling rate [Hz]",
           ylabel="event recall", ylim=(0, 1.05))
    a1.legend(fontsize=7.5, loc="lower right")
    feat = [sweep["results"][r]["pca"]["feature_latency"]["p50_us"] for r in rates]
    ingest = [sweep["results"][r]["pca"]["ingest_ns_per_sample"] * int(r) / 1e3 for r in rates]
    a2.plot(rates, feat, "-o", color=INK, lw=2, ms=6, label="feature extraction per window")
    a2.plot(rates, ingest, "-s", color=INK2, lw=2, ms=6, label="ingest per second of signal")
    a2.set(title="Edge cost vs sampling rate (PCA pipeline)", xlabel="sampling rate [Hz]",
           ylabel="p50 [µs]")
    a2.legend(fontsize=7.5)
    save(fig, "sampling_rate_tradeoff.png")


# ----------------------------------------------------------------------------- MetroPT
def fig_metropt(mb):
    names = ["zscore", "pca", "iforest", "autoencoder"]
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 6.4), sharex=True)
    for ax, name in zip(axes, names):
        z = np.load(RESULTS / "traces" / f"metropt_{name}.npz")
        t = z["t"] / 86400
        ratio = np.clip(z["score"] / z["threshold"], 0, 6)
        # Break the line across logger gaps instead of drawing a straight segment over them.
        gap = np.flatnonzero(np.diff(z["t"]) > 3 * 300) + 1
        t = np.insert(t, gap, np.nan)
        ratio = np.insert(ratio, gap, np.nan)
        shade_faults(ax, z["events"], scale=86400)
        ax.plot(t, ratio, color=FAMILY[name], lw=0.5)
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        for a, b in z["alerts"]:
            ax.plot([a / 86400, b / 86400], [5.6, 5.6], color=ALERT, lw=4, solid_capstyle="butt")
        r = mb["results"][name]
        ax.set_ylim(0, 6)
        ax.set_ylabel(LABEL[name], rotation=0, ha="right", va="center")
        ax.text(1.0, 0.72, f"{r['event_strict']['n_detected']}/4 reports (chance "
                f"{r['chance']['chance_mean']:.1f}, p={r['chance']['p_value']:.2f}) · "
                f"{r['false_alarms_per_day']:.2f} false alarms/day · "
                f"{100 * r['alarm_time_fraction_healthy']:.0f} % of healthy time in alarm",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    axes[0].set_title("MetroPT-3 (REAL compressor data), test period Apr-Aug 2020. Grey = company "
                      "failure report, red bars = alerts", loc="left")
    axes[-1].set_xlabel("days since 2020-04-01")
    save(fig, "metropt_timeline.png")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    cfg = load_config("configs/simulated.toml")
    fig_architecture()
    fig_vibration_signature()
    bench = load("simulated_benchmark.json")
    if bench:
        seed = bench["test_seeds"][0]
        fig_sensor_stream(cfg, seed)
        fig_score_timeline(seed)
        fig_model_comparison(bench)
        fig_latency(bench)
        fig_tradeoff(bench)
    sweep = load("sampling_rate_sweep.json")
    if sweep:
        fig_sampling(sweep)
    mb = load("metropt_benchmark.json")
    if mb:
        fig_metropt(mb)


if __name__ == "__main__":
    main()
