"""Computational measurements. Every number comes from a clock or the OS.

- latency:    time.perf_counter_ns around a call, per window, warm-up excluded
- throughput: samples replayed / wall seconds, unpaced
- CPU:        time.process_time delta / wall delta, paced at the sensor rate
- memory:     tracemalloc peak of Python allocations during scoring, and the
              process RSS before/after loading the model (psutil)
- size:       pickled bytes of the trained detector; ONNX file bytes

What this does NOT measure: power, and anything about hardware other than the
machine recorded by environment().
"""

from __future__ import annotations

import os
import pickle
import platform
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import psutil


def latency_summary(ns: np.ndarray, warmup: int = 20) -> dict:
    x = np.asarray(ns, dtype=float)[warmup:] / 1e3  # microseconds
    if len(x) == 0:
        return {}
    return {
        "n": int(len(x)),
        "mean_us": float(x.mean()),
        "p50_us": float(np.percentile(x, 50)),
        "p95_us": float(np.percentile(x, 95)),
        "p99_us": float(np.percentile(x, 99)),
        "max_us": float(x.max()),
    }


def scoring_memory(detector, X: np.ndarray, repeats: int = 200) -> dict:
    """Peak Python-heap allocation while scoring one window at a time."""
    rows = [X[i % len(X)][None, :] for i in range(repeats)]
    tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    for r in rows:
        detector.score(r)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"scoring_peak_kib": (peak - base) / 1024}


def model_load_rss(detector) -> dict:
    """RSS growth when a serialized detector is loaded. Coarse: the allocator
    may reuse freed pages, so small models can show 0."""
    blob = pickle.dumps(detector, protocol=pickle.HIGHEST_PROTOCOL)
    proc = psutil.Process()
    before = proc.memory_info().rss
    loaded = pickle.loads(blob)
    after = proc.memory_info().rss
    del loaded
    return {"serialized_kib": len(blob) / 1024, "load_rss_delta_kib": (after - before) / 1024}


def microbench(fn, n: int = 2000, warmup: int = 50) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = np.empty(n, dtype=np.int64)
    clock = time.perf_counter_ns
    for i in range(n):
        a = clock()
        fn()
        out[i] = clock() - a
    return out


def environment() -> dict:
    def version(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    cpu = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "sklearn": version("sklearn"),
        "onnxruntime": version("onnxruntime"),
        # Other processes compete for the same cores: record how busy the machine was.
        "load_average_1_5_15": [round(x, 2) for x in os.getloadavg()],
        "thread_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "note": "Development laptop, NOT an edge device. See docs/hardware_deployment.md.",
    }
