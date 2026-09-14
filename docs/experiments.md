# Experiment log

Only experiments that were actually run. Numbers are copied from
`results/summary.md`, which `scripts/summarize_results.py` computes from
`results/*.json`. Raw console output is in `results/logs/`.

## Common environment (EXP-002 to EXP-010)

| Item | Value |
|---|---|
| Machine | Apple M4 laptop, 10 logical CPUs, macOS 26.5 (arm64). **Not an edge device.** |
| Software | Python 3.12.13, NumPy 2.5.3, scikit-learn 1.9.1, ONNX Runtime 1.30.0 |
| Threads | `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = VECLIB_MAXIMUM_THREADS = 1`; ONNX Runtime intra/inter-op threads = 1 |
| Timer | `time.perf_counter_ns`, one call per window, inside the streaming loop |
| Warm-up | first 20 windows of every run discarded |
| Background load | **other processes were running.** Load average is recorded in each JSON: EXP-002 started at 2.8 and ended at 22.8 (1-minute average) |
| Date | 2026-09-14 |

Latency definitions used everywhere:

- **Model inference latency:** `detector.score` on one 12-feature vector.
- **Window pipeline latency:** feature extraction + inference + alert update for one window.
- **Ingest cost:** cleaning + ring-buffer write of one 50 ms chunk, divided by 50 samples.
- **Throughput:** samples replayed / wall-clock seconds, unpaced, everything included.

Noise floor: feature extraction is identical code in every variant, yet its p50
ranged from 62 to 140 µs between variants of EXP-002 as background load
changed. **Latency differences below about 2x are not meaningful in this
environment.**

---

## EXP-001: development iterations (dev seed 101, not reported as results)

- **Dataset:** SIMULATED, train/calibration healthy runs, development run seed 101.
- **Purpose:** build the pipeline and the evaluation protocol without touching test seeds.
- **Observations, in order:**
  1. 20-minute runs could not fit six faults. Moved to 60-minute runs with 5 faults.
  2. With 30 min of training and a fixed 3-window persistence: 4 to 7 false alarms
     per 40-minute run, **all starting during 5 s load/speed ramps**
     (`current_std` dominated the z-score).
  3. Thermal relaxation after a fault ended was counted as false alarms. Added a
     180 s recovery exclusion (1.5 τ).
  4. After 60 min of training and label-free persistence calibration: 1 to 2
     false alarms per run.
- **Frozen afterwards:** `configs/simulated.toml`.

## EXP-002: detector comparison, simulated track

- **Dataset:** SIMULATED rotating machine. Train 60 min (seed 1), calibration 30 min
  (seed 2), test 5 x 60 min (seeds 201-205) with 5 faults each (25 faults).
- **Configuration:** 1 kHz, window 1 s, hop 0.5 s, chunk 50 ms. Threshold =
  99.5th percentile of calibration scores; persistence calibrated (cap 20).
- **Measurements:** 35,895 windows per variant.

| Model | Faults detected | Event F1 (mean ± std over 5 runs) | False alarms / h | Mean delay | Window PR-AUC | Inference p50 / p99 | Window pipeline p50 / p99 | Serialized size | Parameters |
|---|---|---|---|---|---|---|---|---|---|
| Z-score | 8/25 | 0.41 ± 0.13 | 2.64 | 51 s | 0.77 ± 0.08 | 2.8 / 3.4 µs | 65 / 75 µs | 0.4 KiB | 24 |
| PCA (SPE+T²) | 18/25 | 0.73 ± 0.15 | 2.64 | 76 s | 0.85 ± 0.03 | 6.9 / 8.6 µs | 70 / 80 µs | 1.0 KiB | 78 |
| Autoencoder | 18/25 | 0.76 ± 0.05 | 2.26 | 55 s | 0.86 ± 0.02 | 13.0 / 18.9 µs | 100 / 138 µs | 2.8 KiB | 295 |
| Isolation Forest (100 trees) | 14/25 | 0.69 ± 0.08 | 0.79 | 35 s | 0.78 ± 0.06 | 2111 / 8570 µs | 2201 / 9048 µs | 1253 KiB | 35,125 |

Detection by fault type (5 of each):

| Model | bearing | imbalance | cooling | cavitation | electrical overload |
|---|---|---|---|---|---|
| Z-score | 5 | 0 | 0 | 2 | 1 |
| PCA | 5 | 3 | 4 | 5 | 1 |
| Autoencoder | 5 | 4 | 3 | 5 | 1 |
| Isolation Forest | 5 | 3 | 1 | 5 | 0 |

Chance control (same alerts at random circular offsets, 500 shifts per run):
Z-score 4.2, PCA 7.5, autoencoder 7.9, Isolation Forest 5.6 faults expected by
chance, against 8, 18, 18, 14 observed. Healthy time in alarm: 0.05 to 0.20 %.

**Interpretation.** Modelling correlations (PCA, autoencoder) more than doubles
the detected faults compared with per-feature limits. The non-linear autoencoder
does not detect more faults than PCA. Isolation Forest trades recall for the
lowest false-alarm rate. Electrical overload is missed by construction (load is
unobserved). With 25 events, the ± on event F1 is large: PCA and autoencoder
are not separable.

## EXP-003: same models, three runtimes

- **Dataset / configuration:** as EXP-002.
- **Question:** how much of the cost is the algorithm, how much the runtime?

| Model | NumPy / scikit-learn p50 | Packed NumPy p50 | ONNX Runtime p50 | Detection |
|---|---|---|---|---|
| Z-score | 2.8 µs | n/a | 9.7 µs | identical |
| PCA | 6.9 µs | n/a | 8.4 µs | identical |
| Autoencoder | 13.0 µs | n/a | 14.9 µs | identical |
| Isolation Forest | 2111 µs | 118 µs | 466 µs | identical |

ONNX thresholds were recalibrated on float32 scores; detection counts did not
change. Packed scores equal scikit-learn's to 1e-12 (tested).

**Interpretation.** For tiny linear models, the ONNX Runtime call overhead is as
large as the computation, so it brings portability, not speed. For Isolation
Forest, scikit-learn's per-call overhead dominates: the same trees cost 18x less
as packed arrays on this machine. Serialized size also depends on format:
1253 KiB pickle, 861 KiB ONNX, 592 KiB packed.

## EXP-004: Isolation Forest size sweep

| Trees | Faults detected | Event F1 | PR-AUC | Packed p50 | scikit-learn p50 | Packed size |
|---|---|---|---|---|---|---|
| 10 | 15/25 | 0.67 | 0.75 | 96 µs | 432 µs | 52 KiB |
| 25 | 12/25 | 0.63 | 0.76 | 101 µs | 634 µs | 148 KiB |
| 50 | 11/25 | 0.60 | 0.78 | 106 µs | 2085 µs | 296 KiB |
| 100 | 14/25 | 0.69 | 0.78 | 118 µs | 2111 µs | 592 KiB |
| 200 | 12/25 | 0.62 | 0.77 | 147 µs | 4172 µs | 1183 KiB |

**Interpretation.** More trees do not buy detection here: PR-AUC moves by 0.03
and event F1 is not monotonic. Model size grows linearly. Packed latency
grows slowly because the NumPy call overhead, not the tree walk, dominates at
these sizes. On this data, 10 trees detect as many faults as 100 (15 vs 14,
within noise) at one eleventh of the size.

## EXP-005: alert persistence ablation

| Model | Calibrated k | F1 / false alarms per h / delay | Fixed k = 3: F1 / false alarms per h / delay |
|---|---|---|---|
| Z-score | 10 | 0.41 / 2.64 / 51 s | 0.45 / 5.25 / 72 s |
| PCA | 10 | 0.73 / 2.64 / 76 s | 0.65 / 5.25 / 60 s |
| Autoencoder | 8 | 0.76 / 2.26 / 55 s | 0.77 / 3.79 / 42 s |
| Isolation Forest | 8 | 0.69 / 0.79 / 35 s | 0.64 / 10.47 / 60 s |

**Interpretation.** Label-free persistence halves the false alarms for z-score
and PCA and cuts Isolation Forest's by 13x. Mean delay does not rise
consistently: the two rules detect different sets of faults, and the mean is
taken over detected faults only, so the delays are not directly comparable.

## EXP-006: CPU and memory at the sensor rate

- **Configuration:** first 60 s of test seed 201, replayed paced at 1 kHz.
- **CPU:** process CPU seconds / wall seconds.

| Variant | CPU (% of one core) | Max lateness | Scoring heap peak |
|---|---|---|---|
| Z-score | 0.35 | 0.0 ms | 1.6 KiB |
| PCA | 0.44 | 0.0 ms | 2.1 KiB |
| Autoencoder | 0.36 | 0.0 ms | 2.2 KiB |
| Isolation Forest, scikit-learn | 1.73 | 0.0 ms | 286 KiB |
| Isolation Forest, packed | 0.45 | 0.0 ms | 8.6 KiB |
| Isolation Forest, ONNX Runtime | 0.76 | 2.0 ms | 1.5 KiB |

The heap peak is Python allocations traced by `tracemalloc`; native ONNX Runtime
memory is invisible to it. Process RSS was also recorded but is dominated by the
interpreter and allocator and is not reported as model memory. In the first,
heavily loaded run of this benchmark (superseded, its output was overwritten by
the rerun), one paced replay reached 49 ms of lateness; the rerun above peaked at
2 ms. Lateness is measured, not bounded.

**Interpretation.** At 1 kHz every variant needs under 2 % of one laptop core,
and the baseline cost of the loop (ingest, windowing, features) is most of it.

## EXP-007: sampling-rate sweep

- **Dataset:** SIMULATED, same seeds; everything regenerated and refitted per rate
  (window stays 1 s, so 250 to 2000 samples).
- **Isolation Forest scored with the packed runtime.**

| Rate | Z-score | PCA | Autoencoder | Isolation Forest | Feature p50 | Bearing faults (PCA) |
|---|---|---|---|---|---|---|
| 250 Hz | 2/25 (7 FA) | 12/25 (7 FA) | 21/25 (11 FA) | 9/25 (2 FA) | 40 µs | 3/5 |
| 500 Hz | 7/25 (7) | 17/25 (7) | 20/25 (7) | 12/25 (2) | 46 µs | 5/5 |
| 1000 Hz | 8/25 (7) | 18/25 (7) | 18/25 (6) | 14/25 (2) | 63 µs | 5/5 |
| 2000 Hz | 10/25 (7) | 18/25 (7) | 17/25 (7) | 18/25 (2) | 93 µs | 5/5 |

(FA = false alarms over the 5 runs.)

**Interpretation.** At 250 and 500 Hz the simulated 350 Hz bearing resonance is
above Nyquist and removed by the anti-aliasing filter. Bearing faults stay
partly visible through lower defect harmonics: PCA still finds 5/5 at 500 Hz but
only 3/5 at 250 Hz, z-score 4/5 and 1/5. The
autoencoder's high recall at 250 Hz comes with its highest false-alarm count.
Feature cost grows 2.3x for 8x the samples, so on this machine the sampling rate
is limited by the physics of the fault, not by compute. 1000 Hz results match
EXP-002 exactly (consistency check).

## EXP-008: MetroPT-3, pre-registered configuration (REAL data)

- **Dataset:** MetroPT-3, train 2020-02-01 to 03-15, calibration 03-15 to 04-01,
  test 04-01 to 09-01 (153 days, 4 air-leak reports).
- **Configuration:** 0.1 Hz grid, window 30 min, hop 5 min, 14 features, same
  detectors and calibration rules (persistence cap 12 windows = 1 h).
  Windows with more than 20 % missing samples skipped (2008 of the training windows).

| Model | Reports detected | Chance (p) | False alarms / day | Healthy time in alarm | ROC-AUC | PR-AUC | Inference p50 |
|---|---|---|---|---|---|---|---|
| Z-score | 4/4 | 1.23 (p = 0.006) | 0.18 | 16.4 % | 0.952 | 0.233 | 3 µs |
| PCA | 4/4 | 1.19 (p = 0.007) | 0.16 | 16.0 % | 0.959 | 0.253 | 7 µs |
| Autoencoder | 4/4 | 1.00 (p = 0.007) | 0.16 | 8.6 % | 0.938 | 0.184 | 10 µs |
| Isolation Forest | 1/4 | 0.16 (p = 0.157) | 0.01 | 0.9 % | 0.987 | 0.508 | 1723 µs (68 µs packed) |

Chance: 5000 circular shifts of the real alert timeline.

**Interpretation.** Three detectors flag every report well above chance, but
they are in alarm for 9 to 16 % of healthy time, concentrated in August, which no
operator would accept. The best ranking (Isolation Forest, ROC-AUC 0.987)
produced the worst alerting (1/4, not distinguishable from chance), because its
scores sit in a narrow band just under a static threshold. On real data, **ranking
quality and alerting quality diverge; calibration is the weak link.** A first
run before two code fixes (autoencoder iteration cap, alarm-time metric) gave
the same detection counts.

## EXP-009: MetroPT-3 without `oil_level_fraction` (POST-HOC)

- **Why post-hoc:** decided **after** EXP-008 showed that August alarms were
  driven by the `Oil_level` digital channel (training mean 1.0, August 0.2,
  median |z| ≈ 20) together with a +6 °C oil temperature. This is not a
  pre-registered result and must not be quoted as one.

| Model | Reports detected | Chance (p) | False alarms / day | Healthy time in alarm |
|---|---|---|---|---|
| Z-score | 4/4 | 0.66 (p = 0.002) | 0.10 | 4.5 % |
| PCA | 4/4 | 0.63 (p = 0.003) | 0.08 | 4.3 % |
| Autoencoder | 4/4 | 0.59 (p = 0.003) | 0.07 | 4.3 % |
| Isolation Forest | 2/4 | 0.18 (p = 0.011) | 0.04 | 0.9 % |

**Interpretation.** One channel explains about three quarters of the time in
alarm. Whether its change is drift or an unreported condition cannot be decided
from the data.

## EXP-010: hot-loop vs paced latency

- **Dataset:** SIMULATED dev seed, first 60 s; 3 repeats alternating unpaced and paced.
- **Load:** heavy background load during this run (1-minute load average ≈ 20), so
  absolute values are higher than in EXP-002. Only the ratio is of interest.

| Model | Unpaced inference p50 (3 repeats) | Paced inference p50 | Unpaced pipeline p50 | Paced pipeline p50 |
|---|---|---|---|---|
| Z-score | 7.7 / 10.3 / 6.3 µs | 16.2 / 15.8 / 16.2 µs | 177 / 235 / 148 µs | 375 / 367 / 370 µs |
| PCA | 25.1 / 19.5 / 19.0 µs | 38.5 / 38.6 / 38.1 µs | 255 / 197 / 191 µs | 399 / 394 / 386 µs |
| Autoencoder | 27.5 / 30.0 / 29.9 µs | 49.7 / 54.0 / 52.8 µs | 207 / 241 / 243 µs | 391 / 419 / 419 µs |

**Interpretation.** When the stream is paced, the CPU idles between chunks and
each call is 1.5 to 2.6x slower than in a hot loop, probably because of
frequency scaling and core migration (not verified). **The hot-loop numbers of
EXP-002 are optimistic for a real monitor that mostly waits.** A device
benchmark must be paced.
