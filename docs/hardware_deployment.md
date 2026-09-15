# Hardware deployment path

> **Status: PROPOSED, not executed.** No part of this repository has run on a
> Raspberry Pi, a Jetson, a microcontroller or an industrial gateway. Every
> measurement comes from a local benchmark on a laptop (Apple M4, macOS,
> Python 3.12, single-threaded runtimes). Statements about devices below are
> expectations to verify, not results.

## 1. Implemented vs proposed

| Topic | IMPLEMENTED (in this repository, measured on the laptop) | PROPOSED (not done) |
|---|---|---|
| Streaming | chunked ingestion, fixed-size ring buffer, one window scored at a time | same code on a Pi; C port for an MCU |
| Serialization | pickle (development), ONNX float32 for all four detectors, flat arrays for the forest; parity tests | C header export of PCA matrices and forest tables |
| Runtime | NumPy, scikit-learn, ONNX Runtime CPU, all pinned to one thread | ONNX Runtime on aarch64; TensorRT only if a neural model is added |
| Quantization | none | int8 autoencoder, int16 forest thresholds, threshold recalibration |
| Memory | scoring heap peak (tracemalloc), serialized size | process RSS on the device, including the Python runtime |
| CPU | CPU seconds / wall seconds during a 60 s paced replay at 1 kHz | same measurement on the device, over hours, with thermal state |
| Power | not measured | USB power meter, idle vs paced run |
| Sensor interface | replay from memory (simulated arrays, MetroPT-3 CSV) | SPI accelerometer or DAQ, OPC UA / Modbus / MQTT inputs |
| Edge-cloud | none | MQTT upload of alerts and feature summaries |

## 2. What would need to change, by topic

### Serialization

- Pickle is a development convenience and unsafe to load from untrusted sources.
  Deployment would ship the ONNX graphs (`src/inference/onnx_export.py`) or, for
  an MCU, the raw arrays.
- The benchmark already checks that ONNX and packed scores match native scores
  and recalibrates the threshold for float32. That check must be repeated on the
  target, since float behaviour can differ across CPUs and runtimes.

### Quantization

- z-score and PCA: a few hundred float32 values. Quantizing saves bytes that
  do not matter.
- Autoencoder: int8 weights would shrink 295 parameters further and enable
  integer-only inference on an MCU or NPU.
- Isolation Forest: thresholds as int16 after fixed-point feature scaling.
- Required afterwards: recalibrate threshold and persistence on the quantized
  model, then rerun detection metrics. Quantization can preserve ranking while
  shifting the score distribution.

### Dependencies

- Laptop stack: NumPy, SciPy, scikit-learn, pandas, psutil, ONNX Runtime.
- Minimum to run inference: NumPy + ONNX Runtime (or NumPy only with packed
  arrays). scikit-learn, SciPy and pandas are only needed for training, the
  simulator and MetroPT-3 loading.
- MCU: no Python at all; CMSIS-DSP for the FFT features.

### Sampling

- Real bearing vibration (Paderborn) is recorded at 64 kHz. Measured on the laptop
  core, feature extraction for a 0.5 s window costs 1213 µs p50 at 64 kHz, 498 µs at
  16 kHz and 191 µs at 8 kHz, while model inference stays at 3-54 µs (packed
  forest). On a Raspberry-class CPU the feature stage is therefore the first thing
  to measure, and the first thing to port to C (CMSIS-DSP or a fixed-point FFT).
- The sampling rate is a detection decision as much as a cost one: lowering it from
  64 to 8 kHz cut PCA's false alarms from 28 % to 7 % but also its detections from
  78 % to 51 % of damaged recordings (EXP-012).
- Measured full pipeline at 64 kHz on the laptop: 170 to 180 times faster than real
  time for all models except scikit-learn Isolation Forest (83x), about 4 % of one
  core at the sensor rate. That margin, not a device measurement, is why a Pi-class
  device is plausible.
- Slow channels (temperature, pressure) need 1-10 Hz and could be decimated
  before the ring buffer.

### Memory

- Per-stream state: one window. On Paderborn at 64 kHz that is 32,000 x 5 float64 =
  1.2 MiB, dominated by two high-rate channels; float32 halves it. At 1 kHz on the
  simulator it is 40 KiB.
- Models: from 24 numbers (z-score) to about 590 KiB (100-tree packed forest).
- Unmeasured and probably dominant on a Pi: the Python interpreter, NumPy and
  ONNX Runtime themselves (tens of MB).
- MCU: the window buffer, not the model, is the first RAM constraint at kHz rates.

### Power

- Not measured anywhere in this repository.
- Minimum credible measurement: USB power meter on the device, idle baseline vs
  a paced 1-hour replay, per detector.
- The laptop CPU share during paced replay (below 2 % of one core for every
  variant) suggests acquisition and the idle floor would dominate energy, but
  that is an expectation.

### Sensor interface

- Current input is an in-memory array replayed in 50 ms chunks. The chunk size
  was chosen to mimic a DAQ driver handing over a buffer.
- Real inputs: an SPI/I²C MEMS accelerometer with a FIFO read by a C driver or
  a DAQ, or process values from a PLC over OPC UA or Modbus TCP.
- New failure modes to handle: clock drift between channels, reordered or
  duplicated packets, and a stuck sensor that keeps sending a plausible constant.
  The cleaner currently handles only out-of-range values and short gaps.

## 3. Target-specific notes

### Raspberry Pi 4 / 5

Run `scripts/run_benchmark.py` unchanged, plus `vcgencmd measure_temp` and the
throttling flags during a paced run of at least one hour. Question to answer:
does p99 window pipeline latency stay far below the 500 ms hop under thermal
throttling?

### NVIDIA Jetson Orin Nano

For these four models a GPU is not expected to help: a single feature vector
per window is too small to amortise a host-to-device copy. This is an
expectation; the Jetson becomes relevant for heavier models (spectrogram CNNs,
multi-sensor fusion).

### Industrial gateway

Container with engine, ONNX model and TOML configuration. Inputs at 1-100 Hz from
a PLC. The concerns shift from compute to watchdogs, versioned updates, rollback
and time synchronisation.

## 4. Edge-cloud split (proposed)

- **Edge:** acquisition, features, scoring, alerting. Alerts must not depend
  on the network.
- **To the cloud:** alerts, feature summaries (15 float64 every 0.25 s is 480 B/s,
  vs 1 MB/s for vibration and current at 64 kHz as float64), and short raw
  snippets around alerts.
- **From the cloud:** retrained and recalibrated models, versioned, with
  rollback. This is where drift, observed on MetroPT-3 in August, would be
  handled.
