# Hardware deployment path

> **Status: not deployed on any edge hardware.** Every measurement in this
> repository comes from a development laptop (Apple M4, macOS, one pinned
> thread, see `environment` in `results/*.json`). This page is a plan, and each
> statement about a device below is an expectation to verify, not a result.

## What already makes the pipeline deployable

| Property | Where | Why it matters on a device |
|---|---|---|
| Fixed memory per stream | `SlidingWindow` circular buffer, `SampleCleaner` state | no growth over weeks of uptime |
| One window scored at a time, bounded cost | `StreamingEngine.process_chunk` | worst case known in advance |
| No deep learning framework | NumPy, scikit-learn only for training | small image, no GPU needed |
| Models as plain arrays | z-score (24 numbers), PCA matrices, autoencoder weights, packed forest arrays | portable to C or a microcontroller |
| ONNX export with parity tests | `src/inference/onnx_export.py`, `tests/test_onnx.py` | one runtime across Pi, Jetson, x86 gateway |
| Single-threaded measurements | `scripts/_threads.py` | closer to a core shared with acquisition |

## Target 1: Raspberry Pi 4 / 5 (Cortex-A72 / A76, 4-8 GB)

- **Runtime.** Python 3.11+ with NumPy and `onnxruntime` (aarch64 wheels exist).
  scikit-learn is only needed if the model is retrained on the device.
- **Expected change.** Per-call latency will be higher than on the laptop; the
  magnitude must be measured, not extrapolated. The one design question that
  matters: does p99 pipeline latency stay far below the 500 ms hop? On the laptop
  the heaviest variant is below 10 ms, so the margin is large, but thermal
  throttling of a fanless Pi under continuous load has to be observed over hours.
- **Acquisition.** An accelerometer at 1 kHz or more needs a real ADC front-end
  (e.g. an SPI MEMS accelerometer or an I²S/SPI ADC with DMA-like buffering),
  not Python polling. The engine already consumes 50 ms chunks for that reason.
- **What to measure first:** `scripts/run_benchmark.py` unchanged, plus
  `vcgencmd measure_temp` and throttling flags during a paced 1-hour run.

## Target 2: NVIDIA Jetson Orin Nano

- The GPU brings nothing for these models: a 12-feature PCA or a 100-tree forest
  is cheaper on one CPU core than a host-to-GPU copy. The Jetson becomes
  relevant only for heavy models (spectrogram CNNs, multi-sensor fusion) listed
  in `docs/research_extension.md`.
- ONNX Runtime with the CPU provider first; TensorRT only if a neural model is
  introduced and profiled.

## Target 3: industrial edge gateway (x86 or ARM, Linux, often containerised)

- Deliver as a container: engine + ONNX model + TOML config.
- Inputs usually come from a PLC or DAQ over OPC UA, Modbus TCP or MQTT rather
  than from a raw ADC; at those rates (1-100 Hz) the cost of this pipeline is
  negligible and the constraints become determinism, watchdogs and updates.

## Microcontroller (TinyML) path

The z-score and PCA detectors need a few hundred floats and matrix products; the
packed forest is five small integer/float tables walked with a loop. Both fit in
C on a Cortex-M4F. The costly part on an MCU is the feature stage (a 1024-point
FFT per window), which CMSIS-DSP provides. Not implemented here.

## Model serialization

| Format | Used for | Measured size |
|---|---|---|
| pickle (scikit-learn / NumPy objects) | development only; unsafe to load from untrusted sources | `cost.serialized_kib` in results |
| ONNX (float32) | portable deployment | `serialized_kib` of `*-onnx` variants |
| flat arrays (packed forest) | C header / MCU | `serialized_kib` of `*-packed` variants |

## Quantization

- Not applied. The four detectors are already small; for z-score and PCA the
  float32 ONNX graphs are the practical minimum.
- Where it would matter: the autoencoder (int8 weights via ONNX Runtime dynamic
  quantization) and forest thresholds (int16 fixed-point after feature scaling).
  In both cases the calibrated threshold must be recomputed on the quantized
  model, exactly as `run_benchmark.py` already does for float32 ONNX.

## Memory constraints

The dominant memory is not the model but the runtime: the Python interpreter
plus NumPy is tens of MB, ONNX Runtime adds more. The per-stream state is one
window (1000 x 5 float64 = 40 KiB at 1 kHz). On a gateway monitoring many
machines, state scales linearly with the number of streams while models can be
shared.

## Sampling rates

`results/sampling_rate_sweep.json` shows how detection and cost change from 250
to 2000 Hz on the simulator. On real bearings, impacts excite kHz resonances, so
a vibration channel would need 10 kHz or more (or envelope analysis in analog
front-end hardware), which multiplies the feature cost per window by 10; slow
channels stay at 1-10 Hz.

## Power consumption

Not measured. A USB power meter on the Pi during a paced run, compared with an
idle baseline, is the minimum credible measurement. Duty cycling (acquire,
compute, sleep) is the main lever for battery-powered nodes.

## Connectivity and edge-cloud split

- **On the edge:** acquisition, features, scoring, alerting. Alerts must not
  depend on the network.
- **To the cloud:** alerts, periodic feature summaries (12 floats every 0.5 s is
  about 2 kB/s before compression, vs 40 kB/s of raw vibration), and short raw
  snippets around alerts for diagnosis.
- **From the cloud:** retrained models and recalibrated thresholds, versioned,
  with a rollback path. Retraining is where drift (seen on MetroPT-3 in August)
  is handled.
- Transport: MQTT with store-and-forward for intermittent links.
