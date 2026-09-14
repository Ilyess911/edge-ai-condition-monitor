# From prototype to a 9-12 week research project

## Starting point (what exists)

A streaming pipeline with measured latency, four one-class detectors with
label-free calibration, a simulated track with exact labels and a real track
(MetroPT-3) with four failure reports. Its two most interesting open problems
came out of the experiments, not out of a wish list:

1. **Calibration, not ranking, is the bottleneck.** On MetroPT-3 the detectors
   rank failure windows well (ROC-AUC 0.94 to 0.99) yet disagree completely on
   what to alert on, because the threshold and persistence are set on a short
   calibration period.
2. **Drift is indistinguishable from an unreported fault.** In August 2020 the
   MetroPT-3 scores shift for weeks, driven by the `Oil_level` channel and a
   seasonal oil-temperature rise, with no failure report to explain it.

## Proposed question

> How can an edge anomaly detector keep a controlled false-alarm rate under
> operating-condition drift, with a compute and memory budget small enough for
> a microcontroller or a gateway core, and without fault labels?

## Plan

| Weeks | Work package | Output |
|---|---|---|
| 1-2 | **Real sensors.** Accelerometer (e.g. ADXL355 or IIS3DWB) + current clamp on a small motor test bench or a lab fan with controllable imbalance; acquisition at 10+ kHz on a Raspberry Pi / STM32. Or, without a bench, CWRU / Paderborn bearing data replayed at native rate. | real kHz track replacing the simulator for vibration |
| 3-4 | **On-device benchmark.** Run the existing benchmark on a Raspberry Pi 5 and a Cortex-M class board: latency, CPU, RSS, power (USB meter), thermal throttling over 24 h. | the laptop-only limitation removed |
| 5-6 | **TinyML and quantization.** Port features (CMSIS-DSP FFT) and the PCA / packed-forest scorers to C; int8 autoencoder via ONNX Runtime quantization; measure detection loss vs flash, RAM and energy per inference. | quality-vs-cost curve down to MCU scale |
| 7-8 | **Online learning and concept drift.** Streaming PCA (incremental covariance with forgetting factor), Half-Space Trees, ADWIN / Page-Hinkley on the score distribution; label-free threshold tracking (conformal prediction on a sliding calibration buffer, extreme value theory tail fit). Evaluated on MetroPT-3 August and on injected drift in the simulator. | false-alarm rate under drift, with and without adaptation |
| 9-10 | **Edge-cloud collaboration.** Edge sends feature summaries and alert snippets over MQTT; cloud retrains and pushes versioned models; measure bandwidth, update latency, behaviour when the link drops. | reference architecture with measured bandwidth |
| 11 | **Federated learning (scoped).** Several simulated machines with different operating envelopes train a shared PCA / autoencoder with federated averaging; compare to local-only and centralised models. Feasible in simulation within a week; out of scope on hardware. | whether sharing helps a machine with little healthy data |
| 12 | Write-up, reproducibility package, workshop-paper draft. | report + tagged release |

## Industrial IoT angle

Multimodal fusion follows naturally from the channels already present:
vibration localises mechanical faults, current covers electrical ones, and the
simulator shows that `electrical_overload` is unobservable without a load or
torque measurement. A study of *which sensor subset is sufficient for which
fault class, at which cost* is directly useful to anyone specifying a retrofit
monitoring kit.

## Risks

- **No test bench available:** fall back to public bearing datasets at native
  rates (CWRU 12/48 kHz, Paderborn 64 kHz); lose the power measurements.
- **MetroPT-3 has only four reports:** every result keeps a chance-level
  control; a second real dataset with more events (e.g. SKAB, Tennessee
  Eastman for process data) should be added early.
- **Online adaptation can learn the fault as normal:** adaptation must be gated
  (freeze updates while alerting), and this failure mode must be measured,
  not assumed away.
