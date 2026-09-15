# Roadmap

Checked = implemented, run and committed. Unchecked = not done; nothing here is
claimed before it exists.

## Phase 1 — Streaming Engine

- [x] Simulated 5-channel rotating machine with anti-aliased ADC stage
- [x] Five fault types with random severity, labelled sensor artefacts
- [x] Chunked ingestion, plausibility check, sample-and-hold with bounded hold
- [x] Circular-buffer sliding windows, chunk-size invariant (tested)
- [x] Paced (real-time) and unpaced (max throughput) replay
- [x] Configurable sampling frequency, window, hop, chunk (TOML)

## Phase 2 — Baselines

- [x] Z-score alarm limits
- [x] PCA with combined SPE + T² index
- [x] Isolation Forest, with a tree-count sweep
- [x] Small autoencoder, justified as the non-linear counterpart of PCA
- [x] Real-data track: MetroPT-3 with checksum-verified download and failure reports
- [x] Real vibration track: Paderborn bearings (6 healthy, 14 real damage), 64 kHz, split by bearing
- [x] Envelope-analysis features at bearing defect frequencies

## Phase 3 — Real-Time Inference (online inference simulation, soft real-time)

- [x] One window scored at a time, per-stage latency timing
- [x] Label-free threshold calibration on held-out healthy data
- [x] Label-free alert persistence calibration, hysteresis
- [x] ONNX export of all four detectors, parity-tested
- [x] Packed-array Isolation Forest runtime, exact parity with scikit-learn
- [ ] Online (incremental) model update with drift detection
- [ ] Per-machine calibration for bearing-to-bearing variation

## Phase 4 — Edge Benchmarking

- [x] Latency p50/p95/p99, throughput, real-time factor
- [x] CPU utilisation at the sensor rate, lateness behind the sample clock
- [x] Serialized size, parameter count, scoring heap peak, load RSS delta
- [x] Window- and event-level detection metrics over 5 unseen test runs
- [x] Chance-level control for event detection (circular shift)
- [x] Sampling-rate sweep 250-2000 Hz (simulated) and 2-64 kHz (real bearings)
- [x] Feature-group ablation on real bearings
- [x] Full pipeline timing and paced CPU at 64 kHz
- [x] Fixed-persistence ablation
- [x] Paced vs hot-loop latency check (EXP-010)
- [ ] Measurements on a quiet machine (other workloads were running; load recorded)
- [ ] Measurements on real edge hardware (Raspberry Pi, Jetson, MCU)
- [ ] Power measurement

## Phase 5 — Visualization

- [x] Architecture diagram (figure + Mermaid)
- [x] Sensor stream, vibration signature
- [x] Anomaly score timeline with alerts
- [x] Model comparison and per-fault detection matrix
- [x] Latency distribution
- [x] Detection vs latency trade-off with Pareto frontier
- [x] Sampling-rate trade-off
- [x] MetroPT-3 timeline
- [x] Optional Streamlit dashboard, replaying real Paderborn bearings

## Phase 6 — Deployment Path

- [x] `docs/hardware_deployment.md` (plan, explicitly not deployed)
- [x] Single-threaded benchmark configuration
- [ ] Container image for a gateway
- [ ] C port of features and PCA / packed-forest scoring
- [ ] int8 quantization study

## Phase 7 — Release

- [x] README with measured results and limitations
- [x] `docs/experiments.md` experiment log
- [x] `docs/research_extension.md`
- [x] Critical methodology review applied (JOURNAL.md, session 2)
- [x] CI running the test suite
