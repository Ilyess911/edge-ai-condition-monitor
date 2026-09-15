# Journal

One entry per working session. Written on the day, never back-filled.

## 2026-09-14 — Session 1: from empty folder to measured benchmark

### Objective

Build a complete, defensible Edge AI condition-monitoring prototype: sensor
stream, edge preprocessing, features, lightweight detectors, online scoring,
alerting, and measured computational cost, in a public repository.

### Work completed

- Repository, `uv` project, MIT licence, public GitHub repository, CI.
- Simulated rotating machine (5 channels, 5 fault types, anti-aliased ADC stage,
  labelled-normal sensor artefacts).
- Streaming engine: chunked ingestion, plausibility check, sample-and-hold,
  circular-buffer windows, per-stage `perf_counter_ns` timing, paced and unpaced
  replay.
- Four detectors (z-score, PCA SPE+T², Isolation Forest, autoencoder),
  label-free threshold and persistence calibration, alert engine with hysteresis.
- ONNX export of all four (parity-tested) and a packed-array Isolation Forest
  runtime (exact parity with scikit-learn).
- MetroPT-3 real-data track: checksum-verified download, 10 s grid alignment,
  duty-cycle features, chronological split, failure reports transcribed.
- Benchmarks: simulated (5 unseen test runs), sampling-rate sweep, MetroPT-3,
  MetroPT-3 post-hoc ablation. Figures, dashboard, terminal demo, docs.

### Technical decisions

- **Two data tracks.** A simulator gives exact labels and kHz vibration; real
  data keeps the simulator honest. Neither alone was enough.
- **One code path.** Training features are produced by the same streaming engine
  as test features (tested for equality with an offline reference).
- **Calibration without labels.** Threshold = 99.5th percentile of scores on a
  held-out healthy run. Persistence = longest healthy excursion + 1 window.
- **Dev seed vs test seeds.** Seed 101 was used for all debugging; seeds 201-205
  were first evaluated in the final benchmark run.
- **Single-threaded measurements** to approximate a shared edge core and avoid
  flattering BLAS parallelism.
- **Autoencoder kept** only as the non-linear counterpart of PCA, trained with
  scikit-learn to avoid a deep-learning dependency.

### Experiments

EXP-001 to EXP-009 in `docs/experiments.md`.

### Results

See `results/summary.md` and the README. In short: on the simulator, PCA and the
autoencoder detect 18/25 faults, Isolation Forest 14/25 with the fewest false
alarms, z-score 8/25. On MetroPT-3, three detectors flag all four air-leak
reports (p ≈ 0.007 against chance) but spend 9-16 % of healthy time in alarm.

### Problems encountered

1. **Simulator too short for its fault schedule** (20-minute runs could not fit
   six faults). Runs lengthened to 60 minutes with five faults.
2. **False alarms on the development run came from load/speed ramps**, not
   noise: every false alarm on seed 101 started during a 5 s ramp
   (`current_std` dominated). Fixes, all decided on the dev seed: longer
   training (30 to 60 min) for regime coverage, and label-free persistence
   calibration. Dev-seed false alarms fell from 5-7 to 1-2 per run.
3. **Thermal aftermath counted as false alarms.** Temperature keeps relaxing
   after a cooling fault ends. Added a 180 s recovery exclusion (1.5 τ), applied
   identically to all detectors.
4. **PCA exploded in float32 ONNX** when all components were kept (SPE = rounding
   noise). PCA now keeps at least one residual direction.
5. **scikit-learn Isolation Forest costs milliseconds per window** because it
   loops over trees in Python. Wrote the packed runtime.
6. **MetroPT-3 documentation contradicts itself** (1 Hz vs 0.1 Hz). The file
   says 10 s. One maintenance date precedes its failure. Both documented.
7. **The "alarm time" metric was wrong**: sampled on a 60 s grid, it reported
   0 % in alarm next to 2.6 false alarms per hour. Replaced by exact interval
   overlap, test added, benchmark rerun.
8. **ONNX Runtime aborts during interpreter teardown on macOS**, after results
   are written, which stopped the benchmark chain. The script now exits
   explicitly.
9. **The machine was shared.** Two other projects used up to ~8 cores during
   the first benchmark run (load average up to 24). Load is now recorded in
   every result file, and the simulated benchmark was rerun when load was lower.

### Lessons learned

- A false-alarm count hides a week-long alarm. Time in alarm must be reported
  next to it. The MetroPT-3 detectors looked fine (0.16 false alarms per day)
  until time in alarm showed 16 %.
- With four events, "4/4 detected" means little without a chance level.
- For single-window inference, runtime overhead dominates algorithm cost: the
  same trees cost milliseconds in scikit-learn and around 0.1 ms as packed
  arrays.
- A sanity check that disagrees with another metric (0 % in alarm, yet false
  alarms) is a bug report, not a curiosity.

### Next steps

1. Run the unchanged benchmark on a Raspberry Pi 5 with a power meter.
2. Online threshold adaptation under drift (conformal / EVT) evaluated on
   MetroPT-3 August.
3. Replace the simulated vibration with a real bearing dataset at native rate
   (CWRU or Paderborn).

## 2026-09-14 — Session 2: measurement audit and documentation cleanup

### Objective

Review every claim against the result files, tighten the benchmark methodology,
and remove wording the measurements do not support.

### Work completed

- Final benchmark rerun (simulated track, sampling sweep, MetroPT-3, post-hoc
  ablation), figures regenerated, `results/summary.md` generated from the JSON.
- New EXP-010: hot-loop vs paced latency.
- `docs/experiments.md` and README rewritten from the
  generated tables; `docs/hardware_deployment.md` split into implemented vs
  proposed.
- Code cleanup: removed unused helpers (`microbench`, an unused property and
  attributes), merged a duplicated null detector, unified the Isolation Forest
  parameter count between runtimes.
- Removed development-seed traces that had been committed before
  `results/traces/` was ignored.

### Technical decisions

- "Real-time" replaced by "online inference simulation" or "paced replay"
  wherever no timing guarantee exists.
- The "edge suitability" column became "deployment class (rule, untested)",
  with the rule written next to the table.
- Serialized size, parameters, heap peak and CPU share are reported; process RSS
  is recorded but not presented as model memory.

### Experiments

EXP-010 (paced vs unpaced, 3 repeats, 3 models).

### Results

- Paced replay made each call 1.5 to 2.6x slower than the hot loop.
- Feature extraction, identical code in every variant, varied from 62 to 140 µs
  p50 across variants: the noise floor under background load is about 2x.
- scikit-learn Isolation Forest: p99 8.6 ms, maximum 362 ms.

### Problems encountered

1. Background load from other workloads rose again during the rerun (1-minute
   load average 2.8 at start, 22.8 at end). Recorded, not controllable.
2. Several documentation statements were wrong or unsupported and were
   corrected: feature summaries are 192 B/s, not 2 kB/s; the 500 Hz sweep point
   is already above the simulated resonance's Nyquist limit; the MetroPT-3
   interval range is 9 to 13 s for 99.97 % of samples, not "8 to 13 s"; the
   "25x" Isolation Forest speed-up became the measured 18x.
3. The first paced replay with 49 ms lateness belongs to a superseded run whose
   output was overwritten.

### Lessons learned

- Latency measured in a hot loop is not the latency of a monitor that waits for
  data. A device benchmark must be paced.
- When identical code varies 2x between runs, the benchmark ranks orders of
  magnitude, not percentages.

### Next steps

Unchanged from session 1, with one addition: run EXP-010 on a quiet machine and on
the target device before quoting any absolute latency.

## 2026-09-15 — Session 3: real bearing vibration (Paderborn)

### Objective

Replace the simulator as the main evidence with real kHz vibration data.

### Work completed

- Rejected two candidate datasets after inspecting them: a 500-row CSV whose
  columns are statistically independent (position autocorrelation near 0,
  uniform distributions, target column identical to an input column), and an
  IEEE DataPort set described as simulated, sampled every 10 minutes, a few KB.
- Adopted the Paderborn University (KAt) bearing data: read the measuring log and
  damage profiles in the archives, downloaded 6 healthy and 14 real-damage
  bearings (Zenodo mirror, MD5-verified; six archives confirmed byte-identical
  to the university server), SHA-256 recorded.
- Loader reading `.mat` files straight from the RAR archives, envelope-analysis
  features, damage-profile parser, bearing-held-out folds, sampling-rate sweep,
  feature-group ablation, full-engine timing at 64 kHz, figures, docs.

### Technical decisions

- **Split by bearing.** Recordings of one bearing share its mounting; a random
  split by recording would let the model recognise the bearing.
- **Recording-level decision** (flagged if any alert in 4 s), because bearings
  are healthy or damaged for the whole recording: no onset to time.
- **Real-damage bearings only**; artificial damage left out.
- **Features before results:** the 15 features were fixed from bearing physics and
  a check on two bearings before any detector was evaluated.

### Experiments

EXP-011 to EXP-014 in `docs/experiments.md`.

### Results

- 64 kHz, PCA: 78 % of damaged recordings flagged, 28 % of healthy ones.
- Envelope + context features only: PCA false alarms 11 %, Isolation Forest
  detection 35 → 56 %.
- Feature extraction 1213 µs per window at 64 kHz vs 7.5 µs PCA inference;
  176x faster than real time on a laptop core.

### Problems encountered

1. Kurtosis, the textbook impulsiveness indicator, is higher on several healthy
   bearings (up to 14.8) than on the damaged KA04 (5.6): mounting effects.
2. The university server throttled to about 70 kB/s per connection after the
   first files; the Python downloader stalled at 100 kB/s. Switched to curl and
   the Zenodo mirror with checksum verification.
3. Two archives arrived corrupted or truncated (KI04 after a resumed transfer,
   KI14 cut at 98 MB); the MD5 check stopped the pipeline both times, and both were
   downloaded again.
4. The damage-profile parser missed the extent of three multi-damage bearings
   ("1 n/a"); fixed, parser moved to a tested pure function, metadata re-parsed.
5. The laptop went to sleep with the lid closed during feature extraction.
   Latencies are unaffected; the reported total run time (3.5 h) is not
   meaningful.

### Lessons learned

- On real machines, the healthy population is the difficult class.
- A feature tied to physics (defect frequency from geometry and speed) is worth
  more for false alarms than any change of model.
- At high sampling rates the edge budget is spent on signal processing, not on
  the model.

### Next steps

1. Explain healthy K005 (flagged on 65 % of recordings by envelope features).
2. Per-bearing calibration on a short healthy segment of each new machine.
3. Profile and port the feature stage (FFT, envelope) to C before any device run.
