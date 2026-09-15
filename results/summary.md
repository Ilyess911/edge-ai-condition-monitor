### Simulated track: 5 unseen test runs (seeds 201-205), 25 injected faults, 1000 Hz

Machine: Apple M4, 10 logical CPUs, Python 3.12.13, one thread. Load average at start [2.79, 3.22, 6.34], at end [22.78, 21.82, 16.11] (other workloads were running).

| Model | Runtime | Faults detected | Event F1 | False alarms / h | Mean delay | Window PR-AUC | Inference p50 / p99 | Throughput | Size | Deployment class (rule, untested) |
|---|---|---|---|---|---|---|---|---|---|---|
| Z-score | NumPy / scikit-learn | 8/25 | 0.41 ± 0.13 | 2.64 | 51 s | 0.77 ± 0.08 | 3 / 3 µs | 3298 k samples/s | 0.4 KiB | MCU-portable (matrix ops, small); p99 pipeline 0.02 % of the 500 ms hop |
| PCA (SPE+T²) | NumPy / scikit-learn | 18/25 | 0.73 ± 0.15 | 2.64 | 76 s | 0.85 ± 0.03 | 7 / 9 µs | 3218 k samples/s | 1.0 KiB | MCU-portable (matrix ops, small); p99 pipeline 0.02 % of the 500 ms hop |
| Autoencoder | NumPy / scikit-learn | 18/25 | 0.76 ± 0.05 | 2.26 | 55 s | 0.86 ± 0.02 | 13 / 19 µs | 2331 k samples/s | 2.8 KiB | MCU-portable (matrix ops, small); p99 pipeline 0.03 % of the 500 ms hop |
| Isolation Forest | NumPy / scikit-learn | 14/25 | 0.69 ± 0.08 | 0.79 | 35 s | 0.78 ± 0.06 | 2111 / 8570 µs | 181 k samples/s | 1252.7 KiB | gateway only (framework overhead); p99 pipeline 1.81 % of the 500 ms hop |
| Isolation Forest | packed NumPy | 14/25 | 0.69 ± 0.08 | 0.79 | 35 s | 0.78 ± 0.06 | 118 / 261 µs | 1100 k samples/s | 591.6 KiB | SBC / gateway; p99 pipeline 0.12 % of the 500 ms hop |
| Z-score | ONNX Runtime | 8/25 | 0.41 ± 0.13 | 2.64 | 51 s | 0.77 ± 0.08 | 10 / 21 µs | 1579 k samples/s | 0.3 KiB | MCU-portable (matrix ops, small); p99 pipeline 0.04 % of the 500 ms hop |
| PCA (SPE+T²) | ONNX Runtime | 18/25 | 0.73 ± 0.15 | 2.64 | 76 s | 0.85 ± 0.03 | 8 / 13 µs | 2862 k samples/s | 1.0 KiB | MCU-portable (matrix ops, small); p99 pipeline 0.02 % of the 500 ms hop |
| Autoencoder | ONNX Runtime | 18/25 | 0.76 ± 0.05 | 2.26 | 55 s | 0.86 ± 0.02 | 15 / 81 µs | 1466 k samples/s | 1.7 KiB | MCU-portable (matrix ops, small); p99 pipeline 0.08 % of the 500 ms hop |
| Isolation Forest | ONNX Runtime | 14/25 | 0.69 ± 0.08 | 0.79 | 35 s | 0.78 ± 0.06 | 466 / 1133 µs | 698 k samples/s | 860.6 KiB | SBC / gateway; p99 pipeline 0.27 % of the 500 ms hop |

Deployment class is a stated rule, not a hardware result: matrix-operation or table models under 256 KiB serialized are marked MCU-portable, larger ones SBC / gateway, scikit-learn Isolation Forest gateway only. The budget column compares laptop p99 window pipeline latency (features + inference + alert) with the 500 ms hop.

#### Chance level and alarm time

| Model | Detected | Same alerts at random offsets | Healthy time in alarm | Persistence (windows) |
|---|---|---|---|---|
| Z-score | 8 | 4.2 | 0.18 % | 10 |
| PCA (SPE+T²) | 18 | 7.5 | 0.18 % | 10 |
| Autoencoder | 18 | 7.9 | 0.20 % | 8 |
| Isolation Forest | 14 | 5.6 | 0.05 % | 8 |

#### Detection by fault type (pooled over test runs)

| Model | bearing_outer_race | imbalance | cooling_failure | cavitation | electrical_overload |
|---|---|---|---|---|---|
| Z-score | 5/5 | 0/5 | 0/5 | 2/5 | 1/5 |
| PCA (SPE+T²) | 5/5 | 3/5 | 4/5 | 5/5 | 1/5 |
| Autoencoder | 5/5 | 4/5 | 3/5 | 5/5 | 1/5 |
| Isolation Forest | 5/5 | 3/5 | 1/5 | 5/5 | 0/5 |

#### Isolation Forest size sweep (packed runtime)

| Trees | Event F1 | Faults detected | PR-AUC | Inference p50 packed | p50 scikit-learn | Size packed |
|---|---|---|---|---|---|---|
| 10 | 0.67 | 15/25 | 0.75 | 96 µs | 432 µs | 52 KiB |
| 25 | 0.63 | 12/25 | 0.76 | 101 µs | 634 µs | 148 KiB |
| 50 | 0.60 | 11/25 | 0.78 | 106 µs | 2085 µs | 296 KiB |
| 100 | 0.69 | 14/25 | 0.78 | 118 µs | 2111 µs | 592 KiB |
| 200 | 0.62 | 12/25 | 0.77 | 147 µs | 4172 µs | 1183 KiB |

#### Ablation: fixed persistence (3 windows) vs calibrated

| Model | Calibrated k | Event F1 calibrated | False alarms / h | Delay | Event F1 k=3 | False alarms / h k=3 | Delay k=3 |
|---|---|---|---|---|---|---|---|
| Z-score | 10 | 0.41 | 2.64 | 51 s | 0.45 | 5.25 | 72 s |
| PCA (SPE+T²) | 10 | 0.73 | 2.64 | 76 s | 0.65 | 5.25 | 60 s |
| Autoencoder | 8 | 0.76 | 2.26 | 55 s | 0.77 | 3.79 | 42 s |
| Isolation Forest | 8 | 0.69 | 0.79 | 35 s | 0.64 | 10.47 | 60 s |

#### Resource use at the real sensor rate (paced 60 s, 1 kHz)

| Variant | CPU (% of one core) | Max lateness | Scoring heap peak | Parameters |
|---|---|---|---|---|
| zscore | 0.35 | 0.0 ms | 1.6 KiB | 24 |
| pca | 0.44 | 0.0 ms | 2.1 KiB | 78 |
| autoencoder | 0.36 | 0.0 ms | 2.2 KiB | 295 |
| iforest | 1.73 | 0.0 ms | 286.3 KiB | 35125 |
| iforest-packed | 0.45 | 0.0 ms | 8.6 KiB | 35125 |
| zscore-onnx | 0.40 | 0.0 ms | 16.7 KiB |  |
| pca-onnx | 0.38 | 0.0 ms | 1.0 KiB |  |
| autoencoder-onnx | 0.40 | 0.0 ms | 1.0 KiB |  |
| iforest-onnx | 0.76 | 2.0 ms | 1.5 KiB |  |

### Sampling-rate sweep (SIMULATED, same 5 test seeds)

| Rate | Model | Faults detected | False alarms | PR-AUC | Feature p50 | Real-time factor |
|---|---|---|---|---|---|---|
| 250 Hz | Z-score | 2/25 | 7 | 0.60 | 42 µs | 3635x |
| 250 Hz | PCA (SPE+T²) | 12/25 | 7 | 0.70 | 40 µs | 3653x |
| 250 Hz | Autoencoder | 21/25 | 11 | 0.79 | 41 µs | 3543x |
| 250 Hz | Isolation Forest | 9/25 | 2 | 0.68 | 41 µs | 2679x |
| 500 Hz | Z-score | 7/25 | 7 | 0.71 | 47 µs | 3683x |
| 500 Hz | PCA (SPE+T²) | 17/25 | 7 | 0.82 | 46 µs | 3702x |
| 500 Hz | Autoencoder | 20/25 | 7 | 0.85 | 46 µs | 3629x |
| 500 Hz | Isolation Forest | 12/25 | 2 | 0.75 | 47 µs | 2753x |
| 1000 Hz | Z-score | 8/25 | 7 | 0.77 | 62 µs | 3237x |
| 1000 Hz | PCA (SPE+T²) | 18/25 | 7 | 0.85 | 63 µs | 3054x |
| 1000 Hz | Autoencoder | 18/25 | 6 | 0.86 | 63 µs | 3081x |
| 1000 Hz | Isolation Forest | 14/25 | 2 | 0.78 | 63 µs | 2380x |
| 2000 Hz | Z-score | 10/25 | 7 | 0.77 | 93 µs | 2646x |
| 2000 Hz | PCA (SPE+T²) | 18/25 | 7 | 0.85 | 93 µs | 2595x |
| 2000 Hz | Autoencoder | 17/25 | 7 | 0.84 | 93 µs | 2556x |
| 2000 Hz | Isolation Forest | 18/25 | 2 | 0.77 | 93 µs | 2094x |

Bearing faults detected per rate:

| Model | 250 Hz | 500 Hz | 1000 Hz | 2000 Hz |
|---|---|---|---|---|
| Z-score | 1/5 | 4/5 | 5/5 | 5/5 |
| PCA (SPE+T²) | 3/5 | 5/5 | 5/5 | 5/5 |
| Autoencoder | 4/5 | 5/5 | 5/5 | 5/5 |
| Isolation Forest | 3/5 | 4/5 | 5/5 | 5/5 |

### MetroPT-3 (REAL data), pre-registered features

Test period 153 days, 10371 training windows, 4159 calibration windows.

| Model | Reports detected | Chance level (p) | Detected with 2 h early window | False alarms / day | Healthy time in alarm | ROC-AUC | PR-AUC | Inference p50 |
|---|---|---|---|---|---|---|---|---|
| Z-score | 4/4 | 1.23 (p = 0.006) | 4/4 | 0.18 | 16.4 % | 0.952 | 0.233 | 3 µs |
| PCA (SPE+T²) | 4/4 | 1.19 (p = 0.007) | 4/4 | 0.16 | 16.0 % | 0.959 | 0.253 | 7 µs |
| Autoencoder | 4/4 | 1.00 (p = 0.007) | 4/4 | 0.16 | 8.6 % | 0.938 | 0.184 | 10 µs |
| Isolation Forest | 1/4 | 0.16 (p = 0.157) | 1/4 | 0.01 | 0.9 % | 0.987 | 0.508 | 1723 µs |
| Isolation Forest (packed) | 1/4 | 0.16 (p = 0.157) | 1/4 | 0.01 | 0.9 % | 0.987 | 0.508 | 68 µs |

Delay per report (minutes from reported start to first alert):

| Model | #1 | #2 | #3 | #4 |
|---|---|---|---|---|
| Z-score | 75 | 40 | 45 | 215 |
| PCA (SPE+T²) | 80 | 45 | 50 | 55 |
| Autoencoder | 0 | 30 | 35 | 45 |
| Isolation Forest | missed | missed | 85 | missed |

### MetroPT-3 POST-HOC ablation: without oil_level_fraction

Test period 153 days, 10371 training windows, 4159 calibration windows.

| Model | Reports detected | Chance level (p) | Detected with 2 h early window | False alarms / day | Healthy time in alarm | ROC-AUC | PR-AUC | Inference p50 |
|---|---|---|---|---|---|---|---|---|
| Z-score | 4/4 | 0.66 (p = 0.002) | 4/4 | 0.10 | 4.5 % | 0.959 | 0.244 | 3 µs |
| PCA (SPE+T²) | 4/4 | 0.63 (p = 0.003) | 4/4 | 0.08 | 4.3 % | 0.959 | 0.254 | 9 µs |
| Autoencoder | 4/4 | 0.59 (p = 0.003) | 4/4 | 0.07 | 4.3 % | 0.958 | 0.246 | 10 µs |
| Isolation Forest | 2/4 | 0.18 (p = 0.011) | 2/4 | 0.04 | 0.9 % | 0.977 | 0.388 | 2031 µs |
| Isolation Forest (packed) | 2/4 | 0.18 (p = 0.011) | 2/4 | 0.04 | 0.9 % | 0.977 | 0.388 | 61 µs |

Delay per report (minutes from reported start to first alert):

| Model | #1 | #2 | #3 | #4 |
|---|---|---|---|---|
| Z-score | 75 | 40 | 45 | 215 |
| PCA (SPE+T²) | 80 | 45 | 50 | 55 |
| Autoencoder | 85 | 50 | 55 | 55 |
| Isolation Forest | missed | missed | 155 | 205 |

### Paderborn bearings (REAL), 6 healthy + 14 real-damage bearings, 3 folds

Machine: Apple M4, Python 3.12.13, one thread. Load average at start [0.96, 1.18, 1.55], at end [1.97, 3.11, 3.54]. The laptop slept (lid closed) during feature extraction; perf_counter latencies exclude sleep, total wall time does not.

| Model | Damaged recordings flagged | Healthy recordings flagged | Window ROC-AUC | Damaged bearings flagged on >50 % of recordings (A/B/C of 14) | Inference p50 | Full pipeline p50 / p99 | Real-time factor | Paced CPU |
|---|---|---|---|---|---|---|---|---|
| Z-score | 72 ± 5 % | 18 ± 5 % | 0.860 ± 0.024 | 9/9/13 | 3.2 µs | 1213 / 1297 µs | 176x | 4.2 % |
| PCA (SPE+T²) | 78 ± 15 % | 28 ± 14 % | 0.868 ± 0.062 | 11/9/14 | 7.5 µs | 1201 / 1980 µs | 176x | 3.9 % |
| Autoencoder | 78 ± 17 % | 35 ± 15 % | 0.860 ± 0.079 | 11/8/14 | 9.8 µs | 1179 / 1232 µs | 181x | 3.9 % |
| Isolation Forest (packed) | 35 ± 29 % | 7 ± 6 % | 0.795 ± 0.133 | 0/2/11 | 54.0 µs | 1221 / 2040 µs | 174x | 4.0 % |

Isolation Forest through scikit-learn: inference p50 1706 µs, pipeline p50 2880 µs, real-time factor 83x. Sizes: zscore 0.5 KiB, pca 1.4 KiB, iforest 1510.6 KiB, iforest-packed 688.1 KiB, autoencoder 3.3 KiB.
Max lateness in paced replay: zscore 48.8 ms, pca 14.3 ms, iforest 23.2 ms, iforest-packed 14.7 ms, autoencoder 68.7 ms.

#### Sampling-rate sweep (mean of 3 folds)

| Rate | Feature p50 / p99 | Z-score detected / false | PCA (SPE+T²) detected / false | Autoencoder detected / false | Isolation Forest detected / false |
|---|---|---|---|---|---|
| 64 kHz | 1213 / 3933 µs | 72 % / 18 % | 78 % / 28 % | 78 % / 35 % | 35 % / 7 % |
| 32 kHz | 1101 / 2707 µs | 61 % / 9 % | 71 % / 21 % | 75 % / 32 % | 38 % / 5 % |
| 16 kHz | 498 / 1333 µs | 53 % / 8 % | 65 % / 12 % | 66 % / 20 % | 39 % / 4 % |
| 8 kHz | 191 / 337 µs | 42 % / 4 % | 51 % / 7 % | 58 % / 6 % | 36 % / 2 % |
| 4 kHz | 120 / 186 µs | 42 % / 3 % | 44 % / 5 % | 56 % / 22 % | 34 % / 0 % |
| 2 kHz | 87 / 143 µs | 51 % / 2 % | 52 % / 9 % | 59 % / 17 % | 7 % / 0 % |

#### Feature-group ablation at 64 kHz (mean of 3 folds)

| Features | PCA detected | PCA false | PCA ROC-AUC | IForest detected | IForest false | IForest ROC-AUC |
|---|---|---|---|---|---|---|
| all 15 | 78 % | 28 % | 0.868 | 35 % | 7 % | 0.795 |
| no_envelope | 70 % | 24 % | 0.842 | 34 % | 10 % | 0.792 |
| envelope_context | 60 % | 11 % | 0.842 | 56 % | 9 % | 0.848 |

#### Per bearing at 64 kHz: % of 80 recordings flagged (PCA)

| Bearing | Damage (from profile PDF) | All features | Envelope + context |
|---|---|---|---|
| K001 | healthy | 0 % | 0 % |
| K002 | healthy | 91 % | 0 % |
| K003 | healthy | 0 % | 0 % |
| K004 | healthy | 24 % | 0 % |
| K005 | healthy | 49 % | 65 % |
| K006 | healthy | 5 % | 0 % |
| KA04 | OR, combination S, extent 1 | 100 % | 100 % |
| KA15 | OR, combination S, extent 1 | 30 % | 23 % |
| KA16 | OR OR, combination R R, extent 2 2 | 100 % | 100 % |
| KA22 | OR, combination S, extent 1 | 53 % | 9 % |
| KA30 | OR, combination R, extent 1 | 59 % | 40 % |
| KB23 | IR IR OR, combination R R R, extent 1 2 2 | 100 % | 100 % |
| KB24 | IR OR, combination M M, extent 3 n/a | 100 % | 42 % |
| KB27 | OR IR, combination M M, extent 1 1 | 47 % | 65 % |
| KI04 | IR OR, combination M M, extent 1 n/a | 70 % | 83 % |
| KI14 | IR OR, combination M M, extent 1 n/a | 69 % | 28 % |
| KI16 | IR, combination S, extent 3 | 100 % | 71 % |
| KI17 | IR IR, combination R R, extent 1 1 | 98 % | 41 % |
| KI18 | IR, combination S, extent 2 | 100 % | 92 % |
| KI21 | IR, combination S, extent 1 | 66 % | 42 % |
