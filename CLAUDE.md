# CLAUDE.md

Research prototype: streaming condition monitoring with lightweight anomaly
detection and measured edge cost. Public repository, English everywhere.

## Commands

```bash
uv sync --all-extras                                  # install
uv run pytest                                         # tests (~10 s)
uv run python scripts/stream_demo.py --speed 8        # terminal demo
uv run python scripts/run_benchmark.py --quick        # dev seed only, ~5 min
make reproduce                                        # every result and figure, ~45 min
```

## Engineering rules (non-negotiable)

1. **Never fabricate a measurement.** Every number in README or docs comes from
   `results/*.json` produced by a script. No typed-in latency, no extrapolated
   device figures.
2. **Simulated data is labelled SIMULATED** in every figure title, table and
   sentence that uses it.
3. **Test seeds (201-205) and the MetroPT-3 test period are for reporting only.**
   Debug on `dev_seed = 101`. A configuration change after seeing test results
   is a new, explicitly post-hoc experiment in `docs/experiments.md`.
4. **Training and test features go through the same `StreamingEngine`.** No
   offline shortcut for evaluation; `tests/test_streaming.py` enforces equality.
5. **Thresholds and persistence are calibrated on healthy data only**, never on
   fault labels.
6. **A control tests one specific failure.** Chance-level detection is tested by
   circular shift; ONNX and packed runtimes by score parity; streaming by
   offline equivalence. Adding a new claim means adding its control.
7. **Benchmarks run single-threaded** (`scripts/_threads.py` imported first) and
   record machine load. Say "laptop", never "edge device", for current numbers.
8. No em dashes as sentence connectors in prose.

## Definition of done

Tests pass, the affected script was rerun, figures regenerated, numbers in
README / `docs/experiments.md` match the JSON, JOURNAL.md has an entry.
