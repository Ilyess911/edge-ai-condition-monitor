.PHONY: install test demo benchmark benchmark-quick sweep metropt paderborn figures summary dashboard reproduce

install:
	uv sync --all-extras

test:
	uv run pytest

demo:
	uv run python scripts/stream_demo.py

benchmark:
	uv run python scripts/run_benchmark.py

benchmark-quick:
	uv run python scripts/run_benchmark.py --quick

sweep:
	uv run python scripts/sampling_rate_sweep.py

metropt:
	uv run python scripts/download_metropt.py
	uv run python scripts/run_metropt.py
	uv run python scripts/run_metropt.py --drop-features oil_level_fraction

paderborn:
	uv run python scripts/download_paderborn.py
	uv run python scripts/run_paderborn.py

figures:
	uv run python scripts/make_figures.py

summary:
	uv run python scripts/summarize_results.py > results/summary.md

dashboard:
	uv run streamlit run src/dashboard/app.py

reproduce: test paderborn metropt benchmark sweep figures summary
