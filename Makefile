.PHONY: install test demo benchmark benchmark-quick sweep metropt figures dashboard reproduce

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

figures:
	uv run python scripts/make_figures.py

dashboard:
	uv run streamlit run src/dashboard/app.py

reproduce: test benchmark sweep metropt figures
