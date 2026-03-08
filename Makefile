.PHONY: venv
venv:
	uv sync

.PHONY: tests
tests:
	uv run pytest -W ignore tests/
	uv run lit -j $$(nproc 2>/dev/null || sysctl -n hw.logicalcpu) tests/filecheck/

.PHONY: fmt
fmt:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
	uvx ruff check --fix --ignore F403,F405,F821,E731 .

.PHONY: benchmark
benchmark:
	uv run python benchmarks/run.py

.PHONY: precommit
precommit: fmt tests benchmark
