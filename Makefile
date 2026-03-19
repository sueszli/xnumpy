.PHONY: venv
venv:
	uv sync
	uv pip install -e .
	@echo "\nActivate with: source .venv/bin/activate"

.PHONY: tests
tests:
	uv run pytest -W ignore tests/
	uv run lit -j $$(nproc 2>/dev/null || sysctl -n hw.logicalcpu) tests/filecheck/

.PHONY: fmt
fmt:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
	uvx ruff check --fix --ignore F403,F405,F821,E731,E402 .
	find . -not -path './.venv/*' \( -name "*.c" -o -name "*.h" \) | xargs clang-format -i

.PHONY: benchmark
benchmark:
	uv run python benchmarks/run.py
	chmod +x leaderboard/run.sh && ./leaderboard/run.sh

.PHONY: precommit
precommit: fmt tests benchmark
