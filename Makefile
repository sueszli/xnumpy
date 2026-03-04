.PHONY: venv
venv:
	uv sync

.PHONY: tests
tests:
	uv run pytest -W ignore tests/
	uv run lit tests/filecheck/

.PHONY: precommit
precommit: tests
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
	uvx ruff check --fix --ignore F403,F405,F821 .
	find . -name "*.c" -o -name "*.h" | xargs clang-format -i
