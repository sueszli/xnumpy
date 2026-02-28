.PHONY: install
install:
	brew install llvm pkg-config ninja ccache

.PHONY: venv
venv:
	uv sync

.PHONY: precommit
precommit:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
	uvx ruff check --fix --ignore F403,F405,F821 .

.PHONY: tests
tests:
	uv run pytest -W ignore tests/
	uv run lit tests/filecheck/
