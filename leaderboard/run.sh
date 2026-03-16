#!/usr/bin/env bash
set -euox pipefail

cd "$(dirname "$0")"

[[ ! -f weights.json ]] && uv run original.py

uv run vanilla.py
uv run numpygpt.py
uv run torchgpt.py
uv run jaxgpt.py
