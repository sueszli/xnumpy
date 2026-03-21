#!/usr/bin/env bash
set -euox pipefail

cd "$(dirname "$0")"

[ -f weights.json ] || uv run original.py

for file in *.py; do
  [[ "$file" == "utils.py" || "$file" == "original.py" ]] || uv run "$file"
done

rm README
toilet -t -f pagga "microgpt benchmarks game" >> README
cat >> README << 'EOF'


community inference benchmarks game for karpathy's microgpt.
a minimal gpt trained on a names dataset: https://karpathy.github.io/2026/02/12/microgpt/

can you beat me? to submit: add a .py file prefixed with your username and open a pr.
any python library is welcome, but no other language embedded within the file.

EOF
uv run utils.py | tee >(perl -pe 's/\e\[[0-9;]*m//g' >> README)
