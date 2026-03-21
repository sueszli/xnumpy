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


community speed benchmark for karpathy's microgpt. a minimal gpt trained on a names dataset,
one forward + backward pass per step. ref: https://karpathy.github.io/2026/02/12/microgpt/

beat each other's times! to submit: add a .py file prefixed with your username and open a pr.
one rule: any python library is welcome, but no other language embedded within the file.


results
-------

EOF
uv run utils.py | tee >(perl -pe 's/\e\[[0-9;]*m//g; s/^/    /' >> README)
