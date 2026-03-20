import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "weights.json"


def dump_weights(state_dict) -> None:
    WEIGHTS_PATH.write_text(json.dumps({k: [[v.data for v in row] for row in mat] for k, mat in state_dict.items()}))


def assert_weights_match(state_dict, atol: float = 1e-5) -> None:
    assert WEIGHTS_PATH.exists(), f"weights file not found: {WEIGHTS_PATH}"

    ref = json.load(open(WEIGHTS_PATH))
    cur = {k: [[v.data for v in row] for row in mat] for k, mat in state_dict.items()}
    assert set(ref) == set(cur), f"key mismatch: ref={set(ref)-set(cur)} cur={set(cur)-set(ref)}"

    for k in ref:
        assert len(ref[k]) == len(cur[k]) and len(ref[k][0]) == len(cur[k][0]), f"shape mismatch '{k}': {len(ref[k])}x{len(ref[k][0])} vs {len(cur[k])}x{len(cur[k][0])}"

    max_diff = 0.0
    max_loc = ""
    violations = 0
    total = 0
    rows = ((k, i, rr, cr) for k in ref for i, (rr, cr) in enumerate(zip(ref[k], cur[k])))
    all_cells = itertools.chain.from_iterable(((k, i, j, r, c) for j, (r, c) in enumerate(zip(rr, cr))) for k, i, rr, cr in rows)
    for k, i, j, r, c in all_cells:
        d = abs(r - c)
        total += 1
        violations += d > atol
        if d <= max_diff:
            continue
        max_diff, max_loc = d, f"{k}[{i}][{j}]"
    assert violations == 0, f"weights mismatch (atol={atol}): {violations}/{total} params exceed tolerance, max diff={max_diff:.2e} at {max_loc}"
