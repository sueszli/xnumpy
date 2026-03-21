# /// script
# requires-python = "==3.14.*"
# dependencies = ["termgraph"]
# ///

import csv
import inspect
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights.json"
TIMES_DIR = ROOT / "times"


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


def save_times(step_times: list[float]) -> None:
    name = Path(inspect.stack()[1].filename).stem
    path = TIMES_DIR / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_ms"])
        writer.writerows([[i + 1, f"{t * 1000:.3f}"] for i, t in enumerate(step_times)])
    print_times(path)


def print_times(path: Path) -> None:
    if not path.exists():
        return
    with open(path, "r") as f:
        times = [float(row["time_ms"]) * 1000 for row in csv.DictReader(f)]
    if not times:
        return
    n = len(times)
    mean = sum(times) / n
    variance = sum((x - mean) ** 2 for x in times) / (n - 1) if n > 1 else 0
    stddev = math.sqrt(variance)
    print(f"'{path.stem}'")
    print(f"  Time (mean \u00b1 \u03c3):    {mean:8.0f} \u03bcs \u00b1 {stddev:8.0f} \u03bcs    [User: 0 \u03bcs, System: 0 \u03bcs]")
    print(f"  Range (min \u2026 max):  {min(times):8.0f} \u03bcs \u2026 {max(times):8.0f} \u03bcs    {n} runs")
    print()


def print_times_all() -> None:
    from termgraph.args import Args
    from termgraph.chart import BarChart
    from termgraph.data import Data

    if not TIMES_DIR.exists():
        return

    entries = []
    for path in sorted(TIMES_DIR.glob("*.csv")):
        with open(path) as f:
            times = [float(row["time_ms"]) for row in csv.DictReader(f)]
        if times:
            entries.append((path.stem, sum(times) / len(times)))
    if not entries:
        return
    entries.sort(key=lambda x: x[1], reverse=False)

    baseline_ms = max(mean for _, mean in entries)
    speedups = [(name, baseline_ms / mean) for name, mean in entries]
    speedup_data = Data([[s] for _, s in speedups], [name for name, _ in speedups])
    speedup_args = Args(title="speedup over baseline", width=80, format="{:.0f}x", space_between=True)
    BarChart(speedup_data, speedup_args).draw()

    print()
    print("–" * 100, "\n")

    for name, _ in entries:
        print_times(TIMES_DIR / f"{name}.csv")


if __name__ == "__main__":
    print_times_all()
