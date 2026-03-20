# /// script
# requires-python = "==3.14.*"
# dependencies = ["termgraph"]
# ///

import csv
import inspect
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMES_DIR = ROOT / "times"


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

    entries.sort(key=lambda x: x[1], reverse=True)
    for name, _ in entries:
        print_times(TIMES_DIR / f"{name}.csv")

    chart_data = Data([[mean_ms * 1000] for _, mean_ms in entries], [name for name, _ in entries])
    chart_args = Args(title="mean train step time [\u03bcs]", width=60, format="{:.0f}", space_between=True)
    BarChart(chart_data, chart_args).draw()


def save_times(step_times: list[float]) -> None:
    name = Path(inspect.stack()[1].filename).stem
    path = TIMES_DIR / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_ms"])
        writer.writerows([[i + 1, f"{t * 1000:.3f}"] for i, t in enumerate(step_times)])
    print_times(path)


if __name__ == "__main__":
    print_times_all()
