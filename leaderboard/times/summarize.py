import csv
import math
from pathlib import Path


def calculate_stats(times):
    if not times:
        return None
    n = len(times)
    mean = sum(times) / n
    variance = sum((x - mean) ** 2 for x in times) / (n - 1) if n > 1 else 0
    stddev = math.sqrt(variance)
    return {"mean": mean, "stddev": stddev, "min": min(times), "max": max(times), "runs": n}


def get_results():
    base_dir = Path(__file__).parent
    files = sorted(base_dir.glob("*.csv"))
    results = []
    for file_path in files:
        with file_path.open("r") as f:
            reader = csv.DictReader(f)
            times = [float(row["time_ms"]) for row in reader]
            stats = calculate_stats(times)
            if not stats:
                continue
            results.append((file_path.stem, stats))
    return results


def main():
    results = get_results()
    if not results:
        return

    results.sort(key=lambda x: x[1]["mean"])

    for name, stats in results:
        print(f"'{name}'")
        print(f"  Time (mean \u00b1 \u03c3):    {stats['mean']:8.1f} ms \u00b1 {stats['stddev']:8.1f} ms    [User: 0.0 ms, System: 0.0 ms]")
        print(f"  Range (min \u2026 max):  {stats['min']:8.1f} ms \u2026 {stats['max']:8.1f} ms    {stats['runs']} runs")
        print()

    if len(results) <= 1:
        return

    # Summary section
    sorted_results = sorted(results, key=lambda x: x[1]["mean"])
    fastest_name, fastest_stats = sorted_results[0]

    print("Summary")
    print(f"  '{fastest_name}' ran")
    for name, stats in sorted_results[1:]:
        ratio = stats["mean"] / fastest_stats["mean"]
        rel_err_fastest = fastest_stats["stddev"] / fastest_stats["mean"] if fastest_stats["mean"] > 0 else 0
        rel_err_slower = stats["stddev"] / stats["mean"] if stats["mean"] > 0 else 0
        ratio_stddev = ratio * math.sqrt(rel_err_fastest**2 + rel_err_slower**2)
        print(f"    {ratio:6.2f} \u00b1 {ratio_stddev:6.2f} times faster than '{name}'")


if __name__ == "__main__":
    main()
