import csv
import gc as _gc
import json
import tempfile
from pathlib import Path

# xDSL IRDL holds raw ctypes pointers. GC finalizer ordering -> dangling ptr -> segfault
_gc.disable()
_gc.set_threshold(0)
_gc.enable = lambda: None
_gc.collect = lambda *a, **kw: 0


#
# benchmark e2e tests
#

_BENCH_DIR = Path(tempfile.gettempdir()) / "xdsl_exo_bench"
_BENCH_JSONL = _BENCH_DIR / "timings.jsonl"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BENCH_CSV_OUT = _PROJECT_ROOT / "benchmark.csv"
_BENCH_PDF_OUT = _PROJECT_ROOT / "benchmark.pdf"


def pytest_sessionstart(session):
    if not hasattr(session.config, "workerinput"):  # controller only
        _BENCH_DIR.mkdir(exist_ok=True)
        _BENCH_JSONL.unlink(missing_ok=True)


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):
        return  # workers skip
    if not _BENCH_JSONL.exists():
        return
    rows: list[dict] = []
    for line in _BENCH_JSONL.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return
    rows.sort(key=lambda r: r["kernel"])
    with open(_BENCH_CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "exo_c", "xdsl_mlir", "jit"])
        w.writeheader()
        w.writerows(rows)
    _generate_pdf(_BENCH_CSV_OUT, _BENCH_PDF_OUT)


def _generate_pdf(csv_path: Path, pdf_path: Path) -> None:
    import polars as pl
    from plotnine import aes, coord_flip, element_text, geom_col, ggplot, labs, scale_fill_manual, theme, theme_minimal

    df = pl.read_csv(csv_path)
    df = df.group_by("kernel").agg(
        pl.col("exo_c").mean(),
        pl.col("xdsl_mlir").mean(),
        pl.col("jit").mean(),
    )
    long = df.unpivot(index="kernel", on=["exo_c", "xdsl_mlir", "jit"], variable_name="backend", value_name="time_s").sort("kernel")

    pdf = long.to_pandas()
    colors = {"exo_c": "#4e79a7", "xdsl_mlir": "#f28e2b", "jit": "#e15759"}
    n_kernels = df.height

    p = ggplot(pdf, aes(x="kernel", y="time_s", fill="backend")) + geom_col(position="dodge") + coord_flip() + scale_fill_manual(values=colors) + labs(x="", y="time (s)", fill="backend", title="kernel benchmark") + theme_minimal() + theme(figure_size=(10, max(6, n_kernels * 0.22))) + theme(axis_text_y=element_text(size=7))
    p.save(pdf_path, verbose=False)
