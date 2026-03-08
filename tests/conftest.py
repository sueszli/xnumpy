from __future__ import annotations

import csv
import fcntl
import gc as _gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from exo.API import Procedure
from plotnine import aes, coord_flip, element_text, geom_col, ggplot, labs, scale_fill_manual, theme, theme_minimal

from xnumpy.backends import Backend, compile_and_load

# xDSL IRDL holds raw ctypes pointers. GC finalizer ordering -> dangling ptr -> segfault
_gc.disable()
_gc.set_threshold(0)
_gc.enable = lambda: None
_gc.collect = lambda *a, **kw: 0

_JSONL = Path(tempfile.gettempdir()) / "bench" / "timings.jsonl"


def pytest_sessionstart(session):
    if not hasattr(session.config, "workerinput"):
        _JSONL.parent.mkdir(exist_ok=True)
        _JSONL.unlink(missing_ok=True)


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput") or not _JSONL.exists():
        return
    rows = [json.loads(line) for line in _JSONL.read_text().splitlines() if line.strip()]
    if not rows:
        return
    rows.sort(key=lambda r: r["kernel"])

    e2e = Path(__file__).resolve().parent / "e2e"
    fields = ["kernel", "exo_c", "xdsl_mlir", "jit"]
    csv_out = e2e / "e2e_test_times.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    backends = fields[1:]
    df = pl.read_csv(csv_out)
    df = df.group_by("kernel").agg(*(pl.col(b).mean() for b in backends))
    long = df.unpivot(index="kernel", on=backends, variable_name="backend", value_name="time_s").sort("kernel")
    pdf = long.to_pandas()
    colors = {"exo_c": "#4e79a7", "xdsl_mlir": "#f28e2b", "jit": "#e15759"}
    p = ggplot(pdf, aes(x="kernel", y="time_s", fill="backend")) + geom_col(position="dodge") + coord_flip() + scale_fill_manual(values=colors) + labs(x="", y="time (s)", fill="backend", title="e2e test runtimes") + theme_minimal() + theme(figure_size=(10, max(6, df.height * 0.22))) + theme(axis_text_y=element_text(size=7)) + theme(legend_position="bottom")
    p.save(e2e / "e2e_test_times.pdf", verbose=False)


def assert_match(proc: Procedure, **kwargs: Any) -> None:
    results: dict[Backend, dict[str, np.ndarray]] = {}
    times: dict[Backend, float] = {}

    for backend in Backend:
        fn = compile_and_load(proc, backend)
        t0 = time.perf_counter()
        results[backend] = fn(**kwargs)
        times[backend] = time.perf_counter() - t0

    keys = {Backend.EXO_C: "exo_c", Backend.MLIR: "xdsl_mlir", Backend.JIT: "jit"}
    _JSONL.parent.mkdir(exist_ok=True)
    row = json.dumps({"kernel": proc._loopir_proc.name} | {keys[b]: t for b, t in times.items()})
    with open(_JSONL, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    ref = results[Backend.EXO_C]
    for backend in (Backend.MLIR, Backend.JIT):
        for name in ref:
            np.testing.assert_allclose(results[backend][name], ref[name], atol=1e-6, err_msg=f"{backend.name.lower()} mismatch on buffer '{name}'")
