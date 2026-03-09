from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from kernels.matmul import matmul
from kernels.matmul_neon import matmul_neon

REPEATS = 200


bench = lambda fn: min(timeit.repeat(fn, number=1, repeat=REPEATS))

np.random.seed(42)
rows: list[dict[str, object]] = []


#
# matmul
#


matmul_sizes = [32, 64, 128, 256]


for n in tqdm(matmul_sizes, desc="matmul"):
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    expected = A @ B
    flops = 2 * n**3

    # numpy
    t_np = bench(lambda A=A, B=B: A @ B)

    # exo auto-vectorized
    fn_exo = matmul(n, n, n)
    C_exo = np.zeros((n, n), dtype=np.float32)
    fn_exo(C_exo, A, B)
    assert np.allclose(C_exo, expected, atol=1e-3)
    t_exo = bench(lambda C=C_exo, A=A, B=B: fn_exo(C, A, B))

    # exo NEON intrinsics
    fn_neon = matmul_neon(n, n, n)
    C_neon = np.zeros((n, n), dtype=np.float32)
    fn_neon(C_neon, A, B)
    assert np.allclose(C_neon, expected, atol=1e-3)
    t_neon = bench(lambda C=C_neon, A=A, B=B: fn_neon(C, A, B))

    rows.append(
        {
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 1),
            "exo_gflops": round(flops / t_exo / 1e9, 1),
            "neon_gflops": round(flops / t_neon / 1e9, 1),
        }
    )


if __name__ == "__main__":
    df = pl.DataFrame(rows)
    print(df)
    df.write_csv(Path(__file__).parent / "results.csv")
