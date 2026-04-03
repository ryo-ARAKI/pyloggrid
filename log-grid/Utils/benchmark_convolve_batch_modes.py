from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(1, str(ROOT))

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)

from pyloggrid.LogGrid.Grid import Grid


def run_once(*, n_threads: int, n_batch: int, N_points: int, D: int, mode: str) -> float:
    grid = Grid(D=D, l_params={"plastic": False, "a": 1, "b": 2}, N_points=N_points, fields_name=[], n_threads=n_threads, k0=False)
    fgs = [(np.random.randn(*grid.shape).astype(complex), np.random.randn(*grid.shape).astype(complex)) for _ in range(n_batch)]
    fn = grid.maths._convolve_batch_grouped if mode == "grouped" else grid.maths._convolve_batch_list_legacy
    t0 = time.perf_counter()
    fn(fgs)
    return time.perf_counter() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, nargs="+", required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--dim", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for mode in ("legacy", "grouped"):
        for n_threads in args.threads:
            for n_batch in args.batches:
                rows.append(
                    {
                        "mode": mode,
                        "n_threads": n_threads,
                        "n_batch": n_batch,
                        "elapsed": run_once(n_threads=n_threads, n_batch=n_batch, N_points=args.n_points, D=args.dim, mode=mode),
                    }
                )
    print(json.dumps(rows, indent=2))
