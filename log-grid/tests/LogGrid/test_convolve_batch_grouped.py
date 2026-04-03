import os
import sys

import pytest

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)

from pyloggrid.Libs.datasci import randcomplex_like
from pyloggrid.LogGrid.Grid import Grid


def make_grid(*, D: int = 3, N: int = 5, k0: bool = True, n_threads: int = 2) -> Grid:
    return Grid(
        D=D,
        l_params={"a": 1, "b": 2, "plastic": False},
        N_points=N,
        fields_name=[],
        k0=k0,
        n_threads=n_threads,
    )


@pytest.mark.parametrize("D", [1, 2, 3])
def test_grouped_kernel_row_ptr_covers_flat_kernel(D):
    grid = make_grid(D=D, n_threads=2)
    maths = grid.maths

    assert maths.convolution_row_ptr.dtype == np.uint32
    assert maths.convolution_row_ptr[0] == 0
    assert maths.convolution_row_ptr[-1] == maths.convolution_kernel.size
    assert maths.convolution_row_ptr.size == grid.ks_modulus.size + 1
    assert np.all(maths.convolution_row_ptr[1:] >= maths.convolution_row_ptr[:-1])


@pytest.mark.parametrize("n_threads", [1, 2, 4])
@pytest.mark.parametrize("n_batch", [1, 2, 5])
def test_grouped_batch_matches_scalar_convolve(n_threads, n_batch):
    grid = make_grid(D=3, N=4, k0=True, n_threads=n_threads)
    fgs = [
        (
            grid.maths.enforce_grid_symmetry_arr(randcomplex_like(grid.ks_modulus)),
            grid.maths.enforce_grid_symmetry_arr(randcomplex_like(grid.ks_modulus)),
        )
        for _ in range(n_batch)
    ]
    expected = np.asarray([grid.maths.convolve(f, g) for f, g in fgs])
    got = np.asarray(grid.maths._convolve_batch_grouped(fgs))
    assert np.allclose(got, expected)


def test_threaded_batch_dispatch_prefers_grouped_path():
    grid = make_grid(D=3, N=4, k0=True, n_threads=2)
    assert grid.maths.convolve_batch.__func__ is grid.maths._convolve_batch_grouped.__func__
    assert hasattr(grid.maths, "_convolve_batch_list_legacy")
