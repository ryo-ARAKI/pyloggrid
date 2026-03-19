"""
Short tutorial run derived from NS3D.py.
"""

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)

from pyloggrid.Libs.custom_logger import setup_custom_logger
from pyloggrid.Libs.datasci import randcomplex_like, randcomplex_seeded_by_array
from pyloggrid.LogGrid.Framework import Grid, Solver

logger = setup_custom_logger(__name__)

f0, fx, fy, fz = 1, None, None, None
N_points = 4


def get_forcing(grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global fx, fy, fz
    if fx is None:
        f_rot = randcomplex_seeded_by_array(grid.ks, 1337)
        f = grid.maths.rot3D_inv(f_rot)
        f[:, grid.ks_modulus > grid.k_min * grid.l**3] = 0
        f = f / np.sqrt(grid.maths.self_inner_product(f))
        f = f * f0
        fx, fy, fz = f
    return fx, fy, fz


def equation_nonlinear(_t: float, grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
    M = grid.maths
    ux, uy, uz = grid.field("ux", "uy", "uz")
    uxdxux, uydyux, uxdxuy, uydyuy, uzdzux, uzdzuy, uxdxuz, uydyuz, uzdzuz = M.convolve_batch(
        (
            (ux, M.dx * ux),
            (uy, M.dy * ux),
            (ux, M.dx * uy),
            (uy, M.dy * uy),
            (uz, M.dz * ux),
            (uz, M.dz * uy),
            (ux, M.dx * uz),
            (uy, M.dy * uz),
            (uz, M.dz * uz),
        )
    )

    fx, fy, fz = get_forcing(grid)
    dux_dt = -uxdxux - uydyux - uzdzux + fx
    duy_dt = -uxdxuy - uydyuy - uzdzuy + fy
    duz_dt = -uxdxuz - uydyuz - uzdzuz + fz
    dux_dt, duy_dt, duz_dt = grid.maths.P_projector([dux_dt, duy_dt, duz_dt])
    return {"ux": dux_dt, "uy": duy_dt, "uz": duz_dt}


def equation_linear(_t: float, grid: Grid, simu_params: dict) -> dict[str, np.ndarray]:
    Re_F = simu_params["Re_F"]
    nu = np.sqrt(f0) * grid.L**1.5 / Re_F
    visc = grid.maths.laplacian * nu
    return {"ux": visc, "uy": visc, "uz": visc}


def initial_conditions(fields: dict[str, np.ndarray], grid: Grid, _simu_params: dict) -> dict[str, np.ndarray]:
    grid = grid.to_new_size_empty(N_points)
    u = get_forcing(grid)
    randu = grid.maths.rot3D_inv(randcomplex_like(np.array(u)))
    u = u + randu * 1e-200
    ux, uy, uz = u
    fields["ux"] = ux
    fields["uy"] = uy
    fields["uz"] = uz
    return fields


def update_gridsize(grid: Grid) -> int | None:
    global fx
    E = grid.physics.energy()
    ux, uy, uz = grid.field("ux", "uy", "uz")
    mask = grid.ks_modulus > grid.k_min * grid.l ** (grid.N_points - 1)
    comp = np.max(np.abs(ux[mask]) + np.abs(uy[mask]) + np.abs(uz[mask]))
    if comp / np.sqrt(E) > 1e-100:
        fx = None
        return grid.N_points + 1
    if comp / np.sqrt(E) < 1e-170 and grid.N_points > 4:
        fx = None
        return grid.N_points - 1
    return None


if __name__ == "__main__":
    logger.info("Running short NS3D tutorial simulation")
    solver = Solver(
        fields_names=["ux", "uy", "uz"],
        equation_nl=equation_nonlinear,
        equation_l=equation_linear,
        D=3,
        l_params={"plastic": False, "a": 1, "b": 2},
        simu_params={"Re_F": 1e3},
        n_threads=1,
    )
    solver.solve(
        solver_params={"rtol": 1e-4},
        save_path="results/tutorial_ns3d_quick",
        save_one_in=1,
        end_simulation={"elapsed_time": 8, "ode_step": 200},
        update_gridsize_cb=update_gridsize,
        initial_conditions=initial_conditions,
        solver="ViscDopri",
    )
