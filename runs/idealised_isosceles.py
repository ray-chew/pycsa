"""Idealised isosceles-triangle CSA experiment.

Reference run used by the reproducibility suite. Generates synthetic terrain
from a fixed seed, runs six experiments through the CSA pipeline (reference,
pure LSFF, regularized LSFF, optimal CSA, sub-optimal CSA, pure LSFF on the
quadrilateral domain), and returns the spectra plus error metrics.

Numerics match the JAMES paper baseline:

* L2 errors    ≈ [0.0, 164291.57, 115.71, 85.68, 111.37, 164291.57]
* amplitudes   ≈ [1243.30, 1110972.58, 1861.67, 1243.32, 1146.83, 1110972.58]
* %errors      ≈ [0.0, 89256.997, 49.737, 0.002, 7.759, 89256.997]

Usage::

    # Programmatic
    from runs.idealised_isosceles import run
    result = run()

    # CLI
    python -m runs.idealised_isosceles --summary
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pycsa import var, utils, interface

EXPERIMENT_LABELS: tuple[str, ...] = (
    "reference",
    "pLSFF",
    "regLSFF",
    "optCSA",
    "subCSA",
    "pLSFF_quad",
)


@dataclass
class IdealisedResult:
    """Outputs of one full idealised isosceles run.

    Arrays are ordered by ``EXPERIMENT_LABELS``.
    """

    freqs_arr: np.ndarray  # (n_experiments, nhi, nhj)
    dat_arr: np.ndarray  # (n_experiments, H, W) physical-domain reconstructions
    errs: np.ndarray  # L2 error vs reference, per experiment
    sums: np.ndarray  # total amplitude, per experiment
    sum_errs: np.ndarray  # relative amplitude error vs reference
    freqs_ref: np.ndarray  # the reference spectrum (== freqs_arr[0])
    num_modes: int  # number of unique modes in the synthetic terrain


def _generate_terrain(seed: int = 777, sz: int = 25):
    """Deterministic synthetic-terrain spectral mode generator.

    Returns the (deduplicated) mode coordinates and per-mode amplitudes /
    phase selectors. The dedup means the final mode count is typically
    smaller than ``sz`` (22 for the default seed).
    """
    np.random.seed(seed)
    nk = np.random.randint(0, 12, size=sz)
    nl = np.random.randint(-5, 7, size=sz)

    for ii in range(sz):
        if nk[ii] == 0 and nl[ii] < 0:
            nk[ii] += np.random.randint(1, 11)

    pts = np.array(list(set(zip(nk, nl))))
    nk = pts[:, 0]
    nl = pts[:, 1]
    sz = len(pts)

    Ak = np.random.random(size=sz) * 100.0
    Al = np.random.random(size=sz) * 100.0
    sck = np.random.randint(0, 2, size=sz)
    scl = np.random.randint(0, 2, size=sz)  # captured for parity; unused below
    return pts, nk, nl, Ak, Al, sck, scl


def _build_cell(pts, nk, nl, Ak, Al, sck, nhi: int, nhj: int):
    grid = var.grid()
    cell = var.topo_cell()
    vid = utils.isosceles(grid, cell)
    lat_v = grid.clat_vertices[vid, :]
    lon_v = grid.clon_vertices[vid, :]
    cell.gen_mgrids()

    cell.topo = np.zeros_like(cell.lat_grid)
    sz = len(pts)
    for ii in range(sz):
        nk_scaled = 2.0 * np.pi * nk[ii] / cell.lon.max()
        nl_scaled = 2.0 * np.pi * nl[ii] / cell.lat.max()
        if sck[ii] == 0:
            bf = Ak[ii] * np.cos(nk_scaled * cell.lon_grid + nl_scaled * cell.lat_grid)
        else:
            bf = Al[ii] * np.sin(nk_scaled * cell.lon_grid + nl_scaled * cell.lat_grid)
        cell.topo += bf

    triangle = utils.gen_triangle(lon_v, lat_v)
    cell.get_masked(triangle=triangle)
    cell.wlat = np.diff(cell.lat).mean()
    cell.wlon = np.diff(cell.lon).mean()
    return cell, triangle


def run(
    *,
    seed: int = 777,
    nhi: int = 12,
    nhj: int = 12,
    n_modes: int = 14,
    lmbda_reg: float = 8.0e-5,
    lmbda_fg: float = 1.0e-1,
    lmbda_sg: float = 1.0e-6,
) -> IdealisedResult:
    """Run the idealised isosceles CSA experiment deterministically."""
    pts, nk, nl, Ak, Al, sck, _scl = _generate_terrain(seed=seed)
    sz = len(pts)

    freqs_ref = np.zeros((nhi, nhj))
    for cnt, (kk, ll) in enumerate(pts):
        freqs_ref[ll + 5, kk] = Ak[cnt]

    cell, triangle = _build_cell(pts, nk, nl, Ak, Al, sck, nhi, nhj)

    cell_quad = deepcopy(cell)
    cell_quad.get_masked(mask=np.ones_like(cell.topo).astype("bool"))

    U, V = 1.0, 1.0  # artificial winds, not used in the idealised test
    pure_lsff = interface.get_pmf(nhi, nhj, U, V)
    reg_lsff = interface.get_pmf(nhi, nhj, U, V)

    def csa_run(n: int) -> tuple[np.ndarray, np.ndarray]:
        first_guess = interface.get_pmf(nhi, nhj, U, V)
        cell.get_masked(mask=np.ones_like(cell.topo).astype("bool"))
        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()
        freqs_fg, _, _ = first_guess.sappx(cell, lmbda=lmbda_fg, iter_solve=False)

        fq_cpy = np.copy(freqs_fg)
        # Necessary: otherwise NaNs win the argmax race.
        fq_cpy[np.isnan(fq_cpy)] = 0.0

        indices = []
        for _ in range(n):
            mx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            indices.append(mx)
            fq_cpy[mx] = 0.0
        k_idxs = [p[1] for p in indices]
        l_idxs = [p[0] for p in indices]

        second_guess = interface.get_pmf(nhi, nhj, U, V)
        second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
        cell.get_masked(triangle=triangle)
        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()
        freqs, _, dat = second_guess.sappx(
            cell, lmbda=lmbda_sg, updt_analysis=True, scale=1.0, iter_solve=False
        )
        return freqs, dat

    num_experiments = 6
    freqs_arr = np.zeros((num_experiments, nhi, nhj))
    dat_list: list[np.ndarray] = [None] * num_experiments  # type: ignore[list-item]

    # Order matches the original script + JAMES baseline indexing.
    freqs_arr[0] = freqs_ref
    dat_list[0] = cell.topo * cell.mask  # original masked topography
    freqs_arr[1], _, dat_list[1] = pure_lsff.sappx(
        cell, lmbda=0.0, iter_solve=False, save_am=True
    )
    freqs_arr[5], _, dat_list[5] = pure_lsff.recompute_rhs(
        cell_quad, pure_lsff.fobj, save_coeffs=True
    )
    freqs_arr[2], _, dat_list[2] = reg_lsff.sappx(
        cell, lmbda=lmbda_reg, iter_solve=False
    )
    freqs_arr[3], dat_list[3] = csa_run(sz)
    freqs_arr[4], dat_list[4] = csa_run(n_modes)

    freqs_arr = np.array([np.nan_to_num(f) for f in freqs_arr])
    dat_arr = np.array([np.nan_to_num(np.asarray(d)) for d in dat_list])

    errs = np.array([np.linalg.norm(f - freqs_ref) for f in freqs_arr])
    sums = np.array([f.sum() for f in freqs_arr])
    sum_errs = np.array(
        [np.abs(f.sum() - freqs_arr[0].sum()) / freqs_arr[0].sum() for f in freqs_arr]
    )

    return IdealisedResult(
        freqs_arr=freqs_arr,
        dat_arr=dat_arr,
        errs=errs,
        sums=sums,
        sum_errs=sum_errs,
        freqs_ref=freqs_ref,
        num_modes=sz,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the idealised isosceles CSA experiment."
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--n-modes", type=int, default=14)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the numerical summary table after running (default).",
    )
    args = parser.parse_args(argv)

    result = run(seed=args.seed, n_modes=args.n_modes)

    np.set_printoptions(suppress=True)
    print(f"num_modes: {result.num_modes}")
    print(f"experiments: {EXPERIMENT_LABELS}")
    print(f"L2 errors:   {result.errs}")
    print(f"amplitudes:  {result.sums}")
    print(f"%errors:     {np.around(result.sum_errs, 5) * 100}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
