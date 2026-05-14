"""Minimal real-data CSA example — Aleutians (ICON cell 2311, MERIT).

A reviewer or new user can clone the repo and run::

    python examples/icon_regional_minimal.py

…to see the full CSA pipeline (load topography → first approximation →
mode selection → second approximation → reconstruction) on a real, but
tiny, ICON+MERIT case. Total runtime ~10 s on a laptop; the example
ships with all the data it needs (``examples/data/`` is ~260 KB).

The cell is the same one pinned by the reproducibility suite's
``regional_merit`` fixture — chosen because it covers a stretch of the
Aleutian arc (real topography, not ocean) and exercises the
false-positive-dateline branch of the MERIT loader (lon vertices all
negative near -180° but span < 180°).

Numerics here are computed live, not pinned. The reproducibility suite
(``tests/reproducibility/``) is the place to gate against numerical
drift; this script is for human inspection.
"""

from __future__ import annotations

# Import netCDF4 BEFORE pycsa to avoid a libhdf5 init-order quirk —
# importing pycsa.core.io first can leave the HDF5 layer in a state that
# fails subsequent `nc.Dataset(...)` opens with "NetCDF: HDF error".
# Pytest doesn't hit this because conftest.py imports cartopy/numpy
# upstream, which changes the order. Belt-and-braces.
import netCDF4 as _nc  # noqa: F401

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycsa.compute import ComputeContext
from pycsa.config.params import params as Params
from pycsa.core import io as pcio, utils, physics
from pycsa.data.cell import grid as Grid, topo_cell as TopoCell
from pycsa.data.cell import topo as Topo
from pycsa.wrappers import interface

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
ICON_GRID = DATA_DIR / "icon_grid.nc"
MERIT_DIR = DATA_DIR / "merit"

# Same numerics as the regional_merit reproducibility fixture so users
# can reach for that for a tighter, automated check.
NHI, NHJ = 24, 48
N_MODES = 50
LMBDA_SG = 1e-1
U, V = 10.0, 0.0
PADDING = 10


def run() -> dict:
    """Run the minimal CSA pipeline and return a dict of useful arrays."""
    # ------------------------------------------------------------------
    # 1. Load the bundled 1-cell ICON grid.
    # ------------------------------------------------------------------
    grid = Grid()
    pcio.ncdata().read_dat(str(ICON_GRID), grid)

    # Cell 2311 (Aleutians) was renumbered to 0 in the bundled grid.
    lat_verts_orig = np.degrees(grid.clat_vertices[0])
    lon_verts_orig = np.degrees(grid.clon_vertices[0])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts_orig.copy(), lon_verts_orig.copy()
    )

    # ------------------------------------------------------------------
    # 2. Set up run parameters + point the MERIT loader at the bundled tile.
    # ------------------------------------------------------------------
    params = Params()
    params.path_merit = str(MERIT_DIR) + "/"
    params.lat_extent = clat_verts
    params.lon_extent = clon_verts
    # The bundled MERIT tile has already been coarse-grained at capture time
    # (merit_cg=20), so the loader's own coarse-graining is a no-op here.
    params.merit_cg = 1
    params.padding = PADDING

    # ------------------------------------------------------------------
    # 3. Load topography via the real production loader.
    # ------------------------------------------------------------------
    topo = TopoCell()
    reader = pcio.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_merit_topo(topo, params)
    # Match production: clamp deep-ocean fill values so they don't pollute the spectrum.
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()

    # Extract the cell-specific subset.
    cell = TopoCell()
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=True)
    topo_orig = np.copy(cell.topo)

    # ------------------------------------------------------------------
    # 4. First approximation: full spectrum, unregularised.
    # ------------------------------------------------------------------
    ctx = ComputeContext()
    fa = interface.get_pmf(NHI, NHJ, U, V, ctx=ctx)
    freqs_fa, _, dat_fa = fa.sappx(cell, lmbda=0.0)

    # ------------------------------------------------------------------
    # 5. Select the top-N modes by amplitude.
    # ------------------------------------------------------------------
    fq_cpy = np.copy(freqs_fa)
    fq_cpy[np.isnan(fq_cpy)] = 0.0
    indices, max_ampls = [], []
    for _ in range(N_MODES):
        max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
        indices.append(max_idx)
        max_ampls.append(fq_cpy[max_idx])
        fq_cpy[max_idx] = 0.0
    k_idxs = [p[1] for p in indices]
    l_idxs = [p[0] for p in indices]

    # ------------------------------------------------------------------
    # 6. Second approximation: only the selected modes, regularised.
    # ------------------------------------------------------------------
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=False)
    sa = interface.get_pmf(NHI, NHJ, U, V, ctx=ctx)
    sa.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    freqs_sa, uw_sa, dat_sa = sa.sappx(cell, lmbda=LMBDA_SG, updt_analysis=True)

    cell.topo = topo_orig
    ideal = physics.ideal_pmf(U=U, V=V)
    uw_comp = ideal.compute_uw_pmf(cell.analysis)

    return {
        "topo_input": np.nan_to_num(topo_orig),
        "dat_fa": np.nan_to_num(np.asarray(dat_fa)),
        "dat_sa": np.nan_to_num(np.asarray(dat_sa)),
        "freqs_fa": np.nan_to_num(freqs_fa),
        "freqs_sa": np.nan_to_num(freqs_sa),
        "max_ampls": np.asarray(max_ampls),
        "uw_comp": float(uw_comp),
    }


def _render_figure(result: dict, out_path: Path) -> None:
    """2-row figure: physical-domain reconstructions / spectra. Same layout as the fixture's figure.png."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))

    topo = result["topo_input"]
    phys_vmin, phys_vmax = float(topo.min()), float(topo.max())
    for col, (arr, title) in enumerate(
        [
            (topo, "original topography"),
            (result["dat_fa"], "first-approx reconstruction"),
            (result["dat_sa"], "final (SA) reconstruction"),
        ]
    ):
        im = axs[0, col].imshow(
            arr,
            origin="lower",
            aspect="auto",
            cmap="terrain",
            vmin=phys_vmin,
            vmax=phys_vmax,
        )
        axs[0, col].set_title(title, fontsize=10)
        axs[0, col].set_xlabel("lon index")
        axs[0, col].set_ylabel("lat index")
        plt.colorbar(im, ax=axs[0, col], fraction=0.046, pad=0.04)

    im_fa = axs[1, 0].imshow(
        result["freqs_fa"], origin="lower", aspect="auto", cmap="viridis"
    )
    axs[1, 0].set_title("first-approx spectrum", fontsize=10)
    axs[1, 0].set_xlabel("k")
    axs[1, 0].set_ylabel("l")
    plt.colorbar(im_fa, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im_sa = axs[1, 1].imshow(
        result["freqs_sa"], origin="lower", aspect="auto", cmap="viridis"
    )
    axs[1, 1].set_title("second-approx spectrum (final)", fontsize=10)
    axs[1, 1].set_xlabel("k")
    axs[1, 1].set_ylabel("l")
    plt.colorbar(im_sa, ax=axs[1, 1], fraction=0.046, pad=0.04)

    axs[1, 2].bar(range(len(result["max_ampls"])), result["max_ampls"])
    axs[1, 2].set_title(f"top {N_MODES} mode amplitudes (sorted)", fontsize=10)
    axs[1, 2].set_xlabel("mode rank")

    fig.suptitle(
        "Minimal CSA example: ICON cell 2311 (Aleutians ~52°N) — bundled MERIT slice",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("Running minimal regional MERIT CSA example (ICON cell 2311) ...")
    result = run()
    print(f"  top 5 mode amplitudes:     {result['max_ampls'][:5]}")
    print(f"  computed PMF (uw_comp):    {result['uw_comp']:.4g}")
    print(f"  freqs_sa shape:            {result['freqs_sa'].shape}")

    out_dir = HERE / "output"
    figure_path = out_dir / "icon_regional_minimal.png"
    _render_figure(result, figure_path)
    print(f"  figure written:            {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
