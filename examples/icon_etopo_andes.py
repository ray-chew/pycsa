"""Showcase CSA example — central Andes (ICON cell, ETOPO).

A reviewer or new user can clone the repo and run::

    python examples/icon_etopo_andes.py

…to see the full CSA pipeline (load topography -> first approximation ->
mode selection -> second approximation -> reconstruction) end to end on a
single, geographically interesting ICON grid cell over the central Andes
near Aconcagua (~32S) -- a strong orographic gravity-wave source with
dramatic coast-to-summit relief. The example ships with all the data it
needs (``examples/data/etopo_andes/`` is ~130 KB: a single-cell ICON grid
subset plus a coarse-grained ETOPO slice); total runtime is a few seconds.

The two harder "corner" cells (a false-positive-dateline cell and a
south-pole cell) are pinned by the reproducibility suite
(``tests/reproducibility/``) and gated in CI; see the Quality Control
section of the software paper. Numerics here are computed live, not pinned
-- this script is for human inspection.
"""

from __future__ import annotations

# Import netCDF4 BEFORE pycsa to avoid a libhdf5 init-order quirk -- importing
# pycsa.core.io first can leave the HDF5 layer in a state that fails subsequent
# `nc.Dataset(...)` opens with "NetCDF: HDF error".
import netCDF4 as _nc  # noqa: F401

from pathlib import Path

import numpy as np

from pycsa.compute import ComputeContext
from pycsa.config.params import params as Params
from pycsa.core import io as pcio, utils
from pycsa.data.cell import grid as Grid, topo_cell as TopoCell
from pycsa.wrappers import interface
from pycsa.plotting.diagnostics import plot_cell_diagnostics

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "etopo_andes"
ICON_GRID = DATA_DIR / "icon_grid.nc"
ETOPO_DIR = DATA_DIR / "etopo"

C_IDX = 19442  # the bundled cell (central Andes, ~32S, Aconcagua region)
# Same spectral resolution as the etopo_single_cell reproducibility fixture.
NHI, NHJ = 32, 64
N_MODES = 100
LMBDA_SG = 1e-1
U, V = 10.0, 0.0
PADDING = 10
# Ocean config matches the production global run (runs/icon_etopo_global.py):
# clamp deep bathymetry to -500 m for the fit, then exclude ocean (< -200 m) from
# the mask so the atmosphere "sees" the sea surface, not the seafloor. Deep ocean
# and the cell exterior render white; the shallow shelf (-200..0 m) stays blue.
DEEP_OCEAN_CLAMP = -500.0
OCEAN_EXCLUDE = -200.0


def run() -> dict:
    """Run the CSA pipeline on the bundled Andes cell; return useful arrays."""
    # 1. Load the bundled single-cell ICON grid (the chosen cell is index 0).
    grid = Grid()
    pcio.ncdata().read_dat(str(ICON_GRID), grid)
    lat_verts = np.degrees(grid.clat_vertices[0])
    lon_verts = np.degrees(grid.clon_vertices[0])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts.copy(), lon_verts.copy()
    )

    # 2. Parameters; point the ETOPO loader at the bundled (pre-coarsened) slice.
    params = Params()
    params.nhi, params.nhj = NHI, NHJ
    params.path_etopo = str(ETOPO_DIR) + "/"
    params.lat_extent = clat_verts
    params.lon_extent = clon_verts
    params.etopo_cg = 1  # bundle is already coarse-grained at capture time
    params.padding = PADDING

    # 3. Load topography via the production ETOPO loader.
    topo = TopoCell()
    reader = pcio.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_etopo_topo(topo, params)
    topo.gen_mgrids()

    # Extract the cell on its rectangular cover, then floor the abyssal ocean
    # (the floor must be applied to the extracted cell, which is rebuilt from the
    # raw tile, not to topo.topo).
    cell = TopoCell()
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=True)
    cell.topo[cell.topo < DEEP_OCEAN_CLAMP] = DEEP_OCEAN_CLAMP

    # 4. First approximation: full spectrum, unregularised.
    ctx = ComputeContext()
    fa = interface.get_pmf(NHI, NHJ, U, V, ctx=ctx)
    freqs_fa, _, dat_fa = fa.sappx(cell, lmbda=0.0)

    # 5. Select the top-N modes by amplitude.
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

    # 6. Second approximation: selected modes only, on the triangle-masked cell.
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=False)
    cell.topo[cell.topo < DEEP_OCEAN_CLAMP] = DEEP_OCEAN_CLAMP
    sa = interface.get_pmf(NHI, NHJ, U, V, ctx=ctx)
    sa.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    freqs_sa, uw_sa, dat_sa = sa.sappx(cell, lmbda=LMBDA_SG, updt_analysis=True)

    # Exclude ocean (< -200 m) from the mask for the analysis and the figure, exactly
    # as the production per-cell pipeline does: the GW source is the sea surface, not
    # the seafloor. Deep ocean + the cell exterior become NaN (white) in the plot.
    ocean_mask = cell.topo < OCEAN_EXCLUDE
    cell.mask = cell.mask & ~ocean_mask
    cell.get_masked(mask=cell.mask)

    return {
        "cell": cell,
        "freqs_sa": np.asarray(freqs_sa),
        "dat_sa": np.asarray(dat_sa),
        "params": params,
        "max_ampls": np.asarray(max_ampls),
    }


def main() -> int:
    print("Running Andes ETOPO CSA showcase ...")
    result = run()
    print(f"  top 5 mode amplitudes: {np.round(result['max_ampls'][:5], 3)}")
    print(f"  spectrum shape:        {result['freqs_sa'].shape}")
    out = HERE / "output" / "icon_etopo_andes.png"
    # Reuse the production per-cell diagnostic plot (same ocean-aware colormap and
    # masking as runs/icon_etopo_global.py) for a figure consistent with the global run.
    plot_cell_diagnostics(
        C_IDX,
        result["cell"],
        result["freqs_sa"],
        result["dat_sa"],
        out.parent,
        result["params"],
        out_path=out,
        cell_label="Central Andes (ICON cell %d)" % C_IDX,
    )
    print(f"  figure written:        {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
