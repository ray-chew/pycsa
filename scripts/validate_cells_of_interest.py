"""One-off cells-of-interest validation for the kernel-spike defaults.

Runs the same 3-way prior comparison (baseline / isotropic@λ_GCV /
SpectralPrior@λ_GCV) and the same 3-way selector comparison
(Greedy / OMP / Lasso) as ``scripts/validate_hyperparam_defaults.py``,
but on **named real-data cells** picked from an ICON grid using
**ETOPO** topography (not MERIT — MERIT is regional and doesn't cover
Himalayas / Andes / Greenland / etc.).

Not part of CI. Run it by hand when you want to see how the structured
prior and the sparsity-inducing selectors behave on orographically
interesting regions.

Usage::

    ~/anaconda3/envs/playground/bin/python scripts/validate_cells_of_interest.py \\
        --icon-grid /path/to/icon_grid.nc \\
        --etopo-dir /path/to/etopo/tiles/

Add ``--regions himalayas andes`` to subset. Without ``--icon-grid``
and ``--etopo-dir`` the script falls back to the bundled
``tests/reproducibility/fixtures/etopo_single_cell`` (single polar
cell only — useful for end-to-end smoke testing of the script).

Each region produces:

- a tabular summary line with α (slope + stderr + R²), λ_GCV, the
  improvement vs. baseline under isotropic-at-λ and SpectralPrior-at-λ,
  and the structure-only contribution; then
- a 4-panel topography plot (input | baseline | isotropic | selected)
  in ``scripts/validate_outputs/<region>_reconstruction.png``;
- a 4-panel selector plot (input | Greedy | OMP | Lasso) in
  ``scripts/validate_outputs/<region>_selectors_reconstruction.png``.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

# netCDF4 must be imported before pycsa.core.io per the playground env quirk.
import netCDF4  # noqa: F401
import numpy as np

from pycsa.core import io as pcio, utils, var

# Reuse the validation helpers from the sibling script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from validate_hyperparam_defaults import (  # noqa: E402
    _diagnose,
    _selector_compare,
    OUTPUT_DIR,
)


# ----------------------------------------------------------------------
# Region catalog
# ----------------------------------------------------------------------
#
# (lat, lon) centers in degrees. These are *targets*; the script picks
# the ICON cell whose centroid is closest to each target. Add or
# override on the CLI with --regions.

REGIONS = {
    "aleutians":         ( 52.0, -175.0),
    "himalayas":         ( 28.0,   87.0),
    "andes":             (-22.0,  -68.0),
    "alps":              ( 46.5,    8.5),
    "greenland":         ( 72.0,  -40.0),
    "east_african_rift": (  0.5,   36.0),
    "pacific_nw":        ( 47.0, -123.0),
    "south_pole":        (-89.0,    0.0),
}

# Production-baseline parameters reused across cells (mirror the
# Aleutians MERIT fixture's lmbda choices and the ETOPO single-cell
# fixture's NHI/NHJ). Real cells may want per-region tuning; for the
# one-off check we keep one set of numbers so cross-cell comparison
# is apples-to-apples.
NHI, NHJ = 32, 64
N_MODES = 100
LMBDA_FA = 0.0
LMBDA_SA = 1.0e-1
U, V = 10.0, 0.0
PADDING = 10


# ----------------------------------------------------------------------
# Grid + ETOPO loaders
# ----------------------------------------------------------------------


def _great_circle_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance (radians) between two (lat, lon) points in degrees."""
    a1 = np.deg2rad(lat1); o1 = np.deg2rad(lon1)
    a2 = np.deg2rad(lat2); o2 = np.deg2rad(lon2)
    return np.arccos(
        np.clip(
            np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(o2 - o1),
            -1.0, 1.0,
        )
    )


def _find_nearest_cell(grid, lat_deg, lon_deg) -> int:
    """Return the ICON cell index whose centroid is closest to (lat, lon)."""
    clat = np.degrees(np.asarray(grid.clat))
    clon = np.degrees(np.asarray(grid.clon))
    d = _great_circle_distance(lat_deg, lon_deg, clat, clon)
    return int(np.argmin(d))


def _build_cell_from_etopo(grid, cell_idx: int, etopo_dir: str, etopo_cg: int = 1):
    """Construct a CSA ``topo_cell`` for ICON cell ``cell_idx`` using ETOPO data.

    Mirrors the load path in
    ``tests/reproducibility/capture/capture_etopo_single_cell._run_pipeline``.
    """
    lat_verts_orig = np.degrees(grid.clat_vertices[cell_idx])
    lon_verts_orig = np.degrees(grid.clon_vertices[cell_idx])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts_orig.copy(), lon_verts_orig.copy()
    )

    params = var.params()
    params.path_etopo = etopo_dir if etopo_dir.endswith("/") else etopo_dir + "/"
    params.lat_extent = clat_verts
    params.lon_extent = clon_verts
    params.etopo_cg = etopo_cg
    params.padding = PADDING

    topo = var.topo_cell()
    reader = pcio.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_etopo_topo(topo, params)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()

    cell = var.topo_cell()
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=True)
    return cell, clat_verts, clon_verts


# ----------------------------------------------------------------------
# Per-region runner
# ----------------------------------------------------------------------


def _run_region(name, grid, etopo_dir: str, etopo_cg: int):
    lat_deg, lon_deg = REGIONS[name]
    cell_idx = _find_nearest_cell(grid, lat_deg, lon_deg)
    clat_deg = float(np.degrees(grid.clat[cell_idx]))
    clon_deg = float(np.degrees(grid.clon[cell_idx]))
    print(f"\n========================================")
    print(f"region: {name}")
    print(f"  target  (lat, lon) = ({lat_deg:.2f}°, {lon_deg:.2f}°)")
    print(f"  nearest cell idx   = {cell_idx}  centroid = "
          f"({clat_deg:.2f}°, {clon_deg:.2f}°)")
    try:
        cell, *_ = _build_cell_from_etopo(grid, cell_idx, etopo_dir, etopo_cg)
    except Exception as exc:
        print(f"  ETOPO load failed: {exc} — SKIPPING")
        return

    # Reuse helpers from sibling validate script.
    from pycsa.core.priors import IsotropicPrior  # local import to keep top tidy

    # Prior comparison (3-way) at the production baseline.
    result = _diagnose(
        name, cell,
        nhi=NHI, nhj=NHJ, U=U, V=V,
        lmbda_fa=LMBDA_FA, lmbda_sa=LMBDA_SA, n_modes=N_MODES,
        truth_freqs=None,  # no ground truth for real-data cells
    )
    # Selectors under the production regime, then under the
    # isotropic-at-λ_GCV regime (the one that may be the empirical
    # winner per cell). Side-by-side both lets the reader see whether
    # selector preference flips with the regularization regime.
    _selector_compare(
        name, cell,
        nhi=NHI, nhj=NHJ, U=U, V=V,
        lmbda_fa=LMBDA_FA, lmbda_sa=LMBDA_SA, n_modes=N_MODES,
        prior_for_compare=None, truth_freqs=None,
    )
    _selector_compare(
        f"{name}_gcv", cell,
        nhi=NHI, nhj=NHJ, U=U, V=V,
        lmbda_fa=result["lambda"], lmbda_sa=result["lambda"], n_modes=N_MODES,
        prior_for_compare=IsotropicPrior(), truth_freqs=None,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--icon-grid",
        type=str,
        default=None,
        help="Path to an ICON grid .nc file (e.g. R2B4). If omitted, falls back "
             "to the bundled etopo_single_cell fixture grid (single polar cell only).",
    )
    p.add_argument(
        "--etopo-dir",
        type=str,
        default=None,
        help="Directory of ETOPO 15s NetCDF tiles. If omitted, uses the bundled "
             "single-cell fixture tiles.",
    )
    p.add_argument(
        "--etopo-cg",
        type=int,
        default=4,
        help="ETOPO coarse-graining factor (default 4 = production). The "
             "bundled fixture is pre-downsampled by 20× so the demo path "
             "auto-sets this to 1.",
    )
    p.add_argument(
        "--regions",
        nargs="*",
        default=list(REGIONS.keys()),
        choices=list(REGIONS.keys()),
        help="Subset of regions to run (default: all).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    if args.icon_grid is None or args.etopo_dir is None:
        print("No --icon-grid / --etopo-dir provided — falling back to bundled "
              "etopo_single_cell fixture (single polar cell only).")
        fixture_dir = repo_root / "tests" / "reproducibility" / "fixtures" / "etopo_single_cell" / "input"
        icon_grid_path = str(fixture_dir / "icon_grid.nc")
        etopo_dir = str(fixture_dir / "etopo")
        etopo_cg = 1  # bundle is pre-downsampled
        regions_to_run = ["south_pole"]
    else:
        icon_grid_path = args.icon_grid
        etopo_dir = args.etopo_dir
        etopo_cg = args.etopo_cg
        regions_to_run = args.regions

    grid = var.grid()
    pcio.ncdata().read_dat(icon_grid_path, grid)
    print(f"loaded ICON grid: {icon_grid_path}")
    print(f"  n_cells = {len(grid.clat)}")
    print(f"output dir for plots: {OUTPUT_DIR}")

    for name in regions_to_run:
        try:
            _run_region(name, grid, etopo_dir, etopo_cg)
        except Exception as exc:
            print(f"\nregion {name} failed: {exc}")
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
