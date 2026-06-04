"""Capture the regional MERIT fixture for ICON cell 2311 (Aleutians, ~52°N).

This cell stresses the *false-positive* dateline branch of the MERIT loader:
all three vertex lons are negative and close to -180° (-173.55° to -176.84°),
which is exactly the case where a naive dateline detector could mistakenly
trigger ``split_EW``. The cell also contains real Aleutian-arc topography so
the CSA pipeline produces a meaningful spectrum (cell 1074 — the true
dateline crosser at 80°N — was Arctic Ocean and clamped to a constant -500,
giving a degenerate signal).

We use ``MERIT_CG=20`` rather than the regional-pipeline default of 100 —
that default was tuned for *regional* bboxes spanning 14-18°, where 100×
downsampling still leaves enough resolution. For a single ~3° ICON cell,
100× collapses Aleutian islands (~10 km wide) to ~2 pixels and the grid
(~51×37) is below Nyquist for the (24, 48) wavenumber spectrum. 20× gives
~1.1 km/px (matches the ETOPO single-cell fixture's effective scale of
15″×4 ≈ 3″×20), ~257×184 grid, and islands ~9 pixels wide so the figure is
actually verifiable by eye.

Pre-downsamples both MERIT tiles by ``MERIT_CG=100`` (the regional default)
at capture time and sets the test ``merit_cg=1`` so the loader's
coarse-graining step is a no-op on the already-downsampled bundle.
End-to-end output corresponds to production at ``merit_cg=100``.

Usage::

    python -m tests.reproducibility.capture.capture_regional_merit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CASE = "regional_merit"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / CASE
C_IDX_ORIGINAL = 2311  # Aleutians, ~52°N — false-positive dateline path with real land
MERIT_CG = 20  # ~1.1 km/px at 52°N — matches ETOPO etopo_cg=4 effective scale, satisfies Nyquist for (24, 48) spectrum (production regional default of 100 was tuned for 14-18° bboxes, too coarse for a single ICON cell)
NHI, NHJ = 24, 48
N_MODES = 50
LMBDA_SG = 1e-1
U, V = 10.0, 0.0
PADDING = 10


# ---------------------------------------------------------------------------
# 1. Bundle production-shape input slices
# ---------------------------------------------------------------------------


def _tile_filename(lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float) -> str:
    """Reproduce read_merit_topo.__get_fns naming for one tile.

    ``(lat_lo, lat_hi)`` are the southern/northern edges; ``(lon_lo, lon_hi)``
    are the western/eastern edges.
    """

    def tag_lat(v):
        return "N" if v >= 0 else "S"

    def tag_lon(v):
        return "E" if v >= 0 else "W"

    return "MERIT_%s%02d-%s%02d_%s%03d-%s%03d.nc4" % (
        tag_lat(lat_hi),
        abs(int(lat_hi)),
        tag_lat(lat_lo),
        abs(int(lat_lo)),
        tag_lon(lon_lo),
        abs(int(lon_lo)),
        tag_lon(lon_hi),
        abs(int(lon_hi)),
    )


def _bundle_inputs(fixture_dir: Path, real_icon_grid: Path, real_merit_dir: Path):
    """Slice the real ICON grid + MERIT tiles into the fixture's input/ dir."""
    from pycsa.core import io as pcio, var, utils
    from tests.reproducibility.capture._slicing import (
        crop_and_downsample_tile,
        subset_icon_grid,
    )

    grid = var.grid()
    pcio.ncdata().read_dat(str(real_icon_grid), grid)

    lat_verts_orig = np.degrees(grid.clat_vertices[C_IDX_ORIGINAL])
    lon_verts_orig = np.degrees(grid.clon_vertices[C_IDX_ORIGINAL])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts_orig.copy(), lon_verts_orig.copy()
    )

    split_EW = (clon_verts.max() - clon_verts.min()) > 180.0
    lat_min = float(clat_verts.min())
    lat_max = float(clat_verts.max())

    input_dir = fixture_dir / "input"
    merit_input = input_dir / "merit"

    # ICON grid: keep cell 1074 only (renumbered to 0 in the bundle).
    subset_icon_grid(real_icon_grid, input_dir / "icon_grid.nc", [C_IDX_ORIGINAL])

    # MERIT tile boundaries (mirrors pycsa.core.io.read_merit_topo.__init__).
    fn_lat = [90, 60, 30, 0, -30, -60, -90]
    fn_lon = list(range(-180, 210, 30))

    def _find_tile_lat(lat: float) -> tuple[int, int]:
        for i in range(len(fn_lat) - 1):
            if fn_lat[i] >= lat >= fn_lat[i + 1]:
                return fn_lat[i + 1], fn_lat[i]  # (lo, hi)
        raise ValueError(f"lat {lat} not in any MERIT tile")

    def _find_tile_lon(lon: float) -> tuple[int, int]:
        for i in range(len(fn_lon) - 1):
            if fn_lon[i] <= lon < fn_lon[i + 1]:
                return fn_lon[i], fn_lon[i + 1]  # (lo, hi)
        raise ValueError(f"lon {lon} not in any MERIT tile")

    sliced = []
    if not split_EW:
        lon_min = float(clon_verts.min())
        lon_max = float(clon_verts.max())
        tile_lat_lo, tile_lat_hi = _find_tile_lat((lat_min + lat_max) / 2)
        tile_lon_lo, tile_lon_hi = _find_tile_lon((lon_min + lon_max) / 2)
        fn = _tile_filename(tile_lat_lo, tile_lat_hi, tile_lon_lo, tile_lon_hi)
        info = crop_and_downsample_tile(
            real_merit_dir / fn,
            merit_input / fn,
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max),
            factor=MERIT_CG,
            topo_var="Elevation",
            lat_pad=0.5,
            lon_pad=0.5,
        )
        sliced.append((fn, info))
    else:
        # Dateline crossing: bundle one east-of-dateline tile and one
        # west-of-dateline tile from the same lat band.
        tile_lat_lo, tile_lat_hi = _find_tile_lat((lat_min + lat_max) / 2)
        lons_east = clon_verts[clon_verts >= 0]
        lons_west = clon_verts[clon_verts < 0]
        tile_lon_lo_e, tile_lon_hi_e = _find_tile_lon(float(lons_east.min()))
        fn_e = _tile_filename(tile_lat_lo, tile_lat_hi, tile_lon_lo_e, tile_lon_hi_e)
        info_e = crop_and_downsample_tile(
            real_merit_dir / fn_e,
            merit_input / fn_e,
            lat_range=(lat_min, lat_max),
            lon_range=(float(lons_east.min()), 180.0),
            factor=MERIT_CG,
            topo_var="Elevation",
            lat_pad=0.5,
            lon_pad=0.5,
        )
        sliced.append((fn_e, info_e))
        tile_lon_lo_w, tile_lon_hi_w = _find_tile_lon(float(lons_west.max()))
        fn_w = _tile_filename(tile_lat_lo, tile_lat_hi, tile_lon_lo_w, tile_lon_hi_w)
        info_w = crop_and_downsample_tile(
            real_merit_dir / fn_w,
            merit_input / fn_w,
            lat_range=(lat_min, lat_max),
            lon_range=(-180.0, float(lons_west.max())),
            factor=MERIT_CG,
            topo_var="Elevation",
            lat_pad=0.5,
            lon_pad=0.5,
        )
        sliced.append((fn_w, info_w))

    return {
        "split_EW": split_EW,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "clat_verts": clat_verts.tolist(),
        "clon_verts": clon_verts.tolist(),
        "sliced_tiles": sliced,
    }


# ---------------------------------------------------------------------------
# 2. Run the regional CSA pipeline against the bundled inputs
# ---------------------------------------------------------------------------


def _run_pipeline(fixture_dir: Path, return_cell: bool = False):
    """Re-runs the regional MERIT pipeline using the bundled inputs.

    By default returns the ``dict[str, np.ndarray]`` of pinned + figure
    variables (the contract the reproducibility test relies on). With
    ``return_cell=True`` it also returns the ``cell`` (ocean-masked for the
    figure) and ``params``, for the shared ``plot_cell_diagnostics``.
    """
    from pycsa.core import io as pcio, var, utils, physics
    from pycsa.wrappers import interface

    input_dir = fixture_dir / "input"
    bundled_grid = input_dir / "icon_grid.nc"
    bundled_merit_dir = str(input_dir / "merit") + "/"

    grid = var.grid()
    pcio.ncdata().read_dat(str(bundled_grid), grid)

    # In the bundle, cell 1074 is at index 0.
    lat_verts_orig = np.degrees(grid.clat_vertices[0])
    lon_verts_orig = np.degrees(grid.clon_vertices[0])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts_orig.copy(), lon_verts_orig.copy()
    )

    params = var.params()
    params.path_merit = bundled_merit_dir
    params.lat_extent = clat_verts
    params.lon_extent = clon_verts
    params.merit_cg = 1  # bundle is already pre-downsampled
    params.padding = PADDING

    # Load the regional MERIT topography. read_merit_topo populates topo
    # in place when is_parallel=False.
    topo = var.topo_cell()
    reader = pcio.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_merit_topo(topo, params)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()

    cell = var.topo_cell()
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=True)
    topo_orig = np.copy(cell.topo)

    # First approximation on the rectangular cover.
    first_guess = interface.get_pmf(NHI, NHJ, U, V)
    freqs_fg, uw_pmf_fg, dat_fg = first_guess.sappx(cell, lmbda=0.0)

    fq_cpy = np.copy(freqs_fg)
    fq_cpy[np.isnan(fq_cpy)] = 0.0
    indices, max_ampls = [], []
    for _ in range(N_MODES):
        max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
        indices.append(max_idx)
        max_ampls.append(fq_cpy[max_idx])
        fq_cpy[max_idx] = 0.0
    k_idxs = [p[1] for p in indices]
    l_idxs = [p[0] for p in indices]

    # Second approximation on the triangular mask.
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=False)
    second_guess = interface.get_pmf(NHI, NHJ, U, V)
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    freqs_sa, uw_sa, dat_sa = second_guess.sappx(
        cell, lmbda=LMBDA_SG, updt_analysis=True
    )

    cell.topo = topo_orig
    ideal = physics.ideal_pmf(U=U, V=V)
    uw_comp = ideal.compute_uw_pmf(cell.analysis)

    # k_idxs / l_idxs are deliberately NOT pinned: the order in which argmax
    # picks among nearly-equal mode amplitudes is platform-dependent
    # (different LAPACK builds → different ULP-level floats → different
    # argmax tie-breaks). The mode SELECTION is already pinned via
    # `max_ampls` (sorted by amplitude) and `freqs_sa` (final spectrum).
    variables = {
        "topo_input": np.nan_to_num(topo_orig),
        "dat_fa": np.nan_to_num(np.asarray(dat_fg)),
        "dat_sa": np.nan_to_num(np.asarray(dat_sa)),
        "freqs_fa": np.nan_to_num(freqs_fg),
        "freqs_sa": np.nan_to_num(freqs_sa),
        "uw_sa": np.nan_to_num(uw_sa),
        "max_ampls": np.asarray(max_ampls, dtype=np.float64),
        "uw_comp": np.asarray(uw_comp, dtype=np.float64),
    }
    if not return_cell:
        return variables

    # Figure path only (does not affect the pinned variables above): exclude
    # ocean from the mask so the shared per-cell diagnostic renders it white and
    # the RMSE is computed over land. MERIT marks ocean as a non-finite fill;
    # also drop anything below -200 m (deep bathymetry) for parity with ETOPO.
    params.nhi, params.nhj = NHI, NHJ
    ocean_mask = ~np.isfinite(cell.topo) | (cell.topo < -200.0)
    cell.mask = cell.mask & ~ocean_mask
    cell.get_masked(mask=cell.mask)
    return variables, cell, params


# ---------------------------------------------------------------------------
# 3. Figure + main
# ---------------------------------------------------------------------------


def capture(out_dir: Path, real_icon_grid: Path, real_merit_dir: Path) -> None:
    from tests.reproducibility.comparator import save_netcdf
    from tests.reproducibility.manifest import Manifest

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"slicing inputs into {out_dir / 'input'} ...")
    bundle_info = _bundle_inputs(out_dir, real_icon_grid, real_merit_dir)
    for fn, info in bundle_info["sliced_tiles"]:
        print(
            f"  {fn}: {info['lat_points']}x{info['lon_points']} (factor {info['factor']})"
        )

    print("running pipeline against the bundle ...")
    variables, cell, params = _run_pipeline(out_dir, return_cell=True)

    # Physical-domain fields (topo_input, dat_fa, dat_sa) are used for the
    # figure but not pinned in the fixture: their numerics are derivable
    # from the pinned spectra, and bundling the (32, 1963)-class arrays
    # would more than double the fixture footprint.
    FIGURE_ONLY = {"topo_input", "dat_fa", "dat_sa"}
    pinned = {k: v for k, v in variables.items() if k not in FIGURE_ONLY}

    save_netcdf(out_dir / "output.nc", pinned)

    manifest = Manifest.build(
        fixture=CASE,
        variables=pinned,
        notes=(
            f"Regional MERIT CSA on ICON cell {C_IDX_ORIGINAL} "
            f"(Aleutians, ~52°N — exercises the false-positive dateline path: "
            f"lons all-negative near -180° but span < 180°). Bundled MERIT "
            f"tile pre-downsampled by merit_cg={MERIT_CG} (~1.1 km/px, "
            f"matches the ETOPO fixture's effective scale; the regional-"
            f"pipeline default of 100 collapses Aleutian islands to ~2 "
            f"pixels). Test reads with merit_cg=1 so the loader's own "
            f"coarse-graining is a no-op. "
            f"NHI={NHI}, NHJ={NHJ}, N_MODES={N_MODES}, U={U}, V={V}, "
            f"lmbda_sg={LMBDA_SG}."
        ),
    )
    manifest.save(out_dir / "manifest.yml")

    _render(out_dir, variables, cell, params)

    print(f"captured regional_merit fixture → {out_dir}")
    print(f"  variables: {list(variables)}")
    print(f"  freqs_sa shape: {variables['freqs_sa'].shape}")
    print(f"  top mode amplitudes (first 5): {variables['max_ampls'][:5]}")


def _render(case_dir: Path, variables, cell, params) -> None:
    """Write ``figure.png`` from already-computed pipeline outputs."""
    from pycsa.plotting.diagnostics import plot_cell_diagnostics

    plot_cell_diagnostics(
        C_IDX_ORIGINAL,
        cell,
        variables["freqs_sa"],
        variables["dat_sa"],
        case_dir,
        params,
        out_path=case_dir / "figure.png",
        cell_label="Aleutians (ICON cell %d)" % C_IDX_ORIGINAL,
    )


def render_only(case_dir: Path = DEFAULT_DIR) -> None:
    """Regenerate ``figure.png`` from the bundled fixture input only.

    Unlike :func:`capture`, this needs no real source data — it re-runs the
    pipeline against the committed ``<case_dir>/input`` slices and re-renders
    the figure, leaving ``output.nc`` / ``manifest.yml`` untouched. Used by
    ``tests.reproducibility.render_figures`` to refresh figures in CI.
    """
    variables, cell, params = _run_pipeline(case_dir, return_cell=True)
    _render(case_dir, variables, cell, params)


def main(argv=None) -> int:
    from pycsa import local_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--icon-grid", type=Path, default=Path(local_paths.paths.icon_grid)
    )
    parser.add_argument("--merit-dir", type=Path, default=Path(local_paths.paths.merit))
    args = parser.parse_args(argv)
    capture(args.out, args.icon_grid, args.merit_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
