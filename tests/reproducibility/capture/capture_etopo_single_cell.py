"""Capture the ETOPO single-cell fixture for ICON cell 17408 (south polar).

This cell sits at lat ~-89°S and exercises the polar branch of the ETOPO
loader (``iint *= 5`` at ``max(cell.lat) < -85.0``). Its triangle has one
vertex effectively at the South Pole and two vertices around -88.2°S spread
across ~163° of longitude — so the loader assembles topography from a
~12-tile-wide strip at 15° longitudinal intervals.

Bundled ETOPO tiles are pre-downsampled by ``ETOPO_CG=20``. At test time we
set ``etopo_cg=1``; the loader's polar 5× multiplier still fires, yielding
an effective 100× total downsampling. This is heavier than production
``etopo_cg=4`` and trades resolution for fixture-suite runtime (~1 min for
all three cases vs ~7 min at ETOPO_CG=4). The output still gates Phase B
refactors meaningfully — it's a deterministic function of the same code
paths, just at coarser resolution.

Usage::

    python -m tests.reproducibility.capture.capture_etopo_single_cell
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CASE = "etopo_single_cell"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / CASE
C_IDX_ORIGINAL = 17408  # south polar (~-89°S), exercises polar 5× branch
ETOPO_CG = 20  # heavier than production etopo_cg=4 for fast fixture; loader's polar 5× brings effective downsampling to 100×
NHI, NHJ = 32, 64
N_MODES = 100
LMBDA_SG = 1e-1
U, V = 10.0, 0.0
PADDING = 10


# ---------------------------------------------------------------------------
# 1. Bundle the ETOPO + ICON-grid slices
# ---------------------------------------------------------------------------


def _etopo_tile_name(lat_bound: float, lon_bound: float) -> str:
    """Reproduce read_etopo_topo.__get_fns naming.

    Tile is named after its NW corner (largest lat, smallest lon).
    """
    lat_tag = "N" if lat_bound >= 0 else "S"
    lon_tag = "E" if lon_bound >= 0 else "W"
    return "ETOPO_2022_v1_15s_%s%02d%s%03d_surface.nc" % (
        lat_tag,
        abs(int(lat_bound)),
        lon_tag,
        abs(int(lon_bound)),
    )


def _bundle_inputs(fixture_dir: Path, real_icon_grid: Path, real_etopo_dir: Path):
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
    lon_min = float(clon_verts.min())
    lon_max = float(clon_verts.max())

    input_dir = fixture_dir / "input"
    etopo_input = input_dir / "etopo"
    subset_icon_grid(real_icon_grid, input_dir / "icon_grid.nc", [C_IDX_ORIGINAL])

    # ETOPO tile boundaries (matches pycsa.core.io.read_etopo_topo.__init__).
    fn_lat = [90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90]
    fn_lon = list(range(-180, 195, 15))

    # Compute the (lat_idx, lon_idx) ranges of tiles that overlap the cell bbox.
    def lat_tile_indices(lat_lo: float, lat_hi: float) -> list[int]:
        idx = []
        for i in range(len(fn_lat) - 1):
            tile_top, tile_bot = fn_lat[i], fn_lat[i + 1]
            # Tile covers (tile_bot, tile_top]; include if overlap with [lat_lo, lat_hi].
            if not (tile_bot >= lat_hi or tile_top <= lat_lo):
                idx.append(i)
        return idx

    def lon_tile_indices(lon_lo: float, lon_hi: float) -> list[int]:
        idx = []
        for i in range(len(fn_lon) - 1):
            tile_lo, tile_hi = fn_lon[i], fn_lon[i + 1]
            if not (tile_hi <= lon_lo or tile_lo >= lon_hi):
                idx.append(i)
        return idx

    sliced = []
    if split_EW:
        # Two lon halves; handle each strip separately. (Cell 17408 doesn't
        # actually trigger this branch, but other polar cells could.)
        east_idx = lon_tile_indices(float(clon_verts[clon_verts >= 0].min()), 180.0)
        west_idx = lon_tile_indices(-180.0, float(clon_verts[clon_verts < 0].max()))
        lon_groups = [
            (east_idx, float(clon_verts[clon_verts >= 0].min()), 180.0),
            (west_idx, -180.0, float(clon_verts[clon_verts < 0].max())),
        ]
    else:
        lon_groups = [(lon_tile_indices(lon_min, lon_max), lon_min, lon_max)]

    for lat_idx in lat_tile_indices(lat_min, lat_max):
        tile_lat_top = fn_lat[lat_idx]
        for lon_indices, lo_group, hi_group in lon_groups:
            for li in lon_indices:
                tile_lon_lo = fn_lon[li]
                fn = _etopo_tile_name(tile_lat_top, tile_lon_lo)
                # Only crop to the cell's actual lat/lon bbox within this tile.
                info = crop_and_downsample_tile(
                    real_etopo_dir / fn,
                    etopo_input / fn,
                    lat_range=(lat_min, lat_max),
                    lon_range=(
                        max(lo_group, tile_lon_lo),
                        min(hi_group, fn_lon[li + 1]),
                    ),
                    factor=ETOPO_CG,
                    topo_var="z",
                    extra_vars=("crs",),
                    lat_pad=0.5,
                    lon_pad=0.5,
                )
                sliced.append((fn, info))

    return {
        "split_EW": split_EW,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "clat_verts": clat_verts.tolist(),
        "clon_verts": clon_verts.tolist(),
        "sliced_tiles": sliced,
    }


# ---------------------------------------------------------------------------
# 2. Run the ETOPO single-cell pipeline against the bundle
# ---------------------------------------------------------------------------


def _run_pipeline(fixture_dir: Path) -> dict[str, np.ndarray]:
    """Mirrors the per-cell ETOPO pipeline used by runs/icon_etopo_global.do_cell."""
    from pycsa.core import io as pcio, var, utils, physics
    from pycsa.wrappers import interface

    input_dir = fixture_dir / "input"
    bundled_grid = input_dir / "icon_grid.nc"
    bundled_etopo_dir = str(input_dir / "etopo") + "/"

    grid = var.grid()
    pcio.ncdata().read_dat(str(bundled_grid), grid)

    lat_verts_orig = np.degrees(grid.clat_vertices[0])
    lon_verts_orig = np.degrees(grid.clon_vertices[0])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts_orig.copy(), lon_verts_orig.copy()
    )

    params = var.params()
    params.path_etopo = bundled_etopo_dir
    params.lat_extent = clat_verts
    params.lon_extent = clon_verts
    params.etopo_cg = 1  # bundle is already pre-downsampled
    params.padding = PADDING

    topo = var.topo_cell()
    reader = pcio.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_etopo_topo(topo, params)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()

    cell = var.topo_cell()
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=True)
    topo_orig = np.copy(cell.topo)

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

    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=False)
    second_guess = interface.get_pmf(NHI, NHJ, U, V)
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    freqs_sa, uw_sa, dat_sa = second_guess.sappx(
        cell, lmbda=LMBDA_SG, updt_analysis=True
    )

    cell.topo = topo_orig
    ideal = physics.ideal_pmf(U=U, V=V)
    uw_comp = ideal.compute_uw_pmf(cell.analysis)

    # k_idxs / l_idxs are deliberately NOT pinned: argmax tie-breaking
    # among nearly-equal mode amplitudes is platform-dependent. The mode
    # SELECTION is already pinned via `max_ampls` (sorted) and `freqs_sa`.
    return {
        "topo_input": np.nan_to_num(topo_orig),
        "dat_fa": np.nan_to_num(np.asarray(dat_fg)),
        "dat_sa": np.nan_to_num(np.asarray(dat_sa)),
        "freqs_fa": np.nan_to_num(freqs_fg),
        "freqs_sa": np.nan_to_num(freqs_sa),
        "uw_sa": np.nan_to_num(uw_sa),
        "max_ampls": np.asarray(max_ampls, dtype=np.float64),
        "uw_comp": np.asarray(uw_comp, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 3. Figure + main
# ---------------------------------------------------------------------------


def _render_figure(variables: dict[str, np.ndarray], path: Path) -> None:
    """2-row figure: physical-domain reconstructions on top, spectra below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topo = variables["topo_input"]
    dat_fa = variables["dat_fa"]
    dat_sa = variables["dat_sa"]
    fa = variables["freqs_fa"]
    sa = variables["freqs_sa"]
    ampls = variables["max_ampls"]

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))

    phys_vmin, phys_vmax = float(topo.min()), float(topo.max())
    for col, (arr, title) in enumerate(
        [
            (topo, "original topography"),
            (dat_fa, "first-approx reconstruction"),
            (dat_sa, "final (SA) reconstruction"),
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

    im_fa = axs[1, 0].imshow(fa, origin="lower", aspect="auto", cmap="viridis")
    axs[1, 0].set_title("first-approx spectrum", fontsize=10)
    axs[1, 0].set_xlabel("k")
    axs[1, 0].set_ylabel("l")
    plt.colorbar(im_fa, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im_sa = axs[1, 1].imshow(sa, origin="lower", aspect="auto", cmap="viridis")
    axs[1, 1].set_title("second-approx spectrum (final)", fontsize=10)
    axs[1, 1].set_xlabel("k")
    axs[1, 1].set_ylabel("l")
    plt.colorbar(im_sa, ax=axs[1, 1], fraction=0.046, pad=0.04)

    axs[1, 2].bar(range(len(ampls)), ampls)
    axs[1, 2].set_title(f"top {N_MODES} mode amplitudes (sorted)", fontsize=10)
    axs[1, 2].set_xlabel("mode rank")

    fig.suptitle(
        f"ETOPO single-cell — ICON cell {C_IDX_ORIGINAL} (south polar ~-89°S), "
        f"etopo_cg={ETOPO_CG} (×5 polar = {ETOPO_CG * 5}× effective)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def capture(out_dir: Path, real_icon_grid: Path, real_etopo_dir: Path) -> None:
    from tests.reproducibility.comparator import save_netcdf
    from tests.reproducibility.manifest import Manifest

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"slicing inputs into {out_dir / 'input'} ...")
    bundle_info = _bundle_inputs(out_dir, real_icon_grid, real_etopo_dir)
    for fn, info in bundle_info["sliced_tiles"]:
        print(
            f"  {fn}: {info['lat_points']}x{info['lon_points']} (factor {info['factor']})"
        )

    print("running pipeline against the bundle ...")
    variables = _run_pipeline(out_dir)

    # Physical-domain fields (topo_input, dat_fa, dat_sa) are used for the
    # figure but not pinned in the fixture — they're (32, 1963)-class and
    # bundling them would more than double the fixture footprint. Numerics
    # are derivable from the pinned spectra.
    FIGURE_ONLY = {"topo_input", "dat_fa", "dat_sa"}
    pinned = {k: v for k, v in variables.items() if k not in FIGURE_ONLY}

    save_netcdf(out_dir / "output.nc", pinned)

    manifest = Manifest.build(
        fixture=CASE,
        variables=pinned,
        notes=(
            f"ETOPO single-cell CSA on ICON cell {C_IDX_ORIGINAL} "
            f"(south polar ~-89°S — exercises the loader's polar 5× "
            f"multiplier at lat < -85°). Bundled ETOPO tiles pre-downsampled "
            f"by ETOPO_CG={ETOPO_CG}; test reads with etopo_cg=1 + loader's "
            f"polar 5× → {ETOPO_CG * 5}× effective downsampling (heavier than "
            f"production etopo_cg=4 for fast suite runtime — still gates "
            f"refactors deterministically). "
            f"NHI={NHI}, NHJ={NHJ}, N_MODES={N_MODES}, U={U}, V={V}, "
            f"lmbda_sg={LMBDA_SG}."
        ),
    )
    manifest.save(out_dir / "manifest.yml")

    _render_figure(variables, out_dir / "figure.png")

    print(f"captured etopo_single_cell fixture → {out_dir}")
    print(f"  variables: {list(variables)}")
    print(f"  freqs_sa shape: {variables['freqs_sa'].shape}")
    print(f"  top mode amplitudes (first 5): {variables['max_ampls'][:5]}")


def main(argv=None) -> int:
    from pycsa import local_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--icon-grid", type=Path, default=Path(local_paths.paths.icon_grid)
    )
    parser.add_argument("--etopo-dir", type=Path, default=Path(local_paths.paths.etopo))
    args = parser.parse_args(argv)
    capture(args.out, args.icon_grid, args.etopo_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
