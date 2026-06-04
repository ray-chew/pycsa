"""Per-cell diagnostic plotting shared by the global run and the examples.

A single source of truth for the 3-panel per-cell figure (loaded topography /
second-approximation reconstruction / amplitude spectrum) and the ocean-aware
topography colormap, so that ``runs/icon_etopo_global.py`` and the bundled
examples produce identical figures.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-GUI backend (safe under Dask workers / headless runs)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors

from pycsa.plotting import plotter

logger = logging.getLogger(__name__)


def get_topo_colormap():
    """
    Create a topography colormap with blue for ocean (< 0m) and terrain colors for land (> 0m).
    Transition occurs exactly at sea level (0m) with smooth blending.

    For TwoSlopeNorm to work correctly, we need equal colors on each side:
    128 colors for ocean (< 0m) + 128 colors for land (> 0m) = 256 total
    """
    # Ocean colors (blue shades from deep to shallow)
    ocean_colors = plt.cm.Blues_r(np.linspace(0.4, 0.95, 120))

    # Smooth transition zone around sea level (8 colors on each side)
    # Get the last ocean color and first land color
    last_ocean = plt.cm.Blues_r(0.95)
    first_land = plt.cm.terrain(0.25)

    # Create smooth blend from ocean to land
    transition_colors = np.zeros((16, 4))
    for i in range(4):  # RGBA channels
        transition_colors[:, i] = np.linspace(last_ocean[i], first_land[i], 16)

    # Land colors (terrain-like: green to brown to white)
    land_colors = plt.cm.terrain(np.linspace(0.28, 1.0, 120))

    # Combine: 120 ocean + 16 transition + 120 land = 256 total
    # Transition centered at index 128 (sea level)
    colors = np.vstack((ocean_colors, transition_colors, land_colors))
    return mcolors.LinearSegmentedColormap.from_list("topo", colors)


def plot_cell_diagnostics(
    c_idx,
    cell_sa,
    ampls_sa,
    dat_2D_sa,
    output_dir,
    params,
    out_path=None,
    cell_label=None,
):
    """
    Create 3-panel diagnostic plot for a single cell.

    Panel 1: Loaded topography (original ETOPO data within cell)
    Panel 2: Reconstructed topography after second approximation
    Panel 3: Computed spectrum

    Parameters
    ----------
    c_idx : int
        Cell index
    cell_sa : topo_cell
        Cell object after second approximation (contains original topo in cell.topo)
    ampls_sa : ndarray
        Amplitude spectrum from second approximation
    dat_2D_sa : ndarray
        Reconstructed topography from second approximation
    output_dir : Path
        Output directory for the default ``cell_{c_idx:05d}.png`` filename.
    params : params object
        Parameters object
    out_path : Path, optional
        Explicit output path; overrides ``output_dir / f"cell_{c_idx:05d}.png"``.
    cell_label : str, optional
        Title prefix for panel 1; defaults to ``f"Cell {c_idx}"``.
    """
    # Create figure with 3 panels
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Get elevation extent for consistent color scaling
    vmin = -200.0  # Always fix ocean floor at -500m (blue portion)
    vmax = np.nanmax(cell_sa.topo)

    # Ensure vmax is positive (land)
    if vmax <= 0:
        vmax = 100.0  # Force some land color even if all ocean

    # Create custom colormap with blue for ocean, terrain colors for land
    topo_cmap = get_topo_colormap()

    # Create normalization centered at sea level (0m)
    # This makes the colormap transition exactly at 0m
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Panel 1: Original topography within cell
    topo_original = cell_sa.topo.copy()
    topo_original[~cell_sa.mask] = np.nan

    label = cell_label if cell_label is not None else f"Cell {c_idx}"

    im1 = axs[0].imshow(
        topo_original, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0].set_title(
        f"{label}: Loaded Topography\nRange: [{vmin:.0f}, {vmax:.0f}] m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0].set_xlabel("Longitude index")
    axs[0].set_ylabel("Latitude index")
    cbar1 = plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Elevation [m]", rotation=270, labelpad=15)

    # Panel 2: Reconstructed topography (masked)
    dat_2D_masked = dat_2D_sa.copy()
    dat_2D_masked[~cell_sa.mask] = np.nan

    # Compute reconstruction error
    diff = cell_sa.topo - dat_2D_sa
    rmse = np.sqrt(np.mean(diff[cell_sa.mask] ** 2))
    rel_rmse = rmse / (vmax - vmin) * 100

    im2 = axs[1].imshow(
        dat_2D_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[1].set_title(
        f"Reconstructed (2nd Approx)\nRMSE: {rmse:.1f} m ({rel_rmse:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    axs[1].set_xlabel("Longitude index")
    axs[1].set_ylabel("Latitude index")
    cbar2 = plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Elevation [m]", rotation=270, labelpad=15)

    # Panel 3: Amplitude spectrum in (k,l) wavenumber space
    fig_obj = plotter.fig_obj(fig, params.nhi, params.nhj, cbar=True, set_label=True)
    axs[2] = fig_obj.freq_panel(
        axs[2], ampls_sa, title="Amplitude Spectrum", v_extent=None
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(out_path) if out_path is not None else (
        output_dir / f"cell_{c_idx:05d}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Explicit memory cleanup - delete ALL objects to prevent memory leaks
    del fig, axs, fig_obj, im1, im2, topo_original, dat_2D_masked
    del cbar1, cbar2, norm, topo_cmap, diff
    gc.collect()  # Force garbage collection after plotting

    logger.info(f"  Plot saved: {output_path}")
