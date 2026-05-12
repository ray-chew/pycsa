"""
Test script to compare old (corner-based) vs. new (centered) planar projection.

Tests 10 pre-selected polar cells (5 Arctic, 5 Antarctic) to evaluate improvement
in pyCSA RMSE when using centered projection instead of corner-based projection.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from pathlib import Path

from pycsa.core import io, var, utils
from pycsa.wrappers import interface
from scipy import interpolate

# Pre-selected cell indices from ICON grid
# Users can comment/uncomment cells to test different scenarios
# Focus on EXTREME POLAR cells where projection distortion is maximum

POLAR_CELLS = [
    # ========================================================================
    # ARCTIC CELLS (Greenland, 80-82°N)
    # ========================================================================
    # Moderate latitude - smaller projection differences expected
    # 3091,   # Arctic: 80.35°N, -92.11°E - Greenland
    # 3105,   # Arctic: 79.77°N, -65.63°E - Greenland
    # 3107,   # Arctic: 79.77°N, -78.37°E - Greenland
    # 3108,   # Arctic: 81.28°N, -57.03°E - Greenland
    # 3109,   # Arctic: 82.56°N, -45.32°E - Greenland
    # ========================================================================
    # EXTREME ANTARCTIC CELLS (87-89°S)
    # ========================================================================
    # These cells are within 1-3 degrees of the South Pole where corner
    # projection creates MAXIMUM distortion. This is where centered projection
    # should show the biggest improvement!
    # MOST EXTREME: -88.90°S (within 1.1° of South Pole!)
    17408,  # Antarctic: -88.90°S, -108.00°E - Interior plateau, 100% land, elev=2699m
    16384,  # Antarctic: -88.90°S, 180.00°E - Interior plateau, 100% land, elev=2761m
    18432,  # Antarctic: -88.90°S, -36.00°E - Interior plateau, 100% land, elev=2649m
    15360,  # Antarctic: -88.90°S, 108.00°E - Interior plateau, 100% land, elev=2941m
    19456,  # Antarctic: -88.90°S, 36.00°E - Interior plateau, 100% land, elev=2835m
    # VERY EXTREME: -88.07°S
    15362,  # Antarctic: -88.07°S, 108.00°E - Interior plateau, 100% land, elev=3055m
    16386,  # Antarctic: -88.07°S, 180.00°E - Interior plateau, 100% land, elev=2754m
    16387,
    17410,  # Antarctic: -88.07°S, -108.00°E - Interior plateau, 100% land, elev=2554m
    19458,  # Antarctic: -88.07°S, 36.00°E - Interior plateau, 100% land, elev=2882m
    18434,  # Antarctic: -88.07°S, -36.00°E - Interior plateau, 100% land, elev=2445m
    # EXTREME: -87.21°S
    15361,  # Antarctic: -87.21°S, 129.75°E - Interior plateau, 100% land, elev=3023m
    15363,  # Antarctic: -87.21°S, 86.25°E - Interior plateau, 100% land, elev=3105m
    16387,  # Antarctic: -87.21°S, 158.25°E - Interior plateau, 100% land, elev=2698m
    17409,  # Antarctic: -87.21°S, -86.25°E - Interior plateau, 100% land, elev=2384m
    19457,  # Antarctic: -87.21°S, 57.75°E - Interior plateau, 100% land, elev=3059m
    # ========================================================================
    # LESS EXTREME ANTARCTIC CELLS (85-86°S)
    # ========================================================================
    # Still very high latitude but slightly less extreme than above
    # 15364,  # Antarctic: -85.39°S, 135.26°E - Interior plateau, 100% land, elev=2896m
    # 15369,  # Antarctic: -86.34°S, 90.55°E - Interior plateau, 100% land, elev=3214m
    # 15370,  # Antarctic: -85.75°S, 108.00°E - Interior plateau, 100% land, elev=3109m
    # 15371,  # Antarctic: -86.34°S, 125.45°E - Interior plateau, 100% land, elev=2987m
    # 15372,  # Antarctic: -85.39°S, 80.74°E - Interior plateau, 100% land, elev=3328m
]

# Equatorial/mid-latitude cells - to test if centered projection helps more here
# Will be populated dynamically to find land cells near equator
EQUATORIAL_CELLS_CANDIDATES = list(range(0, 25000))  # Will filter for equatorial land
# EQUATORIAL_CELLS = [340, 992, 1015]  # To be filled in


def get_topo_colormap():
    """Create topography colormap with blue for ocean, terrain for land."""
    ocean_colors = plt.cm.Blues_r(np.linspace(0.4, 0.95, 120))
    last_ocean = plt.cm.Blues_r(0.95)
    first_land = plt.cm.terrain(0.25)

    transition_colors = np.zeros((16, 4))
    for i in range(4):
        transition_colors[:, i] = np.linspace(last_ocean[i], first_land[i], 16)

    land_colors = plt.cm.terrain(np.linspace(0.28, 1.0, 120))
    colors = np.vstack((ocean_colors, transition_colors, land_colors))
    return mcolors.LinearSegmentedColormap.from_list("topo", colors)


def interpolate_to_reference_grid(data_2D, source_cell, target_cell):
    """
    Interpolate 2D data from source planar grid to target planar grid.

    This is needed when comparing CSA outputs from different projection methods
    (corner vs centered) against a common reference topography.

    Parameters
    ----------
    data_2D : ndarray
        2D data on source grid (e.g., CSA reconstruction)
    source_cell : topo_cell
        Cell with source planar coordinates (lat, lon in meters)
    target_cell : topo_cell
        Cell with target planar coordinates (lat, lon in meters)

    Returns
    -------
    ndarray
        Data interpolated onto target grid, same shape as target_cell.topo
    """
    # Create source grid coordinates (meshgrid of lat/lon in meters)
    source_lon_grid, source_lat_grid = np.meshgrid(source_cell.lon, source_cell.lat)

    # Create target grid coordinates
    target_lon_grid, target_lat_grid = np.meshgrid(target_cell.lon, target_cell.lat)

    # Flatten source coordinates and data
    source_points = np.column_stack([source_lon_grid.ravel(), source_lat_grid.ravel()])
    source_values = data_2D.ravel()

    # Flatten target coordinates
    target_points = np.column_stack([target_lon_grid.ravel(), target_lat_grid.ravel()])

    # Interpolate using griddata (linear interpolation)
    interpolated_values = interpolate.griddata(
        source_points,
        source_values,
        target_points,
        method="linear",
        fill_value=0.0,  # Fill any out-of-bounds points with 0
    )

    # Reshape back to 2D grid
    interpolated_2D = interpolated_values.reshape(target_cell.topo.shape)

    return interpolated_2D


def create_cell_with_projection(lat_verts, lon_verts, topo, use_center=True, rect=True):
    """
    Create cell using production code path (utils.get_lat_lon_segments).

    Parameters
    ----------
    lat_verts, lon_verts : array
        Vertex coordinates in degrees (processed by handle_latlon_expansion)
    topo : topo_cell
        Topography object
    use_center : bool
        If True, use center of domain as projection origin (NEW method)
        If False, use corner of domain as projection origin (OLD method)
    rect : bool
        If True, use rectangular mask (for FA)
        If False, use triangular mask (for SA)

    Returns
    -------
    cell : topo_cell
        Configured cell object
    """
    cell = var.topo_cell()

    # Use production code path - this includes all preprocessing!
    if rect:
        # FA: Create rectangular cell with filtered topography
        utils.get_lat_lon_segments(
            lat_verts,
            lon_verts,
            cell,
            topo,
            rect=True,
            filtered=True,  # Remove features < 5km
            padding=0,
            use_center=use_center,
        )
    else:
        # SA: Create triangular cell
        # Production calls this twice on the same cell: first rect=True to load topo,
        # then rect=False to apply triangular mask
        # We'll do the same
        utils.get_lat_lon_segments(
            lat_verts,
            lon_verts,
            cell,
            topo,
            rect=True,
            filtered=True,
            padding=0,
            use_center=use_center,
        )
        # Now apply triangular mask
        utils.get_lat_lon_segments(
            lat_verts,
            lon_verts,
            cell,
            topo,
            rect=False,
            filtered=False,
            padding=0,
            use_center=use_center,
        )

    print(f"    use_center={use_center}, rect={rect}")
    print(
        f"    Mask: {cell.mask.sum()} / {cell.mask.size} points ({100*cell.mask.sum()/cell.mask.size:.1f}%)"
    )
    print(f"    cell.lat range: [{cell.lat.min():.1f}, {cell.lat.max():.1f}] m")
    print(f"    cell.lon range: [{cell.lon.min():.1f}, {cell.lon.max():.1f}] m")

    return cell


def run_full_csa(cell, params, use_mode_selection=False):
    """
    Run full CSA algorithm (first + second approximation) on a cell.

    Parameters
    ----------
    cell : topo_cell
        Cell object with topography
    params : params object
        Parameters
    use_mode_selection : bool, optional
        If True, select top n_modes wavenumbers in SA (spectral compression)
        If False, use ALL wavenumbers in SA (full spectrum, better RMSE)
        Default: False (full spectrum)

    Returns
    -------
    tuple : (ampls_fa, ampls_sa, dat_2D_sa, rmse_fa, rmse_sa)
    """
    # First approximation
    fa = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
    ampls_fa, uw_fa, dat_2D_fa = fa.sappx(
        cell, lmbda=params.lmbda_fa, iter_solve=params.fa_iter_solve
    )

    # Compute first approximation RMSE
    diff_fa = cell.topo - dat_2D_fa
    mask = cell.mask if hasattr(cell, "mask") else np.ones_like(cell.topo, dtype=bool)
    rmse_fa = np.sqrt(np.mean(diff_fa[mask] ** 2))

    # Second approximation
    if use_mode_selection:
        # COMPRESSED MODE: Select top n_modes wavenumbers
        # Extract top modes from FA spectrum
        fq_cpy = np.copy(ampls_fa)
        fq_cpy[np.isnan(fq_cpy)] = 0.0

        indices = []
        modes_cnt = 0
        while modes_cnt < params.n_modes:
            max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            indices.append(max_idx)
            fq_cpy[max_idx] = 0.0
            modes_cnt += 1

        k_idxs = [pair[1] for pair in indices]
        l_idxs = [pair[0] for pair in indices]

        # Create new PMF with selected modes only
        sa = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
        sa.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
        ampls_sa, uw_sa, dat_2D_sa = sa.sappx(
            cell, lmbda=params.lmbda_sa, iter_solve=params.sa_iter_solve
        )
    else:
        # FULL SPECTRUM MODE: Use ALL wavenumbers
        sa = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
        ampls_sa, uw_sa, dat_2D_sa = sa.sappx(
            cell, lmbda=params.lmbda_sa, iter_solve=params.sa_iter_solve
        )

    # Compute second approximation RMSE
    diff_sa = cell.topo - dat_2D_sa
    rmse_sa = np.sqrt(np.mean(diff_sa[mask] ** 2))

    return ampls_fa, ampls_sa, dat_2D_sa, rmse_fa, rmse_sa


def plot_single_method(
    c_idx,
    lat,
    topo_orig,
    recon_fa,
    recon_sa,
    rmse_fa,
    rmse_sa,
    mask,
    output_dir,
    method_name,
):
    """
    Create 5-panel plot for a single projection method.

    Panels:
    1. Reference topography
    2. First Approximation reconstruction
    3. Second Approximation reconstruction
    4. First Approximation error map (absolute error)
    5. Second Approximation error map (absolute error)

    Parameters
    ----------
    c_idx : int
        Cell index
    lat : float
        Cell latitude in degrees
    topo_orig : ndarray
        Reference topography
    recon_fa : ndarray
        First approximation reconstruction
    recon_sa : ndarray
        Second approximation reconstruction
    rmse_fa : float
        First approximation RMSE
    rmse_sa : float
        Second approximation RMSE
    mask : ndarray
        Boolean mask for triangular cell
    output_dir : Path
        Output directory
    method_name : str
        'OLD' or 'NEW' for labeling
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Mask the reconstructions for visualization (show only triangular cell)
    recon_fa_masked = np.ma.masked_where(~mask, recon_fa)
    recon_sa_masked = np.ma.masked_where(~mask, recon_sa)
    topo_orig_masked = np.ma.masked_where(~mask, topo_orig)

    vmin = topo_orig[mask].min()
    vmax = topo_orig[mask].max()

    topo_cmap = get_topo_colormap()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    method_label = "Corner-based" if method_name == "OLD" else "Centered"

    # Panel 1: Reference topography
    im1 = axs[0, 0].imshow(
        topo_orig_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0, 0].set_title(
        f"Cell {c_idx} at {lat:.1f}°: Reference Topo\nRange: [{vmin:.0f}, {vmax:.0f}] m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0, 0].set_xlabel("Longitude index")
    axs[0, 0].set_ylabel("Latitude index")
    plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    # Panel 2: First Approximation
    im2 = axs[0, 1].imshow(
        recon_fa_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0, 1].set_title(
        f"{method_name} ({method_label}): 1st Approx\nRMSE: {rmse_fa:.1f} m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0, 1].set_xlabel("Longitude index")
    axs[0, 1].set_ylabel("Latitude index")
    plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    # Panel 3: Second Approximation
    im3 = axs[0, 2].imshow(
        recon_sa_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0, 2].set_title(
        f"{method_name} ({method_label}): 2nd Approx\nRMSE: {rmse_sa:.1f} m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0, 2].set_xlabel("Longitude index")
    axs[0, 2].set_ylabel("Latitude index")
    plt.colorbar(im3, ax=axs[0, 2], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    # Panel 4: First Approximation Error Map
    error_fa = np.abs(topo_orig - recon_fa)
    error_fa_masked = np.ma.masked_where(~mask, error_fa)
    error_max_fa = error_fa[mask].max()

    im4 = axs[1, 0].imshow(
        error_fa_masked,
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=error_max_fa,
        aspect="auto",
    )
    axs[1, 0].set_title(
        f"1st Approx: Absolute Error\nMax: {error_max_fa:.1f} m",
        fontsize=11,
        fontweight="bold",
    )
    axs[1, 0].set_xlabel("Longitude index")
    axs[1, 0].set_ylabel("Latitude index")
    plt.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04).set_label(
        "Absolute Error [m]", rotation=270, labelpad=15
    )

    # Panel 5: Second Approximation Error Map
    error_sa = np.abs(topo_orig - recon_sa)
    error_sa_masked = np.ma.masked_where(~mask, error_sa)
    error_max_sa = error_sa[mask].max()

    im5 = axs[1, 1].imshow(
        error_sa_masked,
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=error_max_sa,
        aspect="auto",
    )
    axs[1, 1].set_title(
        f"2nd Approx: Absolute Error\nMax: {error_max_sa:.1f} m",
        fontsize=11,
        fontweight="bold",
    )
    axs[1, 1].set_xlabel("Longitude index")
    axs[1, 1].set_ylabel("Latitude index")
    plt.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04).set_label(
        "Absolute Error [m]", rotation=270, labelpad=15
    )

    # Panel 6: Statistics summary (text panel)
    axs[1, 2].axis("off")
    stats_text = f"""
    Method: {method_name} ({method_label})
    Cell: {c_idx}
    Latitude: {lat:.2f}°

    Topography Range:
      Min: {vmin:.1f} m
      Max: {vmax:.1f} m

    1st Approximation:
      RMSE: {rmse_fa:.1f} m
      Max Error: {error_max_fa:.1f} m
      Mean Error: {error_fa[mask].mean():.1f} m

    2nd Approximation:
      RMSE: {rmse_sa:.1f} m
      Max Error: {error_max_sa:.1f} m
      Mean Error: {error_sa[mask].mean():.1f} m

    Improvement (FA → SA):
      RMSE: {rmse_fa - rmse_sa:.1f} m
      Reduction: {((rmse_fa - rmse_sa)/rmse_fa*100):.1f}%
    """
    axs[1, 2].text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        transform=axs[1, 2].transAxes,
    )

    plt.tight_layout()
    output_path = (
        output_dir / f"{method_name.lower()}_cell_{c_idx}_lat_{lat:.1f}deg.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plot saved: {output_path}")


def plot_comparison(
    c_idx,
    lat,
    topo_orig,
    recon_old_fa,
    recon_old_sa,
    recon_new_fa,
    recon_new_sa,
    rmse_old_fa,
    rmse_old_sa,
    rmse_new_fa,
    rmse_new_sa,
    mask,
    output_dir,
):
    """
    Create 6-panel comparison plot (FA and SA for both methods).

    All data is on the same grid (centered projection reference).
    OLD method reconstructions have been interpolated to this reference grid.
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Mask the reconstructions for visualization (show only triangular cell)
    recon_old_fa_masked = np.ma.masked_where(~mask, recon_old_fa)
    recon_old_sa_masked = np.ma.masked_where(~mask, recon_old_sa)
    recon_new_fa_masked = np.ma.masked_where(~mask, recon_new_fa)
    recon_new_sa_masked = np.ma.masked_where(~mask, recon_new_sa)
    topo_orig_masked = np.ma.masked_where(~mask, topo_orig)

    vmin = topo_orig[mask].min()
    vmax = topo_orig[mask].max()

    topo_cmap = get_topo_colormap()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Panel 1: Reference topography (centered projection)
    im1 = axs[0, 0].imshow(
        topo_orig_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0, 0].set_title(
        f"Cell {c_idx} at {lat:.1f}°: Reference (Centered)\nRange: [{vmin:.0f}, {vmax:.0f}] m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0, 0].set_xlabel("Longitude index")
    axs[0, 0].set_ylabel("Latitude index")
    plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    # Panel 2: OLD - First Approximation
    im2 = axs[0, 1].imshow(
        recon_old_fa_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0, 1].set_title(
        f"OLD (Corner): 1st Approx\nRMSE: {rmse_old_fa:.1f} m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0, 1].set_xlabel("Longitude index")
    axs[0, 1].set_ylabel("Latitude index")
    plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    # Panel 3: OLD - Second Approximation
    im3 = axs[0, 2].imshow(
        recon_old_sa_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0, 2].set_title(
        f"OLD (Corner): 2nd Approx\nRMSE: {rmse_old_sa:.1f} m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0, 2].set_xlabel("Longitude index")
    axs[0, 2].set_ylabel("Latitude index")
    plt.colorbar(im3, ax=axs[0, 2], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    # Panel 4: Error map (FA)
    error_old_fa = np.abs(topo_orig - recon_old_fa)
    error_new_fa = np.abs(topo_orig - recon_new_fa)
    error_diff_fa = error_old_fa - error_new_fa
    error_diff_fa_masked = np.ma.masked_where(~mask, error_diff_fa)
    error_max_fa = max(
        np.abs(error_diff_fa[mask].min()), np.abs(error_diff_fa[mask].max())
    )

    im4 = axs[1, 0].imshow(
        error_diff_fa_masked,
        origin="lower",
        cmap="RdYlGn",
        vmin=-error_max_fa,
        vmax=error_max_fa,
        aspect="auto",
    )
    imp_fa = (
        ((rmse_old_fa - rmse_new_fa) / rmse_old_fa * 100) if rmse_old_fa > 0 else 0.0
    )
    axs[1, 0].set_title(
        f"1st Approx Improvement\nGreen=Better | Imp: {imp_fa:.1f}%",
        fontsize=11,
        fontweight="bold",
        color="green" if imp_fa > 0 else "red",
    )
    axs[1, 0].set_xlabel("Longitude index")
    axs[1, 0].set_ylabel("Latitude index")
    plt.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04).set_label(
        "Error Reduction [m]", rotation=270, labelpad=15
    )

    # Panel 5: NEW - First Approximation
    im5 = axs[1, 1].imshow(
        recon_new_fa_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[1, 1].set_title(
        f"NEW (Centered): 1st Approx\nRMSE: {rmse_new_fa:.1f} m",
        fontsize=11,
        fontweight="bold",
        color="green",
    )
    axs[1, 1].set_xlabel("Longitude index")
    axs[1, 1].set_ylabel("Latitude index")
    plt.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    # Panel 6: NEW - Second Approximation
    im6 = axs[1, 2].imshow(
        recon_new_sa_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    imp_sa = (
        ((rmse_old_sa - rmse_new_sa) / rmse_old_sa * 100) if rmse_old_sa > 0 else 0.0
    )
    axs[1, 2].set_title(
        f"NEW (Centered): 2nd Approx\nRMSE: {rmse_new_sa:.1f} m | Imp: {imp_sa:.1f}%",
        fontsize=11,
        fontweight="bold",
        color="green",
    )
    axs[1, 2].set_xlabel("Longitude index")
    axs[1, 2].set_ylabel("Latitude index")
    plt.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04).set_label(
        "Elevation [m]", rotation=270, labelpad=15
    )

    plt.tight_layout()
    output_path = output_dir / f"comparison_cell_{c_idx}_lat_{lat:.1f}deg.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plot saved: {output_path}")
    return imp_fa, imp_sa


def main():
    """
    Main test function.

    Tests OLD (corner-based) vs NEW (centered) planar projection methods.

    KEY METHODOLOGY:
    - Creates a SHARED REFERENCE topography using centered projection (geometrically accurate)
    - OLD method: Runs CSA on corner-projection grid, then interpolates to reference grid
    - NEW method: Runs CSA on centered-projection grid (same as reference, no interpolation)
    - Both methods compared against the SAME reference for fair comparison
    """
    # ========================================================================
    # USER CONFIGURATION - MODIFY THESE VALUES
    # ========================================================================

    # PROJECTION METHOD TOGGLE
    # Options: 'BOTH', 'OLD', 'NEW'
    # - 'BOTH': Compare OLD (corner-based) vs NEW (centered) methods side-by-side
    # - 'OLD': Run only OLD (corner-based) projection method
    # - 'NEW': Run only NEW (centered) projection method
    RUN_METHOD = "NEW"  # Change to 'OLD' or 'NEW' to run single method

    # TOPOGRAPHY COARSENING FACTOR
    # Higher values = coarser topography (faster, less memory)
    # Typical values: 1 (full resolution), 2, 4, 8
    ETOPO_CG = 12

    # SPECTRAL COMPRESSION TOGGLE
    # Toggle between full spectrum vs compressed spectrum in second approximation:
    #
    # False (FULL SPECTRUM - default for this test): Use ALL wavenumbers
    #   - Pros: Best reconstruction quality
    #   - Cons: No compression benefit, larger output
    #
    # True (COMPRESSED): Use top n_modes=100 wavenumbers
    #   - Pros: Spectral compression (20x smaller)
    #   - Cons: ~20% higher RMSE
    USE_MODE_SELECTION = True  # Set to True to test compressed mode

    # ========================================================================
    # END USER CONFIGURATION
    # ========================================================================

    print("=" * 80)
    print("CENTERED PROJECTION TEST: Old vs. New Planar Projection")
    print("Testing polar cells (Arctic + Antarctic) at extreme latitudes")
    if RUN_METHOD == "BOTH":
        print("Both methods compared against SHARED REFERENCE (centered projection)")
    elif RUN_METHOD == "OLD":
        print("Running ONLY OLD (corner-based) projection method")
    elif RUN_METHOD == "NEW":
        print("Running ONLY NEW (centered) projection method")
    else:
        raise ValueError(
            f"Invalid RUN_METHOD='{RUN_METHOD}'. Must be 'BOTH', 'OLD', or 'NEW'"
        )
    print("=" * 80)

    # Setup parameters
    from inputs.icon_global_run import params

    params.fn_output = "centered_projection_test"
    params.etopo_cg = ETOPO_CG
    params.dfft_first_guess = False
    params.recompute_rhs = False
    params.plot_output = False

    # CSA parameters
    params.lmbda_fa = 1e-2
    params.lmbda_sa = 1e-1
    params.fa_iter_solve = True
    params.sa_iter_solve = True

    if USE_MODE_SELECTION:
        print(f"*** COMPRESSED MODE: Using top {params.n_modes} wavenumbers ***")
    else:
        print(
            f"*** FULL SPECTRUM MODE: Using ALL {params.nhi * params.nhj} wavenumbers ***"
        )

    if not params.self_test():
        print("ERROR: Parameters failed self-test")
        return

    # Create output directory
    output_dir = Path("outputs/planar_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load ICON grid
    print("\nLoading ICON grid...")
    grid = var.grid()
    reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_dat(params.path_icon_grid, grid)

    clat_rad = np.copy(grid.clat)
    clon_rad = np.copy(grid.clon)
    grid.apply_f(utils.rad2deg)

    # Use pre-selected extreme polar cells
    # These cells are at -88.90°S to -87.21°S (within 1-3° of South Pole)
    # where corner projection creates maximum distortion
    ALL_TEST_CELLS = POLAR_CELLS

    if len(ALL_TEST_CELLS) == 0:
        print("\nERROR: No test cells found. Exiting.")
        return

    print(f"\nTesting {len(ALL_TEST_CELLS)} polar cells (Arctic + Antarctic)")

    # Results storage
    results = []

    # Test each cell
    for c_idx in ALL_TEST_CELLS:
        actual_lat = grid.clat[c_idx]
        actual_lon = grid.clon[c_idx]

        print(f"\n{'='*80}")
        print(
            f"Testing cell {c_idx} at latitude {actual_lat:.2f}°, longitude {actual_lon:.2f}°"
        )
        print(f"{'='*80}")

        # Get cell vertices
        lat_verts = grid.clat_vertices[c_idx]
        lon_verts = grid.clon_vertices[c_idx]
        lat_extent, lon_extent = utils.handle_latlon_expansion(
            lat_verts, lon_verts, lat_expand=0.0, lon_expand=0.0
        )

        params.lat_extent = lat_extent
        params.lon_extent = lon_extent

        # Load topography
        print(f"  Loading topography...")
        topo = var.topo_cell()
        etopo_reader = reader.read_etopo_topo(None, params, is_parallel=True)
        etopo_reader.get_topo(topo)
        topo.topo[np.where(topo.topo < -500.0)] = -500.0
        topo.gen_mgrids()

        # Handle dateline crossing BEFORE processing vertices (like production code)
        if etopo_reader.split_EW:
            lon_verts = lon_verts.copy()  # Don't modify the grid object
            lon_verts[lon_verts < 0.0] += 360.0

        # Process vertices exactly like production code (using dateline-corrected lon_verts!)
        lat_verts_processed, lon_verts_processed = utils.handle_latlon_expansion(
            lat_verts,
            lon_verts,  # Use corrected vertices, not grid originals
            lat_expand=0.0,
            lon_expand=0.0,
        )

        print(
            f"  Vertices (degrees): lat={lat_verts_processed}, lon={lon_verts_processed}"
        )

        # ================================================================
        # CREATE SHARED REFERENCE CELL (Centered Projection - Ground Truth)
        # ================================================================
        # This is the canonical reference topography that BOTH methods will be compared against.
        # Using centered projection (use_center=True) because it's more geometrically accurate,
        # especially at polar latitudes where corner projection introduces maximum distortion.
        print(f"  Creating shared reference cell (centered projection)...")
        cell_reference = create_cell_with_projection(
            lat_verts_processed,
            lon_verts_processed,
            topo,
            use_center=True,
            rect=False,  # Triangular mask for final comparison
        )
        print(
            f"  REFERENCE: {cell_reference.mask.sum()} masked points, "
            f"topo range: [{cell_reference.topo[cell_reference.mask].min():.1f}, "
            f"{cell_reference.topo[cell_reference.mask].max():.1f}] m"
        )

        # Initialize variables for optional methods
        rmse_old_fa, rmse_old_sa = None, None
        rmse_new_fa, rmse_new_sa = None, None
        dat_2D_old_fa_interp, dat_2D_old_sa_interp = None, None
        dat_2D_new_fa, dat_2D_new_sa = None, None

        # TEST 1: OLD projection (corner-based)
        if RUN_METHOD in ["BOTH", "OLD"]:
            print(f"  Running CSA with OLD projection (corner-based)...")

            # FA: Rectangular domain
            print(
                f"    [FA] Creating cell with OLD (corner) projection + rectangular mask..."
            )
            cell_old_fa = create_cell_with_projection(
                lat_verts_processed,
                lon_verts_processed,
                topo,
                use_center=False,
                rect=True,
            )

            # Run FA
            fa_old = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
            ampls_old_fa, uw_old_fa, dat_2D_old_fa = fa_old.sappx(
                cell_old_fa, lmbda=params.lmbda_fa, iter_solve=params.fa_iter_solve
            )

            # SA: Triangular domain
            print(
                f"    [SA] Creating cell with OLD (corner) projection + triangular mask..."
            )
            cell_old_sa = create_cell_with_projection(
                lat_verts_processed,
                lon_verts_processed,
                topo,
                use_center=False,
                rect=False,
            )

            # Run SA
            if USE_MODE_SELECTION:
                # COMPRESSED MODE: Select top n_modes wavenumbers from FA
                ampls_old_fa_copy = np.copy(ampls_old_fa)
                ampls_old_fa_copy[np.isnan(ampls_old_fa_copy)] = 0.0
                indices = []
                for _ in range(params.n_modes):
                    max_idx = np.unravel_index(
                        ampls_old_fa_copy.argmax(), ampls_old_fa_copy.shape
                    )
                    indices.append(max_idx)
                    ampls_old_fa_copy[max_idx] = 0.0
                k_idxs = [pair[1] for pair in indices]
                l_idxs = [pair[0] for pair in indices]
                sa_old = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
                sa_old.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
                ampls_old_sa, uw_old_sa, dat_2D_old_sa = sa_old.sappx(
                    cell_old_sa, lmbda=params.lmbda_sa, iter_solve=params.sa_iter_solve
                )
            else:
                # FULL SPECTRUM MODE: Use all wavenumbers
                sa_old = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
                ampls_old_sa, uw_old_sa, dat_2D_old_sa = sa_old.sappx(
                    cell_old_sa, lmbda=params.lmbda_sa, iter_solve=params.sa_iter_solve
                )

            # Interpolate OLD method outputs from corner-projection grid to reference grid
            print(f"    Interpolating OLD method outputs to reference grid...")
            dat_2D_old_fa_interp = interpolate_to_reference_grid(
                dat_2D_old_fa, cell_old_sa, cell_reference
            )
            dat_2D_old_sa_interp = interpolate_to_reference_grid(
                dat_2D_old_sa, cell_old_sa, cell_reference
            )

            # Compute RMSE against shared reference (centered projection)
            diff_fa = cell_reference.topo - dat_2D_old_fa_interp
            diff_sa = cell_reference.topo - dat_2D_old_sa_interp
            rmse_old_fa = np.sqrt(np.mean(diff_fa[cell_reference.mask] ** 2))
            rmse_old_sa = np.sqrt(np.mean(diff_sa[cell_reference.mask] ** 2))

            print(
                f"    OLD - 1st Approx RMSE (vs shared reference): {rmse_old_fa:.1f} m"
            )
            print(
                f"    OLD - 2nd Approx RMSE (vs shared reference): {rmse_old_sa:.1f} m"
            )

        # TEST 2: NEW projection (centered)
        if RUN_METHOD in ["BOTH", "NEW"]:
            print(f"  Running CSA with NEW projection (centered)...")

            # FA: Rectangular domain
            print(
                f"    [FA] Creating cell with NEW (centered) projection + rectangular mask..."
            )
            cell_new_fa = create_cell_with_projection(
                lat_verts_processed,
                lon_verts_processed,
                topo,
                use_center=True,
                rect=True,
            )

            # Run FA
            fa_new = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
            ampls_new_fa, uw_new_fa, dat_2D_new_fa = fa_new.sappx(
                cell_new_fa, lmbda=params.lmbda_fa, iter_solve=params.fa_iter_solve
            )

            # SA: Triangular domain
            print(
                f"    [SA] Creating cell with NEW (centered) projection + triangular mask..."
            )
            cell_new_sa = create_cell_with_projection(
                lat_verts_processed,
                lon_verts_processed,
                topo,
                use_center=True,
                rect=False,
            )

            # Run SA
            if USE_MODE_SELECTION:
                # COMPRESSED MODE: Select top n_modes wavenumbers from FA
                ampls_new_fa_copy = np.copy(ampls_new_fa)
                ampls_new_fa_copy[np.isnan(ampls_new_fa_copy)] = 0.0
                indices = []
                for _ in range(params.n_modes):
                    max_idx = np.unravel_index(
                        ampls_new_fa_copy.argmax(), ampls_new_fa_copy.shape
                    )
                    indices.append(max_idx)
                    ampls_new_fa_copy[max_idx] = 0.0
                k_idxs = [pair[1] for pair in indices]
                l_idxs = [pair[0] for pair in indices]
                sa_new = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
                sa_new.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
                ampls_new_sa, uw_new_sa, dat_2D_new_sa = sa_new.sappx(
                    cell_new_sa, lmbda=params.lmbda_sa, iter_solve=params.sa_iter_solve
                )
            else:
                # FULL SPECTRUM MODE: Use all wavenumbers
                sa_new = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
                ampls_new_sa, uw_new_sa, dat_2D_new_sa = sa_new.sappx(
                    cell_new_sa, lmbda=params.lmbda_sa, iter_solve=params.sa_iter_solve
                )

            # Compute RMSE against shared reference (no interpolation needed - same grid!)
            # Note: cell_new_sa and cell_reference both use centered projection,
            # so they're on the same planar grid and can be compared directly
            diff_fa = cell_reference.topo - dat_2D_new_fa
            diff_sa = cell_reference.topo - dat_2D_new_sa
            rmse_new_fa = np.sqrt(np.mean(diff_fa[cell_reference.mask] ** 2))
            rmse_new_sa = np.sqrt(np.mean(diff_sa[cell_reference.mask] ** 2))

            print(
                f"    NEW - 1st Approx RMSE (vs shared reference): {rmse_new_fa:.1f} m"
            )
            print(
                f"    NEW - 2nd Approx RMSE (vs shared reference): {rmse_new_sa:.1f} m"
            )

        # Compute improvements (only if BOTH methods were run)
        if RUN_METHOD == "BOTH":
            imp_fa = (
                ((rmse_old_fa - rmse_new_fa) / rmse_old_fa * 100)
                if rmse_old_fa > 0
                else 0.0
            )
            imp_sa = (
                ((rmse_old_sa - rmse_new_sa) / rmse_old_sa * 100)
                if rmse_old_sa > 0
                else 0.0
            )
            print(f"    IMPROVEMENT - 1st Approx: {imp_fa:.1f}%")
            print(f"    IMPROVEMENT - 2nd Approx: {imp_sa:.1f}%")

            # Generate comparison plot using shared reference topography
            # Note: All reconstructions are now on the reference grid (centered projection)
            print(f"  Generating comparison plot...")
            plot_comparison(
                c_idx,
                actual_lat,
                cell_reference.topo,  # Shared reference (centered projection)
                dat_2D_old_fa_interp,
                dat_2D_old_sa_interp,  # OLD method (interpolated to reference grid)
                dat_2D_new_fa,
                dat_2D_new_sa,  # NEW method (already on reference grid)
                rmse_old_fa,
                rmse_old_sa,
                rmse_new_fa,
                rmse_new_sa,
                cell_reference.mask,
                output_dir,  # Use reference mask
            )
        elif RUN_METHOD == "OLD":
            imp_fa = 0.0
            imp_sa = 0.0
            print(f"  Generating visualization plot for OLD method...")
            plot_single_method(
                c_idx,
                actual_lat,
                cell_reference.topo,  # Reference topography
                dat_2D_old_fa_interp,
                dat_2D_old_sa_interp,  # OLD method reconstructions
                rmse_old_fa,
                rmse_old_sa,  # RMSE values
                cell_reference.mask,
                output_dir,  # Mask and output
                method_name="OLD",
            )
        elif RUN_METHOD == "NEW":
            imp_fa = 0.0
            imp_sa = 0.0
            print(f"  Generating visualization plot for NEW method...")
            plot_single_method(
                c_idx,
                actual_lat,
                cell_reference.topo,  # Reference topography
                dat_2D_new_fa,
                dat_2D_new_sa,  # NEW method reconstructions
                rmse_new_fa,
                rmse_new_sa,  # RMSE values
                cell_reference.mask,
                output_dir,  # Mask and output
                method_name="NEW",
            )

        # Store results with region tag
        if actual_lat > 75.0:
            region = "ARCTIC"
        elif actual_lat < -75.0:
            region = "ANTARCTIC"
        else:
            region = "MID-LATITUDE"

        # Only store results if we have data to store
        if RUN_METHOD == "BOTH":
            results.append(
                {
                    "cell_idx": c_idx,
                    "lat": actual_lat,
                    "lon": actual_lon,
                    "region": region,
                    "rmse_old_fa": rmse_old_fa,
                    "rmse_old_sa": rmse_old_sa,
                    "rmse_new_fa": rmse_new_fa,
                    "rmse_new_sa": rmse_new_sa,
                    "imp_fa": imp_fa,
                    "imp_sa": imp_sa,
                }
            )
        elif RUN_METHOD == "OLD":
            results.append(
                {
                    "cell_idx": c_idx,
                    "lat": actual_lat,
                    "lon": actual_lon,
                    "region": region,
                    "rmse_old_fa": rmse_old_fa,
                    "rmse_old_sa": rmse_old_sa,
                    "rmse_new_fa": None,
                    "rmse_new_sa": None,
                    "imp_fa": None,
                    "imp_sa": None,
                }
            )
        elif RUN_METHOD == "NEW":
            results.append(
                {
                    "cell_idx": c_idx,
                    "lat": actual_lat,
                    "lon": actual_lon,
                    "region": region,
                    "rmse_old_fa": None,
                    "rmse_old_sa": None,
                    "rmse_new_fa": rmse_new_fa,
                    "rmse_new_sa": rmse_new_sa,
                    "imp_fa": None,
                    "imp_sa": None,
                }
            )

    # Separate results by region
    arctic_results = [r for r in results if r["region"] == "ARCTIC"]
    antarctic_results = [r for r in results if r["region"] == "ANTARCTIC"]
    mid_lat_results = [r for r in results if r["region"] == "MID-LATITUDE"]

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")

    # Helper function to format RMSE values (handle None)
    def fmt_rmse(val):
        return f"{val:>10.1f}" if val is not None else f"{'N/A':>10}"

    def fmt_imp(val):
        return f"{val:>7.1f}%" if val is not None else f"{'N/A':>8}"

    if arctic_results:
        print("\nARCTIC CELLS (lat > 75°N):")
        print(
            f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}"
        )
        print(f"{'-'*80}")
        for r in arctic_results:
            print(
                f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                f"{fmt_rmse(r['rmse_old_fa'])} {fmt_rmse(r['rmse_new_fa'])} {fmt_imp(r['imp_fa'])} "
                f"{fmt_rmse(r['rmse_old_sa'])} {fmt_rmse(r['rmse_new_sa'])} {fmt_imp(r['imp_sa'])}"
            )
        if RUN_METHOD == "BOTH":
            avg_arctic_fa = np.mean(
                [r["imp_fa"] for r in arctic_results if r["imp_fa"] is not None]
            )
            avg_arctic_sa = np.mean(
                [r["imp_sa"] for r in arctic_results if r["imp_sa"] is not None]
            )
            print(f"  {'Arctic Average - 1st Approx:':>58} {avg_arctic_fa:>7.1f}%")
            print(f"  {'Arctic Average - 2nd Approx:':>58} {avg_arctic_sa:>7.1f}%")

    if antarctic_results:
        print("\nANTARCTIC CELLS (lat < -75°S):")
        print(
            f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}"
        )
        print(f"{'-'*80}")
        for r in antarctic_results:
            print(
                f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                f"{fmt_rmse(r['rmse_old_fa'])} {fmt_rmse(r['rmse_new_fa'])} {fmt_imp(r['imp_fa'])} "
                f"{fmt_rmse(r['rmse_old_sa'])} {fmt_rmse(r['rmse_new_sa'])} {fmt_imp(r['imp_sa'])}"
            )
        if RUN_METHOD == "BOTH":
            avg_antarctic_fa = np.mean(
                [r["imp_fa"] for r in antarctic_results if r["imp_fa"] is not None]
            )
            avg_antarctic_sa = np.mean(
                [r["imp_sa"] for r in antarctic_results if r["imp_sa"] is not None]
            )
            print(
                f"  {'Antarctic Average - 1st Approx:':>58} {avg_antarctic_fa:>7.1f}%"
            )
            print(
                f"  {'Antarctic Average - 2nd Approx:':>58} {avg_antarctic_sa:>7.1f}%"
            )

    if mid_lat_results:
        print("\nMID-LATITUDE CELLS (|lat| < 75°):")
        print(
            f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}"
        )
        print(f"{'-'*80}")
        for r in mid_lat_results:
            print(
                f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                f"{fmt_rmse(r['rmse_old_fa'])} {fmt_rmse(r['rmse_new_fa'])} {fmt_imp(r['imp_fa'])} "
                f"{fmt_rmse(r['rmse_old_sa'])} {fmt_rmse(r['rmse_new_sa'])} {fmt_imp(r['imp_sa'])}"
            )
        if RUN_METHOD == "BOTH":
            avg_mid_lat_fa = np.mean(
                [r["imp_fa"] for r in mid_lat_results if r["imp_fa"] is not None]
            )
            avg_mid_lat_sa = np.mean(
                [r["imp_sa"] for r in mid_lat_results if r["imp_sa"] is not None]
            )
            print(
                f"  {'Mid-Latitude Average - 1st Approx:':>58} {avg_mid_lat_fa:>7.1f}%"
            )
            print(
                f"  {'Mid-Latitude Average - 2nd Approx:':>58} {avg_mid_lat_sa:>7.1f}%"
            )

    # Calculate overall averages (only for BOTH mode)
    if RUN_METHOD == "BOTH":
        avg_imp_fa = np.mean([r["imp_fa"] for r in results if r["imp_fa"] is not None])
        avg_imp_sa = np.mean([r["imp_sa"] for r in results if r["imp_sa"] is not None])
        print(f"\n{'OVERALL Average - 1st Approx:':>60} {avg_imp_fa:>7.1f}%")
        print(f"{'OVERALL Average - 2nd Approx:':>60} {avg_imp_sa:>7.1f}%")

    print(f"\n{'='*80}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*80}")

    # Save results to file
    results_file = output_dir / "results_summary.txt"
    with open(results_file, "w") as f:
        f.write("CENTERED PROJECTION TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Testing {len(results)} cells:\n")
        f.write(f"  Arctic cells (lat > 75°N): {len(arctic_results)}\n")
        f.write(f"  Antarctic cells (lat < -75°S): {len(antarctic_results)}\n")
        f.write(f"  Mid-latitude cells (|lat| < 75°): {len(mid_lat_results)}\n\n")

        if RUN_METHOD == "BOTH":
            f.write(
                f"Comparing OLD (corner-based) vs NEW (centered) planar projection\n"
            )
            f.write(
                f"Running FULL pyCSA: First Approximation + Second Approximation\n\n"
            )
            f.write(
                f"IMPORTANT: Both methods are compared against the SAME reference topography\n"
            )
            f.write(f"           (centered projection, geometrically accurate).\n")
            f.write(
                f"           OLD method reconstructions interpolated to reference grid.\n\n"
            )
        elif RUN_METHOD == "OLD":
            f.write(f"Testing OLD (corner-based) planar projection ONLY\n")
            f.write(
                f"Running FULL pyCSA: First Approximation + Second Approximation\n\n"
            )
        elif RUN_METHOD == "NEW":
            f.write(f"Testing NEW (centered) planar projection ONLY\n")
            f.write(
                f"Running FULL pyCSA: First Approximation + Second Approximation\n\n"
            )

        # Helper function for file writing
        def fmt_rmse_file(val):
            return f"{val:>10.1f}" if val is not None else f"{'N/A':>10}"

        def fmt_imp_file(val):
            return f"{val:>7.1f}%" if val is not None else f"{'N/A':>8}"

        if arctic_results:
            f.write("ARCTIC CELLS (lat > 75°N):\n")
            f.write(
                f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}\n"
            )
            f.write("-" * 80 + "\n")
            for r in arctic_results:
                f.write(
                    f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                    f"{fmt_rmse_file(r['rmse_old_fa'])} {fmt_rmse_file(r['rmse_new_fa'])} {fmt_imp_file(r['imp_fa'])} "
                    f"{fmt_rmse_file(r['rmse_old_sa'])} {fmt_rmse_file(r['rmse_new_sa'])} {fmt_imp_file(r['imp_sa'])}\n"
                )
            if RUN_METHOD == "BOTH":
                avg_arctic_fa = np.mean(
                    [r["imp_fa"] for r in arctic_results if r["imp_fa"] is not None]
                )
                avg_arctic_sa = np.mean(
                    [r["imp_sa"] for r in arctic_results if r["imp_sa"] is not None]
                )
                f.write(
                    f"  {'Arctic Average - 1st Approx:':>58} {avg_arctic_fa:>7.1f}%\n"
                )
                f.write(
                    f"  {'Arctic Average - 2nd Approx:':>58} {avg_arctic_sa:>7.1f}%\n\n"
                )
            else:
                f.write("\n")

        if antarctic_results:
            f.write("ANTARCTIC CELLS (lat < -75°S):\n")
            f.write(
                f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}\n"
            )
            f.write("-" * 80 + "\n")
            for r in antarctic_results:
                f.write(
                    f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                    f"{fmt_rmse_file(r['rmse_old_fa'])} {fmt_rmse_file(r['rmse_new_fa'])} {fmt_imp_file(r['imp_fa'])} "
                    f"{fmt_rmse_file(r['rmse_old_sa'])} {fmt_rmse_file(r['rmse_new_sa'])} {fmt_imp_file(r['imp_sa'])}\n"
                )
            if RUN_METHOD == "BOTH":
                avg_antarctic_fa = np.mean(
                    [r["imp_fa"] for r in antarctic_results if r["imp_fa"] is not None]
                )
                avg_antarctic_sa = np.mean(
                    [r["imp_sa"] for r in antarctic_results if r["imp_sa"] is not None]
                )
                f.write(
                    f"  {'Antarctic Average - 1st Approx:':>58} {avg_antarctic_fa:>7.1f}%\n"
                )
                f.write(
                    f"  {'Antarctic Average - 2nd Approx:':>58} {avg_antarctic_sa:>7.1f}%\n\n"
                )
            else:
                f.write("\n")

        if mid_lat_results:
            f.write("MID-LATITUDE CELLS (|lat| < 75°):\n")
            f.write(
                f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}\n"
            )
            f.write("-" * 80 + "\n")
            for r in mid_lat_results:
                f.write(
                    f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                    f"{fmt_rmse_file(r['rmse_old_fa'])} {fmt_rmse_file(r['rmse_new_fa'])} {fmt_imp_file(r['imp_fa'])} "
                    f"{fmt_rmse_file(r['rmse_old_sa'])} {fmt_rmse_file(r['rmse_new_sa'])} {fmt_imp_file(r['imp_sa'])}\n"
                )
            if RUN_METHOD == "BOTH":
                avg_mid_lat_fa = np.mean(
                    [r["imp_fa"] for r in mid_lat_results if r["imp_fa"] is not None]
                )
                avg_mid_lat_sa = np.mean(
                    [r["imp_sa"] for r in mid_lat_results if r["imp_sa"] is not None]
                )
                f.write(
                    f"  {'Mid-Latitude Average - 1st Approx:':>58} {avg_mid_lat_fa:>7.1f}%\n"
                )
                f.write(
                    f"  {'Mid-Latitude Average - 2nd Approx:':>58} {avg_mid_lat_sa:>7.1f}%\n\n"
                )
            else:
                f.write("\n")

        f.write("-" * 80 + "\n")
        if RUN_METHOD == "BOTH":
            avg_imp_fa = np.mean(
                [r["imp_fa"] for r in results if r["imp_fa"] is not None]
            )
            avg_imp_sa = np.mean(
                [r["imp_sa"] for r in results if r["imp_sa"] is not None]
            )
            f.write(f"{'OVERALL Average - 1st Approx:':>60} {avg_imp_fa:>7.1f}%\n")
            f.write(f"{'OVERALL Average - 2nd Approx:':>60} {avg_imp_sa:>7.1f}%\n")

    print(f"\nResults summary saved to: {results_file}")


if __name__ == "__main__":
    main()
