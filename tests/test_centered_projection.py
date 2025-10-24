"""
Test script to compare old (corner-based) vs. new (centered) planar projection.

Tests 10 pre-selected polar cells (5 Arctic, 5 Antarctic) to evaluate improvement
in pyCSA RMSE when using centered projection instead of corner-based projection.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from pathlib import Path

from pycsa.core import io, var, utils
from pycsa.wrappers import interface


# Pre-selected cell indices from ICON grid
# Testing both POLAR and EQUATORIAL cells to see where centered projection helps

# Polar cells (|lat| > 79.5°) - from previous run, these showed minimal improvement
POLAR_CELLS = [
    3091,   # Arctic: 80.35°N, -92.11°E - Greenland
    # 3105,   # Arctic: 79.77°N, -65.63°E - Greenland
    # 3107,   # Arctic: 79.77°N, -78.37°E - Greenland
    # 3108,   # Arctic: 81.28°N, -57.03°E - Greenland
    # 3109,   # Arctic: 82.56°N, -45.32°E - Greenland
    # 15360,  # Antarctic: -88.90°S, 108.00°E - Interior plateau
    # 15361,  # Antarctic: -87.21°S, 129.75°E - Interior plateau
    # 15362,  # Antarctic: -88.07°S, 108.00°E - Interior plateau
    # 15363,  # Antarctic: -87.21°S, 86.25°E - Interior plateau
    # 15364,  # Antarctic: -85.39°S, 135.26°E - Interior plateau
]

# Equatorial/mid-latitude cells - to test if centered projection helps more here
# Will be populated dynamically to find land cells near equator
# EQUATORIAL_CELLS_CANDIDATES = list(range(0, 25000))  # Will filter for equatorial land
EQUATORIAL_CELLS = [340, 992, 1015]  # To be filled in

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
    return mcolors.LinearSegmentedColormap.from_list('topo', colors)


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
            lat_verts, lon_verts, cell, topo,
            rect=True,
            filtered=True,  # Remove features < 5km
            padding=0,
            use_center=use_center
        )
    else:
        # SA: Create triangular cell
        # Production calls this twice on the same cell: first rect=True to load topo,
        # then rect=False to apply triangular mask
        # We'll do the same
        utils.get_lat_lon_segments(
            lat_verts, lon_verts, cell, topo,
            rect=True,
            filtered=True,
            padding=0,
            use_center=use_center
        )
        # Now apply triangular mask
        utils.get_lat_lon_segments(
            lat_verts, lon_verts, cell, topo,
            rect=False,
            filtered=False,
            padding=0,
            use_center=use_center
        )

    print(f"    use_center={use_center}, rect={rect}")
    print(f"    Mask: {cell.mask.sum()} / {cell.mask.size} points ({100*cell.mask.sum()/cell.mask.size:.1f}%)")
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
    mask = cell.mask if hasattr(cell, 'mask') else np.ones_like(cell.topo, dtype=bool)
    rmse_fa = np.sqrt(np.mean(diff_fa[mask]**2))

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
    rmse_sa = np.sqrt(np.mean(diff_sa[mask]**2))

    return ampls_fa, ampls_sa, dat_2D_sa, rmse_fa, rmse_sa


def plot_comparison(c_idx, lat, topo_orig, recon_old_fa, recon_old_sa,
                    recon_new_fa, recon_new_sa,
                    rmse_old_fa, rmse_old_sa, rmse_new_fa, rmse_new_sa,
                    mask, output_dir):
    """Create 6-panel comparison plot (FA and SA for both methods)."""
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

    # Panel 1: Original topography
    im1 = axs[0, 0].imshow(topo_orig_masked, origin='lower', cmap=topo_cmap, norm=norm, aspect='auto')
    axs[0, 0].set_title(f'Cell {c_idx} at {lat:.1f}°: Original\nRange: [{vmin:.0f}, {vmax:.0f}] m',
                        fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('Longitude index')
    axs[0, 0].set_ylabel('Latitude index')
    plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04).set_label('Elevation [m]', rotation=270, labelpad=15)

    # Panel 2: OLD - First Approximation
    im2 = axs[0, 1].imshow(recon_old_fa_masked, origin='lower', cmap=topo_cmap, norm=norm, aspect='auto')
    axs[0, 1].set_title(f'OLD (Corner): 1st Approx\nRMSE: {rmse_old_fa:.1f} m',
                        fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('Longitude index')
    axs[0, 1].set_ylabel('Latitude index')
    plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04).set_label('Elevation [m]', rotation=270, labelpad=15)

    # Panel 3: OLD - Second Approximation
    im3 = axs[0, 2].imshow(recon_old_sa_masked, origin='lower', cmap=topo_cmap, norm=norm, aspect='auto')
    axs[0, 2].set_title(f'OLD (Corner): 2nd Approx\nRMSE: {rmse_old_sa:.1f} m',
                        fontsize=11, fontweight='bold')
    axs[0, 2].set_xlabel('Longitude index')
    axs[0, 2].set_ylabel('Latitude index')
    plt.colorbar(im3, ax=axs[0, 2], fraction=0.046, pad=0.04).set_label('Elevation [m]', rotation=270, labelpad=15)

    # Panel 4: Error map (FA)
    error_old_fa = np.abs(topo_orig - recon_old_fa)
    error_new_fa = np.abs(topo_orig - recon_new_fa)
    error_diff_fa = error_old_fa - error_new_fa
    error_diff_fa_masked = np.ma.masked_where(~mask, error_diff_fa)
    error_max_fa = max(np.abs(error_diff_fa[mask].min()), np.abs(error_diff_fa[mask].max()))

    im4 = axs[1, 0].imshow(error_diff_fa_masked, origin='lower', cmap='RdYlGn',
                           vmin=-error_max_fa, vmax=error_max_fa, aspect='auto')
    imp_fa = ((rmse_old_fa - rmse_new_fa) / rmse_old_fa * 100) if rmse_old_fa > 0 else 0.0
    axs[1, 0].set_title(f'1st Approx Improvement\nGreen=Better | Imp: {imp_fa:.1f}%',
                        fontsize=11, fontweight='bold', color='green' if imp_fa > 0 else 'red')
    axs[1, 0].set_xlabel('Longitude index')
    axs[1, 0].set_ylabel('Latitude index')
    plt.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04).set_label('Error Reduction [m]', rotation=270, labelpad=15)

    # Panel 5: NEW - First Approximation
    im5 = axs[1, 1].imshow(recon_new_fa_masked, origin='lower', cmap=topo_cmap, norm=norm, aspect='auto')
    axs[1, 1].set_title(f'NEW (Centered): 1st Approx\nRMSE: {rmse_new_fa:.1f} m',
                        fontsize=11, fontweight='bold', color='green')
    axs[1, 1].set_xlabel('Longitude index')
    axs[1, 1].set_ylabel('Latitude index')
    plt.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04).set_label('Elevation [m]', rotation=270, labelpad=15)

    # Panel 6: NEW - Second Approximation
    im6 = axs[1, 2].imshow(recon_new_sa_masked, origin='lower', cmap=topo_cmap, norm=norm, aspect='auto')
    imp_sa = ((rmse_old_sa - rmse_new_sa) / rmse_old_sa * 100) if rmse_old_sa > 0 else 0.0
    axs[1, 2].set_title(f'NEW (Centered): 2nd Approx\nRMSE: {rmse_new_sa:.1f} m | Imp: {imp_sa:.1f}%',
                        fontsize=11, fontweight='bold', color='green')
    axs[1, 2].set_xlabel('Longitude index')
    axs[1, 2].set_ylabel('Latitude index')
    plt.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04).set_label('Elevation [m]', rotation=270, labelpad=15)

    plt.tight_layout()
    output_path = output_dir / f"comparison_cell_{c_idx}_lat_{lat:.1f}deg.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Plot saved: {output_path}")
    return imp_fa, imp_sa


def main():
    """Main test function."""
    print("="*80)
    print("CENTERED PROJECTION TEST: Old vs. New Planar Projection")
    print("Testing equatorial cells (|lat| < 30°) to see if centered projection helps")
    print("="*80)

    # ========================================================================
    # SPECTRAL COMPRESSION TOGGLE
    # ========================================================================
    # Toggle between full spectrum vs compressed spectrum in second approximation:
    #
    # False (FULL SPECTRUM - default for this test): Use ALL wavenumbers
    #   - Pros: Best reconstruction quality
    #   - Cons: No compression benefit, larger output
    #
    # True (COMPRESSED): Use top n_modes=100 wavenumbers
    #   - Pros: Spectral compression (20x smaller)
    #   - Cons: ~20% higher RMSE
    #
    USE_MODE_SELECTION = True  # Set to True to test compressed mode

    # Setup parameters
    from inputs.icon_global_run import params

    params.fn_output = "centered_projection_test"
    params.etopo_cg = 4
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
        print(f"*** FULL SPECTRUM MODE: Using ALL {params.nhi * params.nhj} wavenumbers ***")

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

    # Find equatorial land cells (|lat| < 30° and mean elevation > 100m)
    print("\nSearching for equatorial/mid-latitude land cells...")
    print("Criteria: |latitude| < 30° AND mean elevation > 100m")

    equatorial_land_cells = []

    # Check cells near equator for land
    equatorial_candidates = [i for i in range(len(grid.clat))
                            if abs(grid.clat[i]) < 30.0]

    print(f"Found {len(equatorial_candidates)} equatorial cells (|lat| < 30°)")
    print("Checking which cells are over land with complex terrain...")

    for c_idx in equatorial_candidates:
        if len(equatorial_land_cells) >= 10:
            break

        lat_verts = grid.clat_vertices[c_idx]
        lon_verts = grid.clon_vertices[c_idx]
        lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)

        params.lat_extent = lat_extent
        params.lon_extent = lon_extent

        # Quick check: load topography and check mean elevation + variance
        try:
            topo_check = var.topo_cell()
            etopo_reader = reader.read_etopo_topo(None, params, is_parallel=True)
            etopo_reader.get_topo(topo_check)
            mean_elev = topo_check.topo.mean()
            std_elev = topo_check.topo.std()

            # Land cell with complex terrain (high variance = mountains)
            if mean_elev > 100.0 and std_elev > 200.0:
                equatorial_land_cells.append(c_idx)
                print(f"  Equatorial land cell: {c_idx} at {grid.clat[c_idx]:.2f}°, "
                      f"mean_elev={mean_elev:.0f}m, std={std_elev:.0f}m")
        except:
            continue

    if len(equatorial_land_cells) < 5:
        print(f"\nWARNING: Only found {len(equatorial_land_cells)} equatorial land cells!")
        print("Will combine polar and equatorial cells for testing")

    print(f"\nSelected {len(equatorial_land_cells)} equatorial land cells for testing")

    # Only test equatorial cells
    ALL_TEST_CELLS = POLAR_CELLS#equatorial_land_cells

    if len(ALL_TEST_CELLS) == 0:
        print("\nERROR: No equatorial land cells found. Exiting.")
        return

    print(f"\nTOTAL CELLS TO TEST: {len(ALL_TEST_CELLS)}")

    # Results storage
    results = []

    # Test each cell
    for c_idx in ALL_TEST_CELLS:
        actual_lat = grid.clat[c_idx]
        actual_lon = grid.clon[c_idx]

        print(f"\n{'='*80}")
        print(f"Testing cell {c_idx} at latitude {actual_lat:.2f}°, longitude {actual_lon:.2f}°")
        print(f"{'='*80}")

        # Get cell vertices
        lat_verts = grid.clat_vertices[c_idx]
        lon_verts = grid.clon_vertices[c_idx]
        lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)

        params.lat_extent = lat_extent
        params.lon_extent = lon_extent

        # Load topography
        print(f"  Loading topography...")
        topo = var.topo_cell()
        etopo_reader = reader.read_etopo_topo(None, params, is_parallel=True)
        etopo_reader.get_topo(topo)
        topo.topo[np.where(topo.topo < -500.0)] = -500.0
        topo.gen_mgrids()

        # Handle dateline crossing
        if etopo_reader.split_EW:
            lon_verts[lon_verts < 0.0] += 360.0

        # Process vertices exactly like production code
        lat_verts_processed, lon_verts_processed = utils.handle_latlon_expansion(
            grid.clat_vertices[c_idx], grid.clon_vertices[c_idx],
            lat_expand=0.0, lon_expand=0.0
        )

        print(f"  Vertices (degrees): lat={lat_verts_processed}, lon={lon_verts_processed}")

        # TEST 1: OLD projection (corner-based)
        print(f"  Running CSA with OLD projection (corner-based)...")

        # FA: Rectangular domain
        print(f"    [FA] Creating cell with OLD (corner) projection + rectangular mask...")
        cell_old_fa = create_cell_with_projection(
            lat_verts_processed, lon_verts_processed, topo,
            use_center=False, rect=True
        )

        # Run FA
        fa_old = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
        ampls_old_fa, uw_old_fa, dat_2D_old_fa = fa_old.sappx(
            cell_old_fa, lmbda=params.lmbda_fa, iter_solve=params.fa_iter_solve
        )

        # SA: Triangular domain
        print(f"    [SA] Creating cell with OLD (corner) projection + triangular mask...")
        cell_old_sa = create_cell_with_projection(
            lat_verts_processed, lon_verts_processed, topo,
            use_center=False, rect=False
        )

        # Run SA
        if USE_MODE_SELECTION:
            # COMPRESSED MODE: Select top n_modes wavenumbers from FA
            ampls_old_fa_copy = np.copy(ampls_old_fa)
            ampls_old_fa_copy[np.isnan(ampls_old_fa_copy)] = 0.0
            indices = []
            for _ in range(params.n_modes):
                max_idx = np.unravel_index(ampls_old_fa_copy.argmax(), ampls_old_fa_copy.shape)
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

        # Compute RMSE on triangular mask only
        diff_fa = cell_old_sa.topo - dat_2D_old_fa  # Use SA cell's topo (same domain, just different mask)
        diff_sa = cell_old_sa.topo - dat_2D_old_sa
        rmse_old_fa = np.sqrt(np.mean(diff_fa[cell_old_sa.mask]**2))
        rmse_old_sa = np.sqrt(np.mean(diff_sa[cell_old_sa.mask]**2))

        print(f"    OLD - 1st Approx RMSE: {rmse_old_fa:.1f} m")
        print(f"    OLD - 2nd Approx RMSE: {rmse_old_sa:.1f} m")

        # TEST 2: NEW projection (centered)
        print(f"  Running CSA with NEW projection (centered)...")

        # FA: Rectangular domain
        print(f"    [FA] Creating cell with NEW (centered) projection + rectangular mask...")
        cell_new_fa = create_cell_with_projection(
            lat_verts_processed, lon_verts_processed, topo,
            use_center=True, rect=True
        )

        # Run FA
        fa_new = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
        ampls_new_fa, uw_new_fa, dat_2D_new_fa = fa_new.sappx(
            cell_new_fa, lmbda=params.lmbda_fa, iter_solve=params.fa_iter_solve
        )

        # SA: Triangular domain
        print(f"    [SA] Creating cell with NEW (centered) projection + triangular mask...")
        cell_new_sa = create_cell_with_projection(
            lat_verts_processed, lon_verts_processed, topo,
            use_center=True, rect=False
        )

        # Run SA
        if USE_MODE_SELECTION:
            # COMPRESSED MODE: Select top n_modes wavenumbers from FA
            ampls_new_fa_copy = np.copy(ampls_new_fa)
            ampls_new_fa_copy[np.isnan(ampls_new_fa_copy)] = 0.0
            indices = []
            for _ in range(params.n_modes):
                max_idx = np.unravel_index(ampls_new_fa_copy.argmax(), ampls_new_fa_copy.shape)
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

        # Compute RMSE on triangular mask only
        diff_fa = cell_new_sa.topo - dat_2D_new_fa
        diff_sa = cell_new_sa.topo - dat_2D_new_sa
        rmse_new_fa = np.sqrt(np.mean(diff_fa[cell_new_sa.mask]**2))
        rmse_new_sa = np.sqrt(np.mean(diff_sa[cell_new_sa.mask]**2))

        print(f"    NEW - 1st Approx RMSE: {rmse_new_fa:.1f} m")
        print(f"    NEW - 2nd Approx RMSE: {rmse_new_sa:.1f} m")

        # Compute improvements
        imp_fa = ((rmse_old_fa - rmse_new_fa) / rmse_old_fa * 100) if rmse_old_fa > 0 else 0.0
        imp_sa = ((rmse_old_sa - rmse_new_sa) / rmse_old_sa * 100) if rmse_old_sa > 0 else 0.0
        print(f"    IMPROVEMENT - 1st Approx: {imp_fa:.1f}%")
        print(f"    IMPROVEMENT - 2nd Approx: {imp_sa:.1f}%")

        # Generate comparison plot (use SA cell's triangular mask)
        print(f"  Generating comparison plot...")
        plot_comparison(
            c_idx, actual_lat,
            cell_old_sa.topo, dat_2D_old_fa, dat_2D_old_sa,
            dat_2D_new_fa, dat_2D_new_sa,
            rmse_old_fa, rmse_old_sa, rmse_new_fa, rmse_new_sa,
            cell_old_sa.mask, output_dir
        )

        # Store results with region tag
        is_polar = abs(actual_lat) > 79.5
        results.append({
            'cell_idx': c_idx,
            'lat': actual_lat,
            'lon': actual_lon,
            'region': 'POLAR' if is_polar else 'EQUATOR',
            'rmse_old_fa': rmse_old_fa,
            'rmse_old_sa': rmse_old_sa,
            'rmse_new_fa': rmse_new_fa,
            'rmse_new_sa': rmse_new_sa,
            'imp_fa': imp_fa,
            'imp_sa': imp_sa,
        })

    # Separate results by region
    polar_results = [r for r in results if r['region'] == 'POLAR']
    equatorial_results = [r for r in results if r['region'] == 'EQUATOR']

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")

    if polar_results:
        print("\nPOLAR CELLS (|lat| > 79.5°):")
        print(f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}")
        print(f"{'-'*80}")
        for r in polar_results:
            print(f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                  f"{r['rmse_old_fa']:>10.1f} {r['rmse_new_fa']:>10.1f} {r['imp_fa']:>7.1f}% "
                  f"{r['rmse_old_sa']:>10.1f} {r['rmse_new_sa']:>10.1f} {r['imp_sa']:>7.1f}%")
        avg_polar_fa = np.mean([r['imp_fa'] for r in polar_results])
        avg_polar_sa = np.mean([r['imp_sa'] for r in polar_results])
        print(f"  {'Polar Average - 1st Approx:':>58} {avg_polar_fa:>7.1f}%")
        print(f"  {'Polar Average - 2nd Approx:':>58} {avg_polar_sa:>7.1f}%")

    if equatorial_results:
        print("\nEQUATORIAL CELLS (|lat| < 30°):")
        print(f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}")
        print(f"{'-'*80}")
        for r in equatorial_results:
            print(f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                  f"{r['rmse_old_fa']:>10.1f} {r['rmse_new_fa']:>10.1f} {r['imp_fa']:>7.1f}% "
                  f"{r['rmse_old_sa']:>10.1f} {r['rmse_new_sa']:>10.1f} {r['imp_sa']:>7.1f}%")
        avg_equator_fa = np.mean([r['imp_fa'] for r in equatorial_results])
        avg_equator_sa = np.mean([r['imp_sa'] for r in equatorial_results])
        print(f"  {'Equatorial Average - 1st Approx:':>58} {avg_equator_fa:>7.1f}%")
        print(f"  {'Equatorial Average - 2nd Approx:':>58} {avg_equator_sa:>7.1f}%")

    # Calculate overall averages
    avg_imp_fa = np.mean([r['imp_fa'] for r in results])
    avg_imp_sa = np.mean([r['imp_sa'] for r in results])
    print(f"\n{'OVERALL Average - 1st Approx:':>60} {avg_imp_fa:>7.1f}%")
    print(f"{'OVERALL Average - 2nd Approx:':>60} {avg_imp_sa:>7.1f}%")

    print(f"\n{'='*80}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*80}")

    # Save results to file
    results_file = output_dir / "results_summary.txt"
    with open(results_file, 'w') as f:
        f.write("CENTERED PROJECTION TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Testing {len(results)} cells:\n")
        f.write(f"  Polar cells (|lat| > 79.5°): {len(polar_results)}\n")
        f.write(f"  Equatorial cells (|lat| < 30°): {len(equatorial_results)}\n")
        f.write(f"Comparing OLD (corner-based) vs NEW (centered) planar projection\n")
        f.write(f"Running FULL pyCSA: First Approximation + Second Approximation\n\n")

        if polar_results:
            f.write("POLAR CELLS (|lat| > 79.5°):\n")
            f.write(f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}\n")
            f.write("-"*80 + "\n")
            for r in polar_results:
                f.write(f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                       f"{r['rmse_old_fa']:>10.1f} {r['rmse_new_fa']:>10.1f} {r['imp_fa']:>7.1f}% "
                       f"{r['rmse_old_sa']:>10.1f} {r['rmse_new_sa']:>10.1f} {r['imp_sa']:>7.1f}%\n")
            f.write(f"  {'Polar Average - 1st Approx:':>58} {avg_polar_fa:>7.1f}%\n")
            f.write(f"  {'Polar Average - 2nd Approx:':>58} {avg_polar_sa:>7.1f}%\n\n")

        if equatorial_results:
            f.write("EQUATORIAL CELLS (|lat| < 30°):\n")
            f.write(f"{'Cell':>6} {'Lat':>8} {'Lon':>8} {'OLD FA':>10} {'NEW FA':>10} {'Imp FA':>8} {'OLD SA':>10} {'NEW SA':>10} {'Imp SA':>8}\n")
            f.write("-"*80 + "\n")
            for r in equatorial_results:
                f.write(f"{r['cell_idx']:>6d} {r['lat']:>8.2f} {r['lon']:>8.2f} "
                       f"{r['rmse_old_fa']:>10.1f} {r['rmse_new_fa']:>10.1f} {r['imp_fa']:>7.1f}% "
                       f"{r['rmse_old_sa']:>10.1f} {r['rmse_new_sa']:>10.1f} {r['imp_sa']:>7.1f}%\n")
            f.write(f"  {'Equatorial Average - 1st Approx:':>58} {avg_equator_fa:>7.1f}%\n")
            f.write(f"  {'Equatorial Average - 2nd Approx:':>58} {avg_equator_sa:>7.1f}%\n\n")

        f.write("-"*80 + "\n")
        f.write(f"{'OVERALL Average - 1st Approx:':>60} {avg_imp_fa:>7.1f}%\n")
        f.write(f"{'OVERALL Average - 2nd Approx:':>60} {avg_imp_sa:>7.1f}%\n")

    print(f"\nResults summary saved to: {results_file}")


if __name__ == '__main__':
    main()
