#!/usr/bin/env python3
"""
Verify ETOPO Land/Ocean Cell Counts

This script loads the ICON grid and ETOPO topography data, counts how many
cells are land vs ocean, and creates comprehensive plots.

Usage:
    python verify_icon_etopo_land_ocean.py              # Full verification + plotting
    python verify_icon_etopo_land_ocean.py --plot-only  # Load saved data and plot only
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from pathlib import Path

def get_topo_colormap():
    """
    Create a topography colormap with blue for ocean (< 0m) and terrain colors for land (> 0m).
    """
    # Ocean colors (blue shades from deep to shallow)
    ocean_colors = plt.cm.Blues_r(np.linspace(0.4, 0.95, 120))

    # Smooth transition zone around sea level
    last_ocean = plt.cm.Blues_r(0.95)
    first_land = plt.cm.terrain(0.25)

    # Create smooth blend from ocean to land
    transition_colors = np.zeros((16, 4))
    for i in range(4):  # RGBA channels
        transition_colors[:, i] = np.linspace(last_ocean[i], first_land[i], 16)

    # Land colors (terrain-like: green to brown to white)
    land_colors = plt.cm.terrain(np.linspace(0.28, 1.0, 120))

    # Combine: 120 ocean + 16 transition + 120 land = 256 total
    colors = np.vstack((ocean_colors, transition_colors, land_colors))
    return mcolors.LinearSegmentedColormap.from_list('topo', colors)


def count_land_ocean_cells(grid, params, reader):
    """
    Count how many cells in the ICON grid are land vs ocean based on ETOPO data.
    Also computes land fraction for each cell for gradient visualization.

    Parameters
    ----------
    grid : grid object
        ICON grid (in degrees)
    params : params object
        Parameters with ETOPO settings
    reader : ncdata object
        Data reader

    Returns
    -------
    tuple
        (land_count, ocean_count, land_cells, ocean_cells, land_fractions)
        land_cells and ocean_cells are lists of cell indices
        land_fractions is array of land fraction [0-1] for each cell
    """
    n_cells = grid.clat.size
    land_cells = []
    ocean_cells = []
    land_fractions = np.zeros(n_cells)  # Store land fraction for each cell

    print(f"Checking {n_cells} cells for land/ocean classification...")

    for c_idx in range(n_cells):
        if c_idx % 1000 == 0:
            print(f"  Processing cell {c_idx}/{n_cells}...")

        topo = var.topo_cell()

        lat_verts = grid.clat_vertices[c_idx]
        lon_verts = grid.clon_vertices[c_idx]

        # Determine lat/lon extents
        lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)

        params.lat_extent = lat_extent
        params.lon_extent = lon_extent

        # Load topography data
        etopo_reader = reader.read_etopo_topo(None, params, is_parallel=True)
        etopo_reader.get_topo(topo)

        # Clip deep bathymetry to -500m
        topo.topo[np.where(topo.topo < -500.0)] = -500.0
        topo.gen_mgrids()

        # Handle dateline crossing
        if etopo_reader.split_EW:
            lon_verts = lon_verts.copy()
            lon_verts[lon_verts < 0.0] += 360.0

        # Process vertices for CSA
        lat_verts, lon_verts = utils.handle_latlon_expansion(
            lat_verts, lon_verts, lat_expand=0.0, lon_expand=0.0
        )

        # Initialize cell objects
        tri_idx = 0
        cell = var.topo_cell()
        tri = var.obj()

        # Set up triangles
        clon_vertices = np.array([lon_verts])
        clat_vertices = np.array([lat_verts])
        ncells = 1
        nv = clon_vertices[0].size

        triangles = np.zeros((ncells, nv, 2))
        triangles[0, :, 0] = clon_vertices[0, :]
        triangles[0, :, 1] = clat_vertices[0, :]

        tri.tri_lon_verts = triangles[:, :, 0]
        tri.tri_lat_verts = triangles[:, :, 1]

        simplex_lat = tri.tri_lat_verts[tri_idx]
        simplex_lon = tri.tri_lon_verts[tri_idx]

        # Check if land (binary classification)
        is_land_cell = utils.is_land(cell, simplex_lat, simplex_lon, topo)

        # Calculate land fraction (fraction of cell with elevation > 0m)
        land_points = np.sum(cell.topo > 0.0)
        total_points = cell.topo.size
        land_fractions[c_idx] = land_points / total_points if total_points > 0 else 0.0

        if is_land_cell:
            land_cells.append(c_idx)
        else:
            ocean_cells.append(c_idx)

    return len(land_cells), len(ocean_cells), land_cells, ocean_cells, land_fractions


def create_comprehensive_plots(clat_deg, clon_deg, land_cells, ocean_cells, land_fractions, output_dir):
    """
    Create comprehensive plots of land/ocean classification.

    Parameters
    ----------
    clat_deg : array
        Cell latitudes in degrees
    clon_deg : array
        Cell longitudes in degrees
    land_cells : list
        List of land cell indices
    ocean_cells : list
        List of ocean cell indices
    land_fractions : array
        Array of land fraction [0-1] for each cell
    output_dir : Path
        Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    land_count = len(land_cells)
    ocean_count = len(ocean_cells)

    # Convert to Mollweide projection coordinates
    lon_plot = np.deg2rad(clon_deg)
    lon_plot[lon_plot > np.pi] -= 2*np.pi
    lat_plot = np.deg2rad(clat_deg)

    # Custom colormap from blue (ocean) to green (land)
    colors_gradient = ['#0033aa', '#0066cc', '#3399ff', '#66ccff',
                       '#99ff99', '#66cc66', '#339933', '#006600']
    cmap_land_ocean = LinearSegmentedColormap.from_list('land_ocean', colors_gradient, N=256)

    # ========================================================================
    # Figure 1: Multiple global views with different thresholds
    # ========================================================================
    print("  Creating global overview plots...")
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Continuous land fraction
    ax1 = fig.add_subplot(231, projection='mollweide')
    scatter1 = ax1.scatter(lon_plot, lat_plot,
                          c=land_fractions,
                          cmap=cmap_land_ocean,
                          s=5,
                          alpha=0.9,
                          vmin=0.0,
                          vmax=1.0,
                          edgecolors='none')
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar1.set_label('Land Fraction', fontsize=10)
    ax1.set_title(f'Continuous Land Fraction\n(All gradations)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Binary classification (>50% land = green, else blue)
    ax2 = fig.add_subplot(232, projection='mollweide')
    binary_colors = np.where(land_fractions > 0.5, '#228B22', '#1E90FF')
    ax2.scatter(lon_plot, lat_plot,
               c=binary_colors,
               s=5,
               alpha=0.9,
               edgecolors='none')
    ax2.set_title(f'Binary: >50% Land = Green\nLand: {land_count}, Ocean: {ocean_count}',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Highlight mixed coastal cells (10-90% land)
    ax3 = fig.add_subplot(233, projection='mollweide')
    coastal_mask = (land_fractions > 0.1) & (land_fractions < 0.9)
    pure_land_mask = land_fractions >= 0.9
    pure_ocean_mask = land_fractions <= 0.1

    if np.any(pure_ocean_mask):
        ax3.scatter(lon_plot[pure_ocean_mask], lat_plot[pure_ocean_mask],
                   c='#B0E0E6', s=4, alpha=0.5, label='Pure Ocean (<10% land)')
    if np.any(pure_land_mask):
        ax3.scatter(lon_plot[pure_land_mask], lat_plot[pure_land_mask],
                   c='#90EE90', s=4, alpha=0.5, label='Pure Land (>90% land)')
    if np.any(coastal_mask):
        ax3.scatter(lon_plot[coastal_mask], lat_plot[coastal_mask],
                   c='#FF6347', s=8, alpha=0.9, label=f'Mixed Coastal (10-90% land)')

    ax3.set_title(f'Coastal/Mixed Cells Highlighted\n{np.sum(coastal_mask)} mixed cells',
                  fontsize=11, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=8, markerscale=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Grid structure
    ax4 = fig.add_subplot(234, projection='mollweide')
    ax4.scatter(lon_plot, lat_plot,
               c='gray', s=2, alpha=0.6)
    ax4.set_title(f'ICON R2B4 Grid Structure\n{len(clat_deg)} cells total',
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Only cells with ANY land (>5% threshold)
    ax5 = fig.add_subplot(235, projection='mollweide')
    any_land_mask = land_fractions > 0.05
    if np.any(~any_land_mask):
        ax5.scatter(lon_plot[~any_land_mask], lat_plot[~any_land_mask],
                   c='#1E90FF', s=3, alpha=0.3, label='Pure Ocean')
    if np.any(any_land_mask):
        scatter5 = ax5.scatter(lon_plot[any_land_mask], lat_plot[any_land_mask],
                              c=land_fractions[any_land_mask],
                              cmap=cmap_land_ocean,
                              s=8,
                              alpha=0.9,
                              vmin=0.0,
                              vmax=1.0,
                              edgecolors='none',
                              label='Has Land')
    ax5.set_title(f'Cells with >5% Land Highlighted\n{np.sum(any_land_mask)} cells with land',
                  fontsize=11, fontweight='bold')
    ax5.legend(loc='lower left', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Latitude distribution
    ax6 = fig.add_subplot(236)
    lat_bins = np.linspace(-90, 90, 37)

    pure_ocean_hist, _ = np.histogram(clat_deg[land_fractions <= 0.1], bins=lat_bins)
    coastal_hist, _ = np.histogram(clat_deg[coastal_mask], bins=lat_bins)
    pure_land_hist, _ = np.histogram(clat_deg[land_fractions >= 0.9], bins=lat_bins)

    bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    width = 5

    ax6.barh(bin_centers, pure_ocean_hist, height=width,
            color='#1E90FF', alpha=0.6, label='Pure Ocean (≤10% land)')
    ax6.barh(bin_centers, coastal_hist, height=width, left=pure_ocean_hist,
            color='#FF6347', alpha=0.6, label='Coastal (10-90% land)')
    ax6.barh(bin_centers, pure_land_hist, height=width,
            left=pure_ocean_hist+coastal_hist,
            color='#228B22', alpha=0.6, label='Pure Land (≥90% land)')

    ax6.set_xlabel('Number of cells', fontsize=10)
    ax6.set_ylabel('Latitude [degrees]', fontsize=10)
    ax6.set_title('Cell Distribution by Latitude', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "improved_verification_plots.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # ========================================================================
    # Figure 2: Pacific region details
    # ========================================================================
    print("  Creating Pacific region detail plots...")

    regions = {
        'Hawaii': (15, 25, -165, -150),
        'Micronesia': (0, 15, 130, 170),
        'Polynesia': (-30, 0, -180, -130),
        'Indonesia': (-10, 10, 95, 140),
    }

    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (name, (lat_min, lat_max, lon_min, lon_max)) in enumerate(regions.items()):
        ax = axes[idx]

        # Find cells in region
        mask = (
            (clat_deg >= lat_min) & (clat_deg <= lat_max) &
            (clon_deg >= lon_min) & (clon_deg <= lon_max)
        )

        # Separate by land fraction
        pure_ocean = mask & (land_fractions < 0.05)
        has_land = mask & (land_fractions >= 0.05)

        # Plot
        if np.any(pure_ocean):
            ax.scatter(clon_deg[pure_ocean], clat_deg[pure_ocean],
                      c='#E0F2F7', s=80, alpha=0.5,
                      edgecolors='gray', linewidths=0.3,
                      label='Ocean (<5% land)')

        sc = None  # Initialize scatter plot variable
        if np.any(has_land):
            sc = ax.scatter(clon_deg[has_land], clat_deg[has_land],
                           c=land_fractions[has_land],
                           cmap=cmap_land_ocean,
                           s=120,
                           alpha=0.95,
                           vmin=0.0,
                           vmax=1.0,
                           edgecolors='black',
                           linewidths=0.8)

            # Add cell percentages for high land fraction
            high_land = has_land & (land_fractions > 0.3)
            for cell_idx in np.where(high_land)[0]:
                ax.text(clon_deg[cell_idx], clat_deg[cell_idx],
                       f'{100*land_fractions[cell_idx]:.0f}%',
                       fontsize=7, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Format
        ax.set_xlabel('Longitude [°]', fontsize=10)
        ax.set_ylabel('Latitude [°]', fontsize=10)
        ax.set_title(f'{name} Region\n{np.sum(has_land)} cells with ≥5% land, '
                    f'{np.sum(pure_ocean)} pure ocean cells',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    plt.tight_layout()

    # Add colorbar at the bottom (if we have scatter data)
    if sc is not None:
        cbar_ax = fig2.add_axes([0.25, -0.02, 0.5, 0.02])
        cbar = fig2.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Land Fraction (0=Ocean, 1=Land)', fontsize=11)

    output_file2 = output_dir / "pacific_islands_detail.png"
    plt.savefig(output_file2, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file2}")
    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Pure ocean cells (≤10% land): {np.sum(land_fractions <= 0.1)}")
    print(f"Coastal/mixed cells (10-90% land): {np.sum(coastal_mask)}")
    print(f"Pure land cells (≥90% land): {np.sum(land_fractions >= 0.9)}")
    print()
    print(f"Mean land fraction: {np.mean(land_fractions):.3f}")
    print(f"Median land fraction: {np.median(land_fractions):.3f}")
    print()

    # Pacific statistics
    for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
        mask = (
            (clat_deg >= lat_min) & (clat_deg <= lat_max) &
            (clon_deg >= lon_min) & (clon_deg <= lon_max)
        )
        has_land = mask & (land_fractions >= 0.05)

        if np.any(has_land):
            print(f"{name}:")
            print(f"  Cells with land: {np.sum(has_land)}")
            print(f"  Max land fraction: {np.max(land_fractions[has_land]):.1%}")
            print(f"  Mean land fraction: {np.mean(land_fractions[has_land]):.1%}")

    print("="*80)


def load_saved_data(data_file):
    """Load previously saved verification data."""
    if not data_file.exists():
        print(f"Error: {data_file} not found.")
        print("Please run verification first without --plot-only flag.")
        sys.exit(1)

    data = np.load(data_file)
    print(f"Loaded verification data from: {data_file}")
    print(f"  Total cells: {data['n_cells']}")
    print(f"  Land cells: {data['land_count']}")
    print(f"  Ocean cells: {data['ocean_count']}")
    print(f"  ETOPO coarse-graining: {data['etopo_cg']}")
    print()

    return (
        data['clat_deg'],
        data['clon_deg'],
        list(data['land_cells']),
        list(data['ocean_cells']),
        data['land_fractions']
    )


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Verify ETOPO land/ocean classification and create plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_icon_etopo_land_ocean.py              # Full verification + plotting
  python verify_icon_etopo_land_ocean.py --plot-only  # Load saved data and plot only
        """
    )
    parser.add_argument('--plot-only', action='store_true',
                        help='Only create plots from saved data (skip verification)')
    args = parser.parse_args()

    print("="*80)
    print("ETOPO LAND/OCEAN VERIFICATION")
    print("="*80)

    output_dir = Path("outputs") / "verification"
    data_file = output_dir / "verification_data.npz"

    if args.plot_only:
        # Plot-only mode: Load saved data
        print("\nMode: PLOT ONLY (loading saved data)")
        print("="*80)
        clat_deg, clon_deg, land_cells, ocean_cells, land_fractions = load_saved_data(data_file)

    else:
        # Full verification mode
        print("\nMode: FULL VERIFICATION (compute + save + plot)")
        print("="*80)

        # Import modules needed for verification
        from pycsa.core import io, var, utils
        from inputs.icon_global_run import params

        # Load ICON grid
        print("\nLoading ICON grid...")
        grid = var.grid()
        reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
        reader.read_dat(params.path_icon_grid, grid)

        # Store radians for later use
        clat_rad = np.copy(grid.clat)
        clon_rad = np.copy(grid.clon)

        # Convert to degrees for processing
        grid.apply_f(utils.rad2deg)

        n_cells = grid.clat.size
        print(f"  Total cells in grid: {n_cells}")

        # Set ETOPO parameters
        params.etopo_cg = 4  # Coarse-graining factor (matches processing used in icon_etopo_global_hpc.py)

        # Count land/ocean cells
        print("\nCounting land/ocean cells...")
        land_count, ocean_count, land_cells, ocean_cells, land_fractions = count_land_ocean_cells(
            grid, params, reader
        )

        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Total cells: {n_cells}")
        print(f"Land cells (is_land=1): {land_count}")
        print(f"Ocean cells (is_land=0): {ocean_count}")
        print(f"Land/Ocean ratio: {land_count}/{ocean_count} = {land_count/ocean_count:.3f}")
        print(f"Land percentage: {100*land_count/(land_count+ocean_count):.2f}%")
        print("="*80)

        # Save plotting data for debugging
        print("\nSaving verification data...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert grid coordinates to degrees for saving
        clat_deg = np.rad2deg(clat_rad)
        clon_deg = np.rad2deg(clon_rad)

        # Save as compressed numpy file
        np.savez_compressed(
            data_file,
            clat_deg=clat_deg,
            clon_deg=clon_deg,
            land_cells=np.array(land_cells),
            ocean_cells=np.array(ocean_cells),
            land_fractions=land_fractions,
            n_cells=n_cells,
            land_count=land_count,
            ocean_count=ocean_count,
            etopo_cg=params.etopo_cg
        )
        print(f"  Data saved: {data_file}")
        print(f"  Contains: cell coordinates, land/ocean classifications, land fractions, and counts")

    # Create comprehensive plots (both modes)
    print("\nCreating comprehensive plots...")
    create_comprehensive_plots(clat_deg, clon_deg, land_cells, ocean_cells, land_fractions, output_dir)

    print("\n✓ Complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Plots created:")
    print(f"    - improved_verification_plots.png")
    print(f"    - pacific_islands_detail.png")
