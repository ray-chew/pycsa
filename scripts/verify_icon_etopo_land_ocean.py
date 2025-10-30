#!/usr/bin/env python3
"""
Verify ETOPO Land/Ocean Cell Counts

This script loads the ICON grid and ETOPO topography data, counts how many
cells are land vs ocean, and creates plots of:
1. Global ETOPO topography
2. ICON R2B4 grid overlay on topography
3. Land/Ocean cell distribution
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from pathlib import Path

from pycsa.core import io, var, utils
from inputs.icon_global_run import params

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


def plot_global_topography_and_grid(grid, params, reader, land_cells, ocean_cells, land_fractions, output_dir):
    """
    Create plots of global topography and ICON grid.

    Parameters
    ----------
    grid : grid object
        ICON grid (in degrees)
    params : params object
        Parameters
    reader : ncdata object
        Data reader
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

    # Convert grid coordinates to degrees for plotting
    clat_deg = np.rad2deg(grid.clat)
    clon_deg = np.rad2deg(grid.clon)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))

    # Plot 1: Land/Ocean classification on grid with gradient
    ax1 = fig.add_subplot(131, projection='mollweide')

    # Convert to Mollweide projection coordinates (radians, but centered at 0)
    lon_plot = np.deg2rad(clon_deg)
    lon_plot[lon_plot > np.pi] -= 2*np.pi  # Wrap to [-π, π]
    lat_plot = np.deg2rad(clat_deg)

    # Create a custom colormap from blue (ocean) to green (land)
    from matplotlib.colors import LinearSegmentedColormap
    colors_gradient = ['#0066cc', '#3399ff', '#66ccff', '#99ff99', '#66cc66', '#339933', '#006600']
    n_bins = 100
    cmap_land_ocean = LinearSegmentedColormap.from_list('land_ocean', colors_gradient, N=n_bins)

    # Plot all cells with color based on land fraction
    scatter = ax1.scatter(lon_plot, lat_plot,
                         c=land_fractions,
                         cmap=cmap_land_ocean,
                         s=3,
                         alpha=0.8,
                         vmin=0.0,
                         vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Land Fraction (0=Ocean, 1=Land)', fontsize=9)

    ax1.set_title(f'ICON R2B4 Grid: Land/Ocean Classification\n'
                  f'Land: {len(land_cells)} cells, Ocean: {len(ocean_cells)} cells\n'
                  f'Land %: {100*len(land_cells)/(len(land_cells)+len(ocean_cells)):.2f}%',
                  fontsize=12, fontweight='bold')
    ax1.grid(True)

    # Plot 2: All grid cells
    ax2 = fig.add_subplot(132, projection='mollweide')
    ax2.scatter(lon_plot, lat_plot, c='black', s=0.5, alpha=0.3)
    ax2.set_title(f'ICON R2B4 Grid\nTotal cells: {len(clat_deg)}',
                  fontsize=12, fontweight='bold')
    ax2.grid(True)

    # Plot 3: Cell size distribution by latitude
    ax3 = fig.add_subplot(133)

    # Bin cells by latitude
    lat_bins = np.linspace(-90, 90, 37)  # 5-degree bins
    land_hist, _ = np.histogram(clat_deg[land_cells], bins=lat_bins)
    ocean_hist, _ = np.histogram(clat_deg[ocean_cells], bins=lat_bins)

    bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    ax3.barh(bin_centers, ocean_hist, height=5, color='blue', alpha=0.5, label='Ocean')
    ax3.barh(bin_centers, land_hist, height=5, color='green', alpha=0.5,
             left=ocean_hist, label='Land')

    ax3.set_xlabel('Number of cells', fontsize=11)
    ax3.set_ylabel('Latitude [degrees]', fontsize=11)
    ax3.set_title('Cell Distribution by Latitude', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "etopo_land_ocean_verification.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    print("="*80)
    print("ETOPO LAND/OCEAN VERIFICATION")
    print("="*80)

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

    # Create plots
    print("\nCreating plots...")
    output_dir = Path("outputs") / "verification"
    plot_global_topography_and_grid(grid, params, reader, land_cells, ocean_cells, land_fractions, output_dir)

    # Save plotting data for debugging
    print("\nSaving plotting data for debugging...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert grid coordinates to degrees for saving
    clat_deg = np.rad2deg(clat_rad)
    clon_deg = np.rad2deg(clon_rad)

    # Save as compressed numpy file
    data_file = output_dir / "verification_data.npz"
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

    print("\n✓ Verification complete!")
    print(f"  Land cells: {land_count}")
    print(f"  Ocean cells: {ocean_count}")
