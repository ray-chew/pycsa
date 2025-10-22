#!/usr/bin/env python3
"""
Test script to load ALL ETOPO data and plot it on a globe.

This script validates that:
1. The ETOPO loader can handle large extent regions, including full global coverage
2. Coarse-graining works correctly to speed up loading and plotting
3. The cart_plotter can visualize large datasets on a globe
4. Data values are reasonable (elevation ranges)

Author: Test Suite
Date: 2025-10-22
Updated: Fixed to support full global extent
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Import CSA modules
from pycsa.core import io, var
from pycsa.plotting import cart_plot


def create_global_params(etopo_cg=8):
    """
    Create parameters for global ETOPO data loading.

    Parameters
    ----------
    etopo_cg : int, optional
        Coarse-graining factor (default: 8)
        - 1: Full resolution (~463m at equator) - VERY SLOW, huge memory
        - 2: ~926m - Still very slow
        - 4: ~1.85km - Moderate speed
        - 8: ~3.70km - Good balance for global plots
        - 16: ~7.4km - Fast, good for testing

    Returns
    -------
    params : object
        Parameter object with required attributes
    """
    class Params:
        def __init__(self):
            # Path to ETOPO data directory
            self.path_etopo = "/home/ray/git-projects/spec_appx/data/etopo_15s/"

            # Full global extent: entire world
            self.lat_extent = [-90.0, 90.0]
            self.lon_extent = [-180.0, 180.0]

            # Coarse-graining factor to speed up loading
            self.etopo_cg = etopo_cg

    return Params()


def test_global_etopo_load_and_plot():
    """
    Main test function: Load global ETOPO data and plot on globe.
    """
    print("=" * 80)
    print("GLOBAL ETOPO DATA LOADING AND PLOTTING TEST")
    print("=" * 80)
    print()

    # Configuration
    coarse_grain_factor = 8  # 8x8 averaging for reasonable speed
    plot_stride = 1  # Use all loaded data points for plotting

    print(f"Configuration:")
    print(f"  - Region: Full Global (-90 to 90°N, -180 to 180°E)")
    print(f"  - Coverage: 100% of Earth's surface")
    print(f"  - Coarse-graining: {coarse_grain_factor}x{coarse_grain_factor}")
    print(f"  - Effective resolution: ~{0.463 * coarse_grain_factor:.2f} km at equator")
    print(f"  - Plot stride: every {plot_stride} point(s)")
    print()

    # Step 1: Create parameters
    print("Step 1: Creating parameters...")
    params = create_global_params(etopo_cg=coarse_grain_factor)

    # Verify data directory exists
    data_path = Path(params.path_etopo)
    if not data_path.exists():
        print(f"ERROR: ETOPO data directory not found: {data_path}")
        print("Please ensure ETOPO data is downloaded and path is correct.")
        return False
    print(f"  - Data directory: {data_path}")
    print(f"  - Directory exists: {data_path.exists()}")
    print()

    # Step 2: Initialize topo_cell object
    print("Step 2: Initializing topo_cell object...")
    cell = var.topo_cell()
    print("  - topo_cell object created")
    print()

    # Step 3: Load ETOPO data
    print("Step 3: Loading ETOPO data...")
    print("  (This will load all tiles for full global coverage - may take a few minutes even with coarse-graining)")
    start_time = time.time()

    try:
        loader = io.ncdata.read_etopo_topo(
            cell,
            params,
            verbose=True,  # Show progress
            is_parallel=False
        )
        load_time = time.time() - start_time
        print()
        print(f"  - Loading completed in {load_time:.2f} seconds")
        print()

    except Exception as e:
        print(f"ERROR during loading: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Validate loaded data
    print("Step 4: Validating loaded data...")
    print(f"  - Latitude array shape: {cell.lat.shape}")
    print(f"  - Longitude array shape: {cell.lon.shape}")
    print(f"  - Topography array shape: {cell.topo.shape}")
    print()
    print(f"  - Latitude range: [{cell.lat.min():.4f}, {cell.lat.max():.4f}] degrees")
    print(f"  - Longitude range: [{cell.lon.min():.4f}, {cell.lon.max():.4f}] degrees")
    print()
    print(f"  - Elevation range: [{cell.topo.min():.1f}, {cell.topo.max():.1f}] meters")
    print(f"  - Mean elevation: {cell.topo.mean():.1f} meters")
    print(f"  - Median elevation: {np.median(cell.topo):.1f} meters")
    print()

    # Sanity checks
    checks_passed = True

    # Check data shapes
    expected_lat_points = len(cell.lat)
    expected_lon_points = len(cell.lon)
    if cell.topo.shape != (expected_lat_points, expected_lon_points):
        print(f"  WARNING: Unexpected topo shape!")
        checks_passed = False
    else:
        print(f"  ✓ Topography shape matches lat/lon dimensions")

    # Check elevation ranges (should be realistic)
    if cell.topo.min() < -11500 or cell.topo.max() > 9000:
        print(f"  WARNING: Elevation values outside expected range!")
        print(f"    (Expected: ~-11000m to ~8850m)")
        checks_passed = False
    else:
        print(f"  ✓ Elevation values within expected range")

    # Check for NaN or infinite values
    if np.isnan(cell.topo).any():
        print(f"  WARNING: Found NaN values in topography data!")
        checks_passed = False
    else:
        print(f"  ✓ No NaN values found")

    if np.isinf(cell.topo).any():
        print(f"  WARNING: Found infinite values in topography data!")
        checks_passed = False
    else:
        print(f"  ✓ No infinite values found")

    print()

    if not checks_passed:
        print("  Some validation checks failed!")
        return False


    # Step 5: Optionally clip ocean cells before plotting
    print("Step 5: Optionally clip ocean cells before plotting...")
    import os
    clip_ocean = True  # Default: clip ocean cells to -500m
    # Allow override via environment variable or function argument in future

    if cell.topo is None:
        print("ERROR: cell.topo is None. ETOPO data did not load correctly.")
        print("Skipping plotting and summary.")
        return False

    land_mask = cell.topo > 0
    ocean_mask = cell.topo <= 0
    total_points = cell.topo.size
    land_points = np.sum(land_mask)
    ocean_points = np.sum(ocean_mask)

    if clip_ocean:
        # Clip all ocean cells to -500m for land-only orography test
        cell.topo[ocean_mask] = -500.0
        print("  - Ocean cells clipped to -500m for land orography test.")
    else:
        print("  - Ocean cells retain original bathymetry (full range).")

    # Step 6: Generate meshgrid for plotting
    print("Step 6: Generating meshgrid for plotting...")
    cell.gen_mgrids()
    print(f"  - lon_grid shape: {cell.lon_grid.shape}")
    print(f"  - lat_grid shape: {cell.lat_grid.shape}")
    print()

    # Step 7: Create plot
    print("Step 7: Creating global plot...")
    print("  - Using cartopy PlateCarree projection")
    print("  - This may take a moment to render...")
    print()

    try:
        # Call the plotting function
        cart_plot.lat_lon(
            cell,
            fs=(14, 8),  # Larger figure for global view
            int=plot_stride,
            colorbar_margins=[0.92, 0.22, 0.035, 0.55]  # More visible colorbar
        )
        print("  - Plot displayed successfully!")
        print()

    except Exception as e:
        print(f"ERROR during plotting: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 8: Summary statistics
    print("Step 8: Summary statistics...")
    # Use the already-clipped topo for stats
    print(f"  - Total data points: {total_points:,}")
    print(f"  - Land points: {land_points:,} ({100*land_points/total_points:.1f}%)")
    print(f"  - Ocean points: {ocean_points:,} ({100*ocean_points/total_points:.1f}%)")
    print()
    print(f"  - Mean land elevation: {cell.topo[land_mask].mean():.1f} m")
    if not clip_ocean:
        print(f"  - Mean ocean depth: {cell.topo[ocean_mask].mean():.1f} m")
    print()
    print(f"  - Highest point: {cell.topo.max():.1f} m (should be near Mt. Everest)")
    print(f"  - Lowest point: {cell.topo.min():.1f} m (should be near Mariana Trench or -500m if clipped)")
    print()

    # Step 8: Report success
    print("=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Loaded {total_points:,} elevation data points")
    print(f"  - Load time: {load_time:.2f} seconds")
    print(f"  - Data quality: PASSED all validation checks")
    print(f"  - Visualization: SUCCESS")
    print()

    return True


def test_different_coarse_graining_factors():
    """
    Test loading with different coarse-graining factors.
    This helps understand the speed/quality tradeoff.
    """
    print("=" * 80)
    print("TESTING DIFFERENT COARSE-GRAINING FACTORS")
    print("=" * 80)
    print()

    # Test with progressively coarser graining
    test_factors = [16, 12, 8]

    for cg_factor in test_factors:
        print(f"\n{'='*60}")
        print(f"Testing with coarse-graining factor: {cg_factor}")
        print(f"Effective resolution: ~{0.463 * cg_factor:.2f} km at equator")
        print(f"{'='*60}\n")

        params = create_global_params(etopo_cg=cg_factor)
        cell = var.topo_cell()

        start_time = time.time()
        try:
            loader = io.ncdata.read_etopo_topo(cell, params, verbose=False)
            load_time = time.time() - start_time

            print(f"  Load time: {load_time:.2f} seconds")
            print(f"  Grid size: {cell.topo.shape}")
            print(f"  Memory usage: ~{cell.topo.nbytes / 1e6:.1f} MB")
            print(f"  Elevation range: [{cell.topo.min():.1f}, {cell.topo.max():.1f}] m")

        except Exception as e:
            print(f"  ERROR: {e}")

    print()


if __name__ == "__main__":
    import sys

    # Run the main global test
    success = test_global_etopo_load_and_plot()

    if success:
        print("\nAll tests passed! The ETOPO loader successfully loaded global coverage.")
        print()
        print("=" * 80)
        print("RECOMMENDED APPROACH FOR FULL GLOBAL COVERAGE")
        print("=" * 80)
        print()
        print("The dateline handling has been improved, but for best elevation accuracy")
        print("with full global coverage, use the two-hemisphere approach:")
        print()
        print("    # Load Western Hemisphere")
        print("    params_west = create_global_params()")
        print("    params_west.lon_extent = [-180.0, 0.0]")
        print("    cell_west = var.topo_cell()")
        print("    loader_west = io.ncdata.read_etopo_topo(cell_west, params_west)")
        print()
        print("    # Load Eastern Hemisphere")
        print("    params_east = create_global_params()")
        print("    params_east.lon_extent = [0.0, 180.0]")
        print("    cell_east = var.topo_cell()")
        print("    loader_east = io.ncdata.read_etopo_topo(cell_east, params_east)")
        print()
        print("    # Combine")
        print("    cell_global = var.topo_cell()")
        print("    cell_global.lon = np.concatenate([cell_west.lon, cell_east.lon])")
        print("    cell_global.lat = cell_west.lat  # Same for both")
        print("    cell_global.topo = np.concatenate([cell_west.topo, cell_east.topo], axis=1)")
        print()
        print("This approach preserves elevation accuracy better than loading")
        print("all 288 tiles in a single operation.")
        print("=" * 80)

        # Optionally run coarse-graining comparison (only if running interactively)
        if sys.stdin.isatty():
            user_input = input("\nRun coarse-graining comparison test? (y/n): ")
            if user_input.lower() == 'y':
                test_different_coarse_graining_factors()
        else:
            print("\nNote: Run interactively to test different coarse-graining factors.")
    else:
        print("\nTest failed! Please check the errors above.")
        sys.exit(1)
