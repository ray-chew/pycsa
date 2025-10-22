"""
Test script to load ETOPO data and generate a plot using existing infrastructure.

This script:
1. Loads ETOPO 2022 15 arc-second data for a test region
2. Generates a meshgrid for plotting
3. Uses the existing cart_plot.lat_lon() function to create a visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path

from pycsa.core import io, var
from pycsa.plotting import cart_plot


def test_etopo_plot():
    """Load ETOPO data and create a plot."""

    # Setup parameters for a test region (California Sierra Nevada)
    class params:
        def __init__(self):
            self.path_etopo = str(Path(__file__).parent.parent / "data" / "etopo_15s") + "/"
            # Region covering Lake Tahoe and surrounding Sierra Nevada
            self.lat_extent = [38.5, 39.5]
            self.lon_extent = [-120.5, -119.5]
            self.etopo_cg = 4  # Use some coarse-graining for reasonable file size

    # Load the data
    print("Loading ETOPO data...")
    test_params = params()
    cell = var.topo_cell()

    loader = io.ncdata.read_etopo_topo(cell, test_params, verbose=True)

    # Print statistics
    print(f"\nLoaded data statistics:")
    print(f"  Shape: {len(cell.lat)} x {len(cell.lon)} = {cell.topo.shape}")
    print(f"  Lat range: {cell.lat.min():.4f} to {cell.lat.max():.4f}")
    print(f"  Lon range: {cell.lon.min():.4f} to {cell.lon.max():.4f}")
    print(f"  Elevation range: {cell.topo.min():.1f} to {cell.topo.max():.1f} meters")
    print(f"  Mean elevation: {cell.topo.mean():.1f} meters")

    # Generate meshgrid (required by the plotting function)
    cell.gen_mgrids()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Generate plot using existing infrastructure
    print("\nGenerating plot...")

    try:
        # Use the existing lat_lon plotting function
        # Note: This requires cartopy to be installed
        cart_plot.lat_lon(cell, fs=(10, 8), int=1)

        # Save the figure
        output_file = output_dir / "etopo_test_plot.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")

    except ImportError as e:
        print(f"Warning: Could not use cartopy plotting: {e}")
        print("Falling back to simple matplotlib plot...")

        # Fallback: Simple matplotlib plot without cartopy
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.contourf(
            cell.lon_grid,
            cell.lat_grid,
            cell.topo,
            levels=20,
            cmap="terrain"
        )

        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_title(f"ETOPO 2022 Test Region\n"
                     f"Lake Tahoe & Sierra Nevada\n"
                     f"Elevation: {cell.topo.min():.0f} to {cell.topo.max():.0f} m")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Elevation (m)")

        ax.grid(True, alpha=0.3, linestyle='--')

        output_file = output_dir / "etopo_test_plot_simple.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Simple plot saved to: {output_file}")

    finally:
        plt.close('all')

    print("\nTest completed successfully!")

    return cell


if __name__ == "__main__":
    cell = test_etopo_plot()
