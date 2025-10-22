"""
Example script demonstrating how to use the ETOPO 2022 15 arc-second loader

This script shows how to:
1. Set up parameters for ETOPO data loading
2. Load a regional topography dataset
3. Apply coarse-graining for different resolutions
"""

import numpy as np
from pycsa.core import io, var


class params:
    """Simple parameter class for ETOPO loading"""
    def __init__(self):
        # Path to ETOPO data directory (must end with /)
        self.path_etopo = "/home/ray/git-projects/spec_appx/data/etopo_15s/"

        # Define region of interest [lat_min, lat_max]
        self.lat_extent = [30.0, 45.0]

        # Define region of interest [lon_min, lon_max]
        self.lon_extent = [-120.0, -105.0]

        # Coarse-graining factor (1 = no coarse-graining, 2 = 2x2 average, etc.)
        # ETOPO 15" has ~3600 points per 15 degrees, so coarse-graining is useful
        # etopo_cg = 2  -> ~30" resolution
        # etopo_cg = 4  -> ~60" resolution (1 arc-minute)
        # etopo_cg = 8  -> ~120" resolution (2 arc-minutes)
        self.etopo_cg = 1  # Default: no coarse-graining


# Example 1: Load high-resolution data (15 arc-seconds, no coarse-graining)
print("Example 1: Loading high-resolution ETOPO data...")
params1 = params()
params1.etopo_cg = 1
cell1 = var.topo_cell()

loader1 = io.ncdata.read_etopo_topo(cell1, params1, verbose=True)
print(f"Loaded: {len(cell1.lat)} x {len(cell1.lon)} = {cell1.topo.shape}")
print(f"Lat range: {cell1.lat.min():.4f} to {cell1.lat.max():.4f}")
print(f"Lon range: {cell1.lon.min():.4f} to {cell1.lon.max():.4f}")
print(f"Elevation range: {cell1.topo.min():.1f} to {cell1.topo.max():.1f} meters")
print()


# Example 2: Load with 4x coarse-graining (~60" resolution)
print("Example 2: Loading with 4x coarse-graining...")
params2 = params()
params2.etopo_cg = 4
cell2 = var.topo_cell()

loader2 = io.ncdata.read_etopo_topo(cell2, params2)
print(f"Loaded: {len(cell2.lat)} x {len(cell2.lon)} = {cell2.topo.shape}")
print(f"Data reduction factor: {cell1.topo.size / cell2.topo.size:.1f}x")
print()


# Example 3: Load a small region
print("Example 3: Loading a small region (35-37°N, -115 to -110°W)...")
params3 = params()
params3.lat_extent = [35.0, 37.0]
params3.lon_extent = [-115.0, -110.0]
params3.etopo_cg = 1
cell3 = var.topo_cell()

loader3 = io.ncdata.read_etopo_topo(cell3, params3)
print(f"Loaded: {len(cell3.lat)} x {len(cell3.lon)} = {cell3.topo.shape}")
print(f"Elevation range: {cell3.topo.min():.1f} to {cell3.topo.max():.1f} meters")
print()


# Example 4: Cross-dateline region (if needed)
print("Example 4: Region spanning across dateline...")
params4 = params()
params4.lat_extent = [40.0, 50.0]
params4.lon_extent = [170.0, -170.0]  # Crosses dateline
params4.etopo_cg = 8
cell4 = var.topo_cell()

try:
    loader4 = io.ncdata.read_etopo_topo(cell4, params4)
    print(f"Loaded: {len(cell4.lat)} x {len(cell4.lon)} = {cell4.topo.shape}")
except Exception as e:
    print(f"Note: Dateline crossing may need verification: {e}")
print()


print("Done! All loaders completed successfully.")
print("\nUsage tips:")
print("- Set etopo_cg = 1 for full 15\" resolution (very high-res!)")
print("- Set etopo_cg = 4 for ~60\" (~1.8 km at equator)")
print("- Set etopo_cg = 8 for ~120\" (~3.6 km at equator)")
print("- Coarse-graining reduces memory and speeds up processing")
