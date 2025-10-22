"""
Debug script to test ETOPO loading WITH coarse-graining
"""

import numpy as np
from pycsa.core import io, var

class params:
    def __init__(self):
        self.path_etopo = "./data/etopo_15s/"
        self.lat_extent = [48.0, 64.0, 64.0]
        self.lon_extent = [-148.0, -148.0, -112.0]
        self.etopo_cg = 10  # Add coarse-graining

test_params = params()

print("Testing ETOPO loader with Alaska parameters + CG=10...")
print(f"lat_extent: {test_params.lat_extent}")
print(f"lon_extent: {test_params.lon_extent}")
print(f"etopo_cg: {test_params.etopo_cg}")
print(f"lat range: {np.array(test_params.lat_extent).min():.1f} to {np.array(test_params.lat_extent).max():.1f}")
print(f"lon range: {np.array(test_params.lon_extent).min():.1f} to {np.array(test_params.lon_extent).max():.1f}")

cell = var.topo_cell()

try:
    loader = io.ncdata.read_etopo_topo(cell, test_params, verbose=False)

    print(f"\n✓ Loading successful!")
    print(f"  Loaded shape: {cell.topo.shape}")
    print(f"  Lat: {len(cell.lat)} points from {cell.lat.min():.4f} to {cell.lat.max():.4f}")
    print(f"  Lon: {len(cell.lon)} points from {cell.lon.min():.4f} to {cell.lon.max():.4f}")
    print(f"  Topo range: {cell.topo.min():.1f} to {cell.topo.max():.1f} m")
    print(f"  Topo mean: {cell.topo.mean():.1f} m")

    print(f"\n  Data reduction: {(3838*8638)/(cell.topo.size):.1f}x")

    # Check for suspicious values
    if np.any(cell.topo == 0):
        n_zeros = np.sum(cell.topo == 0)
        print(f"\n⚠ Warning: {n_zeros} zero values found ({100*n_zeros/cell.topo.size:.1f}%)")

    if np.any(np.isnan(cell.topo)):
        print(f"⚠ Warning: NaN values found!")

    if np.all(cell.topo == cell.topo[0,0]):
        print(f"⚠ Warning: All values are the same!")

    # Test meshgrid generation
    print(f"\n  Testing meshgrid generation...")
    cell.gen_mgrids()
    print(f"  ✓ Meshgrid generated: {cell.lat_grid.shape}")

except Exception as e:
    print(f"\n✗ Loading failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
