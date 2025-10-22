"""
Compare MERIT vs ETOPO loading for the same Alaska region
"""

import numpy as np
from pycsa.core import io, var

print("=" * 60)
print("COMPARING MERIT vs ETOPO for Alaska region")
print("=" * 60)

# Test ETOPO
class params_etopo:
    def __init__(self):
        self.path_etopo = "./data/etopo_15s/"
        self.lat_extent = [48.0, 64.0, 64.0]
        self.lon_extent = [-148.0, -148.0, -112.0]
        self.etopo_cg = 10

print("\n1. LOADING ETOPO...")
cell_etopo = var.topo_cell()
params_e = params_etopo()
loader_e = io.ncdata.read_etopo_topo(cell_etopo, params_e, verbose=False)

print(f"   Shape: {cell_etopo.topo.shape}")
print(f"   Lat: {cell_etopo.lat.min():.2f} to {cell_etopo.lat.max():.2f}")
print(f"   Lon: {cell_etopo.lon.min():.2f} to {cell_etopo.lon.max():.2f}")
print(f"   Elevation: {cell_etopo.topo.min():.1f} to {cell_etopo.topo.max():.1f} m")
print(f"   Mean: {cell_etopo.topo.mean():.1f} m")
print(f"   Std: {cell_etopo.topo.std():.1f} m")

# Test MERIT
try:
    class params_merit:
        def __init__(self):
            self.path_merit = "/data/MERIT/"  # Adjust path as needed
            self.lat_extent = [48.0, 64.0, 64.0]
            self.lon_extent = [-148.0, -148.0, -112.0]
            self.merit_cg = 10

    print("\n2. LOADING MERIT...")
    cell_merit = var.topo_cell()
    params_m = params_merit()
    loader_m = io.ncdata.read_merit_topo(cell_merit, params_m, verbose=False)

    print(f"   Shape: {cell_merit.topo.shape}")
    print(f"   Lat: {cell_merit.lat.min():.2f} to {cell_merit.lat.max():.2f}")
    print(f"   Lon: {cell_merit.lon.min():.2f} to {cell_merit.lon.max():.2f}")
    print(f"   Elevation: {cell_merit.topo.min():.1f} to {cell_merit.topo.max():.1f} m")
    print(f"   Mean: {cell_merit.topo.mean():.1f} m")
    print(f"   Std: {cell_merit.topo.std():.1f} m")

    print("\n3. COMPARISON:")
    print(f"   Shape difference: ETOPO {cell_etopo.topo.shape} vs MERIT {cell_merit.topo.shape}")
    print(f"   Mean difference: {cell_etopo.topo.mean() - cell_merit.topo.mean():.1f} m")

except Exception as e:
    print(f"\n   Could not load MERIT: {e}")
    print("   (This is expected if MERIT data is not available)")

# Check for data quality issues in ETOPO
print("\n4. ETOPO DATA QUALITY CHECKS:")
if np.any(np.isnan(cell_etopo.topo)):
    print(f"   ✗ WARNING: NaN values present!")
else:
    print(f"   ✓ No NaN values")

if np.any(cell_etopo.topo == -99999):
    print(f"   ✗ WARNING: Fill values (-99999) present!")
else:
    print(f"   ✓ No fill values")

if np.all(cell_etopo.topo == cell_etopo.topo[0, 0]):
    print(f"   ✗ WARNING: All values identical!")
else:
    print(f"   ✓ Values vary")

# Check array types
print(f"\n5. ARRAY TYPES:")
print(f"   lat type: {type(cell_etopo.lat)}, dtype: {cell_etopo.lat.dtype}")
print(f"   lon type: {type(cell_etopo.lon)}, dtype: {cell_etopo.lon.dtype}")
print(f"   topo type: {type(cell_etopo.topo)}, dtype: {cell_etopo.topo.dtype}")

# Sample a few points
print(f"\n6. SAMPLE VALUES (first 3x3):")
print(cell_etopo.topo[:3, :3])
