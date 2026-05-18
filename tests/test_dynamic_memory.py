#!/usr/bin/env python3
"""
Test script for dynamic memory allocation based on cell latitude.

This verifies that:
1. Memory estimation function works correctly
2. Cells are properly grouped by memory requirements
3. Configuration makes sense for different hardware setups

Uses test_grid_5cells.npz when the full ICON grid is not available.
"""

import numpy as np
import pytest
from pathlib import Path

# Import the memory functions from runs/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runs"))
from icon_etopo_global import estimate_cell_memory_gb, group_cells_by_memory


DATA_DIR = Path(__file__).parent.parent / "data" / "test"


def _load_clat_rad():
    """Load cell latitudes in radians. Use test grid if full ICON grid unavailable."""
    grid_path = DATA_DIR / "test_grid_5cells.npz"
    if not grid_path.exists():
        pytest.skip(f"Test grid not found: {grid_path}")
    data = np.load(grid_path, allow_pickle=True)
    return data['clat_rad']


def test_memory_estimation():
    """Test that memory estimation scales appropriately with latitude."""
    test_latitudes = [0, 30, 45, 60, 70, 75, 80, 85, 89]

    base_mem = estimate_cell_memory_gb(0)
    for lat in test_latitudes:
        mem_gb = estimate_cell_memory_gb(lat)
        scale = mem_gb / base_mem
        print(f"  {lat:>3}° → {mem_gb:>6.1f} GB ({scale:>5.2f}x)")

    assert estimate_cell_memory_gb(0) == 10.0, "Equatorial cells should need 10 GB"
    assert estimate_cell_memory_gb(85) >= 50.0, "Polar cells (~85°) should need >= 50 GB"


def test_memory_monotonic():
    """Memory requirement should increase with latitude (cells stretch near poles)."""
    lats = [0, 15, 30, 45, 60, 75, 85, 89]
    mems = [estimate_cell_memory_gb(lat) for lat in lats]

    for i in range(1, len(mems)):
        assert mems[i] >= mems[i-1], \
            f"Memory should increase with latitude: {lats[i-1]}°={mems[i-1]} GB > {lats[i]}°={mems[i]} GB"


def test_cell_grouping():
    """Test that cells are properly grouped by memory requirements."""
    clat_rad = _load_clat_rad()
    n_cells = len(clat_rad)

    print(f"\n  {n_cells} cells, lat range: "
          f"[{np.rad2deg(clat_rad.min()):.1f}°, {np.rad2deg(clat_rad.max()):.1f}°]")

    # Test grouping with a small memory budget
    batches = group_cells_by_memory(clat_rad, max_memory_per_batch_gb=60.0)

    total_cells_batched = 0
    for i, batch in enumerate(batches):
        n = len(batch['cell_indices'])
        total_cells_batched += n
        print(f"  Batch {i}: {n} cells, "
              f"{batch['memory_per_cell_gb']:.1f} GB/cell, "
              f"{batch['n_workers']} workers")

    assert total_cells_batched == n_cells, \
        f"All cells should be batched (got {total_cells_batched}, expected {n_cells})"

    # Every batch should have at least 1 worker
    for batch in batches:
        assert batch['n_workers'] >= 1, "Batch should have at least 1 worker"


def test_cell_grouping_hpc_vs_laptop():
    """HPC (more memory) should allow more workers per batch."""
    clat_rad = _load_clat_rad()

    batches_laptop = group_cells_by_memory(clat_rad, max_memory_per_batch_gb=60.0)
    batches_hpc = group_cells_by_memory(clat_rad, max_memory_per_batch_gb=240.0)

    avg_workers_laptop = np.mean([b['n_workers'] for b in batches_laptop])
    avg_workers_hpc = np.mean([b['n_workers'] for b in batches_hpc])

    print(f"\n  Laptop (60 GB): avg {avg_workers_laptop:.1f} workers/batch")
    print(f"  HPC   (240 GB): avg {avg_workers_hpc:.1f} workers/batch")

    assert avg_workers_hpc >= avg_workers_laptop, \
        "HPC should have >= workers per batch than laptop"


def test_specific_cells():
    """Test memory estimation for the 5 extracted test cells."""
    clat_rad = _load_clat_rad()
    grid = np.load(DATA_DIR / "test_grid_5cells.npz", allow_pickle=True)
    names = list(grid['cell_names'])
    clat_deg = np.rad2deg(clat_rad)

    print()
    for i, name in enumerate(names):
        lat = clat_deg[i]
        mem = estimate_cell_memory_gb(lat)
        print(f"  {name:>15s}: lat={lat:>7.2f}°, est. memory={mem:>5.1f} GB")

    # Greenland (74°N) should need more memory than equatorial Pacific
    pacific_idx = names.index('pacific')
    greenland_idx = names.index('greenland')
    mem_pacific = estimate_cell_memory_gb(clat_deg[pacific_idx])
    mem_greenland = estimate_cell_memory_gb(clat_deg[greenland_idx])
    assert mem_greenland > mem_pacific, \
        f"Greenland ({mem_greenland} GB) should need more memory than Pacific ({mem_pacific} GB)"


if __name__ == '__main__':
    test_memory_estimation()
    test_cell_grouping()
    test_cell_grouping_hpc_vs_laptop()
    test_specific_cells()
    print("\nAll tests passed!")
