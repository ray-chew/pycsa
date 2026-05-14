#!/usr/bin/env python3
"""
Test script for dynamic memory allocation based on cell latitude.

This verifies that:
1. Memory estimation function works correctly
2. Cells are properly grouped by memory requirements
3. Configuration makes sense for different hardware setups
"""

import numpy as np

from pycsa.core import io, var, utils
from pycsa.scheduling import estimate_cell_memory_gb, group_cells_by_memory


def test_memory_estimation():
    """Test that memory estimation scales appropriately with latitude."""
    print("=" * 80)
    print("TEST 1: Memory Estimation Function")
    print("=" * 80)

    test_latitudes = [0, 30, 45, 60, 70, 75, 80, 85, 89]

    print("\nMemory requirements by latitude:")
    print(f"{'Latitude':<12} {'Memory (GB)':<15} {'Scale Factor':<15}")
    print("-" * 42)

    base_mem = estimate_cell_memory_gb(0)
    for lat in test_latitudes:
        mem_gb = estimate_cell_memory_gb(lat)
        scale = mem_gb / base_mem
        print(f"{lat:>3}°        {mem_gb:>6.1f} GB        {scale:>5.2f}x")

    # Verify expectations
    assert estimate_cell_memory_gb(0) == 10.0, "Equatorial cells should need 10 GB"
    assert (
        estimate_cell_memory_gb(85) >= 50.0
    ), "Polar cells (~85°) should need >= 50 GB"
    print("\n✓ Memory estimation function passes basic tests")


def test_cell_grouping():
    """Test that cells are properly grouped by memory requirements."""
    print("\n" + "=" * 80)
    print("TEST 2: Cell Grouping by Memory")
    print("=" * 80)

    # Load actual ICON grid to get realistic cell latitudes
    print("\nLoading ICON grid...")
    from inputs.icon_global_run import params

    grid = var.grid()
    reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_dat(params.path_icon_grid, grid)

    clat_rad = grid.clat
    n_cells = len(clat_rad)

    print(f"Loaded {n_cells} cells")
    print(
        f"Latitude range: {np.rad2deg(clat_rad.min()):.1f}° to {np.rad2deg(clat_rad.max()):.1f}°"
    )

    # Test for laptop configuration (60 GB total)
    print("\n--- LAPTOP CONFIGURATION (60 GB total) ---")
    batches_laptop = group_cells_by_memory(clat_rad, max_memory_per_batch_gb=60.0)

    print(f"\nCreated {len(batches_laptop)} memory batches:")
    total_cells_batched = 0
    for i, batch in enumerate(batches_laptop):
        n = len(batch["cell_indices"])
        total_cells_batched += n
        print(
            f"  Batch {i}: {n:>6} cells, "
            f"{batch['memory_per_cell_gb']:>5.1f} GB/cell, "
            f"{batch['n_workers']:>2} workers × {batch['memory_per_worker_gb']:>5.1f} GB = "
            f"{batch['n_workers'] * batch['memory_per_worker_gb']:>6.1f} GB total"
        )

    assert (
        total_cells_batched == n_cells
    ), f"All cells should be batched (got {total_cells_batched}, expected {n_cells})"
    print(f"\n✓ All {n_cells} cells properly batched")

    # Test for HPC configuration (240 GB total)
    print("\n--- HPC CONFIGURATION (240 GB total) ---")
    batches_hpc = group_cells_by_memory(clat_rad, max_memory_per_batch_gb=240.0)

    print(f"\nCreated {len(batches_hpc)} memory batches:")
    total_cells_batched = 0
    for i, batch in enumerate(batches_hpc):
        n = len(batch["cell_indices"])
        total_cells_batched += n
        print(
            f"  Batch {i}: {n:>6} cells, "
            f"{batch['memory_per_cell_gb']:>5.1f} GB/cell, "
            f"{batch['n_workers']:>2} workers × {batch['memory_per_worker_gb']:>5.1f} GB = "
            f"{batch['n_workers'] * batch['memory_per_worker_gb']:>6.1f} GB total"
        )

    assert (
        total_cells_batched == n_cells
    ), f"All cells should be batched (got {total_cells_batched}, expected {n_cells})"
    print(f"\n✓ All {n_cells} cells properly batched")

    # Verify that HPC has better parallelism (more workers on average)
    avg_workers_laptop = np.mean([b["n_workers"] for b in batches_laptop])
    avg_workers_hpc = np.mean([b["n_workers"] for b in batches_hpc])

    print(f"\nAverage workers per batch:")
    print(f"  Laptop: {avg_workers_laptop:.1f}")
    print(f"  HPC:    {avg_workers_hpc:.1f}")

    assert (
        avg_workers_hpc > avg_workers_laptop
    ), "HPC should have more workers on average"
    print("✓ HPC configuration properly utilizes more workers")


def test_specific_cells():
    """Test memory estimation for specific problematic cells."""
    print("\n" + "=" * 80)
    print("TEST 3: Specific Cell Memory Requirements")
    print("=" * 80)

    from inputs.icon_global_run import params

    grid = var.grid()
    reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_dat(params.path_icon_grid, grid)

    clat_rad = grid.clat
    clat_deg = np.rad2deg(clat_rad)

    # Test cell 16384 (known to need 60 GB)
    test_cell_idx = 16384
    if test_cell_idx < len(clat_deg):
        cell_lat = clat_deg[test_cell_idx]
        estimated_mem = estimate_cell_memory_gb(cell_lat)

        print(f"\nCell {test_cell_idx}:")
        print(f"  Latitude: {cell_lat:.2f}°")
        print(f"  Estimated memory: {estimated_mem:.1f} GB")
        print(f"  Actual requirement (from tests): 60 GB")

        if estimated_mem >= 50.0:
            print("  ✓ Estimation is in the right ballpark")
        else:
            print(
                f"  ⚠ Estimation may be too low (got {estimated_mem:.1f} GB, expected >= 50 GB)"
            )

    # Show top 10 most memory-intensive cells
    cell_memory_gb = np.array([estimate_cell_memory_gb(lat) for lat in clat_deg])
    top_indices = np.argsort(cell_memory_gb)[-10:][::-1]

    print(f"\nTop 10 most memory-intensive cells:")
    print(f"{'Cell Index':<12} {'Latitude':<12} {'Est. Memory':<15}")
    print("-" * 39)
    for idx in top_indices:
        print(f"{idx:<12} {clat_deg[idx]:>7.2f}°     {cell_memory_gb[idx]:>6.1f} GB")


if __name__ == "__main__":
    try:
        test_memory_estimation()
        test_cell_grouping()
        test_specific_cells()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
