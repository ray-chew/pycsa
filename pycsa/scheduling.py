"""HPC scheduling helpers: per-cell memory estimation and memory-aware batching.

Pure-numpy functions used by ``runs/icon_etopo_global.py`` to size Dask
workers based on each ICON cell's latitude (polar cells cover more
longitudinal range in degree-space, so they need more topographic data
loaded). Lives in ``pycsa.*`` rather than ``runs/`` so tests can import
without going through a run script (or pulling Dask in at collection time).
"""

from __future__ import annotations

import numpy as np


def estimate_cell_memory_gb(lat_deg: float) -> float:
    """
    Estimate memory requirements (in GB) for processing a cell based on its latitude.

    At polar latitudes, cells cover a larger longitudinal range in degree-space,
    requiring more topographic data points to be loaded with coarse-graining.

    Parameters
    ----------
    lat_deg : float
        Cell center latitude in degrees (-90 to 90)

    Returns
    -------
    float
        Estimated memory requirement in GB

    Notes
    -----
    - Equatorial cells (~0°): ~10 GB sufficient
    - Mid-latitude cells (~45°): ~10 GB
    - High-latitude cells (~70°): ~25 GB
    - High-latitude cells (~80°): ~43 GB
    - Polar cells (~85-89°): ~60 GB required

    Memory scales approximately with (1/cos(lat))^0.7 due to meridian
    convergence, capped at 60 GB at the poles.

    **Tuning history.** Original cap was 60 GB (scale 6.0) but worked
    only nominally because the planner's interior path used
    safety_factor=1.0 (bug; the final-batch path used 1.5). The bug
    caused 8 workers × 60 GB = 480 GB on the 510 GB node and OOMs.
    Briefly retuned 60 → 30 → 45 GB chasing the OOM symptom; the
    real fix was the safety_factor=1.5 update in 2026-05. With
    safety_factor=1.5 honored consistently, the original 60 GB cap
    gives 3 workers × 90 GB on a 256 GB budget — well within the
    per-worker memory_limit that Dask actually enforces.
    """
    abs_lat = np.abs(lat_deg)

    # Base memory requirement at equator
    base_memory_gb = 10.0

    # Scale factor based on latitude (empirical fit)
    if abs_lat < 60.0:
        # Below 60°, memory is fairly constant
        scale_factor = 1.0
    elif abs_lat < 85.0:
        # Between 60° and 85°, use (1/cos(lat))^0.7 scaling:
        #   at 70°: (1/0.342)^0.7 ≈ 2.5  → 25 GB
        #   at 80°: (1/0.174)^0.7 ≈ 4.3  → 43 GB
        lat_rad = np.deg2rad(abs_lat)
        cos_lat = np.cos(lat_rad)
        scale_factor = (1.0 / cos_lat) ** 0.7
    else:
        # Above 85°, cap at 6.0x base (60 GB). See "tuning history"
        # in the docstring.
        scale_factor = 6.0

    return base_memory_gb * scale_factor


def group_cells_by_memory(
    clat_rad: np.ndarray, max_memory_per_batch_gb: float = 240.0
) -> list[dict]:
    """
    Group cells into batches with similar memory requirements.

    Parameters
    ----------
    clat_rad : ndarray
        Cell center latitudes in radians
    max_memory_per_batch_gb : float
        Maximum total memory available for a batch (default: 240 GB for 6 workers × 40 GB)

    Returns
    -------
    list of dict
        List of batch configurations, each containing:
        - 'cell_indices': list of cell indices in this batch
        - 'memory_per_cell_gb': average memory per cell in GB
        - 'n_workers': recommended number of workers
        - 'memory_per_worker_gb': recommended memory per worker
    """
    n_cells = len(clat_rad)
    clat_deg = np.rad2deg(clat_rad)

    # Estimate memory for each cell
    cell_memory_gb = np.array([estimate_cell_memory_gb(lat) for lat in clat_deg])

    # Sort cells by memory requirement (process high-memory cells first)
    sorted_indices = np.argsort(cell_memory_gb)[::-1]

    batches = []
    current_batch_indices = []
    current_batch_memory = []

    for idx in sorted_indices:
        mem = cell_memory_gb[idx]

        # Check if adding this cell would exceed batch memory limit
        if current_batch_indices:
            avg_mem = np.mean(current_batch_memory + [mem])
            # Ensure we can fit at least 1 worker with this memory
            if avg_mem * len(current_batch_indices) > max_memory_per_batch_gb:
                # Finalize current batch
                avg_mem_current = np.mean(current_batch_memory)
                # Use 50% safety margin for diskless NetCDF loading
                # (must match the final-batch branch below — was 1.0
                # here for a long time, causing polar batches to allocate
                # more workers than the node could actually hold)
                safety_factor = 1.5
                n_workers = max(
                    1, int(max_memory_per_batch_gb / (avg_mem_current * safety_factor))
                )
                mem_per_worker = avg_mem_current * safety_factor
                # When only one worker fits, give it the entire budget
                # (matches the runner's "Single-worker mode" path that
                # allocates the full machine memory to the single
                # worker). Without this, mem_per_worker can exceed
                # max_memory_per_batch_gb on memory-tight presets.
                if n_workers == 1:
                    mem_per_worker = min(mem_per_worker, max_memory_per_batch_gb)
                # Defensive: never let n_workers × mem_per_worker exceed
                # the node budget. The int(...) above + the n_workers==1
                # clamp together enforce this; the assert is just so a
                # future refactor can't regress silently.
                assert n_workers * mem_per_worker <= max_memory_per_batch_gb, (
                    f"planner overcommit: {n_workers} × {mem_per_worker:.1f} GB "
                    f"> {max_memory_per_batch_gb} GB"
                )

                batches.append(
                    {
                        "cell_indices": sorted(
                            current_batch_indices
                        ),  # Sort by original index order
                        "memory_per_cell_gb": avg_mem_current,
                        "n_workers": n_workers,
                        "memory_per_worker_gb": mem_per_worker,
                    }
                )

                # Start new batch
                current_batch_indices = [idx]
                current_batch_memory = [mem]
            else:
                current_batch_indices.append(idx)
                current_batch_memory.append(mem)
        else:
            current_batch_indices.append(idx)
            current_batch_memory.append(mem)

    # Finalize last batch
    if current_batch_indices:
        avg_mem = np.mean(current_batch_memory)
        # Use 50% safety margin for diskless NetCDF loading
        safety_factor = 1.5
        n_workers = max(1, int(max_memory_per_batch_gb / (avg_mem * safety_factor)))
        mem_per_worker = avg_mem * safety_factor
        if n_workers == 1:
            mem_per_worker = min(mem_per_worker, max_memory_per_batch_gb)
        assert n_workers * mem_per_worker <= max_memory_per_batch_gb, (
            f"planner overcommit (last batch): {n_workers} × {mem_per_worker:.1f} GB "
            f"> {max_memory_per_batch_gb} GB"
        )

        batches.append(
            {
                "cell_indices": sorted(current_batch_indices),
                "memory_per_cell_gb": avg_mem,
                "n_workers": n_workers,
                "memory_per_worker_gb": mem_per_worker,
            }
        )

    return batches
