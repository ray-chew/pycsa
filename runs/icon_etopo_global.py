#!/usr/bin/env python3
"""
ICON ETOPO Global Processing Script

IMPORTANT: Thread control environment variables must be set BEFORE numpy/numba import
to prevent thread over-subscription with Dask workers.
"""

import os

# ============================================================================
# CRITICAL: Set thread limits BEFORE importing numpy/numba/scipy
# This prevents thread over-subscription when using Dask with threads_per_worker > 1
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = (
    "1"  # Critical: prevents Numba parallel=True conflicts
)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for parallel processing
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from pathlib import Path
import gc
import logging
from datetime import datetime

from pycsa.core import io, var, utils, tile_cache
from pycsa.wrappers import interface, diagnostics
from pycsa.plotting import plotter
from pycsa.scheduling import estimate_cell_memory_gb, group_cells_by_memory

# Initialize logger (will be configured in main)
logger = logging.getLogger(__name__)


def setup_logger(log_dir="logs"):
    """Thin wrapper around :func:`pycsa.logging_config.configure_logging`
    that preserves the historical ``icon_etopo_global_*`` filename prefix
    and the previous default level. New code should call
    ``configure_logging`` directly.
    """
    from pycsa.logging_config import configure_logging

    return configure_logging(log_dir=log_dir, name_prefix="icon_etopo_global")


def get_topo_colormap():
    """
    Create a topography colormap with blue for ocean (< 0m) and terrain colors for land (> 0m).
    Transition occurs exactly at sea level (0m) with smooth blending.

    For TwoSlopeNorm to work correctly, we need equal colors on each side:
    128 colors for ocean (< 0m) + 128 colors for land (> 0m) = 256 total
    """
    # Ocean colors (blue shades from deep to shallow)
    ocean_colors = plt.cm.Blues_r(np.linspace(0.4, 0.95, 120))

    # Smooth transition zone around sea level (8 colors on each side)
    # Get the last ocean color and first land color
    last_ocean = plt.cm.Blues_r(0.95)
    first_land = plt.cm.terrain(0.25)

    # Create smooth blend from ocean to land
    transition_colors = np.zeros((16, 4))
    for i in range(4):  # RGBA channels
        transition_colors[:, i] = np.linspace(last_ocean[i], first_land[i], 16)

    # Land colors (terrain-like: green to brown to white)
    land_colors = plt.cm.terrain(np.linspace(0.28, 1.0, 120))

    # Combine: 120 ocean + 16 transition + 120 land = 256 total
    # Transition centered at index 128 (sea level)
    colors = np.vstack((ocean_colors, transition_colors, land_colors))
    return mcolors.LinearSegmentedColormap.from_list("topo", colors)


def plot_cell_diagnostics(c_idx, cell_sa, ampls_sa, dat_2D_sa, output_dir, params):
    """
    Create 3-panel diagnostic plot for a single cell.

    Panel 1: Loaded topography (original ETOPO data within cell)
    Panel 2: Reconstructed topography after second approximation
    Panel 3: Computed spectrum

    Parameters
    ----------
    c_idx : int
        Cell index
    cell_sa : topo_cell
        Cell object after second approximation (contains original topo in cell.topo)
    ampls_sa : ndarray
        Amplitude spectrum from second approximation
    dat_2D_sa : ndarray
        Reconstructed topography from second approximation
    output_dir : Path
        Output directory for saving plots
    params : params object
        Parameters object
    """
    # Create figure with 3 panels
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Get elevation extent for consistent color scaling
    vmin = -200.0  # Always fix ocean floor at -500m (blue portion)
    vmax = np.nanmax(cell_sa.topo)

    # Ensure vmax is positive (land)
    if vmax <= 0:
        vmax = 100.0  # Force some land color even if all ocean

    # Create custom colormap with blue for ocean, terrain colors for land
    topo_cmap = get_topo_colormap()

    # Create normalization centered at sea level (0m)
    # This makes the colormap transition exactly at 0m
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Panel 1: Original topography within cell
    topo_original = cell_sa.topo.copy()
    topo_original[~cell_sa.mask] = np.nan

    im1 = axs[0].imshow(
        topo_original, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[0].set_title(
        f"Cell {c_idx}: Loaded Topography\nRange: [{vmin:.0f}, {vmax:.0f}] m",
        fontsize=11,
        fontweight="bold",
    )
    axs[0].set_xlabel("Longitude index")
    axs[0].set_ylabel("Latitude index")
    cbar1 = plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Elevation [m]", rotation=270, labelpad=15)

    # Panel 2: Reconstructed topography (masked)
    dat_2D_masked = dat_2D_sa.copy()
    dat_2D_masked[~cell_sa.mask] = np.nan

    # Compute reconstruction error
    diff = cell_sa.topo - dat_2D_sa
    rmse = np.sqrt(np.mean(diff[cell_sa.mask] ** 2))
    rel_rmse = rmse / (vmax - vmin) * 100

    im2 = axs[1].imshow(
        dat_2D_masked, origin="lower", cmap=topo_cmap, norm=norm, aspect="auto"
    )
    axs[1].set_title(
        f"Reconstructed (2nd Approx)\nRMSE: {rmse:.1f} m ({rel_rmse:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    axs[1].set_xlabel("Longitude index")
    axs[1].set_ylabel("Latitude index")
    cbar2 = plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Elevation [m]", rotation=270, labelpad=15)

    # Panel 3: Amplitude spectrum in (k,l) wavenumber space
    fig_obj = plotter.fig_obj(fig, params.nhi, params.nhj, cbar=True, set_label=True)
    axs[2] = fig_obj.freq_panel(
        axs[2], ampls_sa, title="Amplitude Spectrum", v_extent=None
    )

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"cell_{c_idx:05d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Explicit memory cleanup - delete ALL objects to prevent memory leaks
    del fig, axs, fig_obj, im1, im2, topo_original, dat_2D_masked
    del cbar1, cbar2, norm, topo_cmap, diff
    gc.collect()  # Force garbage collection after plotting

    logger.info(f"  Plot saved: {output_path}")


def do_cell(
    c_idx,
    grid,
    params,
    reader,
    writer,
    chunk_output_dir,
    clat_rad,
    clon_rad,
):
    """
    Process a single ICON grid cell with ETOPO topography.

    Parameters
    ----------
    c_idx : int
        Cell index in the grid
    grid : grid object
        ICON grid (in degrees)
    params : params object
        Parameters
    reader : ncdata object
        Data reader
    writer : nc_writer object
        NetCDF writer
    chunk_output_dir : Path
        Output directory for this chunk
    clat_rad : ndarray
        Cell center latitudes in radians
    clon_rad : ndarray
        Cell center longitudes in radians

    Returns
    -------
    grp_struct
        Result structure for NetCDF output
    """

    import sys
    import traceback

    try:
        logger.info(f"[START] Processing cell {c_idx}")

        # Compute context: one BufferPool per cell, shared across the FA/SA
        # get_pmf instances below; tile_cache accessor resolves to the
        # worker-local singleton initialised by client.run(...) in the main
        # loop. Explicit ctx replaces the previous pattern of implicit
        # BufferPool creation inside each get_pmf + module-global access to
        # tile_cache.get_worker_cache.
        from pycsa.compute import ComputeContext

        ctx = ComputeContext.default()

        topo = var.topo_cell()

        lat_verts = grid.clat_vertices[c_idx]
        lon_verts = grid.clon_vertices[c_idx]

        # Determine lat/lon extents with appropriate expansion for data loading
        lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)

        params.lat_extent = lat_extent
        params.lon_extent = lon_extent

        # Load topography for this cell from the worker-local tile cache.
        # The cache is initialised once per memory batch via init_worker_cache
        # (see the per-batch loop below); handles stay open across cells in
        # the same worker so we don't re-open the same ETOPO tile per cell.
        cache = ctx.tile_cache()
        topo.lat, topo.lon, topo.topo = cache.get_etopo_data(
            lat_extent, lon_extent, etopo_cg=params.etopo_cg
        )
        split_EW = tile_cache.compute_split_EW(lon_extent)

        # Clip deep bathymetry to -500m (same as test_etopo_pole_cells.py)
        # This prevents issues with extreme ocean depths creating artifacts
        topo.topo[np.where(topo.topo < -500.0)] = -500.0
        topo.gen_mgrids()

        # Handle dateline crossing BEFORE processing vertices for CSA
        # This must be done before handle_latlon_expansion() to ensure consistent coordinates
        if split_EW:
            lon_verts = lon_verts.copy()  # Don't modify the grid object
            lon_verts[lon_verts < 0.0] += 360.0

        # Process vertices for CSA (after dateline correction!)
        lat_verts, lon_verts = utils.handle_latlon_expansion(
            lat_verts, lon_verts, lat_expand=0.0, lon_expand=0.0
        )

        # Set up cell center and vertices
        clon = np.array([grid.clon[c_idx]])
        clat = np.array([grid.clat[c_idx]])
        clon_vertices = np.array([lon_verts])
        clat_vertices = np.array([lat_verts])

        ncells = 1
        nv = clon_vertices[0].size

        triangles = np.zeros((ncells, nv, 2))

        for i in range(0, ncells, 1):
            triangles[i, :, 0] = np.array(clon_vertices[i, :])
            triangles[i, :, 1] = np.array(clat_vertices[i, :])

        # Initialize cell objects for CSA algorithm
        tri_idx = 0
        cell = var.topo_cell()
        tri = var.obj()

        nhi = params.nhi
        nhj = params.nhj

        fa = interface.first_appx(nhi, nhj, params, topo, ctx=ctx)
        sa = interface.second_appx(nhi, nhj, params, topo, tri, ctx=ctx)

        tri.tri_lon_verts = triangles[:, :, 0]
        tri.tri_lat_verts = triangles[:, :, 1]

        simplex_lat = tri.tri_lat_verts[tri_idx]
        simplex_lon = tri.tri_lon_verts[tri_idx]

        if not utils.is_land(cell, simplex_lat, simplex_lon, topo):
            logger.info(f"[OCEAN] Cell {c_idx} is ocean, skipping")
            return writer.grp_struct(
                c_idx, clat_rad[c_idx], clon_rad[c_idx], 0, None, grid.cell_area[c_idx]
            )
        else:
            is_land = 1
            logger.info(f"[LAND] Cell {c_idx} is land, processing...")

        # First approximation
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(
            simplex_lat, simplex_lon, use_center=True
        )

        # Second approximation
        if USE_MODE_SELECTION:
            # COMPRESSED MODE: Use sa.do() to select top n_modes wavenumbers
            # This is the original workflow with spectral compression
            if params.recompute_rhs:
                sols, _ = sa.do(tri_idx, ampls_fa, use_center=True)
            else:
                sols = sa.do(tri_idx, ampls_fa, use_center=True)
            cell_sa, ampls_sa, uw_sa, dat_2D_sa = sols

            # Exclude ocean from spectral analysis (same as FULL SPECTRUM mode)
            ocean_mask = cell_sa.topo < -200.0
            cell_sa.mask = cell_sa.mask & ~ocean_mask
            cell_sa.get_masked(mask=cell_sa.mask)
        else:
            # FULL SPECTRUM MODE: Use ALL wavenumbers (no mode selection)
            # This gives ~20% better RMSE but no compression
            cell_sa = var.topo_cell()

            # Step 1: Load topo with rectangular mask
            utils.get_lat_lon_segments(
                simplex_lat,
                simplex_lon,
                cell_sa,
                topo,
                rect=True,
                filtered=True,
                padding=0,
                use_center=True,
            )

            # Step 2: Apply triangular mask
            utils.get_lat_lon_segments(
                simplex_lat,
                simplex_lon,
                cell_sa,
                topo,
                rect=False,
                filtered=False,
                padding=0,
                use_center=True,
            )

            # Run SA with ALL wavenumbers
            sa_pmf = interface.get_pmf(
                params.nhi, params.nhj, params.U, params.V, ctx=ctx
            )
            ampls_sa, uw_sa, dat_2D_sa = sa_pmf.sappx(
                cell_sa,
                lmbda=params.lmbda_sa,
                iter_solve=params.sa_iter_solve,
                updt_analysis=True,  # Populate cell_sa.analysis for NetCDF output
            )

            # Exclude ocean from spectral analysis for orographic gravity waves
            # The atmosphere flows over ocean SURFACE (0m), not the seafloor
            # Threshold: -200m distinguishes deep ocean from below-sea-level land
            #   - Most below-sea-level land features: -200m to 0m (Death Valley -86m, etc.)
            #   - Coastal ocean bathymetry: typically < -200m
            ocean_mask = cell_sa.topo < -200.0
            cell_sa.mask = cell_sa.mask & ~ocean_mask
            cell_sa.get_masked(mask=cell_sa.mask)

        # Store analysis results
        result = writer.grp_struct(
            c_idx,
            clat_rad[c_idx],
            clon_rad[c_idx],
            is_land,
            cell_sa.analysis,
            grid.cell_area[c_idx],
            topo_mean=getattr(cell_sa, "topo_mean", None),
        )

        # Generate 3-panel plot
        if params.plot_output:
            plot_cell_diagnostics(
                c_idx, cell_sa, ampls_sa, dat_2D_sa, chunk_output_dir, params
            )

        logger.info(f"[DONE] Cell {c_idx} analysis complete")

        # Explicit memory cleanup to help Dask workers
        del (
            topo,
            cell_fa,
            cell_sa,
            ampls_fa,
            ampls_sa,
            uw_fa,
            uw_sa,
            dat_2D_fa,
            dat_2D_sa,
        )
        del fa, sa, tri, cell, etopo_reader
        gc.collect()  # Force garbage collection

        return result

    except Exception as e:
        # Catch ALL exceptions and log them before worker dies
        error_msg = (
            f"[FATAL ERROR] Cell {c_idx} crashed with {type(e).__name__}: {str(e)}"
        )
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        # Print to stderr so it appears in worker logs
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

        # Re-raise to let Dask handle it
        raise


def parallel_wrapper(
    grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad
):
    return lambda ii: do_cell(
        ii, grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad
    )


from inputs.icon_global_run import params
from dask.distributed import Client, progress
import dask
from tqdm import tqdm

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION SELECTOR
    # ========================================================================
    # Choose one: 'generic_laptop', 'dkrz_hpc', 'laptop_performance'
    SYSTEM_CONFIG = "laptop_performance"  # ← Edit this line to switch configs
    # ========================================================================

    # ========================================================================
    # QUICK START GUIDE - Processing Specific Cell Ranges
    # ========================================================================
    # To process specific cell ranges (e.g., to regenerate corrupted chunks):
    #
    # 1. Scroll down to "CELL RANGE CONFIGURATION" section (around line 690)
    # 2. Set cell_start and cell_end:
    #
    #    Examples:
    #      cell_start = 0,    cell_end = 100    → Process cells 0-99 only
    #      cell_start = 2900, cell_end = 3000   → Process cells 2900-2999 only
    #      cell_start = 0,    cell_end = None   → Process all cells from 0 to end
    #      cell_start = 3000, cell_end = None   → Process from 3000 to end
    #
    # 3. Run the script - it will create appropriately named NetCDF files
    #
    # Note: Files are created in chunks of netcdf_chunk_size (default: 100)
    #       Example: cells 0-99 → icon_etopo_global_cells_00000-00099.nc
    # ========================================================================

    CONFIGS = {
        "generic_laptop": {
            "total_cores": 12,  # Conservative: use 12 of 16 threads
            "total_memory_gb": 12.0,
            "netcdf_chunk_size": 100,
            "threads_per_worker": 1,  # Set to None for auto-compute
            "memory_per_cpu_mb": None,  # Will calculate dynamically
            "description": "Generic laptop (16 threads, 16GB RAM)",
        },
        "dkrz_hpc": {
            "total_cores": 250,
            "total_memory_gb": 240.0,
            "netcdf_chunk_size": 100,
            "threads_per_worker": None,  # Auto-compute based on worker memory
            "memory_per_cpu_mb": None,  # SLURM quota on interactive partition
            "description": "DKRZ HPC interactive partition (standard memory node)",
        },
        "laptop_performance": {
            "total_cores": 20,  # Use 20 of 24 threads (leave 4 for background)
            "total_memory_gb": 80.0,
            "netcdf_chunk_size": 100,
            "threads_per_worker": None,  # Auto-compute based on worker memory
            "memory_per_cpu_mb": None,  # Will calculate dynamically
            "description": "AMD Ryzen AI 9 HX 370 (24 threads, 94GB RAM)",
        },
    }

    # Validate configuration selection
    if SYSTEM_CONFIG not in CONFIGS:
        raise ValueError(
            f"Invalid SYSTEM_CONFIG '{SYSTEM_CONFIG}'. Choose from: {list(CONFIGS.keys())}"
        )

    config = CONFIGS[SYSTEM_CONFIG]

    # Set up logging first
    log_file = setup_logger(log_dir="logs")
    print(f"Logging to: {log_file}")
    print("=" * 80)
    print(f"SYSTEM CONFIG: {SYSTEM_CONFIG}")
    print(f"  {config['description']}")
    print(f"  Cores: {config['total_cores']}, Memory: {config['total_memory_gb']} GB")
    print("=" * 80)

    # Override/add ETOPO-specific parameters
    params.fn_output = "icon_etopo_global"
    params.etopo_cg = (
        4  # Coarse-graining factor (1.8km at equator, ~0.9-1.8km at Drake Passage)
    )

    # Use traditional first approximation
    params.dfft_first_guess = False
    params.recompute_rhs = False

    # Disable plotting by default (set to True if you want diagnostic plots for each cell)
    params.plot_output = True

    # ========================================================================
    # SPECTRAL COMPRESSION TOGGLE
    # ========================================================================
    # Toggle between full spectrum vs compressed spectrum in second approximation:
    #
    # False (COMPRESSED - default): Use top n_modes=100 wavenumbers
    #   - Pros: 20x smaller NetCDF files, fast I/O, spectral compression feature
    #   - Cons: ~20% higher RMSE (e.g., 150.9m vs 121.0m for cell 3091)
    #
    # True (FULL SPECTRUM): Use ALL nhi*nhj=2048 wavenumbers
    #   - Pros: Best reconstruction quality (~20% lower RMSE)
    #   - Cons: 20x larger NetCDF files, no compression benefit
    #
    USE_FULL_SPECTRUM = False  # Set to True to disable spectral compression

    if USE_FULL_SPECTRUM:
        logger.info(
            "*** FULL SPECTRUM MODE: Using ALL wavenumbers (no compression) ***"
        )
        params.n_modes = params.nhi * params.nhj  # 2048 modes
        USE_MODE_SELECTION = False  # Use all modes in SA
    else:
        logger.info("*** COMPRESSED SPECTRUM MODE: Using top 100 wavenumbers ***")
        # params.n_modes already set to 100 in icon_global_run
        USE_MODE_SELECTION = True  # Select top n_modes in SA
    # ========================================================================

    if params.self_test():
        params.print()

    grid = var.grid()

    # Read ICON grid
    reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_dat(params.path_icon_grid, grid)

    clat_rad = np.copy(grid.clat)
    clon_rad = np.copy(grid.clon)

    grid.apply_f(utils.rad2deg)

    n_cells = grid.clat.size

    # Create base output directory
    base_output_dir = Path("outputs") / params.fn_output
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base output directory: {base_output_dir}")

    # ========================================================================
    # DYNAMIC MEMORY ALLOCATION SETUP
    # ========================================================================
    # Instead of fixed worker configuration, we'll dynamically adjust based on
    # the memory requirements of cells being processed (latitude-dependent)

    import multiprocessing
    import os

    # Use configuration values
    total_cores = config["total_cores"]
    total_memory_gb = config["total_memory_gb"]
    netcdf_chunk_size = config["netcdf_chunk_size"]

    logger.info("=" * 80)
    logger.info(f"RESOURCE CONFIGURATION: {SYSTEM_CONFIG}")
    logger.info(f"  Description: {config['description']}")
    logger.info(f"  Available cores: {total_cores}")
    logger.info(f"  Available memory: {total_memory_gb} GB")
    logger.info(f"  NetCDF chunk size: {netcdf_chunk_size} cells")

    # Threading configuration display
    if config["threads_per_worker"] is not None:
        logger.info(
            f"  Threading mode: MANUAL (threads_per_worker = {config['threads_per_worker']})"
        )
    else:
        logger.info(f"  Threading mode: AUTO (will compute based on worker count)")

    if config["memory_per_cpu_mb"] is not None:
        logger.info(f"  SLURM quota: {config['memory_per_cpu_mb']} MB per CPU")
    logger.info("=" * 80)

    # Group cells by memory requirements for dynamic worker allocation
    logger.info(f"\nAnalyzing cells by latitude for dynamic memory allocation...")
    memory_batches = group_cells_by_memory(
        clat_rad, max_memory_per_batch_gb=total_memory_gb
    )

    logger.info(f"Created {len(memory_batches)} memory-based batches:")
    for i, batch in enumerate(memory_batches):
        logger.info(
            f"  Batch {i}: {len(batch['cell_indices'])} cells, "
            f"{batch['memory_per_cell_gb']:.1f} GB/cell, "
            f"{batch['n_workers']} workers × {batch['memory_per_worker_gb']:.1f} GB"
        )

    # We'll create Dask client dynamically for each memory batch
    # Start with None (will be created per batch)
    client = None
    current_batch_idx = None

    logger.info(f"Total cells in grid: {n_cells}")

    # ========================================================================
    # CELL RANGE CONFIGURATION
    # ========================================================================
    # Set cell_start and cell_end to process specific ranges
    # Examples:
    #   cell_start = 0,    cell_end = None     → Process all cells (0 to n_cells-1)
    #   cell_start = 2900, cell_end = 3000     → Process cells 2900-2999 only
    #   cell_start = 0,    cell_end = 100      → Process cells 0-99 only
    cell_start = 0  # First cell to process (inclusive)
    cell_end = None  # Last cell to process (exclusive), None means process to end
    # ========================================================================

    # Validate and set cell_end
    if cell_end is None:
        cell_end = n_cells
    else:
        cell_end = min(cell_end, n_cells)  # Don't exceed total cells

    if cell_start >= cell_end:
        raise ValueError(
            f"Invalid cell range: cell_start ({cell_start}) >= cell_end ({cell_end})"
        )

    # Progress tracking
    cells_to_process = cell_end - cell_start
    total_netcdf_chunks = (
        cells_to_process + netcdf_chunk_size - 1
    ) // netcdf_chunk_size
    logger.info(
        f"\nProcessing cell range: {cell_start} to {cell_end-1} ({cells_to_process} cells)"
    )
    logger.info(
        f"  NetCDF chunks: {total_netcdf_chunks} files ({netcdf_chunk_size} cells each)\n"
    )

    # Statistics
    total_land_cells = 0
    total_ocean_cells = 0

    # Configure task retries and logging (do this once)
    import dask
    import logging

    dask.config.set({"distributed.scheduler.allowed-failures": 0})
    logging.getLogger("distributed.worker.memory").setLevel(logging.ERROR)

    # Create a mapping from cell_idx to memory batch index for quick lookup
    cell_to_batch = {}
    for batch_idx, batch in enumerate(memory_batches):
        for cell_idx in batch["cell_indices"]:
            cell_to_batch[cell_idx] = batch_idx

    # ========================================================================
    # SEQUENTIAL PROCESSING BY MEMORY BATCH
    # ========================================================================
    # Process memory batches sequentially (equatorial → mid-lat → polar)
    # This allows easy restart: if script crashes, you know all previous
    # memory batches are complete and can skip to the current batch.
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING STRATEGY: Sequential by Memory Batch")
    logger.info("=" * 80)
    for batch_idx, batch_config in enumerate(memory_batches):
        logger.info(f"\n{'='*80}")
        logger.info(
            f"MEMORY BATCH {batch_idx}/{len(memory_batches)-1}: {len(batch_config['cell_indices'])} cells"
        )
        logger.info(f"  Memory per cell: {batch_config['memory_per_cell_gb']:.1f} GB")
        logger.info(f"  Workers: {batch_config['n_workers']}")
        logger.info(f"{'='*80}\n")

        # Get all cells in this memory batch
        batch_cell_indices = set(batch_config["cell_indices"])

        # Create Dask client for this memory batch
        n_workers = batch_config["n_workers"]
        # Single-worker batches (high-memory polar cells) get the full machine
        # memory; multi-worker batches share by config.
        if n_workers == 1:
            memory_per_worker = f"{int(total_memory_gb)}GB"
            logger.info(
                f"  Single-worker mode: allowing full memory access ({total_memory_gb} GB)"
            )
        else:
            memory_per_worker = f"{int(batch_config['memory_per_worker_gb'])}GB"
        threads_per_worker = 1  # HDF5 not thread-safe

        logger.info(f"Starting Dask client for memory batch {batch_idx}:")
        logger.info(f"  Workers: {n_workers} × {memory_per_worker}")
        logger.info(f"  Threads per worker: {threads_per_worker}")

        client = Client(
            threads_per_worker=threads_per_worker,
            n_workers=n_workers,
            processes=True,
            memory_limit=memory_per_worker,
            silence_logs="ERROR",
        )
        logger.info(f"  Dashboard: {client.dashboard_link}\n")

        # Initialise the per-worker tile cache. Each worker is a separate
        # process, so this populates a module-level _WORKER_CACHE inside that
        # process; do_cell then reaches it via tile_cache.get_worker_cache().
        # The cache opens ETOPO tile files lazily on first access and keeps
        # the handles for the rest of the worker's lifetime.
        init_results = client.run(
            tile_cache.init_worker_cache, params.path_etopo, "ETOPO"
        )
        logger.info(
            f"  Initialised tile cache on {sum(bool(v) for v in init_results.values())} "
            f"of {len(init_results)} workers"
        )

        # Inner loop: NetCDF file creation (one file per netcdf_chunk_size cells)
        # Only process NetCDF chunks that contain cells from this memory batch
        for netcdf_chunk_idx, netcdf_chunk_start in enumerate(
            tqdm(
                range(cell_start, n_cells, netcdf_chunk_size),
                desc=f"NetCDF chunks (batch {batch_idx})",
                total=total_netcdf_chunks,
            )
        ):
            netcdf_chunk_end = min(netcdf_chunk_start + netcdf_chunk_size, n_cells)

            # Filter: only process cells in this NetCDF chunk that belong to current memory batch
            cell_indices_in_chunk = []
            for c_idx in range(netcdf_chunk_start, netcdf_chunk_end):
                if c_idx in batch_cell_indices:
                    cell_indices_in_chunk.append(c_idx)

            # Skip this NetCDF chunk if no cells belong to current memory batch
            if not cell_indices_in_chunk:
                continue

            logger.info(
                f"\n  Processing NetCDF chunk {netcdf_chunk_idx}: cells {netcdf_chunk_start}-{netcdf_chunk_end-1}"
            )
            logger.info(f"    Cells in this batch: {len(cell_indices_in_chunk)}")

            # Create subdirectory for this NetCDF chunk's plots
            chunk_output_dir = (
                base_output_dir
                / f"cells_{netcdf_chunk_start:05d}-{netcdf_chunk_end-1:05d}"
            )
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            # Writer object for this NetCDF chunk
            sfx = f"_cells_{netcdf_chunk_start:05d}-{netcdf_chunk_end-1:05d}"
            writer = io.nc_writer(params, sfx)

            pw_run = parallel_wrapper(
                grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad
            )

            # Process cells in smaller batches to avoid overwhelming scheduler
            processing_batch_size = min(n_workers * 2, len(cell_indices_in_chunk))

            for i in range(0, len(cell_indices_in_chunk), processing_batch_size):
                batch_cells = cell_indices_in_chunk[i : i + processing_batch_size]

                # Submit batch to Dask
                lazy_results = []
                for c_idx in batch_cells:
                    lazy_result = dask.delayed(pw_run)(c_idx)
                    lazy_results.append(lazy_result)

                # Compute batch
                results = dask.compute(*lazy_results)

                # Write results to NetCDF file
                for item in results:
                    writer.duplicate(item.c_idx, item)
                    if item.is_land:
                        total_land_cells += 1
                    else:
                        total_ocean_cells += 1

            # Cleanup after each NetCDF chunk
            if hasattr(reader, "close_cached_files"):
                reader.close_cached_files()

            gc.collect()

            logger.info(
                f"    NetCDF chunk complete: {len(cell_indices_in_chunk)} cells processed"
            )
            logger.info(
                f"    Running totals - Land: {total_land_cells}, Ocean: {total_ocean_cells}"
            )

        # Close Dask client after finishing this memory batch
        client.close()
        logger.info(f"\n{'='*80}")
        logger.info(f"MEMORY BATCH {batch_idx} COMPLETE")
        logger.info(f"  Processed {len(batch_cell_indices)} cells")
        logger.info(f"{'='*80}\n")

    # Cleanup: close all cached NetCDF files
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total cells processed: {total_land_cells + total_ocean_cells}")
    logger.info(f"  Land cells: {total_land_cells}")
    logger.info(f"  Ocean cells: {total_ocean_cells}")
    logger.info(f"\nNetCDF files created: {total_netcdf_chunks}")
    logger.info(f"  Location: {params.path_output}datasets/")
    logger.info(f"  Pattern: icon_etopo_global_cells_XXXXX-XXXXX.nc")
    logger.info(f"\nTo merge into single file, run:")
    logger.info(f"  python3 -m runs.merge_netcdf_chunks")
    logger.info("=" * 80)

    if hasattr(reader, "close_cached_files"):
        reader.close_cached_files()
        logger.info("\n✓ Closed cached topography files")

    # Final console message
    print("=" * 80)
    print(f"PROCESSING COMPLETE - Check log file: {log_file}")
    print("=" * 80)
