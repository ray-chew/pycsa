import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for parallel processing
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from pathlib import Path
import gc
import logging
from datetime import datetime

from pycsa.core import io, var, utils
from pycsa.wrappers import interface, diagnostics
from pycsa.plotting import plotter

# Initialize logger (will be configured in main)
logger = logging.getLogger(__name__)


def setup_logger(log_dir="logs"):
    """
    Set up logging configuration for ETOPO global run.

    Parameters
    ----------
    log_dir : str
        Directory for log files (default: "logs")

    Returns
    -------
    Path
        Path to the log file
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"icon_etopo_global_{timestamp}.log"

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers.clear()

    # File handler - logs everything
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Also silence matplotlib and other libraries from console
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('distributed').setLevel(logging.WARNING)

    return log_file


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
    return mcolors.LinearSegmentedColormap.from_list('topo', colors)


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

    im1 = axs[0].imshow(topo_original, origin='lower', cmap=topo_cmap,
                        norm=norm, aspect='auto')
    axs[0].set_title(f'Cell {c_idx}: Loaded Topography\nRange: [{vmin:.0f}, {vmax:.0f}] m',
                     fontsize=11, fontweight='bold')
    axs[0].set_xlabel('Longitude index')
    axs[0].set_ylabel('Latitude index')
    cbar1 = plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Elevation [m]', rotation=270, labelpad=15)

    # Panel 2: Reconstructed topography (masked)
    dat_2D_masked = dat_2D_sa.copy()
    dat_2D_masked[~cell_sa.mask] = np.nan

    # Compute reconstruction error
    diff = cell_sa.topo - dat_2D_sa
    rmse = np.sqrt(np.mean(diff[cell_sa.mask]**2))
    rel_rmse = rmse / (vmax - vmin) * 100

    im2 = axs[1].imshow(dat_2D_masked, origin='lower', cmap=topo_cmap,
                        norm=norm, aspect='auto')
    axs[1].set_title(f'Reconstructed (2nd Approx)\nRMSE: {rmse:.1f} m ({rel_rmse:.1f}%)',
                     fontsize=11, fontweight='bold')
    axs[1].set_xlabel('Longitude index')
    axs[1].set_ylabel('Latitude index')
    cbar2 = plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Elevation [m]', rotation=270, labelpad=15)

    # Panel 3: Amplitude spectrum in (k,l) wavenumber space
    fig_obj = plotter.fig_obj(fig, params.nhi, params.nhj, cbar=True, set_label=True)
    axs[2] = fig_obj.freq_panel(
        axs[2],
        ampls_sa,
        title="Amplitude Spectrum",
        v_extent=None
    )

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"cell_{c_idx:05d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Explicit memory cleanup
    del fig, axs, fig_obj, im1, im2, topo_original, dat_2D_masked

    logger.info(f"  Plot saved: {output_path}")


def do_cell(c_idx,
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

    logger.info(f"[START] Processing cell {c_idx}")

    topo = var.topo_cell()

    lat_verts = grid.clat_vertices[c_idx]
    lon_verts = grid.clon_vertices[c_idx]

    # Determine lat/lon extents with appropriate expansion for data loading
    lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)

    params.lat_extent = lat_extent
    params.lon_extent = lon_extent

    # Load topography data for this cell (ETOPO instead of MERIT)
    etopo_reader = reader.read_etopo_topo(None, params, is_parallel=True)
    etopo_reader.get_topo(topo)

    # Clip deep bathymetry to -500m (same as test_etopo_pole_cells.py)
    # This prevents issues with extreme ocean depths creating artifacts
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()

    # Handle dateline crossing BEFORE processing vertices for CSA
    # This must be done before handle_latlon_expansion() to ensure consistent coordinates
    if etopo_reader.split_EW:
        lon_verts = lon_verts.copy()  # Don't modify the grid object
        lon_verts[lon_verts < 0.0] += 360.0

    # Process vertices for CSA (after dateline correction!)
    lat_verts, lon_verts = utils.handle_latlon_expansion(lat_verts, lon_verts, lat_expand = 0.0, lon_expand = 0.0)

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

    fa = interface.first_appx(nhi, nhj, params, topo)
    sa = interface.second_appx(nhi, nhj, params, topo, tri)

    tri.tri_lon_verts = triangles[:, :, 0]
    tri.tri_lat_verts = triangles[:, :, 1]

    simplex_lat = tri.tri_lat_verts[tri_idx]
    simplex_lon = tri.tri_lon_verts[tri_idx]

    if not utils.is_land(cell, simplex_lat, simplex_lon, topo):
        logger.info(f"[OCEAN] Cell {c_idx} is ocean, skipping")
        return writer.grp_struct(c_idx, clat_rad[c_idx], clon_rad[c_idx], 0)
    else:
        is_land = 1
        logger.info(f"[LAND] Cell {c_idx} is land, processing...")

    # First approximation
    cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon, use_center=True)

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
            simplex_lat, simplex_lon, cell_sa, topo,
            rect=True, filtered=True, padding=0, use_center=True
        )

        # Step 2: Apply triangular mask
        utils.get_lat_lon_segments(
            simplex_lat, simplex_lon, cell_sa, topo,
            rect=False, filtered=False, padding=0, use_center=True
        )

        # Run SA with ALL wavenumbers
        sa_pmf = interface.get_pmf(params.nhi, params.nhj, params.U, params.V)
        ampls_sa, uw_sa, dat_2D_sa = sa_pmf.sappx(
            cell_sa,
            lmbda=params.lmbda_sa,
            iter_solve=params.sa_iter_solve,
            updt_analysis=True  # Populate cell_sa.analysis for NetCDF output
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
    result = writer.grp_struct(c_idx, clat_rad[c_idx], clon_rad[c_idx], is_land, cell_sa.analysis)

    # Generate 3-panel plot
    if params.plot_output:
        plot_cell_diagnostics(
            c_idx, cell_sa, ampls_sa, dat_2D_sa,
            chunk_output_dir, params
        )

    logger.info(f"[DONE] Cell {c_idx} analysis complete")

    # Explicit memory cleanup to help Dask workers
    del topo, cell_fa, cell_sa, ampls_fa, ampls_sa, uw_fa, uw_sa, dat_2D_fa, dat_2D_sa
    del fa, sa, tri, cell, etopo_reader
    gc.collect()  # Force garbage collection

    return result


def estimate_cell_memory_gb(lat_deg):
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
    - Polar cells (~80-89°): ~60 GB required

    Memory scales approximately with 1/cos(lat) due to meridian convergence,
    but caps at ~60 GB for cells very close to the poles.
    """
    abs_lat = np.abs(lat_deg)

    # Base memory requirement at equator
    base_memory_gb = 10.0

    # Scale factor based on latitude (empirical fit)
    if abs_lat < 60.0:
        # Below 60°, memory is fairly constant
        scale_factor = 1.0
    elif abs_lat < 85.0:
        # Between 60° and 85°, use power law scaling
        # At 70°: (1/0.342)^0.7 ≈ 2.5, giving 25 GB
        # At 80°: (1/0.174)^0.7 ≈ 4.3, giving 43 GB
        lat_rad = np.deg2rad(abs_lat)
        cos_lat = np.cos(lat_rad)
        scale_factor = (1.0 / cos_lat) ** 0.7
    else:
        # Above 85°, cap at 6x base (60 GB) to avoid unrealistic estimates
        # Very close to poles, the ICON grid cells are smaller and don't
        # actually require infinite memory despite cos(lat)→0
        scale_factor = 6.0

    return base_memory_gb * scale_factor


def group_cells_by_memory(clat_rad, max_memory_per_batch_gb=240.0):
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
                n_workers = max(1, int(max_memory_per_batch_gb / (avg_mem_current * 1.2)))  # 20% safety margin
                mem_per_worker = avg_mem_current * 1.2

                batches.append({
                    'cell_indices': sorted(current_batch_indices),  # Sort by original index order
                    'memory_per_cell_gb': avg_mem_current,
                    'n_workers': n_workers,
                    'memory_per_worker_gb': mem_per_worker
                })

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
        n_workers = max(1, int(max_memory_per_batch_gb / (avg_mem * 1.2)))
        mem_per_worker = avg_mem * 1.2

        batches.append({
            'cell_indices': sorted(current_batch_indices),
            'memory_per_cell_gb': avg_mem,
            'n_workers': n_workers,
            'memory_per_worker_gb': mem_per_worker
        })

    return batches


def parallel_wrapper(grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad):
    return lambda ii : do_cell(ii, grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad)


from inputs.icon_global_run import params
from dask.distributed import Client, progress
import dask
from tqdm import tqdm

if __name__ == '__main__':
    # ========================================================================
    # CONFIGURATION SELECTOR
    # ========================================================================
    # Choose one: 'generic_laptop', 'dkrz_hpc', 'laptop_performance'
    SYSTEM_CONFIG = 'laptop_performance'  # ← Edit this line to switch configs
    # ========================================================================

    CONFIGS = {
        'generic_laptop': {
            'total_cores': 12,  # Conservative: use 12 of 16 threads
            'total_memory_gb': 12.0,
            'netcdf_chunk_size': 100,
            'memory_per_cpu_mb': None,  # Will calculate dynamically
            'description': 'Generic laptop (16 threads, 16GB RAM)'
        },
        'dkrz_hpc': {
            'total_cores': 128,
            'total_memory_gb': 240.0,
            'netcdf_chunk_size': 1000,
            'memory_per_cpu_mb': 1940,  # SLURM quota on interactive partition
            'description': 'DKRZ HPC interactive partition (standard memory node)'
        },
        'laptop_performance': {
            'total_cores': 20,  # Use 20 of 24 threads (leave 4 for background)
            'total_memory_gb': 80.0,
            'netcdf_chunk_size': 100,
            'memory_per_cpu_mb': None,  # Will calculate dynamically
            'description': 'AMD Ryzen AI 9 HX 370 (24 threads, 94GB RAM)'
        }
    }

    # Validate configuration selection
    if SYSTEM_CONFIG not in CONFIGS:
        raise ValueError(f"Invalid SYSTEM_CONFIG '{SYSTEM_CONFIG}'. Choose from: {list(CONFIGS.keys())}")

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
    params.etopo_cg = 4  # Coarse-graining factor (1.8km at equator, ~0.9-1.8km at Drake Passage)

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
        logger.info("*** FULL SPECTRUM MODE: Using ALL wavenumbers (no compression) ***")
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
    total_cores = config['total_cores']
    total_memory_gb = config['total_memory_gb']
    netcdf_chunk_size = config['netcdf_chunk_size']

    logger.info("=" * 80)
    logger.info(f"RESOURCE CONFIGURATION: {SYSTEM_CONFIG}")
    logger.info(f"  Description: {config['description']}")
    logger.info(f"  Available cores: {total_cores}")
    logger.info(f"  Available memory: {total_memory_gb} GB")
    logger.info(f"  NetCDF chunk size: {netcdf_chunk_size} cells")
    if config['memory_per_cpu_mb'] is not None:
        logger.info(f"  SLURM quota: {config['memory_per_cpu_mb']} MB per CPU")
        logger.info(f"  Mode: HPC (threads scale with worker memory)")
    else:
        logger.info(f"  Mode: Laptop (threads distributed evenly)")
    logger.info("=" * 80)

    # Group cells by memory requirements for dynamic worker allocation
    logger.info(f"\nAnalyzing cells by latitude for dynamic memory allocation...")
    memory_batches = group_cells_by_memory(clat_rad, max_memory_per_batch_gb=total_memory_gb)

    logger.info(f"Created {len(memory_batches)} memory-based batches:")
    for i, batch in enumerate(memory_batches):
        logger.info(f"  Batch {i}: {len(batch['cell_indices'])} cells, "
                   f"{batch['memory_per_cell_gb']:.1f} GB/cell, "
                   f"{batch['n_workers']} workers × {batch['memory_per_worker_gb']:.1f} GB")

    # We'll create Dask client dynamically for each memory batch
    # Start with None (will be created per batch)
    client = None
    current_batch_idx = None

    logger.info(f"Total cells to process: {n_cells}")

    cell_start = 0  # Start from beginning (can be modified for restart)

    # Progress tracking
    total_netcdf_chunks = (n_cells - cell_start + netcdf_chunk_size - 1) // netcdf_chunk_size
    logger.info(f"\nProcessing {n_cells - cell_start} cells:")
    logger.info(f"  NetCDF chunks: {total_netcdf_chunks} files ({netcdf_chunk_size} cells each)\n")

    # Statistics
    total_land_cells = 0
    total_ocean_cells = 0

    # Configure task retries and logging (do this once)
    import dask
    import logging
    dask.config.set({'distributed.scheduler.allowed-failures': 0})
    logging.getLogger('distributed.worker.memory').setLevel(logging.ERROR)

    # Create a mapping from cell_idx to memory batch index for quick lookup
    cell_to_batch = {}
    for batch_idx, batch in enumerate(memory_batches):
        for cell_idx in batch['cell_indices']:
            cell_to_batch[cell_idx] = batch_idx

    # Outer loop: NetCDF file creation (one file per netcdf_chunk_size cells)
    for netcdf_chunk_idx, netcdf_chunk_start in enumerate(tqdm(
        range(cell_start, n_cells, netcdf_chunk_size),
        desc="NetCDF chunks",
        total=total_netcdf_chunks
    )):
        netcdf_chunk_end = min(netcdf_chunk_start + netcdf_chunk_size, n_cells)

        # Create subdirectory for this NetCDF chunk's plots
        chunk_output_dir = base_output_dir / f"cells_{netcdf_chunk_start:05d}-{netcdf_chunk_end-1:05d}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)

        # Writer object for this NetCDF chunk
        sfx = f"_cells_{netcdf_chunk_start:05d}-{netcdf_chunk_end-1:05d}"
        writer = io.nc_writer(params, sfx)

        pw_run = parallel_wrapper(grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad)

        # Group cells in this NetCDF chunk by memory batch
        cells_by_memory_batch = {}
        for c_idx in range(netcdf_chunk_start, netcdf_chunk_end):
            if c_idx in cell_to_batch:
                mem_batch_idx = cell_to_batch[c_idx]
                if mem_batch_idx not in cells_by_memory_batch:
                    cells_by_memory_batch[mem_batch_idx] = []
                cells_by_memory_batch[mem_batch_idx].append(c_idx)

        # Process each memory batch with appropriate Dask configuration
        for mem_batch_idx in sorted(cells_by_memory_batch.keys()):
            cell_indices = cells_by_memory_batch[mem_batch_idx]
            batch_config = memory_batches[mem_batch_idx]

            # Check if we need to reconfigure Dask client
            if current_batch_idx != mem_batch_idx:
                # Shutdown previous client if it exists
                if client is not None:
                    client.close()
                    logger.info(f"\n  Closed previous Dask client")

                # Create new client with appropriate memory configuration
                n_workers = batch_config['n_workers']
                memory_per_worker = f"{int(batch_config['memory_per_worker_gb'])}GB"

                # Calculate threads per worker based on configuration
                if config['memory_per_cpu_mb'] is not None:
                    # HPC mode: Use SLURM's memory-per-CPU quota
                    # Each worker gets CPUs proportional to its memory allocation
                    threads_per_worker = max(1, int(
                        batch_config['memory_per_worker_gb'] * 1000 / config['memory_per_cpu_mb']
                    ))
                else:
                    # Laptop mode: Calculate based on total available resources
                    # How many workers can we fit given memory constraints?
                    max_workers_by_memory = max(1, int(
                        config['total_memory_gb'] / batch_config['memory_per_worker_gb']
                    ))
                    # Limit workers to what we actually configured
                    actual_workers = min(max_workers_by_memory, n_workers)
                    # Distribute threads evenly across workers
                    threads_per_worker = max(1, config['total_cores'] // actual_workers)

                logger.info(f"\n  Starting Dask client for memory batch {mem_batch_idx}:")
                logger.info(f"    Workers: {n_workers} × {memory_per_worker}")
                logger.info(f"    Threads per worker: {threads_per_worker}")
                logger.info(f"    Expected memory per cell: {batch_config['memory_per_cell_gb']:.1f} GB")

                client = Client(
                    threads_per_worker=threads_per_worker,
                    n_workers=n_workers,
                    processes=True,
                    memory_limit=memory_per_worker,
                    silence_logs='ERROR',
                )
                logger.info(f"    Dashboard: {client.dashboard_link}")

                current_batch_idx = mem_batch_idx

            # Process cells in smaller batches to avoid overwhelming scheduler
            processing_batch_size = min(batch_config['n_workers'] * 2, len(cell_indices))

            for i in range(0, len(cell_indices), processing_batch_size):
                batch_cells = cell_indices[i:i+processing_batch_size]

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
        if hasattr(reader, 'close_cached_files'):
            reader.close_cached_files()

        gc.collect()

        logger.info(f"\n  NetCDF chunk {netcdf_chunk_idx}: Cells {netcdf_chunk_start}-{netcdf_chunk_end-1} complete")
        logger.info(f"    Land: {total_land_cells}, Ocean: {total_ocean_cells}, Total: {total_land_cells + total_ocean_cells}")

    # Cleanup: close all cached NetCDF files and shut down Dask client
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total cells processed: {total_land_cells + total_ocean_cells}")
    logger.info(f"  Land cells: {total_land_cells}")
    logger.info(f"  Ocean cells: {total_ocean_cells}")
    logger.info(f"\nNetCDF files created: {total_netcdf_chunks}")
    logger.info(f"  Location: {params.path_output}datasets/")
    logger.info(f"  Pattern: icon_etopo_global_cells_XXXXX-XXXXX.nc")
    logger.info(f"\nTo merge into single file, run:")
    logger.info(f"  python3 -m runs.merge_netcdf_chunks")
    logger.info("="*80)

    if hasattr(reader, 'close_cached_files'):
        reader.close_cached_files()
        logger.info("\n✓ Closed cached topography files")

    if client is not None:
        client.close()
        logger.info("✓ Shut down Dask client")

    # Final console message
    print("="*80)
    print(f"PROCESSING COMPLETE - Check log file: {log_file}")
    print("="*80)
