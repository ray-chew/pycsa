import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for parallel processing
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
from pathlib import Path
import gc

from pycsa.core import io, var, utils
from pycsa.wrappers import interface, diagnostics
from pycsa.plotting import plotter


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
    vmin = -500.0  # Always fix ocean floor at -500m (blue portion)
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

    print(f"  Plot saved: {output_path}")


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

    print(c_idx)

    topo = var.topo_cell()

    lat_verts = grid.clat_vertices[c_idx]
    lon_verts = grid.clon_vertices[c_idx]

    # Determine lat/lon extents with appropriate expansion for data loading
    lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)
    lat_verts, lon_verts = utils.handle_latlon_expansion(lat_verts, lon_verts, lat_expand = 0.0, lon_expand = 0.0)

    params.lat_extent = lat_extent
    params.lon_extent = lon_extent


    # Load topography data for this cell (ETOPO instead of MERIT)
    etopo_reader = reader.read_etopo_topo(None, params, is_parallel=True)
    etopo_reader.get_topo(topo)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()

    # Set up cell center and vertices
    clon = np.array([grid.clon[c_idx]])
    clat = np.array([grid.clat[c_idx]])
    clon_vertices = np.array([lon_verts])
    clat_vertices = np.array([lat_verts])

    ncells = 1
    nv = clon_vertices[0].size

    # Handle dateline crossing
    if etopo_reader.split_EW:
        clon_vertices[clon_vertices < 0.0] += 360.0

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
        print("--> skipping ocean cell")
        return writer.grp_struct(c_idx, clat_rad[c_idx], clon_rad[c_idx], 0)
    else:
        is_land = 1

    # Traditional first approximation (not DFFT first guess)
    cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

    kls_fa = None  # Traditional approach doesn't use DFFT wavenumbers

    sols = (cell_fa, ampls_fa, uw_fa, dat_2D_fa)

    # Second approximation
    if params.recompute_rhs:
        sols, _ = sa.do(tri_idx, ampls_fa)
    else:
        sols = sa.do(tri_idx, ampls_fa)

    cell_sa, ampls_sa, uw_sa, dat_2D_sa = sols

    # Store analysis results
    result = writer.grp_struct(c_idx, clat_rad[c_idx], clon_rad[c_idx], is_land, cell_sa.analysis)

    # Generate 3-panel plot
    if params.plot_output:
        plot_cell_diagnostics(
            c_idx, cell_sa, ampls_sa, dat_2D_sa,
            chunk_output_dir, params
        )

    print("--> analysis done")

    # Explicit memory cleanup to help Dask workers
    del topo, cell_fa, cell_sa, ampls_fa, ampls_sa, uw_fa, uw_sa, dat_2D_fa, dat_2D_sa
    del fa, sa, tri, cell, etopo_reader
    gc.collect()  # Force garbage collection

    return result


def parallel_wrapper(grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad):
    return lambda ii : do_cell(ii, grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad)


from inputs.icon_global_run import params
from dask.distributed import Client, progress
import dask
from tqdm import tqdm

if __name__ == '__main__':
    # Override/add ETOPO-specific parameters
    params.fn_output = "icon_etopo_global"
    params.etopo_cg = 4  # Coarse-graining factor (1.8km at equator, ~0.9-1.8km at Drake Passage)

    # Use traditional first approximation
    params.dfft_first_guess = False
    params.recompute_rhs = False

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
    print(f"Base output directory: {base_output_dir}")

    # Configure Dask for parallel processing
    # Use processes (not threads) to avoid NetCDF file locking issues
    # Each worker gets 1 thread to avoid GIL contention
    # MEMORY OPTIMIZATION: Fewer workers with more memory each for ETOPO full resolution
    import multiprocessing
    n_workers = 6  # Reduced from 20 to give each worker more memory
    print(f"Initializing Dask with {n_workers} workers...")
    print(f"Memory optimization: 6 workers × 10GB = ~60GB total")

    client = Client(
        threads_per_worker=1,
        n_workers=n_workers,
        processes=True,
        memory_limit='10GB'  # Increased from 4GB for ETOPO CG=4 data volumes
    )
    print(f"Dask dashboard available at: {client.dashboard_link}")

    print(f"Total cells to process: {n_cells}")

    chunk_sz = 10
    chunk_start = 0  # Start from beginning (can be modified for restart)

    # Progress tracking
    total_chunks = (n_cells - chunk_start + chunk_sz - 1) // chunk_sz
    print(f"\nProcessing {n_cells - chunk_start} cells in {total_chunks} chunks of {chunk_sz}...")

    for chunk_idx, chunk in enumerate(tqdm(range(chunk_start, n_cells, chunk_sz), desc="Processing chunks")):
        # Create subdirectory for this chunk
        chunk_output_dir = base_output_dir / f"chunk_{chunk:05d}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)

        # Writer object for this chunk
        sfx = "_" + str(chunk+chunk_sz)
        writer = io.nc_writer(params, sfx)

        pw_run = parallel_wrapper(grid, params, reader, writer, chunk_output_dir, clat_rad, clon_rad)

        lazy_results = []

        if chunk+chunk_sz > n_cells:
            chunk_end = n_cells
        else:
            chunk_end = chunk+chunk_sz

        for c_idx in range(chunk, chunk_end):
            lazy_result = dask.delayed(pw_run)(c_idx)
            lazy_results.append(lazy_result)

        results = dask.compute(*lazy_results)

        for item in results:
            writer.duplicate(item.c_idx, item)

    # Cleanup: close all cached NetCDF files and shut down Dask client
    print("\nCleaning up...")
    if hasattr(reader, 'close_cached_files'):
        reader.close_cached_files()
        print("✓ Closed cached topography files")

    client.close()
    print("✓ Shut down Dask client")
    print("Processing complete!")
