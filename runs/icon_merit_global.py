import numpy as np

from pycsa.core import io, var, utils
from pycsa.wrappers import interface, diagnostics
from pycsa.plotting import cart_plot


def do_cell(
    c_idx,
    grid,
    params,
    reader,
    writer,
):

    print(c_idx)

    topo = var.topo_cell()

    lat_verts = grid.clat_vertices[c_idx]
    lon_verts = grid.clon_vertices[c_idx]

    # Determine lat/lon extents with appropriate expansion for data loading
    lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)
    lat_verts, lon_verts = utils.handle_latlon_expansion(
        lat_verts, lon_verts, lat_expand=0.0, lon_expand=0.0
    )

    params.lat_extent = lat_extent
    params.lon_extent = lon_extent

    # Load topography data for this cell
    reader = reader.read_merit_topo(None, params, is_parallel=True)
    reader.get_topo(topo)
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
    if reader.split_EW:
        clon_vertices[clon_vertices < 0.0] += 360.0

    triangles = np.zeros((ncells, nv, 2))

    for i in range(0, ncells, 1):
        triangles[i, :, 0] = np.array(clon_vertices[i, :])
        triangles[i, :, 1] = np.array(clat_vertices[i, :])

    if params.plot or params.plot_output:
        output_fn = params.path_output + str(c_idx) + ".png"
        cart_plot.lat_lon_icon(
            topo,
            triangles,
            ncells=ncells,
            clon=clon,
            clat=clat,
            title=c_idx,
            fn=output_fn,
            output_fig=True,
        )

    # Initialize cell objects for CSA algorithm
    tri_idx = 0
    cell = var.topo_cell()
    tri = var.obj()

    nhi = params.nhi
    nhj = params.nhj

    fa = interface.first_appx(nhi, nhj, params, topo)
    sa = interface.second_appx(nhi, nhj, params, topo, tri)

    dplot = diagnostics.diag_plotter(params, nhi, nhj)
    dplot.output_dir = params.path_output

    tri.tri_lon_verts = triangles[:, :, 0]
    tri.tri_lat_verts = triangles[:, :, 1]

    simplex_lat = tri.tri_lat_verts[tri_idx]
    simplex_lon = tri.tri_lon_verts[tri_idx]

    if not utils.is_land(cell, simplex_lat, simplex_lon, topo):
        # writer.output(c_idx, clat_rad[c_idx], clon_rad[c_idx], 0)
        print("--> skipping ocean cell")
        return writer.grp_struct(c_idx, clat_rad[c_idx], clon_rad[c_idx], 0)
    else:
        is_land = 1

    if params.dfft_first_guess:
        # do tapering
        if params.taper_fa:
            interface.taper_quad(params, simplex_lat, simplex_lon, cell, topo)
        else:
            utils.get_lat_lon_segments(
                simplex_lat, simplex_lon, cell, topo, rect=params.rect
            )

        dfft_run = interface.get_pmf(nhi, nhj, params.U, params.V)
        ampls_fa, uw_fa, dat_2D_fa, kls_fa = dfft_run.dfft(cell)

        cell_fa = cell

        nhi = len(cell_fa.lon)
        nhj = len(cell_fa.lat)

        sa.nhi = nhi
        sa.nhj = nhj
    else:
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

    sols = (cell_fa, ampls_fa, uw_fa, dat_2D_fa)

    v_extent = [dat_2D_fa.min(), dat_2D_fa.max()]

    if params.plot:
        if params.dfft_first_guess:
            dplot.show(
                tri_idx,
                sols,
                kls=kls_fa,
                v_extent=v_extent,
                dfft_plot=True,
                output_fig=False,
            )
        else:
            dplot.show(c_idx, sols, v_extent=v_extent, output_fig=False)

    if params.recompute_rhs:
        sols, _ = sa.do(tri_idx, ampls_fa)
    else:
        sols = sa.do(tri_idx, ampls_fa)

    cell, ampls_sa, uw_sa, dat_2D_sa = sols
    v_extent = [dat_2D_sa.min(), dat_2D_sa.max()]

    # writer.output(c_idx, clat_rad[c_idx], clon_rad[c_idx], is_land, cell.analysis)
    result = writer.grp_struct(
        c_idx,
        clat_rad[c_idx],
        clon_rad[c_idx],
        is_land,
        cell.analysis,
        topo_mean=getattr(cell, "topo_mean", None),
    )

    if params.plot:
        if params.dfft_first_guess:
            dplot.show(
                tri_idx,
                sols,
                kls=kls_fa,
                v_extent=v_extent,
                dfft_plot=True,
                output_fig=False,
            )
        else:
            dplot.show(c_idx, sols, v_extent=v_extent, output_fig=False)

    print("--> analysis done")

    return result


def parallel_wrapper(grid, params, reader, writer):
    return lambda ii: do_cell(ii, grid, params, reader, writer)


from pycsa.inputs.icon_global_run import params
from dask.distributed import Client, progress
import dask
from tqdm import tqdm

if __name__ == "__main__":
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

    # Configure Dask for parallel processing
    # Use processes (not threads) to avoid NetCDF file locking issues
    # Each worker gets 1 thread to avoid GIL contention
    import multiprocessing

    n_workers = min(multiprocessing.cpu_count() - 2, 20)  # Leave 2 cores for system
    print(f"Initializing Dask with {n_workers} workers...")

    client = Client(
        threads_per_worker=1,
        n_workers=n_workers,
        processes=True,
        memory_limit="4GB",  # Per worker
    )
    print(f"Dask dashboard available at: {client.dashboard_link}")

    print(f"Total cells to process: {n_cells}")

    chunk_sz = 10
    chunk_start = 20400

    # Progress tracking
    total_chunks = (n_cells - chunk_start + chunk_sz - 1) // chunk_sz
    print(
        f"\nProcessing {n_cells - chunk_start} cells in {total_chunks} chunks of {chunk_sz}..."
    )

    for chunk_idx, chunk in enumerate(
        tqdm(range(chunk_start, n_cells, chunk_sz), desc="Processing chunks")
    ):
        # Writer object for this chunk
        sfx = "_" + str(chunk + chunk_sz)
        writer = io.nc_writer(params, sfx)

        pw_run = parallel_wrapper(grid, params, reader, writer)

        lazy_results = []

        if chunk + chunk_sz > n_cells:
            chunk_end = n_cells
        else:
            chunk_end = chunk + chunk_sz

        for c_idx in range(chunk, chunk_end):
            lazy_result = dask.delayed(pw_run)(c_idx)
            lazy_results.append(lazy_result)

        results = dask.compute(*lazy_results)

        for item in results:
            writer.duplicate(item.c_idx, item)

    # Cleanup: close all cached NetCDF files and shut down Dask client
    print("\nCleaning up...")
    if hasattr(reader, "close_cached_files"):
        reader.close_cached_files()
        print("✓ Closed cached topography files")

    client.close()
    print("✓ Shut down Dask client")
    print("Processing complete!")
