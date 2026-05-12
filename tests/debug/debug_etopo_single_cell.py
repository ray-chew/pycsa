"""
Debug test for individual cells with verbose plotting and diagnostics.

Usage:
    # Edit CELL_INDICES list below, then run:
    pytest tests/test_single_cell_debug.py -v -s

This will create detailed plots and logs for debugging specific cell failures.
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import sys

from pycsa.core import io, var, utils
from pycsa.wrappers import interface

# =============================================================================
# CONFIGURE WHICH CELLS TO DEBUG HERE
# =============================================================================
CELL_INDICES = [
    1086,  # FileNotFoundError: E180 tile (N90E180)
    # 1027,   # FileNotFoundError: E180 tile (N90E180)
    # 1219,   # FileNotFoundError: E180 tile (N75E180)
]
# =============================================================================


@pytest.fixture(params=CELL_INDICES, ids=lambda x: f"cell_{x}")
def cell_idx(request):
    """Get cell index from parameter list."""
    return request.param


@pytest.fixture
def output_dir(cell_idx):
    """Create output directory for this specific cell."""
    base_dir = Path(__file__).parent.parent / "outputs" / "cell_debug"
    base_dir.mkdir(parents=True, exist_ok=True)

    cell_dir = base_dir / f"cell_{cell_idx}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Debug output directory: {cell_dir}")
    return cell_dir


@pytest.fixture
def test_params():
    """Create test parameters using ETOPO data."""
    params = var.params()

    # Import local paths
    try:
        from pycsa import local_paths

        utils.transfer_attributes(params, local_paths.paths, prefix="path")
    except ImportError as e:
        pytest.skip(f"Could not import local_paths: {e}")

    # Verify ETOPO path exists
    if not hasattr(params, "path_etopo") or not Path(params.path_etopo).exists():
        pytest.skip(f"ETOPO data path not found")

    # Test region: Alaska (will be overridden per cell)
    params.lat_extent = [48.0, 64.0, 64.0]
    params.lon_extent = [-148.0, -148.0, -112.0]

    # ETOPO coarse-graining factor
    params.etopo_cg = 50

    # CSA parameters
    params.nhi = 24
    params.nhj = 48
    params.n_modes = 50
    params.padding = 10

    params.U, params.V = 10.0, 0.0
    params.rect = True

    # Enable verbose mode
    params.plot = False
    params.plot_output = False
    params.debug = False
    params.dfft_first_guess = False
    params.refine = False
    params.verbose = True

    return params


@pytest.fixture
def test_grid(test_params):
    """Load ICON grid."""
    grid = var.grid()

    try:
        reader = io.ncdata()
        reader.read_dat(test_params.path_icon_grid, grid)
    except Exception as e:
        pytest.skip(f"Could not load ICON grid: {e}")

    # Convert to degrees
    grid.apply_f(utils.rad2deg)

    return grid


def test_debug_cell(cell_idx, output_dir, test_params, test_grid):
    """Debug a single cell with verbose output and plotting."""

    print(f"\n{'='*70}")
    print(f"DEBUGGING CELL {cell_idx}")
    print(f"{'='*70}\n")

    # Create log file
    log_file = output_dir / "debug_log.txt"

    def log_and_print(msg):
        """Print and log message."""
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log_and_print(f"Cell Index: {cell_idx}")
    log_and_print(f"Output Directory: {output_dir}")
    log_and_print("")

    # Step 1: Get cell geometry
    log_and_print("=" * 70)
    log_and_print("STEP 1: Cell Geometry")
    log_and_print("=" * 70)

    try:
        lat_verts = test_grid.clat_vertices[cell_idx]
        lon_verts = test_grid.clon_vertices[cell_idx]
        cell_lat = test_grid.clat[cell_idx]
        cell_lon = test_grid.clon[cell_idx]

        log_and_print(f"Cell center: lat={cell_lat:.4f}°, lon={cell_lon:.4f}°")
        log_and_print(f"Vertices (lat): {lat_verts}")
        log_and_print(f"Vertices (lon): {lon_verts}")
        log_and_print("")

    except Exception as e:
        log_and_print(f"ERROR getting cell geometry: {e}")
        log_and_print(traceback.format_exc())
        raise

    # Step 2: Handle lat/lon expansion
    log_and_print("=" * 70)
    log_and_print("STEP 2: Lat/Lon Expansion")
    log_and_print("=" * 70)

    try:
        lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)
        lat_verts_expanded, lon_verts_expanded = utils.handle_latlon_expansion(
            lat_verts, lon_verts, lat_expand=0.0, lon_expand=0.0
        )

        log_and_print(f"Original vertices:")
        log_and_print(f"  lat: {lat_verts}")
        log_and_print(f"  lon: {lon_verts}")
        log_and_print(f"")
        log_and_print(f"Expanded extents:")
        log_and_print(f"  lat_extent: {lat_extent}")
        log_and_print(f"  lon_extent: {lon_extent}")
        log_and_print(f"")
        log_and_print(f"Expanded vertices:")
        log_and_print(f"  lat: {lat_verts_expanded}")
        log_and_print(f"  lon: {lon_verts_expanded}")
        log_and_print("")

        # Update params
        test_params.lat_extent = lat_extent
        test_params.lon_extent = lon_extent

    except Exception as e:
        log_and_print(f"ERROR in lat/lon expansion: {e}")
        log_and_print(traceback.format_exc())
        raise

    # Step 3: Initialize ETOPO reader
    log_and_print("=" * 70)
    log_and_print("STEP 3: Initialize ETOPO Reader")
    log_and_print("=" * 70)

    try:
        reader = io.ncdata(padding=test_params.padding)
        topo = var.topo_cell()

        log_and_print(f"Creating ETOPO reader with:")
        log_and_print(f"  padding: {test_params.padding}")
        log_and_print(f"  lat_extent: {test_params.lat_extent}")
        log_and_print(f"  lon_extent: {test_params.lon_extent}")
        log_and_print(f"  etopo_cg: {test_params.etopo_cg}")
        log_and_print("")

        etopo_reader = reader.read_etopo_topo(
            None, test_params, is_parallel=True, verbose=True
        )

        log_and_print(f"ETOPO reader created successfully")
        log_and_print(f"  split_EW: {etopo_reader.split_EW}")
        if hasattr(etopo_reader, "split_NS"):
            log_and_print(f"  split_NS: {etopo_reader.split_NS}")
        if hasattr(etopo_reader, "file_cache"):
            log_and_print(f"  file_cache size: {len(etopo_reader.file_cache)}")
        log_and_print("")

    except Exception as e:
        log_and_print(f"ERROR initializing ETOPO reader: {e}")
        log_and_print(traceback.format_exc())
        raise

    # Step 4: Load topography data
    log_and_print("=" * 70)
    log_and_print("STEP 4: Load Topography Data")
    log_and_print("=" * 70)

    try:
        log_and_print("Calling etopo_reader.get_topo()...")
        etopo_reader.get_topo(topo)

        log_and_print(f"Topography loaded successfully!")
        log_and_print(f"  Shape: {topo.topo.shape}")
        log_and_print(f"  Min elevation: {np.min(topo.topo):.2f} m")
        log_and_print(f"  Max elevation: {np.max(topo.topo):.2f} m")
        log_and_print(f"  Mean elevation: {np.mean(topo.topo):.2f} m")
        log_and_print(f"  Lat shape: {topo.lat.shape}")
        log_and_print(f"  Lon shape: {topo.lon.shape}")
        log_and_print(f"  Lat range: [{np.min(topo.lat):.4f}, {np.max(topo.lat):.4f}]")
        log_and_print(f"  Lon range: [{np.min(topo.lon):.4f}, {np.max(topo.lon):.4f}]")
        log_and_print("")

        # Apply elevation floor
        below_floor = np.sum(topo.topo < -500.0)
        if below_floor > 0:
            log_and_print(f"Applying elevation floor: {below_floor} points below -500m")
            topo.topo[np.where(topo.topo < -500.0)] = -500.0

        topo.gen_mgrids()
        log_and_print("Generated mesh grids")
        log_and_print("")

        # Save topography data for inspection
        np.save(output_dir / "topo_elevation.npy", topo.topo)
        np.save(output_dir / "topo_lat.npy", topo.lat)
        np.save(output_dir / "topo_lon.npy", topo.lon)
        log_and_print(f"Saved topography arrays to {output_dir}")
        log_and_print("")

    except Exception as e:
        log_and_print(f"ERROR loading topography: {e}")
        log_and_print(traceback.format_exc())

        # Try to get more debug info from the reader
        if hasattr(etopo_reader, "__get_fns"):
            try:
                log_and_print("\nAttempting to get file info...")
                # This might fail but could give us useful info
                lat_idx_rng = getattr(etopo_reader, "lat_idx_rng", None)
                lon_idx_rng = getattr(etopo_reader, "lon_idx_rng", None)
                log_and_print(f"  lat_idx_rng: {lat_idx_rng}")
                log_and_print(f"  lon_idx_rng: {lon_idx_rng}")
            except:
                pass

        raise

    # Step 5: Set up cell geometry for land check
    log_and_print("=" * 70)
    log_and_print("STEP 5: Cell Geometry Setup")
    log_and_print("=" * 70)

    try:
        clon = np.array([test_grid.clon[cell_idx]])
        clat = np.array([test_grid.clat[cell_idx]])
        clon_vertices = np.array([lon_verts_expanded])
        clat_vertices = np.array([lat_verts_expanded])

        log_and_print(f"Cell geometry:")
        log_and_print(f"  clon: {clon}")
        log_and_print(f"  clat: {clat}")
        log_and_print(f"  clon_vertices: {clon_vertices}")
        log_and_print(f"  clat_vertices: {clat_vertices}")
        log_and_print("")

        ncells = 1
        nv = clon_vertices[0].size

        # Handle dateline crossing
        if etopo_reader.split_EW:
            log_and_print("Handling dateline crossing (split_EW=True)")
            orig_clon_vertices = clon_vertices.copy()
            clon_vertices[clon_vertices < 0.0] += 360.0
            log_and_print(f"  Before: {orig_clon_vertices}")
            log_and_print(f"  After: {clon_vertices}")
            log_and_print("")

        triangles = np.zeros((ncells, nv, 2))
        for i in range(0, ncells, 1):
            triangles[i, :, 0] = np.array(clon_vertices[i, :])
            triangles[i, :, 1] = np.array(clat_vertices[i, :])

        log_and_print(f"Triangle vertices:")
        log_and_print(f"  {triangles}")
        log_and_print("")

    except Exception as e:
        log_and_print(f"ERROR setting up cell geometry: {e}")
        log_and_print(traceback.format_exc())
        raise

    # Step 6: Check if land
    log_and_print("=" * 70)
    log_and_print("STEP 6: Land/Ocean Check")
    log_and_print("=" * 70)

    try:
        tri_idx = 0
        cell = var.topo_cell()
        tri = var.obj()

        tri.tri_lon_verts = triangles[:, :, 0]
        tri.tri_lat_verts = triangles[:, :, 1]
        simplex_lat = tri.tri_lat_verts[tri_idx]
        simplex_lon = tri.tri_lon_verts[tri_idx]

        log_and_print(f"Simplex vertices for land check:")
        log_and_print(f"  simplex_lat: {simplex_lat}")
        log_and_print(f"  simplex_lon: {simplex_lon}")
        log_and_print("")

        # This is where the error happens in some cells
        log_and_print("Calling utils.is_land()...")
        is_land = utils.is_land(cell, simplex_lat, simplex_lon, topo)

        log_and_print(f"is_land result: {is_land}")
        log_and_print(
            f"Cell lat shape: {cell.lat.shape if hasattr(cell, 'lat') and cell.lat is not None else 'None'}"
        )
        log_and_print(
            f"Cell lon shape: {cell.lon.shape if hasattr(cell, 'lon') and cell.lon is not None else 'None'}"
        )
        log_and_print("")

        if not is_land:
            log_and_print("Cell is OCEAN - skipping CSA processing")
            # Still plot the topography
            plot_topography(
                output_dir, topo, simplex_lat, simplex_lon, cell_idx, is_land=False
            )
            return

        log_and_print("Cell is LAND - proceeding with CSA")

        # Save cell data for inspection
        if hasattr(cell, "lat") and cell.lat is not None:
            np.save(output_dir / "cell_lat.npy", cell.lat)
            np.save(output_dir / "cell_lon.npy", cell.lon)
            if hasattr(cell, "topo") and cell.topo is not None:
                np.save(output_dir / "cell_topo.npy", cell.topo)
            log_and_print(f"Saved cell arrays to {output_dir}")
        log_and_print("")

    except Exception as e:
        log_and_print(f"ERROR in land check: {e}")
        log_and_print(traceback.format_exc())

        # Try to plot what we have so far
        try:
            plot_topography(
                output_dir,
                topo,
                simplex_lat,
                simplex_lon,
                cell_idx,
                is_land=None,
                error=str(e),
            )
        except:
            pass

        raise

    # Step 7: Get lat/lon segments
    log_and_print("=" * 70)
    log_and_print("STEP 7: Get Lat/Lon Segments")
    log_and_print("=" * 70)

    try:
        log_and_print(f"Calling utils.get_lat_lon_segments()...")
        log_and_print(f"  simplex_lat: {simplex_lat}")
        log_and_print(f"  simplex_lon: {simplex_lon}")
        log_and_print(f"  rect: {test_params.rect}")
        log_and_print("")

        utils.get_lat_lon_segments(
            simplex_lat, simplex_lon, cell, topo, rect=test_params.rect
        )

        log_and_print(f"Segments extracted successfully!")
        log_and_print(f"  cell.lat shape: {cell.lat.shape}")
        log_and_print(f"  cell.lon shape: {cell.lon.shape}")
        log_and_print(f"  cell.topo shape: {cell.topo.shape}")
        log_and_print("")

    except Exception as e:
        log_and_print(f"ERROR getting lat/lon segments: {e}")
        log_and_print(traceback.format_exc())
        raise

    # Step 8: Run spectral approximation
    log_and_print("=" * 70)
    log_and_print("STEP 8: Spectral Approximation")
    log_and_print("=" * 70)

    try:
        nhi = test_params.nhi
        nhj = test_params.nhj

        log_and_print(f"Running CSA with:")
        log_and_print(f"  nhi: {nhi}")
        log_and_print(f"  nhj: {nhj}")
        log_and_print(f"  U, V: {test_params.U}, {test_params.V}")
        log_and_print(f"  n_modes: {test_params.n_modes}")
        log_and_print("")

        pmf = interface.get_pmf(nhi, nhj, test_params.U, test_params.V)
        ampls, uw_pmf, dat_2D = pmf.sappx(cell, lmbda=0.1)

        # Filter out NaNs from spectrum
        ampls_valid = ampls[~np.isnan(ampls)]

        log_and_print(f"CSA complete!")
        log_and_print(f"  ampls shape: {ampls.shape}")
        log_and_print(f"  ampls total elements: {ampls.size}")
        log_and_print(f"  ampls valid (non-NaN): {len(ampls_valid)}")
        if len(ampls_valid) > 0:
            log_and_print(f"  ampls max (valid): {np.max(ampls_valid):.6e}")
            log_and_print(f"  ampls sum (valid): {np.sum(ampls_valid):.6e}")
        else:
            log_and_print(f"  ampls max: No valid values (all NaN)")
        log_and_print("")

        # Save spectrum
        np.save(output_dir / "spectrum.npy", ampls)
        log_and_print(f"Saved spectrum to {output_dir}/spectrum.npy")
        log_and_print("")

    except Exception as e:
        log_and_print(f"ERROR in spectral approximation: {e}")
        log_and_print(traceback.format_exc())
        raise

    # Step 9: Generate plots
    log_and_print("=" * 70)
    log_and_print("STEP 9: Generate Diagnostic Plots")
    log_and_print("=" * 70)

    try:
        plot_topography(
            output_dir,
            topo,
            simplex_lat,
            simplex_lon,
            cell_idx,
            is_land=True,
            cell=cell,
            ampls=ampls,
        )
        log_and_print("✓ Generated diagnostic plots")
    except Exception as e:
        log_and_print(f"ERROR generating plots: {e}")
        log_and_print(traceback.format_exc())

    log_and_print("")
    log_and_print("=" * 70)
    log_and_print(f"DEBUG COMPLETE FOR CELL {cell_idx}")
    log_and_print("=" * 70)
    log_and_print(f"All outputs saved to: {output_dir}")
    log_and_print("")

    print(f"\n✓ Debug complete! Check {output_dir} for detailed outputs")


def plot_topography(
    output_dir,
    topo,
    simplex_lat,
    simplex_lon,
    cell_idx,
    is_land=None,
    cell=None,
    ampls=None,
    error=None,
):
    """Generate comprehensive topography plots."""

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Full topography with cell outline
    ax1 = plt.subplot(2, 3, 1)
    if topo.topo is not None and topo.topo.size > 0:
        im1 = ax1.contourf(topo.lon, topo.lat, topo.topo, levels=50, cmap="terrain")
        plt.colorbar(im1, ax=ax1, label="Elevation (m)")

        # Overlay cell polygon
        if simplex_lat is not None and simplex_lon is not None and len(simplex_lat) > 0:
            # Close the polygon
            poly_lat = np.append(simplex_lat, simplex_lat[0])
            poly_lon = np.append(simplex_lon, simplex_lon[0])
            ax1.plot(poly_lon, poly_lat, "r-", linewidth=2, label="Cell boundary")
            ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No topography data", ha="center", va="center")

    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_title(f"Cell {cell_idx}: Full Topography")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Topography 3D view
    ax2 = plt.subplot(2, 3, 2, projection="3d")
    if topo.topo is not None and topo.topo.size > 0:
        # Downsample for 3D plotting if too large
        stride = max(1, topo.topo.shape[0] // 50)
        X, Y = np.meshgrid(topo.lon[::stride], topo.lat[::stride])
        Z = topo.topo[::stride, ::stride]
        ax2.plot_surface(X, Y, Z, cmap="terrain", alpha=0.8)
        ax2.set_xlabel("Longitude (°)")
        ax2.set_ylabel("Latitude (°)")
        ax2.set_zlabel("Elevation (m)")
    else:
        ax2.text2D(
            0.5,
            0.5,
            "No topography data",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
    ax2.set_title("3D View")

    # Plot 3: Elevation histogram
    ax3 = plt.subplot(2, 3, 3)
    if topo.topo is not None and topo.topo.size > 0:
        ax3.hist(topo.topo.flatten(), bins=50, edgecolor="black", alpha=0.7)
        ax3.axvline(0, color="blue", linestyle="--", linewidth=2, label="Sea level")
        ax3.axvline(
            -500, color="red", linestyle="--", linewidth=2, label="Floor (-500m)"
        )
        ax3.set_xlabel("Elevation (m)")
        ax3.set_ylabel("Count")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No topography data", ha="center", va="center")
    ax3.set_title("Elevation Distribution")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cell topography (if extracted)
    ax4 = plt.subplot(2, 3, 4)
    if (
        cell is not None
        and hasattr(cell, "topo")
        and cell.topo is not None
        and cell.topo.size > 0
    ):
        im4 = ax4.contourf(cell.lon, cell.lat, cell.topo, levels=50, cmap="terrain")
        plt.colorbar(im4, ax=ax4, label="Elevation (m)")
        ax4.set_xlabel("Longitude (°)")
        ax4.set_ylabel("Latitude (°)")
        ax4.set_title("Extracted Cell Topography")
        ax4.grid(True, alpha=0.3)
    else:
        status = "OCEAN" if is_land == False else "ERROR" if error else "No cell data"
        ax4.text(
            0.5, 0.5, status, ha="center", va="center", fontsize=14, fontweight="bold"
        )
        if error:
            ax4.text(
                0.5,
                0.3,
                f"Error: {error[:50]}...",
                ha="center",
                va="center",
                fontsize=8,
                color="red",
            )
    ax4.set_title("Cell Data")

    # Plot 5: Spectrum (if available)
    ax5 = plt.subplot(2, 3, 5)
    if ampls is not None and ampls.size > 0:
        # Plot non-NaN values
        ampls_valid = ampls[~np.isnan(ampls)]
        if len(ampls_valid) > 0:
            # Find indices of valid values for proper x-axis
            valid_indices = np.where(~np.isnan(ampls.flatten()))[0]
            ax5.semilogy(valid_indices, ampls_valid, "o-", markersize=4)
            ax5.set_xlabel("Mode index")
            ax5.set_ylabel("Amplitude")
            ax5.set_title(
                f"Spectral Amplitudes ({len(ampls_valid)}/{ampls.size} valid)"
            )
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(
                0.5,
                0.5,
                "No valid spectrum values\n(all NaN)",
                ha="center",
                va="center",
                fontsize=10,
            )
    else:
        ax5.text(0.5, 0.5, "No spectrum computed", ha="center", va="center")

    # Plot 6: Summary info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    info_lines = [
        f"Cell Index: {cell_idx}",
        f"",
        f"Topography Grid:",
        f"  Shape: {topo.topo.shape if topo.topo is not None else 'None'}",
        (
            f"  Lat: [{np.min(topo.lat):.4f}, {np.max(topo.lat):.4f}]°"
            if topo.lat is not None
            else "  Lat: None"
        ),
        (
            f"  Lon: [{np.min(topo.lon):.4f}, {np.max(topo.lon):.4f}]°"
            if topo.lon is not None
            else "  Lon: None"
        ),
        f"",
        f"Elevation:",
        f"  Min: {np.min(topo.topo):.1f} m" if topo.topo is not None else "  Min: None",
        f"  Max: {np.max(topo.topo):.1f} m" if topo.topo is not None else "  Max: None",
        (
            f"  Mean: {np.mean(topo.topo):.1f} m"
            if topo.topo is not None
            else "  Mean: None"
        ),
        f"",
        f"Land Classification: {is_land if is_land is not None else 'Unknown'}",
    ]

    if cell is not None and hasattr(cell, "topo") and cell.topo is not None:
        info_lines.extend(
            [
                f"",
                f"Cell Data:",
                f"  Shape: {cell.topo.shape}",
                f"  Points: {cell.topo.size}",
            ]
        )

    if ampls is not None:
        ampls_valid = ampls[~np.isnan(ampls)]
        info_lines.extend(
            [
                f"",
                f"Spectrum:",
                f"  Total modes: {ampls.size}",
                f"  Valid modes: {len(ampls_valid)}",
            ]
        )
        if len(ampls_valid) > 0:
            info_lines.append(f"  Max: {np.max(ampls_valid):.6e}")
        else:
            info_lines.append(f"  Max: N/A (all NaN)")

    if error:
        info_lines.extend(
            [
                f"",
                f"ERROR:",
                f"  {error[:60]}",
            ]
        )

    info_text = "\n".join(info_lines)
    ax6.text(
        0.1,
        0.9,
        info_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )

    plt.suptitle(f"Cell {cell_idx} Debug Plots", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"cell_{cell_idx}_debug.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved plot: {output_dir / f'cell_{cell_idx}_debug.png'}")


if __name__ == "__main__":
    # Run directly
    print(f"Testing cells: {CELL_INDICES}")
    pytest.main([__file__, "-v", "-s"])
