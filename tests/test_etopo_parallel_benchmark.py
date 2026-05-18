"""
Comprehensive benchmark test for ETOPO data processing with Dask parallelization.

This test:
1. Uses ETOPO input data instead of MERIT
2. Processes 320 cells using 16+ cores
3. Verifies Dask is working correctly
4. Saves diagnostic outputs (topography plots, spectra)
"""

import pytest
import numpy as np
import time
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

from pycsa.core import io, var, utils
from pycsa.wrappers import interface, diagnostics
from pycsa.plotting import cart_plot

# Dask imports
from dask.distributed import Client, as_completed
import dask


class TestETOPOParallelBenchmark:
    """Benchmark test for parallel ETOPO processing."""

    @pytest.fixture(scope="class")
    def output_dir(self, tmp_path_factory):
        """Create output directory for test results."""
        # Use a permanent directory instead of tmp for inspection
        base_dir = Path(__file__).parent.parent / "outputs" / "benchmark_etopo"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = base_dir / f"run_{timestamp}"
        test_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Output directory: {test_dir}")
        return test_dir

    @pytest.fixture(scope="class")
    def test_params(self):
        """Create test parameters using ETOPO data."""
        params = var.params()

        # Import local paths
        try:
            from pycsa import local_paths
            utils.transfer_attributes(params, local_paths.paths, prefix="path")
        except ImportError as e:
            print(f"ERROR: Could not import local_paths: {e}")
            raise

        # Verify ETOPO path exists
        if not hasattr(params, 'path_etopo') or not Path(params.path_etopo).exists():
            pytest.skip(f"ETOPO data path not found: {params.path_etopo if hasattr(params, 'path_etopo') else 'not set'}")

        # Test region: Alaska (good for testing, has varied topography)
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

        # Disable plotting during cell processing (we'll plot diagnostics separately)
        params.plot = False
        params.plot_output = False

        params.debug = False
        params.dfft_first_guess = False
        params.refine = False
        params.verbose = False

        return params

    @pytest.fixture(scope="class")
    def test_grid(self, test_params):
        """Load a subset of ICON grid for testing."""
        grid = var.grid()

        # Read ICON grid
        try:
            reader = io.ncdata()
            reader.read_dat(test_params.path_icon_grid, grid)
        except Exception as e:
            pytest.skip(f"Could not load ICON grid: {e}")

        # Convert to degrees
        grid.apply_f(utils.rad2deg)

        return grid

    def test_dask_initialization(self, output_dir):
        """Test 1: Verify Dask initializes correctly with 16+ cores."""
        import multiprocessing

        n_workers = min(multiprocessing.cpu_count() - 2, 20)
        if n_workers < 16:
            pytest.skip(f"Not enough cores for benchmark: {n_workers} (need 16+)")

        print(f"\n🚀 Initializing Dask with {n_workers} workers...")

        client = Client(
            threads_per_worker=1,
            n_workers=n_workers,
            processes=True,
            memory_limit='4GB'
        )

        # Verify client is running
        assert client.status == 'running', "Dask client not running!"

        # Verify workers
        workers = client.scheduler_info()['workers']
        if len(workers) < 16:
            client.close()
            pytest.skip(f"Only {len(workers)} workers started (need 16+) — resource limits on this node")

        print(f"✓ Dask running with {len(workers)} workers")
        print(f"✓ Dashboard: {client.dashboard_link}")

        # Save Dask info to output
        with open(output_dir / "dask_info.txt", "w") as f:
            f.write(f"Dask Benchmark Test\n")
            f.write(f"===================\n\n")
            f.write(f"Workers: {len(workers)}\n")
            f.write(f"Threads per worker: 1\n")
            f.write(f"Memory limit per worker: 4GB\n")
            f.write(f"Dashboard: {client.dashboard_link}\n")
            f.write(f"\nWorker details:\n")
            for worker_id, worker_info in workers.items():
                f.write(f"  {worker_id}: {worker_info['memory_limit'] / 1e9:.1f}GB\n")

        client.close()
        print("✓ Dask client closed cleanly")

    def test_etopo_file_caching(self, test_params, output_dir):
        """Test 2: Verify ETOPO file caching works correctly."""
        print("\n📦 Testing ETOPO file caching...")

        # Create a test cell
        test_cell = var.topo_cell()

        # Initialize ETOPO reader with caching
        reader = io.ncdata(padding=test_params.padding)
        etopo_reader = reader.read_etopo_topo(test_cell, test_params, verbose=True, is_parallel=True)

        # Verify cache exists
        assert hasattr(etopo_reader, 'file_cache'), "ETOPO reader missing file_cache attribute!"
        assert hasattr(etopo_reader, '_get_cached_file'), "ETOPO reader missing _get_cached_file method!"
        assert hasattr(etopo_reader, 'close_cached_files'), "ETOPO reader missing close_cached_files method!"

        # Load data (this should populate the cache)
        etopo_reader.get_topo(test_cell)

        # Verify data was loaded
        assert test_cell.topo is not None, "Topography not loaded!"
        assert test_cell.lon is not None, "Longitude not loaded!"
        assert test_cell.lat is not None, "Latitude not loaded!"

        # Verify cache was used
        cache_size = len(etopo_reader.file_cache)
        print(f"✓ File cache contains {cache_size} open files")
        assert cache_size > 0, "File cache is empty (caching not working!)"

        # Load same region again - should reuse cache
        test_cell2 = var.topo_cell()
        etopo_reader.get_topo(test_cell2)

        # Cache size should not have increased
        cache_size_after = len(etopo_reader.file_cache)
        assert cache_size_after == cache_size, f"Cache size increased ({cache_size} -> {cache_size_after}), files not being reused!"

        print(f"✓ File cache correctly reused (size unchanged: {cache_size})")

        # Clean up
        etopo_reader.close_cached_files()
        assert len(etopo_reader.file_cache) == 0, "Cache not cleared after close_cached_files()!"
        print("✓ Cache cleared successfully")

        # Save cache info
        with open(output_dir / "cache_info.txt", "w") as f:
            f.write("ETOPO File Caching Test\n")
            f.write("=======================\n\n")
            f.write(f"Cache size (unique files): {cache_size}\n")
            f.write(f"Cache reuse verified: Yes\n")
            f.write(f"Cache cleanup verified: Yes\n")

    def test_parallel_320_cells(self, test_params, test_grid, output_dir):
        """Test 3: Process 320 cells in parallel with full diagnostics."""
        print(f"\n🔬 Processing 320 cells in parallel...")

        n_test_cells = 320
        total_cells = test_grid.clat.size

        # Make sure we have enough cells
        if total_cells < n_test_cells:
            pytest.skip(f"Grid only has {total_cells} cells, need {n_test_cells}")

        # Select cells to process (spread across the grid)
        cell_indices = np.linspace(0, total_cells - 1, n_test_cells, dtype=int)

        # Initialize Dask
        import multiprocessing
        n_workers = min(multiprocessing.cpu_count() - 2, 20)
        print(f"  Starting Dask with {n_workers} workers...")

        client = Client(
            threads_per_worker=1,
            n_workers=n_workers,
            processes=True,
            memory_limit='4GB'
        )
        print(f"  Dashboard: {client.dashboard_link}")

        # Initialize reader with ETOPO
        reader = io.ncdata(padding=test_params.padding, padding_tol=(60 - test_params.padding))

        # Store pre-computation info
        clat_rad = np.copy(test_grid.clat)
        clon_rad = np.copy(test_grid.clon)

        # Scatter large objects to workers (avoid serialization overhead)
        print(f"\n  Scattering grid data to workers...")
        grid_future = client.scatter(test_grid, broadcast=True)
        params_future = client.scatter(test_params, broadcast=True)
        clat_rad_future = client.scatter(clat_rad, broadcast=True)
        clon_rad_future = client.scatter(clon_rad, broadcast=True)

        # Diagnostic storage
        processing_times = []
        cell_results = []
        error_cells = []

        # Progress tracking
        from tqdm import tqdm

        print(f"\n  Processing {n_test_cells} cells...")
        start_time = time.time()

        # Process cells
        futures = []
        for c_idx in cell_indices:
            future = client.submit(
                self._process_single_cell,
                c_idx, grid_future, params_future, reader, clat_rad_future, clon_rad_future
            )
            futures.append((c_idx, future))

        # Collect results with progress bar
        for c_idx, future in tqdm(futures, desc="Processing cells"):
            try:
                result = future.result(timeout=120)  # 2 min timeout per cell
                if result is not None:
                    cell_results.append(result)
                    if 'error' not in result:
                        processing_times.append(result['processing_time'])
                    else:
                        error_cells.append(result)
                        if len(error_cells) <= 3:  # Only print first 3 errors
                            print(f"\n  Cell {c_idx} error: {result['error']}")
            except Exception as e:
                print(f"\n  Warning: Cell {c_idx} timed out: {e}")
                error_cells.append({'c_idx': c_idx, 'error': f'Timeout: {e}'})

        total_time = time.time() - start_time

        # Close cached files
        if hasattr(reader, 'close_cached_files'):
            reader.close_cached_files()

        # Shut down Dask
        client.close()

        # Analysis
        n_total = len(cell_results)
        n_errors = len(error_cells)
        valid_results = [r for r in cell_results if 'error' not in r]
        n_successful = len(valid_results)
        n_land = sum(1 for r in valid_results if r.get('is_land', False))
        n_ocean = sum(1 for r in valid_results if r.get('is_land') == False)
        success_rate = 100 * n_successful / n_test_cells

        # Separate land and ocean processing times
        land_times = [r['processing_time'] for r in valid_results if r.get('is_land') == True]
        ocean_times = [r['processing_time'] for r in valid_results if r.get('is_land') == False]

        print(f"\n📊 Results:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Cells processed: {n_successful}/{n_test_cells} ({success_rate:.1f}%)")
        if n_successful > 0:
            print(f"    - Land cells: {n_land} ({100*n_land/n_successful:.0f}%)")
            print(f"    - Ocean cells: {n_ocean} ({100*n_ocean/n_successful:.0f}%) [skipped CSA]")
        print(f"  Errors/failures: {n_errors}")

        if land_times:
            print(f"\n  Land cell timing (CSA processed):")
            print(f"    Avg: {np.mean(land_times):.2f}s")
            print(f"    Min: {np.min(land_times):.2f}s")
            print(f"    Max: {np.max(land_times):.2f}s")

        if ocean_times:
            print(f"\n  Ocean cell timing (skipped):")
            print(f"    Avg: {np.mean(ocean_times):.3f}s")

        if processing_times:
            print(f"\n  Overall throughput: {n_successful / total_time:.1f} cells/sec")
            if land_times:
                print(f"  Land-only throughput: {n_land / sum(land_times):.1f} cells/sec")

        # Assertions (relaxed for initial benchmarking)
        # Note: Success rate depends on grid coverage of test region
        assert success_rate >= 60, f"Success rate too low: {success_rate:.1f}% (expected ≥60%)"
        if processing_times:
            assert np.mean(processing_times) < 10, f"Average processing time too high: {np.mean(processing_times):.1f}s"

        # Print error summary if needed
        if n_errors > 0:
            print(f"\n⚠️  Warning: {n_errors} cells had errors. Check outputs/benchmark_etopo/*/errors.txt for details")

        # Save results
        self._save_benchmark_results(output_dir, valid_results, processing_times, total_time, n_test_cells, error_cells)

        # Generate diagnostic plots
        self._generate_diagnostic_plots(output_dir, cell_results, test_params)

        print(f"\n✓ Benchmark complete! Results saved to {output_dir}")

    @staticmethod
    def _process_single_cell(c_idx, grid, params, reader, clat_rad, clon_rad):
        """Process a single cell (executed by Dask worker)."""
        try:
            start_time = time.time()

            # Create cell object
            topo = var.topo_cell()

            # Get cell vertices
            lat_verts = grid.clat_vertices[c_idx]
            lon_verts = grid.clon_vertices[c_idx]

            # Handle lat/lon expansion
            lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)
            lat_verts, lon_verts = utils.handle_latlon_expansion(
                lat_verts, lon_verts, lat_expand=0.0, lon_expand=0.0
            )

            params.lat_extent = lat_extent
            params.lon_extent = lon_extent

            # Load ETOPO topography data
            etopo_reader = reader.read_etopo_topo(None, params, is_parallel=True)
            etopo_reader.get_topo(topo)

            # Apply elevation floor
            topo.topo[np.where(topo.topo < -500.0)] = -500.0
            topo.gen_mgrids()

            # Set up cell geometry
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

            # Check if land
            tri_idx = 0
            cell = var.topo_cell()
            tri = var.obj()

            tri.tri_lon_verts = triangles[:, :, 0]
            tri.tri_lat_verts = triangles[:, :, 1]
            simplex_lat = tri.tri_lat_verts[tri_idx]
            simplex_lon = tri.tri_lon_verts[tri_idx]

            is_land = utils.is_land(cell, simplex_lat, simplex_lon, topo)

            if not is_land:
                return {
                    'c_idx': c_idx,
                    'is_land': False,
                    'processing_time': time.time() - start_time
                }

            # Run CSA (simplified - just first approximation for benchmark)
            nhi = params.nhi
            nhj = params.nhj

            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=params.rect)

            # Run spectral approximation
            pmf = interface.get_pmf(nhi, nhj, params.U, params.V)
            ampls, uw_pmf, dat_2D = pmf.sappx(cell, lmbda=0.1)

            processing_time = time.time() - start_time

            # Filter out NaNs from spectrum for meaningful statistics
            ampls_valid = ampls[~np.isnan(ampls)]
            spectrum_max = float(np.max(ampls_valid)) if len(ampls_valid) > 0 else np.nan
            n_valid_modes = len(ampls_valid)

            return {
                'c_idx': c_idx,
                'is_land': True,
                'processing_time': processing_time,
                'topo_shape': topo.topo.shape,
                'topo_min': float(np.min(topo.topo)),
                'topo_max': float(np.max(topo.topo)),
                'spectrum_max': spectrum_max,
                'n_modes': ampls.size,
                'n_valid_modes': n_valid_modes,
                'lat_extent': params.lat_extent,
                'lon_extent': params.lon_extent,
            }

        except Exception as e:
            import traceback
            return {
                'c_idx': c_idx,
                'is_land': None,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _save_benchmark_results(self, output_dir, cell_results, processing_times, total_time, n_test_cells, error_cells):
        """Save benchmark results to file."""
        with open(output_dir / "benchmark_results.txt", "w") as f:
            f.write("ETOPO Parallel Processing Benchmark\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Test Configuration:\n")
            f.write(f"  Total cells attempted: {n_test_cells}\n")
            f.write(f"  Successful cells: {len(cell_results)}\n")
            f.write(f"  Error/failed cells: {len(error_cells)}\n")
            f.write(f"\n")

            f.write(f"Timing Results:\n")
            f.write(f"  Total time: {total_time:.2f}s\n")
            f.write(f"  Average per cell: {np.mean(processing_times):.2f}s\n")
            f.write(f"  Median per cell: {np.median(processing_times):.2f}s\n")
            f.write(f"  Min per cell: {np.min(processing_times):.2f}s\n")
            f.write(f"  Max per cell: {np.max(processing_times):.2f}s\n")
            f.write(f"  Throughput: {len(cell_results) / total_time:.2f} cells/sec\n")
            f.write(f"\n")

            # Land/ocean statistics
            land_cells = sum(1 for r in cell_results if r.get('is_land'))
            ocean_cells = sum(1 for r in cell_results if r.get('is_land') == False)
            f.write(f"Cell Statistics:\n")
            f.write(f"  Land cells: {land_cells}\n")
            f.write(f"  Ocean cells: {ocean_cells}\n")

            # Error summary
            if error_cells:
                f.write(f"\nErrors:\n")
                error_types = {}
                for err in error_cells:
                    err_msg = err.get('error', 'Unknown error')
                    # Group by error type (first line of error)
                    err_type = err_msg.split('\n')[0][:100]
                    error_types[err_type] = error_types.get(err_type, 0) + 1

                for err_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {count}x: {err_type}\n")

        # Save detailed error log
        if error_cells:
            with open(output_dir / "errors.txt", "w") as f:
                f.write(f"Detailed Error Log ({len(error_cells)} errors)\n")
                f.write("=" * 70 + "\n\n")
                for i, err in enumerate(error_cells[:10]):  # First 10 errors
                    f.write(f"Error {i+1}: Cell {err.get('c_idx', 'unknown')}\n")
                    f.write(f"{'-' * 70}\n")
                    f.write(f"{err.get('error', 'No error message')}\n")
                    if 'traceback' in err:
                        f.write(f"\nTraceback:\n{err['traceback']}\n")
                    f.write(f"\n{'=' * 70}\n\n")
                if len(error_cells) > 10:
                    f.write(f"\n... and {len(error_cells) - 10} more errors (see benchmark_results.txt for summary)\n")

        print(f"  ✓ Saved benchmark results")

    def _generate_diagnostic_plots(self, output_dir, cell_results, params):
        """Generate diagnostic plots from results."""
        print("\n  Generating diagnostic plots...")

        # Filter land cells only
        land_results = [r for r in cell_results if r['is_land']]

        if len(land_results) < 5:
            print("    Skipping plots (not enough land cells)")
            return

        # Plot 1: Processing time distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        times = [r['processing_time'] for r in cell_results]
        axes[0, 0].hist(times, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Processing Time (s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Processing Time Distribution')
        axes[0, 0].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.2f}s')
        axes[0, 0].legend()

        # Plot 2: Topography elevation ranges
        topo_mins = [r['topo_min'] for r in land_results]
        topo_maxs = [r['topo_max'] for r in land_results]
        axes[0, 1].scatter(topo_mins, topo_maxs, alpha=0.5)
        axes[0, 1].set_xlabel('Min Elevation (m)')
        axes[0, 1].set_ylabel('Max Elevation (m)')
        axes[0, 1].set_title('Topography Elevation Ranges')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Spectrum amplitudes
        spectrum_maxs = [r['spectrum_max'] for r in land_results if not np.isnan(r['spectrum_max'])]
        if len(spectrum_maxs) > 0:
            axes[1, 0].hist(spectrum_maxs, bins=30, edgecolor='black', alpha=0.7)
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid spectrum data', ha='center', va='center')
        axes[1, 0].set_xlabel('Max Spectrum Amplitude')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Spectral Amplitude Distribution')

        # Plot 4: Topography grid sizes
        topo_sizes = [r['topo_shape'][0] * r['topo_shape'][1] for r in land_results]
        axes[1, 1].hist(topo_sizes, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Grid Points')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Loaded Topography Grid Sizes')

        plt.tight_layout()
        plt.savefig(output_dir / 'diagnostics_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved diagnostics_summary.png")

        # Save a few example topography samples
        n_samples = min(6, len(land_results))
        sample_cells = np.random.choice(len(land_results), n_samples, replace=False)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, sample_idx in enumerate(sample_cells):
            result = land_results[sample_idx]
            ax = axes[idx]

            # Just show basic info since we don't have the actual topo data
            spectrum_str = f"{result['spectrum_max']:.2e}" if not np.isnan(result['spectrum_max']) else "N/A"
            n_valid = result.get('n_valid_modes', '?')
            n_total = result.get('n_modes', '?')

            info_text = (
                f"Cell {result['c_idx']}\n"
                f"Grid: {result['topo_shape']}\n"
                f"Elev: [{result['topo_min']:.0f}, {result['topo_max']:.0f}]m\n"
                f"Spectrum max: {spectrum_str}\n"
                f"Valid modes: {n_valid}/{n_total}\n"
                f"Time: {result['processing_time']:.2f}s"
            )
            ax.text(0.5, 0.5, info_text, ha='center', va='center',
                   fontsize=10, family='monospace')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        plt.suptitle('Sample Cell Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_cells.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved sample_cells.png")


if __name__ == "__main__":
    # Run the test directly
    pytest.main([__file__, "-v", "-s"])
