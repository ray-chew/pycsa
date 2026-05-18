"""
Integration test for Delaunay decomposition workflow.

Tests the full pipeline using the correct first_appx/second_appx API.
Uses Phase 0 extracted test data (ETOPO patches from data/test/).
Generates diagnostic plots in plots/tests/integration/.
"""

import pytest
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from pycsa.core import var, delaunay
from pycsa.wrappers import interface


PLOT_DIR = Path(__file__).parent.parent.parent / "plots" / "tests" / "integration"


def _save_fig(name):
    """Save figure to plot directory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {path}")


@pytest.mark.integration
class TestDelaunayWorkflow:
    """Test Delaunay decomposition and triangle pair processing."""

    @pytest.fixture
    def mock_params(self):
        """Create mock params object for interface classes."""
        class MockParams:
            U = 10.0
            V = 0.0
            n_modes = 20
            lmbda_fa = 1e-1
            lmbda_sa = 1e-6
            taper_ref = False
            taper_fa = False
            taper_sa = False
            dfft_first_guess = False
            rect = True
            no_corrections = True
            recompute_rhs = False
            run_case = "TEST"
            rect_set = [0, 2]
            padding = 10
            taper_art_it = 20
            fa_iter_solve = False
            sa_iter_solve = False
            cg_spsp = False

        return MockParams()

    @pytest.fixture
    def test_topo(self):
        """Load Himalayas ETOPO patch from Phase 0 test data."""
        data_dir = Path(__file__).parent.parent.parent / "data" / "test"
        etopo_path = data_dir / "etopo_himalayas.npz"

        if not etopo_path.exists():
            pytest.skip(f"Test data not found: {etopo_path}")

        data = np.load(etopo_path)
        topo = var.topo_cell()
        topo.lat = data["lat"]
        topo.lon = data["lon"]
        topo.topo = data["topo"]
        topo.topo[topo.topo < -500.0] = -500.0
        topo.gen_mgrids()

        return topo

    def test_delaunay_decomposition(self, test_topo):
        """Test Delaunay triangulation of domain."""
        tri = delaunay.get_decomposition(
            test_topo, xnp=5, ynp=4, padding=0
        )

        # Verify triangulation structure
        assert hasattr(tri, 'simplices'), "Triangulation missing simplices"
        assert hasattr(tri, 'points'), "Triangulation missing points"
        assert tri.simplices is not None, "Simplices not computed"
        assert tri.points is not None, "Points not computed"
        assert len(tri.simplices) > 0, "No triangles created"
        assert tri.simplices.shape[1] == 3, "Triangles should have 3 vertices"
        assert tri.simplices.min() >= 0, "Invalid vertex index"
        assert tri.simplices.max() < len(tri.points), "Vertex index out of range"
        assert hasattr(tri, 'tri_lat_verts'), "Triangle lat vertices missing"
        assert hasattr(tri, 'tri_lon_verts'), "Triangle lon vertices missing"
        assert len(tri.tri_lat_verts) == len(tri.simplices), "Lat vertices count mismatch"
        assert len(tri.tri_lon_verts) == len(tri.simplices), "Lon vertices count mismatch"

        # --- Plot: topo + Delaunay mesh overlay ---
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.contourf(test_topo.lon_grid, test_topo.lat_grid, test_topo.topo,
                         levels=20, cmap='terrain')
        plt.colorbar(im, ax=ax, label='Elevation (m)')

        # Draw triangles
        patches = []
        for i in range(len(tri.simplices)):
            verts = np.column_stack([tri.tri_lon_verts[i], tri.tri_lat_verts[i]])
            patches.append(Polygon(verts, closed=True))
        pc = PatchCollection(patches, edgecolor='k', facecolor='none',
                             linewidth=0.8, alpha=0.7)
        ax.add_collection(pc)

        # Mark centroids
        ax.scatter(tri.tri_clons, tri.tri_clats, c='red', s=15, zorder=5,
                   label=f'{len(tri.simplices)} triangles')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Delaunay Decomposition (5x4 grid, {len(tri.simplices)} triangles)\nHimalayas ETOPO patch')
        ax.legend()
        _save_fig('delaunay_mesh.png')

    def test_first_appx_interface(self, test_topo, mock_params):
        """Test first approximation interface."""
        tri = delaunay.get_decomposition(
            test_topo, xnp=5, ynp=4, padding=0
        )

        rect_idx = 0
        nhi = 12
        nhj = 12

        simplex_lat = tri.tri_lat_verts[rect_idx]
        simplex_lon = tri.tri_lon_verts[rect_idx]

        fa = interface.first_appx(nhi, nhj, mock_params, test_topo)
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

        assert cell_fa is not None, "Cell not returned"
        assert ampls_fa is not None, "Amplitudes not computed"
        assert uw_fa is not None, "PMF not computed"
        assert dat_2D_fa is not None, "Reconstruction not computed"
        assert ampls_fa.shape == (nhj, nhi), f"Unexpected amplitude shape: {ampls_fa.shape}"

        # --- Plot: 3-panel (topo, reconstruction, difference) ---
        cell_fa.gen_mgrids()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original topo (masked)
        masked_topo = np.where(cell_fa.mask, cell_fa.topo, np.nan)
        im0 = axes[0].contourf(cell_fa.lon_grid, cell_fa.lat_grid, masked_topo,
                               levels=20, cmap='terrain')
        plt.colorbar(im0, ax=axes[0], label='m')
        axes[0].set_title('Original Topography')

        # Reconstruction
        im1 = axes[1].contourf(cell_fa.lon_grid, cell_fa.lat_grid, dat_2D_fa,
                               levels=20, cmap='terrain')
        plt.colorbar(im1, ax=axes[1], label='m')
        axes[1].set_title(f'FA Reconstruction ({nhi}x{nhj} modes)')

        # Difference
        diff = masked_topo - dat_2D_fa
        vmax = np.nanpercentile(np.abs(diff), 98)
        im2 = axes[2].contourf(cell_fa.lon_grid, cell_fa.lat_grid, diff,
                               levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im2, ax=axes[2], label='m')
        rmse = np.sqrt(np.nanmean(diff**2))
        axes[2].set_title(f'Difference (RMSE={rmse:.0f} m)')

        for ax in axes:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        fig.suptitle(f'First Approximation — Triangle {rect_idx}, UW={np.sum(uw_fa):.2e} Pa', y=1.02)
        _save_fig('first_appx_reconstruction.png')

        # --- Plot: amplitude spectrum ---
        fig, ax = plt.subplots(figsize=(7, 6))
        ampls_clean = np.nan_to_num(ampls_fa, nan=0.0)
        im = ax.imshow(np.abs(ampls_clean), cmap='hot_r', aspect='auto',
                       origin='lower')
        plt.colorbar(im, ax=ax, label='|Amplitude| (m)')
        ax.set_xlabel('k index')
        ax.set_ylabel('l index')
        ax.set_title(f'FA Amplitude Spectrum ({nhi}x{nhj})')
        _save_fig('first_appx_spectrum.png')

    def test_second_appx_interface(self, test_topo, mock_params):
        """Test second approximation interface."""
        tri = delaunay.get_decomposition(
            test_topo, xnp=5, ynp=4, padding=0
        )

        rect_idx = 0
        nhi = 12
        nhj = 12

        simplex_lat = tri.tri_lat_verts[rect_idx]
        simplex_lon = tri.tri_lon_verts[rect_idx]

        fa = interface.first_appx(nhi, nhj, mock_params, test_topo)
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

        sa = interface.second_appx(nhi, nhj, mock_params, test_topo, tri)
        cell_sa, ampls_sa, uw_sa, dat_2D_sa = sa.do(rect_idx, ampls_fa)

        assert cell_sa is not None, "Cell not returned"
        assert ampls_sa is not None, "Second approx amplitudes not computed"
        assert uw_sa is not None, "PMF not computed"
        assert dat_2D_sa is not None, "Reconstruction not computed"

        # --- Plot: FA vs SA comparison (4-panel) ---
        cell_fa.gen_mgrids()
        cell_sa.gen_mgrids()

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # FA reconstruction
        im0 = axes[0, 0].contourf(cell_fa.lon_grid, cell_fa.lat_grid, dat_2D_fa,
                                  levels=20, cmap='terrain')
        plt.colorbar(im0, ax=axes[0, 0], label='m')
        axes[0, 0].set_title(f'FA Reconstruction (UW={np.sum(uw_fa):.2e})')

        # SA reconstruction
        im1 = axes[0, 1].contourf(cell_sa.lon_grid, cell_sa.lat_grid, dat_2D_sa,
                                  levels=20, cmap='terrain')
        plt.colorbar(im1, ax=axes[0, 1], label='m')
        axes[0, 1].set_title(f'SA Reconstruction (UW={np.sum(uw_sa):.2e})')

        # SA topo (masked original within triangle)
        masked_topo = np.where(cell_sa.mask, cell_sa.topo, np.nan)
        im2 = axes[1, 0].contourf(cell_sa.lon_grid, cell_sa.lat_grid, masked_topo,
                                  levels=20, cmap='terrain')
        plt.colorbar(im2, ax=axes[1, 0], label='m')
        axes[1, 0].set_title('SA Original (masked)')

        # SA difference
        diff = masked_topo - dat_2D_sa
        vmax = np.nanpercentile(np.abs(diff), 98)
        im3 = axes[1, 1].contourf(cell_sa.lon_grid, cell_sa.lat_grid, diff,
                                  levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im3, ax=axes[1, 1], label='m')
        rmse = np.sqrt(np.nanmean(diff**2))
        axes[1, 1].set_title(f'SA Difference (RMSE={rmse:.0f} m)')

        for ax in axes.flat:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        fig.suptitle(f'FA vs SA — Triangle {rect_idx}', fontsize=14)
        plt.tight_layout()
        _save_fig('second_appx_comparison.png')

        # --- Plot: amplitude comparison ---
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ampls_fa_clean = np.nan_to_num(ampls_fa, nan=0.0)
        ampls_sa_clean = np.nan_to_num(ampls_sa, nan=0.0)
        vmax = max(np.abs(ampls_fa_clean).max(), np.abs(ampls_sa_clean).max())

        im0 = axes[0].imshow(np.abs(ampls_fa_clean), cmap='hot_r', aspect='auto',
                             origin='lower', vmin=0, vmax=vmax)
        plt.colorbar(im0, ax=axes[0], label='|Amplitude| (m)')
        axes[0].set_title('FA Amplitudes')

        im1 = axes[1].imshow(np.abs(ampls_sa_clean), cmap='hot_r', aspect='auto',
                             origin='lower', vmin=0, vmax=vmax)
        plt.colorbar(im1, ax=axes[1], label='|Amplitude| (m)')
        axes[1].set_title('SA Amplitudes')

        for ax in axes:
            ax.set_xlabel('k index')
            ax.set_ylabel('l index')

        fig.suptitle('Amplitude Spectra: FA vs SA')
        plt.tight_layout()
        _save_fig('fa_vs_sa_spectrum.png')

    def test_triangle_pair_workflow(self, test_topo, mock_params):
        """Test complete triangle pair processing workflow."""
        tri = delaunay.get_decomposition(
            test_topo, xnp=5, ynp=4, padding=0
        )

        rect_idx = 0
        nhi = 12
        nhj = 12

        simplex_lat = tri.tri_lat_verts[rect_idx]
        simplex_lon = tri.tri_lon_verts[rect_idx]

        fa = interface.first_appx(nhi, nhj, mock_params, test_topo)
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

        sa = interface.second_appx(nhi, nhj, mock_params, test_topo, tri)

        triangle_pair = []
        recons = []
        for idx in [rect_idx, rect_idx + 1]:
            cell, ampls_sa, uw_sa, dat_2D_sa = sa.do(idx, ampls_fa)
            cell.uw = uw_sa
            triangle_pair.append(cell)
            recons.append(dat_2D_sa)

        assert len(triangle_pair) == 2, "Triangle pair should contain 2 triangles"
        assert triangle_pair[0].topo is not None
        assert triangle_pair[1].topo is not None
        assert triangle_pair[0].analysis is not None
        assert triangle_pair[1].analysis is not None

        # --- Plot: triangle pair side-by-side ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        for col, (cell, recon, idx) in enumerate(
                zip(triangle_pair, recons, [rect_idx, rect_idx + 1])):
            cell.gen_mgrids()
            masked_topo = np.where(cell.mask, cell.topo, np.nan)

            im0 = axes[0, col].contourf(cell.lon_grid, cell.lat_grid, masked_topo,
                                        levels=20, cmap='terrain')
            plt.colorbar(im0, ax=axes[0, col], label='m')
            axes[0, col].set_title(f'Triangle {idx}: Topo (masked)')

            im1 = axes[1, col].contourf(cell.lon_grid, cell.lat_grid, recon,
                                        levels=20, cmap='terrain')
            plt.colorbar(im1, ax=axes[1, col], label='m')
            diff = masked_topo - recon
            rmse = np.sqrt(np.nanmean(diff**2))
            axes[1, col].set_title(f'Triangle {idx}: SA Recon (RMSE={rmse:.0f} m, UW={np.sum(cell.uw):.2e})')

        for ax in axes.flat:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        fig.suptitle('Triangle Pair Workflow — Adjacent Triangles', fontsize=14)
        plt.tight_layout()
        _save_fig('triangle_pair_workflow.png')


@pytest.mark.integration
class TestDelaunayDiagnostics:
    """Test diagnostics for Delaunay workflow."""

    @pytest.fixture
    def mock_triangle_pair(self):
        """Create mock triangle pair for diagnostics testing."""
        cell1 = var.topo_cell()
        cell1.topo = np.random.randn(50, 50) * 100
        cell1.lat = np.linspace(60, 61, 50)
        cell1.lon = np.linspace(-150, -149, 50)
        cell1.mask = np.ones((50, 50), dtype=bool)
        cell1.uw = 1500.0

        analysis1 = var.analysis()
        analysis1.ampls = np.random.randn(12, 12) * 10
        analysis1.recon = np.random.randn(50, 50) * 80
        cell1.analysis = analysis1

        cell2 = var.topo_cell()
        cell2.topo = np.random.randn(50, 50) * 100
        cell2.lat = np.linspace(60, 61, 50)
        cell2.lon = np.linspace(-150, -149, 50)
        cell2.mask = np.ones((50, 50), dtype=bool)
        cell2.uw = 1200.0

        analysis2 = var.analysis()
        analysis2.ampls = np.random.randn(12, 12) * 10
        analysis2.recon = np.random.randn(50, 50) * 80
        cell2.analysis = analysis2

        return [cell1, cell2]

    def test_diagnostics_basic(self):
        """Test basic diagnostics initialization."""
        from pycsa.wrappers import diagnostics

        class MockParams:
            run_case = "TEST"
            rect_set = [0, 2]
            padding = 10

        class MockTri:
            simplices = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        diag = diagnostics.delaunay_metrics(MockParams(), MockTri(), writer=None)

        assert diag is not None
        assert diag.params.rect_set == [0, 2]
        assert diag.pmf_diff == []
        assert diag.writer is None
