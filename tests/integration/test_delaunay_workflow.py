"""
Integration test for Delaunay decomposition workflow (FIXED).

Tests the full pipeline using the correct first_appx/second_appx API.
"""

import pytest
import numpy as np
from pathlib import Path
from pycsa.core import io, var, utils, delaunay
from pycsa.wrappers import interface, diagnostics


@pytest.mark.integration
class TestDelaunayWorkflow:
    """Test Delaunay decomposition and triangle pair processing."""

    @pytest.fixture
    def data_dir(self):
        """Return path to test data directory."""
        return Path(__file__).parent.parent.parent / "data"

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
            taper_fa = True
            taper_sa = True
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
    def test_data(self, data_dir):
        """Load test data (grid and topography)."""
        grid_path = data_dir / "icon_compact_alaska.nc"
        topo_path = data_dir / "topo_compact_alaska.nc"

        if not grid_path.exists() or not topo_path.exists():
            pytest.skip("Test data not available")

        # Initialize data objects
        grid = var.grid()
        topo = var.topo_cell()

        # Read data
        reader = io.ncdata(padding=10, padding_tol=50)
        reader.read_dat(str(grid_path), grid)
        grid.apply_f(utils.rad2deg)

        reader.read_dat(str(topo_path), topo)

        # Define Alaska region
        lat_verts = np.array([60.0, 64.0])
        lon_verts = np.array([-148.0, -140.0])

        # Extract topography for region
        reader.read_topo(topo, topo, lon_verts, lat_verts)

        # Clean up unrealistic values
        topo.topo[np.where(topo.topo < -500.0)] = -500.0

        topo.gen_mgrids()

        return grid, topo, reader

    def test_delaunay_decomposition(self, test_data):
        """Test Delaunay triangulation of domain."""
        grid, topo, reader = test_data

        # Perform Delaunay decomposition with small grid for testing
        tri = delaunay.get_decomposition(topo, xnp=5, ynp=4, padding=reader.padding)

        # Verify triangulation structure
        assert hasattr(tri, "simplices"), "Triangulation missing simplices"
        assert hasattr(tri, "points"), "Triangulation missing points"
        assert tri.simplices is not None, "Simplices not computed"
        assert tri.points is not None, "Points not computed"

        # Check that we have triangles
        assert len(tri.simplices) > 0, "No triangles created"

        # Each triangle should have 3 vertices
        assert tri.simplices.shape[1] == 3, "Triangles should have 3 vertices"

        # Vertex indices should be valid
        assert tri.simplices.min() >= 0, "Invalid vertex index"
        assert tri.simplices.max() < len(tri.points), "Vertex index out of range"

        # Check triangle vertex coordinates
        assert hasattr(tri, "tri_lat_verts"), "Triangle lat vertices missing"
        assert hasattr(tri, "tri_lon_verts"), "Triangle lon vertices missing"
        assert len(tri.tri_lat_verts) == len(
            tri.simplices
        ), "Lat vertices count mismatch"
        assert len(tri.tri_lon_verts) == len(
            tri.simplices
        ), "Lon vertices count mismatch"

    # @pytest.mark.skip(reason="Requires complete params object - advanced test")
    def test_first_appx_interface(self, test_data, mock_params):
        """Test first approximation interface."""
        grid, topo, reader = test_data

        # Delaunay decomposition
        tri = delaunay.get_decomposition(topo, xnp=5, ynp=4, padding=reader.padding)

        rect_idx = 0
        nhi = 12
        nhj = 12

        # Get reference cell
        simplex_lat = tri.tri_lat_verts[rect_idx]
        simplex_lon = tri.tri_lon_verts[rect_idx]

        # Create first approximation object
        fa = interface.first_appx(nhi, nhj, mock_params, topo)

        # Run first approximation
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

        # Verify results
        assert cell_fa is not None, "Cell not returned"
        assert ampls_fa is not None, "Amplitudes not computed"
        assert uw_fa is not None, "PMF not computed"
        assert dat_2D_fa is not None, "Reconstruction not computed"
        assert ampls_fa.shape == (
            nhj,
            nhi,
        ), f"Unexpected amplitude shape: {ampls_fa.shape}"

    # @pytest.mark.skip(reason="Requires complete params object - advanced test")
    def test_second_appx_interface(self, test_data, mock_params):
        """Test second approximation interface."""
        grid, topo, reader = test_data

        # Delaunay decomposition
        tri = delaunay.get_decomposition(topo, xnp=5, ynp=4, padding=reader.padding)

        rect_idx = 0
        nhi = 12
        nhj = 12

        # First approximation
        simplex_lat = tri.tri_lat_verts[rect_idx]
        simplex_lon = tri.tri_lon_verts[rect_idx]

        fa = interface.first_appx(nhi, nhj, mock_params, topo)
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

        # Second approximation
        sa = interface.second_appx(nhi, nhj, mock_params, topo, tri)

        # Process first triangle
        idx = rect_idx
        sols = sa.do(idx, ampls_fa)

        cell, ampls_sa, uw_sa, dat_2D_sa = sols

        # Verify results
        assert cell is not None, "Cell not returned"
        assert ampls_sa is not None, "Second approx amplitudes not computed"
        assert uw_sa is not None, "PMF not computed"
        assert dat_2D_sa is not None, "Reconstruction not computed"

    # @pytest.mark.skip(reason="Requires complete params object - advanced test")
    def test_triangle_pair_workflow(self, test_data, mock_params):
        """Test complete triangle pair processing workflow."""
        grid, topo, reader = test_data

        # Delaunay decomposition
        tri = delaunay.get_decomposition(topo, xnp=5, ynp=4, padding=reader.padding)

        rect_idx = 0
        nhi = 12
        nhj = 12

        # First approximation
        simplex_lat = tri.tri_lat_verts[rect_idx]
        simplex_lon = tri.tri_lon_verts[rect_idx]

        fa = interface.first_appx(nhi, nhj, mock_params, topo)
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

        # Second approximation on both triangles
        sa = interface.second_appx(nhi, nhj, mock_params, topo, tri)

        triangle_pair = []
        for idx in [rect_idx, rect_idx + 1]:
            cell, ampls_sa, uw_sa, dat_2D_sa = sa.do(idx, ampls_fa)
            cell.uw = uw_sa
            triangle_pair.append(cell)

        # Verify triangle pair
        assert len(triangle_pair) == 2, "Triangle pair should contain 2 triangles"
        assert triangle_pair[0].topo is not None
        assert triangle_pair[1].topo is not None
        assert triangle_pair[0].analysis is not None
        assert triangle_pair[1].analysis is not None


@pytest.mark.integration
class TestDelaunayDiagnostics:
    """Test diagnostics for Delaunay workflow."""

    @pytest.fixture
    def mock_params(self):
        """Create mock params."""

        class MockParams:
            run_case = "TEST"
            rect_set = [0, 2]
            padding = 10

        return MockParams()

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

    @pytest.mark.skip(reason="Diagnostics API needs verification")
    def test_diagnostics_basic(self, mock_params):
        """Test basic diagnostics initialization."""

        # Create mock triangulation
        class MockTri:
            simplices = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        tri = MockTri()

        diag = diagnostics.delaunay_metrics(mock_params, tri, writer=None)

        # Just check it initializes without error
        assert diag is not None
        assert hasattr(diag, "rect_set")
