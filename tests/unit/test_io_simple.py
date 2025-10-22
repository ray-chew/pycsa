"""
Simplified unit tests for I/O routines.

Tests basic NetCDF reading functionality for topographic data.
"""

import pytest
import numpy as np
from pathlib import Path
from pycsa.core import io, var


class TestNetCDFReader:
    """Test NetCDF data reading functionality."""

    @pytest.fixture
    def data_dir(self):
        """Return path to test data directory."""
        return Path(__file__).parent.parent.parent / "data"

    def test_ncdata_initialization(self):
        """Test ncdata object initialization."""
        reader = io.ncdata(padding=10, padding_tol=50)
        assert reader.padding == 60
        assert reader.read_merit == False

    def test_read_grid_data(self, data_dir):
        """Test reading grid data from NetCDF file."""
        grid_path = data_dir / "icon_compact_alaska.nc"
        if not grid_path.exists():
            pytest.skip(f"Test data not found: {grid_path}")

        grid = var.grid()
        reader = io.ncdata()
        reader.read_dat(str(grid_path), grid)

        assert grid.clat is not None
        assert grid.clon is not None
        assert len(grid.clat) > 0

    def test_read_topography_data(self, data_dir):
        """Test reading topography data from NetCDF file."""
        topo_path = data_dir / "topo_compact_alaska.nc"
        if not topo_path.exists():
            pytest.skip(f"Test data not found: {topo_path}")

        topo = var.topo_cell()
        reader = io.ncdata()
        reader.read_dat(str(topo_path), topo)

        assert topo.lat is not None
        assert topo.lon is not None
        assert topo.topo is not None
        assert topo.topo.size > 0


class TestETOPOLoader:
    """Test ETOPO 2022 15 arc-second data loading."""

    @pytest.fixture
    def etopo_dir(self, project_root):
        """Return path to ETOPO data directory."""
        etopo_path = project_root / "data" / "etopo_15s"
        if not etopo_path.exists():
            pytest.skip(f"ETOPO data not found: {etopo_path}")
        return etopo_path

    @pytest.fixture
    def test_params(self, etopo_dir):
        """Create test parameters for ETOPO loading."""
        class TestParams:
            def __init__(self):
                self.path_etopo = str(etopo_dir) + "/"
                self.lat_extent = [35.0, 40.0]
                self.lon_extent = [-120.0, -115.0]
                self.etopo_cg = 4  # Use coarse-graining for faster testing
        return TestParams()

    def test_etopo_loader_initialization(self, test_params, etopo_dir):
        """Test ETOPO loader initialization and basic loading."""
        cell = var.topo_cell()

        loader = io.ncdata.read_etopo_topo(cell, test_params, verbose=False)

        # Check that data was loaded
        assert cell.lat is not None, "Latitude not loaded"
        assert cell.lon is not None, "Longitude not loaded"
        assert cell.topo is not None, "Topography not loaded"

        # Check dimensions
        assert len(cell.lat) > 0, "Latitude array is empty"
        assert len(cell.lon) > 0, "Longitude array is empty"
        assert cell.topo.size > 0, "Topography array is empty"

        # Check that loaded region matches requested extent (with small tolerance)
        # Note: Due to coarse-graining, exact boundaries may not be matched
        assert cell.lat.min() <= test_params.lat_extent[0] + 0.1
        assert cell.lat.max() >= test_params.lat_extent[1] - 0.1
        assert cell.lon.min() <= test_params.lon_extent[0] + 0.1
        assert cell.lon.max() >= test_params.lon_extent[1] - 0.1

    def test_etopo_data_values(self, test_params, etopo_dir):
        """Test that loaded ETOPO data has reasonable values."""
        cell = var.topo_cell()

        loader = io.ncdata.read_etopo_topo(cell, test_params, verbose=False)

        # Check for reasonable elevation values (California coast to Sierra Nevada)
        # Should have values from below sea level to several thousand meters
        assert cell.topo.min() >= -11000, "Topography minimum too low (deepest ocean ~11km)"
        assert cell.topo.max() <= 9000, "Topography maximum too high (Mt Everest ~9km)"

        # Check for fill values (should not be present after loading)
        assert not np.any(cell.topo == -99999), "Fill values present in loaded data"

        # Check that data is not all zeros
        assert not np.all(cell.topo == 0), "Topography data is all zeros"

    def test_etopo_coarse_graining(self, etopo_dir):
        """Test that coarse-graining reduces data size as expected."""
        class ParamsCG1:
            def __init__(self):
                self.path_etopo = str(etopo_dir) + "/"
                self.lat_extent = [36.0, 37.0]
                self.lon_extent = [-119.0, -118.0]
                self.etopo_cg = 1

        class ParamsCG4:
            def __init__(self):
                self.path_etopo = str(etopo_dir) + "/"
                self.lat_extent = [36.0, 37.0]
                self.lon_extent = [-119.0, -118.0]
                self.etopo_cg = 4

        # Load with no coarse-graining
        cell1 = var.topo_cell()
        loader1 = io.ncdata.read_etopo_topo(cell1, ParamsCG1(), verbose=False)

        # Load with 4x coarse-graining
        cell4 = var.topo_cell()
        loader4 = io.ncdata.read_etopo_topo(cell4, ParamsCG4(), verbose=False)

        # Check that coarse-graining reduces size
        size_ratio = cell1.topo.size / cell4.topo.size

        # Should be approximately 4x4 = 16 times reduction
        assert size_ratio > 10, f"Coarse-graining didn't reduce size enough: {size_ratio}x"
        assert size_ratio < 20, f"Coarse-graining reduced size too much: {size_ratio}x"

    def test_etopo_grid_structure(self, test_params, etopo_dir):
        """Test that loaded grid has correct structure."""
        cell = var.topo_cell()

        loader = io.ncdata.read_etopo_topo(cell, test_params, verbose=False)

        # Check that lat/lon are 1D arrays
        assert cell.lat.ndim == 1, "Latitude should be 1D"
        assert cell.lon.ndim == 1, "Longitude should be 1D"

        # Check that topo is 2D
        assert cell.topo.ndim == 2, "Topography should be 2D"

        # Check that dimensions match
        assert cell.topo.shape == (len(cell.lat), len(cell.lon)), \
            f"Topography shape {cell.topo.shape} doesn't match lat/lon ({len(cell.lat)}, {len(cell.lon)})"

        # Check that lat/lon are sorted
        assert np.all(np.diff(cell.lat) > 0), "Latitude should be sorted ascending"
        assert np.all(np.diff(cell.lon) > 0), "Longitude should be sorted ascending"
