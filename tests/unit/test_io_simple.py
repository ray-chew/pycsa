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
