"""
Shared pytest fixtures and utilities for pyCSA tests.
"""

import os
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def baseline_dir(project_root):
    """Return the baseline results directory."""
    return project_root / "outputs" / "baseline_results"


@pytest.fixture
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "data" / "test"


@pytest.fixture
def test_output_dir(project_root, tmp_path):
    """Return a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def simple_rect_cell():
    """Create a 64x64 rectangular topo_cell with all-True mask on [0, 2pi] domain."""
    from pycsa.core.var import topo_cell

    res = 64
    cell = topo_cell()
    cell.lon = np.linspace(0, 2 * np.pi, res, endpoint=False)
    cell.lat = np.linspace(0, 2 * np.pi, res, endpoint=False)
    cell.topo = np.zeros((res, res))
    cell.wlon = np.diff(cell.lon).mean()
    cell.wlat = np.diff(cell.lat).mean()
    cell.gen_mgrids()
    cell.mask = np.ones((res, res), dtype=bool)
    cell.lon_m = cell.lon_grid[cell.mask]
    cell.lat_m = cell.lat_grid[cell.mask]
    cell.topo_m = cell.topo[cell.mask] - cell.topo[cell.mask].mean()
    return cell


@pytest.fixture
def make_sinusoidal_cell():
    """Factory fixture: create a cell with cos(2pi*k*x/L + 2pi*l*y/L) topography."""
    from pycsa.core.var import topo_cell

    def _make(k, l, amplitude=100.0, res=64):
        cell = topo_cell()
        cell.lon = np.linspace(0, 2 * np.pi, res, endpoint=False)
        cell.lat = np.linspace(0, 2 * np.pi, res, endpoint=False)
        cell.wlon = np.diff(cell.lon).mean()
        cell.wlat = np.diff(cell.lat).mean()

        lon_grid, lat_grid = np.meshgrid(cell.lon, cell.lat)
        L = 2 * np.pi
        cell.topo = amplitude * np.cos(
            2 * np.pi * k * lon_grid / L + 2 * np.pi * l * lat_grid / L
        )
        cell.gen_mgrids()
        cell.mask = np.ones((res, res), dtype=bool)
        cell.lon_m = cell.lon_grid[cell.mask]
        cell.lat_m = cell.lat_grid[cell.mask]
        cell.topo_m = cell.topo[cell.mask] - cell.topo[cell.mask].mean()
        return cell

    return _make


def assert_arrays_close(actual, expected, rtol=1e-5, atol=1e-8, name="array"):
    """
    Assert that two numpy arrays are close within tolerance.

    Parameters
    ----------
    actual : np.ndarray
        The actual computed array
    expected : np.ndarray
        The expected baseline array
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    name : str
        Name of the array for error messages
    """
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{name} does not match baseline within tolerance (rtol={rtol}, atol={atol})"
    )


def assert_values_close(actual, expected, rtol=1e-5, atol=1e-8, name="value"):
    """
    Assert that two scalar values are close within tolerance.

    Parameters
    ----------
    actual : float
        The actual computed value
    expected : float
        The expected baseline value
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    name : str
        Name of the value for error messages
    """
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{name} = {actual} does not match baseline {expected} within tolerance"
    )


class BaselineComparison:
    """Helper class for comparing test results against baseline."""

    def __init__(self, rtol=1e-5, atol=1e-8):
        """
        Initialize baseline comparison.

        Parameters
        ----------
        rtol : float
            Relative tolerance for comparisons
        atol : float
            Absolute tolerance for comparisons
        """
        self.rtol = rtol
        self.atol = atol
        self.results = {}

    def add_result(self, name, actual, expected):
        """Add a result to compare."""
        self.results[name] = {
            'actual': actual,
            'expected': expected,
            'passed': None
        }

    def compare_all(self):
        """Compare all added results and return summary."""
        summary = {
            'passed': 0,
            'failed': 0,
            'failures': []
        }

        for name, data in self.results.items():
            try:
                if isinstance(data['actual'], np.ndarray):
                    assert_arrays_close(
                        data['actual'],
                        data['expected'],
                        self.rtol,
                        self.atol,
                        name
                    )
                else:
                    assert_values_close(
                        data['actual'],
                        data['expected'],
                        self.rtol,
                        self.atol,
                        name
                    )
                self.results[name]['passed'] = True
                summary['passed'] += 1
            except AssertionError as e:
                self.results[name]['passed'] = False
                summary['failed'] += 1
                summary['failures'].append({
                    'name': name,
                    'error': str(e)
                })

        return summary
