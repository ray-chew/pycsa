"""
Shared pytest fixtures and utilities for pyCSA tests.
"""

# ---------------------------------------------------------------------------
# Cartopy stub — let tests run in environments without cartopy installed.
# pycsa.__init__ eagerly imports pycsa.plotting.cart_plot which imports
# cartopy. The tests don't actually call any plotting functions, so a stub
# is enough to satisfy the import chain. If real cartopy is installed, this
# is a no-op.
# ---------------------------------------------------------------------------
try:
    import cartopy  # noqa: F401
except ImportError:
    import sys
    import types

    def _stub_pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []  # marks as package so submodule imports work
        sys.modules[name] = m
        return m

    def _stub_attrs(mod, *names):
        for n in names:
            setattr(mod, n, type(n, (), {}))

    _stub_pkg("cartopy")
    _crs = _stub_pkg("cartopy.crs")
    _stub_attrs(_crs, "PlateCarree", "Mollweide", "Robinson", "Geodetic")
    _stub_pkg("cartopy.mpl")
    _ticker = _stub_pkg("cartopy.mpl.ticker")
    _stub_attrs(_ticker, "LongitudeFormatter", "LatitudeFormatter",
                "LongitudeLocator", "LatitudeLocator")
    _stub_pkg("cartopy.feature")
    _stub_pkg("cartopy.io")
    _stub_pkg("cartopy.io.shapereader")

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
def test_output_dir(project_root, tmp_path):
    """Return a temporary directory for test outputs."""
    return tmp_path


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
