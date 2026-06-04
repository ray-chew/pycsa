"""Fixtures for the reproducibility suite.

The pyCSA pipelines consume input paths from ``pycsa.local_paths.paths``.
That file is gitignored (per-developer), so CI runs without it. This conftest
installs a stub ``pycsa.local_paths`` module if the real one is missing, then
the ``patch_local_paths_for`` fixture redirects ``paths.*`` attributes to a
case's bundled ``input/`` directory.

Bundled inputs are kept in production filename/schema shape so the real IO
loaders are smoke-tested by the reproducibility suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.reproducibility.local_paths_stub import ensure_local_paths_stub

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def patch_local_paths_for(monkeypatch, tmp_path):
    """Redirect ``pycsa.local_paths.paths`` to a case's bundled ``input/`` dir.

    Usage::

        def test_foo(patch_local_paths_for):
            patch_local_paths_for(FIXTURES_DIR / 'etopo_single_cell')
            ...  # pipeline now reads from the bundled inputs
    """
    ensure_local_paths_stub()
    from pycsa import local_paths

    def _patch(case_dir: Path | str) -> None:
        case_dir = Path(case_dir)
        input_dir = case_dir / "input"

        icon_grid = input_dir / "icon_grid.nc"
        if icon_grid.exists():
            monkeypatch.setattr(local_paths.paths, "icon_grid", str(icon_grid))

        merit_dir = input_dir / "merit"
        if merit_dir.is_dir():
            monkeypatch.setattr(local_paths.paths, "merit", str(merit_dir) + "/")

        etopo_dir = input_dir / "etopo"
        if etopo_dir.is_dir():
            monkeypatch.setattr(local_paths.paths, "etopo", str(etopo_dir) + "/")

        # Output always goes to tmp_path for test runs; the comparator works
        # with the pipeline's in-memory return value, not on-disk artifacts.
        monkeypatch.setattr(local_paths.paths, "output", str(tmp_path) + "/")

    return _patch
