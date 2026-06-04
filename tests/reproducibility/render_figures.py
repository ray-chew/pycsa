"""Regenerate every reproducibility fixture's ``figure.png`` from bundled input.

Runs each capture script's ``render_only`` (which re-runs the pipeline against
the committed ``<fixture>/input`` slices and re-renders the figure) — no real
source data required, so it works in CI. The pinned ``output.nc`` /
``manifest.yml`` are left untouched; only ``figure.png`` is rewritten.

Usage::

    MPLBACKEND=Agg python -m tests.reproducibility.render_figures
"""

from __future__ import annotations

import sys

# netCDF4 must be imported before pycsa.core.io (HDF5 init ordering quirk), or
# reading the bundled grid tiles raises "NetCDF: HDF error".
import netCDF4  # noqa: F401

from tests.reproducibility.local_paths_stub import ensure_local_paths_stub


def main(argv=None) -> int:
    ensure_local_paths_stub()

    from tests.reproducibility.capture import (
        capture_idealised,
        capture_regional_merit,
        capture_etopo_single_cell,
    )

    renderers = [
        ("idealised", capture_idealised.render_only),
        ("regional_merit", capture_regional_merit.render_only),
        ("etopo_single_cell", capture_etopo_single_cell.render_only),
    ]
    for name, render in renderers:
        print(f"rendering {name} figure ...")
        render()
        print(f"  -> {name} figure.png written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
