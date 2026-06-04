"""Shared helper: install a stub ``pycsa.local_paths`` when the real (gitignored,
per-developer) module is absent — e.g. in CI.

Used by both the reproducibility ``conftest`` and the standalone
``render_figures`` entry point so neither has to import ``pycsa.local_paths``
eagerly.
"""

from __future__ import annotations

import sys
import types

PATHS_ATTRS = (
    "compact_grid",
    "compact_topo",
    "icon_grid",
    "output",
    "merit",
    "rema",
    "etopo",
)


def ensure_local_paths_stub() -> None:
    """If ``pycsa.local_paths`` can't be imported, install a stub in sys.modules."""
    if "pycsa.local_paths" in sys.modules:
        return
    try:
        import pycsa.local_paths  # noqa: F401

        return
    except Exception:
        # Real file missing or broken — install a stub.
        pass

    import pycsa  # noqa: F401  (parent package; must import without error)

    stub = types.ModuleType("pycsa.local_paths")
    stub.paths = types.SimpleNamespace(**{a: "" for a in PATHS_ATTRS})
    sys.modules["pycsa.local_paths"] = stub
    setattr(sys.modules["pycsa"], "local_paths", stub)
