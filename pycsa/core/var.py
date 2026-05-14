"""Deprecation shim for the old monolithic ``var.py``.

This module previously held six classes; they have been split into
thematic submodules:

* :class:`grid`, :class:`topo`, :class:`topo_cell` → :mod:`pycsa.data.cell`
* :class:`analysis` → :mod:`pycsa.data.results`
* :class:`params` → :mod:`pycsa.config.params`
* :class:`obj` → deprecated; use :class:`types.SimpleNamespace`

Re-exports keep ``from pycsa.core import var`` (and ``var.grid()`` /
``var.params()`` / etc.) working for at least one release. New code
should import from the new locations directly.
"""

import warnings as _warnings

from pycsa.config.params import params
from pycsa.data.cell import grid, topo, topo_cell
from pycsa.data.results import analysis


class obj:
    """Generic attribute bag.

    .. deprecated::
       Use :class:`types.SimpleNamespace` instead. Retained as a
       deprecation alias because there are ~15 external callers
       (test fixtures, archived run scripts, inputs/) that still do
       ``tri = var.obj()`` to build attribute bags at runtime. Those
       call sites can migrate at their own pace.
    """

    def __init__(self):
        _warnings.warn(
            "pycsa.core.var.obj is deprecated; use types.SimpleNamespace instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def print(self):
        """Print all attributes to stdout (legacy diagnostic helper)."""
        for name in vars(self):
            print(name, getattr(self, name))


__all__ = ["grid", "topo", "topo_cell", "analysis", "params", "obj"]
