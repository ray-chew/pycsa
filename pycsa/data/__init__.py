"""Data containers for grids, topography, and analysis results.

Reorganises what used to live in ``pycsa.core.var`` into thematic
submodules:

* :mod:`pycsa.data.cell` — :class:`grid`, :class:`topo`, :class:`topo_cell`
* :mod:`pycsa.data.results` — :class:`analysis`

User-parameter config moves to :mod:`pycsa.config.params`.
"""

from pycsa.data.cell import grid, topo, topo_cell
from pycsa.data.results import analysis

__all__ = ["grid", "topo", "topo_cell", "analysis"]
