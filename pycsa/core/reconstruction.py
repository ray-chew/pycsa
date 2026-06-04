"""Reassembly of masked reconstruction vectors back to 2D fields.

Provides :func:`recon_2D`, which scatters a flat per-point
reconstruction vector (as produced by the linear-fit step) back onto
the 2D grid of a topography cell using the cell's boolean mask, leaving
out-of-mask points at zero.
"""

import numpy as np


def recon_2D(recons_z, cell):
    """
    Reassembles the vector-like ``recons_z`` into a 2D representation given by the properties of :class:`cell <pycsa.data.cell.topo_cell>`.

    Parameters
    ----------
    recons_z : array-like
        reconstructed topography from :func:`pycsa.core.lin_reg.do`
    cell : :class:`pycsa.data.cell.topo_cell`
        instance of the ``cell`` object

    Returns
    -------
    array-like
        2D reconstructed topography, values outside the mask are set to zero.
    """
    # Vectorized implementation - replaces nested Python loops with NumPy indexing
    recons_z_2D = np.zeros(np.shape(cell.topo))
    recons_z_2D[cell.mask] = recons_z

    return recons_z_2D
