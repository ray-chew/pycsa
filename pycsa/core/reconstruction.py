import numpy as np


def recon_2D(recons_z, cell):
    """
    Reassembles the vector-like ``recons_z`` into a 2D representation given by the properties of :class:`cell <src.var.topo_cell>`.

    Parameters
    ----------
    recons_z : list
        reconstructed topography from :func:`src.lin_reg.do`
    cell : :class:`src.var.topo_cell`
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
