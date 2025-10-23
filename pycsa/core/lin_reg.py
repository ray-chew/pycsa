"""
Linear regression module
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import gmres
from scipy.linalg import blas


def get_coeffs(fobj):
    """Assembles the Fourier coefficients from the sine and cosine terms generated in the :class:`Fourier transformer class <src.fourier.f_trans>`.

    Parameters
    ----------
    fobj : :class:`src.fourier.f_trans` instance
        instance of the Fourier transformer class.

    Returns
    -------
    array-like
        2D array corresponding to the ``M`` matrix.
    """
    Ncos = fobj.bf_cos
    Nsin = fobj.bf_sin

    coeff = np.hstack([Ncos, Nsin])

    del fobj.bf_cos
    del fobj.bf_sin

    if fobj.grad:
        coeff = np.vstack([coeff, coeff])

    return coeff


def do(fobj, cell, lmbda=0.0, iter_solve=True, save_coeffs=False):
    """
    Does the linear regression

    Parameters
    ----------
    fobj : :class:`src.fourier.f_trans` instance
        instance of the Fourier transformer class.
    cell : :class:`src.var.topo_cell` instance
        cell object instance
    lmbda : float, optional
        regularisation parameter, by default 0.0
    iter_solve : bool, optional
        toggles between using direct or iterative solver, by default True
    save_coeffs : bool, optional
        skips the linear regression and just saves the generated ``M`` matrix for diagnostics and debugging, by default False

    Returns
    -------
    a_m : list
        list of Fourier amplitudes corresponding to the unknown vector in the linear problem
    data_recons : like
        vector-like topography reconstructed from ``a_m``
    """
    if fobj.grad:
        cell.get_grad()
        data = cell.grad_topo_m
    else:
        data = cell.topo_m

    coeff = get_coeffs(fobj)

    if save_coeffs:
        fobj.coeff = coeff
        return None, None

    # Compute RHS and LHS efficiently
    h_tilda_l = np.dot(coeff.T, data.reshape(-1, 1)).flatten()
    E_tilda_lm = np.dot(coeff.T, coeff)

    # Add regularization to diagonal (vectorized for speed)
    if lmbda > 0:
        trace = np.trace(E_tilda_lm) / E_tilda_lm.shape[0] * lmbda
        np.fill_diagonal(E_tilda_lm, np.diag(E_tilda_lm) + trace)

    # E_tilda_lm is symmetric positive definite (M^T M form with regularization)
    # Use Cholesky decomposition for 2-5x speedup vs GMRES
    if iter_solve:
        try:
            # Attempt Cholesky factorization (fastest for SPD matrices)
            # scipy.linalg.cho_factor checks for positive definiteness
            c, lower = la.cho_factor(E_tilda_lm, lower=True, check_finite=False)
            a_m = la.cho_solve((c, lower), h_tilda_l, check_finite=False)
        except la.LinAlgError:
            # Fallback to GMRES if matrix is not positive definite
            # Add tolerance and iteration controls for better convergence
            a_m, info = gmres(E_tilda_lm, h_tilda_l,
                             tol=1e-8,           # Convergence tolerance
                             atol=1e-10,         # Absolute tolerance
                             maxiter=min(szc, 100))  # Limit iterations
            if info != 0:
                # GMRES didn't converge, warn user
                import warnings
                warnings.warn(f"GMRES did not converge (info={info}), solution may be inaccurate")
    else:
        # Direct inversion (slower, but kept for compatibility)
        a_m = la.inv(E_tilda_lm).dot(h_tilda_l)

    # regular FFT considers normalization by total nu  mber of datapoints N=100
    # so multiply the Fourier coefficients by N here
    # a_m = a_m#*len(data)

    data_recons = coeff.dot(a_m)

    return a_m, data_recons
