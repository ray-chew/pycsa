"""
Linear regression module with buffer pool and sparse solver support
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import gmres
from scipy.linalg import blas
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve


def get_coeffs(fobj, ctx=None, buffer_pool=None):
    """Assembles the Fourier coefficients from the sine and cosine terms generated in the :class:`Fourier transformer class <src.fourier.f_trans>`.

    Parameters
    ----------
    fobj : :class:`src.fourier.f_trans` instance
        instance of the Fourier transformer class.
    ctx : ComputeContext, optional
        Compute context bundling the buffer pool. If absent, defaults to
        ``fobj.ctx`` (set by ``f_trans.__init__``).
    buffer_pool : BufferPool, optional
        **Deprecated.** Pass ``ctx=ComputeContext(buffer_pool=...)`` instead.

    Returns
    -------
    array-like
        2D array corresponding to the ``M`` matrix.
    """
    from pycsa.compute.context import _resolve_ctx

    if ctx is None and buffer_pool is None:
        # Inherit from fobj when neither is explicitly supplied — that's
        # the common path now that f_trans owns a ctx.
        ctx = getattr(fobj, "ctx", None)
    ctx = _resolve_ctx(ctx, buffer_pool)
    buffer_pool = ctx.buffer_pool

    Ncos = fobj.bf_cos
    Nsin = fobj.bf_sin

    n_points = Ncos.shape[0]
    n_modes = Ncos.shape[1] + Nsin.shape[1]

    if buffer_pool:
        # Use buffer pool - handles variable sizes dynamically
        coeff = buffer_pool.get_or_create("coeff", (n_points, n_modes), Ncos.dtype)
        coeff[:, : Ncos.shape[1]] = Ncos
        coeff[:, Ncos.shape[1] :] = Nsin
    else:
        # Fallback for backward compatibility
        coeff = np.hstack([Ncos, Nsin])

    del fobj.bf_cos
    del fobj.bf_sin

    if fobj.grad:
        if buffer_pool:
            # Allocate larger buffer for gradient stacking
            coeff_grad = buffer_pool.get_or_create(
                "coeff_grad", (2 * n_points, n_modes), Ncos.dtype
            )
            coeff_grad[:n_points] = coeff
            coeff_grad[n_points:] = coeff
            return coeff_grad
        else:
            coeff = np.vstack([coeff, coeff])

    return coeff


def do(
    fobj,
    cell,
    lmbda=0.0,
    iter_solve=True,
    save_coeffs=False,
    ctx=None,
    buffer_pool=None,
    use_sparse=False,
    prior=None,
):
    """
    Does the linear regression with optional buffer pool and sparse solver

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
    ctx : ComputeContext, optional
        Compute context bundling the buffer pool. If absent, defaults to
        ``fobj.ctx``. If ``ctx.prior`` is set and the ``prior`` kwarg is
        ``None``, the context's prior is used.
    buffer_pool : BufferPool, optional
        **Deprecated.** Pass ``ctx=ComputeContext(buffer_pool=...)`` instead.
    use_sparse : bool, optional
        Use sparse matrix solver (automatic for few modes), by default False
    prior : :class:`pycsa.core.priors.Prior`, optional
        Pluggable Tikhonov diagonal generator. When ``None`` (default),
        the existing scalar-trace branch is used and numerics are
        bit-identical to historical behavior. When supplied, the prior is
        called as ``prior(fobj, E_tilda_lm, lmbda=lmbda)`` and the returned
        diagonal is added to ``diag(E_tilda_lm)``. Opt-in only; no
        production caller passes this kwarg.

    Returns
    -------
    a_m : list
        list of Fourier amplitudes corresponding to the unknown vector in the linear problem
    data_recons : like
        vector-like topography reconstructed from ``a_m``
    """
    from pycsa.compute.context import _resolve_ctx

    if ctx is None and buffer_pool is None:
        ctx = getattr(fobj, "ctx", None)
    ctx = _resolve_ctx(ctx, buffer_pool)
    buffer_pool = ctx.buffer_pool

    # ctx-level prior fallback: explicit kwarg wins, otherwise pick up
    # whatever the context carries (None for default ComputeContexts).
    if prior is None:
        prior = getattr(ctx, "prior", None)

    if fobj.grad:
        cell.get_grad()
        data = cell.grad_topo_m
    else:
        data = cell.topo_m

    coeff = get_coeffs(fobj, ctx=ctx)

    if save_coeffs:
        fobj.coeff = coeff
        return None, None

    # Determine if sparse solver should be used
    # Criteria: pick_kls enabled AND <10% of total modes selected
    use_sparse_solver = use_sparse or (
        getattr(fobj, "pick_kls", False)
        and hasattr(fobj, "k_idx")
        and len(fobj.k_idx) < 0.1 * (fobj.nhar_i * fobj.nhar_j)
    )

    if use_sparse_solver:
        # ============================================================
        # SPARSE PATH: For Second Approximation with few modes
        # ============================================================
        # Convert to sparse matrix (CSR format is efficient for matrix ops)
        coeff_sparse = csr_matrix(coeff)
        coeff_T_sparse = coeff_sparse.T

        # Compute sparse normal equations
        h_tilda_l_sparse = coeff_T_sparse @ data.reshape(-1, 1)
        E_tilda_lm_sparse = coeff_T_sparse @ coeff_sparse

        # Add regularization to sparse matrix
        if lmbda > 0:
            if prior is None:
                trace = E_tilda_lm_sparse.diagonal().mean() * lmbda
                E_tilda_lm_sparse = E_tilda_lm_sparse + trace * eye(
                    E_tilda_lm_sparse.shape[0]
                )
            else:
                # Densify just for the prior call (it expects the dense
                # normal-equations matrix); the actual solve stays sparse.
                E_dense_view = E_tilda_lm_sparse.toarray()
                diag_add = np.asarray(
                    prior(fobj, E_dense_view, lmbda=lmbda), dtype=float
                )
                if diag_add.shape != (E_tilda_lm_sparse.shape[0],):
                    raise ValueError(
                        "prior returned diag of shape %s; expected %s"
                        % (diag_add.shape, (E_tilda_lm_sparse.shape[0],))
                    )
                # Inject the prior diagonal into the sparse matrix in-place
                # by adding a sparse diag matrix.
                from scipy.sparse import diags as _sp_diags
                E_tilda_lm_sparse = E_tilda_lm_sparse + _sp_diags(
                    diag_add, 0, shape=E_tilda_lm_sparse.shape
                )

        # Solve with sparse solver (direct solver for sparse SPD matrices)
        # Convert RHS to dense array if it's sparse, otherwise use as-is
        if hasattr(h_tilda_l_sparse, "toarray"):
            rhs = h_tilda_l_sparse.toarray().flatten()
        else:
            rhs = np.asarray(h_tilda_l_sparse).flatten()
        a_m = spsolve(E_tilda_lm_sparse, rhs)

        # Reconstruct (sparse @ dense is efficient)
        recons_result = coeff_sparse @ a_m
        if hasattr(recons_result, "toarray"):
            data_recons = recons_result.toarray().flatten()
        else:
            data_recons = np.asarray(recons_result).flatten()

    else:
        # ============================================================
        # DENSE PATH: Standard approach with optional buffer reuse
        # ============================================================
        # Compute RHS
        h_tilda_l = np.dot(coeff.T, data.reshape(-1, 1)).flatten()

        # Compute LHS with optional buffer reuse
        if buffer_pool:
            n_modes = coeff.shape[1]
            E_tilda_lm = buffer_pool.get_or_create(
                "E_tilda_lm", (n_modes, n_modes), np.float64
            )
            # Compute and store in buffer
            E_tilda_lm[:] = np.dot(coeff.T, coeff)
        else:
            E_tilda_lm = np.dot(coeff.T, coeff)

        # Add regularization to diagonal (vectorized for speed)
        if lmbda > 0:
            if prior is None:
                # Preserved scalar-trace branch — what the reproducibility
                # fixtures pin down. Bit-identical to historical behavior.
                trace = np.trace(E_tilda_lm) / E_tilda_lm.shape[0] * lmbda
                np.fill_diagonal(E_tilda_lm, np.diag(E_tilda_lm) + trace)
            else:
                diag_add = np.asarray(
                    prior(fobj, E_tilda_lm, lmbda=lmbda), dtype=E_tilda_lm.dtype
                )
                if diag_add.shape != (E_tilda_lm.shape[0],):
                    raise ValueError(
                        "prior returned diag of shape %s; expected %s"
                        % (diag_add.shape, (E_tilda_lm.shape[0],))
                    )
                np.fill_diagonal(E_tilda_lm, np.diag(E_tilda_lm) + diag_add)

        # E_tilda_lm is symmetric positive definite (M^T M form with regularization)
        # Use Cholesky decomposition for 2-5x speedup vs GMRES
        if iter_solve:
            try:
                # Attempt Cholesky factorization (fastest for SPD matrices)
                c, lower = la.cho_factor(E_tilda_lm, lower=True, check_finite=False)
                a_m = la.cho_solve((c, lower), h_tilda_l, check_finite=False)
            except la.LinAlgError:
                # Fallback to GMRES if matrix is not positive definite
                szc = E_tilda_lm.shape[0]
                a_m, info = gmres(
                    E_tilda_lm,
                    h_tilda_l,
                    rtol=1e-8,  # Convergence tolerance (renamed from tol in SciPy 1.12)
                    atol=1e-10,  # Absolute tolerance
                    maxiter=min(szc, 100),
                )  # Limit iterations
                if info != 0:
                    # GMRES didn't converge, warn user
                    import warnings

                    warnings.warn(
                        f"GMRES did not converge (info={info}), solution may be inaccurate"
                    )
        else:
            # Direct inversion (slower, but kept for compatibility)
            a_m = la.inv(E_tilda_lm).dot(h_tilda_l)

        data_recons = coeff.dot(a_m)

    return a_m, data_recons
