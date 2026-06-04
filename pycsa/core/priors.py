"""Pluggable Tikhonov priors for the linear-fit step.

The default fit in :func:`pycsa.core.lin_reg.do` adds a scalar
``lmbda * trace(MᵀM)/N`` to the diagonal of the normal-equations
matrix. That branch is preserved unchanged — it is what every
existing call site hits and what the reproducibility fixtures
pin down.

This module introduces a ``Prior`` protocol so spike scripts can
experiment with structured (per-mode) regularization without
touching the default code path. A prior is a callable that returns
the diagonal vector to add to ``diag(E_tilda_lm)``.

``IsotropicPrior`` is a parallel implementation of the existing
scalar-trace branch. It is NOT routed through by default; passing
``prior=IsotropicPrior()`` to ``lin_reg.do`` should produce the
same numerical result as ``prior=None``, within floating-point
re-association noise (the parallel path computes the same scalar
but the assignment sequence is different). Unit-tested for parity
on a synthetic ``E_tilda_lm``.

``SpectralPrior`` is the alternative under test in the spike: a
per-mode prior whose diagonal grows with wavenumber norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class Prior(Protocol):
    """Callable that returns the Tikhonov diagonal to add to ``diag(MᵀM)``.

    Parameters
    ----------
    fobj
        The :class:`pycsa.core.fourier.f_trans` instance used to build
        the design matrix. Carries the mode-index attributes
        (``m_i``, ``m_j``, optional ``k_idx``/``l_idx``) that structured
        priors need to compute per-mode weights.
    E_tilda_lm
        The dense normal-equations matrix (``MᵀM``), shape ``(N, N)``.
        Priors may use ``trace(E)`` for scale normalization.
    lmbda
        The overall regularization scale (the same scalar that the
        existing ``lin_reg.do`` accepts).

    Returns
    -------
    diag : np.ndarray, shape (N,)
        Non-negative diagonal to be added to ``diag(E_tilda_lm)``.
    """

    def __call__(self, fobj, E_tilda_lm: np.ndarray, *, lmbda: float) -> np.ndarray: ...


@dataclass
class IsotropicPrior:
    """Parallel implementation of the current scalar-trace diagonal.

    Returns ``lmbda * trace(E) / N * ones(N)``. Equivalent to the
    inline scalar-trace branch in ``lin_reg.do`` (the dense and sparse
    ``prior is None`` paths), within reassociation noise —
    the inline branch computes one scalar and broadcasts; this
    returns an array. Provided so spike scripts can compose
    it as an explicit baseline alongside other priors.

    Not the default path. ``lin_reg.do(prior=None)`` continues to
    use the inline branch and produces bit-identical fixture output.
    """

    def __call__(self, fobj, E_tilda_lm: np.ndarray, *, lmbda: float) -> np.ndarray:
        n = E_tilda_lm.shape[0]
        trace_scale = float(np.trace(E_tilda_lm)) / n * lmbda
        return np.full(n, trace_scale, dtype=E_tilda_lm.dtype)


@dataclass
class SpectralPrior:
    """Per-mode prior with diagonal growing with wavenumber norm.

    For each mode ``m`` with wavenumber pair ``(k_m, l_m)``:

    .. math::

        \\lambda_m = \\lambda \\cdot \\frac{\\mathrm{tr}(E)}{N}
                    \\cdot \\left(
                        \\frac{\\Vert k_m \\Vert}{\\Vert k \\Vert_{\\max}}
                        + \\varepsilon
                    \\right)^\\alpha

    Equivalent to a zero-mean Gaussian prior on the coefficients
    with variance proportional to ``‖k_m‖^(-alpha)``. Implies an
    assumed topographic power spectrum behaving like ``‖k‖^(-alpha)``,
    so ``alpha`` should be calibrated from the input fixture's
    empirical power-law slope rather than picked from generic
    physics constants. ``alpha=0`` recovers the isotropic baseline
    (modulo the ``+ eps`` floor).

    Parameters
    ----------
    alpha
        Spectral decay exponent. ``alpha=0`` reduces to isotropic
        (constant diagonal at ``(1 + eps)^0 == 1``).
    eps
        Small additive floor that keeps the DC mode (``‖k‖ = 0``)
        from receiving zero regularization. Documented knob, not
        a free hyperparameter.
    """

    alpha: float
    eps: float = 1e-3

    def __call__(self, fobj, E_tilda_lm: np.ndarray, *, lmbda: float) -> np.ndarray:
        n = E_tilda_lm.shape[0]
        weights = _wavenumber_weights(fobj, n)
        norm = np.max(weights) if weights.size > 0 else 1.0
        if norm == 0.0:
            norm = 1.0
        scaled = (weights / norm + self.eps) ** self.alpha
        trace_scale = float(np.trace(E_tilda_lm)) / n * lmbda
        return (trace_scale * scaled).astype(E_tilda_lm.dtype, copy=False)


def _wavenumber_weights(fobj, n_modes: int) -> np.ndarray:
    """Per-column wavenumber norm ``‖k_m‖`` for the design matrix's columns.

    The design matrix produced by ``f_trans.do_full`` has columns
    laid out as ``[cos_columns | sin_columns]`` after the slicing
    in ``f_trans.do_full``. For the spike, we accept a small
    simplification: we compute the wavenumber norm per ``(m_i, m_j)``
    pair from ``fobj.m_i`` and ``fobj.m_j``, then duplicate across
    cos/sin columns. If the column count doesn't divide cleanly,
    we fall back to a uniform weight (effectively recovering
    isotropic behavior for that fixture — a safe degradation).

    When ``fobj.pick_kls`` is True, the mode indices come from
    ``fobj.k_idx``/``fobj.l_idx`` directly.
    """
    m_i = getattr(fobj, "m_i", None)
    m_j = getattr(fobj, "m_j", None)
    if m_i is None or m_j is None:
        # fobj not yet configured — fall back to uniform
        return np.ones(n_modes)

    if getattr(fobj, "pick_kls", False) and hasattr(fobj, "k_idx"):
        # Sparse mode set: each selected (k_idx[i], l_idx[i]) is one mode.
        # The selected indices are remapped into the ``m_i``/``m_j`` axes
        # via modulo (``k_idx % len(m_i)``, ``l_idx % len(m_j)``) so an
        # out-of-range pick still lands on a valid wavenumber slot; the
        # per-mode norm is computed from those remapped values FIRST, then
        # duplicated across the cos/sin column halves below.
        k_idx = np.asarray(fobj.k_idx)
        l_idx = np.asarray(fobj.l_idx)
        mi_sel = np.asarray(m_i)[k_idx % len(m_i)]
        mj_sel = np.asarray(m_j)[l_idx % len(m_j)]
        per_mode = np.hypot(mi_sel.astype(float), mj_sel.astype(float))
        # Duplicate across cos/sin halves if dimensions match
        if n_modes == 2 * per_mode.size:
            return np.concatenate([per_mode, per_mode])
        if n_modes == per_mode.size:
            return per_mode
        return np.ones(n_modes)

    # Dense (full) basis case: mode count is roughly nhar_i * nhar_j
    # split between cos and sin halves. We compute the full
    # (nhar_i, nhar_j) grid of ‖(m_i, m_j)‖ and flatten in the same
    # order as the design matrix's columns. The slicing in
    # f_trans.do_full removes a fixed slab of cos and sin columns;
    # we approximate by tiling and trimming to ``n_modes``.
    mi_arr = np.asarray(m_i, dtype=float)
    mj_arr = np.asarray(m_j, dtype=float)
    grid = np.hypot(mi_arr[:, None], mj_arr[None, :]).ravel()
    if grid.size == 0:
        return np.ones(n_modes)
    # cos and sin halves both draw from the same wavenumber grid
    half = (n_modes + 1) // 2
    tiled = np.concatenate(
        [
            np.tile(grid, (half + grid.size) // grid.size + 1)[:half],
            np.tile(grid, (half + grid.size) // grid.size + 1)[: n_modes - half],
        ]
    )
    return tiled[:n_modes]


__all__ = ["Prior", "IsotropicPrior", "SpectralPrior"]
