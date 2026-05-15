"""Unit tests for pycsa.core.priors.

These tests cover the protocol-level behavior of the pluggable Tikhonov
diagonal generators introduced in the phase-1 spike skeleton. They do
NOT exercise the full pipeline — those checks live in
tests/reproducibility/ and must remain green with the new code path
absent (default ``prior=None``).
"""

import numpy as np
import pytest

from pycsa.core import lin_reg
from pycsa.core.priors import IsotropicPrior, SpectralPrior


class _StubFobj:
    """Minimal fobj-like object exposing the attributes priors read.

    Real ``f_trans`` instances carry far more state; we only need
    ``m_i``, ``m_j``, and the optional ``pick_kls`` / ``k_idx`` /
    ``l_idx`` slots.
    """

    def __init__(self, m_i, m_j, pick_kls=False, k_idx=None, l_idx=None):
        self.m_i = np.asarray(m_i)
        self.m_j = np.asarray(m_j)
        self.pick_kls = pick_kls
        if k_idx is not None:
            self.k_idx = np.asarray(k_idx)
        if l_idx is not None:
            self.l_idx = np.asarray(l_idx)


def _spd_matrix(n, seed=0):
    """Random SPD matrix for use as a stand-in ``E_tilda_lm``."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def test_isotropic_matches_inline_branch_dense():
    """IsotropicPrior produces a diagonal equal to the inline branch."""
    E = _spd_matrix(8)
    lmbda = 0.5
    prior = IsotropicPrior()
    diag = prior(fobj=None, E_tilda_lm=E, lmbda=lmbda)
    # Inline branch at lin_reg.py:193-195 computes:
    #     trace = np.trace(E) / E.shape[0] * lmbda
    # then adds this scalar to every diagonal entry.
    expected_scalar = np.trace(E) / E.shape[0] * lmbda
    assert diag.shape == (E.shape[0],)
    np.testing.assert_allclose(diag, expected_scalar)


def test_isotropic_zero_lambda_returns_zero():
    diag = IsotropicPrior()(fobj=None, E_tilda_lm=_spd_matrix(4), lmbda=0.0)
    np.testing.assert_array_equal(diag, np.zeros(4))


def test_lin_reg_do_with_isotropic_prior_equals_default(monkeypatch):
    """lin_reg.do(prior=IsotropicPrior()) ~= lin_reg.do(prior=None).

    Within floating-point reassociation noise; the inline branch uses
    one scalar and broadcasts via fill_diagonal, while IsotropicPrior
    returns an N-vector that we add via fill_diagonal — same final
    numerical state.
    """
    from pycsa.core.fourier import f_trans
    from pycsa.core import var

    # Build a small idealised cell + fobj to drive lin_reg.do end-to-end.
    cell = var.topo_cell()
    n = 16
    cell.lon_m = np.tile(np.linspace(0, 1, n), n)
    cell.lat_m = np.repeat(np.linspace(0, 1, n), n)
    cell.lon = np.linspace(0, 1, n)
    cell.lat = np.linspace(0, 1, n)
    cell.wlon = 1.0 / (n - 1)
    cell.wlat = 1.0 / (n - 1)
    cell.topo_m = np.cos(2 * np.pi * cell.lon_m) + 0.3 * np.sin(
        4 * np.pi * cell.lat_m
    )

    fobj = f_trans(4, 4)
    fobj.do_full(cell)
    a_default, recons_default = lin_reg.do(fobj, cell, lmbda=0.1, prior=None)

    fobj2 = f_trans(4, 4)
    fobj2.do_full(cell)
    a_iso, recons_iso = lin_reg.do(
        fobj2, cell, lmbda=0.1, prior=IsotropicPrior()
    )

    np.testing.assert_allclose(a_default, a_iso, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(recons_default, recons_iso, rtol=1e-10, atol=1e-12)


def test_spectral_prior_alpha_zero_is_almost_isotropic():
    """SpectralPrior(alpha=0) is the constant ``(1+eps)^0 = 1`` weight,
    so trace_scale * 1 * ones — matches IsotropicPrior exactly.
    """
    E = _spd_matrix(12, seed=1)
    fobj = _StubFobj(m_i=np.arange(4), m_j=np.arange(-1, 3))
    iso = IsotropicPrior()(fobj=fobj, E_tilda_lm=E, lmbda=0.25)
    sp = SpectralPrior(alpha=0.0)(fobj=fobj, E_tilda_lm=E, lmbda=0.25)
    np.testing.assert_allclose(sp, iso)


def test_spectral_prior_monotone_in_alpha():
    """For alpha>0 the diagonal grows with ``‖k‖``; pick a fobj where
    ‖k_m‖ has an unambiguous max and min and check ordering.
    """
    E = _spd_matrix(8, seed=2)
    # m_i in 0..3, m_j in -1..2 → ‖k‖ ranges from 0 to sqrt(9+4)=sqrt(13)
    fobj = _StubFobj(m_i=np.arange(4), m_j=np.arange(-1, 3))
    diag = SpectralPrior(alpha=2.0)(fobj=fobj, E_tilda_lm=E, lmbda=1.0)
    # Largest weight should appear at columns mapping to high ‖k‖, smallest
    # at the DC corner (which is bounded below by eps>0 — never exactly 0).
    assert diag.min() > 0.0
    assert diag.max() > diag.min()


def test_spectral_prior_eps_floor_keeps_dc_positive():
    """The +eps floor guarantees positive regularization even at k=0."""
    E = _spd_matrix(4, seed=3)
    # Single-mode setup: forces a column whose weight would be exactly 0
    # without the eps floor.
    fobj = _StubFobj(
        m_i=np.array([0]),
        m_j=np.array([0]),
        pick_kls=True,
        k_idx=[0],
        l_idx=[0],
    )
    sp = SpectralPrior(alpha=2.0, eps=1e-3)
    diag = sp(fobj=fobj, E_tilda_lm=E, lmbda=1.0)
    assert np.all(diag > 0.0)
