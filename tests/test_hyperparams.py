"""Unit tests for pycsa.core.hyperparams.

Exercise each selector at the protocol level on synthetic problems
with known answers. End-to-end validation against the real fixtures
is the job of scripts/validate_hyperparam_defaults.py, not these.
"""

import numpy as np
import pytest

from pycsa.core.hyperparams import (
    EmpiricalSpectralSlope,
    FixedAlpha,
    FixedLambda,
    GCVSelector,
    Hyperparams,
    JointGCVSelector,
    MarginalLikelihoodSelector,
    SpatialCVSelector,
    SlopeFit,
    build_spectral_prior,
)
from pycsa.core.priors import SpectralPrior


# -----------------------------------------------------------------------------
# Synthetic helpers
# -----------------------------------------------------------------------------


def _planted_power_law_field(shape, beta, rng):
    """Generate a 2D field whose periodogram follows ``P(k) ∝ k^(-beta)``.

    Build amplitudes ``|F(k)| ∝ k^(-beta/2)`` with random phases,
    then inverse-FFT to the spatial domain. The resulting field is
    real (we symmetrize) and its empirical periodogram should match
    the planted slope within sampling noise.
    """
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    kmag = np.hypot(ky, kx)
    kmag[0, 0] = 1.0  # avoid div-by-zero at DC
    amp = kmag ** (-beta / 2.0)
    amp[0, 0] = 0.0  # zero out DC
    phase = rng.uniform(0, 2 * np.pi, size=shape)
    F = amp * np.exp(1j * phase)
    field = np.fft.ifft2(F).real
    return field


def _simple_design(n_points=120, n_modes=20, seed=0):
    """Random Gaussian design matrix with unit-norm columns."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n_points, n_modes))
    M /= np.linalg.norm(M, axis=0, keepdims=True)
    return M, rng


# -----------------------------------------------------------------------------
# Alpha selectors
# -----------------------------------------------------------------------------


def test_fixed_alpha_passthrough():
    assert FixedAlpha(1.7)(np.zeros((4, 4))) == 1.7


def test_empirical_slope_recovers_planted_power_law():
    rng = np.random.default_rng(42)
    target_beta = 2.0
    field = _planted_power_law_field((128, 128), beta=target_beta, rng=rng)
    fit = EmpiricalSpectralSlope()(field)
    assert isinstance(fit, SlopeFit)
    assert abs(fit.alpha - target_beta) < 0.3
    assert fit.stderr >= 0.0
    assert 0.0 <= fit.r_squared <= 1.0


def test_empirical_slope_returns_stderr_and_r2():
    """Whatever the recovered slope is, the returned record must
    carry usable stderr and R² (so downstream callers can detect
    poorly-fit fields)."""
    rng = np.random.default_rng(7)
    field = rng.normal(size=(64, 64))  # white noise — flat spectrum
    fit = EmpiricalSpectralSlope()(field)
    assert np.isfinite(fit.stderr)
    assert np.isfinite(fit.r_squared)


def test_empirical_slope_raises_on_1d_input():
    with pytest.raises(ValueError, match="2D"):
        EmpiricalSpectralSlope()(np.zeros(64))


def test_empirical_slope_stable_across_band_perturbations():
    """β should not move much when the fit band is widened/narrowed
    by reasonable amounts — checks the +/- stderr is meaningful."""
    rng = np.random.default_rng(11)
    field = _planted_power_law_field((128, 128), beta=2.0, rng=rng)
    fits = [
        EmpiricalSpectralSlope(k_min_frac=0.02, k_max_frac=0.4)(field),
        EmpiricalSpectralSlope(k_min_frac=0.03, k_max_frac=0.5)(field),
        EmpiricalSpectralSlope(k_min_frac=0.04, k_max_frac=0.45)(field),
    ]
    alphas = np.array([f.alpha for f in fits])
    # Spread should be on the order of the largest reported stderr.
    max_stderr = max(f.stderr for f in fits)
    assert alphas.std() < max(5 * max_stderr, 0.3)


# -----------------------------------------------------------------------------
# Lambda selectors
# -----------------------------------------------------------------------------


def _trial_prior_factory(alpha):
    return SpectralPrior(alpha=alpha)


def test_fixed_lambda_passthrough():
    val = FixedLambda(0.42)(
        design_matrix=np.zeros((4, 3)),
        data=np.zeros(4),
        alpha=0.0,
        prior_factory=_trial_prior_factory,
    )
    assert val == 0.42


def test_gcv_returns_positive_finite_lambda():
    M, rng = _simple_design(n_points=80, n_modes=15)
    y = M @ rng.normal(size=15) + 0.05 * rng.normal(size=80)
    lam = GCVSelector()(
        M, y, alpha=0.0, prior_factory=_trial_prior_factory
    )
    assert np.isfinite(lam)
    assert lam > 0.0


def test_gcv_prefers_smaller_lambda_at_high_snr():
    """At very high SNR, GCV should prefer near-zero regularization."""
    M, rng = _simple_design(n_points=200, n_modes=20, seed=3)
    truth = rng.normal(size=20)
    y_clean = M @ truth
    y_noisy = y_clean + 0.5 * rng.normal(size=200)

    lam_high_snr = GCVSelector()(
        M, y_clean, alpha=0.0, prior_factory=_trial_prior_factory
    )
    lam_low_snr = GCVSelector()(
        M, y_noisy, alpha=0.0, prior_factory=_trial_prior_factory
    )
    # High-SNR lambda should be no larger than low-SNR lambda.
    # Loose bound — GCV grids are coarse — but the ordering should hold.
    assert lam_high_snr <= 1.5 * lam_low_snr or lam_high_snr < 1e-4


def test_marginal_likelihood_returns_positive_finite_lambda():
    M, rng = _simple_design(n_points=80, n_modes=15, seed=5)
    y = M @ rng.normal(size=15) + 0.05 * rng.normal(size=80)
    lam = MarginalLikelihoodSelector()(
        M, y, alpha=0.0, prior_factory=_trial_prior_factory
    )
    assert np.isfinite(lam)
    assert lam > 0.0


def test_marginal_likelihood_agrees_with_gcv_in_gaussian_regime():
    """Under homoscedastic Gaussian noise the two selectors should agree
    within ~half a decade. (Diverge under heteroscedastic / non-Gaussian
    residuals — that's the documented regime where MarginalLikelihood
    is preferred.)"""
    M, rng = _simple_design(n_points=400, n_modes=30, seed=9)
    truth = rng.normal(size=30)
    y = M @ truth + 0.1 * rng.normal(size=400)
    lam_gcv = GCVSelector()(
        M, y, alpha=0.0, prior_factory=_trial_prior_factory
    )
    lam_ml = MarginalLikelihoodSelector()(
        M, y, alpha=0.0, prior_factory=_trial_prior_factory
    )
    # log10 ratio within 0.7 ≈ within factor of 5. Loose because both
    # selectors have edge-case behavior near small lambda.
    assert abs(np.log10(lam_gcv) - np.log10(lam_ml)) < 0.7


def test_spatial_cv_selector_returns_positive_finite_lambda():
    rng = np.random.default_rng(13)
    n_points = 200
    M = rng.normal(size=(n_points, 20))
    y = M @ rng.normal(size=20) + 0.1 * rng.normal(size=n_points)
    # Random 2D coords for the spatial fold builder
    coords = rng.uniform(0, 1, size=(n_points, 2))
    lam = SpatialCVSelector(coords=coords, n_folds=4)(
        M, y, alpha=0.0, prior_factory=_trial_prior_factory
    )
    assert np.isfinite(lam)
    assert lam > 0.0


# -----------------------------------------------------------------------------
# build_spectral_prior end-to-end
# -----------------------------------------------------------------------------


def test_build_spectral_prior_defaults_return_hyperparams_record():
    rng = np.random.default_rng(0)
    field = _planted_power_law_field((64, 64), beta=2.0, rng=rng)
    M, _ = _simple_design(n_points=field.size, n_modes=30, seed=4)
    y = field.ravel() + 0.01 * rng.normal(size=field.size)
    hp = build_spectral_prior(field, M, y)
    assert isinstance(hp, Hyperparams)
    assert np.isfinite(hp.alpha)
    assert np.isfinite(hp.lmbda)
    assert hp.lmbda > 0.0
    assert isinstance(hp.prior, SpectralPrior)
    assert hp.prior.alpha == hp.alpha
    # Default uses EmpiricalSpectralSlope ⇒ slope_fit populated
    assert hp.slope_fit is not None
    assert hp.slope_fit.alpha == hp.alpha


def test_build_spectral_prior_with_fixed_alpha_skips_slope_fit():
    rng = np.random.default_rng(1)
    field = rng.normal(size=(48, 48))
    M, _ = _simple_design(n_points=field.size, n_modes=20, seed=2)
    y = field.ravel()
    hp = build_spectral_prior(
        field, M, y, alpha_selector=FixedAlpha(1.5)
    )
    assert hp.alpha == 1.5
    assert hp.slope_fit is None


def test_build_spectral_prior_with_fixed_lambda_returns_that_lambda():
    rng = np.random.default_rng(2)
    field = rng.normal(size=(32, 32))
    M, _ = _simple_design(n_points=field.size, n_modes=15, seed=3)
    y = field.ravel()
    hp = build_spectral_prior(
        field, M, y,
        alpha_selector=FixedAlpha(0.0),
        lambda_selector=FixedLambda(0.123),
    )
    assert hp.lmbda == 0.123


# -----------------------------------------------------------------------------
# JointGCVSelector
# -----------------------------------------------------------------------------


def test_joint_gcv_returns_finite_pair_within_grids():
    """Joint GCV should pick (alpha, lmbda) inside its grid bounds and
    return finite values."""
    M, rng = _simple_design(n_points=120, n_modes=20, seed=4)
    y = M @ rng.normal(size=20) + 0.05 * rng.normal(size=120)
    alpha, lmbda = JointGCVSelector(
        alpha_grid=np.array([0.0, 1.0, 2.0]),
    )(M, y)
    assert 0.0 <= alpha <= 2.0
    assert np.isfinite(lmbda)
    assert lmbda > 0.0


def test_joint_gcv_with_alpha_zero_only_matches_gcv_selector():
    """If the alpha grid contains only 0, JointGCV reduces to plain
    1-D GCV over lambda — should give a similar lambda as
    GCVSelector on the same data."""
    M, rng = _simple_design(n_points=160, n_modes=18, seed=5)
    y = M @ rng.normal(size=18) + 0.05 * rng.normal(size=160)
    grid = np.logspace(-6, 1, 21)
    _, lam_joint = JointGCVSelector(
        alpha_grid=np.array([0.0]), lambda_grid=grid,
    )(M, y)
    lam_solo = GCVSelector(lambda_grid=grid)(
        M, y, alpha=0.0, prior_factory=lambda a: SpectralPrior(alpha=a),
    )
    # Both run on the same eigendecomp + grid — should match exactly.
    np.testing.assert_allclose(lam_joint, lam_solo)


def test_build_spectral_prior_joint_mode_returns_hyperparams():
    rng = np.random.default_rng(6)
    field = rng.normal(size=(40, 40))
    M, _ = _simple_design(n_points=field.size, n_modes=20, seed=7)
    y = field.ravel() + 0.01 * rng.normal(size=field.size)
    hp = build_spectral_prior(
        field, M, y,
        joint_selector=JointGCVSelector(
            alpha_grid=np.array([0.0, 1.0, 2.0]),
        ),
    )
    assert isinstance(hp, Hyperparams)
    assert hp.slope_fit is None  # joint mode skips the slope fit
    assert hp.alpha in (0.0, 1.0, 2.0)
    assert np.isfinite(hp.lmbda) and hp.lmbda > 0.0
    assert hp.prior.alpha == hp.alpha
