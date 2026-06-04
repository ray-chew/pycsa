"""Hyperparameter selection for the structured Tikhonov prior.

Phase 2 of the kernel-spike work introduces a separation between the
two hyperparameters that ``SpectralPrior`` carries:

- ``alpha`` — the spectral decay exponent. A structural belief about
  the topography: chosen from the input field's empirical power-law
  slope, or fixed by the user.
- ``lmbda`` — the overall regularization scale. A data-driven scalar:
  selected by GCV, marginal likelihood, or spatial cross-validation.

Two protocols (:class:`AlphaSelector`, :class:`LambdaSelector`) expose
these as pluggable extension points. Concrete strategies are provided
with documented defaults: :class:`EmpiricalSpectralSlope` for alpha,
:class:`GCVSelector` for lambda. The convenience constructor
:func:`build_spectral_prior` wires them in the canonical order
(alpha first, then lambda given alpha) and returns an explicit
:class:`Hyperparams` record — no silent override of the kwarg-level
``lmbda`` that :func:`pycsa.core.lin_reg.do` already accepts.

**Honest framing on defaults.** ``EmpiricalSpectralSlope`` is grounded
in the fact that topography spectra are typically well-approximated
by a power law over a resolved-but-not-aliased wavenumber band; it
returns the empirical slope plus its standard error, so the
uncertainty is observable. ``GCVSelector`` is the textbook
closed-form-LOO surrogate (Golub, Heath, and Wahba, 1979). Neither
choice has been benchmarked against alternatives on the project's
reproducibility fixtures yet — a one-script empirical check at
``scripts/validate_hyperparam_defaults.py`` performs that comparison
without being a full sweep.

**Composition with sparse selection (mode_selection.py).** The
recommended pattern is: tune ``(alpha, lmbda)`` with the FA
``GreedyArgmax`` selector, then fix the resulting prior and switch
to a sparsity-inducing selector for SA. Do *not* jointly tune
``(alpha, lmbda, selector)``; the search space explodes and the
selectors interact with the prior through the SA basis, not the
FA basis where ``alpha``/``lmbda`` are calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Protocol

import numpy as np

from pycsa.core.priors import SpectralPrior

# -----------------------------------------------------------------------------
# Records and protocols
# -----------------------------------------------------------------------------


class SlopeFit(NamedTuple):
    """Output of :class:`EmpiricalSpectralSlope`.

    ``alpha`` is the positive power-law slope (``power ∝ ‖k‖^(-alpha)``).
    ``stderr`` is the standard error from the linear regression in
    log-log space. ``r_squared`` is the regression's coefficient of
    determination — values much below ~0.9 mean the spectrum is not
    well approximated by a single power law and downstream selectors
    should be re-examined.
    """

    alpha: float
    stderr: float
    r_squared: float


@dataclass
class Hyperparams:
    """Explicit (alpha, lmbda, prior) bundle returned by ``build_spectral_prior``.

    Use the fields directly when calling ``lin_reg.do``::

        hp = build_spectral_prior(topography, fobj)
        a_m, recons = lin_reg.do(fobj, cell, lmbda=hp.lmbda, prior=hp.prior)

    Both ``lmbda`` and ``prior`` are needed: the prior knows ``alpha``
    and the per-mode shape, but ``lin_reg.do``'s ``lmbda`` kwarg is the
    overall scale that the prior multiplies into. Passing one without
    the other defeats the selection.
    """

    alpha: float
    lmbda: float
    prior: SpectralPrior
    slope_fit: Optional[SlopeFit] = None


class AlphaSelector(Protocol):
    """Chooses the spectral decay exponent ``alpha`` from the input field."""

    def __call__(self, topography: np.ndarray) -> float | SlopeFit: ...


class LambdaSelector(Protocol):
    """Chooses the regularization scale ``lmbda`` given the design matrix and alpha.

    Parameters
    ----------
    design_matrix
        Dense ``M`` matrix, shape ``(n_points, n_modes)``. Most selectors
        operate on the normal-equations form ``MᵀM`` internally.
    data
        Target vector, shape ``(n_points,)``.
    alpha
        The chosen ``alpha`` (used by selectors that build trial priors).
    prior_factory
        Callable ``alpha -> Prior`` for selectors that need to instantiate
        trial priors at the candidate ``lmbda``. The factory should
        produce a prior parameterized only by ``alpha``; ``lmbda`` is
        the kwarg passed at call time by the selector.

    Returns
    -------
    lmbda : float
        The selected regularization scale.
    """

    def __call__(
        self,
        design_matrix: np.ndarray,
        data: np.ndarray,
        *,
        alpha: float,
        prior_factory: Callable[[float], SpectralPrior],
    ) -> float: ...


# -----------------------------------------------------------------------------
# Alpha selectors
# -----------------------------------------------------------------------------


@dataclass
class FixedAlpha:
    """Pass-through alpha selector. No default — caller must specify.

    Use this when the user has a principled reason to fix ``alpha``
    (e.g. matching a published spectrum estimate, debugging, or
    side-by-side comparison with another tool). The lack of a default
    value is deliberate: if you don't know what ``alpha`` should be,
    use :class:`EmpiricalSpectralSlope` instead.
    """

    value: float

    def __call__(self, topography: np.ndarray) -> float:
        return float(self.value)


@dataclass
class EmpiricalSpectralSlope:
    """Documented default alpha selector. Fits a power law to the
    radially-averaged 2D periodogram of the input topography.

    Returns a :class:`SlopeFit` carrying ``alpha`` plus its standard
    error and R². Downstream callers can inspect the standard error to
    decide whether a single-power-law model is appropriate, or use it
    as a sweep width when building sensitivity tests.

    Parameters
    ----------
    k_min_frac, k_max_frac
        Lower and upper bound of the wavenumber band over which the
        power law is fit, expressed as a fraction of Nyquist. Defaults
        exclude the very-low-wavenumber region (poorly resolved on a
        finite cell) and the near-Nyquist region (aliasing-prone).
        These knobs *are* themselves hyperparameters and the returned
        ``stderr`` partly reflects sensitivity to them — perturb and
        re-fit if you want to bound that.
    n_bins
        Number of radial bins for the periodogram. Default 32 trades
        off variance per bin against resolution.
    """

    k_min_frac: float = 0.02
    k_max_frac: float = 0.5
    n_bins: int = 32

    def __call__(self, topography: np.ndarray) -> SlopeFit:
        if topography.ndim != 2:
            raise ValueError(
                "EmpiricalSpectralSlope expects a 2D topography field; "
                f"got ndim={topography.ndim}"
            )
        ny, nx = topography.shape
        # Subtract mean to avoid a giant DC spike dominating the radial average.
        field = topography - float(np.mean(topography))
        power = np.abs(np.fft.fft2(field)) ** 2 / (ny * nx)

        # Build a radial wavenumber grid in cycles/sample.
        ky = np.fft.fftfreq(ny)[:, None]
        kx = np.fft.fftfreq(nx)[None, :]
        kmag = np.hypot(ky, kx)
        # Mask to the resolved-but-not-aliased band.
        nyq = 0.5
        lo, hi = self.k_min_frac * nyq, self.k_max_frac * nyq
        in_band = (kmag >= lo) & (kmag <= hi) & np.isfinite(power) & (power > 0)
        if not np.any(in_band):
            raise ValueError(
                "No periodogram samples in the requested band "
                f"[{lo:.4g}, {hi:.4g}] cycles/sample — check k_min_frac/k_max_frac"
            )
        # Radially bin in log-k.
        log_k = np.log(kmag[in_band])
        log_P = np.log(power[in_band])
        bins = np.linspace(log_k.min(), log_k.max(), self.n_bins + 1)
        which = np.digitize(log_k, bins) - 1
        which = np.clip(which, 0, self.n_bins - 1)
        bin_log_k = np.full(self.n_bins, np.nan)
        bin_log_P = np.full(self.n_bins, np.nan)
        for b in range(self.n_bins):
            sel = which == b
            if np.any(sel):
                bin_log_k[b] = log_k[sel].mean()
                bin_log_P[b] = log_P[sel].mean()
        keep = np.isfinite(bin_log_k) & np.isfinite(bin_log_P)
        if int(np.sum(keep)) < 3:
            raise ValueError(
                "Fewer than 3 populated bins — increase k_max_frac or n_bins"
            )
        # Linear fit in log-log space.
        from scipy.stats import linregress

        result = linregress(bin_log_k[keep], bin_log_P[keep])
        # P ~ k^(-alpha) ⇒ slope = -alpha
        alpha = float(-result.slope)
        stderr = float(result.stderr)
        r2 = float(result.rvalue**2)
        return SlopeFit(alpha=alpha, stderr=stderr, r_squared=r2)


# -----------------------------------------------------------------------------
# Lambda selectors
# -----------------------------------------------------------------------------


def _default_lambda_grid(design_matrix: np.ndarray) -> np.ndarray:
    """Log-spaced grid scaled by ``trace(MᵀM)/N``.

    Matches the scale of the existing scalar-trace branch so the grid
    spans the regime where ``lin_reg.do`` historically operates.
    """
    n = design_matrix.shape[1]
    trace = float(np.einsum("ij,ij->", design_matrix, design_matrix)) / max(n, 1)
    return np.logspace(-8, 2, 41) * max(trace, 1e-12)


@dataclass
class FixedLambda:
    """Pass-through lambda selector. No default — caller must specify.

    Mirrors :class:`FixedAlpha`. Use when ``lmbda`` is known a priori
    (e.g. matching the existing scalar default in ``lin_reg.do``).
    """

    value: float

    def __call__(self, design_matrix, data, *, alpha, prior_factory) -> float:
        return float(self.value)


@dataclass
class GCVSelector:
    """Cheap closed-form lambda selector. Generalized cross-validation.

    For each candidate lambda on a log-spaced grid, computes::

        GCV(lambda) = || y - M â(lambda) ||² / (n - tr(H(lambda)))²

    where ``â(lambda)`` solves the regularized normal equations and
    ``H(lambda) = M (MᵀM + Λ(lambda))⁻¹ Mᵀ`` is the hat matrix. Returns
    the lambda that minimizes GCV.

    Closed-form approximation to leave-one-out cross-validation
    (Golub, Heath, Wahba 1979). Cheap, well-behaved on this problem
    class, no held-out machinery required. The implicit assumption is
    that the prior form (the structured ``Λ`` shape) is approximately
    correct; if you don't trust the prior form, use
    :class:`SpatialCVSelector` instead.

    Parameters
    ----------
    lambda_grid
        Candidate lambdas. If ``None``, falls back to a 41-point
        log-spaced grid scaled by ``trace(MᵀM)/N``.
    """

    lambda_grid: Optional[np.ndarray] = None

    def __call__(self, design_matrix, data, *, alpha, prior_factory) -> float:
        M = np.asarray(design_matrix, dtype=float)
        y = np.asarray(data, dtype=float).reshape(-1)
        n = M.shape[0]
        # Eigendecomposition of MᵀM lets us evaluate every candidate
        # lambda in O(N) per candidate instead of O(N³). The hat trace
        # and residual norm both reduce to sums over eigenvalues.
        E = M.T @ M
        # MᵀM is symmetric PSD; use eigh.
        eigvals, eigvecs = np.linalg.eigh(E)
        Mt_y = M.T @ y  # shape (N,)
        proj = eigvecs.T @ Mt_y  # shape (N,)
        y_norm2 = float(y @ y)
        prior = prior_factory(alpha)
        # Trial-prior diagonal: build a representative diag at lmbda=1
        # then scale it linearly per candidate (the spec defines
        # Λ_m ∝ lmbda).
        unit_diag = np.asarray(prior(fobj=None, E_tilda_lm=E, lmbda=1.0), dtype=float)
        # GCV does not care about the eigenvector basis for Λ — we
        # approximate Λ as diagonal in the eigenbasis of MᵀM by taking
        # its mean. This is exact for IsotropicPrior and a good
        # approximation for SpectralPrior on regular Fourier grids.
        lambda_grid = (
            self.lambda_grid
            if self.lambda_grid is not None
            else _default_lambda_grid(M)
        )
        scores = np.full_like(lambda_grid, np.inf, dtype=float)
        for i, lam in enumerate(lambda_grid):
            shift = float(np.mean(unit_diag * lam))
            denom = eigvals + shift
            if np.any(denom <= 0):
                continue
            # â projected onto eigenbasis: proj * eigvals / denom; then
            # residual² = ||y||² - 2 yᵀM â + âᵀ MᵀM â reduces to:
            ratio = eigvals / denom
            residual2 = y_norm2 - float(np.sum((proj**2) * (2 * ratio - ratio**2)))
            trH = float(np.sum(ratio))
            denom_gcv = (n - trH) ** 2
            if denom_gcv <= 0:
                continue
            scores[i] = max(residual2, 0.0) / denom_gcv
        return float(lambda_grid[int(np.argmin(scores))])


@dataclass
class MarginalLikelihoodSelector:
    """Empirical-Bayes lambda selector via type-II MLE.

    Wraps ``sklearn.linear_model.BayesianRidge``, which performs the
    standard MacKay-1992 evidence-approximation fixed-point iteration
    jointly over the noise precision ``α`` and prior precision ``λ``
    with weak Gamma hyperpriors. The returned scalar is the effective
    ridge weight :math:`\\lambda_{\\text{eff}} = \\lambda / \\alpha`
    (sklearn's convention), which is what ``lin_reg.do``'s ``lmbda``
    kwarg expects.

    **When this differs from GCV.** GCV approximates leave-one-out
    CV under the implicit assumption of homoscedastic Gaussian noise.
    Marginal likelihood agrees with GCV in that regime. The two
    diverge when (a) the residual distribution is heavily non-Gaussian
    (heavy tails, skew — common with topography in coastal/glacial
    cells), or (b) the noise variance varies systematically with
    location (heteroscedasticity from data-source mixing — e.g. MERIT
    coastal masking). Prefer this selector over GCV in those regimes;
    otherwise GCV is cheaper and produces equivalent results.

    **Limitation.** ``BayesianRidge`` assumes an isotropic prior on
    the coefficients. For ``SpectralPrior(alpha != 0)`` this selector
    returns the scalar overall scale ``lmbda``; the per-mode shape
    still comes from the ``Prior`` callable. If you need per-mode
    marginal likelihood, use :class:`SpatialCVSelector` instead.

    Parameters
    ----------
    max_iter, tol
        Passed through to ``BayesianRidge``.
    """

    max_iter: int = 300
    tol: float = 1e-3

    def __call__(self, design_matrix, data, *, alpha, prior_factory) -> float:
        try:
            from sklearn.linear_model import BayesianRidge
        except ImportError as exc:
            raise ImportError(
                "MarginalLikelihoodSelector requires scikit-learn. "
                "Install it or use GCVSelector."
            ) from exc
        M = np.asarray(design_matrix, dtype=float)
        y = np.asarray(data, dtype=float).reshape(-1)
        br = BayesianRidge(
            alpha_init=1.0,
            lambda_init=1.0,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=False,
            compute_score=False,
        )
        br.fit(M, y)
        # sklearn's convention: alpha_ = 1/σ², lambda_ = prior precision.
        # Effective ridge weight in the MAP objective is lambda_/alpha_.
        return float(br.lambda_ / max(br.alpha_, 1e-12))


@dataclass
class SpatialCVSelector:
    """Lambda selector via spatial k-fold cross-validation.

    Partitions the rows of ``design_matrix`` into spatial patches with
    a buffer zone (see :func:`pycsa.core.validation.spatial_cv_score`
    for the patch geometry), fits the prior at each candidate
    ``lmbda`` on the training rows, and evaluates reconstruction MSE
    on the held-out patch's rows. The lambda with the smallest mean
    held-out MSE wins.

    **Why this is the recommended selector for topography.** Real
    topography residuals are spatially correlated, which breaks the
    i.i.d. assumption GCV / marginal likelihood rely on — those
    selectors then under-regularize. Spatial CV evaluates held-out
    *patches*, so its notion of "out-of-sample" matches how the fit is
    actually used on a constrained cell, and it is the only selector
    here that *detects* misspecified priors: if GCV and SpatialCV pick
    wildly different ``lmbda`` values, the prior form is doing more
    work than it should be, and the user should revisit ``alpha`` (or
    pick a different prior altogether). It is the default in
    :func:`build_spectral_prior` whenever per-row ``coords`` are
    supplied. (Note pyCSA's production runs do not invoke this
    selection API at all — they use a hand-tuned ``lmbda`` baseline.)

    Parameters
    ----------
    coords
        Row coordinates as a ``(n_points, 2)`` array of
        ``(x, y)`` pairs in any consistent metric — used by
        :func:`spatial_cv_score` to build patches. If ``None``,
        falls back to row-index splitting (only sensible if rows are
        already in geographic order).
    n_folds
        Number of spatial folds. Default 5.
    buffer_fraction
        Half-width of the buffer zone around each patch as a
        fraction of patch size. Default 0.1.
    lambda_grid
        Same fallback as :class:`GCVSelector`.
    rng_seed
        Seed for reproducible fold assignment.
    """

    coords: Optional[np.ndarray] = None
    n_folds: int = 5
    buffer_fraction: float = 0.1
    lambda_grid: Optional[np.ndarray] = None
    rng_seed: Optional[int] = None

    def __call__(self, design_matrix, data, *, alpha, prior_factory) -> float:
        from pycsa.core.validation import spatial_cv_score

        M = np.asarray(design_matrix, dtype=float)
        y = np.asarray(data, dtype=float).reshape(-1)
        lambda_grid = (
            self.lambda_grid
            if self.lambda_grid is not None
            else _default_lambda_grid(M)
        )
        prior = prior_factory(alpha)
        scores = np.full_like(lambda_grid, np.inf, dtype=float)
        for i, lam in enumerate(lambda_grid):
            cv = spatial_cv_score(
                prior=prior,
                lmbda=float(lam),
                design_matrix=M,
                data=y,
                coords=self.coords,
                n_folds=self.n_folds,
                buffer_fraction=self.buffer_fraction,
                rng_seed=self.rng_seed,
            )
            scores[i] = cv["mean_heldout_mse"]
        return float(lambda_grid[int(np.argmin(scores))])


# -----------------------------------------------------------------------------
# Joint (alpha, lambda) selector
# -----------------------------------------------------------------------------


@dataclass
class JointGCVSelector:
    """Joint 2-D GCV minimization over (alpha, lmbda) for SpectralPrior.

    Unlike :class:`GCVSelector` (which picks ``lmbda`` only, with ``alpha``
    set externally — usually by ``EmpiricalSpectralSlope`` from the
    input periodogram), this selector lets the GCV objective pick
    *both* hyperparameters. The motivation: the periodogram fit can
    pick an ``alpha`` that's too aggressive on real cells (south_pole
    α ≈ 10 over-regularized signal-bearing high-k modes); a
    held-out-error-driven search will usually pick something milder
    or zero, when that's empirically better.

    The search is a 2-D grid (``alpha_grid × lambda_grid``). One
    eigendecomposition of ``MᵀM`` is reused across all candidates;
    each evaluation is O(N) given the eigenbasis. Cost is therefore
    ``|alpha_grid|`` × the cost of plain 1-D GCV.

    **Approximation note.** Like :class:`GCVSelector`, the score is
    computed by treating the prior's per-mode diagonal in the
    eigenbasis of ``MᵀM`` (approximated by its mean, exact for
    isotropic priors, good for slowly-varying structured priors on
    regular Fourier grids). The chosen ``alpha`` still flows through
    properly when the resulting :class:`SpectralPrior` is used in
    ``lin_reg.do``, where the full per-mode diagonal IS applied — so
    the GCV ranking is approximate but the post-selection fit is
    exact.

    Parameters
    ----------
    alpha_grid
        Candidate exponents. Default ``[0, 0.5, 1, 1.5, 2, 3]`` —
        spans isotropic through "more aggressive than typical
        atmospheric / topographic spectra." ``0`` reduces to plain
        isotropic GCV, recovered automatically when that's best.
    lambda_grid
        As :class:`GCVSelector`. ``None`` falls back to a 41-point
        log-spaced grid scaled by ``trace(MᵀM)/N``.
    eps
        DC-mode floor passed to :class:`SpectralPrior`.
    """

    alpha_grid: Optional[np.ndarray] = None
    lambda_grid: Optional[np.ndarray] = None
    eps: float = 1e-3

    def __call__(
        self,
        design_matrix: np.ndarray,
        data: np.ndarray,
    ) -> tuple[float, float]:
        M = np.asarray(design_matrix, dtype=float)
        y = np.asarray(data, dtype=float).reshape(-1)
        n = M.shape[0]

        alpha_grid = (
            np.asarray(self.alpha_grid)
            if self.alpha_grid is not None
            else np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        )
        lambda_grid = (
            self.lambda_grid
            if self.lambda_grid is not None
            else _default_lambda_grid(M)
        )

        E = M.T @ M
        eigvals, eigvecs = np.linalg.eigh(E)
        proj = eigvecs.T @ (M.T @ y)
        y_norm2 = float(y @ y)

        best_score = np.inf
        best_alpha = float(alpha_grid[0])
        best_lmbda = float(lambda_grid[0])

        for alpha in alpha_grid:
            prior = SpectralPrior(alpha=float(alpha), eps=self.eps)
            unit_diag = np.asarray(
                prior(fobj=None, E_tilda_lm=E, lmbda=1.0), dtype=float
            )
            unit_shift_mean = float(np.mean(unit_diag))
            for lam in lambda_grid:
                shift = unit_shift_mean * float(lam)
                denom = eigvals + shift
                if np.any(denom <= 0):
                    continue
                ratio = eigvals / denom
                residual2 = max(
                    y_norm2 - float(np.sum((proj**2) * (2 * ratio - ratio**2))),
                    0.0,
                )
                trH = float(np.sum(ratio))
                denom_gcv = (n - trH) ** 2
                if denom_gcv <= 0:
                    continue
                score = residual2 / denom_gcv
                if score < best_score:
                    best_score = score
                    best_alpha = float(alpha)
                    best_lmbda = float(lam)
        return best_alpha, best_lmbda


# -----------------------------------------------------------------------------
# Convenience constructor
# -----------------------------------------------------------------------------


def build_spectral_prior(
    topography: np.ndarray,
    design_matrix: np.ndarray,
    data: np.ndarray,
    *,
    coords: Optional[np.ndarray] = None,
    alpha_selector: Optional[AlphaSelector] = None,
    lambda_selector: Optional[LambdaSelector] = None,
    joint_selector: Optional[JointGCVSelector] = None,
    eps: float = 1e-3,
) -> Hyperparams:
    """One-call construction of a fully-specified ``SpectralPrior``.

    Two modes:

    - **Sequential (default):** alpha selected first (from the
      ``topography`` field), then lambda selected given alpha (from
      ``design_matrix``, ``data``). Pass ``alpha_selector`` and/or
      ``lambda_selector`` to override the defaults
      (``EmpiricalSpectralSlope`` + ``SpatialCVSelector`` when ``coords``
      are supplied, else ``GCVSelector``).
    - **Joint:** pass ``joint_selector=JointGCVSelector(...)`` to
      pick both alpha and lambda from one held-out-error
      optimization on ``(design_matrix, data)``. Overrides any
      sequential selectors. Recommended when ``EmpiricalSpectralSlope``
      gives an alpha you don't trust (e.g., on cells where the
      periodogram fit has low R² or yields very steep slopes).

    Parameters
    ----------
    topography
        2D topography field used only for alpha selection in the
        sequential mode. Ignored when ``joint_selector`` is set.
    design_matrix
        Dense ``M`` matrix, shape ``(n_points, n_modes)``.
    data
        Target vector that the LSQ fit targets. Typically
        ``cell.topo_m``.
    coords
        Per-row ``(n_points, 2)`` spatial coordinates of the
        ``design_matrix`` rows — typically
        ``np.column_stack([cell.lon_m, cell.lat_m])``. When supplied,
        the default lambda selector is :class:`SpatialCVSelector`
        (spatial k-fold CV, the recommended choice for spatially-
        correlated topography); when omitted it falls back to
        :class:`GCVSelector`. Ignored if ``lambda_selector`` is given.
    alpha_selector, lambda_selector
        Sequential-mode selector overrides. Pass ``FixedAlpha`` /
        ``FixedLambda`` to short-circuit either step. Ignored when
        ``joint_selector`` is set.
    joint_selector
        Joint-mode override. Currently :class:`JointGCVSelector` is
        the only implementation.
    eps
        Passed through to :class:`SpectralPrior` (DC-mode floor).

    Returns
    -------
    Hyperparams
        Bundle of ``alpha``, ``lmbda``, ``prior``, and (only in
        sequential mode with ``EmpiricalSpectralSlope``)
        ``slope_fit``. Thread both ``lmbda`` and ``prior`` into
        ``lin_reg.do`` — passing one without the other defeats the
        selection.
    """
    if joint_selector is not None:
        alpha, lmbda = joint_selector(design_matrix, data)
        return Hyperparams(
            alpha=float(alpha),
            lmbda=float(lmbda),
            prior=SpectralPrior(alpha=float(alpha), eps=eps),
            slope_fit=None,
        )

    alpha_selector = alpha_selector or EmpiricalSpectralSlope()
    if lambda_selector is None:
        # Default to spatial CV when per-row coords are available — it is the
        # only selector here that respects spatial correlation in the residual
        # (GCV / marginal likelihood assume i.i.d. rows and under-regularize
        # spatially-correlated topography). Without coords a spatial split is
        # meaningless, so fall back to the closed-form GCV surrogate.
        lambda_selector = (
            SpatialCVSelector(coords=coords) if coords is not None else GCVSelector()
        )

    alpha_result = alpha_selector(topography)
    if isinstance(alpha_result, SlopeFit):
        alpha = alpha_result.alpha
        slope_fit: Optional[SlopeFit] = alpha_result
    else:
        alpha = float(alpha_result)
        slope_fit = None

    def factory(a: float) -> SpectralPrior:
        return SpectralPrior(alpha=a, eps=eps)

    lmbda = lambda_selector(design_matrix, data, alpha=alpha, prior_factory=factory)
    return Hyperparams(
        alpha=alpha,
        lmbda=float(lmbda),
        prior=SpectralPrior(alpha=alpha, eps=eps),
        slope_fit=slope_fit,
    )


__all__ = [
    "SlopeFit",
    "Hyperparams",
    "AlphaSelector",
    "LambdaSelector",
    "FixedAlpha",
    "EmpiricalSpectralSlope",
    "FixedLambda",
    "GCVSelector",
    "MarginalLikelihoodSelector",
    "SpatialCVSelector",
    "JointGCVSelector",
    "build_spectral_prior",
]
