"""Unit tests for pycsa.core.validation.spatial_cv_score.

Cover fold construction (no leakage through buffer; balanced sizes;
raises on too-sparse or zero-extent coords) and the scoring loop
(finite per-fold MSE; mean = mean(per_fold)).
"""

import numpy as np
import pytest

from pycsa.core.priors import IsotropicPrior, SpectralPrior
from pycsa.core.validation import _build_spatial_folds, spatial_cv_score


def _grid_coords(n_per_side):
    """Regular n×n coord grid in [0, 1] × [0, 1]."""
    xs = np.linspace(0.0, 1.0, n_per_side)
    ys = np.linspace(0.0, 1.0, n_per_side)
    XY = np.array(np.meshgrid(xs, ys, indexing="ij"))
    return XY.reshape(2, -1).T  # shape (n_per_side**2, 2)


def test_build_spatial_folds_returns_disjoint_eval_sets():
    """No coord index appears in more than one fold's eval set."""
    coords = _grid_coords(20)
    folds = _build_spatial_folds(coords, n_folds=4, buffer_fraction=0.0, rng_seed=0)
    seen = set()
    for _, eval_idx in folds:
        eval_set = set(eval_idx.tolist())
        assert seen.isdisjoint(eval_set), "fold eval sets overlap"
        seen |= eval_set


def test_build_spatial_folds_buffer_excludes_from_both():
    """A non-zero buffer means buffer points are absent from both
    train_idx and eval_idx of that fold."""
    coords = _grid_coords(30)
    folds = _build_spatial_folds(coords, n_folds=4, buffer_fraction=0.1, rng_seed=1)
    for train_idx, eval_idx in folds:
        # train ∩ eval is empty
        assert not (set(train_idx.tolist()) & set(eval_idx.tolist()))
        # train ∪ eval covers fewer than all coords (buffer was excised)
        n = coords.shape[0]
        assert len(set(train_idx.tolist()) | set(eval_idx.tolist())) < n


def test_build_spatial_folds_returns_n_folds():
    coords = _grid_coords(15)
    folds = _build_spatial_folds(coords, n_folds=5, buffer_fraction=0.05, rng_seed=2)
    assert len(folds) == 5
    for train_idx, eval_idx in folds:
        assert train_idx.size > 1
        assert eval_idx.size > 0


def test_build_spatial_folds_raises_on_zero_extent():
    coords = np.column_stack([np.zeros(20), np.linspace(0, 1, 20)])
    with pytest.raises(ValueError, match="zero extent"):
        _build_spatial_folds(coords, n_folds=4, buffer_fraction=0.1, rng_seed=0)


def test_build_spatial_folds_raises_on_too_few_points():
    coords = _grid_coords(2)  # 4 points
    with pytest.raises(ValueError, match="at least"):
        _build_spatial_folds(coords, n_folds=5, buffer_fraction=0.0, rng_seed=0)


def test_spatial_cv_score_basic_shape():
    """Returns the documented dict structure with finite numbers."""
    rng = np.random.default_rng(0)
    coords = _grid_coords(20)  # 400 points
    n = coords.shape[0]
    M = rng.normal(size=(n, 25))
    truth = rng.normal(size=25)
    y = M @ truth + 0.1 * rng.normal(size=n)

    result = spatial_cv_score(
        prior=IsotropicPrior(),
        lmbda=0.1,
        design_matrix=M,
        data=y,
        coords=coords,
        n_folds=4,
        buffer_fraction=0.05,
        rng_seed=42,
    )
    assert "per_fold_mse" in result
    assert "mean_heldout_mse" in result
    assert "fold_sizes" in result
    assert result["per_fold_mse"].shape == (4,)
    assert np.all(np.isfinite(result["per_fold_mse"]))
    np.testing.assert_allclose(
        result["mean_heldout_mse"], float(np.mean(result["per_fold_mse"]))
    )
    # fold_sizes is (n_train, n_eval) per fold; both should be > 0
    assert np.all(result["fold_sizes"] > 0)


def test_spatial_cv_score_isotropic_vs_strong_regularization():
    """Stronger regularization should *not* make held-out MSE arbitrarily
    smaller — the validation is supposed to detect over-regularization.
    """
    rng = np.random.default_rng(0)
    coords = _grid_coords(20)
    n = coords.shape[0]
    M = rng.normal(size=(n, 25))
    truth = rng.normal(size=25)
    y = M @ truth + 0.05 * rng.normal(size=n)

    mse_light = spatial_cv_score(
        prior=IsotropicPrior(),
        lmbda=1e-4,
        design_matrix=M,
        data=y,
        coords=coords,
        n_folds=4,
        buffer_fraction=0.05,
        rng_seed=0,
    )["mean_heldout_mse"]
    mse_heavy = spatial_cv_score(
        prior=IsotropicPrior(),
        lmbda=1e3,
        design_matrix=M,
        data=y,
        coords=coords,
        n_folds=4,
        buffer_fraction=0.05,
        rng_seed=0,
    )["mean_heldout_mse"]
    # Heavy regularization → underfit → larger held-out MSE.
    assert mse_heavy > mse_light


def test_spatial_cv_score_works_with_spectral_prior():
    """SpectralPrior should be a valid plug-in. Smoke test only."""
    rng = np.random.default_rng(0)
    coords = _grid_coords(15)
    n = coords.shape[0]
    M = rng.normal(size=(n, 20))
    y = M @ rng.normal(size=20) + 0.05 * rng.normal(size=n)
    result = spatial_cv_score(
        prior=SpectralPrior(alpha=1.5),
        lmbda=0.01,
        design_matrix=M,
        data=y,
        coords=coords,
        n_folds=4,
        buffer_fraction=0.05,
        rng_seed=1,
    )
    assert np.all(np.isfinite(result["per_fold_mse"]))
