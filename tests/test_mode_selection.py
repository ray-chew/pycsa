"""Unit tests for pycsa.core.mode_selection.

Cover the protocol-level behavior of GreedyArgmax (bit-equivalent to
the inline loop), OMPSelector (planted-mode recovery on a deterministic
fixture), and LassoSelector (returns ≤ n_modes indices with finite
coefficients). End-to-end pipeline behavior is the gate's job, not
these tests'.
"""

import numpy as np
import pytest

from pycsa.core.mode_selection import (
    GreedyArgmax,
    LassoSelector,
    OMPSelector,
)


def _greedy_inline_loop(fa_spectrum, n_modes):
    """Verbatim replica of the inline loop at interface.py:556-571.

    Used as the ground truth for the GreedyArgmax parity test.
    """
    fq = np.copy(fa_spectrum)
    indices = []
    while len(indices) < n_modes:
        max_idx = np.unravel_index(fq.argmax(), fq.shape)
        indices.append(max_idx)
        fq[max_idx] = 0.0
    k_idxs = [pair[1] for pair in indices]
    l_idxs = [pair[0] for pair in indices]
    return k_idxs, l_idxs


def test_greedy_argmax_matches_inline_loop():
    """GreedyArgmax produces bit-identical (k_idxs, l_idxs) to the
    historical inline loop on a synthetic FA spectrum.
    """
    rng = np.random.default_rng(0)
    fa = rng.uniform(0, 1, size=(6, 8))  # (nhar_j, nhar_i) convention
    fa[2, 3] = 10.0  # plant a dominant mode
    fa[5, 1] = 7.5
    fa[0, 7] = 5.0

    k_inline, l_inline = _greedy_inline_loop(fa, n_modes=5)
    k_sel, l_sel = GreedyArgmax()(fa, n_modes=5)
    assert k_sel == k_inline
    assert l_sel == l_inline


def test_greedy_argmax_handles_uniform_spectrum():
    """When the spectrum is constant, argmax always picks the first
    flat element. The selector should follow numpy's tiebreaking,
    same as the inline loop.
    """
    fa = np.ones((3, 3))
    k_inline, l_inline = _greedy_inline_loop(fa, n_modes=4)
    k_sel, l_sel = GreedyArgmax()(fa, n_modes=4)
    assert (k_sel, l_sel) == (k_inline, l_inline)


def test_omp_recovers_planted_modes():
    """Plant 3 modes in a small Fourier design matrix and verify OMP
    recovers their column indices.

    Synthetic setup: M is a random Gaussian matrix (this stands in for
    the design matrix without going through f_trans); y = M @ a* + tiny
    noise, where a* has exactly 3 non-zero entries.
    """
    rng = np.random.default_rng(42)
    n_points = 200
    n_cols = 30
    M = rng.normal(size=(n_points, n_cols))
    M /= np.linalg.norm(M, axis=0, keepdims=True)  # unit-norm columns

    truth = np.zeros(n_cols)
    planted = [3, 11, 22]
    truth[planted] = [2.0, -1.5, 1.0]
    y = M @ truth + 1e-4 * rng.normal(size=n_points)

    # FA spectrum is unused by OMP — pass a placeholder shape
    # consistent with the spectrum convention. column_to_kl is provided
    # so we can verify the recovered indices directly.
    placeholder_spectrum = np.zeros((5, 6))  # nhar_j=5, nhar_i=6 → 30 cols
    recovered_columns: list[int] = []

    def column_to_kl(c):
        recovered_columns.append(c)
        return (c % 6, c // 6)

    k_idxs, l_idxs = OMPSelector()(
        placeholder_spectrum,
        n_modes=3,
        design_matrix=M,
        data=y,
        column_to_kl=column_to_kl,
    )
    assert sorted(recovered_columns) == sorted(planted)


def test_omp_raises_without_design_matrix():
    fa = np.zeros((4, 4))
    with pytest.raises(ValueError, match="design_matrix"):
        OMPSelector()(fa, n_modes=2)


def test_omp_batch_mode_returns_n_modes():
    """Batched OMP returns exactly n_modes columns, even if the batch
    size doesn't divide cleanly.
    """
    rng = np.random.default_rng(1)
    M = rng.normal(size=(80, 20))
    y = M[:, 3] + 0.5 * M[:, 7] + 1e-3 * rng.normal(size=80)
    fa = np.zeros((4, 5))
    k_idxs, l_idxs = OMPSelector(batch_size=3)(
        fa, n_modes=5, design_matrix=M, data=y
    )
    assert len(k_idxs) == 5
    assert len(l_idxs) == 5


def test_lasso_selector_returns_n_modes():
    """LassoSelector returns exactly n_modes indices on a synthetic
    problem with a clear sparse support.
    """
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    M = rng.normal(size=(120, 25))
    M /= np.linalg.norm(M, axis=0, keepdims=True)
    truth = np.zeros(25)
    truth[[2, 9, 15]] = [1.5, -0.8, 1.0]
    y = M @ truth + 1e-3 * rng.normal(size=120)
    fa = np.zeros((5, 5))
    k_idxs, l_idxs = LassoSelector()(
        fa, n_modes=3, design_matrix=M, data=y
    )
    assert len(k_idxs) == 3
    assert len(l_idxs) == 3


def test_lasso_selector_raises_without_design_matrix():
    pytest.importorskip("sklearn")
    fa = np.zeros((4, 4))
    with pytest.raises(ValueError, match="design_matrix"):
        LassoSelector()(fa, n_modes=2)
