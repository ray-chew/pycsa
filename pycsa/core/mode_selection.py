"""Pluggable mode selectors for the FA -> SA bridge.

The default selection step in :class:`pycsa.wrappers.interface.second_appx`
is a greedy ``argmax`` loop on the FA spectrum (lines 556-571): pick the
largest amplitude, zero it, repeat ``n_modes`` times. That loop is
preserved unchanged — it is what every existing call site hits and
what the reproducibility fixtures pin down.

This module introduces a ``ModeSelector`` protocol so spike scripts
can experiment with sparsity-inducing alternatives (OMP, Lasso)
without touching the default code path. A selector takes the FA
spectrum (and optionally the design matrix + data) and returns the
``(k_idxs, l_idxs)`` pair that ``fobj.set_kls`` expects.

``GreedyArgmax`` is a 1-to-1 reimplementation of the inline loop.
Unit-tested for bit-equivalent output on a synthetic FA spectrum.

``OMPSelector`` and ``LassoSelector`` are the alternatives under
test in the spike. They require access to the design matrix and
the residual signal — these are passed through optional kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

# k_idxs and l_idxs are passed to ``fobj.set_kls`` which currently
# accepts list-like inputs; we keep the return type as lists to
# match the existing call convention at interface.py:570-571.
IndexPair = Tuple[list, list]


class ModeSelector(Protocol):
    """Callable that selects ``n_modes`` Fourier modes from the FA spectrum.

    Parameters
    ----------
    fa_spectrum
        2D real array of FA amplitudes, shape ``(nhar_j, nhar_i)`` —
        the same array currently passed to the inline ``argmax``
        loop at interface.py:556-571. Indexing matches the existing
        convention: ``axis 0 == l``, ``axis 1 == k``.
    n_modes
        Number of modes to select.
    design_matrix
        Optional dense design matrix ``M`` of shape
        ``(n_points, n_columns)`` for selectors that need it (OMP,
        Lasso). ``GreedyArgmax`` ignores this.
    data
        Optional 1D data vector, shape ``(n_points,)``, that the
        FA fit targeted. Selectors that fit a sparse representation
        of ``data`` against the columns of ``design_matrix`` use it.
    column_to_kl
        Optional callable mapping column index in ``design_matrix``
        to a ``(k, l)`` index pair compatible with ``fa_spectrum``
        indexing. Used by OMP/Lasso to convert their selected
        columns back to the ``(k_idxs, l_idxs)`` format. If absent,
        the selector assumes columns are laid out in
        ``np.ndindex(fa_spectrum.shape)`` order.

    Returns
    -------
    (k_idxs, l_idxs)
        Two lists of length ``n_modes`` ready to feed
        ``fobj.set_kls(k_idxs, l_idxs, ...)``.
    """

    def __call__(
        self,
        fa_spectrum: np.ndarray,
        *,
        n_modes: int,
        design_matrix: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        column_to_kl=None,
    ) -> IndexPair: ...


@dataclass
class GreedyArgmax:
    """Reproduces the existing inline argmax loop verbatim.

    Iterates ``n_modes`` times: take ``argmax`` of the current
    spectrum, record the unravel-index pair, zero that entry,
    repeat. The recorded pairs are then split as
    ``k_idxs = [pair[1] for pair in indices]`` and
    ``l_idxs = [pair[0] for pair in indices]`` to match the
    convention at interface.py:570-571.

    This is the default selector. Passing ``selector=GreedyArgmax()``
    to a wrapped ``second_appx`` should produce bit-identical output
    versus the inline loop.
    """

    def __call__(
        self,
        fa_spectrum: np.ndarray,
        *,
        n_modes: int,
        design_matrix: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        column_to_kl=None,
    ) -> IndexPair:
        # Match the inline loop exactly: operate on a copy, zero entries
        # as they are picked, do not pre-mask anything.
        fq = np.copy(fa_spectrum)
        # NaN-handling is done by the caller (interface.second_appx.do
        # already zeros NaNs before passing the spectrum in). We do not
        # re-do it here so behavior remains identical.
        indices = []
        for _ in range(n_modes):
            max_idx = np.unravel_index(fq.argmax(), fq.shape)
            indices.append(max_idx)
            fq[max_idx] = 0.0
        k_idxs = [pair[1] for pair in indices]
        l_idxs = [pair[0] for pair in indices]
        return k_idxs, l_idxs


@dataclass
class OMPSelector:
    """Orthogonal matching pursuit on the design matrix.

    At each step, picks the column of ``design_matrix`` most
    correlated (in absolute value) with the current residual,
    re-fits the active set by ordinary least squares, and updates
    the residual. Repeats until ``n_modes`` columns are selected.

    Requires ``design_matrix`` and ``data``. Raises ``ValueError``
    if either is missing.

    Parameters
    ----------
    batch_size
        Number of correlations selected per step. ``batch_size=1``
        is canonical OMP. Larger values (e.g. 5) trade some
        optimality for cost — only one ``do_full`` + active-set
        solve per batch.
    column_offset
        Index into ``design_matrix`` columns to skip (e.g. the DC
        mode if the caller prefers OMP to ignore it). Defaults to
        0, meaning all columns are candidates.
    """

    batch_size: int = 1
    column_offset: int = 0

    def __call__(
        self,
        fa_spectrum: np.ndarray,
        *,
        n_modes: int,
        design_matrix: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        column_to_kl=None,
    ) -> IndexPair:
        if design_matrix is None or data is None:
            raise ValueError(
                "OMPSelector requires design_matrix and data kwargs; "
                "got design_matrix=%r, data=%r" % (design_matrix, data)
            )
        M = np.asarray(design_matrix, dtype=float)
        y = np.asarray(data, dtype=float).reshape(-1)
        if M.shape[0] != y.shape[0]:
            raise ValueError(
                f"design_matrix rows ({M.shape[0]}) and data length "
                f"({y.shape[0]}) disagree"
            )

        n_cols = M.shape[1]
        candidates = list(range(self.column_offset, n_cols))
        active: list[int] = []
        residual = y.copy()

        col_norms = np.linalg.norm(M[:, candidates], axis=0)
        col_norms[col_norms == 0.0] = 1.0  # avoid div-by-zero

        while len(active) < n_modes and candidates:
            # Correlation of each candidate column with the current residual,
            # normalized by column norm (cosine-like score).
            sub = M[:, candidates]
            scores = np.abs(sub.T @ residual) / col_norms
            step = min(self.batch_size, n_modes - len(active), len(candidates))
            top = np.argpartition(-scores, step - 1)[:step]
            picked = [candidates[i] for i in top]
            active.extend(picked)
            # Remove picked columns and their norms from the candidate pool.
            keep = [i for i in range(len(candidates)) if i not in set(top)]
            candidates = [candidates[i] for i in keep]
            col_norms = col_norms[keep]
            # Re-fit active set via lstsq, update residual.
            M_active = M[:, active]
            coef, *_ = np.linalg.lstsq(M_active, y, rcond=None)
            residual = y - M_active @ coef

        return _columns_to_kl(active, fa_spectrum.shape, column_to_kl)


@dataclass
class LassoSelector:
    """L1-penalised regression via coordinate descent.

    Wraps ``sklearn.linear_model.Lasso``. Selects the ``n_modes``
    largest-magnitude non-zero coefficients (pads with zeros if
    Lasso returns fewer non-zeros than ``n_modes``).

    Parameters
    ----------
    alpha
        Lasso penalty strength. ``None`` triggers a small k-fold CV
        over a log-spaced grid (left to the spike-script layer in
        phase 1; in the protocol layer here, ``alpha=None`` falls
        back to a sensible default of ``1e-3 * ‖Mᵀ y‖∞``).
    column_offset
        See :class:`OMPSelector`.
    max_iter
        Lasso solver max iterations.
    """

    alpha: Optional[float] = None
    column_offset: int = 0
    max_iter: int = 10_000

    def __call__(
        self,
        fa_spectrum: np.ndarray,
        *,
        n_modes: int,
        design_matrix: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        column_to_kl=None,
    ) -> IndexPair:
        if design_matrix is None or data is None:
            raise ValueError("LassoSelector requires design_matrix and data kwargs")
        try:
            from sklearn.linear_model import Lasso
        except ImportError as exc:
            raise ImportError(
                "LassoSelector requires scikit-learn. Install it or use a "
                "different selector."
            ) from exc

        M = np.asarray(design_matrix, dtype=float)
        y = np.asarray(data, dtype=float).reshape(-1)
        sub = M[:, self.column_offset :]

        if self.alpha is None:
            scale = float(np.max(np.abs(sub.T @ y))) / max(sub.shape[0], 1)
            alpha = 1e-3 * max(scale, 1e-12)
        else:
            alpha = float(self.alpha)

        model = Lasso(alpha=alpha, max_iter=self.max_iter, fit_intercept=False)
        model.fit(sub, y)
        coef = np.abs(model.coef_)
        # Take up to n_modes largest-magnitude indices among non-zeros;
        # if fewer non-zeros than n_modes, fill the remainder with the
        # next largest |coefficients| (still informative for the spike).
        order = np.argsort(-coef)
        picked = order[:n_modes]
        active = (picked + self.column_offset).tolist()
        return _columns_to_kl(active, fa_spectrum.shape, column_to_kl)


def _columns_to_kl(columns, spectrum_shape, column_to_kl) -> IndexPair:
    """Convert a list of design-matrix column indices into ``(k_idxs, l_idxs)``.

    If ``column_to_kl`` is provided, applies it. Otherwise assumes
    columns are laid out in ``np.ndindex(spectrum_shape)`` order —
    i.e. row-major over ``(l, k)`` to match the FA spectrum's
    ``(nhar_j, nhar_i)`` layout.
    """
    if column_to_kl is not None:
        pairs = [column_to_kl(c) for c in columns]
    else:
        nhar_j, nhar_i = spectrum_shape
        pairs = []
        for c in columns:
            # Same order as np.ndindex((nhar_j, nhar_i))
            l = c // nhar_i
            k = c % nhar_i
            pairs.append((k, l))
    k_idxs = [p[0] for p in pairs]
    l_idxs = [p[1] for p in pairs]
    return k_idxs, l_idxs


__all__ = [
    "ModeSelector",
    "GreedyArgmax",
    "OMPSelector",
    "LassoSelector",
]
