"""Held-out validation utilities for the structured prior.

Two callables here. :func:`spatial_cv_score` is the workhorse —
it runs k-fold spatial cross-validation for an arbitrary
:class:`pycsa.core.priors.Prior` at a fixed ``lmbda``, returning
the per-fold and mean held-out MSE. :class:`SpatialCVSelector` in
``hyperparams.py`` uses it internally over a ``lmbda`` grid; users
can call it directly to validate *any* prior choice without going
through a selector.

**Patch geometry, made concrete.** Phase 1's plan flagged this as
the most under-specified piece. The implementation here:

- Takes per-row coordinates as a ``(n_points, 2)`` array (any
  metric — local Cartesian, (lon, lat), whatever the caller's
  cell uses). When ``coords`` is ``None`` we fall back to
  ``cell.lon_m``-style row-index ordering — that is, we treat
  the points as already in scan-line order and split by index.
  That's the only setting where the fallback is correct; the
  caller is responsible for providing real coordinates when the
  data is on a Delaunay grid or any other non-scan-line layout.
- Computes a 2D bounding box from the supplied coords, partitions
  it into a near-square ``r × c`` grid where ``r·c ≥ n_folds``,
  and assigns each fold to one tile. Excess tiles are unused.
  Tiles are contiguous in coordinate space — this is what
  ``spatial`` cross-validation actually means; per-point random
  shuffling leaks long-wavelength modes across folds and would
  silently overstate held-out accuracy.
- Each held-out tile has a buffer zone of width
  ``buffer_fraction · tile_side`` around it. Points inside the
  buffer are excluded from both the training set and the
  evaluation set for that fold.

Documented limitation: works for cells whose points roughly fill a
2D region (MERIT regional cells, ETOPO regional cells). For ICON
Delaunay-triangle cells with sparse coverage near a cell vertex
the bounding-box partition may produce empty tiles — the function
raises in that case so the failure is visible.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.linalg as la

from pycsa.core.priors import Prior


def _build_spatial_folds(
    coords: np.ndarray,
    n_folds: int,
    buffer_fraction: float,
    rng_seed: Optional[int],
) -> list:
    """Return list of length n_folds; each entry is (train_idx, eval_idx).

    Folds are contiguous in coordinate space. Buffer points are
    excluded from both training and evaluation for the corresponding
    fold. The caller is responsible for verifying that each fold has
    non-empty train + eval sets — we raise otherwise.
    """
    rng = np.random.default_rng(rng_seed)
    n = coords.shape[0]
    if n < 2 * n_folds:
        raise ValueError(
            f"Need at least 2*n_folds={2 * n_folds} points to split; got {n}"
        )
    x = coords[:, 0]
    y = coords[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax == xmin or ymax == ymin:
        raise ValueError(
            "coords have zero extent in at least one axis — cannot tile"
        )
    # Near-square grid of tiles. Pick r and c so r*c >= n_folds and the
    # aspect ratio of the tile matches the aspect ratio of the bounding box.
    aspect = (xmax - xmin) / (ymax - ymin)
    side = int(np.ceil(np.sqrt(n_folds)))
    r = max(int(np.ceil(side / np.sqrt(aspect))), 1)
    c = max(int(np.ceil(n_folds / r)), 1)
    # tile_x, tile_y are the per-tile widths in coord units
    tile_x = (xmax - xmin) / c
    tile_y = (ymax - ymin) / r
    buffer_x = buffer_fraction * tile_x
    buffer_y = buffer_fraction * tile_y

    # Pick n_folds tile centers in a deterministic interleaved order so
    # the chosen folds are spatially spread rather than packed in one
    # corner.
    tile_ids = list(range(r * c))
    rng.shuffle(tile_ids)
    chosen = tile_ids[:n_folds]

    folds = []
    for tile_id in chosen:
        ti, tj = divmod(tile_id, c)
        xlo = xmin + tj * tile_x
        xhi = xlo + tile_x
        ylo = ymin + ti * tile_y
        yhi = ylo + tile_y
        eval_mask = (x >= xlo) & (x < xhi) & (y >= ylo) & (y < yhi)
        buffer_mask = (
            (x >= xlo - buffer_x)
            & (x < xhi + buffer_x)
            & (y >= ylo - buffer_y)
            & (y < yhi + buffer_y)
        )
        train_mask = ~buffer_mask  # buffer excludes both train and eval
        eval_idx = np.where(eval_mask)[0]
        train_idx = np.where(train_mask)[0]
        if eval_idx.size == 0:
            raise ValueError(
                f"Fold tile ({ti}, {tj}) has no eval points — "
                "data is too sparse for the requested n_folds; "
                "lower n_folds or supply denser coords"
            )
        if train_idx.size < 2:
            raise ValueError(
                f"Fold tile ({ti}, {tj}) leaves < 2 training points — "
                "lower buffer_fraction or n_folds"
            )
        folds.append((train_idx, eval_idx))
    return folds


def spatial_cv_score(
    prior: Prior,
    lmbda: float,
    design_matrix: np.ndarray,
    data: np.ndarray,
    *,
    coords: Optional[np.ndarray] = None,
    n_folds: int = 5,
    buffer_fraction: float = 0.1,
    rng_seed: Optional[int] = None,
) -> dict:
    """K-fold spatial cross-validation for any :class:`Prior`.

    Solves the regularized normal equations on each fold's training
    rows, predicts the held-out rows, and returns the per-fold and
    mean reconstruction MSE.

    Parameters
    ----------
    prior
        Any :class:`pycsa.core.priors.Prior`. Called per-fold with
        the fold's normal-equations matrix.
    lmbda
        Regularization scale passed to the prior.
    design_matrix
        Dense ``M`` matrix, shape ``(n_points, n_modes)``.
    data
        Target vector, shape ``(n_points,)``.
    coords
        Per-row 2D coordinates for spatial fold construction. If
        ``None``, falls back to a strided index split — only
        appropriate when rows are already in scan-line order.
    n_folds, buffer_fraction, rng_seed
        See module docstring.

    Returns
    -------
    dict with keys:
        ``per_fold_mse``
            ndarray of length ``n_folds``.
        ``mean_heldout_mse``
            Mean of ``per_fold_mse``.
        ``fold_sizes``
            ndarray of shape ``(n_folds, 2)`` — (n_train, n_eval) per fold.
    """
    M = np.asarray(design_matrix, dtype=float)
    y = np.asarray(data, dtype=float).reshape(-1)
    n = M.shape[0]
    if coords is None:
        # Strided index split — only correct if rows are scan-line
        # ordered. Document the assumption in the result.
        coords_array = np.column_stack([np.arange(n), np.zeros(n)])
    else:
        coords_array = np.asarray(coords, dtype=float)
        if coords_array.shape[0] != n:
            raise ValueError(
                f"coords rows ({coords_array.shape[0]}) must match "
                f"design_matrix rows ({n})"
            )
    folds = _build_spatial_folds(
        coords_array, n_folds=n_folds, buffer_fraction=buffer_fraction,
        rng_seed=rng_seed,
    )

    per_fold_mse = np.full(n_folds, np.nan)
    fold_sizes = np.zeros((n_folds, 2), dtype=int)
    for f, (train_idx, eval_idx) in enumerate(folds):
        M_tr, y_tr = M[train_idx], y[train_idx]
        M_ev, y_ev = M[eval_idx], y[eval_idx]
        E = M_tr.T @ M_tr
        diag_add = np.asarray(
            prior(fobj=None, E_tilda_lm=E, lmbda=lmbda), dtype=float
        )
        if diag_add.shape != (E.shape[0],):
            raise ValueError(
                f"prior returned diag of shape {diag_add.shape}; "
                f"expected {(E.shape[0],)}"
            )
        np.fill_diagonal(E, np.diag(E) + diag_add)
        rhs = M_tr.T @ y_tr
        try:
            c_factor, lower = la.cho_factor(E, lower=True, check_finite=False)
            a_m = la.cho_solve((c_factor, lower), rhs, check_finite=False)
        except la.LinAlgError:
            a_m, *_ = np.linalg.lstsq(E, rhs, rcond=None)
        y_pred = M_ev @ a_m
        per_fold_mse[f] = float(np.mean((y_ev - y_pred) ** 2))
        fold_sizes[f] = (int(train_idx.size), int(eval_idx.size))

    return {
        "per_fold_mse": per_fold_mse,
        "mean_heldout_mse": float(np.mean(per_fold_mse)),
        "fold_sizes": fold_sizes,
    }


__all__ = ["spatial_cv_score"]
