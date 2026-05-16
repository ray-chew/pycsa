"""One-script empirical check for the structured-prior hyperparameter defaults.

Runs :func:`pycsa.core.hyperparams.build_spectral_prior` on the two
fixtures the kernel-spike plan singled out (idealised + Aleutians MERIT,
the third fixture — polar ETOPO — is skipped here for speed; run by
hand if you want it) and reports three things per fixture:

1. the alpha chosen by :class:`EmpiricalSpectralSlope` (with stderr,
   R² — so you can spot a poorly-fit periodogram before trusting it);
2. the lambda chosen by :class:`GCVSelector`;
3. a *quality* comparison vs. the existing production pipeline (no
   prior, fixture lmbda):

   - ``‖Δuw_pmf‖∞ / ‖uw_pmf_baseline‖∞`` — *difference* from baseline.
   - **Improvement metric** — for idealised the script knows the
     planted-modes ground truth (``freqs_ref`` from the deterministic
     22-mode superposition) and reports
     ``‖freqs_sa − freqs_ref‖_F`` for both pipelines; for Aleutians
     there is no ground truth so we report 4-fold spatial-holdout
     reconstruction MSE via :func:`pycsa.core.validation.spatial_cv_score`.

A 3-panel side-by-side topography plot lands under
``scripts/validate_outputs/`` for each fixture: truth (or input) |
baseline reconstruction | selected reconstruction.

**Not a benchmark.** This script gives one comparison per fixture, at
one choice of (alpha_selector, lambda_selector). It does *not* prove
GCV is the right default in general; it confirms the defaults are
wired correctly and produces concrete numbers that the JORS
subsection can cite.

Run::

    ~/anaconda3/envs/playground/bin/python scripts/validate_hyperparam_defaults.py

Requires playground env (Python 3.12). For Aleutians, the bundled MERIT
fixture under tests/reproducibility/fixtures/regional_merit/input/ must
exist (it ships in the repo).
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

# netCDF4 must be imported before pycsa.core.io per the playground env quirk.
import netCDF4  # noqa: F401
import numpy as np

from pycsa.core import fourier, lin_reg, physics, utils, var
from pycsa.core import io as pcio
from pycsa.core.hyperparams import JointGCVSelector, build_spectral_prior
from pycsa.core.mode_selection import GreedyArgmax, LassoSelector, OMPSelector
from pycsa.core.priors import IsotropicPrior
from pycsa.core.validation import spatial_cv_score
from pycsa.wrappers import interface


OUTPUT_DIR = Path(__file__).resolve().parent / "validate_outputs"


def _column_to_kl_for_fobj(fobj):
    """Build a column_to_kl callable matching ``f_trans.do_full``'s slicing.

    The FA design matrix M = ``hstack([Ncos, Nsin])`` is produced by
    ``f_trans.do_full`` which:

    1. builds ``tt_sum`` of shape ``(n_points, nhar_i, nhar_j)``,
    2. flattens to ``(n_points, nhar_i*nhar_j)`` in C order,
    3. computes ``bcos``, ``bsin`` elementwise on the flat array,
    4. slices the flat axis at ``nhar_j/2 - 1`` (cos) and ``nhar_j/2``
       (sin) — see [fourier.py:215-231].

    So a column index ``c`` in ``M`` maps to a flat ``(mi, mj)`` pair
    via different offsets in the cos-half and sin-half. This helper
    inverts that mapping. The returned tuple is ``(k, l) = (mi, mj)``
    — the same convention ``fobj.set_kls`` expects.

    Even/odd ``nhar_j`` use the same offsets per the slicing branches
    (the odd branch happens to match the even branch byte-for-byte
    at the relevant lines).
    """
    nhar_i = int(fobj.nhar_i)
    nhar_j = int(fobj.nhar_j)
    cos_off = nhar_j // 2 - 1
    sin_off = nhar_j // 2
    n_cos = nhar_i * nhar_j - cos_off

    def column_to_kl(c: int):
        if c < n_cos:
            flat = c + cos_off
        else:
            flat = (c - n_cos) + sin_off
        return (flat // nhar_j, flat % nhar_j)

    return column_to_kl, n_cos


def _dedupe_kl(k_idxs, l_idxs, target_n):
    """Drop duplicate (k, l) pairs (cos and sin columns for the same
    mode collide) while preserving selection order. Pads with the
    next available greedy picks if dedup leaves fewer than target_n
    — but only as a safety net; for well-behaved selectors the
    dedupe rate should be small.
    """
    seen = set()
    out_k, out_l = [], []
    for k, l in zip(k_idxs, l_idxs):
        key = (int(k), int(l))
        if key in seen:
            continue
        seen.add(key)
        out_k.append(int(k))
        out_l.append(int(l))
        if len(out_k) == target_n:
            break
    return out_k, out_l


# ----------------------------------------------------------------------
# Idealised case (deterministic synthetic terrain, no external data)
# ----------------------------------------------------------------------


def _build_idealised_cell(nhi: int = 12, nhj: int = 12, seed: int = 777):
    """Replicate runs.idealised_isosceles._build_cell + _generate_terrain.

    Returns ``(cell, triangle, remask_for_sa)``. The cell is left
    **un-masked** (rectangular cover) so the FA fit runs on the full
    grid; ``remask_for_sa`` closes over ``triangle`` and re-masks
    the cell to the ICON triangle before the SA fit, matching the
    canonical flow at ``runs/idealised_isosceles.py:141-167``.
    """
    np.random.seed(seed)
    sz = 25
    nk = np.random.randint(0, 12, size=sz)
    nl = np.random.randint(-5, 7, size=sz)
    for ii in range(sz):
        if nk[ii] == 0 and nl[ii] < 0:
            nk[ii] += np.random.randint(1, 11)
    pts = np.array(list(set(zip(nk, nl))))
    nk = pts[:, 0]
    nl = pts[:, 1]
    sz = len(pts)
    Ak = np.random.random(size=sz) * 100.0
    sck = np.random.randint(0, 2, size=sz)

    grid = var.grid()
    cell = var.topo_cell()
    vid = utils.isosceles(grid, cell)
    lat_v = grid.clat_vertices[vid, :]
    lon_v = grid.clon_vertices[vid, :]
    cell.gen_mgrids()
    cell.topo = np.zeros_like(cell.lat_grid)
    for ii in range(sz):
        nk_scaled = 2.0 * np.pi * nk[ii] / cell.lon.max()
        nl_scaled = 2.0 * np.pi * nl[ii] / cell.lat.max()
        bf_amp = Ak[ii]
        if sck[ii] == 0:
            cell.topo += bf_amp * np.cos(
                nk_scaled * cell.lon_grid + nl_scaled * cell.lat_grid
            )
        else:
            cell.topo += bf_amp * np.sin(
                nk_scaled * cell.lon_grid + nl_scaled * cell.lat_grid
            )
    triangle = utils.gen_triangle(lon_v, lat_v)

    # Start un-masked (rectangular cover) so FA sees the full grid.
    cell.get_masked(mask=np.ones_like(cell.topo).astype("bool"))
    cell.wlat = float(np.diff(cell.lat).mean())
    cell.wlon = float(np.diff(cell.lon).mean())

    def remask_for_sa(c):
        c.get_masked(triangle=triangle)
        c.wlat = float(np.diff(c.lat).mean())
        c.wlon = float(np.diff(c.lon).mean())

    return cell, triangle, remask_for_sa


# ----------------------------------------------------------------------
# Aleutians MERIT case (bundled fixture inputs)
# ----------------------------------------------------------------------


def _build_aleutians_cell():
    """Reproduce ``tests/reproducibility/capture/capture_regional_merit._run_pipeline``
    up to the point where the rectangular-cover cell is ready.

    Returns ``(cell, remask_for_sa)``. ``cell`` is the rectangular
    cover (``rect=True``); ``remask_for_sa`` re-applies
    ``utils.get_lat_lon_segments(..., rect=False)`` to constrain
    subsequent fits to the ICON triangle. Matches the canonical
    FA → mode-select → re-mask → SA flow in the capture script.
    """
    repo_root = Path(__file__).resolve().parents[1]
    fixture_dir = repo_root / "tests" / "reproducibility" / "fixtures" / "regional_merit"
    input_dir = fixture_dir / "input"
    bundled_grid = input_dir / "icon_grid.nc"
    bundled_merit_dir = str(input_dir / "merit") + "/"

    grid = var.grid()
    pcio.ncdata().read_dat(str(bundled_grid), grid)

    lat_verts_orig = np.degrees(grid.clat_vertices[0])
    lon_verts_orig = np.degrees(grid.clon_vertices[0])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts_orig.copy(), lon_verts_orig.copy()
    )

    params = var.params()
    params.path_merit = bundled_merit_dir
    params.lat_extent = clat_verts
    params.lon_extent = clon_verts
    params.merit_cg = 1
    params.padding = 10

    topo = var.topo_cell()
    reader = pcio.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_merit_topo(topo, params)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()

    cell = var.topo_cell()
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=True)

    def remask_for_sa(c):
        utils.get_lat_lon_segments(clat_verts, clon_verts, c, topo, rect=False)

    return cell, remask_for_sa


# ----------------------------------------------------------------------
# Diagnostic core
# ----------------------------------------------------------------------


def _materialize_design_matrix(nhi: int, nhj: int, cell):
    """Build the FA design matrix ``M`` for hyperparam selection.

    Side-effect free with respect to the cell: we work on a deep copy
    of the cell so the subsequent ``sappx`` calls see pristine state.
    """
    cell_copy = deepcopy(cell)
    fobj = fourier.f_trans(nhi, nhj)
    fobj.do_full(cell_copy)
    M = lin_reg.get_coeffs(fobj, ctx=fobj.ctx)
    return np.asarray(M)


def _run_fa_sa(
    nhi, nhj, U, V, cell, *, lmbda_fa, lmbda_sa, n_modes, prior=None,
    remask_for_sa=None,
):
    """Run FA (rectangular cover) → greedy mode select → re-mask to
    ICON triangle → SA.

    Returns ``(uw_sa, freqs_sa, dat_2D_sa, k_idxs, l_idxs)``.
    ``remask_for_sa(cell)`` is the thunk returned by the cell-builder
    that switches the cell from the rectangular cover to the
    triangle-masked state. When ``None``, behaves as before (no
    re-masking) — kept for backward compatibility but production
    semantics require the thunk.
    """
    work_cell = deepcopy(cell)

    # ---- FA on the rectangular cover ----
    first_guess = interface.get_pmf(nhi, nhj, U, V)
    if prior is not None:
        first_guess.ctx.prior = prior
    freqs_fa, _, _ = first_guess.sappx(work_cell, lmbda=lmbda_fa)

    # ---- mode selection (greedy, matching the existing inline loop
    # used by both capture scripts) ----
    fq = np.copy(freqs_fa)
    fq[np.isnan(fq)] = 0.0
    indices = []
    for _ in range(n_modes):
        mx = np.unravel_index(fq.argmax(), fq.shape)
        indices.append(mx)
        fq[mx] = 0.0
    k_idxs = [p[1] for p in indices]
    l_idxs = [p[0] for p in indices]

    # ---- re-mask to ICON triangle before SA ----
    if remask_for_sa is not None:
        remask_for_sa(work_cell)

    # ---- SA on the triangle-masked cell ----
    second_guess = interface.get_pmf(nhi, nhj, U, V)
    if prior is not None:
        second_guess.ctx.prior = prior
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    freqs_sa, uw_sa, dat_2D = second_guess.sappx(
        work_cell, lmbda=lmbda_sa, updt_analysis=True, scale=1.0
    )
    return uw_sa, freqs_sa, np.asarray(dat_2D), k_idxs, l_idxs


def _pick_sa_lambda_via_gcv(
    cell, nhi, nhj, k_idxs, l_idxs, *, remask_for_sa,
):
    """Run JointGCV on the *SA* design matrix and return Hyperparams.

    The SA design matrix is built on the **triangle-masked** cell
    using the modes that FA + greedy selected. GCV on this matrix is
    methodologically tied to its leave-one-point-out approximation —
    appropriate for IID-noise problems, but on spatially-correlated
    topography tends to under-regularize because dropping one point
    barely perturbs the fit. See
    :func:`_pick_sa_lambda_via_spatial_cv` for a version that uses
    the same notion of out-of-sample as the evaluation metric.
    """
    side_cell = deepcopy(cell)
    if remask_for_sa is not None:
        remask_for_sa(side_cell)
    fobj_sa = fourier.f_trans(nhi, nhj)
    fobj_sa.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    fobj_sa.do_full(side_cell)
    M_sa = np.asarray(lin_reg.get_coeffs(fobj_sa, ctx=fobj_sa.ctx))
    hp = build_spectral_prior(
        topography=side_cell.topo,
        design_matrix=M_sa,
        data=np.asarray(side_cell.topo_m),
        joint_selector=JointGCVSelector(),
    )
    return hp


def _pick_sa_lambda_via_spatial_cv(
    cell, nhi, nhj, k_idxs, l_idxs, *,
    remask_for_sa, n_folds=4, buffer_fraction=0.05, rng_seed=0,
):
    """Pick λ via 4-fold spatial CV on the *SA* design matrix.

    Matches the out-of-sample notion that the evaluation metric uses
    (leave-one-spatial-patch-out), so the selector and the evaluator
    agree on what "generalizes" means. Cost is ~4×–10× a single GCV
    because each candidate λ requires k_folds full ridge fits, but
    still tractable per cell (~seconds for a 100-mode SA basis).
    """
    from pycsa.core.hyperparams import (
        FixedAlpha,
        SpatialCVSelector,
        build_spectral_prior,
    )

    side_cell = deepcopy(cell)
    if remask_for_sa is not None:
        remask_for_sa(side_cell)
    fobj_sa = fourier.f_trans(nhi, nhj)
    fobj_sa.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    fobj_sa.do_full(side_cell)
    M_sa = np.asarray(lin_reg.get_coeffs(fobj_sa, ctx=fobj_sa.ctx))
    coords = np.column_stack(
        [np.asarray(side_cell.lon_m), np.asarray(side_cell.lat_m)]
    )
    selector = SpatialCVSelector(
        coords=coords, n_folds=n_folds, buffer_fraction=buffer_fraction,
        rng_seed=rng_seed,
    )
    hp = build_spectral_prior(
        topography=side_cell.topo,
        design_matrix=M_sa,
        data=np.asarray(side_cell.topo_m),
        alpha_selector=FixedAlpha(0.0),
        lambda_selector=selector,
    )
    return hp


def _holdout_sa_mse(
    cell, nhi, nhj, k_idxs, l_idxs, *, prior, lmbda_sa,
    n_folds=4, buffer_fraction=0.05, remask_for_sa=None,
):
    """4-fold spatial holdout MSE for an SA fit on a specified mode set.

    Builds the SA design matrix on the triangle-masked cell (after
    applying ``remask_for_sa``), then runs :func:`spatial_cv_score`
    against held-out spatial patches. Uses ``prior`` + ``lmbda_sa``
    for the per-fold ridge fit.

    This is the correct out-of-sample metric for the pipeline: SA is
    where regularization is consumed, and the SA basis is what the
    downstream physics actually uses on the constrained ICON cell.
    """
    side_cell = deepcopy(cell)
    if remask_for_sa is not None:
        remask_for_sa(side_cell)
    fobj_sa = fourier.f_trans(nhi, nhj)
    fobj_sa.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    fobj_sa.do_full(side_cell)
    M_sa = np.asarray(lin_reg.get_coeffs(fobj_sa, ctx=fobj_sa.ctx))
    coords = np.column_stack([
        np.asarray(side_cell.lon_m), np.asarray(side_cell.lat_m),
    ])
    result = spatial_cv_score(
        prior=prior if prior is not None else IsotropicPrior(),
        lmbda=max(float(lmbda_sa), 1e-12),
        design_matrix=M_sa, data=np.asarray(side_cell.topo_m),
        coords=coords, n_folds=n_folds, buffer_fraction=buffer_fraction,
        rng_seed=0,
    )
    return result["mean_heldout_mse"], M_sa.shape[1]


# ----------------------------------------------------------------------
# Quality metrics
# ----------------------------------------------------------------------


def _build_idealised_freqs_ref(nhi: int = 12, nhj: int = 12, seed: int = 777):
    """Replicate the planted-modes ground-truth spectrum.

    Mirrors ``runs.idealised_isosceles.run`` lines 128-130: place the
    planted amplitudes ``Ak`` at ``(l+5, k)`` in a ``(nhi, nhj)`` grid.
    Returned as the same shape the SA pipeline produces so they can be
    compared via Frobenius norm directly.
    """
    np.random.seed(seed)
    sz = 25
    nk = np.random.randint(0, 12, size=sz)
    nl = np.random.randint(-5, 7, size=sz)
    for ii in range(sz):
        if nk[ii] == 0 and nl[ii] < 0:
            nk[ii] += np.random.randint(1, 11)
    pts = np.array(list(set(zip(nk, nl))))
    sz = len(pts)
    Ak = np.random.random(size=sz) * 100.0
    freqs_ref = np.zeros((nhi, nhj))
    for cnt, (kk, ll) in enumerate(pts):
        freqs_ref[ll + 5, kk] = Ak[cnt]
    return freqs_ref


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def _run_fa_then_selector(
    nhi, nhj, U, V, cell, *, lmbda_fa, lmbda_sa, n_modes, prior, selector,
    remask_for_sa=None,
):
    """Run FA (rectangular cover) → selector → re-mask → SA.

    Mirrors ``_run_fa_sa`` but calls a pluggable selector instead of
    the inline argmax loop. The selector receives the FA spectrum
    AND the FA design matrix + data — OMP and Lasso need them;
    GreedyArgmax ignores them.
    """
    work_cell = deepcopy(cell)

    # FA stage. We materialize the design matrix in a side cell so
    # the FA fit itself runs unchanged (do_full + lin_reg.do).
    side_cell = deepcopy(cell)
    side_fobj = fourier.f_trans(nhi, nhj)
    side_fobj.do_full(side_cell)
    M = np.asarray(lin_reg.get_coeffs(side_fobj, ctx=side_fobj.ctx))
    column_to_kl, _ = _column_to_kl_for_fobj(side_fobj)

    first_guess = interface.get_pmf(nhi, nhj, U, V)
    if prior is not None:
        first_guess.ctx.prior = prior
    freqs_fa, _, _ = first_guess.sappx(work_cell, lmbda=lmbda_fa)

    # Selector picks (k_idxs, l_idxs).
    fq = np.copy(freqs_fa)
    fq[np.isnan(fq)] = 0.0
    k_picks, l_picks = selector(
        fq,
        n_modes=n_modes * 2,  # over-pick to cover cos/sin duplicates
        design_matrix=M,
        data=np.asarray(side_cell.topo_m),
        column_to_kl=column_to_kl,
    )
    k_idxs, l_idxs = _dedupe_kl(k_picks, l_picks, target_n=n_modes)
    if len(k_idxs) < n_modes:
        # Fall back to greedy padding if selector picked too few unique
        # modes (rare; defensive only).
        from pycsa.core.mode_selection import GreedyArgmax
        gk, gl = GreedyArgmax()(fq, n_modes=n_modes)
        for k, l in zip(gk, gl):
            if (k, l) not in set(zip(k_idxs, l_idxs)):
                k_idxs.append(int(k))
                l_idxs.append(int(l))
                if len(k_idxs) == n_modes:
                    break

    # Re-mask to ICON triangle before SA (matches production).
    if remask_for_sa is not None:
        remask_for_sa(work_cell)

    second_guess = interface.get_pmf(nhi, nhj, U, V)
    if prior is not None:
        second_guess.ctx.prior = prior
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)
    freqs_sa, uw_sa, dat_2D = second_guess.sappx(
        work_cell, lmbda=lmbda_sa, updt_analysis=True, scale=1.0
    )
    return uw_sa, freqs_sa, np.asarray(dat_2D), (k_idxs, l_idxs)


def _selector_compare(
    name, cell, nhi, nhj, U, V, lmbda_fa, lmbda_sa, n_modes,
    *, prior_for_compare, truth_freqs, remask_for_sa=None,
):
    """Compare Greedy / OMP / Lasso on the same (prior, lmbda) setting.

    ``prior_for_compare`` is the prior used during the FA fit *and* the
    SA refit. Pass ``None`` to compare selectors at the production
    baseline; pass ``hp.prior`` to compare them at the GCV-selected
    setting.
    """
    prior_label = (
        type(prior_for_compare).__name__ if prior_for_compare is not None
        else "None"
    )
    print(f"\n--- {name} : selector comparison "
          f"(prior={prior_label}, λ_fa={lmbda_fa:.2e}, λ_sa={lmbda_sa:.2e}) ---")

    selectors = {
        "Greedy": GreedyArgmax(),
        "OMP(b=5)": OMPSelector(batch_size=5),
        "Lasso": LassoSelector(),
    }
    outputs = {}
    for tag, sel in selectors.items():
        uw, freqs_sa, dat, (kis, lis) = _run_fa_then_selector(
            nhi, nhj, U, V, cell,
            lmbda_fa=lmbda_fa, lmbda_sa=lmbda_sa, n_modes=n_modes,
            prior=prior_for_compare, selector=sel,
            remask_for_sa=remask_for_sa,
        )
        outputs[tag] = {
            "uw": uw, "freqs_sa": freqs_sa, "dat": dat,
            "k_idxs": kis, "l_idxs": lis,
        }

    # Pairwise Jaccard of the selected mode sets (greedy as reference).
    ref_set = set(zip(outputs["Greedy"]["k_idxs"], outputs["Greedy"]["l_idxs"]))
    for tag in ("OMP(b=5)", "Lasso"):
        s = set(zip(outputs[tag]["k_idxs"], outputs[tag]["l_idxs"]))
        jaccard = len(ref_set & s) / max(len(ref_set | s), 1)
        print(f"  Jaccard({tag} vs Greedy) = {jaccard:.3f}  "
              f"(mode-set distance, NOT quality)")

    # Quality metric. For idealised: truth distance on freqs_sa (which
    # IS selector-dependent). For real-data: SA-stage holdout MSE on the
    # selected basis (not FA-stage — that would be selector-independent
    # and uninformative).
    print("  quality:")
    if truth_freqs is not None:
        for tag, out in outputs.items():
            d = float(np.linalg.norm(out["freqs_sa"] - truth_freqs))
            print(f"    ‖freqs_sa − truth‖_F  {tag:<10} = {d:.4e}")
    else:
        for tag, out in outputs.items():
            mse, n_sa = _holdout_sa_mse(
                cell, nhi, nhj, out["k_idxs"], out["l_idxs"],
                prior=prior_for_compare, lmbda_sa=lmbda_sa,
                remask_for_sa=remask_for_sa,
            )
            print(f"    4-fold holdout SA-MSE  {tag:<10} = {mse:.4e}  "
                  f"(SA basis size = {n_sa})")

    # Side-by-side plot: input + 3 SA reconstructions.
    plot_path = _plot_panels(
        f"{name}_selectors",
        arrays=[np.nan_to_num(cell.topo)] + [
            np.nan_to_num(outputs[tag]["dat"]) for tag in selectors
        ],
        titles=["input"] + [f"SA: {tag}" for tag in selectors],
        out_dir=OUTPUT_DIR,
    )
    print(f"  selector plot: {plot_path.relative_to(plot_path.parents[2])}")
    return outputs


def _plot_panels(name, arrays, titles, out_dir):
    """Save a side-by-side PNG with ``len(arrays)`` panels."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    vmax = float(np.nanmax(np.abs(arrays[0])))
    vmin = -vmax
    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="elevation (relative)")
    out_path = out_dir / f"{name}_reconstruction.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _diagnose(
    name, cell, nhi, nhj, U, V, lmbda_fa, lmbda_sa, n_modes,
    *, truth_freqs=None, remask_for_sa=None,
):
    """Run baseline vs hyperparam-selected pipeline and report metrics.

    Quality metric branches on ``truth_freqs``:
    - if supplied (idealised): reports ``‖freqs_sa − truth‖_F`` for
      both pipelines — smaller is better.
    - otherwise (real-data fixture): reports spatial 4-fold holdout
      reconstruction MSE for both pipelines — smaller is better.
    """
    print(f"\n=== {name} ===")
    M = _materialize_design_matrix(nhi, nhj, cell)
    print(f"  design matrix shape: {M.shape}")

    hp = build_spectral_prior(
        topography=cell.topo,
        design_matrix=M,
        data=cell.topo_m,
    )
    if hp.slope_fit is not None:
        print(
            "  sequential: alpha = {:.3f}  (stderr={:.3f}, R²={:.3f}) ← "
            "EmpiricalSpectralSlope".format(
                hp.alpha, hp.slope_fit.stderr, hp.slope_fit.r_squared
            )
        )
    else:
        print(f"  sequential: alpha = {hp.alpha:.3f}  ← FixedAlpha")
    print(f"  sequential: lambda = {hp.lmbda:.3e}  ← GCVSelector")

    # Joint 2-D GCV: picks (alpha, lmbda) together from one held-out-
    # error search. Independent comparison to the sequential pick;
    # we don't replace hp because the rest of the diagnostic uses it.
    hp_joint = build_spectral_prior(
        topography=cell.topo,
        design_matrix=M,
        data=cell.topo_m,
        joint_selector=JointGCVSelector(),
    )
    print(
        f"  joint GCV:  alpha = {hp_joint.alpha:.3f}  "
        f"lambda = {hp_joint.lmbda:.3e}  ← JointGCVSelector"
    )
    print(
        "  baseline lmbda_fa = {:.3e}  lmbda_sa = {:.3e}".format(
            lmbda_fa, lmbda_sa
        )
    )

    # Three pipelines, identical mode-selection + SA paths, only the
    # FA/SA prior+lmbda combination differs:
    #   * baseline   — no prior; the fixture's production (lmbda_fa, lmbda_sa)
    #   * isotropic  — IsotropicPrior at the GCV-selected lmbda (the "scale-only" control)
    #   * selected   — SpectralPrior(alpha) at the GCV-selected lmbda (full new defaults)
    # The isotropic-vs-selected comparison isolates the contribution of
    # the per-mode structure from the contribution of just picking a
    # different scalar lambda.
    uw_base, freqs_sa_base, dat_base, k_base, l_base = _run_fa_sa(
        nhi, nhj, U, V, cell,
        lmbda_fa=lmbda_fa, lmbda_sa=lmbda_sa, n_modes=n_modes, prior=None,
        remask_for_sa=remask_for_sa,
    )
    uw_iso, freqs_sa_iso, dat_iso, k_iso, l_iso = _run_fa_sa(
        nhi, nhj, U, V, cell,
        lmbda_fa=hp.lmbda, lmbda_sa=hp.lmbda, n_modes=n_modes,
        prior=IsotropicPrior(), remask_for_sa=remask_for_sa,
    )
    uw_sel, freqs_sa_sel, dat_sel, k_sel, l_sel = _run_fa_sa(
        nhi, nhj, U, V, cell,
        lmbda_fa=hp.lmbda, lmbda_sa=hp.lmbda, n_modes=n_modes, prior=hp.prior,
        remask_for_sa=remask_for_sa,
    )
    # Joint-GCV pipeline: uses hp_joint.prior + hp_joint.lmbda. When
    # joint picks alpha=0 this reduces to isotropic@λ_joint; when it
    # picks alpha>0 this is a non-trivial structured prior at a
    # jointly-optimized scale.
    uw_jnt, freqs_sa_jnt, dat_jnt, k_jnt, l_jnt = _run_fa_sa(
        nhi, nhj, U, V, cell,
        lmbda_fa=hp_joint.lmbda, lmbda_sa=hp_joint.lmbda, n_modes=n_modes,
        prior=hp_joint.prior, remask_for_sa=remask_for_sa,
    )

    # ---- Difference summary vs baseline ----
    norm_base = float(np.max(np.abs(uw_base))) or 1.0
    print("  difference vs. baseline (uw_pmf ∞-norm, relative):")
    print(
        "    isotropic-at-selected-λ : {:.3e}".format(
            float(np.max(np.abs(uw_iso - uw_base))) / norm_base
        )
    )
    print(
        "    selected (SpectralPrior): {:.3e}".format(
            float(np.max(np.abs(uw_sel - uw_base))) / norm_base
        )
    )
    print(
        "    joint GCV (α,λ)         : {:.3e}".format(
            float(np.max(np.abs(uw_jnt - uw_base))) / norm_base
        )
    )

    # ---- Quality metric: improvement vs baseline ----
    print("  quality vs. ground truth or held-out points:")
    if truth_freqs is not None:
        d_base = float(np.linalg.norm(freqs_sa_base - truth_freqs))
        d_iso = float(np.linalg.norm(freqs_sa_iso - truth_freqs))
        d_sel = float(np.linalg.norm(freqs_sa_sel - truth_freqs))
        d_jnt = float(np.linalg.norm(freqs_sa_jnt - truth_freqs))
        print(f"    ‖freqs_sa − truth‖_F")
        print(f"      baseline                  = {d_base:.4e}")
        print(f"      isotropic-at-selected-λ   = {d_iso:.4e}")
        print(f"      selected (SpectralPrior)  = {d_sel:.4e}")
        print(f"      joint GCV (α={hp_joint.alpha:.2f}, λ={hp_joint.lmbda:.2e})"
              f" = {d_jnt:.4e}")
        for tag, d in (("isotropic", d_iso), ("selected", d_sel),
                       ("joint_GCV", d_jnt)):
            if d_base > 0:
                rel = (d_base - d) / d_base * 100.0
                verdict = "better" if d < d_base else "worse" if d > d_base else "tie"
                print(f"      {tag:<10} vs baseline = {rel:+.2f}%  ({verdict})")
        # Structure-vs-scale isolation
        if d_iso > 0:
            structure_rel = (d_iso - d_sel) / d_iso * 100.0
            print(
                f"      structure-only effect (selected vs isotropic) = "
                f"{structure_rel:+.2f}%  "
                f"({'structure helps' if d_sel < d_iso else 'structure hurts' if d_sel > d_iso else 'tie'})"
            )
    else:
        # SA-stage holdout: per pipeline, build the SA design matrix
        # from the modes that pipeline's FA→greedy actually picked,
        # then 4-fold-CV with the pipeline's own (prior, lmbda_sa).
        # FA-stage holdout was an artifact — lmbda_fa=0 is by design,
        # so FA isn't tasked with generalizing to held-out points.
        mse_base, n_sa_base = _holdout_sa_mse(
            cell, nhi, nhj, k_base, l_base,
            prior=None, lmbda_sa=lmbda_sa, remask_for_sa=remask_for_sa,
        )
        mse_iso, n_sa_iso = _holdout_sa_mse(
            cell, nhi, nhj, k_iso, l_iso,
            prior=IsotropicPrior(), lmbda_sa=hp.lmbda,
            remask_for_sa=remask_for_sa,
        )
        mse_sel, n_sa_sel = _holdout_sa_mse(
            cell, nhi, nhj, k_sel, l_sel,
            prior=hp.prior, lmbda_sa=hp.lmbda, remask_for_sa=remask_for_sa,
        )
        mse_jnt, n_sa_jnt = _holdout_sa_mse(
            cell, nhi, nhj, k_jnt, l_jnt,
            prior=hp_joint.prior, lmbda_sa=hp_joint.lmbda,
            remask_for_sa=remask_for_sa,
        )
        print(f"    4-fold spatial holdout SA-MSE (correct metric — "
              f"SA basis built from each pipeline's own picked modes)")
        print(f"      baseline (λ_fa={lmbda_fa:.2e}, λ_sa={lmbda_sa:.2e}, no prior) "
              f"= {mse_base:.4e}  (SA basis size = {n_sa_base})")
        print(f"      isotropic-at-selected-λ              = {mse_iso:.4e}  "
              f"(SA basis size = {n_sa_iso})")
        print(f"      selected (SpectralPrior at GCV λ)    = {mse_sel:.4e}  "
              f"(SA basis size = {n_sa_sel})")
        print(f"      joint GCV (α={hp_joint.alpha:.2f}, λ={hp_joint.lmbda:.2e})"
              f" = {mse_jnt:.4e}  (SA basis size = {n_sa_jnt})")
        for tag, m in (("isotropic", mse_iso), ("selected", mse_sel),
                       ("joint_GCV", mse_jnt)):
            if mse_base > 0:
                rel = (mse_base - m) / mse_base * 100.0
                verdict = "better" if m < mse_base else "worse" if m > mse_base else "tie"
                print(f"      {tag:<10} vs baseline = {rel:+.2f}%  ({verdict})")
        # Structure-vs-scale isolation
        if mse_iso > 0:
            structure_rel = (mse_iso - mse_sel) / mse_iso * 100.0
            print(
                f"      structure-only effect (selected vs isotropic) = "
                f"{structure_rel:+.2f}%  "
                f"({'structure helps' if mse_sel < mse_iso else 'structure hurts' if mse_sel > mse_iso else 'tie'})"
            )

    # ---- Side-by-side topography (4 panels) ----
    plot_path = _plot_panels(
        name,
        arrays=[
            np.nan_to_num(cell.topo),
            np.nan_to_num(dat_base),
            np.nan_to_num(dat_iso),
            np.nan_to_num(dat_sel),
        ],
        titles=[
            "input (idealised: planted topo)" if truth_freqs is not None
            else "input topography",
            f"baseline (λ_fa={lmbda_fa:.0e})",
            f"isotropic at GCV λ={hp.lmbda:.2e}",
            f"selected: SpectralPrior(α={hp.alpha:.2f})",
        ],
        out_dir=OUTPUT_DIR,
    )
    print(f"  side-by-side plot: {plot_path.relative_to(plot_path.parents[2])}")
    return {
        "alpha": hp.alpha,
        "lambda": hp.lmbda,
    }


def main(argv=None):
    print("kernel-spike phase-2 default validation")
    print("---------------------------------------")
    print(
        "Reports the alpha/lambda chosen by the documented defaults\n"
        "(EmpiricalSpectralSlope + GCVSelector) on two fixtures, plus\n"
        "‖Δuw_pmf‖∞ / ‖uw_pmf_baseline‖∞ vs. each fixture's existing\n"
        "production pipeline (no prior, fixture lmbda)."
    )

    # ---- Idealised: numbers from runs/idealised_isosceles.py ----
    cell_id, _triangle, remask_id = _build_idealised_cell(
        nhi=12, nhj=12, seed=777
    )
    truth_freqs = _build_idealised_freqs_ref(nhi=12, nhj=12, seed=777)
    result = _diagnose(
        "idealised",
        cell_id,
        nhi=12, nhj=12, U=1.0, V=1.0,
        lmbda_fa=1.0e-1, lmbda_sa=1.0e-6, n_modes=14,
        truth_freqs=truth_freqs, remask_for_sa=remask_id,
    )
    _selector_compare(
        "idealised", cell_id,
        nhi=12, nhj=12, U=1.0, V=1.0,
        lmbda_fa=1.0e-1, lmbda_sa=1.0e-6, n_modes=14,
        prior_for_compare=None, truth_freqs=truth_freqs,
        remask_for_sa=remask_id,
    )
    _selector_compare(
        "idealised_gcv", cell_id,
        nhi=12, nhj=12, U=1.0, V=1.0,
        lmbda_fa=result["lambda"], lmbda_sa=result["lambda"], n_modes=14,
        prior_for_compare=IsotropicPrior(), truth_freqs=truth_freqs,
        remask_for_sa=remask_id,
    )

    # ---- Aleutians MERIT: numbers from capture_regional_merit.py ----
    try:
        cell_al, remask_al = _build_aleutians_cell()
    except Exception as exc:
        print(f"\n=== aleutians_merit (SKIPPED: {exc}) ===")
        return 0
    result = _diagnose(
        "aleutians_merit",
        cell_al,
        nhi=24, nhj=48, U=10.0, V=0.0,
        lmbda_fa=0.0, lmbda_sa=1.0e-1, n_modes=50,
        truth_freqs=None, remask_for_sa=remask_al,
    )
    _selector_compare(
        "aleutians_merit", cell_al,
        nhi=24, nhj=48, U=10.0, V=0.0,
        lmbda_fa=0.0, lmbda_sa=1.0e-1, n_modes=50,
        prior_for_compare=None, truth_freqs=None,
        remask_for_sa=remask_al,
    )
    _selector_compare(
        "aleutians_merit_gcv", cell_al,
        nhi=24, nhj=48, U=10.0, V=0.0,
        lmbda_fa=result["lambda"], lmbda_sa=result["lambda"], n_modes=50,
        prior_for_compare=IsotropicPrior(), truth_freqs=None,
        remask_for_sa=remask_al,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
