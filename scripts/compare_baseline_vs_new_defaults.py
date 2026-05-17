"""Reference figures comparing baseline vs the new-defaults pipeline.

Produces one PR-#32-style 2×3 figure per fixture under
``scripts/validate_outputs/<fixture>_baseline_vs_new.png``:

    [ input topo            | baseline SA recon       | new-defaults SA recon       ]
    [ reference spectrum    | baseline SA spectrum    | new-defaults SA spectrum    ]

"New defaults" = ``build_spectral_prior(joint_selector=JointGCVSelector())``,
with mode selection via greedy argmax (production default), applied
uniformly at FA *and* SA. On every fixture tested so far the joint
GCV picks ``α = 0`` (no per-mode structure), so the "new defaults"
pipeline is in practice isotropic Tikhonov at a data-picked scalar
``λ`` — but the script doesn't bake that in, so if a future cell
benefits from ``α > 0`` it would surface here automatically.

The "reference spectrum" panel is:

- the planted-modes ground truth for ``idealised`` (a ``(nhar_j,
  nhar_i)`` array with non-zero entries only at the planted
  ``(l + 5, k)`` positions);
- the FA spectrum from the new-defaults run for real-data fixtures
  — the densest, least-biased estimate we have without a known
  truth. The L₂ labels then read as "how much does the SA sparse
  approximation deviate from the dense FA estimate."

Run::

    ~/anaconda3/envs/playground/bin/python scripts/compare_baseline_vs_new_defaults.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import netCDF4  # noqa: F401 — netCDF4 must be imported before pycsa.core.io
import matplotlib.pyplot as plt
import numpy as np

from pycsa.core import fourier, lin_reg, utils, var
from pycsa.core import io as pcio
from pycsa.core.hyperparams import JointGCVSelector, build_spectral_prior
from pycsa.wrappers import interface

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from validate_hyperparam_defaults import (  # noqa: E402
    OUTPUT_DIR,
    _build_aleutians_cell,
    _build_idealised_cell,
    _build_idealised_freqs_ref,
    _holdout_sa_mse,
    _materialize_design_matrix,
    _pick_sa_lambda_via_gcv,
    _pick_sa_lambda_via_spatial_cv,
    _run_fa_sa,
)
from pycsa.core.priors import IsotropicPrior  # noqa: E402


def _build_south_pole_cell():
    """Load the bundled ETOPO single-cell fixture (the polar cell).

    Returns ``(cell, remask_for_sa)``. Cell is built with ``rect=True``
    so FA runs on the rectangular cover; ``remask_for_sa`` switches
    it to the triangle-masked state for SA.
    """
    repo_root = Path(__file__).resolve().parents[1]
    fixture_dir = (
        repo_root
        / "tests"
        / "reproducibility"
        / "fixtures"
        / "etopo_single_cell"
        / "input"
    )
    grid = var.grid()
    pcio.ncdata().read_dat(str(fixture_dir / "icon_grid.nc"), grid)
    lat_verts_orig = np.degrees(grid.clat_vertices[0])
    lon_verts_orig = np.degrees(grid.clon_vertices[0])
    clat_verts, clon_verts = utils.handle_latlon_expansion(
        lat_verts_orig.copy(), lon_verts_orig.copy()
    )
    params = var.params()
    params.path_etopo = str(fixture_dir / "etopo") + "/"
    params.lat_extent = clat_verts
    params.lon_extent = clon_verts
    params.etopo_cg = 1
    params.padding = 10
    topo = var.topo_cell()
    reader = pcio.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    reader.read_etopo_topo(topo, params)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0
    topo.gen_mgrids()
    cell = var.topo_cell()
    utils.get_lat_lon_segments(clat_verts, clon_verts, cell, topo, rect=True)

    def remask_for_sa(c):
        utils.get_lat_lon_segments(clat_verts, clon_verts, c, topo, rect=False)

    return cell, remask_for_sa


def _fa_freqs(nhi, nhj, U, V, cell, lmbda_fa, prior=None):
    """Run FA on the rectangular-cover ``cell`` and return ``freqs_fa``."""
    work_cell = deepcopy(cell)
    first_guess = interface.get_pmf(nhi, nhj, U, V)
    if prior is not None:
        first_guess.ctx.prior = prior
    freqs_fa, _, _ = first_guess.sappx(work_cell, lmbda=lmbda_fa)
    return np.asarray(freqs_fa)


def _render(
    name,
    cell,
    ref_spec,
    ref_label,
    dat_base,
    freqs_sa_base,
    l2_base,
    dat_gcv,
    freqs_sa_gcv,
    l2_gcv,
    hp_gcv,
    dat_spcv,
    freqs_sa_spcv,
    l2_spcv,
    hp_spcv,
    out_dir,
):
    """2×4 grid: original | baseline | new (GCV) | new (SpatialCV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))

    topo_panels = [
        (cell.topo, "original topography"),
        (dat_base, "baseline SA reconstruction"),
        (dat_gcv, f"new (SA-GCV)  α={hp_gcv.alpha:.2f}, λ={hp_gcv.lmbda:.2e}"),
        (dat_spcv, f"new (SA-SpatialCV)  α={hp_spcv.alpha:.2f}, λ={hp_spcv.lmbda:.2e}"),
    ]
    topo_arrays = [np.nan_to_num(np.asarray(a)) for a, _ in topo_panels]
    topo_vmax = float(np.nanmax([np.nanmax(np.abs(a)) for a in topo_arrays])) or 1.0
    for col, ((_, title), arr) in enumerate(zip(topo_panels, topo_arrays)):
        im = axs[0, col].imshow(
            arr,
            origin="lower",
            aspect="auto",
            cmap="terrain",
            vmin=-topo_vmax,
            vmax=topo_vmax,
        )
        axs[0, col].set_title(title, fontsize=10)
        axs[0, col].set_xlabel("lon index")
        axs[0, col].set_ylabel("lat index")
        plt.colorbar(im, ax=axs[0, col], fraction=0.046, pad=0.04)

    spec_panels = [
        (ref_spec, f"{ref_label}"),
        (freqs_sa_base, f"baseline SA  L₂={l2_base:.2f}"),
        (freqs_sa_gcv, f"new (SA-GCV) SA  L₂={l2_gcv:.2f}"),
        (freqs_sa_spcv, f"new (SA-SpatialCV) SA  L₂={l2_spcv:.2f}"),
    ]
    spec_arrays = [np.nan_to_num(np.asarray(a)) for a, _ in spec_panels]
    spec_vmax = float(np.nanmax([np.nanmax(np.abs(a)) for a in spec_arrays])) or 1.0
    for col, ((_, title), arr) in enumerate(zip(spec_panels, spec_arrays)):
        im = axs[1, col].imshow(
            arr,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=spec_vmax,
        )
        axs[1, col].set_title(title, fontsize=10)
        axs[1, col].set_xlabel("k")
        axs[1, col].set_ylabel("l")
        plt.colorbar(im, ax=axs[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{name}: baseline vs new defaults  (SA-stage λ via GCV vs SpatialCV)",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    out_path = out_dir / f"{name}_baseline_vs_new.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _run(
    name,
    cell,
    nhi,
    nhj,
    U,
    V,
    lmbda_fa_base,
    lmbda_sa_base,
    n_modes,
    truth_freqs=None,
    remask_for_sa=None,
):
    print(f"\n=== {name} ===")

    # Baseline pipeline (production defaults). Provides the mode set
    # we re-use for both new-default pipelines below.
    _, freqs_sa_base, dat_base, k_idxs, l_idxs = _run_fa_sa(
        nhi,
        nhj,
        U,
        V,
        cell,
        lmbda_fa=lmbda_fa_base,
        lmbda_sa=lmbda_sa_base,
        n_modes=n_modes,
        prior=None,
        remask_for_sa=remask_for_sa,
    )

    # SA-stage λ via two different selectors.
    hp_gcv = _pick_sa_lambda_via_gcv(
        cell,
        nhi,
        nhj,
        k_idxs,
        l_idxs,
        remask_for_sa=remask_for_sa,
    )
    hp_spcv = _pick_sa_lambda_via_spatial_cv(
        cell,
        nhi,
        nhj,
        k_idxs,
        l_idxs,
        remask_for_sa=remask_for_sa,
    )
    print(f"  SA-stage GCV pick:        α={hp_gcv.alpha:.3f}, λ={hp_gcv.lmbda:.3e}")
    print(f"  SA-stage SpatialCV pick:  α={hp_spcv.alpha:.3f}, λ={hp_spcv.lmbda:.3e}")
    print(f"  production hardcoded λ_sa = {lmbda_sa_base:.3e}")

    # Run both new-defaults pipelines.
    _, freqs_sa_gcv, dat_gcv, k_gcv, l_gcv = _run_fa_sa(
        nhi,
        nhj,
        U,
        V,
        cell,
        lmbda_fa=lmbda_fa_base,
        lmbda_sa=hp_gcv.lmbda,
        n_modes=n_modes,
        prior=hp_gcv.prior,
        remask_for_sa=remask_for_sa,
    )
    _, freqs_sa_spcv, dat_spcv, k_spcv, l_spcv = _run_fa_sa(
        nhi,
        nhj,
        U,
        V,
        cell,
        lmbda_fa=lmbda_fa_base,
        lmbda_sa=hp_spcv.lmbda,
        n_modes=n_modes,
        prior=hp_spcv.prior,
        remask_for_sa=remask_for_sa,
    )

    # 4-fold spatial holdout MSE for each pipeline.
    mse_base, _ = _holdout_sa_mse(
        cell,
        nhi,
        nhj,
        k_idxs,
        l_idxs,
        prior=None,
        lmbda_sa=lmbda_sa_base,
        remask_for_sa=remask_for_sa,
    )
    mse_gcv, _ = _holdout_sa_mse(
        cell,
        nhi,
        nhj,
        k_gcv,
        l_gcv,
        prior=hp_gcv.prior,
        lmbda_sa=hp_gcv.lmbda,
        remask_for_sa=remask_for_sa,
    )
    mse_spcv, _ = _holdout_sa_mse(
        cell,
        nhi,
        nhj,
        k_spcv,
        l_spcv,
        prior=hp_spcv.prior,
        lmbda_sa=hp_spcv.lmbda,
        remask_for_sa=remask_for_sa,
    )
    print(f"  SA-stage holdout MSE:")
    print(f"    baseline       = {mse_base:.3e}")
    rel_gcv = (mse_base - mse_gcv) / mse_base * 100.0
    rel_spcv = (mse_base - mse_spcv) / mse_base * 100.0
    print(f"    new (GCV)      = {mse_gcv:.3e}   ({rel_gcv:+.2f}% vs baseline)")
    print(f"    new (SpatialCV)= {mse_spcv:.3e}   ({rel_spcv:+.2f}% vs baseline)")

    if truth_freqs is not None:
        ref = np.asarray(truth_freqs)
        ref_label = "reference spectrum (planted truth, L₂=0)"
    else:
        # Dense reference: the unregularized FA spectrum (λ_fa=0, no
        # prior — production's FA stage, unchanged across pipelines).
        # The SA panels are sparse approximations of this.
        ref = _fa_freqs(nhi, nhj, U, V, cell, lmbda_fa_base, None)
        ref_label = "FA spectrum (dense, λ_fa=0) — reference"

    # FA / SA spectra come back with NaN sentinels at unused half-grid
    # entries (see fourier.f_trans.get_freq_grid); clean them before
    # taking the norm so a single NaN doesn't poison the whole metric.
    ref_clean = np.nan_to_num(ref)
    l2_base = float(np.linalg.norm(np.nan_to_num(freqs_sa_base) - ref_clean))
    l2_gcv = float(np.linalg.norm(np.nan_to_num(freqs_sa_gcv) - ref_clean))
    l2_spcv = float(np.linalg.norm(np.nan_to_num(freqs_sa_spcv) - ref_clean))
    print(
        f"  L₂(SA, ref): "
        f"baseline={l2_base:.3f}, GCV={l2_gcv:.3f}, SpatialCV={l2_spcv:.3f}"
    )

    out_path = _render(
        name,
        cell,
        ref,
        ref_label,
        dat_base,
        freqs_sa_base,
        l2_base,
        dat_gcv,
        freqs_sa_gcv,
        l2_gcv,
        hp_gcv,
        dat_spcv,
        freqs_sa_spcv,
        l2_spcv,
        hp_spcv,
        OUTPUT_DIR,
    )
    print(f"  figure: {out_path.relative_to(out_path.parents[2])}")


def main():
    print("Baseline vs new-defaults reference figures")
    print("------------------------------------------")

    cell_id, _triangle, remask_id = _build_idealised_cell(nhi=12, nhj=12, seed=777)
    truth = _build_idealised_freqs_ref(nhi=12, nhj=12, seed=777)
    _run(
        "idealised",
        cell_id,
        nhi=12,
        nhj=12,
        U=1.0,
        V=1.0,
        lmbda_fa_base=1.0e-1,
        lmbda_sa_base=1.0e-6,
        n_modes=14,
        truth_freqs=truth,
        remask_for_sa=remask_id,
    )

    try:
        cell_al, remask_al = _build_aleutians_cell()
        _run(
            "aleutians_merit",
            cell_al,
            nhi=24,
            nhj=48,
            U=10.0,
            V=0.0,
            lmbda_fa_base=0.0,
            lmbda_sa_base=1.0e-1,
            n_modes=50,
            truth_freqs=None,
            remask_for_sa=remask_al,
        )
    except Exception as exc:
        print(f"\naleutians_merit SKIPPED: {exc}")

    try:
        cell_sp, remask_sp = _build_south_pole_cell()
        _run(
            "south_pole",
            cell_sp,
            nhi=32,
            nhj=64,
            U=10.0,
            V=0.0,
            lmbda_fa_base=0.0,
            lmbda_sa_base=1.0e-1,
            n_modes=100,
            truth_freqs=None,
            remask_for_sa=remask_sp,
        )
    except Exception as exc:
        print(f"\nsouth_pole SKIPPED: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
