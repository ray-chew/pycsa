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
    _materialize_design_matrix,
    _run_fa_sa,
)


def _build_south_pole_cell():
    """Load the bundled ETOPO single-cell fixture (the polar cell)."""
    repo_root = Path(__file__).resolve().parents[1]
    fixture_dir = (
        repo_root / "tests" / "reproducibility" / "fixtures" / "etopo_single_cell"
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
    return cell


def _fa_freqs(nhi, nhj, U, V, cell, lmbda_fa, prior=None):
    """Run FA only and return ``freqs_fa`` (the dense Fourier amplitudes)."""
    work_cell = deepcopy(cell)
    first_guess = interface.get_pmf(nhi, nhj, U, V)
    if prior is not None:
        first_guess.ctx.prior = prior
    freqs_fa, _, _ = first_guess.sappx(work_cell, lmbda=lmbda_fa)
    return np.asarray(freqs_fa)


def _render(name, cell, ref_spec, dat_base, freqs_sa_base, dat_new,
            freqs_sa_new, hp_joint, l2_base, l2_new, ref_label, out_dir):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))

    # Top row: physical-domain reconstructions
    topo_panels = [
        (cell.topo, "original topography"),
        (dat_base, "baseline SA reconstruction"),
        (
            dat_new,
            f"new defaults SA  (α={hp_joint.alpha:.2f}, λ={hp_joint.lmbda:.2e})",
        ),
    ]
    topo_arrays = [np.nan_to_num(np.asarray(a)) for a, _ in topo_panels]
    topo_vmax = float(np.nanmax([np.nanmax(np.abs(a)) for a in topo_arrays])) or 1.0
    for col, ((_, title), arr) in enumerate(zip(topo_panels, topo_arrays)):
        im = axs[0, col].imshow(
            arr, origin="lower", aspect="auto", cmap="terrain",
            vmin=-topo_vmax, vmax=topo_vmax,
        )
        axs[0, col].set_title(title)
        axs[0, col].set_xlabel("lon index")
        axs[0, col].set_ylabel("lat index")
        plt.colorbar(im, ax=axs[0, col], fraction=0.046, pad=0.04)

    # Bottom row: spectra — shared color scale across all three panels
    # so visual differences mean amplitude differences, not rescales.
    spec_panels = [
        (ref_spec, f"{ref_label}"),
        (freqs_sa_base, f"baseline SA spectrum  (L₂={l2_base:.2f})"),
        (freqs_sa_new, f"new defaults SA spectrum  (L₂={l2_new:.2f})"),
    ]
    spec_arrays = [np.nan_to_num(np.asarray(a)) for a, _ in spec_panels]
    spec_vmax = float(np.nanmax([np.nanmax(np.abs(a)) for a in spec_arrays])) or 1.0
    for col, ((_, title), arr) in enumerate(zip(spec_panels, spec_arrays)):
        im = axs[1, col].imshow(
            arr, origin="lower", aspect="auto", cmap="viridis",
            vmin=0.0, vmax=spec_vmax,
        )
        axs[1, col].set_title(title)
        axs[1, col].set_xlabel("k")
        axs[1, col].set_ylabel("l")
        plt.colorbar(im, ax=axs[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{name}: baseline vs new defaults  (joint-GCV + Greedy)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    out_path = out_dir / f"{name}_baseline_vs_new.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _run(
    name, cell, nhi, nhj, U, V, lmbda_fa_base, lmbda_sa_base, n_modes,
    truth_freqs=None,
):
    print(f"\n=== {name} ===")
    M = _materialize_design_matrix(nhi, nhj, cell)
    hp = build_spectral_prior(
        topography=cell.topo, design_matrix=M, data=cell.topo_m,
        joint_selector=JointGCVSelector(),
    )
    print(f"  joint-GCV pick: α={hp.alpha:.3f}, λ={hp.lmbda:.3e}")

    _, freqs_sa_base, dat_base, _, _ = _run_fa_sa(
        nhi, nhj, U, V, cell,
        lmbda_fa=lmbda_fa_base, lmbda_sa=lmbda_sa_base, n_modes=n_modes,
        prior=None,
    )
    _, freqs_sa_new, dat_new, _, _ = _run_fa_sa(
        nhi, nhj, U, V, cell,
        lmbda_fa=hp.lmbda, lmbda_sa=hp.lmbda, n_modes=n_modes, prior=hp.prior,
    )

    if truth_freqs is not None:
        ref = np.asarray(truth_freqs)
        ref_label = "reference spectrum (planted truth, L₂=0)"
    else:
        ref = _fa_freqs(nhi, nhj, U, V, cell, hp.lmbda, hp.prior)
        ref_label = "FA spectrum (dense, new-defaults run) — reference"

    # FA / SA spectra come back with NaN sentinels at unused half-grid
    # entries (see fourier.f_trans.get_freq_grid); clean them before
    # taking the norm so a single NaN doesn't poison the whole metric.
    base_clean = np.nan_to_num(freqs_sa_base)
    new_clean = np.nan_to_num(freqs_sa_new)
    ref_clean = np.nan_to_num(ref)
    l2_base = float(np.linalg.norm(base_clean - ref_clean))
    l2_new = float(np.linalg.norm(new_clean - ref_clean))
    verdict = (
        "new wins" if l2_new < l2_base
        else "baseline wins" if l2_new > l2_base else "tie"
    )
    print(f"  L₂(SA, ref): baseline={l2_base:.3f}, new={l2_new:.3f}  ({verdict})")

    out_path = _render(
        name, cell, ref, dat_base, freqs_sa_base, dat_new, freqs_sa_new,
        hp, l2_base, l2_new, ref_label, OUTPUT_DIR,
    )
    print(f"  figure: {out_path.relative_to(out_path.parents[2])}")


def main():
    print("Baseline vs new-defaults reference figures")
    print("------------------------------------------")

    cell_id, _ = _build_idealised_cell(nhi=12, nhj=12, seed=777)
    truth = _build_idealised_freqs_ref(nhi=12, nhj=12, seed=777)
    _run(
        "idealised", cell_id,
        nhi=12, nhj=12, U=1.0, V=1.0,
        lmbda_fa_base=1.0e-1, lmbda_sa_base=1.0e-6, n_modes=14,
        truth_freqs=truth,
    )

    try:
        cell_al = _build_aleutians_cell()
        _run(
            "aleutians_merit", cell_al,
            nhi=24, nhj=48, U=10.0, V=0.0,
            lmbda_fa_base=0.0, lmbda_sa_base=1.0e-1, n_modes=50,
            truth_freqs=None,
        )
    except Exception as exc:
        print(f"\naleutians_merit SKIPPED: {exc}")

    try:
        cell_sp = _build_south_pole_cell()
        _run(
            "south_pole", cell_sp,
            nhi=32, nhj=64, U=10.0, V=0.0,
            lmbda_fa_base=0.0, lmbda_sa_base=1.0e-1, n_modes=100,
            truth_freqs=None,
        )
    except Exception as exc:
        print(f"\nsouth_pole SKIPPED: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
