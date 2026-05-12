"""Capture the idealised isosceles fixture.

Runs ``runs.idealised_isosceles.run()``, writes ``output.nc`` + ``manifest.yml``
+ ``figure.png`` into ``tests/reproducibility/fixtures/idealised/``.

Idempotent: re-running overwrites the existing fixture. Capture is a deliberate
fixture-update action — open a PR with the refreshed figure in the description.

Usage::

    python -m tests.reproducibility.capture.capture_idealised
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Module-level imports are deferred into main() so the script's --help and the
# fixture-path resolution work without needing pyCSA's full import chain.


CASE = "idealised"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / CASE


def capture(out_dir: Path) -> None:
    from runs.idealised_isosceles import run, EXPERIMENT_LABELS
    from tests.reproducibility.comparator import save_netcdf
    from tests.reproducibility.manifest import Manifest

    out_dir.mkdir(parents=True, exist_ok=True)

    result = run()

    # Physical-domain reconstructions (`dat_arr`) are used for the figure but
    # not pinned in the fixture — the (480, 480) per-experiment grid would
    # bloat the bundle to ~11 MB, and the spectrum already pins the numerics.
    variables = {
        "freqs_arr": result.freqs_arr,
        "errs": result.errs,
        "sums": result.sums,
        "sum_errs": result.sum_errs,
        "freqs_ref": result.freqs_ref,
        "num_modes": np.array(result.num_modes, dtype=np.int64),
    }

    save_netcdf(out_dir / "output.nc", variables)

    # The pLSFF / pLSFF_quad experiments (indices 1 and 5) use lmbda=0
    # → la.inv() on an ill-conditioned matrix → cross-LAPACK drift up to
    # ~0.2% across platforms (e.g. local SciPy 1.17 vs CI SciPy 1.15).
    # The reference/regLSFF/optCSA/subCSA paths are bit-identical at any
    # platform. We loosen rtol on the affected variables rather than drop
    # pLSFF — refactors that meaningfully change pLSFF behavior would
    # exceed 1% drift and still be caught.
    per_variable_rtol = {
        "freqs_arr": 1e-2,
        "errs": 1e-2,
        "sums": 1e-2,
        "sum_errs": 1e-2,
    }
    manifest = Manifest.build(
        fixture=CASE,
        variables=variables,
        per_variable_rtol=per_variable_rtol,
        notes=(
            "Idealised isosceles CSA experiment. Deterministic, no external\n"
            "data. Variable order in freqs_arr/errs/sums/sum_errs follows\n"
            f"runs.idealised_isosceles.EXPERIMENT_LABELS = {EXPERIMENT_LABELS}.\n"
            "freqs_arr / errs / sums / sum_errs use rtol=1e-2: pLSFF rows\n"
            "(indices 1, 5) are unregularized and drift ~0.2% across LAPACK\n"
            "builds; reference / regLSFF / optCSA / subCSA rows are bit-equal\n"
            "regardless."
        ),
    )
    manifest.save(out_dir / "manifest.yml")

    _render_figure(result, out_dir / "figure.png", EXPERIMENT_LABELS)

    print(f"captured idealised fixture → {out_dir}")
    print(f"  variables: {list(variables)}")
    print(f"  L2 errors: {result.errs}")
    print(f"  amplitudes: {result.sums}")


def _render_figure(result, path: Path, labels) -> None:
    """2-row figure for PR-review eyeballing.

    Row 1 (physical domain): original masked topography, regLSFF
    reconstruction, final subCSA reconstruction.
    Row 2 (spectral domain): reference spectrum, regLSFF spectrum, subCSA
    spectrum.
    pLSFF/pLSFF_quad columns are intentionally omitted from the figure —
    their amplitudes are off by ~1000× and would crush the colorbar; they
    are still pinned in the manifest at rtol=1e-2.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Indices into EXPERIMENT_LABELS for the 3 panels we plot.
    panel_idxs = [0, 2, 4]  # reference, regLSFF, subCSA
    panel_titles = [labels[i] for i in panel_idxs]

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))

    # Physical-domain row: shared color range from the reference field.
    dat_ref = result.dat_arr[0]
    phys_vmin, phys_vmax = float(dat_ref.min()), float(dat_ref.max())
    for col, idx in enumerate(panel_idxs):
        dat = result.dat_arr[idx]
        im = axs[0, col].imshow(
            dat,
            origin="lower",
            aspect="auto",
            cmap="terrain",
            vmin=phys_vmin,
            vmax=phys_vmax,
        )
        prefix = "original" if idx == 0 else "reconstruction"
        axs[0, col].set_title(f"{prefix}: {panel_titles[col]}", fontsize=10)
        axs[0, col].set_xlabel("lon index")
        axs[0, col].set_ylabel("lat index")
        plt.colorbar(im, ax=axs[0, col], fraction=0.046, pad=0.04)

    # Spectral-domain row: shared color range from reference spectrum.
    spec_ref = result.freqs_arr[0]
    spec_vmin, spec_vmax = float(spec_ref.min()), float(spec_ref.max())
    for col, idx in enumerate(panel_idxs):
        spec = result.freqs_arr[idx]
        im = axs[1, col].imshow(
            spec,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=spec_vmin,
            vmax=spec_vmax,
        )
        l2 = result.errs[idx]
        axs[1, col].set_title(
            f"spectrum: {panel_titles[col]} (L2={l2:.2f})", fontsize=10
        )
        axs[1, col].set_xlabel("k")
        axs[1, col].set_ylabel("l")
        plt.colorbar(im, ax=axs[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Idealised isosceles — {result.num_modes} unique modes "
        f"(top: physical / bottom: spectral)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Fixture directory (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args(argv)
    capture(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
