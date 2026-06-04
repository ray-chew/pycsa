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
    """2-row figure using the common pyCSA plotting routines.

    Row 1 (physical domain, ``plotter.fig_obj.phys_panel``): original masked
    topography, regLSFF reconstruction, final subCSA reconstruction.
    Row 2 (spectral domain, ``plotter.fig_obj.freq_panel``): the corresponding
    spectra. These are the same routines used for the JAMES-paper figures.
    Cells outside the isosceles triangle are exactly 0 and are masked to NaN so
    the triangle exterior renders white. pLSFF/pLSFF_quad columns are omitted
    (amplitudes ~1000x off); they remain pinned in the manifest at rtol=1e-2.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycsa.plotting import plotter

    # Indices into EXPERIMENT_LABELS for the 3 panels we plot.
    panel_idxs = [0, 2, 4]  # reference, regLSFF (= the FA), subCSA (= CSA)
    # Display names in the paper's vocabulary. regLSFF is a single-stage
    # regularized least-squares fit, i.e. the first approximation (FA); subCSA
    # is the full constrained spectral approximation (CSA). We avoid the
    # internal "LSFF" acronym, which the paper never introduces.
    display = {"regLSFF": "FA", "subCSA": "CSA"}
    panel_titles = [display.get(labels[i], labels[i]) for i in panel_idxs]

    nhj, nhi = result.freqs_arr.shape[1], result.freqs_arr.shape[2]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    fobj = plotter.fig_obj(fig, nhi, nhj, cbar=True, set_label=True)

    # Physical-domain row (cividis); exterior (exactly 0) -> NaN -> white.
    for col, idx in enumerate(panel_idxs):
        dat = np.where(result.dat_arr[idx] == 0.0, np.nan, result.dat_arr[idx])
        prefix = "original" if idx == 0 else "reconstruction"
        fobj.phys_panel(axs[0, col], dat, title=f"{prefix}: {panel_titles[col]}")

    # Spectral-domain row (grayscale, n/m axes).
    for col, idx in enumerate(panel_idxs):
        l2 = result.errs[idx]
        fobj.freq_panel(
            axs[1, col],
            result.freqs_arr[idx],
            title=f"spectrum: {panel_titles[col]} (L2={l2:.2f})",
        )

    fig.suptitle(
        f"Idealised isosceles — {result.num_modes} unique modes "
        f"(top: physical / bottom: spectral)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_only(case_dir: Path = DEFAULT_DIR) -> None:
    """Regenerate ``figure.png`` only (no external data; idealised is synthetic).

    Used by ``tests.reproducibility.render_figures`` to refresh figures in CI
    without touching the pinned ``output.nc`` / ``manifest.yml``.
    """
    from runs.idealised_isosceles import run, EXPERIMENT_LABELS

    case_dir.mkdir(parents=True, exist_ok=True)
    _render_figure(run(), case_dir / "figure.png", EXPERIMENT_LABELS)


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
