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

    variables = {
        "freqs_arr": result.freqs_arr,
        "errs": result.errs,
        "sums": result.sums,
        "sum_errs": result.sum_errs,
        "freqs_ref": result.freqs_ref,
        "num_modes": np.array(result.num_modes, dtype=np.int64),
    }

    save_netcdf(out_dir / "output.nc", variables)

    manifest = Manifest.build(
        fixture=CASE,
        variables=variables,
        notes=(
            "Idealised isosceles CSA experiment. Deterministic, no external\n"
            "data. Variable order in freqs_arr/errs/sums/sum_errs follows\n"
            f"runs.idealised_isosceles.EXPERIMENT_LABELS = {EXPERIMENT_LABELS}."
        ),
    )
    manifest.save(out_dir / "manifest.yml")

    _render_figure(result, out_dir / "figure.png", EXPERIMENT_LABELS)

    print(f"captured idealised fixture → {out_dir}")
    print(f"  variables: {list(variables)}")
    print(f"  L2 errors: {result.errs}")
    print(f"  amplitudes: {result.sums}")


def _render_figure(result, path: Path, labels) -> None:
    """Minimal 6-panel spectrum figure for PR-review eyeballing.

    Imports matplotlib lazily so capture without --figure (future flag) still
    works in plotless envs. Currently always produces the figure since the PR
    description requires it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    vmin = float(result.freqs_arr[0].min())
    vmax = float(result.freqs_arr[0].max())

    for idx, ax in enumerate(axs.flat):
        if idx >= len(labels):
            ax.axis("off")
            continue
        spec = result.freqs_arr[idx]
        # pLSFF/pLSFF_quad have huge amplitudes (~1e6); clip to ref's range so
        # the other panels remain visible.
        clipped = np.clip(spec, vmin, vmax)
        im = ax.imshow(clipped, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"{labels[idx]} (L2={result.errs[idx]:.2f})", fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("l")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Idealised isosceles — {result.num_modes} unique modes "
        f"(reference vmin/vmax for color)",
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
