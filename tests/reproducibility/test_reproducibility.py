"""Reproducibility gate — re-runs each captured pipeline and compares against
its fixture's ``manifest.yml``.

This test gates the Phase B refactor PRs (#3–#6). A failing variable means
either:

* the refactor changed numerics in an unexpected way (revert or investigate); or
* the change is intentional and the fixture needs a deliberate update
  (re-run the capture script, embed the fresh figure in a fixture-update PR,
  get human sign-off, then merge).

All three cases gate in CI — input data for the MERIT and ETOPO cases is
bundled in production schema next to each fixture's ``manifest.yml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.reproducibility.comparator import NetCDFComparator

FIXTURES_DIR = Path(__file__).parent / "fixtures"


CASES: list[dict] = [
    {"name": "idealised", "needs_local_data": False},
    {"name": "regional_merit", "needs_local_data": True},
    {"name": "etopo_single_cell", "needs_local_data": True},
]


def _run_idealised():
    import numpy as np
    from runs.idealised_isosceles import run

    result = run()
    return {
        "freqs_arr": result.freqs_arr,
        "errs": result.errs,
        "sums": result.sums,
        "sum_errs": result.sum_errs,
        "freqs_ref": result.freqs_ref,
        "num_modes": np.array(result.num_modes, dtype=np.int64),
    }


def _run_regional_merit():
    from tests.reproducibility.capture.capture_regional_merit import _run_pipeline

    return _run_pipeline(FIXTURES_DIR / "regional_merit")


def _run_etopo_single_cell():
    from tests.reproducibility.capture.capture_etopo_single_cell import _run_pipeline

    return _run_pipeline(FIXTURES_DIR / "etopo_single_cell")


RUNNERS = {
    "idealised": _run_idealised,
    "regional_merit": _run_regional_merit,
    "etopo_single_cell": _run_etopo_single_cell,
}


@pytest.mark.parametrize("case", [c["name"] for c in CASES])
def test_reproducibility(case, patch_local_paths_for):
    case_dir = FIXTURES_DIR / case
    if not (case_dir / "manifest.yml").exists():
        pytest.skip(f"fixture {case} not yet captured")

    case_spec = next(c for c in CASES if c["name"] == case)
    if case_spec["needs_local_data"]:
        patch_local_paths_for(case_dir)

    actual = RUNNERS[case]()
    result = NetCDFComparator(case_dir).compare(actual)

    assert result.ok, "\n" + result.render()
