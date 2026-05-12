"""NetCDF-aware comparator for reproducibility fixtures.

Loads a fixture directory (``output.nc`` + ``manifest.yml``) and compares
against an actual ``dict[str, np.ndarray]`` produced by re-running a pipeline.
Per-variable ``rtol``/``atol`` from the manifest drive pass/fail; the stored
SHA256 is a warning-only tripwire that flags when bit-exactness drifts while
results are still within tolerance.

Extends ``tests/conftest.py``'s ``BaselineComparison`` concept; uses
``netCDF4`` (already a pyCSA runtime dep) rather than introducing xarray.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import netCDF4 as nc  # type: ignore[import-untyped]

from tests.reproducibility.manifest import Manifest, compute_sha256


@dataclass
class ComparisonResult:
    fixture: str
    passed: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    drifted: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.missing

    def render(self) -> str:
        lines = [f"Reproducibility check: {self.fixture}"]
        lines.append(
            f"  passed: {len(self.passed)}  failed: {len(self.failed)}  "
            f"drifted: {len(self.drifted)}  missing: {len(self.missing)}  "
            f"extra: {len(self.extra)}"
        )
        for f in self.failed:
            lines.append(f"  FAIL [{f['name']}]: {f['reason']}")
        for d in self.drifted:
            lines.append(
                f"  DRIFT [{d['name']}]: bit-exactness changed but within tolerance"
            )
        for m in self.missing:
            lines.append(f"  MISSING [{m}]: expected in fixture, not in actual")
        for e in self.extra:
            lines.append(f"  EXTRA [{e}]: in actual but not in fixture (ignored)")
        return "\n".join(lines)


def load_netcdf(path: Path | str) -> dict[str, np.ndarray]:
    """Read a NetCDF file into a flat ``name -> ndarray`` dict.

    Variables are loaded eagerly (no Dask). Masked arrays are filled with NaN
    for floats and the variable's ``_FillValue`` otherwise — this matches how
    pyCSA pipelines treat masked regions.
    """
    out: dict[str, np.ndarray] = {}
    with nc.Dataset(str(path), "r") as ds:
        for name, var in ds.variables.items():
            arr = var[...]
            if hasattr(arr, "filled"):
                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.filled(np.nan)
                else:
                    arr = arr.filled()
            out[name] = np.asarray(arr)
    return out


def save_netcdf(path: Path | str, variables: dict[str, np.ndarray]) -> None:
    """Write a flat ``name -> ndarray`` dict as a NetCDF file.

    Dimensions are auto-named ``<var>_dim_<i>`` and unshared. This is fine for
    fixture purposes; the file is not meant to be human-edited or merged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(str(path), "w", format="NETCDF4") as ds:
        for name, arr in variables.items():
            arr = np.asarray(arr)
            dim_names = []
            for i, sz in enumerate(arr.shape):
                dn = f"{name}_dim_{i}"
                ds.createDimension(dn, sz)
                dim_names.append(dn)
            if arr.ndim == 0:
                v = ds.createVariable(name, arr.dtype, ())
                v[...] = arr
            else:
                v = ds.createVariable(name, arr.dtype, tuple(dim_names))
                v[...] = arr


class NetCDFComparator:
    def __init__(self, fixture_dir: Path | str):
        self.fixture_dir = Path(fixture_dir)
        self.manifest = Manifest.load(self.fixture_dir / "manifest.yml")
        self.expected = load_netcdf(self.fixture_dir / "output.nc")

    def compare(self, actual: dict[str, np.ndarray]) -> ComparisonResult:
        result = ComparisonResult(fixture=self.manifest.fixture)

        expected_names = set(self.manifest.variables)
        actual_names = set(actual)

        result.missing = sorted(expected_names - actual_names)
        result.extra = sorted(actual_names - expected_names)

        for name in sorted(expected_names & actual_names):
            spec = self.manifest.variables[name]
            exp = self.expected[name]
            act = np.asarray(actual[name])

            if list(act.shape) != list(spec.shape):
                result.failed.append(
                    {
                        "name": name,
                        "reason": f"shape mismatch: expected {spec.shape}, got {list(act.shape)}",
                    }
                )
                continue
            if str(act.dtype) != spec.dtype:
                # Dtype mismatch is a soft failure — cast and continue, but flag.
                result.failed.append(
                    {
                        "name": name,
                        "reason": f"dtype mismatch: expected {spec.dtype}, got {act.dtype}",
                    }
                )
                continue

            try:
                np.testing.assert_allclose(
                    act,
                    exp,
                    rtol=spec.rtol,
                    atol=spec.atol,
                    equal_nan=spec.equal_nan,
                )
            except AssertionError as e:
                result.failed.append({"name": name, "reason": str(e).strip()})
                continue

            actual_sha = compute_sha256(act)
            if actual_sha != spec.sha256:
                result.drifted.append(
                    {
                        "name": name,
                        "expected_sha": spec.sha256,
                        "actual_sha": actual_sha,
                    }
                )
                warnings.warn(
                    f"[{self.manifest.fixture}/{name}] bit-exactness drifted "
                    f"(within rtol={spec.rtol}, atol={spec.atol}): "
                    f"{spec.sha256[:12]}... -> {actual_sha[:12]}...",
                    stacklevel=2,
                )
            result.passed.append(name)

        return result
