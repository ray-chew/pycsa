"""Manifest schema and YAML I/O for reproducibility fixtures.

A manifest pins the expected output of a captured pipeline run. Per variable
it records ``rtol``/``atol`` (the comparator gate) and ``sha256`` (a warning-
only tripwire that signals bit-exactness drift while staying within tolerance).

The schema is intentionally flat and stdlib-only (PyYAML + dataclasses) so
manifests are diffable in code review.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8


def compute_sha256(arr: np.ndarray) -> str:
    """SHA256 of an array's canonical little-endian byte representation.

    Force-converts to little-endian and contiguous memory so the hash is
    stable across platforms.
    """
    canonical = np.ascontiguousarray(arr).astype(
        arr.dtype.newbyteorder("<"), copy=False
    )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def summarize(arr: np.ndarray) -> dict[str, Any]:
    """Quick human-readable stats for the manifest's ``summary`` block."""
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        return {"size": 0}
    if flat.ndim == 0 or flat.size == 1:
        return {"value": float(flat.item()) if not np.isnan(flat.item()) else "nan"}
    finite = flat[np.isfinite(flat)]
    return {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "nan_count": int(np.isnan(flat).sum()),
        "size": int(flat.size),
    }


@dataclass
class VariableManifest:
    dtype: str
    shape: list[int]
    sha256: str
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    equal_nan: bool = True
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        equal_nan: bool = True,
    ) -> "VariableManifest":
        return cls(
            dtype=str(np.asarray(arr).dtype),
            shape=list(np.asarray(arr).shape),
            sha256=compute_sha256(np.asarray(arr)),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            summary=summarize(np.asarray(arr)),
        )


@dataclass
class CapturedFrom:
    git_sha: str
    python: str
    numpy: str
    scipy: str = ""

    @classmethod
    def collect(cls) -> "CapturedFrom":
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sha = "unknown"
        try:
            import scipy  # local import; scipy may not be present in all envs

            scipy_v = scipy.__version__
        except ImportError:
            scipy_v = ""
        return cls(
            git_sha=sha,
            python=sys.version.split()[0],
            numpy=np.__version__,
            scipy=scipy_v,
        )


@dataclass
class Manifest:
    fixture: str
    captured_at: str
    captured_from: CapturedFrom
    variables: dict[str, VariableManifest] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def build(
        cls,
        fixture: str,
        variables: dict[str, np.ndarray],
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        equal_nan: bool = True,
        notes: str = "",
    ) -> "Manifest":
        return cls(
            fixture=fixture,
            captured_at=_dt.datetime.now(_dt.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            captured_from=CapturedFrom.collect(),
            variables={
                name: VariableManifest.from_array(
                    arr, rtol=rtol, atol=atol, equal_nan=equal_nan
                )
                for name, arr in variables.items()
            },
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "captured_at": self.captured_at,
            "captured_from": dataclasses.asdict(self.captured_from),
            "variables": {
                name: dataclasses.asdict(v) for name, v in self.variables.items()
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        return cls(
            fixture=data["fixture"],
            captured_at=data["captured_at"],
            captured_from=CapturedFrom(**data["captured_from"]),
            variables={
                name: VariableManifest(**v) for name, v in data["variables"].items()
            },
            notes=data.get("notes", ""),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=None)

    @classmethod
    def load(cls, path: Path | str) -> "Manifest":
        with Path(path).open() as f:
            return cls.from_dict(yaml.safe_load(f))
