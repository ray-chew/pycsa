"""Shared slicing helpers for the MERIT and ETOPO capture scripts.

The fixtures need *production-shape* miniatures of the real topography tiles
and ICON grid — same variable names, same filename pattern, same per-tile
dimensions — but small enough to commit to the repo (a few hundred KB total).

We do this by cropping each source tile to the target cell's lat/lon bbox
(plus a small padding) and pre-applying the production coarse-graining factor
via sliding-window mean. The reproducibility tests then run with
``merit_cg=1`` / ``etopo_cg=1`` so the loader's own coarse-graining step is a
no-op on the already-downsampled bundle. End-to-end output corresponds to
production-at-the-given-CG-factor, just with the heavy lifting moved from
load-time to capture-time.

The polar-latitude branch in the ETOPO loader (``iint *= 5`` at lat < -85°)
is preserved — for polar fixtures, pre-downsampling absorbs the base
``etopo_cg`` factor and the polar 5× multiplier still fires at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import netCDF4 as nc  # type: ignore[import-untyped]


def _trim_to_multiple(idx_lo: int, idx_hi: int, factor: int) -> tuple[int, int]:
    """Adjust [lo, hi) so the length is a positive multiple of ``factor``."""
    n = ((idx_hi - idx_lo) // factor) * factor
    if n < factor:
        n = factor  # always keep at least one window
    return idx_lo, idx_lo + n


def crop_and_downsample_tile(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    factor: int,
    topo_var: str,
    extra_vars: Iterable[str] = (),
    lat_pad: float = 1.0,
    lon_pad: float = 1.0,
) -> dict[str, int]:
    """Crop a topography tile to a lat/lon bbox and downsample by ``factor``.

    Writes a NetCDF at ``dst_path`` with the same ``lat``, ``lon``, and
    ``topo_var`` schema as the source, plus any ``extra_vars`` copied
    verbatim (e.g. ETOPO's ``crs`` scalar).

    ``factor`` applies sliding-window mean exactly like the loader's
    coarse-graining; afterwards the bundle is consumable with the loader's
    own factor set to 1. ``lat_pad`` / ``lon_pad`` (degrees) leave room
    around the bbox so the loader's ``argmin``-based extent matching still
    finds clean boundaries.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(str(src_path), "r") as src:
        lat = np.asarray(src["lat"][:])
        lon = np.asarray(src["lon"][:])

        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range
        # Index masks work the same whether lat is ascending or descending.
        lat_mask = (lat >= lat_min - lat_pad) & (lat <= lat_max + lat_pad)
        lon_mask = (lon >= lon_min - lon_pad) & (lon <= lon_max + lon_pad)

        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            raise ValueError(
                f"No points in {src_path.name} cover lat={lat_range} lon={lon_range}"
            )

        lat_lo, lat_hi = _trim_to_multiple(lat_idx[0], lat_idx[-1] + 1, factor)
        lon_lo, lon_hi = _trim_to_multiple(lon_idx[0], lon_idx[-1] + 1, factor)

        lat_crop = lat[lat_lo:lat_hi]
        lon_crop = lon[lon_lo:lon_hi]

        # MERIT stores ocean cells as masked with _FillValue=-32767. The
        # production loader assigns the masked-array slice into a plain ndarray,
        # which fills masked positions with the variable's fill value, and the
        # downstream pipeline clamps via `topo.topo[topo.topo < -500] = -500`.
        # To match that, fill the mask with the variable's fill value BEFORE
        # the sliding-window mean — so mixed-coverage windows carry the same
        # -32767 contamination that production sees, and a follow-up clamp on
        # the test side produces the same -500 values.
        topo_var_obj = src[topo_var]
        fill_value = getattr(topo_var_obj, "_FillValue", None)
        raw = topo_var_obj[lat_lo:lat_hi, lon_lo:lon_hi]
        if np.ma.isMaskedArray(raw):
            raw = raw.filled(fill_value if fill_value is not None else 0)
        topo_crop = np.asarray(raw)

        if factor > 1:
            new_lat = lat_crop.reshape(-1, factor).mean(axis=-1)
            new_lon = lon_crop.reshape(-1, factor).mean(axis=-1)
            new_topo = (
                topo_crop.astype("float64")
                .reshape(new_lat.size, factor, new_lon.size, factor)
                .mean(axis=(1, 3))
            )
        else:
            new_lat, new_lon, new_topo = lat_crop, lon_crop, topo_crop

        topo_dtype = src[topo_var].dtype

        with nc.Dataset(str(dst_path), "w", format="NETCDF4") as dst:
            dst.createDimension("lat", new_lat.size)
            dst.createDimension("lon", new_lon.size)
            v_lat = dst.createVariable("lat", lat.dtype, ("lat",))
            v_lat[:] = new_lat.astype(lat.dtype)
            v_lon = dst.createVariable("lon", lon.dtype, ("lon",))
            v_lon[:] = new_lon.astype(lon.dtype)
            v_topo = dst.createVariable(topo_var, topo_dtype, ("lat", "lon"))
            v_topo[:] = new_topo.astype(topo_dtype)

            for ev in extra_vars:
                if ev in src.variables:
                    vsrc = src[ev]
                    vdst = dst.createVariable(ev, vsrc.dtype, vsrc.dimensions)
                    vdst[...] = vsrc[...]

            dst.setncattr(
                "history",
                f"Cropped from {src_path.name} to lat={lat_range}, lon={lon_range}, "
                f"downsampled by {factor}x via sliding-window mean.",
            )

    return {
        "lat_points": int(new_lat.size),
        "lon_points": int(new_lon.size),
        "factor": int(factor),
    }


def subset_icon_grid(
    src_path: Path | str,
    dst_path: Path | str,
    cell_indices: list[int],
) -> dict[str, int]:
    """Write a miniature ICON grid containing only ``cell_indices``.

    Cells are renumbered 0..N-1 in the output. The pipeline references cells
    by integer index, so consumers using this miniature grid should refer to
    the new index, not the original ``c_idx``.

    Only cell-centred variables (sized ``cell``) and their vertex companions
    (sized ``cell × nv``) are subset. Other dimensions are dropped, on the
    premise that the regional MERIT / single-cell ETOPO pipelines don't read
    vertex- or edge-centred fields.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(cell_indices, dtype=np.int64)

    with nc.Dataset(str(src_path), "r") as src:
        cell_dim = src.dimensions["cell"].name
        # nv (vertices per cell) varies by ICON setup; read from a known var.
        nv = src.variables["clon_vertices"].shape[1]

        with nc.Dataset(str(dst_path), "w", format="NETCDF4") as dst:
            dst.createDimension("cell", len(idx))
            dst.createDimension("nv", nv)

            for name, var in src.variables.items():
                dims = var.dimensions
                if dims == (cell_dim,):
                    out = dst.createVariable(name, var.dtype, ("cell",))
                    out[:] = var[idx]
                elif len(dims) == 2 and dims[0] == cell_dim and var.shape[1] == nv:
                    out = dst.createVariable(name, var.dtype, ("cell", "nv"))
                    out[:] = var[idx, :]
                # other-shaped vars are skipped

            dst.setncattr(
                "history",
                f"Subset of {src_path.name} keeping cells {list(map(int, idx))}, "
                f"renumbered 0..{len(idx) - 1}.",
            )

    return {"n_cells": int(len(idx)), "original_indices": [int(i) for i in idx]}
