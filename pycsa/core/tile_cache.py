"""
Topography tile caching system for efficient parallel processing.

This module provides a caching layer for MERIT/ETOPO topography tiles to avoid
repeatedly opening/closing NetCDF files during parallel cell processing.
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from pycsa.core.io import _NETCDF_GLOBAL_LOCK
from pycsa.core import utils

logger = logging.getLogger(__name__)


# ETOPO 2022 15 arc-second tile grid (15° spacing in both lat and lon)
_ETOPO_FN_LON = np.array([
    -180, -165, -150, -135, -120, -105, -90, -75, -60, -45, -30, -15,
    0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180
])
_ETOPO_FN_LAT = np.array([90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90])


def compute_split_EW(lon_verts: np.ndarray) -> bool:
    """Determine whether a cell's longitude extent truly crosses the dateline.

    Uses the robust span-comparison formula: a true crossing occurs only when
    converting to the [0, 360) representation reduces the span AND the original
    span exceeds 180°. This avoids the false positives that plagued cells in
    the western hemisphere near the dateline (e.g. Aleutian cells).
    """
    lon_verts = np.asarray(lon_verts)
    lon_span = lon_verts.max() - lon_verts.min()
    lon_verts_360 = np.where(lon_verts < 0.0, lon_verts + 360.0, lon_verts)
    span_360 = lon_verts_360.max() - lon_verts_360.min()
    return bool((span_360 < lon_span) and (lon_span > 180.0))


def _etopo_NSEW(vert: float, typ: str) -> str:
    """N/S for latitude, E/W for longitude with the +180° → 'W' convention."""
    if typ == "lat":
        return "N" if vert >= 0.0 else "S"
    # longitude — note ETOPO's quirk: 180° always uses 'W' (since 180°E ≡ 180°W)
    if vert == 180.0:
        return "W"
    return "E" if vert >= 0.0 else "W"


def _etopo_tile_filename(lat_bound: float, lon_bound: float) -> str:
    """ETOPO 2022 15s tile filename for the (lat, lon) tile origin."""
    return "ETOPO_2022_v1_15s_%s%.2d%s%.3d_surface.nc" % (
        _etopo_NSEW(lat_bound, "lat"),
        np.abs(int(lat_bound)),
        _etopo_NSEW(lon_bound, "lon"),
        np.abs(int(lon_bound)),
    )


class TopographyTileCache:
    """
    Cache for topography data tiles.

    Pre-loads all required MERIT/ETOPO/REMA tiles into memory and provides
    fast access to subsets for individual grid cells.

    This dramatically speeds up parallel processing by avoiding repeated
    file I/O operations.

    Parameters
    ----------
    data_dir : str or Path
        Base directory containing topography data tiles
    tile_filenames : list of str
        List of tile filenames to pre-load
    dataset_type : str, optional
        Type of dataset ('MERIT', 'ETOPO', 'REMA'), by default 'MERIT'
    verbose : bool, optional
        Enable verbose logging, by default False

    Attributes
    ----------
    tiles : dict
        Dictionary mapping filenames to opened netCDF4.Dataset objects
    tile_bounds : dict
        Dictionary mapping filenames to (lat_min, lat_max, lon_min, lon_max) bounds
    """

    def __init__(
        self,
        data_dir: str,
        tile_filenames: List[str],
        dataset_type: str = 'MERIT',
        verbose: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.verbose = verbose

        # Cache dictionaries
        self.tiles: Dict[str, nc.Dataset] = {}
        self.tile_bounds: Dict[str, Tuple[float, float, float, float]] = {}
        self.tile_lats: Dict[str, np.ndarray] = {}
        self.tile_lons: Dict[str, np.ndarray] = {}

        # ETOPO with empty tile list = lazy mode: tiles open on first access via
        # get_etopo_data. MERIT keeps the existing eager pre-load behaviour.
        if dataset_type == 'ETOPO' and len(tile_filenames) == 0:
            return

        self._load_tiles(tile_filenames)

    def _load_tiles(self, filenames: List[str]):
        """Pre-load all tile files into memory."""
        logger.info(f"Pre-loading {len(filenames)} topography tiles...")

        for fn in filenames:
            filepath = self.data_dir / fn

            if not filepath.exists():
                logger.warning(f"Tile file not found: {filepath}")
                continue

            try:
                # Open NetCDF file under the shared HDF5 lock (HDF5 is not
                # thread-safe on this system — see pycsa/core/io.py).
                with _NETCDF_GLOBAL_LOCK:
                    ds = nc.Dataset(str(filepath), 'r')
                self.tiles[fn] = ds

                # Cache coordinate arrays
                lat = ds['lat'][:]
                lon = ds['lon'][:]
                self.tile_lats[fn] = lat
                self.tile_lons[fn] = lon

                # Cache bounds for quick lookup
                self.tile_bounds[fn] = (
                    float(lat.min()),
                    float(lat.max()),
                    float(lon.min()),
                    float(lon.max())
                )

                if self.verbose:
                    logger.debug(f"Loaded tile: {fn}")
                    logger.debug(f"  Bounds: lat[{lat.min():.2f}, {lat.max():.2f}], "
                               f"lon[{lon.min():.2f}, {lon.max():.2f}]")

            except Exception as e:
                logger.error(f"Failed to load tile {fn}: {e}")

    def get_data_for_region(
        self,
        lat_extent: np.ndarray,
        lon_extent: np.ndarray,
        merit_cg: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract topography data for a given lat/lon region.

        This is designed to be a drop-in replacement for the current
        read_merit_topo().get_topo() workflow.

        Parameters
        ----------
        lat_extent : array-like
            Latitude extent [lat_min, lat_max, ...]
        lon_extent : array-like
            Longitude extent [lon_min, lon_max, ...]
        merit_cg : int, optional
            Coarse-graining factor, by default 1

        Returns
        -------
        lat : ndarray
            Latitude coordinates
        lon : ndarray
            Longitude coordinates
        topo : ndarray
            Topography data (2D array)
        """
        lat_min = float(np.min(lat_extent))
        lat_max = float(np.max(lat_extent))
        lon_min = float(np.min(lon_extent))
        lon_max = float(np.max(lon_extent))

        # Handle dateline crossing — robust formula matching io.read_etopo_topo;
        # the old `(lon_max - lon_min) > 180.0` test false-positived on western
        # cells near the dateline (e.g. Aleutians).
        crosses_dateline = compute_split_EW(lon_extent)
        if crosses_dateline:
            lon_min = max(np.where(lon_extent < 0.0, lon_extent + 360.0, lon_extent)) - 360.0
            lon_max = min(np.where(lon_extent < 0.0, lon_extent + 360.0, lon_extent))

        # Find tiles that overlap with this region
        overlapping_tiles = self._find_overlapping_tiles(lat_min, lat_max, lon_min, lon_max)

        if not overlapping_tiles:
            logger.warning(f"No tiles found for region: lat[{lat_min}, {lat_max}], lon[{lon_min}, {lon_max}]")
            # Return empty arrays
            return np.array([]), np.array([]), np.zeros((0, 0))

        # Extract and merge data from overlapping tiles
        lat_data, lon_data, topo_data = self._merge_tiles(
            overlapping_tiles, lat_min, lat_max, lon_min, lon_max, crosses_dateline
        )

        # Apply coarse-graining if requested
        if merit_cg > 1:
            from pycsa.core import utils

            # Adjust for high-latitude regions
            iint = merit_cg
            if lat_max < -85.0:
                iint *= 5

            # Coarse-grain using sliding window
            lat_data = utils.sliding_window_view(
                np.sort(lat_data), (iint,), (iint,)
            ).mean(axis=-1)
            lon_data = utils.sliding_window_view(
                np.sort(lon_data), (iint,), (iint,)
            ).mean(axis=-1)
            topo_data = utils.sliding_window_view(
                topo_data, (iint, iint), (iint, iint)
            ).mean(axis=(-1, -2))[::-1, :]

        return lat_data, lon_data, topo_data

    def _find_overlapping_tiles(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ) -> List[str]:
        """Find all tiles that overlap with the given region."""
        overlapping = []

        for fn, (tile_lat_min, tile_lat_max, tile_lon_min, tile_lon_max) in self.tile_bounds.items():
            # Check for overlap
            lat_overlap = not (tile_lat_max < lat_min or tile_lat_min > lat_max)
            lon_overlap = not (tile_lon_max < lon_min or tile_lon_min > lon_max)

            if lat_overlap and lon_overlap:
                overlapping.append(fn)

        return overlapping

    def _merge_tiles(
        self,
        tile_filenames: List[str],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        crosses_dateline: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge data from multiple tiles into a single contiguous array.

        This handles the case where a cell region spans multiple MERIT/ETOPO tiles.
        """
        all_lats = []
        all_lons = []
        all_topos = []

        for fn in tile_filenames:
            ds = self.tiles[fn]
            lat = self.tile_lats[fn]
            lon = self.tile_lons[fn]

            # Find indices within requested bounds
            lat_mask = (lat >= lat_min) & (lat <= lat_max)
            lon_mask = (lon >= lon_min) & (lon <= lon_max)

            lat_idxs = np.where(lat_mask)[0]
            lon_idxs = np.where(lon_mask)[0]

            if len(lat_idxs) == 0 or len(lon_idxs) == 0:
                continue

            # Extract subset
            lat_subset = lat[lat_idxs]
            lon_subset = lon[lon_idxs]

            # Handle elevation variable name (MERIT uses "Elevation", ETOPO may use different)
            if 'Elevation' in ds.variables:
                elev_var = 'Elevation'
            elif 'elevation' in ds.variables:
                elev_var = 'elevation'
            elif 'z' in ds.variables:
                elev_var = 'z'
            else:
                # Try to find any elevation-like variable
                possible_names = ['topo', 'topography', 'height', 'dem']
                elev_var = None
                for name in possible_names:
                    if name in ds.variables:
                        elev_var = name
                        break
                if elev_var is None:
                    logger.error(f"Could not find elevation variable in tile {fn}")
                    continue

            with _NETCDF_GLOBAL_LOCK:
                topo_subset = ds[elev_var][lat_idxs[0]:lat_idxs[-1]+1, lon_idxs[0]:lon_idxs[-1]+1]

            all_lats.append(lat_subset)
            all_lons.append(lon_subset)
            all_topos.append(topo_subset)

        if not all_topos:
            return np.array([]), np.array([]), np.zeros((0, 0))

        # If only one tile, return directly
        if len(all_topos) == 1:
            return all_lats[0], all_lons[0], all_topos[0]

        # Otherwise, need to merge multiple tiles
        # For simplicity, concatenate and remove duplicates
        merged_lat = np.unique(np.concatenate(all_lats))
        merged_lon = np.unique(np.concatenate(all_lons))

        # Create output array
        merged_topo = np.zeros((len(merged_lat), len(merged_lon)))

        # Fill from tiles (simple approach - could be optimized)
        for i, lat_val in enumerate(merged_lat):
            for j, lon_val in enumerate(merged_lon):
                # Find which tile contains this point and extract value
                for k, fn in enumerate(tile_filenames):
                    if (lat_val in all_lats[k]) and (lon_val in all_lons[k]):
                        lat_idx = np.where(all_lats[k] == lat_val)[0][0]
                        lon_idx = np.where(all_lons[k] == lon_val)[0][0]
                        merged_topo[i, j] = all_topos[k][lat_idx, lon_idx]
                        break

        return merged_lat, merged_lon, merged_topo

    # ------------------------------------------------------------------
    # ETOPO path — byte-equivalent port of pycsa.core.io.read_etopo_topo
    # ------------------------------------------------------------------
    # The MERIT methods above (get_data_for_region, _find_overlapping_tiles,
    # _merge_tiles) stay MERIT-specific. ETOPO has a fixed 15° tile grid and
    # dateline handling that doesn't fit cleanly into bounds-based discovery,
    # so the ETOPO path uses its own discovery + assembly mirroring io.py.

    def _open_etopo_tile(self, fn: str) -> nc.Dataset:
        """Open an ETOPO tile on first access; cache the handle thereafter.

        Goes through _NETCDF_GLOBAL_LOCK because HDF5 is not thread-safe on
        the target system. Once opened, the handle (and its lat/lon coordinate
        arrays) stay cached for the lifetime of this TopographyTileCache.
        """
        if fn in self.tiles:
            return self.tiles[fn]
        filepath = str(self.data_dir / fn)
        with _NETCDF_GLOBAL_LOCK:
            ds = nc.Dataset(filepath, "r")
        self.tiles[fn] = ds
        # Coordinate arrays are small; cache so we don't re-read per cell.
        self.tile_lats[fn] = ds["lat"][:]
        self.tile_lons[fn] = ds["lon"][:]
        return ds

    @staticmethod
    def _etopo_compute_idx(vert: float, typ: str, direction: str, split_EW: bool) -> int:
        """Look up which ETOPO tile-boundary index encloses ``vert``.

        Mirrors pycsa.core.io.read_etopo_topo.__compute_idx (io.py:834-870).
        """
        fn_int = _ETOPO_FN_LON if direction == "lon" else _ETOPO_FN_LAT
        where_idx = int(np.argmin(np.abs(fn_int - vert)))

        if typ == "min":
            if (vert - fn_int[where_idx]) < 0.0:
                where_idx += -1 if direction == "lon" else 1
        elif typ == "max":
            if (vert - fn_int[where_idx]) > 0.0:
                if direction == "lon":
                    if not split_EW:
                        where_idx += 1
                else:
                    where_idx -= 1
            if (where_idx == len(fn_int) - 1) and split_EW:
                where_idx -= 1
        return int(where_idx)

    @staticmethod
    def _etopo_get_fns(lat_idx_rng: List[int], lon_idx_rng: List[int]) -> Tuple[List[str], int, int]:
        """Build ETOPO filenames for a rectangular tile range.

        Mirrors pycsa.core.io.read_etopo_topo.__get_fns (io.py:872-898).
        Returns (filenames, lon_cnt, lat_cnt) where the counts are the
        zero-based last enumerations (for __load_topo's row/col arithmetic).
        """
        fns: List[str] = []
        lon_cnt = 0
        lat_cnt = 0
        for lat_cnt, lat_idx in enumerate(lat_idx_rng):
            l_lat_bound = _ETOPO_FN_LAT[lat_idx]
            for lon_cnt, lon_idx in enumerate(lon_idx_rng):
                l_lon_bound = _ETOPO_FN_LON[lon_idx]
                fns.append(_etopo_tile_filename(l_lat_bound, l_lon_bound))
        return fns, lon_cnt, lat_cnt

    @staticmethod
    def _etopo_get_lon_idxs(
        lon: np.ndarray,
        lon_idx_rng: List[int],
        n_col: int,
        split_EW: bool,
        lon_verts: np.ndarray,
    ) -> Tuple[int, int]:
        """Compute per-tile longitude slice indices.

        Mirrors pycsa.core.io.read_etopo_topo.__get_lon_idxs (io.py:1052-1104).
        """
        l_lon_bound = _ETOPO_FN_LON[lon_idx_rng[n_col]]
        r_idx = lon_idx_rng[n_col] + 1
        if r_idx >= len(_ETOPO_FN_LON):
            r_idx = 1  # 180° wraps to -165° (skip index 0 = -180° duplicate)
        r_lon_bound = _ETOPO_FN_LON[r_idx]
        lon_rng = r_lon_bound - l_lon_bound

        lon_in_file = lon_verts[
            ((lon_verts - l_lon_bound) >= 0)
            & ((lon_verts - l_lon_bound) <= lon_rng)
        ]

        if len(lon_in_file) == 0:
            lon_high = int(np.argmin(np.abs(lon - r_lon_bound)))
            lon_low = int(np.argmin(np.abs(lon - l_lon_bound)))
            return lon_low, lon_high

        if not split_EW:
            if lon_in_file.max() == lon_verts.max():
                lon_high = int(np.argmin(np.abs(lon - lon_in_file.max())))
            else:
                lon_high = int(np.argmin(np.abs(lon - r_lon_bound)))
            if lon_in_file.min() == lon_verts.min():
                lon_low = int(np.argmin(np.abs(lon - lon_in_file.min())))
            else:
                lon_low = int(np.argmin(np.abs(lon - l_lon_bound)))
            return lon_low, lon_high

        # split_EW = True (dateline crossing)
        negative_lons = lon_verts[lon_verts < 0.0]
        lon_high = int(np.argmin(np.abs(lon - r_lon_bound)))
        lon_low = int(np.argmin(np.abs(lon - l_lon_bound)))
        if len(negative_lons) > 0:
            wrapped = np.where(lon_verts < 0.0, lon_verts + 360.0, lon_verts)
            if lon_in_file.max() == wrapped.min():
                lon_high = int(np.argmin(np.abs(lon - r_lon_bound)))
                lon_low = int(np.argmin(np.abs(lon - lon_in_file.min())))
            if lon_in_file.min() == (negative_lons.max() + 360.0 - 360.0):
                lon_high = int(np.argmin(np.abs(lon - lon_in_file.max())))
                lon_low = int(np.argmin(np.abs(lon - l_lon_bound)))
        return lon_low, lon_high

    def _etopo_load_topo(
        self,
        fns: List[str],
        lon_cnt: int,
        lat_cnt: int,
        lat_idx_rng: List[int],
        lon_idx_rng: List[int],
        lat_verts: np.ndarray,
        lon_verts: np.ndarray,
        split_EW: bool,
    ) -> Tuple[List[float], List[float], np.ndarray]:
        """Assemble the regional topography array from per-tile slices.

        Mirrors pycsa.core.io.read_etopo_topo.__load_topo (io.py:900-1050)
        as a two-pass over ``fns`` — first pass computes the output shape,
        second pass populates the array. Returns (lat_list, lon_list, topo).
        """
        # First pass: compute output shape (nc_lat, nc_lon).
        n_col = 0
        n_row = 0
        nc_lon = 0
        nc_lat = 0
        for fn in fns:
            ds = self._open_etopo_tile(fn)
            lat = self.tile_lats[fn]
            lon = self.tile_lons[fn]

            lat_min_idx = np.argmin(
                np.abs((lat - np.sign(lat) * 1e-4) - lat_verts.min())
            )
            lat_max_idx = np.argmin(
                np.abs((lat + np.sign(lat) * 1e-4) - lat_verts.max())
            )
            lat_high = int(max(lat_min_idx, lat_max_idx))
            lat_low = int(min(lat_min_idx, lat_max_idx))

            lon_low, lon_high = self._etopo_get_lon_idxs(
                lon, lon_idx_rng, n_col, split_EW, lon_verts
            )

            if n_row == 0:
                nc_lon += lon_high - lon_low
            if n_col == 0:
                nc_lat += lat_high - lat_low

            n_col += 1
            if n_col == (lon_cnt + 1):
                n_col = 0
                n_row += 1

        # Second pass: populate the array.
        topo_arr = np.zeros((nc_lat, nc_lon))
        cell_lat: List[float] = []
        cell_lon: List[float] = []
        n_col = 0
        n_row = 0
        lon_sz_old = 0
        lat_sz_old = 0
        for fn in fns:
            ds = self.tiles[fn]
            lat = self.tile_lats[fn]
            lon = self.tile_lons[fn]

            lat_min_idx = np.argmin(
                np.abs((lat - np.sign(lat) * 1e-4) - lat_verts.min())
            )
            lat_max_idx = np.argmin(
                np.abs((lat + np.sign(lat) * 1e-4) - lat_verts.max())
            )
            lat_high = int(max(lat_min_idx, lat_max_idx))
            lat_low = int(min(lat_min_idx, lat_max_idx))

            lon_low, lon_high = self._etopo_get_lon_idxs(
                lon, lon_idx_rng, n_col, split_EW, lon_verts
            )

            with _NETCDF_GLOBAL_LOCK:
                slab = ds["z"][lat_low:lat_high, lon_low:lon_high].data

            curr_lon = lon[lon_low:lon_high].data.tolist()
            if n_col == 0:
                cell_lat += lat[lat_low:lat_high].data.tolist()
            if n_row == 0:
                cell_lon += curr_lon

            lon_sz = lon_high - lon_low
            lat_sz = lat_high - lat_low
            topo_arr[
                lat_sz_old : lat_sz_old + lat_sz,
                lon_sz_old : lon_sz_old + lon_sz,
            ] = slab

            n_col += 1
            lon_sz_old += lon_sz
            if n_col == (lon_cnt + 1):
                n_col = 0
                lon_sz_old = 0
                n_row += 1
                lat_sz_old += lat_sz

        return cell_lat, cell_lon, topo_arr

    def get_etopo_data(
        self,
        lat_extent: np.ndarray,
        lon_extent: np.ndarray,
        etopo_cg: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load ETOPO topography for a cell's lat/lon vertex extent.

        Byte-equivalent to pycsa.core.io.read_etopo_topo.get_topo + __load_topo
        (io.py:720-1050), but uses this cache's persistent file handles so the
        same tile isn't re-opened across cells within a worker.

        Parameters
        ----------
        lat_extent : array-like
            Cell latitude vertices (1-D).
        lon_extent : array-like
            Cell longitude vertices (1-D), in [-180, 180).
        etopo_cg : int, optional
            Coarse-graining factor (stride). High southern latitudes
            (lat_max < -85°) implicitly multiply this by 5 — see below.

        Returns
        -------
        lat, lon, topo
            1-D coordinate arrays and the 2-D topography slab, sorted in
            ascending lat/lon. ``lon`` is in [0, 360) when the cell crosses
            the dateline; otherwise it stays in [-180, 180).
        """
        lat_verts = np.asarray(lat_extent)
        lon_verts = np.asarray(lon_extent)

        # Dateline detection (robust formula; see compute_split_EW).
        lon_span = lon_verts.max() - lon_verts.min()
        lon_verts_360 = np.where(lon_verts < 0.0, lon_verts + 360.0, lon_verts)
        span_360 = lon_verts_360.max() - lon_verts_360.min()
        split_EW = (span_360 < lon_span) and (lon_span > 180.0)

        # Determine longitude tile range — three branches: global / dateline / normal.
        if lon_span >= 360.0:
            split_EW = False
            lon_idx_rng = list(range(0, len(_ETOPO_FN_LON) - 1))
        elif split_EW:
            min_lon_360 = lon_verts_360.min()
            max_lon_360 = lon_verts_360.max()
            min_lon = min_lon_360 if min_lon_360 <= 180 else min_lon_360 - 360
            max_lon = max_lon_360 if max_lon_360 <= 180 else max_lon_360 - 360
            lon_min_idx = self._etopo_compute_idx(min_lon, "min", "lon", split_EW)
            lon_max_idx = self._etopo_compute_idx(max_lon, "max", "lon", split_EW)
            if lon_min_idx == lon_max_idx:
                lon_idx_rng = [lon_min_idx]
                if lon_min_idx >= len(_ETOPO_FN_LON) - 2:
                    lon_idx_rng.append(0)
            else:
                lon_idx_rng = (
                    list(range(lon_min_idx, len(_ETOPO_FN_LON) - 1))
                    + list(range(0, lon_max_idx + 1))
                )
        else:
            min_lon = lon_verts.min()
            max_lon = lon_verts.max()
            lon_min_idx = self._etopo_compute_idx(min_lon, "min", "lon", split_EW)
            lon_max_idx = self._etopo_compute_idx(max_lon, "max", "lon", split_EW)
            if lon_min_idx == lon_max_idx:
                lon_max_idx += 1
            lon_idx_rng = list(range(lon_min_idx, lon_max_idx))

        # Latitude tile range — same logic across all longitude branches.
        lat_min_tile_idx = self._etopo_compute_idx(lat_verts.min(), "min", "lat", split_EW)
        lat_max_tile_idx = self._etopo_compute_idx(lat_verts.max(), "max", "lat", split_EW)
        lat_idx_rng = list(range(lat_max_tile_idx, lat_min_tile_idx))

        # Build filenames; load + assemble.
        fns, lon_cnt, lat_cnt = self._etopo_get_fns(lat_idx_rng, lon_idx_rng)
        cell_lat, cell_lon, topo_arr = self._etopo_load_topo(
            fns, lon_cnt, lat_cnt, lat_idx_rng, lon_idx_rng,
            lat_verts, lon_verts, split_EW,
        )

        # Wrap longitudes if dateline-crossing, then sort lat/lon and reorder topo.
        lat_arr = np.array(cell_lat)
        lon_arr = np.array(cell_lon)
        if split_EW:
            lon_arr = np.where(lon_arr < 0.0, lon_arr + 360.0, lon_arr)

        lat_sort_idx = np.argsort(lat_arr)
        lon_sort_idx = np.argsort(lon_arr)
        lat_sorted = lat_arr[lat_sort_idx]
        lon_sorted = lon_arr[lon_sort_idx]
        topo_sorted = topo_arr[np.ix_(lat_sort_idx, lon_sort_idx)]

        # Coarse-graining — io.py picks up a 5× multiplier for very-southern cells.
        iint = etopo_cg
        if iint > 1:
            try:
                out_lat = utils.sliding_window_view(
                    lat_sorted, (iint,), (iint,)
                ).mean(axis=-1)
                out_lon = utils.sliding_window_view(
                    lon_sorted, (iint,), (iint,)
                ).mean(axis=-1)
                out_topo = utils.sliding_window_view(
                    topo_sorted, (iint, iint), (iint, iint)
                ).mean(axis=(-1, -2))
                return out_lat, out_lon, out_topo
            except (ValueError, MemoryError) as e:
                logger.warning(f"Coarse-graining failed ({e}); returning full resolution")
        return lat_sorted, lon_sorted, topo_sorted

    def close_all(self):
        """Close all opened NetCDF files."""
        for fn, ds in self.tiles.items():
            try:
                ds.close()
                if self.verbose:
                    logger.debug(f"Closed tile: {fn}")
            except Exception as e:
                logger.error(f"Error closing tile {fn}: {e}")

        self.tiles.clear()
        self.tile_bounds.clear()
        self.tile_lats.clear()
        self.tile_lons.clear()

    def __del__(self):
        """Ensure files are closed when cache is destroyed."""
        self.close_all()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure files are closed."""
        self.close_all()
        return False


def create_tile_cache_from_grid(
    grid,
    params,
    padding: float = 0.5
) -> TopographyTileCache:
    """
    Create a tile cache containing all tiles needed for a given grid.

    This analyzes the grid to determine which tiles are needed, then
    pre-loads them all at once.

    Parameters
    ----------
    grid : pycsa.core.var.grid
        ICON grid object with cell vertices
    params : pycsa.core.var.params
        Parameters object with path_merit, path_etopo, etc.
    padding : float, optional
        Extra padding in degrees to ensure tiles are loaded, by default 0.5

    Returns
    -------
    TopographyTileCache
        Initialized cache with all required tiles loaded
    """
    from pycsa.core import utils

    # Determine global bounds of the grid
    lat_min = np.min(grid.clat_vertices) - padding
    lat_max = np.max(grid.clat_vertices) + padding
    lon_min = np.min(grid.clon_vertices) - padding
    lon_max = np.max(grid.clon_vertices) + padding

    logger.info(f"Grid spans: lat[{lat_min:.2f}, {lat_max:.2f}], lon[{lon_min:.2f}, {lon_max:.2f}]")

    # Determine which tiles to load (using MERIT tile naming convention)
    # TODO: Implement automatic tile discovery based on bounds
    # For now, this is a placeholder - you'll need to implement the logic
    # to determine required tile filenames based on the grid bounds

    # Example: if using MERIT data with standard 30x30 degree tiles
    tile_filenames = _get_merit_tiles_for_bounds(lat_min, lat_max, lon_min, lon_max)

    logger.info(f"Loading {len(tile_filenames)} topography tiles for grid coverage")

    # Create and return cache
    return TopographyTileCache(
        data_dir=params.path_merit,
        tile_filenames=tile_filenames,
        dataset_type='MERIT',
        verbose=params.verbose if hasattr(params, 'verbose') else False
    )


def _get_merit_tiles_for_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> List[str]:
    """
    Determine MERIT tile filenames needed to cover the given bounds.

    MERIT tiles are 30x30 degrees and named like:
    MERIT_N60-N90_W180-W150.nc4
    """
    # MERIT tile boundaries (standard grid)
    merit_lat_bounds = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])
    merit_lon_bounds = np.array([-180.0, -150.0, -120.0, -90.0, -60.0, -30.0,
                                 0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])

    tile_filenames = []

    # Find lat tile indices
    lat_idx_min = np.searchsorted(merit_lat_bounds[::-1], lat_min, side='left')
    lat_idx_max = np.searchsorted(merit_lat_bounds[::-1], lat_max, side='right')

    # Find lon tile indices
    lon_idx_min = np.searchsorted(merit_lon_bounds, lon_min, side='left')
    lon_idx_max = np.searchsorted(merit_lon_bounds, lon_max, side='right')

    def _get_nsew(val, coord_type):
        """Get N/S/E/W tag for coordinate value."""
        if coord_type == 'lat':
            return 'N' if val >= 0 else 'S'
        else:  # lon
            return 'E' if val >= 0 else 'W'

    # Generate filenames
    for lat_idx in range(max(0, lat_idx_min-1), min(len(merit_lat_bounds)-1, lat_idx_max+1)):
        l_lat = merit_lat_bounds[lat_idx]
        r_lat = merit_lat_bounds[lat_idx + 1]
        l_lat_tag = _get_nsew(l_lat, 'lat')
        r_lat_tag = _get_nsew(r_lat, 'lat')

        for lon_idx in range(max(0, lon_idx_min-1), min(len(merit_lon_bounds)-1, lon_idx_max+1)):
            l_lon = merit_lon_bounds[lon_idx]
            r_lon = merit_lon_bounds[lon_idx + 1]
            l_lon_tag = _get_nsew(l_lon, 'lon')
            r_lon_tag = _get_nsew(r_lon, 'lon')

            # Check if this is REMA region (Antarctica)
            if l_lat == -60.0 and r_lat == -90.0:
                dataset_name = "REMA_BKG"
            else:
                dataset_name = "MERIT"

            filename = f"{dataset_name}_{l_lat_tag}{abs(int(l_lat)):02d}-{r_lat_tag}{abs(int(r_lat)):02d}_{l_lon_tag}{abs(int(l_lon)):03d}-{r_lon_tag}{abs(int(r_lon)):03d}.nc4"
            tile_filenames.append(filename)

    return tile_filenames
