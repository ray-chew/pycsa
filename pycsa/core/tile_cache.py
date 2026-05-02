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

logger = logging.getLogger(__name__)


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

        # Pre-load all tiles
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
                # Open NetCDF file (keep it open for fast access)
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

        # Handle dateline crossing
        crosses_dateline = (lon_max - lon_min) > 180.0
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
