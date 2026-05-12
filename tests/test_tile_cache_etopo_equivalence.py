"""Byte-equivalence test for TopographyTileCache.get_etopo_data vs read_etopo_topo.

The cache's ETOPO path is a port of pycsa.core.io.read_etopo_topo.get_topo. This
test loads representative ICON cells via both paths and asserts the returned
(lat, lon, topo) arrays are identical. Run with:

    pytest tests/test_tile_cache_etopo_equivalence.py -v

Skips automatically if data/etopo_15s/ is missing.
"""

from pathlib import Path

import numpy as np
import pytest

from pycsa.core import io as pcio, utils, var
from pycsa import local_paths
from pycsa.core.tile_cache import TopographyTileCache, compute_split_EW

ETOPO_DIR = Path(local_paths.paths.etopo)
ICON_GRID = local_paths.paths.icon_grid


pytestmark = pytest.mark.skipif(
    not ETOPO_DIR.exists() or not Path(ICON_GRID).exists(),
    reason="ETOPO tiles or ICON grid not available locally",
)


# Representative cells covering each branch of the ETOPO loader.
# Each tuple is (c_idx, description).
TEST_CELLS = [
    (1086, "typical non-dateline mid-latitude (lat ~76°N)"),
    (2311, "Aleutians — false-positive dateline (all-negative lons near -176°)"),
    (1074, "genuine dateline crossing (split_EW=True, lat ~80°N)"),
    (17408, "extreme south polar (lat -88.90°S, exercises lat_idx_rng generation)"),
]


@pytest.fixture(scope="module")
def grid():
    """Load the ICON grid once and reuse across cells."""
    g = var.grid()
    pcio.ncdata().read_dat(ICON_GRID, g)
    return g


@pytest.fixture(scope="module")
def params():
    """Minimal params object with what read_etopo_topo needs."""
    p = var.obj()
    p.path_etopo = str(ETOPO_DIR) + "/"
    p.etopo_cg = 4  # matches the default coarse-graining used by the global run
    p.lat_extent = np.array([0.0, 0.0])  # placeholder; set per-cell
    p.lon_extent = np.array([0.0, 0.0])
    return p


def _load_via_reader(grid, params, c_idx):
    """Reference path: pycsa.core.io.read_etopo_topo."""
    lat_verts = np.degrees(grid.clat_vertices[c_idx])
    lon_verts = np.degrees(grid.clon_vertices[c_idx])
    lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)
    params.lat_extent = lat_extent
    params.lon_extent = lon_extent

    topo = var.topo_cell()
    reader = pcio.ncdata().read_etopo_topo(None, params, is_parallel=True)
    reader.get_topo(topo)
    return topo, reader.split_EW, lat_extent, lon_extent


def _load_via_cache(cache, params, lat_extent, lon_extent):
    """Candidate path: TopographyTileCache.get_etopo_data."""
    lat, lon, topo = cache.get_etopo_data(
        lat_extent, lon_extent, etopo_cg=params.etopo_cg
    )
    return lat, lon, topo


@pytest.fixture(scope="module")
def cache():
    """Build a single lazy ETOPO cache used across all cells."""
    return TopographyTileCache(
        data_dir=str(ETOPO_DIR),
        tile_filenames=[],
        dataset_type="ETOPO",
        verbose=False,
    )


def test_worker_cache_lifecycle(grid, params):
    """init_worker_cache / get_worker_cache / close_worker_cache happy path.

    This mirrors what do_cell does inside a Dask worker process: the main
    loop calls client.run(init_worker_cache, ...), then each cell's do_cell
    call retrieves the cache via get_worker_cache().
    """
    from pycsa.core import tile_cache as tc

    # No cache should be initialised yet (or from a prior test).
    tc.close_worker_cache()
    with pytest.raises(RuntimeError):
        tc.get_worker_cache()

    assert tc.init_worker_cache(str(ETOPO_DIR), "ETOPO") is True
    cache = tc.get_worker_cache()
    assert cache.dataset_type == "ETOPO"

    # Idempotency: second init with same dir should be a no-op (same object).
    assert tc.init_worker_cache(str(ETOPO_DIR), "ETOPO") is True
    assert tc.get_worker_cache() is cache

    # Functional check: retrieve topo for one cell through the worker-cache
    # path; should match reader output (this is the same contract used by
    # the wired do_cell).
    c_idx = 1086
    topo_ref, _, lat_extent, lon_extent = _load_via_reader(grid, params, c_idx)
    lat, lon, topo_arr = cache.get_etopo_data(
        lat_extent, lon_extent, etopo_cg=params.etopo_cg
    )
    np.testing.assert_array_equal(topo_arr, topo_ref.topo)

    # Cleanup leaves get_worker_cache failing again.
    tc.close_worker_cache()
    with pytest.raises(RuntimeError):
        tc.get_worker_cache()


@pytest.mark.parametrize("c_idx,description", TEST_CELLS)
def test_etopo_equivalence(grid, params, cache, c_idx, description):
    """Cache output must match the reference reader byte-for-byte for every cell."""
    topo_ref, split_EW_ref, lat_extent, lon_extent = _load_via_reader(
        grid, params, c_idx
    )
    lat_cache, lon_cache, topo_cache = _load_via_cache(
        cache, params, lat_extent, lon_extent
    )

    # The free-function dateline detector must agree with the reader's own
    # internal flag for the same vertex set.
    assert (
        compute_split_EW(lon_extent) == split_EW_ref
    ), f"cell {c_idx}: compute_split_EW disagrees with reader ({description})"

    np.testing.assert_array_equal(
        lat_cache,
        topo_ref.lat,
        err_msg=f"cell {c_idx}: lat arrays differ ({description})",
    )
    np.testing.assert_array_equal(
        lon_cache,
        topo_ref.lon,
        err_msg=f"cell {c_idx}: lon arrays differ ({description})",
    )
    np.testing.assert_array_equal(
        topo_cache,
        topo_ref.topo,
        err_msg=f"cell {c_idx}: topo arrays differ ({description})",
    )
