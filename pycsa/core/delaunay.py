"""Delaunay decomposition and land-cell selection.

Partitions a lat-lon topography domain into Delaunay triangles whose
vertices sit on a coarse regular grid (:func:`get_decomposition`), and
selects the subset of triangles that contain enough above-threshold
topography to be treated as land cells (:func:`get_land_cells`). The
``scipy.spatial.Delaunay`` object returned by :func:`get_decomposition`
is augmented in place with ``.tri_lat_verts`` / ``.tri_lon_verts`` (and
centroid arrays) that downstream consumers, including
:func:`get_land_cells`, rely on.
"""

import logging

import numpy as np
from scipy.spatial import Delaunay

from pycsa.core import utils, var

logger = logging.getLogger(__name__)


def get_decomposition(topo, xnp=11, ynp=6, padding=0):
    """
    Partitions a lat-lon domain into a number of coarser but regularly spaced points that comprises the vertices of the Delaunay triangles.

    Parameters
    ----------
    topo : array-like
        2D topography data
    xnp : int, optional
        number of points in the first horizontal direction, by default 11
    ynp : int, optional
        number of points in the second horizontal direction, by default 6
    padding : int, optional
        number of grid points to include as a boundary (padded) region, by default 0

    Returns
    -------
    :class:`scipy.spatial.Delaunay` instance
        scipy Delaunay triangulation instance, augmented in place with
        ``tri_lat_verts``/``tri_lon_verts`` (per-triangle vertex
        coordinates) and ``tri_clats``/``tri_clons`` (centroids).
    """

    xlen = len(topo.lon) - padding
    ylen = len(topo.lat) - padding
    xPoints = np.linspace(padding, xlen - 1, xnp)
    yPoints = np.linspace(padding, ylen - 1, ynp)

    YY, XX = np.meshgrid(yPoints, xPoints)

    # Now we get the points by index.
    points = np.array([list(item) for item in zip(XX.ravel(), YY.ravel())]).astype(
        "int"
    )

    lat_verts = topo.lat_grid[points[:, 1], points[:, 0]]
    lon_verts = topo.lon_grid[points[:, 1], points[:, 0]]

    # Using these indices, we get the list of points in (lon,lat).
    points = np.array([list(item) for item in zip(lon_verts, lat_verts)])

    lats = points[:, 1]
    lons = points[:, 0]

    # Using scipy spatial, we setup the Delaunay decomposition
    tri = Delaunay(points)

    # Convert the vertices of the simplices to lat-lon values.
    tri.tri_lat_verts = lats[tri.simplices]
    tri.tri_lon_verts = lons[tri.simplices]

    logger.info("Delaunay triangulation object created.")
    logger.info("Number of triangles = %d", len(tri.tri_lat_verts))

    # Compute the centroid for each vertex.
    tri.tri_clats = tri.tri_lat_verts.sum(axis=1) / 3.0
    tri.tri_clons = tri.tri_lon_verts.sum(axis=1) / 3.0

    return tri


def get_land_cells(tri, topo, height_tol=0.5, percent_tol=0.95):
    """
    Land cell selector based on how much of a grid cell contains topography of a certain elevation.

    Parameters
    ----------
    tri : :class:`scipy.spatial.Delaunay`
        Triangulation as returned by :func:`get_decomposition`. Must
        carry the ``.tri_lat_verts`` / ``.tri_lon_verts`` per-triangle
        vertex-coordinate arrays that :func:`get_decomposition` attaches.
    topo : array-like
        2D topographic data
    height_tol : float, optional
        elevation above `height_tol` is considered land, by default 0.5 [m]
    percent_tol : float, optional
        Maximum fraction of a cell that may sit at or below `height_tol`
        before the cell is rejected as ocean, by default 0.95. The cell
        is dropped when more than `percent_tol` of its grid points are at
        or below `height_tol`
        (``(topo <= height_tol).sum() / size > percent_tol``) and kept as
        a land cell otherwise.

    Returns
    -------
    list
        list of land cell indices. Only even-indexed triangles
        (``range(n_tri)[::2]``) are evaluated; odd-indexed triangles are
        skipped because each grid quad is split into two triangles and
        the pair shares the same topography footprint.
    """
    rect_set = []
    n_tri = len(tri.tri_lat_verts)

    for tri_idx in range(n_tri)[::2]:
        cell = var.topo_cell()

        logger.debug("computing idx: %d", tri_idx)

        simplex_lat = tri.tri_lat_verts[tri_idx]
        simplex_lon = tri.tri_lon_verts[tri_idx]

        utils.get_lat_lon_segments(
            simplex_lat, simplex_lon, cell, topo, load_topo=True, filtered=False
        )

        if not (((cell.topo <= height_tol).sum() / cell.topo.size) > percent_tol):
            rect_set.append(tri_idx)

    return rect_set
