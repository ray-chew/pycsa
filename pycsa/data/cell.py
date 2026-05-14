"""Grid and topography data classes.

Moved from ``pycsa.core.var`` and refactored to use ``@dataclass`` for
explicit field declarations. Behavior is preserved including the
``grid.apply_f`` convenience, which is now driven by a ``ClassVar``
exclusion list instead of a runtime-set instance attribute.

``topo_cell`` inherits its on-disk fields (``lon``, ``lat``, ``topo``,
``analysis``) from ``topo`` and adds methods that set further runtime
attributes (``lon_grid``, ``lat_grid``, ``mask``, ``topo_m``, …) — these
are not declared as dataclass fields because they're side products of
``gen_mgrids`` / ``get_masked`` / ``get_grad_topo`` rather than
construction arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from pycsa.core import utils


@dataclass
class grid:
    """ICON triangular grid: cell centers + vertex coordinates + areas.

    ``links`` is a lookup table mapping cells to their topography source
    file; ``cell_area`` is the per-cell area in m². Both are skipped by
    :meth:`apply_f` because they are not (lat, lon)-style data needing
    unit conversion.
    """

    clat: np.ndarray | None = None
    clat_vertices: np.ndarray | None = None
    clon: np.ndarray | None = None
    clon_vertices: np.ndarray | None = None
    links: np.ndarray | None = None
    cell_area: np.ndarray | None = None

    # Field names that ``apply_f`` skips. Was a runtime-set instance list
    # in the old class; now a ClassVar exclusion set, so the data fields
    # and the apply_f exclusion list are in one place.
    NON_CONVERTIBLES: ClassVar[tuple[str, ...]] = ("links", "cell_area")

    def apply_f(self, f):
        """Apply ``f`` to each (lat, lon)-style attribute.

        Skips :attr:`NON_CONVERTIBLES`. Used in the wild as
        ``grid.apply_f(utils.rad2deg)`` to convert radians to degrees
        after loading from a NetCDF.

        Parameters
        ----------
        f : callable
            Function applied in-place to each non-skipped, non-``None``
            field.
        """
        for name in self.__dataclass_fields__:
            if name in self.NON_CONVERTIBLES:
                continue
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, f(value))


@dataclass
class topo:
    """Topography container: 1-D lat/lon arrays + 2-D elevation + analysis.

    The ``analysis`` field is left ``Any`` because it holds a
    :class:`pycsa.data.results.analysis` instance, but importing that
    here would create a circular dependency.
    """

    lon: np.ndarray | None = None
    lat: np.ndarray | None = None
    topo: np.ndarray | None = None
    analysis: Any = None


@dataclass
class topo_cell(topo):
    """Cell-specific topography. Inherits ``lon`` / ``lat`` / ``topo`` /
    ``analysis`` from :class:`topo`."""

    def gen_mgrids(self, grad: bool = False) -> None:
        """Generate meshgrids from the 1-D lat/lon arrays.

        Sets ``self.lon_grid`` and ``self.lat_grid`` (and the gradient
        meshgrids when ``grad=True``). These attributes are runtime
        artifacts; they're not dataclass fields.
        """
        if not grad:
            lat, lon = self.lat, self.lon
            self.lon_grid, self.lat_grid = np.meshgrid(lon, lat)
        else:
            lat, lon = self.lat, self.lon
            grad_lat, grad_lon = self.grad_lat, self.grad_lon
            self.grad_lat_lon_grid, self.grad_lat_lat_grid = np.meshgrid(lon, grad_lat)
            self.grad_lon_lon_grid, self.grad_lon_lat_grid = np.meshgrid(grad_lon, lat)

    def __get_lat_lon_points(self, grad: bool = False) -> np.ndarray:
        """Stack the 2-D grids into a flat ``(N, 2)`` ``(lon, lat)``
        point array, rescaled."""
        if not grad:
            lat_grid, lon_grid = self.lat_grid, self.lon_grid
        else:
            lat_grid, lon_grid = self.grad_lat_grid, self.grad_lon_grid

        lat_grid_tmp = np.expand_dims(np.copy(lat_grid), -1)
        lon_grid_tmp = np.expand_dims(np.copy(lon_grid), -1)

        lat_grid_tmp = utils.rescale(lat_grid_tmp)
        lon_grid_tmp = utils.rescale(lon_grid_tmp)

        return np.stack((lon_grid_tmp, lat_grid_tmp), axis=2).reshape(-1, 2)

    def __get_mask(self, triangle) -> None:
        """Compute a boolean mask of points inside ``triangle``."""
        lat_lon_points = self.__get_lat_lon_points()
        init_poly = triangle.vec_get_mask

        self.mask = (
            np.array([init_poly(elem) for elem in lat_lon_points])
            .reshape(self.lat.size, self.lon.size)
            .astype("bool_")
        )

    def get_masked(self, triangle=None, mask=None) -> None:
        """Populate ``self.{lon_m, lat_m, topo_m}`` from a triangle or
        explicit mask. Subtracts the mean from ``topo_m`` in place.
        """
        if (triangle is not None) and (mask is None):
            self.__get_mask(triangle)
        elif mask is not None:
            self.mask = mask

        self.lon_m = self.lon_grid[self.mask]
        self.lat_m = self.lat_grid[self.mask]
        self.topo_m = self.topo[self.mask]

        self.topo_m -= self.topo_m.mean()

    def get_grad_topo(self, triangle) -> None:
        """Compute the topographic gradient.

        .. deprecated:: 0.90.0
        """
        lat, lon = self.lat, self.lon
        self.grad_lat = lat[:-1] + 0.5 * (lat[1:] - lat[:-1])
        self.grad_lon = lon[:-1] + 0.5 * (lon[1:] - lon[:-1])

        self.gen_mgrids(grad=True)

        dlat = np.diff(self.lat).reshape(1, -1)
        dlon = np.diff(self.lon).reshape(-1, 1)

        grad_lon_topo = (self.topo[1:, :] - self.topo[:-1, :]) / dlon
        grad_lat_topo = (self.topo[:, 1:] - self.topo[:, :-1]) / dlat

        lat_lon_points = self.__get_lat_lon_points(grad=True)
        init_poly = triangle.vec_get_mask

        self.grad_mask = (
            np.array([init_poly(elem) for elem in lat_lon_points])
            .reshape(self.topo.shape)
            .astype("bool_")
        )

        grad_lon_topo = grad_lon_topo[self.grad_mask]
        grad_lat_topo = grad_lat_topo[self.grad_mask]

        self.grad_lon_m = self.grad_lon_grid[self.grad_mask]
        self.grad_lat_m = self.grad_lat_grid[self.grad_mask]
        self.grad_topo_m = np.vstack([grad_lon_topo, grad_lat_topo])
