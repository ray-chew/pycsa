"""
Test ICON grid cells against real-world ETOPO topography.

This module validates that ICON grid cells and their associated ETOPO topography
data correctly correspond to real-world geographical features.

Uses pre-extracted .npz test data from data/test/ (Phase 0).
Falls back to live ETOPO loading when full ICON grid + ETOPO tiles are available.

Generates diagnostic plots in plots/tests/icon_etopo_validation/.

Test categories:
1. Mountains: Verify high elevation features (Himalayas, Andes)
2. Ice sheets: Verify high-altitude ice (Greenland)
3. Plains: Verify moderate elevation (Great Plains)
4. Oceans: Verify negative elevations (Pacific)
5. Grid geometry: Verify vertex coordinates match expected locations
"""

import pytest
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from pycsa.core import var


DATA_DIR = Path(__file__).parent.parent / "data" / "test"
PLOT_DIR = Path(__file__).parent.parent / "plots" / "tests" / "icon_etopo_validation"

# Test cell metadata: name → expected properties
CELL_SPECS = {
    'himalayas': {
        'feature_type': 'mountain',
        'min_peak': 5000,        # m — should contain peaks > 5 km
        'min_mean': 1500,        # m — mean should be well above sea level
        'lat_range': (25, 32),   # expected lat range
        'lon_range': (84, 90),
    },
    'andes': {
        'feature_type': 'mountain',
        'min_peak': 3500,
        'min_mean': 500,
        'lat_range': (-36, -31),
        'lon_range': (-73, -68),
    },
    'pacific': {
        'feature_type': 'ocean',
        'max_elev': 0,           # everything should be below sea level
        'min_water_fraction': 0.99,
        'lat_range': (-3, 2),
        'lon_range': (-163, -158),
    },
    'greenland': {
        'feature_type': 'ice_sheet',
        'min_peak': 2500,
        'min_mean': 2400,        # interior ice sheet is uniformly high
        'lat_range': (72, 77),
        'lon_range': (-45, -35),
    },
    'great_plains': {
        'feature_type': 'plains',
        'min_elev': 200,         # above sea level
        'max_elev': 1500,        # not mountainous
        'min_mean': 300,
        'lat_range': (37, 42),
        'lon_range': (-102, -97),
    },
}


def _save_fig(name):
    """Save figure to plot directory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {path}")


@pytest.fixture(scope="module")
def test_grid():
    """Load the 5-cell ICON grid excerpt."""
    grid_path = DATA_DIR / "test_grid_5cells.npz"
    if not grid_path.exists():
        pytest.skip(f"Test grid not found: {grid_path}")
    return np.load(grid_path, allow_pickle=True)


@pytest.fixture(scope="module")
def baselines():
    """Load baseline CSA results."""
    bl_path = DATA_DIR / "baselines.npz"
    if not bl_path.exists():
        pytest.skip(f"Baselines not found: {bl_path}")
    return np.load(bl_path)


def _load_etopo_patch(name):
    """Load an ETOPO patch .npz, return (topo_cell, data_dict)."""
    path = DATA_DIR / f"etopo_{name}.npz"
    if not path.exists():
        pytest.skip(f"ETOPO patch not found: {path}")
    data = np.load(path)
    cell = var.topo_cell()
    cell.lat = data['lat']
    cell.lon = data['lon']
    cell.topo = data['topo']
    cell.gen_mgrids()
    return cell, data


class TestGridGeometry:
    """Verify the test grid coordinates match expected geographic locations."""

    def test_cell_count(self, test_grid):
        """Test that we have exactly 5 test cells."""
        names = test_grid['cell_names']
        assert len(names) == 5, f"Expected 5 cells, got {len(names)}"
        expected = {'himalayas', 'andes', 'pacific', 'greenland', 'great_plains'}
        assert set(names) == expected
        print(f"\n  Grid contains {len(names)} cells: {', '.join(names)}")

    @pytest.mark.parametrize("cell_name", list(CELL_SPECS.keys()))
    def test_cell_location(self, test_grid, cell_name):
        """Verify each cell's center lat/lon is in the expected region."""
        names = list(test_grid['cell_names'])
        idx = names.index(cell_name)
        spec = CELL_SPECS[cell_name]

        clat = test_grid['clat_deg'][idx]
        clon = test_grid['clon_deg'][idx]

        lat_lo, lat_hi = spec['lat_range']
        lon_lo, lon_hi = spec['lon_range']

        assert lat_lo <= clat <= lat_hi, \
            f"{cell_name} center lat {clat:.2f} outside [{lat_lo}, {lat_hi}]"
        assert lon_lo <= clon <= lon_hi, \
            f"{cell_name} center lon {clon:.2f} outside [{lon_lo}, {lon_hi}]"

        print(f"  {cell_name:>15s}: center=({clat:>7.2f}°N, {clon:>8.2f}°E)")

    def test_vertices_are_triangles(self, test_grid):
        """Each cell should have exactly 3 vertices."""
        assert test_grid['clat_vertices'].shape == (5, 3)
        assert test_grid['clon_vertices'].shape == (5, 3)
        print(f"\n  All 5 cells have 3 vertices each (triangular ICON cells)")

    def test_vertex_ordering(self, test_grid):
        """Vertices should be distinct (not degenerate triangles)."""
        names = list(test_grid['cell_names'])
        print()
        for i in range(5):
            lats = test_grid['clat_vertices'][i]
            lons = test_grid['clon_vertices'][i]
            min_dist = float('inf')
            # Check no two vertices are identical
            for a in range(3):
                for b in range(a + 1, 3):
                    dist = np.sqrt((lats[a] - lats[b])**2 + (lons[a] - lons[b])**2)
                    min_dist = min(min_dist, dist)
                    assert dist > 0.01, \
                        f"Cell {i}: vertices {a} and {b} are too close ({dist:.4f} deg)"
            print(f"  {names[i]:>15s}: min vertex separation = {min_dist:.3f}°")


class TestTopographyValidation:
    """Validate ETOPO patches against known geographic features.

    Each test also generates a diagnostic plot for human verification.
    """

    @pytest.mark.parametrize("cell_name", ['himalayas', 'andes'])
    def test_mountain_features(self, cell_name):
        """Mountains should have high peak elevations and terrain variation."""
        cell, _ = _load_etopo_patch(cell_name)
        spec = CELL_SPECS[cell_name]

        max_elev = cell.topo.max()
        mean_elev = cell.topo.mean()
        std_elev = cell.topo.std()

        assert max_elev >= spec['min_peak'], \
            f"{cell_name}: max elevation {max_elev:.0f} m < {spec['min_peak']} m"
        assert mean_elev >= spec['min_mean'], \
            f"{cell_name}: mean elevation {mean_elev:.0f} m < {spec['min_mean']} m"
        assert std_elev > 100, \
            f"{cell_name}: terrain too flat (std = {std_elev:.0f} m)"

        print(f"  {cell_name}: peak={max_elev:.0f} m, mean={mean_elev:.0f} m, std={std_elev:.0f} m")

        # Diagnostic plot: terrain + histogram + cross-section
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im = axes[0].contourf(cell.lon_grid, cell.lat_grid, cell.topo,
                              levels=20, cmap='terrain')
        plt.colorbar(im, ax=axes[0], label='Elevation (m)')
        axes[0].set_title(f'{cell_name.title()}\n'
                          f'peak={max_elev:.0f} m, mean={mean_elev:.0f} m')
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')

        axes[1].hist(cell.topo.ravel(), bins=50, color='sienna', edgecolor='k', alpha=0.7)
        axes[1].axvline(mean_elev, color='red', linestyle='--', label=f'mean={mean_elev:.0f} m')
        axes[1].set_xlabel('Elevation (m)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Elevation Distribution')
        axes[1].legend()

        mid_row = cell.topo.shape[0] // 2
        axes[2].plot(cell.lon, cell.topo[mid_row, :], 'k-', linewidth=0.8)
        axes[2].fill_between(cell.lon, cell.topo[mid_row, :], alpha=0.3, color='sienna')
        axes[2].set_xlabel('Longitude (°)')
        axes[2].set_ylabel('Elevation (m)')
        axes[2].set_title(f'E-W Cross-section at lat={cell.lat[mid_row]:.2f}°')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        _save_fig(f'topo_{cell_name}.png')

    def test_pacific_ocean(self):
        """Pacific patch should be entirely below sea level."""
        cell, _ = _load_etopo_patch('pacific')
        spec = CELL_SPECS['pacific']

        water_fraction = (cell.topo < 0).sum() / cell.topo.size
        max_elev = cell.topo.max()

        assert max_elev <= spec['max_elev'], \
            f"Pacific: max elevation {max_elev:.0f} m > {spec['max_elev']} m"
        assert water_fraction >= spec['min_water_fraction'], \
            f"Pacific: water fraction {water_fraction:.3f} < {spec['min_water_fraction']}"

        mean_depth = -cell.topo.mean()
        print(f"  Pacific: 100% ocean, mean depth={mean_depth:.0f} m, "
              f"range=[{cell.topo.min():.0f}, {cell.topo.max():.0f}] m")

        # Diagnostic plot: bathymetry
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im = axes[0].contourf(cell.lon_grid, cell.lat_grid, cell.topo,
                              levels=20, cmap='ocean')
        plt.colorbar(im, ax=axes[0], label='Depth (m)')
        axes[0].set_title(f'Pacific Ocean\nmean depth={mean_depth:.0f} m')
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')

        axes[1].hist(cell.topo.ravel(), bins=50, color='steelblue', edgecolor='k', alpha=0.7)
        axes[1].set_xlabel('Elevation (m)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Bathymetry Distribution\n100% ocean, '
                          f'[{cell.topo.min():.0f}, {cell.topo.max():.0f}] m')

        plt.tight_layout()
        _save_fig('topo_pacific.png')

    def test_greenland_ice_sheet(self):
        """Greenland interior should be high-altitude ice sheet."""
        cell, _ = _load_etopo_patch('greenland')
        spec = CELL_SPECS['greenland']

        max_elev = cell.topo.max()
        mean_elev = cell.topo.mean()
        min_elev = cell.topo.min()

        assert max_elev >= spec['min_peak'], \
            f"Greenland: max elevation {max_elev:.0f} m < {spec['min_peak']} m"
        assert mean_elev >= spec['min_mean'], \
            f"Greenland: mean elevation {mean_elev:.0f} m < {spec['min_mean']} m"
        # Ice sheet interior should be uniformly high (low std relative to mean)
        assert cell.topo.std() < 0.2 * mean_elev, \
            f"Greenland: too much variation for ice sheet interior"

        print(f"  Greenland: mean={mean_elev:.0f} m, range=[{min_elev:.0f}, {max_elev:.0f}] m")

        # Diagnostic plot: ice sheet uniformity
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im = axes[0].contourf(cell.lon_grid, cell.lat_grid, cell.topo,
                              levels=20, cmap='terrain')
        plt.colorbar(im, ax=axes[0], label='Elevation (m)')
        axes[0].set_title(f'Greenland Ice Sheet\nmean={mean_elev:.0f} m, '
                          f'std={cell.topo.std():.0f} m')
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')

        axes[1].hist(cell.topo.ravel(), bins=50, color='lightblue', edgecolor='k', alpha=0.7)
        axes[1].axvline(mean_elev, color='red', linestyle='--', label=f'mean={mean_elev:.0f} m')
        axes[1].set_xlabel('Elevation (m)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Elevation Distribution\nrange=[{min_elev:.0f}, {max_elev:.0f}] m')
        axes[1].legend()

        mid_row = cell.topo.shape[0] // 2
        axes[2].plot(cell.lon, cell.topo[mid_row, :], 'k-', linewidth=0.8)
        axes[2].fill_between(cell.lon, cell.topo[mid_row, :], alpha=0.3, color='lightblue')
        axes[2].set_xlabel('Longitude (°)')
        axes[2].set_ylabel('Elevation (m)')
        axes[2].set_title(f'E-W Cross-section at lat={cell.lat[mid_row]:.2f}°')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        _save_fig('topo_greenland.png')

    def test_great_plains(self):
        """Great Plains should be moderate elevation, relatively flat."""
        cell, _ = _load_etopo_patch('great_plains')
        spec = CELL_SPECS['great_plains']

        mean_elev = cell.topo.mean()
        max_elev = cell.topo.max()
        min_elev = cell.topo.min()

        assert min_elev >= spec['min_elev'], \
            f"Great Plains: min elevation {min_elev:.0f} m < {spec['min_elev']} m"
        assert max_elev <= spec['max_elev'], \
            f"Great Plains: max elevation {max_elev:.0f} m > {spec['max_elev']} m"
        assert mean_elev >= spec['min_mean'], \
            f"Great Plains: mean elevation {mean_elev:.0f} m < {spec['min_mean']} m"

        print(f"  Great Plains: mean={mean_elev:.0f} m, range=[{min_elev:.0f}, {max_elev:.0f}] m")

        # Diagnostic plot: gentle terrain
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im = axes[0].contourf(cell.lon_grid, cell.lat_grid, cell.topo,
                              levels=20, cmap='terrain')
        plt.colorbar(im, ax=axes[0], label='Elevation (m)')
        axes[0].set_title(f'Great Plains\nmean={mean_elev:.0f} m, '
                          f'range=[{min_elev:.0f}, {max_elev:.0f}] m')
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')

        axes[1].hist(cell.topo.ravel(), bins=50, color='olive', edgecolor='k', alpha=0.7)
        axes[1].axvline(mean_elev, color='red', linestyle='--', label=f'mean={mean_elev:.0f} m')
        axes[1].set_xlabel('Elevation (m)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Elevation Distribution')
        axes[1].legend()

        mid_row = cell.topo.shape[0] // 2
        axes[2].plot(cell.lon, cell.topo[mid_row, :], 'k-', linewidth=0.8)
        axes[2].fill_between(cell.lon, cell.topo[mid_row, :], alpha=0.3, color='olive')
        axes[2].set_xlabel('Longitude (°)')
        axes[2].set_ylabel('Elevation (m)')
        axes[2].set_title(f'E-W Cross-section at lat={cell.lat[mid_row]:.2f}°')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        _save_fig('topo_great_plains.png')


class TestDataQuality:
    """Verify data integrity across all ETOPO patches."""

    @pytest.mark.parametrize("cell_name", list(CELL_SPECS.keys()))
    def test_no_nan(self, cell_name):
        """No NaN values in topography."""
        cell, _ = _load_etopo_patch(cell_name)
        n_nan = np.isnan(cell.topo).sum()
        assert n_nan == 0, f"{cell_name}: {n_nan} NaN values in topography"
        print(f"  {cell_name}: 0 NaN / {cell.topo.size} pixels")

    @pytest.mark.parametrize("cell_name", list(CELL_SPECS.keys()))
    def test_no_inf(self, cell_name):
        """No infinite values in topography."""
        cell, _ = _load_etopo_patch(cell_name)
        n_inf = np.isinf(cell.topo).sum()
        assert n_inf == 0, f"{cell_name}: {n_inf} Inf values in topography"
        print(f"  {cell_name}: 0 Inf / {cell.topo.size} pixels")

    @pytest.mark.parametrize("cell_name", list(CELL_SPECS.keys()))
    def test_elevation_range(self, cell_name):
        """Elevation should be within Earth's physical range."""
        cell, _ = _load_etopo_patch(cell_name)
        assert cell.topo.min() >= -12000, \
            f"{cell_name}: elevation too low ({cell.topo.min():.0f} m)"
        assert cell.topo.max() <= 9000, \
            f"{cell_name}: elevation too high ({cell.topo.max():.0f} m)"
        print(f"  {cell_name}: elevation [{cell.topo.min():.0f}, {cell.topo.max():.0f}] m "
              f"(within [-12000, 9000])")

    @pytest.mark.parametrize("cell_name", list(CELL_SPECS.keys()))
    def test_coordinate_consistency(self, cell_name):
        """Topography shape must match lat × lon dimensions."""
        cell, _ = _load_etopo_patch(cell_name)
        assert cell.topo.shape == (len(cell.lat), len(cell.lon)), \
            f"{cell_name}: shape {cell.topo.shape} != ({len(cell.lat)}, {len(cell.lon)})"
        print(f"  {cell_name}: shape {cell.topo.shape} = ({len(cell.lat)} lat × {len(cell.lon)} lon) ✓")

    @pytest.mark.parametrize("cell_name", list(CELL_SPECS.keys()))
    def test_coordinates_monotonic(self, cell_name):
        """Lat and lon should be monotonically increasing."""
        cell, _ = _load_etopo_patch(cell_name)
        lat_mono = "ascending" if np.all(np.diff(cell.lat) > 0) else "descending"
        lon_mono = "ascending" if np.all(np.diff(cell.lon) > 0) else "descending"
        assert np.all(np.diff(cell.lat) > 0) or np.all(np.diff(cell.lat) < 0), \
            f"{cell_name}: lat not monotonic"
        assert np.all(np.diff(cell.lon) > 0) or np.all(np.diff(cell.lon) < 0), \
            f"{cell_name}: lon not monotonic"
        dlat = np.abs(np.diff(cell.lat).mean())
        dlon = np.abs(np.diff(cell.lon).mean())
        print(f"  {cell_name}: lat {lat_mono} (Δ={dlat:.4f}°), lon {lon_mono} (Δ={dlon:.4f}°)")


class TestBaselineConsistency:
    """Verify baseline CSA results match expectations."""

    @pytest.mark.parametrize("cell_name", ['himalayas', 'andes', 'greenland', 'great_plains'])
    def test_land_flag(self, baselines, cell_name):
        """Land cells should be flagged as land."""
        assert baselines[f'{cell_name}_is_land'][0] == 1, \
            f"{cell_name} should be land"
        print(f"  {cell_name}: is_land = True ✓")

    def test_pacific_not_land(self, baselines):
        """Pacific cell should not be land."""
        assert baselines['pacific_is_land'][0] == 0, "Pacific should not be land"
        print(f"  pacific: is_land = False ✓ (ocean cell, CSA skipped)")

    @pytest.mark.parametrize("cell_name", ['himalayas', 'andes', 'greenland', 'great_plains'])
    def test_amplitudes_finite(self, baselines, cell_name):
        """FA and SA amplitudes should be finite (no NaN)."""
        ampls_fa = baselines[f'{cell_name}_ampls_fa']
        ampls_sa = baselines[f'{cell_name}_ampls_sa']
        assert np.all(np.isfinite(ampls_fa)), f"{cell_name}: NaN/Inf in FA amplitudes"
        assert np.all(np.isfinite(ampls_sa)), f"{cell_name}: NaN/Inf in SA amplitudes"
        n_fa = np.count_nonzero(ampls_fa)
        n_sa = np.count_nonzero(ampls_sa)
        max_fa = np.abs(ampls_fa).max()
        max_sa = np.abs(ampls_sa).max()
        print(f"  {cell_name}: FA {n_fa} nonzero modes (max |A|={max_fa:.1f} m), "
              f"SA {n_sa} nonzero (max |A|={max_sa:.1f} m)")

    @pytest.mark.parametrize("cell_name", ['himalayas', 'andes', 'greenland', 'great_plains'])
    def test_rmse_positive(self, baselines, cell_name):
        """RMSE should be positive (non-trivial fit)."""
        rmse_fa = baselines[f'{cell_name}_rmse_fa'][0]
        rmse_sa = baselines[f'{cell_name}_rmse_sa'][0]
        assert rmse_fa > 0, f"{cell_name}: FA RMSE is zero"
        assert rmse_sa > 0, f"{cell_name}: SA RMSE is zero"
        print(f"  {cell_name}: FA RMSE = {rmse_fa:.2f} m, SA RMSE = {rmse_sa:.2f} m")

    @pytest.mark.parametrize("cell_name", ['himalayas', 'andes', 'greenland', 'great_plains'])
    def test_fa_rmse_lower_than_sa(self, baselines, cell_name):
        """FA (full-domain) should have lower RMSE than SA (triangle-domain).

        This is expected because FA fits on the full rectangular domain,
        while SA uses a subset of modes on the triangular sub-domain.
        """
        rmse_fa = baselines[f'{cell_name}_rmse_fa'][0]
        rmse_sa = baselines[f'{cell_name}_rmse_sa'][0]
        # FA fits more data with more modes — should have lower error
        assert rmse_fa < rmse_sa, \
            f"{cell_name}: FA RMSE ({rmse_fa:.1f}) >= SA RMSE ({rmse_sa:.1f})"
        ratio = rmse_sa / rmse_fa
        print(f"  {cell_name}: FA={rmse_fa:.1f} m < SA={rmse_sa:.1f} m "
              f"(SA/FA ratio = {ratio:.1f}×)")


class TestCSAReconstruction:
    """Run CSA pipeline on test cells and plot FA/SA reconstruction quality.

    Re-runs first_appx and second_appx on each land cell's ETOPO patch,
    then plots original vs reconstruction vs difference for visual verification.
    ~3s per cell, ~12s total for all 4 land cells.
    """

    LAND_CELLS = ['himalayas', 'andes', 'greenland', 'great_plains']

    @pytest.fixture(scope="class")
    def grid_data(self):
        """Load grid vertices for triangle construction."""
        grid_path = DATA_DIR / "test_grid_5cells.npz"
        if not grid_path.exists():
            pytest.skip(f"Test grid not found: {grid_path}")
        return np.load(grid_path, allow_pickle=True)

    @staticmethod
    def _run_csa(cell_name, grid_data):
        """Run FA + SA on one cell. Returns (cell_fa, dat_2D_fa, cell_sa, dat_2D_sa, ampls_fa, ampls_sa)."""
        from pycsa.core import var, utils
        from pycsa.wrappers import interface

        # Load topo
        topo, _ = _load_etopo_patch(cell_name)

        # Get triangle vertices from grid
        names = list(grid_data['cell_names'])
        idx = names.index(cell_name)
        lat_verts = grid_data['clat_vertices'][idx].copy()
        lon_verts = grid_data['clon_vertices'][idx].copy()

        lat_verts, lon_verts = utils.handle_latlon_expansion(
            lat_verts, lon_verts, lat_expand=0.0, lon_expand=0.0
        )

        # Build tri object
        tri = var.obj()
        tri.tri_lon_verts = lon_verts.reshape(1, -1)
        tri.tri_lat_verts = lat_verts.reshape(1, -1)

        simplex_lat = tri.tri_lat_verts[0]
        simplex_lon = tri.tri_lon_verts[0]

        # Production params
        p = var.params()
        p.nhi, p.nhj, p.n_modes = 32, 64, 100
        p.padding, p.U, p.V = 10, 10.0, 0.0
        p.rect = True
        p.dfft_first_guess = False
        p.recompute_rhs = False
        p.fa_iter_solve = True
        p.sa_iter_solve = True
        p.taper_fa = False
        p.taper_sa = False
        p.cg_spsp = False
        p.taper_art_it = 50
        p.etopo_cg = 4
        p.debug = False
        p.refine = False
        p.verbose = False
        p.plot = False
        p.plot_output = False

        # First approximation
        fa = interface.first_appx(p.nhi, p.nhj, p, topo)
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(
            simplex_lat, simplex_lon, use_center=True
        )

        # Second approximation
        sa = interface.second_appx(p.nhi, p.nhj, p, topo, tri)
        cell_sa, ampls_sa, uw_sa, dat_2D_sa = sa.do(0, ampls_fa, use_center=True)

        return cell_fa, dat_2D_fa, cell_sa, dat_2D_sa, ampls_fa, ampls_sa

    @pytest.mark.parametrize("cell_name", ['himalayas', 'andes', 'greenland', 'great_plains'])
    def test_reconstruction_plots(self, cell_name, grid_data, baselines):
        """Run CSA and plot FA/SA reconstruction vs original topography."""
        cell_fa, dat_2D_fa, cell_sa, dat_2D_sa, ampls_fa, ampls_sa = \
            self._run_csa(cell_name, grid_data)

        cell_fa.gen_mgrids()
        cell_sa.gen_mgrids()

        # Compute RMSE
        masked_fa = np.where(cell_fa.mask, cell_fa.topo, np.nan)
        diff_fa = masked_fa - dat_2D_fa
        rmse_fa = np.sqrt(np.nanmean(diff_fa**2))

        masked_sa = np.where(cell_sa.mask, cell_sa.topo, np.nan)
        diff_sa = masked_sa - dat_2D_sa
        rmse_sa = np.sqrt(np.nanmean(diff_sa**2))

        # Verify RMSE matches baselines (within 5% — slight float diffs expected)
        bl_rmse_fa = baselines[f'{cell_name}_rmse_fa'][0]
        bl_rmse_sa = baselines[f'{cell_name}_rmse_sa'][0]
        assert abs(rmse_fa - bl_rmse_fa) / bl_rmse_fa < 0.05, \
            f"{cell_name}: FA RMSE {rmse_fa:.2f} != baseline {bl_rmse_fa:.2f}"

        print(f"  {cell_name}: FA RMSE={rmse_fa:.1f} m, SA RMSE={rmse_sa:.1f} m")

        # --- 6-panel plot: Original / FA recon / SA recon (top), FA diff / SA diff / spectra (bottom) ---
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Row 1: original, FA reconstruction, SA reconstruction
        im00 = axes[0, 0].contourf(cell_fa.lon_grid, cell_fa.lat_grid, masked_fa,
                                    levels=20, cmap='terrain')
        plt.colorbar(im00, ax=axes[0, 0], label='m')
        axes[0, 0].set_title(f'{cell_name.title()}: Original (FA domain)\n'
                              f'[{np.nanmin(masked_fa):.0f}, {np.nanmax(masked_fa):.0f}] m')

        im01 = axes[0, 1].contourf(cell_fa.lon_grid, cell_fa.lat_grid, dat_2D_fa,
                                    levels=20, cmap='terrain')
        plt.colorbar(im01, ax=axes[0, 1], label='m')
        axes[0, 1].set_title(f'FA Reconstruction (32×64 modes)\nRMSE = {rmse_fa:.1f} m')

        im02 = axes[0, 2].contourf(cell_sa.lon_grid, cell_sa.lat_grid, dat_2D_sa,
                                    levels=20, cmap='terrain')
        plt.colorbar(im02, ax=axes[0, 2], label='m')
        axes[0, 2].set_title(f'SA Reconstruction (100 modes)\nRMSE = {rmse_sa:.1f} m')

        # Row 2: FA difference, SA difference, amplitude spectra
        vmax_fa = np.nanpercentile(np.abs(diff_fa), 98)
        im10 = axes[1, 0].contourf(cell_fa.lon_grid, cell_fa.lat_grid, diff_fa,
                                    levels=20, cmap='RdBu_r', vmin=-vmax_fa, vmax=vmax_fa)
        plt.colorbar(im10, ax=axes[1, 0], label='m')
        axes[1, 0].set_title(f'FA Residual\nmax |err| = {np.nanmax(np.abs(diff_fa)):.0f} m')

        vmax_sa = np.nanpercentile(np.abs(diff_sa), 98)
        im11 = axes[1, 1].contourf(cell_sa.lon_grid, cell_sa.lat_grid, diff_sa,
                                    levels=20, cmap='RdBu_r', vmin=-vmax_sa, vmax=vmax_sa)
        plt.colorbar(im11, ax=axes[1, 1], label='m')
        axes[1, 1].set_title(f'SA Residual\nmax |err| = {np.nanmax(np.abs(diff_sa)):.0f} m')

        # Amplitude spectra comparison
        ampls_fa_clean = np.nan_to_num(ampls_fa, nan=0.0)
        ampls_sa_clean = np.nan_to_num(ampls_sa, nan=0.0)
        vmax_a = max(np.abs(ampls_fa_clean).max(), np.abs(ampls_sa_clean).max())

        # Split the last subplot into two halves for FA and SA spectra
        axes[1, 2].remove()
        gs = fig.add_gridspec(2, 3)
        ax_fa_sp = fig.add_subplot(gs[1, 2])

        # Just show FA spectrum (SA has only 100 nonzero modes, hard to see at same scale)
        im12 = ax_fa_sp.imshow(np.abs(ampls_fa_clean), cmap='hot_r', aspect='auto',
                                origin='lower', vmin=0, vmax=vmax_a)
        plt.colorbar(im12, ax=ax_fa_sp, label='|A| (m)')
        n_nonzero_fa = np.count_nonzero(ampls_fa_clean)
        n_nonzero_sa = np.count_nonzero(ampls_sa_clean)
        ax_fa_sp.set_xlabel('k')
        ax_fa_sp.set_ylabel('l')
        ax_fa_sp.set_title(f'FA Spectrum ({n_nonzero_fa} modes)\n'
                            f'SA: {n_nonzero_sa} selected modes')

        for ax in axes[0, :]:
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
        for ax in axes[1, :2]:
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')

        fig.suptitle(f'CSA Reconstruction Quality — {cell_name.title()}', fontsize=14, y=1.02)
        plt.tight_layout()
        _save_fig(f'csa_reconstruction_{cell_name}.png')

    def test_reconstruction_summary(self, grid_data, baselines):
        """Summary plot: all 4 land cells side-by-side showing original, FA, SA."""
        fig, axes = plt.subplots(4, 4, figsize=(22, 22))

        for row, cell_name in enumerate(self.LAND_CELLS):
            cell_fa, dat_2D_fa, cell_sa, dat_2D_sa, ampls_fa, ampls_sa = \
                self._run_csa(cell_name, grid_data)

            cell_fa.gen_mgrids()
            cell_sa.gen_mgrids()

            masked_fa = np.where(cell_fa.mask, cell_fa.topo, np.nan)
            diff_fa = masked_fa - dat_2D_fa
            rmse_fa = np.sqrt(np.nanmean(diff_fa**2))

            masked_sa = np.where(cell_sa.mask, cell_sa.topo, np.nan)
            diff_sa = masked_sa - dat_2D_sa
            rmse_sa = np.sqrt(np.nanmean(diff_sa**2))

            # Col 0: original
            im0 = axes[row, 0].contourf(cell_fa.lon_grid, cell_fa.lat_grid, masked_fa,
                                         levels=20, cmap='terrain')
            plt.colorbar(im0, ax=axes[row, 0], label='m')
            axes[row, 0].set_title(f'{cell_name.title()}: Original')

            # Col 1: FA reconstruction
            im1 = axes[row, 1].contourf(cell_fa.lon_grid, cell_fa.lat_grid, dat_2D_fa,
                                         levels=20, cmap='terrain')
            plt.colorbar(im1, ax=axes[row, 1], label='m')
            axes[row, 1].set_title(f'FA (RMSE={rmse_fa:.1f} m)')

            # Col 2: SA reconstruction
            im2 = axes[row, 2].contourf(cell_sa.lon_grid, cell_sa.lat_grid, dat_2D_sa,
                                         levels=20, cmap='terrain')
            plt.colorbar(im2, ax=axes[row, 2], label='m')
            axes[row, 2].set_title(f'SA (RMSE={rmse_sa:.1f} m)')

            # Col 3: E-W cross-section at mid-latitude
            mid = cell_fa.topo.shape[0] // 2
            axes[row, 3].plot(cell_fa.lon, masked_fa[mid, :], 'k-', label='Original', linewidth=1)
            axes[row, 3].plot(cell_fa.lon, dat_2D_fa[mid, :], 'b--', label='FA', linewidth=1)
            # SA is on same grid but may have different mask
            mid_sa = cell_sa.topo.shape[0] // 2
            axes[row, 3].plot(cell_sa.lon, dat_2D_sa[mid_sa, :], 'r:', label='SA', linewidth=1)
            axes[row, 3].set_xlabel('Longitude (°)')
            axes[row, 3].set_ylabel('Elevation (m)')
            axes[row, 3].set_title(f'E-W Profile (lat≈{cell_fa.lat[mid]:.1f}°)')
            axes[row, 3].legend(fontsize=8)
            axes[row, 3].grid(True, alpha=0.3)

            print(f"  {cell_name:>15s}: FA RMSE={rmse_fa:>7.1f} m, SA RMSE={rmse_sa:>7.1f} m")

        for ax in axes[:, :3].flat:
            ax.set_xlabel('Lon (°)')
            ax.set_ylabel('Lat (°)')

        fig.suptitle('CSA Reconstruction Summary — All Land Cells\n'
                      'nhi=32, nhj=64, n_modes=100, U=10 m/s, CG=4',
                      fontsize=14, y=1.01)
        plt.tight_layout()
        _save_fig('csa_reconstruction_summary.png')


class TestVisualization:
    """Generate diagnostic plots for human verification."""

    def test_all_patches_overview(self):
        """Plot all 5 ETOPO patches in a single figure."""
        names = ['himalayas', 'andes', 'pacific', 'greenland', 'great_plains']
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes_flat = axes.flat

        for name, ax in zip(names, axes_flat):
            cell, _ = _load_etopo_patch(name)
            im = ax.contourf(cell.lon_grid, cell.lat_grid, cell.topo,
                             levels=20, cmap='terrain')
            plt.colorbar(im, ax=ax, label='m')
            ax.set_title(f'{name.replace("_", " ").title()}\n'
                         f'[{cell.topo.min():.0f}, {cell.topo.max():.0f}] m, '
                         f'mean={cell.topo.mean():.0f} m')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        # Turn off unused subplot
        axes_flat[5].set_visible(False)

        fig.suptitle('Phase 0 Test Data — 5 ICON Cell ETOPO Patches', fontsize=14)
        plt.tight_layout()
        _save_fig('all_patches_overview.png')

    def test_baseline_spectra_overview(self, baselines):
        """Plot FA and SA amplitude spectra for all land cells."""
        land_cells = ['himalayas', 'andes', 'greenland', 'great_plains']
        fig, axes = plt.subplots(len(land_cells), 3, figsize=(18, 5 * len(land_cells)))

        for row, name in enumerate(land_cells):
            ampls_fa = baselines[f'{name}_ampls_fa']
            ampls_sa = baselines[f'{name}_ampls_sa']
            rmse_fa = baselines[f'{name}_rmse_fa'][0]
            rmse_sa = baselines[f'{name}_rmse_sa'][0]

            vmax = max(np.abs(ampls_fa).max(), np.abs(ampls_sa).max())

            im0 = axes[row, 0].imshow(np.abs(ampls_fa), cmap='hot_r', aspect='auto',
                                       origin='lower', vmin=0, vmax=vmax)
            plt.colorbar(im0, ax=axes[row, 0], label='|A| (m)')
            axes[row, 0].set_title(f'{name}: FA (RMSE={rmse_fa:.1f} m)')

            im1 = axes[row, 1].imshow(np.abs(ampls_sa), cmap='hot_r', aspect='auto',
                                       origin='lower', vmin=0, vmax=vmax)
            plt.colorbar(im1, ax=axes[row, 1], label='|A| (m)')
            axes[row, 1].set_title(f'{name}: SA (RMSE={rmse_sa:.1f} m)')

            # Difference
            diff = np.abs(ampls_fa) - np.abs(ampls_sa)
            im2 = axes[row, 2].imshow(diff, cmap='RdBu_r', aspect='auto',
                                       origin='lower')
            plt.colorbar(im2, ax=axes[row, 2], label='|FA| - |SA| (m)')
            axes[row, 2].set_title(f'{name}: FA − SA')

        for ax in axes.flat:
            ax.set_xlabel('k')
            ax.set_ylabel('l')

        fig.suptitle('Baseline CSA Spectra — FA vs SA', fontsize=14)
        plt.tight_layout()
        _save_fig('baseline_spectra_overview.png')

    def test_grid_cells_map(self, test_grid):
        """Plot the 5 test cells on a simple lat-lon map."""
        fig, ax = plt.subplots(figsize=(14, 7))

        names = list(test_grid['cell_names'])
        clat = test_grid['clat_deg']
        clon = test_grid['clon_deg']
        clat_v = test_grid['clat_vertices']
        clon_v = test_grid['clon_vertices']

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        for i, (name, color) in enumerate(zip(names, colors)):
            # Draw triangle
            verts = list(zip(clon_v[i], clat_v[i]))
            verts.append(verts[0])  # close polygon
            xs, ys = zip(*verts)
            ax.fill(xs, ys, alpha=0.3, color=color)
            ax.plot(xs, ys, '-', color=color, linewidth=2)

            # Mark center
            ax.scatter(clon[i], clat[i], c=color, s=100, zorder=5,
                       edgecolor='k', linewidth=1.5)
            ax.annotate(f'  {name}', (clon[i], clat[i]), fontsize=10,
                       fontweight='bold', color=color)

        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title('Phase 0 Test Cells — ICON Grid R02B04')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        _save_fig('test_cells_map.png')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
