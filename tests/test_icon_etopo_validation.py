"""
Test ICON grid cells against real-world ETOPO topography.

This module validates that ICON grid cells and their associated ETOPO topography
data correctly correspond to real-world geographical features. This ensures that
coordinate transformations, data loading, and spatial mapping are functioning correctly.

Test categories:
1. Mountains: Verify high elevation features (Himalayas, Andes, Alps, etc.)
2. Lakes: Verify inland water bodies (Great Lakes, Lake Baikal, etc.)
3. Oceans/Gulfs: Verify marine features (Pacific, Gulf of Mexico, etc.)
4. Coasts: Verify land-ocean transitions
5. Edge cases: Dateline, poles, tile boundaries
"""

import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

from pycsa.core import io, var, utils
from pycsa import local_paths


class GeographicFeature:
    """Represents a known geographic feature for validation."""

    def __init__(
        self,
        name: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        feature_type: str,
        validation_func,
        description: str = "",
    ):
        """
        Initialize a geographic feature.

        Args:
            name: Feature name (e.g., "Himalayas", "Lake Superior")
            lat_range: (min_lat, max_lat) in degrees
            lon_range: (min_lon, max_lon) in degrees
            feature_type: One of "mountain", "lake", "ocean", "gulf", "coast"
            validation_func: Function that validates topography matches feature
            description: Human-readable description
        """
        self.name = name
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.feature_type = feature_type
        self.validation_func = validation_func
        self.description = description

    def get_center(self) -> Tuple[float, float]:
        """Return (center_lat, center_lon) of feature."""
        lat_center = np.mean(self.lat_range)
        lon_center = np.mean(self.lon_range)
        return lat_center, lon_center

    def validate(self, topo_cell: var.topo_cell) -> Dict:
        """
        Validate that topography matches this geographic feature.

        Returns:
            Dict with keys: 'passed', 'message', 'stats'
        """
        return self.validation_func(topo_cell, self)


# Validation functions for different feature types
def validate_mountain(topo_cell: var.topo_cell, feature: GeographicFeature) -> Dict:
    """Validate mountain features have high elevations."""
    max_elev = topo_cell.topo.max()
    min_expected = 3000  # meters

    # Different mountain ranges have different heights
    if "Himalayas" in feature.name or "Karakoram" in feature.name:
        min_expected = 5000  # Should have peaks > 5km
    elif "Andes" in feature.name or "Alps" in feature.name:
        min_expected = 3500
    elif "Rockies" in feature.name or "Appalachian" in feature.name:
        min_expected = 2000

    passed = max_elev >= min_expected
    message = (
        f"{feature.name}: max elevation {max_elev:.0f}m (expected >{min_expected}m)"
    )

    stats = {
        "max_elevation": max_elev,
        "mean_elevation": topo_cell.topo.mean(),
        "min_elevation": topo_cell.topo.min(),
        "std_elevation": topo_cell.topo.std(),
        "high_terrain_fraction": (topo_cell.topo > 1000).sum() / topo_cell.topo.size,
    }

    return {"passed": passed, "message": message, "stats": stats}


def validate_lake(topo_cell: var.topo_cell, feature: GeographicFeature) -> Dict:
    """Validate lake features have appropriate water elevation."""
    # Lakes regions include surrounding terrain, so we check:
    # 1. Minimum elevation should be near expected lake level
    # 2. Should have some low-elevation areas (the actual lake)

    min_elev = topo_cell.topo.min()
    mean_elev = topo_cell.topo.mean()

    # Count how much of the area is near the expected lake elevation
    # Special cases for different lakes
    if "Titicaca" in feature.name:
        expected_lake_elev = 3812  # meters
        tolerance = 300  # Allow surrounding mountains
    elif "Baikal" in feature.name:
        expected_lake_elev = 456  # meters
        tolerance = 500  # Mountainous region
    elif "Great Lakes" in feature.name or "Superior" in feature.name:
        expected_lake_elev = 183  # meters
        tolerance = 200  # Relatively flat region
    else:
        expected_lake_elev = 100  # Generic lake
        tolerance = 300

    # Check that minimum elevation is close to lake level (below it due to lake depth)
    lake_depth_margin = 500  # Lakes can be deep
    min_expected = expected_lake_elev - lake_depth_margin
    max_expected = expected_lake_elev + tolerance

    # Count fraction of area near lake elevation (within tolerance)
    near_lake_level = np.abs(topo_cell.topo - expected_lake_elev) < tolerance
    lake_fraction = near_lake_level.sum() / topo_cell.topo.size

    # Validate: minimum should be below/near lake level, and some area should be at lake level
    has_low_areas = min_elev < expected_lake_elev + 100
    has_lake_level_areas = lake_fraction > 0.05  # At least 5% at lake level

    passed = has_low_areas and has_lake_level_areas
    message = (
        f"{feature.name}: min elev {min_elev:.0f}m, mean {mean_elev:.0f}m, "
        f"{lake_fraction:.1%} near lake level ~{expected_lake_elev}m"
    )

    stats = {
        "mean_elevation": mean_elev,
        "min_elevation": min_elev,
        "max_elevation": topo_cell.topo.max(),
        "std_elevation": topo_cell.topo.std(),
        "expected_lake_elevation": expected_lake_elev,
        "fraction_near_lake_level": lake_fraction,
        "has_low_areas": has_low_areas,
        "has_lake_level_areas": has_lake_level_areas,
    }

    return {"passed": passed, "message": message, "stats": stats}


def validate_ocean(topo_cell: var.topo_cell, feature: GeographicFeature) -> Dict:
    """Validate ocean features have negative (below sea level) elevations."""
    # Oceans should be mostly below sea level
    water_fraction = (topo_cell.topo < 0).sum() / topo_cell.topo.size
    mean_depth = (
        -topo_cell.topo[topo_cell.topo < 0].mean() if (topo_cell.topo < 0).any() else 0
    )

    min_water_fraction = 0.80  # At least 80% should be water

    # Deep ocean should have significant depth
    if "Pacific" in feature.name or "Atlantic" in feature.name:
        min_expected_depth = 3000  # Deep ocean
    else:
        min_expected_depth = 100  # Shallow seas/gulfs

    passed = water_fraction >= min_water_fraction and mean_depth >= min_expected_depth
    message = (
        f"{feature.name}: water fraction {water_fraction:.1%}, "
        f"mean depth {mean_depth:.0f}m (expected >{min_expected_depth}m)"
    )

    stats = {
        "water_fraction": water_fraction,
        "mean_depth": mean_depth,
        "max_depth": -topo_cell.topo.min(),
        "mean_elevation": topo_cell.topo.mean(),
        "land_fraction": (topo_cell.topo >= 0).sum() / topo_cell.topo.size,
    }

    return {"passed": passed, "message": message, "stats": stats}


def validate_gulf(topo_cell: var.topo_cell, feature: GeographicFeature) -> Dict:
    """Validate gulf/bay features have mostly water with some coastline."""
    # Gulfs should be mostly water but may have significant land depending on region bounds
    water_fraction = (topo_cell.topo < 0).sum() / topo_cell.topo.size
    mean_water_depth = (
        -topo_cell.topo[topo_cell.topo < 0].mean() if (topo_cell.topo < 0).any() else 0
    )

    # Adjust thresholds based on specific gulf
    if "Persian Gulf" in feature.name:
        min_water_fraction = 0.70  # Fairly shallow, wide gulf
        min_expected_depth = 30  # Persian Gulf is shallow
    else:
        min_water_fraction = 0.50  # At least 50% should be water
        min_expected_depth = 50  # Should have some depth

    passed = (
        water_fraction >= min_water_fraction and mean_water_depth >= min_expected_depth
    )

    message = (
        f"{feature.name}: water fraction {water_fraction:.1%}, "
        f"mean depth {mean_water_depth:.0f}m (expected >{min_expected_depth}m)"
    )

    stats = {
        "water_fraction": water_fraction,
        "land_fraction": (topo_cell.topo >= 0).sum() / topo_cell.topo.size,
        "mean_water_depth": mean_water_depth,
        "mean_elevation": topo_cell.topo.mean(),
        "elevation_range": topo_cell.topo.max() - topo_cell.topo.min(),
        "min_expected_depth": min_expected_depth,
    }

    return {"passed": passed, "message": message, "stats": stats}


def validate_coast(topo_cell: var.topo_cell, feature: GeographicFeature) -> Dict:
    """Validate coastal features have both land and water."""
    # Coasts should have significant mix of land and water
    water_fraction = (topo_cell.topo < 0).sum() / topo_cell.topo.size
    land_fraction = (topo_cell.topo >= 0).sum() / topo_cell.topo.size

    # Coast should have reasonable mix (20-80% water)
    min_water = 0.20
    max_water = 0.80

    passed = min_water <= water_fraction <= max_water
    message = (
        f"{feature.name}: water {water_fraction:.1%}, land {land_fraction:.1%} "
        f"(expected {min_water:.0%}-{max_water:.0%} water)"
    )

    stats = {
        "water_fraction": water_fraction,
        "land_fraction": land_fraction,
        "mean_elevation": topo_cell.topo.mean(),
        "elevation_range": topo_cell.topo.max() - topo_cell.topo.min(),
        "std_elevation": topo_cell.topo.std(),
    }

    return {"passed": passed, "message": message, "stats": stats}


# Define known geographic features for testing
GEOGRAPHIC_FEATURES = [
    # Mountains
    GeographicFeature(
        "Himalayas",
        (27.0, 30.0),
        (85.0, 90.0),
        "mountain",
        validate_mountain,
        "World's highest mountain range (Everest, K2)",
    ),
    GeographicFeature(
        "Andes (Peru)",
        (-15.0, -10.0),
        (-77.0, -72.0),
        "mountain",
        validate_mountain,
        "Andes mountain range in Peru",
    ),
    GeographicFeature(
        "Alps",
        (45.5, 47.5),
        (6.0, 11.0),
        "mountain",
        validate_mountain,
        "European Alps (Mont Blanc)",
    ),
    GeographicFeature(
        "Rockies (Colorado)",
        (38.0, 41.0),
        (-108.0, -105.0),
        "mountain",
        validate_mountain,
        "Rocky Mountains in Colorado",
    ),
    # Lakes
    GeographicFeature(
        "Lake Superior",
        (46.5, 48.5),
        (-89.0, -85.0),
        "lake",
        validate_lake,
        "Largest Great Lake by area",
    ),
    GeographicFeature(
        "Lake Baikal",
        (51.5, 55.5),
        (103.5, 109.5),
        "lake",
        validate_lake,
        "World's deepest lake in Siberia",
    ),
    GeographicFeature(
        "Lake Titicaca",
        (-16.5, -15.0),
        (-69.5, -68.5),
        "lake",
        validate_lake,
        "High-altitude lake in Andes (Peru/Bolivia border)",
    ),
    # Oceans
    GeographicFeature(
        "Pacific Ocean (mid)",
        (10.0, 15.0),
        (-160.0, -150.0),
        "ocean",
        validate_ocean,
        "Central Pacific Ocean",
    ),
    GeographicFeature(
        "Atlantic Ocean (mid)",
        (25.0, 30.0),
        (-50.0, -40.0),
        "ocean",
        validate_ocean,
        "Central Atlantic Ocean",
    ),
    # Gulfs and Bays
    GeographicFeature(
        "Gulf of Mexico",
        (27.0, 29.5),
        (-94.0, -89.0),
        "gulf",
        validate_gulf,
        "Gulf of Mexico central region with coastal areas",
    ),
    GeographicFeature(
        "Persian Gulf",
        (26.0, 28.0),
        (50.0, 52.0),
        "gulf",
        validate_gulf,
        "Persian Gulf between Iran and Arabia",
    ),
    # Coasts
    GeographicFeature(
        "California Coast",
        (35.0, 37.0),
        (-122.0, -120.0),
        "coast",
        validate_coast,
        "California coastline near Monterey",
    ),
    GeographicFeature(
        "Mediterranean Coast (Spain)",
        (40.0, 42.0),
        (1.0, 3.0),
        "coast",
        validate_coast,
        "Spanish Mediterranean coast",
    ),
]


class TestICONETOPOValidation:
    """Validate ICON grid cells against ETOPO topography."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup test parameters and data structures."""
        params = var.params()
        utils.transfer_attributes(params, local_paths.paths, prefix="path")
        params.etopo_cg = 4  # Use coarse-graining for faster tests
        params.padding = 0

        # Load ICON grid
        grid = var.grid()
        reader = io.ncdata(padding=params.padding, padding_tol=60)
        reader.read_dat(params.path_icon_grid, grid)
        grid.apply_f(utils.rad2deg)

        return {"params": params, "grid": grid, "reader": reader}

    def load_region_topography(
        self,
        setup: Dict,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
    ) -> var.topo_cell:
        """
        Load topography for a specific lat/lon region.

        Args:
            setup: Test setup dictionary with params and reader
            lat_range: (min_lat, max_lat) in degrees
            lon_range: (min_lon, max_lon) in degrees

        Returns:
            topo_cell with loaded topography data
        """
        params = setup["params"]
        reader = setup["reader"]

        # Set region extents
        params.lat_extent = list(lat_range)
        params.lon_extent = list(lon_range)

        # Load topography
        topo = var.topo_cell()
        etopo_reader = reader.read_etopo_topo(
            None, params, is_parallel=True, verbose=False
        )
        etopo_reader.get_topo(topo)
        etopo_reader.close_cached_files()

        # Generate mesh grids
        topo.gen_mgrids()

        return topo

    def load_cell_topography(
        self, setup: Dict, cell_idx: int
    ) -> Tuple[var.topo_cell, np.ndarray, np.ndarray]:
        """
        Load topography for a specific ICON grid cell.

        Args:
            setup: Test setup dictionary
            cell_idx: ICON grid cell index

        Returns:
            (topo_cell, lat_vertices, lon_vertices)
        """
        params = setup["params"]
        grid = setup["grid"]
        reader = setup["reader"]

        # Get cell vertices
        lat_verts = grid.clat_vertices[cell_idx]
        lon_verts = grid.clon_vertices[cell_idx]

        # Handle edge cases (dateline, poles)
        lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)
        params.lat_extent = lat_extent
        params.lon_extent = lon_extent

        # Load topography
        topo = var.topo_cell()
        etopo_reader = reader.read_etopo_topo(
            None, params, is_parallel=True, verbose=False
        )
        etopo_reader.get_topo(topo)
        etopo_reader.close_cached_files()

        topo.gen_mgrids()

        return topo, lat_verts, lon_verts

    def test_topography_data_quality_basic(self, setup):
        """Test that loaded topography has valid data structure."""
        # Load a simple region (central Pacific)
        topo = self.load_region_topography(setup, (10.0, 20.0), (-160.0, -150.0))

        # Basic structure checks
        assert topo.topo is not None, "No topography loaded"
        assert (
            topo.lat is not None and topo.lon is not None
        ), "Missing coordinate arrays"
        assert topo.topo.shape[0] == len(topo.lat), "Latitude dimension mismatch"
        assert topo.topo.shape[1] == len(topo.lon), "Longitude dimension mismatch"

        # Check for NaN values
        nan_count = np.sum(np.isnan(topo.topo))
        assert nan_count == 0, f"Found {nan_count} NaN values in topography"

        # Sanity check elevation range (Earth surface)
        assert (
            topo.topo.min() >= -12000
        ), f"Elevation too low: {topo.topo.min()}m (deepest ocean ~-11km)"
        assert (
            topo.topo.max() <= 9000
        ), f"Elevation too high: {topo.topo.max()}m (Everest ~8.8km)"

        print(
            f"✓ Data quality check passed: shape={topo.topo.shape}, "
            f"elev=[{topo.topo.min():.0f}, {topo.topo.max():.0f}]m"
        )

    @pytest.mark.parametrize("feature", GEOGRAPHIC_FEATURES, ids=lambda f: f.name)
    def test_geographic_feature(self, setup, feature: GeographicFeature):
        """Test that a specific geographic feature validates correctly."""
        print(f"\nTesting: {feature.name} ({feature.feature_type})")
        print(f"  Location: lat={feature.lat_range}, lon={feature.lon_range}")
        print(f"  Description: {feature.description}")

        # Load topography for this region
        topo = self.load_region_topography(setup, feature.lat_range, feature.lon_range)

        # Validate against feature
        result = feature.validate(topo)

        # Print statistics
        print(f"  {result['message']}")
        for key, value in result["stats"].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")

        # Assert validation passed
        assert result[
            "passed"
        ], f"{feature.name} validation failed: {result['message']}"
        print(f"  ✓ Validation PASSED")

    def test_cell_near_himalayas(self, setup):
        """Test loading a cell near the Himalayas and verify high elevations."""
        grid = setup["grid"]

        # Find cell near Himalayas (28°N, 87°E - near Everest)
        cell_idx = utils.pick_cell(lat_ref=28.0, lon_ref=87.0, grid=grid, radius=1.0)
        assert cell_idx is not None, "Could not find cell near Himalayas"

        print(f"\nTesting ICON cell {cell_idx} near Himalayas")

        # Load cell topography
        topo, lat_verts, lon_verts = self.load_cell_topography(setup, cell_idx)

        print(
            f"  Cell vertices: lat={np.rad2deg(lat_verts)}, lon={np.rad2deg(lon_verts)}"
        )
        print(f"  Topography shape: {topo.topo.shape}")
        print(
            f"  Elevation: [{topo.topo.min():.0f}, {topo.topo.max():.0f}]m, mean={topo.topo.mean():.0f}m"
        )

        # Verify high elevations
        assert (
            topo.topo.max() > 4000
        ), f"Expected high peaks in Himalayas, got {topo.topo.max():.0f}m"
        assert (
            topo.topo.mean() > 2000
        ), f"Expected high mean elevation, got {topo.topo.mean():.0f}m"

        print(f"  ✓ Himalayan cell validation PASSED")

    def test_cell_in_pacific_ocean(self, setup):
        """Test loading a cell in the Pacific Ocean and verify it's water."""
        grid = setup["grid"]

        # Find cell in Pacific (15°N, 155°W)
        cell_idx = utils.pick_cell(lat_ref=15.0, lon_ref=-155.0, grid=grid, radius=1.0)
        assert cell_idx is not None, "Could not find cell in Pacific"

        print(f"\nTesting ICON cell {cell_idx} in Pacific Ocean")

        # Load cell topography
        topo, lat_verts, lon_verts = self.load_cell_topography(setup, cell_idx)

        print(
            f"  Cell vertices: lat={np.rad2deg(lat_verts)}, lon={np.rad2deg(lon_verts)}"
        )
        print(f"  Topography shape: {topo.topo.shape}")
        print(
            f"  Elevation: [{topo.topo.min():.0f}, {topo.topo.max():.0f}]m, mean={topo.topo.mean():.0f}m"
        )

        # Verify it's ocean
        water_fraction = (topo.topo < 0).sum() / topo.topo.size
        print(f"  Water fraction: {water_fraction:.1%}")

        assert (
            water_fraction > 0.95
        ), f"Expected mostly water in Pacific, got {water_fraction:.1%}"
        assert (
            topo.topo.mean() < -1000
        ), f"Expected deep ocean, got mean depth {-topo.topo.mean():.0f}m"

        print(f"  ✓ Pacific Ocean cell validation PASSED")

    def test_cell_on_california_coast(self, setup):
        """Test loading a coastal cell and verify land-water mix."""
        grid = setup["grid"]

        # Find cell on California coast (36°N, 122°W)
        cell_idx = utils.pick_cell(lat_ref=36.0, lon_ref=-122.0, grid=grid, radius=1.0)
        assert cell_idx is not None, "Could not find cell on California coast"

        print(f"\nTesting ICON cell {cell_idx} on California coast")

        # Load cell topography
        topo, lat_verts, lon_verts = self.load_cell_topography(setup, cell_idx)

        print(
            f"  Cell vertices: lat={np.rad2deg(lat_verts)}, lon={np.rad2deg(lon_verts)}"
        )
        print(f"  Topography shape: {topo.topo.shape}")
        print(f"  Elevation: [{topo.topo.min():.0f}, {topo.topo.max():.0f}]m")

        # Verify it's coastal (mix of land and water)
        water_fraction = (topo.topo < 0).sum() / topo.topo.size
        land_fraction = (topo.topo >= 0).sum() / topo.topo.size

        print(f"  Water fraction: {water_fraction:.1%}")
        print(f"  Land fraction: {land_fraction:.1%}")

        # Coast should have both land and water
        assert (
            0.10 < water_fraction < 0.90
        ), f"Expected coastal mix, got {water_fraction:.1%} water"

        print(f"  ✓ Coastal cell validation PASSED")

    def test_multiple_cells_consistency(self, setup):
        """Test that multiple cells across different regions load consistently."""
        grid = setup["grid"]

        # Test cells at various locations
        test_locations = [
            (0.0, 0.0, "Equator/Prime Meridian"),
            (45.0, 0.0, "Mid-latitude Europe"),
            (0.0, 180.0, "Equator/Dateline"),
            (-30.0, 150.0, "Australia region"),
            (60.0, -100.0, "Northern Canada"),
        ]

        results = []
        for lat, lon, description in test_locations:
            cell_idx = utils.pick_cell(lat_ref=lat, lon_ref=lon, grid=grid, radius=1.0)
            if cell_idx is None:
                print(f"  ⚠ Could not find cell at {description} ({lat}, {lon})")
                continue

            try:
                topo, lat_verts, lon_verts = self.load_cell_topography(setup, cell_idx)

                result = {
                    "location": description,
                    "cell_idx": cell_idx,
                    "lat": lat,
                    "lon": lon,
                    "shape": topo.topo.shape,
                    "elev_min": topo.topo.min(),
                    "elev_max": topo.topo.max(),
                    "elev_mean": topo.topo.mean(),
                    "has_nan": np.isnan(topo.topo).any(),
                    "success": True,
                }
                results.append(result)

                print(
                    f"  ✓ Cell {cell_idx} ({description}): "
                    f"shape={topo.topo.shape}, elev=[{topo.topo.min():.0f}, {topo.topo.max():.0f}]m"
                )

            except Exception as e:
                print(f"  ✗ Cell {cell_idx} ({description}) FAILED: {str(e)}")
                results.append(
                    {
                        "location": description,
                        "cell_idx": cell_idx,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Verify all succeeded
        success_count = sum(1 for r in results if r["success"])
        print(f"\n  Summary: {success_count}/{len(results)} cells loaded successfully")

        assert success_count == len(
            results
        ), f"Some cells failed to load: {len(results) - success_count} failures"

        # Verify no NaN values in any cell
        nan_count = sum(1 for r in results if r.get("has_nan", False))
        assert nan_count == 0, f"Found NaN values in {nan_count} cells"


class TestICONETOPOVisualization:
    """Optional visualization tests for debugging (requires matplotlib)."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup test parameters and data structures."""
        params = var.params()
        utils.transfer_attributes(params, local_paths.paths, prefix="path")
        params.etopo_cg = 4
        params.padding = 0

        grid = var.grid()
        reader = io.ncdata(padding=params.padding, padding_tol=60)
        reader.read_dat(params.path_icon_grid, grid)
        grid.apply_f(utils.rad2deg)

        return {"params": params, "grid": grid, "reader": reader}

    def test_visualize_feature(self, setup):
        """Visualize a geographic feature for debugging.

        Run with: pytest -v -s -k visualization
        """
        # Pick a feature to visualize (Himalayas)
        feature = GEOGRAPHIC_FEATURES[5]  # Himalayas

        # Load topography
        params = setup["params"]
        reader = setup["reader"]

        params.lat_extent = list(feature.lat_range)
        params.lon_extent = list(feature.lon_range)

        topo = var.topo_cell()
        etopo_reader = reader.read_etopo_topo(
            None, params, is_parallel=True, verbose=True
        )
        etopo_reader.get_topo(topo)
        etopo_reader.close_cached_files()
        topo.gen_mgrids()

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Raw topography
        im1 = axes[0].imshow(topo.topo, origin="lower", cmap="terrain", aspect="auto")
        axes[0].set_title(f"{feature.name} - Raw Topography")
        axes[0].set_xlabel(f"Longitude index")
        axes[0].set_ylabel(f"Latitude index")
        plt.colorbar(im1, ax=axes[0], label="Elevation (m)")

        # Plot 2: Contour plot with coordinates
        levels = 20
        cs = axes[1].contourf(
            topo.lon_grid, topo.lat_grid, topo.topo, levels=levels, cmap="terrain"
        )
        axes[1].set_title(f"{feature.name} - Contour Plot")
        axes[1].set_xlabel("Longitude (°)")
        axes[1].set_ylabel("Latitude (°)")
        plt.colorbar(cs, ax=axes[1], label="Elevation (m)")

        plt.tight_layout()

        # Save figure
        output_dir = Path(__file__).parent.parent / "outputs" / "test_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"validation_{feature.name.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved visualization to: {output_path}")

        plt.show()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
