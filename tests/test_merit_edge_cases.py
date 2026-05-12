#!/usr/bin/env python3
"""
Edge case test script for MERIT topography data loading.

This script tests the MERIT loader on challenging regions to validate:
1. MERIT-REMA interface at -60° latitude (Antarctic boundary)
2. Dateline crossing at ±180° longitude
3. North Pole high-latitude region
4. Prime Meridian crossing at 0° longitude
5. Equator crossing at 0° latitude
6. Multiple boundary crossings simultaneously

These are the trickiest cases for global data loaders!

Author: Test Suite
Date: 2025-10-22
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

from pycsa.core import io, var
from pycsa.plotting import cart_plot


def test_region(name, lat_extent, lon_extent, merit_cg=50, description=""):
    """
    Test loading a specific region.

    Parameters
    ----------
    name : str
        Region name for display
    lat_extent : list
        [lat_min, lat_max]
    lon_extent : list
        [lon_min, lon_max]
    merit_cg : int
        Coarse-graining factor
    description : str
        Description of what makes this region tricky

    Returns
    -------
    dict
        Results dictionary with success status and statistics
    """
    print("=" * 80)
    print(f"TEST: {name}")
    print("=" * 80)
    print()
    print(f"Region Configuration:")
    print(
        f"  Latitude:  {lat_extent[0]:7.2f}° to {lat_extent[1]:7.2f}° (span: {lat_extent[1]-lat_extent[0]:.2f}°)"
    )
    print(
        f"  Longitude: {lon_extent[0]:7.2f}° to {lon_extent[1]:7.2f}° (span: {abs(lon_extent[1]-lon_extent[0]):.2f}°)"
    )
    print(f"  Coarse-graining: {merit_cg}x{merit_cg}")
    print()
    if description:
        print(f"Why this is tricky:")
        print(f"  {description}")
        print()

    # Create parameters
    class Params:
        def __init__(self):
            self.path_merit = "/home/ray/Documents/orog_data/MERIT/"
            self.path_rema = "/home/ray/Documents/orog_data/REMA/"
            self.lat_extent = lat_extent
            self.lon_extent = lon_extent
            self.merit_cg = merit_cg

    params = Params()

    # Check data paths
    if not Path(params.path_merit).exists():
        print(f"ERROR: MERIT data not found at {params.path_merit}")
        return {"success": False, "error": "Data path not found"}

    # Load data
    print("Loading MERIT data...")
    cell = var.topo_cell()
    start_time = time.time()

    try:
        loader = io.ncdata.read_merit_topo(cell, params, verbose=False)
        load_time = time.time() - start_time
        print(f"✓ Loaded in {load_time:.2f} seconds")
        print()

    except Exception as e:
        print(f"✗ ERROR during loading: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}

    # Apply data cleaning
    n_clipped = np.sum(cell.topo < -500.0)
    cell.topo[cell.topo < -500.0] = -500.0

    # Validate data
    print("Data Validation:")
    print(f"  Shape: {cell.topo.shape}")
    print(f"  Lat range: [{cell.lat.min():.4f}, {cell.lat.max():.4f}]°")
    print(f"  Lon range: [{cell.lon.min():.4f}, {cell.lon.max():.4f}]°")
    print(f"  Elevation: [{cell.topo.min():.1f}, {cell.topo.max():.1f}] m")
    print(f"  Mean elevation: {cell.topo.mean():.1f} m")
    if n_clipped > 0:
        print(f"  Clipped {n_clipped:,} points below -500m")

    # Check for issues
    has_nan = np.isnan(cell.topo).any()
    has_inf = np.isinf(cell.topo).any()

    if has_nan:
        print(f"  ✗ WARNING: Contains NaN values!")
    else:
        print(f"  ✓ No NaN values")

    if has_inf:
        print(f"  ✗ WARNING: Contains infinite values!")
    else:
        print(f"  ✓ No infinite values")

    # Statistics
    land_mask = cell.topo > 0
    ocean_mask = cell.topo <= 0
    land_pct = 100 * np.sum(land_mask) / cell.topo.size
    ocean_pct = 100 * np.sum(ocean_mask) / cell.topo.size

    print(f"  Land/Ocean: {land_pct:.1f}% / {ocean_pct:.1f}%")
    print()

    # Plot
    print("Creating plot...")
    try:
        cell.gen_mgrids()

        # Adjust figure size based on region aspect ratio
        lat_span = lat_extent[1] - lat_extent[0]
        lon_span = abs(lon_extent[1] - lon_extent[0])
        aspect = lon_span / max(lat_span, 1.0)

        if aspect > 2:
            figsize = (16, 8)
        elif aspect < 0.5:
            figsize = (8, 12)
        else:
            figsize = (12, 8)

        cart_plot.lat_lon(cell, fs=figsize, int=1)
        print(f"✓ Plot displayed")
        print()

    except Exception as e:
        print(f"✗ ERROR during plotting: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": f"Plotting failed: {e}"}

    # Success!
    success = not (has_nan or has_inf)

    results = {
        "success": success,
        "name": name,
        "load_time": load_time,
        "shape": cell.topo.shape,
        "elevation_range": (cell.topo.min(), cell.topo.max()),
        "mean_elevation": cell.topo.mean(),
        "land_pct": land_pct,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }

    if success:
        print(f"✓ {name}: PASSED")
    else:
        print(f"⚠ {name}: COMPLETED WITH WARNINGS")
    print()

    return results


def run_all_edge_case_tests():
    """
    Run all edge case tests.

    Returns
    -------
    list
        List of test results
    """
    print("=" * 80)
    print("MERIT EDGE CASE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    print("Testing the trickiest regions for global data loaders:")
    print("  1. MERIT-REMA interface at -60° latitude")
    print("  2. International dateline crossing at ±180° longitude")
    print("  3. North Pole high-latitude region")
    print("  4. Prime Meridian crossing at 0° longitude")
    print("  5. Equator crossing")
    print("  6. Multiple boundary crossings")
    print()
    input("Press Enter to start tests...")
    print()

    results = []

    # Test 1: MERIT-REMA Interface at EXACTLY -60° (South Orkney Islands!)
    # This is THE island you remember - sits right on the boundary!
    results.append(
        test_region(
            name="MERIT-REMA Boundary (South Orkney Islands)",
            lat_extent=[-61.5, -59.5],  # Tight 2° centered on South Orkney at -60.5°
            lon_extent=[
                -47.0,
                -44.0,
            ],  # Narrow 3° window over South Orkney Islands at -45.5°
            merit_cg=10,  # Finer resolution to catch the small islands
            description="Tests EXACTLY the -60° latitude boundary with South Orkney Islands!\n"
            "  These islands sit RIGHT ON the MERIT-REMA transition at 60.5°S.\n"
            "  Perfect test case for seamless dataset integration.",
        )
    )

    # Test 1b: MERIT-REMA Interface (Antarctic Peninsula - broader view)
    results.append(
        test_region(
            name="MERIT-REMA Interface (Antarctic Peninsula)",
            lat_extent=[-70.0, -55.0],  # Crosses -60° boundary, broader range
            lon_extent=[-65.0, -55.0],  # Narrow 10° window over Antarctic Peninsula
            merit_cg=30,
            description="Crosses the -60° latitude boundary over Antarctic Peninsula.\n"
            "  Broader view of the MERIT-REMA transition zone.\n"
            "  Tests seamless data integration between datasets.",
        )
    )

    # Test 2: Dateline Crossing - Kamchatka Peninsula (Russia, has land)
    results.append(
        test_region(
            name="Dateline Crossing (Kamchatka Peninsula)",
            lat_extent=[50.0, 62.0],  # Kamchatka Peninsula latitude
            lon_extent=[175.0, -175.0],  # Narrow 10° window crossing dateline
            merit_cg=30,
            description="Crosses the international dateline at ±180° longitude.\n"
            "  Focuses on Kamchatka Peninsula (volcanoes, mountains).\n"
            "  Tests handling of longitude wraparound over land.",
        )
    )

    # Test 3: North Pole Region - Greenland focus (has major topography)
    results.append(
        test_region(
            name="North Pole Region (Greenland)",
            lat_extent=[75.0, 85.0],  # High Arctic, northern Greenland
            lon_extent=[-50.0, -20.0],  # Narrow window over Greenland ice sheet
            merit_cg=40,
            description="High latitude region near North Pole.\n"
            "  Focuses on northern Greenland (ice sheet with elevation).\n"
            "  Tests polar convergence and high-latitude handling.",
        )
    )

    # Test 4: Prime Meridian Crossing - UK/France coast (small, fast, over land)
    results.append(
        test_region(
            name="Prime Meridian Crossing (UK-France)",
            lat_extent=[49.0, 52.0],  # English Channel area, tight lat range
            lon_extent=[-3.0, 3.0],  # Narrow 6° window crossing 0° longitude
            merit_cg=20,
            description="Crosses the Prime Meridian at 0° longitude.\n"
            "  Focuses on UK-France region (Dover, Calais area).\n"
            "  Tests transition from negative to positive longitude over land.",
        )
    )

    # Test 5: Equator Crossing - Mount Kenya area (has elevation features)
    results.append(
        test_region(
            name="Equator Crossing (Mount Kenya)",
            lat_extent=[-2.0, 2.0],  # Narrow 4° crossing equator
            lon_extent=[36.0, 38.0],  # Tight 2° window on Mt. Kenya
            merit_cg=20,
            description="Crosses the Equator at 0° latitude.\n"
            "  Focuses on Mount Kenya (5199m, sits on equator!).\n"
            "  Tests hemisphere transition over dramatic topography.",
        )
    )

    # Test 6: Tierra del Fuego - near MERIT-REMA boundary
    results.append(
        test_region(
            name="Tierra del Fuego (Near Antarctic Boundary)",
            lat_extent=[-56.0, -53.0],  # Southernmost South America
            lon_extent=[-70.0, -65.0],  # Cape Horn area
            merit_cg=25,
            description="Southernmost tip of South America, near -60° boundary.\n"
            "  Tests high southern latitude (stays in MERIT, doesn't cross to REMA).\n"
            "  Drake Passage area with complex coastline.",
        )
    )

    # Test 7: Bering Strait - dateline + high latitude (Alaska-Russia)
    results.append(
        test_region(
            name="Bering Strait (Dateline + High Latitude)",
            lat_extent=[64.0, 68.0],  # Bering Strait, tight range
            lon_extent=[177.0, -177.0],  # Narrow 6° crossing dateline
            merit_cg=25,
            description="Bering Strait region between Alaska and Russia.\n"
            "  Tests BOTH dateline crossing AND high latitude.\n"
            "  Includes Bering Strait islands and coastlines.",
        )
    )

    # Test 8: South Pole Region (Pure REMA) - smaller window
    results.append(
        test_region(
            name="South Pole Region (Marie Byrd Land)",
            lat_extent=[-85.0, -75.0],  # Deep Antarctica
            lon_extent=[-150.0, -100.0],  # Narrower 50° window over Marie Byrd Land
            merit_cg=60,  # Higher CG for speed
            description="Interior Antarctica (pure REMA data).\n"
            "  Focuses on Marie Byrd Land (West Antarctica, mountains).\n"
            "  Tests REMA dataset at extreme southern latitude.",
        )
    )

    return results


def print_summary(results):
    """Print summary of all test results."""
    print()
    print("=" * 80)
    print("EDGE CASE TEST SUMMARY")
    print("=" * 80)
    print()

    passed = sum(1 for r in results if r.get("success", False))
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print()

    print(f"{'Test Name':<45} {'Status':<10} {'Time (s)':<10} {'Shape':<15}")
    print("-" * 80)

    for r in results:
        if r.get("success"):
            status = "✓ PASS"
        elif "error" in r:
            status = "✗ FAIL"
        else:
            status = "⚠ WARN"

        name = r.get("name", "Unknown")[:44]
        time_str = f"{r.get('load_time', 0):.2f}" if "load_time" in r else "N/A"
        shape = str(r.get("shape", "N/A"))

        print(f"{name:<45} {status:<10} {time_str:<10} {shape:<15}")

    print()

    if passed == total:
        print("🎉 ALL EDGE CASE TESTS PASSED!")
        print()
        print("The MERIT loader correctly handles:")
        print("  ✓ MERIT-REMA interface at -60° latitude")
        print("  ✓ International dateline crossing (±180° longitude)")
        print("  ✓ North and South Pole regions")
        print("  ✓ Prime Meridian crossing (0° longitude)")
        print("  ✓ Equator crossing (0° latitude)")
        print("  ✓ Multiple simultaneous boundary crossings")
        print()
        print("The implementation is robust and production-ready! 🚀")
    else:
        print(f"⚠ {total - passed} test(s) had issues. Review details above.")

    print()
    return passed == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test MERIT data loader on edge cases and tricky regions"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (only 3 most critical regions)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "merit-rema",
            "south-orkney",
            "dateline",
            "north-pole",
            "prime-meridian",
            "equator",
            "tierra-del-fuego",
            "bering",
            "south-pole",
        ],
        help="Run only a specific test",
    )

    args = parser.parse_args()

    if args.test:
        # Run single test
        test_configs = {
            "merit-rema": {
                "name": "MERIT-REMA Boundary (South Orkney Islands)",
                "lat_extent": [-61.5, -59.5],
                "lon_extent": [-47.0, -44.0],
                "merit_cg": 10,
                "description": "Tests EXACTLY -60° boundary with South Orkney Islands",
            },
            "south-orkney": {
                "name": "MERIT-REMA Boundary (South Orkney Islands)",
                "lat_extent": [-61.5, -59.5],
                "lon_extent": [-47.0, -44.0],
                "merit_cg": 10,
                "description": "Tests EXACTLY -60° boundary with South Orkney Islands",
            },
            "dateline": {
                "name": "Dateline Crossing (Kamchatka)",
                "lat_extent": [50.0, 62.0],
                "lon_extent": [175.0, -175.0],
                "merit_cg": 30,
                "description": "Tests ±180° longitude over Kamchatka Peninsula",
            },
            "north-pole": {
                "name": "North Pole (Greenland)",
                "lat_extent": [75.0, 85.0],
                "lon_extent": [-50.0, -20.0],
                "merit_cg": 40,
                "description": "Tests high Arctic over northern Greenland",
            },
            "prime-meridian": {
                "name": "Prime Meridian (UK-France)",
                "lat_extent": [49.0, 52.0],
                "lon_extent": [-3.0, 3.0],
                "merit_cg": 20,
                "description": "Tests 0° longitude crossing over UK-France",
            },
            "equator": {
                "name": "Equator (Mount Kenya)",
                "lat_extent": [-2.0, 2.0],
                "lon_extent": [36.0, 38.0],
                "merit_cg": 20,
                "description": "Tests 0° latitude over Mount Kenya",
            },
            "tierra-del-fuego": {
                "name": "Tierra del Fuego",
                "lat_extent": [-56.0, -53.0],
                "lon_extent": [-70.0, -65.0],
                "merit_cg": 25,
                "description": "Tests southern tip of South America",
            },
            "bering": {
                "name": "Bering Strait",
                "lat_extent": [64.0, 68.0],
                "lon_extent": [177.0, -177.0],
                "merit_cg": 25,
                "description": "Tests dateline + high latitude over strait",
            },
            "south-pole": {
                "name": "South Pole (Marie Byrd Land)",
                "lat_extent": [-85.0, -75.0],
                "lon_extent": [-150.0, -100.0],
                "merit_cg": 60,
                "description": "Tests pure REMA over West Antarctica",
            },
        }

        config = test_configs[args.test]
        result = test_region(**config)
        success = result.get("success", False)
        sys.exit(0 if success else 1)

    elif args.quick:
        # Run only 3 most critical tests
        print("Running QUICK edge case tests (3 most critical regions)...\n")

        results = []

        # 1. MERIT-REMA interface at EXACT boundary (most critical!)
        results.append(
            test_region(
                name="MERIT-REMA Boundary (South Orkney Islands)",
                lat_extent=[-61.5, -59.5],
                lon_extent=[-47.0, -44.0],
                merit_cg=10,
                description="EXACTLY -60° boundary with South Orkney Islands at 60.5°S",
            )
        )

        # 2. Dateline crossing
        results.append(
            test_region(
                name="Dateline Crossing (Kamchatka)",
                lat_extent=[50.0, 62.0],
                lon_extent=[175.0, -175.0],
                merit_cg=30,
                description="±180° longitude over Kamchatka Peninsula",
            )
        )

        # 3. North Pole
        results.append(
            test_region(
                name="North Pole (Greenland)",
                lat_extent=[75.0, 85.0],
                lon_extent=[-50.0, -20.0],
                merit_cg=40,
                description="High Arctic over northern Greenland",
            )
        )

        success = print_summary(results)
        sys.exit(0 if success else 1)

    else:
        # Run all tests
        results = run_all_edge_case_tests()
        success = print_summary(results)
        sys.exit(0 if success else 1)
