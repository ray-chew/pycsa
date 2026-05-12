"""
Quick validation script to check NetCDF chunk completeness.

Usage:
    python3 -m runs.validate_chunks [--datasets-dir PATH]
"""

from pathlib import Path
import re
import argparse


def main():
    parser = argparse.ArgumentParser(description="Validate ICON ETOPO NetCDF chunks")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        help="Directory containing chunk files (default: auto-detect)",
    )
    args = parser.parse_args()

    # Find datasets directory
    if args.datasets_dir:
        datasets_dir = Path(args.datasets_dir)
    else:
        possible_paths = [
            Path("outputs/global_run/datasets"),
            Path("../outputs/global_run/datasets"),
            Path("../../outputs/global_run/datasets"),
        ]
        datasets_dir = None
        for path in possible_paths:
            if path.exists():
                datasets_dir = path
                break

        if datasets_dir is None:
            print("❌ Could not find datasets directory")
            return 1

    print(f"Checking: {datasets_dir}\n")

    # Find chunk files
    pattern = re.compile(r"icon_etopo_global_cells_(\d+)-(\d+)\.nc")
    chunks = []

    for filepath in sorted(datasets_dir.glob("icon_etopo_global_cells_*.nc")):
        match = pattern.match(filepath.name)
        if match:
            start_cell = int(match.group(1))
            end_cell = int(match.group(2))
            file_size = filepath.stat().st_size / 1024  # KB
            chunks.append(
                {
                    "filepath": filepath,
                    "start": start_cell,
                    "end": end_cell,
                    "size_kb": file_size,
                }
            )

    chunks = sorted(chunks, key=lambda x: x["start"])

    if not chunks:
        print("❌ No chunk files found!")
        print(f"   Looking for: icon_etopo_global_cells_*.nc")
        return 1

    # Display summary
    print(f"Found {len(chunks)} chunk files:")
    print(f"  First: cells {chunks[0]['start']}-{chunks[0]['end']}")
    print(f"  Last:  cells {chunks[-1]['start']}-{chunks[-1]['end']}")

    # Check for issues
    issues = []

    # Check for gaps
    for i in range(len(chunks) - 1):
        current_end = chunks[i]["end"]
        next_start = chunks[i + 1]["start"]
        if current_end + 1 != next_start:
            issues.append(
                f"Gap: chunk {i} ends at {current_end}, chunk {i+1} starts at {next_start}"
            )

    # Check start
    if chunks[0]["start"] != 0:
        issues.append(
            f"First chunk doesn't start at 0 (starts at {chunks[0]['start']})"
        )

    # Check expected coverage
    expected_cells = 20480
    total_cells = chunks[-1]["end"] + 1 - chunks[0]["start"]

    print(
        f"\nCoverage: {total_cells}/{expected_cells} cells ({total_cells/expected_cells*100:.1f}%)"
    )

    if total_cells < expected_cells:
        issues.append(f"Incomplete: only {total_cells}/{expected_cells} cells")

    # Calculate total size
    total_size_mb = sum(c["size_kb"] for c in chunks) / 1024
    print(f"Total size: {total_size_mb:.1f} MB")

    # Report
    print("\n" + "=" * 60)
    if issues:
        print("⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("=" * 60)
        return 1
    else:
        print("✓ ALL CHECKS PASSED")
        print("  - No gaps in cell coverage")
        print("  - All chunks present")
        print("\nReady to merge with:")
        print("  python3 -m runs.merge_netcdf_chunks")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    exit(main())
