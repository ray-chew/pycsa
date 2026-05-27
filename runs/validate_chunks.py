"""
Quick validation script to check NetCDF chunk completeness.

Usage:
    python3 -m runs.validate_chunks [--datasets-dir PATH]
"""

from pathlib import Path
import re
import argparse

import netCDF4


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

    # Per-chunk cell-count check — catches the nc_writer truncation symptom
    # (filename range says 100 cells, but the file only contains a fraction).
    # Opens every chunk; metadata-only read, takes a few seconds for 200+ chunks.
    print("\nPer-chunk cell counts:")
    incomplete_chunks = []
    claimed_cells = 0
    actual_cells = 0
    for chunk in chunks:
        expected_in_chunk = chunk["end"] - chunk["start"] + 1
        claimed_cells += expected_in_chunk
        with netCDF4.Dataset(chunk["filepath"]) as nc:
            actual = len(nc.groups)
        actual_cells += actual
        if actual != expected_in_chunk:
            incomplete_chunks.append((chunk, actual, expected_in_chunk))
            print(
                f"  {chunk['filepath'].name}: {actual}/{expected_in_chunk}  [INCOMPLETE]"
            )
    if incomplete_chunks:
        issues.append(
            f"{len(incomplete_chunks)}/{len(chunks)} chunk files have fewer cells "
            f"than their filename range claims"
        )
    else:
        print(f"  All {len(chunks)} chunks have the expected cell count.")

    # Coverage: what the filenames claim vs. what's actually inside, both
    # relative to the full grid.
    expected_cells = 20480
    print(
        f"\nCoverage (filename-claimed): {claimed_cells}/{expected_cells} cells "
        f"({claimed_cells / expected_cells * 100:.1f}%)"
    )
    print(
        f"Coverage (actually present): {actual_cells}/{expected_cells} cells "
        f"({actual_cells / expected_cells * 100:.1f}%)"
    )
    if claimed_cells < expected_cells:
        issues.append(
            f"Incomplete: filenames cover only {claimed_cells}/{expected_cells} cells"
        )
    if actual_cells < claimed_cells:
        issues.append(
            f"Underfilled: only {actual_cells} cells written across "
            f"{claimed_cells} claimed"
        )

    # Calculate total size
    total_size_mb = sum(c["size_kb"] for c in chunks) / 1024
    print(f"\nTotal size: {total_size_mb:.1f} MB")

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
