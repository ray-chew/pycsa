"""
Merge NetCDF chunk files into a single final NetCDF file.

This script:
1. Finds all icon_etopo_global_cells_*.nc files
2. Validates that all expected chunks are present
3. Merges them into icon_etopo_global_FINAL.nc
4. Optionally removes intermediate chunk files

Usage:
    python3 -m runs.merge_netcdf_chunks [--cleanup] [--output OUTPUT_NAME]

Options:
    --cleanup     Remove intermediate chunk files after successful merge
    --output      Output filename (default: icon_etopo_global_FINAL.nc)
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
import re
import argparse
from tqdm import tqdm


def find_chunk_files(datasets_dir):
    """Find all NetCDF chunk files and extract their cell ranges."""
    pattern = re.compile(r"icon_etopo_global_cells_(\d+)-(\d+)\.nc")

    chunks = []
    for filepath in sorted(datasets_dir.glob("icon_etopo_global_cells_*.nc")):
        match = pattern.match(filepath.name)
        if match:
            start_cell = int(match.group(1))
            end_cell = int(match.group(2))
            chunks.append(
                {
                    "filepath": filepath,
                    "start": start_cell,
                    "end": end_cell,
                    "size": end_cell - start_cell + 1,
                }
            )

    return sorted(chunks, key=lambda x: x["start"])


def validate_chunks(chunks, expected_total_cells=20480):
    """Validate that chunks cover all cells without gaps or overlaps."""
    if not chunks:
        raise ValueError("No chunk files found!")

    print(f"\nFound {len(chunks)} chunk files")
    print(f"  First chunk: cells {chunks[0]['start']}-{chunks[0]['end']}")
    print(f"  Last chunk: cells {chunks[-1]['start']}-{chunks[-1]['end']}")

    # Check for gaps
    for i in range(len(chunks) - 1):
        current_end = chunks[i]["end"]
        next_start = chunks[i + 1]["start"]
        if current_end + 1 != next_start:
            raise ValueError(
                f"Gap detected: chunk ends at {current_end}, next starts at {next_start}"
            )

    # Check coverage
    total_cells = chunks[-1]["end"] + 1 - chunks[0]["start"]
    if chunks[0]["start"] != 0:
        print(f"\n⚠ Warning: First chunk starts at cell {chunks[0]['start']}, not 0")

    if total_cells < expected_total_cells:
        print(f"\n⚠ Warning: Only {total_cells}/{expected_total_cells} cells covered")

    print(f"\n✓ Validation passed: {total_cells} cells in {len(chunks)} chunks\n")
    return True


def merge_chunks(chunks, output_path, datasets_dir):
    """Merge chunk files into a single NetCDF file."""

    print(f"Merging {len(chunks)} chunks into: {output_path.name}")
    print("=" * 80)

    # Read first chunk to get global attributes and parameters
    first_chunk = nc.Dataset(chunks[0]["filepath"], "r")

    # Create output file
    output_nc = nc.Dataset(output_path, "w", format="NETCDF4")

    # Copy global attributes from first chunk
    print("\nCopying global attributes...")
    for attr_name in first_chunk.ncattrs():
        setattr(output_nc, attr_name, getattr(first_chunk, attr_name))

    # Create dimensions
    nspec = (
        first_chunk.dimensions["nspec"].size
        if "nspec" in first_chunk.dimensions
        else 100
    )
    output_nc.createDimension("nspec", nspec)

    first_chunk.close()

    # Merge all chunks
    print(f"\nMerging chunks...")
    total_land_cells = 0
    total_ocean_cells = 0

    for chunk in tqdm(chunks, desc="Processing chunks"):
        src_nc = nc.Dataset(chunk["filepath"], "r")

        # Iterate through all groups (cells) in this chunk
        for group_name in src_nc.groups:
            src_group = src_nc.groups[group_name]

            # Create group in output
            dst_group = output_nc.createGroup(group_name)

            # Copy variables
            for var_name in src_group.variables:
                src_var = src_group.variables[var_name]

                # Create variable in output
                if src_var.dimensions:
                    dst_var = dst_group.createVariable(
                        var_name, src_var.datatype, src_var.dimensions
                    )
                else:
                    dst_var = dst_group.createVariable(var_name, src_var.datatype)

                # Copy data
                dst_var[:] = src_var[:]

                # Copy attributes
                for attr_name in src_var.ncattrs():
                    setattr(dst_var, attr_name, getattr(src_var, attr_name))

            # Track statistics
            if "is_land" in src_group.variables:
                if src_group.variables["is_land"][:]:
                    total_land_cells += 1
                else:
                    total_ocean_cells += 1

        src_nc.close()

    output_nc.close()

    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nCells merged:")
    print(f"  Land cells: {total_land_cells}")
    print(f"  Ocean cells: {total_ocean_cells}")
    print(f"  Total: {total_land_cells + total_ocean_cells}")
    print("=" * 80)

    return total_land_cells + total_ocean_cells


def cleanup_chunks(chunks):
    """Remove intermediate chunk files."""
    print("\nCleaning up intermediate files...")
    for chunk in tqdm(chunks, desc="Removing chunks"):
        chunk["filepath"].unlink()
    print(f"✓ Removed {len(chunks)} chunk files")


def main():
    parser = argparse.ArgumentParser(description="Merge ICON ETOPO NetCDF chunk files")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate chunk files after merge",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="icon_etopo_global_FINAL.nc",
        help="Output filename (default: icon_etopo_global_FINAL.nc)",
    )
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
        # Try to find it automatically
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
            print("Error: Could not find datasets directory")
            print("Please specify with --datasets-dir")
            return 1

    print(f"Datasets directory: {datasets_dir}")

    # Find chunk files
    chunks = find_chunk_files(datasets_dir)
    if not chunks:
        print("Error: No chunk files found!")
        print(f"Looking for: icon_etopo_global_cells_*.nc in {datasets_dir}")
        return 1

    # Validate
    try:
        validate_chunks(chunks)
    except ValueError as e:
        print(f"\n❌ Validation error: {e}")
        print("\nChunk files found:")
        for chunk in chunks:
            print(f"  {chunk['filepath'].name}: cells {chunk['start']}-{chunk['end']}")
        return 1

    # Merge
    output_path = datasets_dir / args.output
    if output_path.exists():
        response = input(f"\n⚠ {output_path.name} already exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Merge cancelled")
            return 0

    try:
        total_cells = merge_chunks(chunks, output_path, datasets_dir)
    except Exception as e:
        print(f"\n❌ Merge failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Cleanup if requested
    if args.cleanup:
        response = input(f"\nRemove {len(chunks)} chunk files? [y/N] ")
        if response.lower() == "y":
            cleanup_chunks(chunks)

    print(f"\n✓ Success! Merged file: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
