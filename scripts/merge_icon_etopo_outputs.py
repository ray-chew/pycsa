#!/usr/bin/env python3
"""
Merge ETOPO NetCDF Output Files

This script merges all chunked NetCDF outputs from the ETOPO processing into a single file,
ensuring that:
1. All cell IDs (groups) are represented in the merged file
2. Each cell has an 'is_land' attribute
3. Missing cells are filled with ocean placeholders (is_land=0)
"""

import netCDF4
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

def get_expected_cell_range(files):
    """
    Determine the expected cell range from filenames.

    Parameters
    ----------
    files : list of Path
        List of NetCDF files

    Returns
    -------
    tuple
        (min_cell, max_cell) expected in the dataset
    """
    min_cell = float('inf')
    max_cell = float('-inf')

    for f in files:
        parts = f.stem.split('_')
        range_part = parts[-1]  # e.g., '00000-00099'
        start, end = map(int, range_part.split('-'))
        min_cell = min(min_cell, start)
        max_cell = max(max_cell, end)

    return int(min_cell), int(max_cell)


def collect_all_cells(files):
    """
    Collect all cell data from chunked NetCDF files.

    Parameters
    ----------
    files : list of Path
        List of NetCDF files to merge

    Returns
    -------
    dict
        Dictionary mapping cell_id (int) to cell data dict containing:
        - is_land: int (0 or 1)
        - clat: float (radians)
        - clon: float (radians)
        - cell_area: float or None (m^2)
        - analysis: dict of arrays (only for land cells)
    """
    cell_data = {}

    print("Reading cell data from NetCDF files...")
    for nc_file in tqdm(files, desc="Processing files"):
        try:
            nc = netCDF4.Dataset(nc_file, 'r')

            # Iterate over all groups (cell IDs) in this file
            for group_name in nc.groups.keys():
                cell_id = int(group_name)
                group = nc.groups[group_name]

                # Extract cell data
                is_land = int(group.variables['is_land'][:])
                clat = float(group.variables['clat'][:])
                clon = float(group.variables['clon'][:])

                # Extract cell_area if available
                cell_area = None
                if 'cell_area' in group.variables:
                    cell_area = float(group.variables['cell_area'][:])

                cell_info = {
                    'is_land': is_land,
                    'clat': clat,
                    'clon': clon,
                    'cell_area': cell_area,
                }

                # For land cells, also extract analysis data
                if is_land == 1:
                    cell_info['analysis'] = {}
                    for var_name in group.variables.keys():
                        if var_name not in ['is_land', 'clat', 'clon', 'cell_area']:
                            cell_info['analysis'][var_name] = group.variables[var_name][:]

                cell_data[cell_id] = cell_info

            nc.close()

        except Exception as e:
            print(f"Error reading {nc_file.name}: {e}")
            continue

    return cell_data


def create_merged_netcdf(cell_data, output_path, expected_min, expected_max):
    """
    Create merged NetCDF file with all cells.

    Parameters
    ----------
    cell_data : dict
        Dictionary of cell data from collect_all_cells()
    output_path : Path
        Output file path
    expected_min : int
        Expected minimum cell ID
    expected_max : int
        Expected maximum cell ID
    """
    print(f"\nCreating merged NetCDF file: {output_path}")

    # Create new NetCDF file
    nc_out = netCDF4.Dataset(output_path, 'w', format='NETCDF4')

    # Set global attributes
    nc_out.title = "ICON ETOPO Global Topography - Merged Output"
    nc_out.description = "Merged spectral analysis of ETOPO topography on ICON grid"
    nc_out.source = "pycsa spectral approximation framework"

    # Statistics counters
    land_cells = 0
    ocean_cells = 0
    missing_cells = 0

    print(f"Writing cells {expected_min} to {expected_max}...")

    # Iterate through all expected cells
    for cell_id in tqdm(range(expected_min, expected_max + 1), desc="Writing cells"):
        # Create group for this cell
        grp = nc_out.createGroup(str(cell_id))

        if cell_id in cell_data:
            # Cell exists in data
            cell = cell_data[cell_id]
            is_land = cell['is_land']
            clat = cell['clat']
            clon = cell['clon']
            cell_area = cell.get('cell_area', None)

            if is_land:
                land_cells += 1
            else:
                ocean_cells += 1

        else:
            # Missing cell - create ocean placeholder
            print(f"Warning: Cell {cell_id} missing, creating ocean placeholder")
            is_land = 0
            clat = 0.0  # Placeholder
            clon = 0.0  # Placeholder
            cell_area = None
            missing_cells += 1
            ocean_cells += 1

        # Write basic cell attributes (always present)
        var_is_land = grp.createVariable('is_land', 'i4')
        var_is_land[:] = is_land

        var_clat = grp.createVariable('clat', 'f8')
        var_clat[:] = clat
        var_clat.units = "radians"
        var_clat.long_name = "cell center latitude"

        var_clon = grp.createVariable('clon', 'f8')
        var_clon[:] = clon
        var_clon.units = "radians"
        var_clon.long_name = "cell center longitude"

        # Write cell_area if available
        if cell_area is not None:
            var_cell_area = grp.createVariable('cell_area', 'f8')
            var_cell_area[:] = cell_area
            var_cell_area.units = "m^2"
            var_cell_area.long_name = "Area of ICON grid cell"

        # Write analysis data for land cells
        if is_land and cell_id in cell_data:
            analysis = cell_data[cell_id]['analysis']
            for var_name, var_data in analysis.items():
                # Create variable with appropriate dimensions
                if var_data.ndim == 0:
                    # Scalar variable (0-dimensional)
                    var = grp.createVariable(var_name, var_data.dtype)
                    var[:] = var_data
                elif var_data.ndim == 1:
                    dim_name = f"dim_{var_name}"
                    grp.createDimension(dim_name, var_data.shape[0])
                    var = grp.createVariable(var_name, var_data.dtype, (dim_name,))
                    var[:] = var_data
                elif var_data.ndim == 2:
                    dim0_name = f"dim0_{var_name}"
                    dim1_name = f"dim1_{var_name}"
                    grp.createDimension(dim0_name, var_data.shape[0])
                    grp.createDimension(dim1_name, var_data.shape[1])
                    var = grp.createVariable(var_name, var_data.dtype, (dim0_name, dim1_name))
                    var[:] = var_data
                else:
                    print(f"Warning: Skipping variable {var_name} with unsupported dimensions: {var_data.ndim}")
                    continue

    nc_out.close()

    # Print statistics
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"Output file: {output_path}")
    print(f"Total cells: {expected_max - expected_min + 1}")
    print(f"  Land cells (is_land=1): {land_cells}")
    print(f"  Ocean cells (is_land=0): {ocean_cells}")
    if missing_cells > 0:
        print(f"  Missing cells (filled with ocean): {missing_cells}")
    print(f"\nLand/Ocean ratio: {land_cells}/{ocean_cells} = {land_cells/ocean_cells:.3f}" if ocean_cells > 0 else "")
    print(f"Land percentage: {100*land_cells/(land_cells+ocean_cells):.2f}%")
    print("="*80)


def verify_merged_file(output_path, expected_min, expected_max):
    """
    Verify the merged NetCDF file has all cells with is_land attribute.

    Parameters
    ----------
    output_path : Path
        Path to merged NetCDF file
    expected_min : int
        Expected minimum cell ID
    expected_max : int
        Expected maximum cell ID

    Returns
    -------
    bool
        True if verification passes
    """
    print(f"\nVerifying merged file: {output_path}")

    nc = netCDF4.Dataset(output_path, 'r')

    expected_cells = set(range(expected_min, expected_max + 1))
    found_cells = set(int(g) for g in nc.groups.keys())

    # Check all cells present
    missing = expected_cells - found_cells
    if missing:
        print(f"ERROR: Missing cells: {sorted(missing)[:10]}... ({len(missing)} total)")
        nc.close()
        return False

    # Check extra cells
    extra = found_cells - expected_cells
    if extra:
        print(f"Warning: Extra cells: {sorted(extra)[:10]}... ({len(extra)} total)")

    # Check is_land attribute and count land vs ocean
    cells_without_is_land = []
    land_count = 0
    ocean_count = 0
    for group_name in nc.groups.keys():
        group = nc.groups[group_name]
        if 'is_land' not in group.variables:
            cells_without_is_land.append(group_name)
        else:
            is_land_val = int(group.variables['is_land'][:])
            if is_land_val == 1:
                land_count += 1
            else:
                ocean_count += 1

    if cells_without_is_land:
        print(f"ERROR: Cells without is_land attribute: {cells_without_is_land[:10]}... ({len(cells_without_is_land)} total)")
        nc.close()
        return False

    nc.close()

    print("✓ Verification PASSED")
    print(f"  All {len(expected_cells)} cells present")
    print(f"  All cells have 'is_land' attribute")
    print(f"  Land cells (is_land=1): {land_count}")
    print(f"  Ocean cells (is_land=0): {ocean_count}")
    print(f"  Land percentage: {100*land_count/(land_count+ocean_count):.2f}%")

    return True


if __name__ == '__main__':
    # Configuration
    input_dir = Path("datasets")
    output_dir = Path("datasets")
    output_filename = "icon_etopo_global_merged.nc"

    # Find all input files
    input_files = sorted(input_dir.glob("icon_etopo_global_cells_*.nc"))

    if not input_files:
        print(f"ERROR: No NetCDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} NetCDF files to merge")

    # Determine expected cell range
    expected_min, expected_max = get_expected_cell_range(input_files)
    print(f"Expected cell range: {expected_min} to {expected_max} ({expected_max - expected_min + 1} cells)")

    # Collect all cell data
    cell_data = collect_all_cells(input_files)
    print(f"Collected data for {len(cell_data)} cells")

    # Create merged file
    output_path = output_dir / output_filename
    create_merged_netcdf(cell_data, output_path, expected_min, expected_max)

    # Verify merged file
    if verify_merged_file(output_path, expected_min, expected_max):
        print(f"\n✓ Successfully created merged file: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
    else:
        print(f"\n✗ Verification failed for: {output_path}")
        sys.exit(1)
