#!/bin/bash
# Diagnostic script for NetCDF/HDF errors on HPC
# Usage: ./diagnose_netcdf_issue.sh /path/to/etopo_file.nc

NETCDF_FILE="${1}"

if [ -z "$NETCDF_FILE" ]; then
    echo "Usage: $0 /path/to/netcdf_file.nc"
    exit 1
fi

echo "========================================="
echo "NetCDF/HDF Diagnostic Tool"
echo "========================================="
echo ""

echo "File: $NETCDF_FILE"
echo ""

# 1. Check if file exists
echo "1. File existence check:"
if [ -f "$NETCDF_FILE" ]; then
    echo "   ✓ File exists"
else
    echo "   ✗ File does not exist!"
    exit 1
fi
echo ""

# 2. Check file size
echo "2. File size check:"
FILE_SIZE=$(stat -c%s "$NETCDF_FILE" 2>/dev/null || stat -f%z "$NETCDF_FILE" 2>/dev/null)
FILE_SIZE_MB=$((FILE_SIZE / 1048576))
echo "   Size: ${FILE_SIZE} bytes (${FILE_SIZE_MB} MB)"
if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "   ⚠️  WARNING: File seems too small (< 1 MB), likely corrupted"
elif [ "$FILE_SIZE" -gt 50000000 ]; then
    echo "   ⚠️  WARNING: File seems too large (> 50 MB), unusual for 15s tile"
else
    echo "   ✓ File size seems reasonable"
fi
echo ""

# 3. Check file permissions
echo "3. File permissions check:"
FILE_PERMS=$(ls -lh "$NETCDF_FILE" | awk '{print $1}')
echo "   Permissions: $FILE_PERMS"
if [ -r "$NETCDF_FILE" ]; then
    echo "   ✓ File is readable"
else
    echo "   ✗ File is NOT readable!"
fi
echo ""

# 4. Check file type
echo "4. File type check:"
FILE_TYPE=$(file "$NETCDF_FILE" 2>/dev/null || echo "file command not available")
echo "   Type: $FILE_TYPE"
if echo "$FILE_TYPE" | grep -qi "netcdf\|hdf"; then
    echo "   ✓ File appears to be NetCDF/HDF format"
else
    echo "   ⚠️  WARNING: File may not be valid NetCDF/HDF"
fi
echo ""

# 5. Check first few bytes (magic number)
echo "5. File header check (magic number):"
HEADER=$(xxd -l 16 -p "$NETCDF_FILE" 2>/dev/null | tr -d '\n')
echo "   First 16 bytes (hex): $HEADER"

# NetCDF-3: starts with "CDF" (43 44 46)
# NetCDF-4/HDF5: starts with HDF5 signature (89 48 44 46 0d 0a 1a 0a)
if [[ "$HEADER" == 434446* ]]; then
    echo "   ✓ NetCDF-3 format detected"
elif [[ "$HEADER" == 894844460d0a1a0a* ]]; then
    echo "   ✓ NetCDF-4/HDF5 format detected"
else
    echo "   ✗ INVALID: Does not match NetCDF format signature!"
    echo "   This file is corrupted or not a NetCDF file"
fi
echo ""

# 6. Check with ncdump (if available)
echo "6. ncdump validation check:"
if command -v ncdump &> /dev/null; then
    if ncdump -h "$NETCDF_FILE" > /dev/null 2>&1; then
        echo "   ✓ File can be opened with ncdump"
        echo ""
        echo "   Variables in file:"
        ncdump -h "$NETCDF_FILE" | grep -E "^\s+(float|double|int|short|byte)" | head -10
    else
        echo "   ✗ ncdump FAILED to open file"
        echo ""
        echo "   Error output:"
        ncdump -h "$NETCDF_FILE" 2>&1 | head -5
    fi
else
    echo "   ⚠️  ncdump not available (load netcdf module?)"
fi
echo ""

# 7. Try Python netCDF4 library
echo "7. Python netCDF4 library check:"
if command -v python3 &> /dev/null; then
    python3 << EOF
import sys
try:
    import netCDF4 as nc
    print("   ✓ netCDF4 module is available")
    try:
        ds = nc.Dataset("$NETCDF_FILE", "r")
        print("   ✓ File opened successfully with Python netCDF4")
        print(f"   Variables: {list(ds.variables.keys())}")
        ds.close()
    except Exception as e:
        print(f"   ✗ Python netCDF4 FAILED to open file")
        print(f"   Error: {e}")
        sys.exit(1)
except ImportError:
    print("   ⚠️  netCDF4 module not available in Python")
    sys.exit(1)
EOF
else
    echo "   ⚠️  python3 not available"
fi
echo ""

# 8. Check filesystem
echo "8. Filesystem check:"
FILESYSTEM=$(df -T "$NETCDF_FILE" 2>/dev/null | tail -1 | awk '{print $2}')
MOUNT_POINT=$(df "$NETCDF_FILE" 2>/dev/null | tail -1 | awk '{print $NF}')
echo "   Filesystem type: $FILESYSTEM"
echo "   Mount point: $MOUNT_POINT"

# Check if on /scratch (common on HPC)
if [[ "$MOUNT_POINT" == *"scratch"* ]]; then
    echo "   ⚠️  File is on /scratch - check quota and purge policies"
fi
echo ""

# 9. Check disk space
echo "9. Disk space check:"
df -h "$NETCDF_FILE" | tail -1
echo ""

# 10. Suggest fixes
echo "========================================="
echo "DIAGNOSTIC SUMMARY & SUGGESTIONS"
echo "========================================="
echo ""

if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "⚠️  LIKELY ISSUE: File is corrupted/incomplete (too small)"
    echo ""
    echo "SOLUTION:"
    echo "  1. Delete the file:"
    echo "     rm '$NETCDF_FILE'"
    echo ""
    echo "  2. Re-download:"
    filename=$(basename "$NETCDF_FILE")
    echo "     wget https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/15s/15s_surface_elev_netcdf/$filename"
    echo ""
elif ! echo "$HEADER" | grep -qE "^(434446|894844460d0a1a0a)"; then
    echo "⚠️  LIKELY ISSUE: File is corrupted (invalid magic number)"
    echo ""
    echo "SOLUTION: Re-download the file (see above)"
    echo ""
else
    echo "❓ File appears valid, but Python netCDF4 cannot open it."
    echo ""
    echo "Possible causes:"
    echo "  1. HDF5 library version mismatch"
    echo "  2. NetCDF4 compiled with different HDF5 than runtime"
    echo "  3. File locking issues (multiple processes)"
    echo "  4. Filesystem issues (NFS, /scratch)"
    echo ""
    echo "Try:"
    echo "  1. Check loaded modules:"
    echo "     module list"
    echo ""
    echo "  2. Try reloading HDF5/NetCDF modules:"
    echo "     module purge"
    echo "     module load netcdf-c hdf5"
    echo ""
    echo "  3. Check if file is locked by another process:"
    echo "     lsof '$NETCDF_FILE'"
    echo ""
    echo "  4. Copy file to local /tmp and try opening:"
    echo "     cp '$NETCDF_FILE' /tmp/"
    echo "     # Then test with /tmp version"
fi

echo ""
echo "========================================="
