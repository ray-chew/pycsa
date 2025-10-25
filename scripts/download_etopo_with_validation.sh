#!/bin/bash
# Enhanced ETOPO download script with validation
# Checks remote file size and validates after download
# Usage:
#   Download mode: ./download_etopo_with_validation.sh [output_dir]
#   Verify mode:   ./download_etopo_with_validation.sh --verify [output_dir]

set -e

# Check for verify mode
VERIFY_ONLY=false
if [ "$1" = "--verify" ] || [ "$1" = "-v" ]; then
    VERIFY_ONLY=true
    OUTPUT_DIR="${2:-./data/etopo_15s}"
else
    OUTPUT_DIR="${1:-./data/etopo_15s}"
fi

DATA_TYPE="${ETOPO_DATA_TYPE:-surface}"
if [ "$DATA_TYPE" = "bed" ]; then
    BASE_URL="https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/15s/15s_bed_elev_netcdf"
    FILE_SUFFIX="bed"
else
    BASE_URL="https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/15s/15s_surface_elev_netcdf"
    FILE_SUFFIX="surface"
fi

mkdir -p "$OUTPUT_DIR"

if [ "$VERIFY_ONLY" = true ]; then
    echo "ETOPO 2022 15s Verification Mode"
else
    echo "ETOPO 2022 15s Download with Validation"
fi
echo "Data type: $DATA_TYPE"
echo "Directory: $OUTPUT_DIR"
echo "========================================"

# Function to get remote file size
get_remote_size() {
    local url="$1"
    # Use wget --spider to get headers only
    local size=$(wget --spider --server-response "$url" 2>&1 | grep -i Content-Length | tail -1 | awk '{print $2}')
    echo "$size"
}

# Function to get local file size
get_local_size() {
    local file="$1"
    if [ -f "$file" ]; then
        stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null
    else
        echo "0"
    fi
}

# Function to verify a single tile (no download)
verify_tile() {
    local lat="$1"
    local lon="$2"
    local filename="ETOPO_2022_v1_15s_${lat}${lon}_${FILE_SUFFIX}.nc"
    local filepath="${OUTPUT_DIR}/${filename}"
    local url="${BASE_URL}/${filename}"

    echo -n "Verifying ${lat}${lon}... "

    # Check if file exists locally
    local local_size=$(get_local_size "$filepath")

    if [ "$local_size" = "0" ]; then
        echo "✗ Missing"
        return 1
    fi

    # Get remote size
    local remote_size=$(get_remote_size "$url")

    if [ -z "$remote_size" ] || [ "$remote_size" = "0" ]; then
        echo "⚠️  Cannot verify (server unavailable)"
        return 2
    fi

    # Compare sizes
    if [ "$local_size" = "$remote_size" ]; then
        echo "✓ Valid ($(($remote_size / 1048576)) MB)"
        return 0
    else
        local local_mb=$(($local_size / 1048576))
        local remote_mb=$(($remote_size / 1048576))
        echo "✗ Size mismatch! Local: ${local_mb} MB, Expected: ${remote_mb} MB"
        return 1
    fi
}

# Function to download and validate a single tile
download_tile() {
    local lat="$1"
    local lon="$2"
    local filename="ETOPO_2022_v1_15s_${lat}${lon}_${FILE_SUFFIX}.nc"
    local filepath="${OUTPUT_DIR}/${filename}"
    local url="${BASE_URL}/${filename}"

    # Check if file exists and get sizes
    local local_size=$(get_local_size "$filepath")

    echo -n "Checking ${lat}${lon}... "

    # Get remote size
    local remote_size=$(get_remote_size "$url")

    if [ -z "$remote_size" ] || [ "$remote_size" = "0" ]; then
        echo "⚠️  File not available on server"
        return 1
    fi

    # Check if local file matches remote size
    if [ "$local_size" = "$remote_size" ]; then
        echo "✓ Already downloaded ($(($remote_size / 1048576)) MB)"
        return 0
    fi

    # Download the file
    echo "Downloading ($(($remote_size / 1048576)) MB)..."
    if wget -c -O "$filepath" "$url" 2>&1 | grep -v "^--" | grep -v "^Saving" | grep -v "^Length"; then
        # Verify download
        local final_size=$(get_local_size "$filepath")
        if [ "$final_size" = "$remote_size" ]; then
            echo "  ✓ Download verified"
            return 0
        else
            echo "  ✗ Size mismatch! Expected: $remote_size, Got: $final_size"
            echo "  Deleting incomplete file..."
            rm -f "$filepath"
            return 1
        fi
    else
        echo "  ✗ Download failed"
        rm -f "$filepath"
        return 1
    fi
}

# All latitude/longitude combinations
declare -a LATS=(N00 N15 N30 N45 N60 N75 N90 S15 S30 S45 S60 S75)
declare -a LONS=(W180 W165 W150 W135 W120 W105 W090 W075 W060 W045 W030 W015 E000 E015 E030 E045 E060 E075 E090 E105 E120 E135 E150 E165)

# Track statistics
total_tiles=0
valid=0
invalid=0
missing=0
failed=0

echo ""
if [ "$VERIFY_ONLY" = true ]; then
    echo "Verifying existing files..."
else
    echo "Starting download..."
fi
echo ""

# Store corrupted files for optional deletion
declare -a corrupted_files=()

for lat in "${LATS[@]}"; do
    for lon in "${LONS[@]}"; do
        total_tiles=$((total_tiles + 1))

        if [ "$VERIFY_ONLY" = true ]; then
            # Verify mode
            result=$(verify_tile "$lat" "$lon"; echo $?)
            case $result in
                0)
                    valid=$((valid + 1))
                    ;;
                1)
                    invalid=$((invalid + 1))
                    filename="ETOPO_2022_v1_15s_${lat}${lon}_${FILE_SUFFIX}.nc"
                    filepath="${OUTPUT_DIR}/${filename}"
                    if [ -f "$filepath" ]; then
                        corrupted_files+=("$filepath")
                    else
                        missing=$((missing + 1))
                    fi
                    ;;
                2)
                    failed=$((failed + 1))
                    ;;
            esac
        else
            # Download mode
            if download_tile "$lat" "$lon"; then
                valid=$((valid + 1))
            else
                failed=$((failed + 1))
            fi
        fi
    done
done

echo ""
echo "========================================"
if [ "$VERIFY_ONLY" = true ]; then
    echo "Verification Summary:"
    echo "  Total tiles checked: $total_tiles"
    echo "  Valid files: $valid"
    echo "  Invalid/corrupted: $invalid"
    echo "  Missing files: $missing"
    echo "  Could not verify: $failed"

    if [ $invalid -gt 0 ]; then
        echo ""
        echo "⚠️  Found $invalid corrupted/invalid files"
        echo ""
        echo "Corrupted files:"
        for file in "${corrupted_files[@]}"; do
            echo "  - $(basename "$file")"
        done
        echo ""
        read -p "Delete corrupted files and re-download? (yes/no): " delete_confirm
        if [ "$delete_confirm" = "yes" ]; then
            for file in "${corrupted_files[@]}"; do
                echo "Deleting: $(basename "$file")"
                rm -f "$file"
            done
            echo ""
            echo "Deleted $invalid corrupted files"
            echo "Now re-run without --verify to download missing files:"
            echo "  $0 $OUTPUT_DIR"
        fi
        exit 1
    elif [ $missing -gt 0 ]; then
        echo ""
        echo "⚠️  $missing files are missing"
        echo "Run without --verify to download them:"
        echo "  $0 $OUTPUT_DIR"
        exit 1
    else
        echo ""
        echo "✓ All files verified successfully!"
        exit 0
    fi
else
    echo "Download Summary:"
    echo "  Total tiles attempted: $total_tiles"
    echo "  Successfully validated: $valid"
    echo "  Failed/Not available: $failed"
    echo ""

    if [ $failed -gt 0 ]; then
        echo "⚠️  Some tiles failed to download."
        echo "Re-run this script to retry failed downloads."
        exit 1
    else
        echo "✓ All tiles downloaded and validated successfully!"
        exit 0
    fi
fi
