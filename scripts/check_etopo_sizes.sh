#!/bin/bash
# Safer script - just checks file sizes
# ETOPO 15s surface files should be 5-35 MB typically

DATA_DIR="${1:-./data/etopo_15s}"

echo "Checking ETOPO file sizes in: $DATA_DIR"
echo "========================================="

suspicious=()
total=0

for file in "$DATA_DIR"/*.nc; do
    if [ -f "$file" ]; then
        total=$((total + 1))
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        size_mb=$((size / 1048576))
        filename=$(basename "$file")

        # ETOPO 15s tiles are typically 5-35 MB
        if [ "$size" -lt 1000000 ]; then  # Less than 1 MB is definitely wrong
            echo "⚠️  SUSPICIOUS: $filename (${size_mb} MB - too small!)"
            suspicious+=("$file")
        elif [ "$size" -gt 50000000 ]; then  # More than 50 MB is suspicious
            echo "⚠️  SUSPICIOUS: $filename (${size_mb} MB - too large!)"
            suspicious+=("$file")
        else
            echo "✓ OK: $filename (${size_mb} MB)"
        fi
    fi
done

echo ""
echo "========================================="
echo "Total files: $total"
echo "Suspicious files: ${#suspicious[@]}"

if [ ${#suspicious[@]} -gt 0 ]; then
    echo ""
    echo "Suspicious files to check/re-download:"
    for file in "${suspicious[@]}"; do
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo "  - $(basename "$file") ($(($size / 1048576)) MB)"
    done
fi
