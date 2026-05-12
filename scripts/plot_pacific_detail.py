#!/usr/bin/env python3
"""
Detailed Pacific region plot showing island cells more clearly.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Load data
data = np.load("outputs/verification/verification_data.npz")
clat_deg = data["clat_deg"]
clon_deg = data["clon_deg"]
land_fractions = data["land_fractions"]

# Create colormap
colors_gradient = [
    "#0033aa",
    "#0066cc",
    "#3399ff",
    "#66ccff",
    "#99ff99",
    "#66cc66",
    "#339933",
    "#006600",
]
cmap_land_ocean = LinearSegmentedColormap.from_list(
    "land_ocean", colors_gradient, N=256
)

# Define Pacific regions
regions = {
    "Hawaii": (15, 25, -165, -150),
    "Micronesia": (0, 15, 130, 170),
    "Polynesia": (-30, 0, -180, -130),
    "Indonesia": (-10, 10, 95, 140),
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, (lat_min, lat_max, lon_min, lon_max)) in enumerate(regions.items()):
    ax = axes[idx]

    # Find cells in region
    mask = (
        (clat_deg >= lat_min)
        & (clat_deg <= lat_max)
        & (clon_deg >= lon_min)
        & (clon_deg <= lon_max)
    )

    # Separate by land fraction
    pure_ocean = mask & (land_fractions < 0.05)
    has_land = mask & (land_fractions >= 0.05)

    # Plot
    if np.any(pure_ocean):
        ax.scatter(
            clon_deg[pure_ocean],
            clat_deg[pure_ocean],
            c="#E0F2F7",
            s=80,
            alpha=0.5,
            edgecolors="gray",
            linewidths=0.3,
            label="Ocean (<5% land)",
        )

    if np.any(has_land):
        sc = ax.scatter(
            clon_deg[has_land],
            clat_deg[has_land],
            c=land_fractions[has_land],
            cmap=cmap_land_ocean,
            s=120,
            alpha=0.95,
            vmin=0.0,
            vmax=1.0,
            edgecolors="black",
            linewidths=0.8,
        )

        # Add cell numbers for high land fraction
        high_land = has_land & (land_fractions > 0.3)
        for cell_idx in np.where(high_land)[0]:
            ax.text(
                clon_deg[cell_idx],
                clat_deg[cell_idx],
                f"{100*land_fractions[cell_idx]:.0f}%",
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    # Format
    ax.set_xlabel("Longitude [°]", fontsize=10)
    ax.set_ylabel("Latitude [°]", fontsize=10)
    ax.set_title(
        f"{name} Region\n{np.sum(has_land)} cells with ≥5% land, "
        f"{np.sum(pure_ocean)} pure ocean cells",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    if idx == 0:
        ax.legend(loc="best", fontsize=8)

plt.tight_layout()

# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Land Fraction (0=Ocean, 1=Land)", fontsize=11)

output_file = Path("outputs/verification/pacific_islands_detail.png")
plt.savefig(output_file, dpi=200, bbox_inches="tight")
print(f"Saved: {output_file}")

# Print statistics
print("\nPacific Island Statistics:")
for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
    mask = (
        (clat_deg >= lat_min)
        & (clat_deg <= lat_max)
        & (clon_deg >= lon_min)
        & (clon_deg <= lon_max)
    )
    has_land = mask & (land_fractions >= 0.05)

    if np.any(has_land):
        print(f"\n{name}:")
        print(f"  Cells with land: {np.sum(has_land)}")
        print(f"  Max land fraction: {np.max(land_fractions[has_land]):.1%}")
        print(f"  Mean land fraction: {np.mean(land_fractions[has_land]):.1%}")
