#!/usr/bin/env python3
"""
Improved plotting script for ICON ETOPO verification data.
Loads the saved verification data and creates enhanced visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

def load_verification_data():
    """Load the verification data from npz file."""
    data_file = Path("outputs/verification/verification_data.npz")

    if not data_file.exists():
        print(f"Error: {data_file} not found.")
        print("Please run verify_icon_etopo_land_ocean.py first.")
        return None

    data = np.load(data_file)
    print(f"Loaded verification data:")
    print(f"  Total cells: {data['n_cells']}")
    print(f"  Land cells: {data['land_count']}")
    print(f"  Ocean cells: {data['ocean_count']}")
    print(f"  ETOPO coarse-graining: {data['etopo_cg']}")
    print()

    return data


def create_improved_plots(data, output_dir):
    """Create improved visualization plots."""

    clat_deg = data['clat_deg']
    clon_deg = data['clon_deg']
    land_cells = data['land_cells']
    ocean_cells = data['ocean_cells']
    land_fractions = data['land_fractions']
    land_count = data['land_count']
    ocean_count = data['ocean_count']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to Mollweide projection coordinates
    lon_plot = np.deg2rad(clon_deg)
    lon_plot[lon_plot > np.pi] -= 2*np.pi
    lat_plot = np.deg2rad(clat_deg)

    # ========================================================================
    # Figure 1: Multiple views with different thresholds
    # ========================================================================
    fig = plt.figure(figsize=(20, 12))

    # Custom colormap from blue (ocean) to green (land)
    colors_gradient = ['#0033aa', '#0066cc', '#3399ff', '#66ccff',
                       '#99ff99', '#66cc66', '#339933', '#006600']
    cmap_land_ocean = LinearSegmentedColormap.from_list('land_ocean', colors_gradient, N=256)

    # Plot 1: Continuous land fraction (original)
    ax1 = fig.add_subplot(231, projection='mollweide')
    scatter1 = ax1.scatter(lon_plot, lat_plot,
                          c=land_fractions,
                          cmap=cmap_land_ocean,
                          s=5,
                          alpha=0.9,
                          vmin=0.0,
                          vmax=1.0,
                          edgecolors='none')
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar1.set_label('Land Fraction', fontsize=10)
    ax1.set_title(f'Continuous Land Fraction\n(All gradations)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Binary classification (>50% land = green, else blue)
    ax2 = fig.add_subplot(232, projection='mollweide')
    binary_colors = np.where(land_fractions > 0.5, '#228B22', '#1E90FF')
    ax2.scatter(lon_plot, lat_plot,
               c=binary_colors,
               s=5,
               alpha=0.9,
               edgecolors='none')
    ax2.set_title(f'Binary: >50% Land = Green\nLand: {land_count}, Ocean: {ocean_count}',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Highlight mixed coastal cells (10-90% land)
    ax3 = fig.add_subplot(233, projection='mollweide')
    coastal_mask = (land_fractions > 0.1) & (land_fractions < 0.9)
    pure_land_mask = land_fractions >= 0.9
    pure_ocean_mask = land_fractions <= 0.1

    # Plot pure ocean (light blue), pure land (green), coastal (red)
    if np.any(pure_ocean_mask):
        ax3.scatter(lon_plot[pure_ocean_mask], lat_plot[pure_ocean_mask],
                   c='#B0E0E6', s=4, alpha=0.5, label='Pure Ocean (<10% land)')
    if np.any(pure_land_mask):
        ax3.scatter(lon_plot[pure_land_mask], lat_plot[pure_land_mask],
                   c='#90EE90', s=4, alpha=0.5, label='Pure Land (>90% land)')
    if np.any(coastal_mask):
        ax3.scatter(lon_plot[coastal_mask], lat_plot[coastal_mask],
                   c='#FF6347', s=8, alpha=0.9, label=f'Mixed Coastal (10-90% land)')

    ax3.set_title(f'Coastal/Mixed Cells Highlighted\n{np.sum(coastal_mask)} mixed cells',
                  fontsize=11, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=8, markerscale=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Grid structure (all cells same size/color)
    ax4 = fig.add_subplot(234, projection='mollweide')
    ax4.scatter(lon_plot, lat_plot,
               c='gray', s=2, alpha=0.6)
    ax4.set_title(f'ICON R2B4 Grid Structure\n{len(clat_deg)} cells total',
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Only cells with ANY land (>5% threshold)
    ax5 = fig.add_subplot(235, projection='mollweide')
    any_land_mask = land_fractions > 0.05
    if np.any(~any_land_mask):
        ax5.scatter(lon_plot[~any_land_mask], lat_plot[~any_land_mask],
                   c='#1E90FF', s=3, alpha=0.3, label='Pure Ocean')
    if np.any(any_land_mask):
        scatter5 = ax5.scatter(lon_plot[any_land_mask], lat_plot[any_land_mask],
                              c=land_fractions[any_land_mask],
                              cmap=cmap_land_ocean,
                              s=8,
                              alpha=0.9,
                              vmin=0.0,
                              vmax=1.0,
                              edgecolors='none',
                              label='Has Land')
    ax5.set_title(f'Cells with >5% Land Highlighted\n{np.sum(any_land_mask)} cells with land',
                  fontsize=11, fontweight='bold')
    ax5.legend(loc='lower left', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Latitude distribution
    ax6 = fig.add_subplot(236)
    lat_bins = np.linspace(-90, 90, 37)

    # Create histogram for different land fraction ranges
    pure_ocean_hist, _ = np.histogram(clat_deg[land_fractions <= 0.1], bins=lat_bins)
    coastal_hist, _ = np.histogram(clat_deg[coastal_mask], bins=lat_bins)
    pure_land_hist, _ = np.histogram(clat_deg[land_fractions >= 0.9], bins=lat_bins)

    bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    width = 5

    ax6.barh(bin_centers, pure_ocean_hist, height=width,
            color='#1E90FF', alpha=0.6, label='Pure Ocean (≤10% land)')
    ax6.barh(bin_centers, coastal_hist, height=width, left=pure_ocean_hist,
            color='#FF6347', alpha=0.6, label='Coastal (10-90% land)')
    ax6.barh(bin_centers, pure_land_hist, height=width,
            left=pure_ocean_hist+coastal_hist,
            color='#228B22', alpha=0.6, label='Pure Land (≥90% land)')

    ax6.set_xlabel('Number of cells', fontsize=10)
    ax6.set_ylabel('Latitude [degrees]', fontsize=10)
    ax6.set_title('Cell Distribution by Latitude', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "improved_verification_plots.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # ========================================================================
    # Figure 2: Pacific region zoom
    # ========================================================================
    fig2 = plt.figure(figsize=(16, 8))

    # Define Pacific region
    pacific_mask = (
        (clat_deg >= -30) & (clat_deg <= 30) &
        (((clon_deg >= 120) & (clon_deg <= 180)) |
         ((clon_deg >= -180) & (clon_deg <= -100)))
    )

    # Plot 1: Pacific overview with land fraction
    ax1 = fig2.add_subplot(121)
    scatter_pac = ax1.scatter(clon_deg[pacific_mask], clat_deg[pacific_mask],
                              c=land_fractions[pacific_mask],
                              cmap=cmap_land_ocean,
                              s=20,
                              alpha=0.9,
                              vmin=0.0,
                              vmax=1.0,
                              edgecolors='gray',
                              linewidths=0.3)
    cbar = plt.colorbar(scatter_pac, ax=ax1)
    cbar.set_label('Land Fraction', fontsize=10)
    ax1.set_xlabel('Longitude [degrees]', fontsize=10)
    ax1.set_ylabel('Latitude [degrees]', fontsize=10)
    ax1.set_title('Pacific Region: Land Fraction\n(Many islands are correctly detected)',
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([120, -100])

    # Plot 2: Pacific with only significant land (>20%)
    ax2 = fig2.add_subplot(122)
    pacific_ocean = pacific_mask & (land_fractions <= 0.2)
    pacific_land = pacific_mask & (land_fractions > 0.2)

    if np.any(pacific_ocean):
        ax2.scatter(clon_deg[pacific_ocean], clat_deg[pacific_ocean],
                   c='#1E90FF', s=10, alpha=0.4, label='Ocean (≤20% land)')
    if np.any(pacific_land):
        ax2.scatter(clon_deg[pacific_land], clat_deg[pacific_land],
                   c=land_fractions[pacific_land],
                   cmap=cmap_land_ocean,
                   s=30,
                   alpha=0.9,
                   vmin=0.2,
                   vmax=1.0,
                   edgecolors='black',
                   linewidths=0.5,
                   label='Land (>20% land)')

    ax2.set_xlabel('Longitude [degrees]', fontsize=10)
    ax2.set_ylabel('Latitude [degrees]', fontsize=10)
    ax2.set_title(f'Pacific: Cells with >20% Land\n{np.sum(pacific_land)} cells',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([120, -100])

    plt.tight_layout()

    output_file2 = output_dir / "pacific_region_detail.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file2}")
    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Pure ocean cells (≤10% land): {np.sum(land_fractions <= 0.1)}")
    print(f"Coastal/mixed cells (10-90% land): {np.sum(coastal_mask)}")
    print(f"Pure land cells (≥90% land): {np.sum(land_fractions >= 0.9)}")
    print()
    print(f"Mean land fraction: {np.mean(land_fractions):.3f}")
    print(f"Median land fraction: {np.median(land_fractions):.3f}")
    print()
    print(f"Pacific region cells: {np.sum(pacific_mask)}")
    print(f"Pacific cells with >20% land: {np.sum(pacific_land)}")
    print(f"Pacific land fraction: {np.mean(land_fractions[pacific_mask]):.3f}")
    print("="*80)


if __name__ == '__main__':
    print("="*80)
    print("IMPROVED VERIFICATION PLOTTING")
    print("="*80)
    print()

    data = load_verification_data()

    if data is not None:
        output_dir = Path("outputs") / "verification"
        create_improved_plots(data, output_dir)
        print("\n✓ Improved plots created successfully!")
        print(f"  Location: {output_dir}")
