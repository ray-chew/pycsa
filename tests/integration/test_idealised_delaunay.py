"""
Integration test for idealised Delaunay case with Perlin noise terrain.

Tests CSA on synthetic terrain generated using Perlin noise,
which provides more realistic multi-scale topography than pure sinusoids.
Generates diagnostic plots in plots/tests/idealised_delaunay/.
"""

import pytest
import numpy as np
from pathlib import Path
from pycsa import var, utils, interface

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False


PLOT_DIR = Path(__file__).parent.parent.parent / "plots" / "tests" / "idealised_delaunay"


def _save_fig(name):
    """Save figure to plot directory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {path}")


@pytest.mark.integration
@pytest.mark.skipif(not NOISE_AVAILABLE, reason="noise package not available")
class TestIdealisedDelaunay:
    """Test CSA on Perlin noise synthetic terrain."""

    @pytest.fixture
    def perlin_terrain(self):
        """Generate synthetic terrain using Perlin noise."""
        res_x = res_y = 120  # Smaller for faster tests
        scale_fac = 2000.0

        shape = (res_x, res_y)
        scale = 60.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0

        world = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                world[i][j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    base=42,  # Fixed seed for reproducibility
                )

        world -= world.mean()
        world /= world.max()
        world *= scale_fac

        return world, res_x, res_y, scale_fac

    @pytest.fixture
    def cosine_terrain(self):
        """Generate simple cosine background terrain."""
        res_x = res_y = 120
        scale_fac = 2000.0

        xx = np.linspace(0, 2.0 * np.pi * scale_fac, res_x)
        X, Y = np.meshgrid(xx, xx)
        kl = 1.0 / scale_fac

        bg = -(scale_fac / 2.0) * (np.cos(kl * X + kl * Y))

        return bg, res_x, res_y, scale_fac

    def test_perlin_terrain_generation(self, perlin_terrain):
        """Test that Perlin noise terrain is generated correctly."""
        world, res_x, res_y, scale_fac = perlin_terrain

        # Check shape
        assert world.shape == (res_x, res_y), "Terrain shape incorrect"

        # Check values are in expected range
        assert np.abs(world).max() <= scale_fac, "Terrain values exceed scale factor"

        # Check terrain has variation (not constant)
        assert world.std() > 0, "Terrain has no variation"

        # Check mean is close to zero (normalized)
        assert np.abs(world.mean()) < 1.0, "Terrain mean not centered at zero"

        # --- Plot: Perlin terrain ---
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(world, cmap='terrain', origin='lower',
                       extent=[0, res_x, 0, res_y])
        plt.colorbar(im, ax=ax, label='Elevation (m)')
        ax.set_title(f'Perlin Noise Terrain\n'
                     f'shape={world.shape}, range=[{world.min():.0f}, {world.max():.0f}] m, '
                     f'std={world.std():.0f} m')
        ax.set_xlabel('x index')
        ax.set_ylabel('y index')
        _save_fig('perlin_terrain.png')

    def test_csa_on_perlin_terrain(self, perlin_terrain):
        """Test CSA pipeline on Perlin noise terrain."""
        world, res_x, res_y, scale_fac = perlin_terrain

        U, V = 10.0, 0.0
        nhi, nhj = 24, 48

        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = world

        vid = utils.isosceles(
            grid, cell,
            ymax=2.0 * np.pi * scale_fac,
            xmax=2.0 * np.pi * scale_fac,
            res=res_x
        )

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        run = interface.get_pmf(nhi, nhj, U, V)
        ampls, uw, recon = run.sappx(cell, lmbda=1e-3, iter_solve=False)

        assert ampls is not None, "Amplitudes not computed"
        assert ampls.shape == (nhj, nhi), f"Unexpected amplitude shape: {ampls.shape}"
        assert not np.all(np.isnan(ampls)), "All amplitudes are NaN"
        assert uw is not None, "PMF not computed"
        if isinstance(uw, np.ndarray):
            assert uw.size > 0, "PMF array is empty"
        else:
            assert isinstance(uw, (int, float, np.number)), "PMF should be numeric"
        assert recon is not None, "Reconstruction not computed"
        assert recon.shape == cell.topo.shape, "Reconstruction shape mismatch"

        # --- Plot: 3-panel (original, recon, diff) ---
        masked_topo = np.where(cell.mask, cell.topo, np.nan)
        diff = masked_topo - recon
        rmse = np.sqrt(np.nanmean(diff**2))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(masked_topo, cmap='terrain', origin='lower')
        plt.colorbar(im0, ax=axes[0], label='m')
        axes[0].set_title('Original (masked)')

        im1 = axes[1].imshow(recon, cmap='terrain', origin='lower')
        plt.colorbar(im1, ax=axes[1], label='m')
        axes[1].set_title(f'CSA Reconstruction ({nhi}x{nhj})')

        vmax = np.nanpercentile(np.abs(diff), 98)
        im2 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower',
                             vmin=-vmax, vmax=vmax)
        plt.colorbar(im2, ax=axes[2], label='m')
        axes[2].set_title(f'Difference (RMSE={rmse:.0f} m)')

        fig.suptitle(f'CSA on Perlin Terrain — UW={np.sum(uw):.2e}', y=1.02)
        plt.tight_layout()
        _save_fig('perlin_csa_reconstruction.png')

        # --- Plot: amplitude spectrum ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ampls_clean = np.nan_to_num(ampls, nan=0.0)
        im = ax.imshow(np.abs(ampls_clean), cmap='hot_r', aspect='auto',
                       origin='lower')
        plt.colorbar(im, ax=ax, label='|Amplitude| (m)')
        ax.set_xlabel('k index')
        ax.set_ylabel('l index')
        ax.set_title(f'Perlin CSA Spectrum ({nhi}x{nhj})')
        _save_fig('perlin_csa_spectrum.png')

    def test_csa_on_cosine_terrain(self, cosine_terrain):
        """Test CSA on simple cosine terrain (should recover mode perfectly)."""
        bg, res_x, res_y, scale_fac = cosine_terrain

        U, V = 10.0, 0.0
        nhi, nhj = 12, 24

        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = bg

        vid = utils.isosceles(
            grid, cell,
            ymax=2.0 * np.pi * scale_fac,
            xmax=2.0 * np.pi * scale_fac,
            res=res_x
        )

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        run = interface.get_pmf(nhi, nhj, U, V)
        ampls, uw, recon = run.sappx(cell, lmbda=1e-4, iter_solve=False)

        ampls_clean = np.nan_to_num(ampls)
        assert np.any(ampls_clean != 0), "No modes recovered"

        max_ampl = np.abs(ampls_clean).max()
        mean_ampl = np.abs(ampls_clean).mean()
        assert max_ampl > 3 * mean_ampl, "Energy should be concentrated in few modes"

        # --- Plot: cosine terrain + reconstruction ---
        masked_topo = np.where(cell.mask, cell.topo, np.nan)
        diff = masked_topo - recon
        rmse = np.sqrt(np.nanmean(diff**2))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(masked_topo, cmap='terrain', origin='lower')
        plt.colorbar(im0, ax=axes[0], label='m')
        axes[0].set_title('Cosine Terrain (masked)')

        im1 = axes[1].imshow(recon, cmap='terrain', origin='lower')
        plt.colorbar(im1, ax=axes[1], label='m')
        axes[1].set_title(f'CSA Reconstruction ({nhi}x{nhj})')

        vmax = np.nanpercentile(np.abs(diff), 98)
        im2 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower',
                             vmin=-vmax, vmax=vmax)
        plt.colorbar(im2, ax=axes[2], label='m')
        axes[2].set_title(f'Difference (RMSE={rmse:.0f} m)')

        fig.suptitle('CSA on Single Cosine Mode', y=1.02)
        plt.tight_layout()
        _save_fig('cosine_csa_reconstruction.png')

    def test_mode_selection_on_perlin_terrain(self, perlin_terrain):
        """Test mode selection (top-N modes) on Perlin terrain."""
        world, res_x, res_y, scale_fac = perlin_terrain

        U, V = 10.0, 0.0
        nhi, nhj = 24, 48
        n_modes = 20

        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = world

        vid = utils.isosceles(
            grid, cell,
            ymax=2.0 * np.pi * scale_fac,
            xmax=2.0 * np.pi * scale_fac,
            res=res_x
        )

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        # First approximation
        first_appx = interface.get_pmf(nhi, nhj, U, V)
        ampls_fa, uw_fa, recon_fa = first_appx.sappx(cell, lmbda=1e-2, iter_solve=False)

        # Select top N modes
        fq_cpy = np.copy(ampls_fa)
        fq_cpy[np.isnan(fq_cpy)] = 0.0

        indices = []
        for ii in range(n_modes):
            max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            indices.append(max_idx)
            fq_cpy[max_idx] = 0.0

        k_idxs = [pair[1] for pair in indices]
        l_idxs = [pair[0] for pair in indices]

        assert len(k_idxs) == n_modes
        assert len(l_idxs) == n_modes
        assert all(0 <= k < nhi for k in k_idxs)
        assert all(0 <= l < nhj for l in l_idxs)

        # Second approximation with selected modes
        second_appx = interface.get_pmf(nhi, nhj, U, V)
        second_appx.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)

        ampls_sa, uw_sa, recon_sa = second_appx.sappx(
            cell, lmbda=1e-5, updt_analysis=True, scale=1.0, iter_solve=False
        )

        assert ampls_sa is not None, "Second approx failed"
        assert not np.all(np.isnan(ampls_sa)), "Second approx all NaN"

        ampls_sa_clean = np.nan_to_num(ampls_sa)
        n_nonzero = np.sum(ampls_sa_clean != 0)
        assert n_nonzero <= n_modes + 5, f"Too many modes in second approx: {n_nonzero}"

        # --- Plot: FA vs SA comparison ---
        masked_topo = np.where(cell.mask, cell.topo, np.nan)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: reconstructions
        im00 = axes[0, 0].imshow(masked_topo, cmap='terrain', origin='lower')
        plt.colorbar(im00, ax=axes[0, 0], label='m')
        axes[0, 0].set_title('Original (masked)')

        im01 = axes[0, 1].imshow(recon_fa, cmap='terrain', origin='lower')
        plt.colorbar(im01, ax=axes[0, 1], label='m')
        rmse_fa = np.sqrt(np.nanmean((masked_topo - recon_fa)**2))
        axes[0, 1].set_title(f'FA Recon (RMSE={rmse_fa:.0f} m)')

        im02 = axes[0, 2].imshow(recon_sa, cmap='terrain', origin='lower')
        plt.colorbar(im02, ax=axes[0, 2], label='m')
        rmse_sa = np.sqrt(np.nanmean((masked_topo - recon_sa)**2))
        axes[0, 2].set_title(f'SA Recon ({n_modes} modes, RMSE={rmse_sa:.0f} m)')

        # Row 2: spectra + mode selection
        ampls_fa_clean = np.nan_to_num(ampls_fa, nan=0.0)
        vmax = np.abs(ampls_fa_clean).max()

        im10 = axes[1, 0].imshow(np.abs(ampls_fa_clean), cmap='hot_r',
                                 aspect='auto', origin='lower', vmin=0, vmax=vmax)
        plt.colorbar(im10, ax=axes[1, 0], label='|A| (m)')
        axes[1, 0].set_title('FA Spectrum (full)')

        im11 = axes[1, 1].imshow(np.abs(ampls_sa_clean), cmap='hot_r',
                                 aspect='auto', origin='lower', vmin=0, vmax=vmax)
        plt.colorbar(im11, ax=axes[1, 1], label='|A| (m)')
        axes[1, 1].set_title(f'SA Spectrum ({n_modes} modes)')

        # Mode selection scatter
        axes[1, 2].scatter(k_idxs, l_idxs, c='red', s=60, zorder=5, edgecolor='k')
        for i, (k, l) in enumerate(zip(k_idxs, l_idxs)):
            axes[1, 2].annotate(f'{i+1}', (k, l), fontsize=7,
                               ha='center', va='bottom', color='blue')
        axes[1, 2].set_xlim(-0.5, nhi - 0.5)
        axes[1, 2].set_ylim(-0.5, nhj - 0.5)
        axes[1, 2].set_xlabel('k index')
        axes[1, 2].set_ylabel('l index')
        axes[1, 2].set_title(f'Top {n_modes} Selected Modes')
        axes[1, 2].grid(True, alpha=0.3)

        fig.suptitle('Perlin Terrain: FA vs SA Mode Selection', fontsize=14)
        plt.tight_layout()
        _save_fig('perlin_mode_selection.png')

    def test_deterministic_perlin_generation(self):
        """Test that Perlin noise generation is deterministic with fixed seed."""
        def generate_perlin():
            res = 50
            world = np.zeros((res, res))
            for i in range(res):
                for j in range(res):
                    world[i][j] = noise.pnoise2(
                        i / 30.0, j / 30.0,
                        octaves=4,
                        persistence=0.5,
                        lacunarity=2.0,
                        repeatx=1024,
                        repeaty=1024,
                        base=42
                    )
            return world

        world1 = generate_perlin()
        world2 = generate_perlin()

        np.testing.assert_array_equal(
            world1, world2,
            err_msg="Perlin noise generation is not deterministic"
        )

    def test_reconstruction_quality(self, cosine_terrain):
        """Test that reconstruction quality is reasonable for known terrain."""
        bg, res_x, res_y, scale_fac = cosine_terrain

        U, V = 10.0, 0.0
        nhi, nhj = 24, 48

        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = bg

        vid = utils.isosceles(
            grid, cell,
            ymax=2.0 * np.pi * scale_fac,
            xmax=2.0 * np.pi * scale_fac,
            res=res_x
        )

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        run = interface.get_pmf(nhi, nhj, U, V)
        ampls, uw, recon = run.sappx(cell, lmbda=1e-4, iter_solve=False)

        original_masked = cell.topo * cell.mask
        recon_masked = recon * cell.mask

        l2_error = np.linalg.norm(original_masked - recon_masked) / np.linalg.norm(original_masked)

        assert l2_error < 0.5, f"Reconstruction error too high: {l2_error:.3f}"

        # --- Plot: reconstruction quality ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        diff = original_masked - recon_masked
        vmax = np.abs(diff).max()

        im0 = axes[0].imshow(diff, cmap='RdBu_r', origin='lower',
                             vmin=-vmax, vmax=vmax)
        plt.colorbar(im0, ax=axes[0], label='m')
        rmse = np.sqrt(np.mean(diff[cell.mask]**2))
        axes[0].set_title(f'Residual (RMSE={rmse:.1f} m)')

        # 1D profile through center
        mid_row = cell.topo.shape[0] // 2
        axes[1].plot(original_masked[mid_row, :], 'k-', label='Original', linewidth=1.5)
        axes[1].plot(recon_masked[mid_row, :], 'r--', label='Reconstruction', linewidth=1.5)
        axes[1].set_xlabel('x index')
        axes[1].set_ylabel('Elevation (m)')
        axes[1].set_title(f'Profile at row {mid_row}')
        axes[1].legend()

        fig.suptitle(f'Cosine Reconstruction Quality (L2 rel. error = {l2_error:.3f})')
        plt.tight_layout()
        _save_fig('cosine_reconstruction_quality.png')
