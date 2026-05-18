"""
Integration test for idealised isosceles triangle case.

This test runs the full CSA pipeline on synthetic terrain with an isosceles
triangular domain and compares results against baseline values from the
published JAMES paper.
Generates diagnostic plots in plots/tests/idealised_isosceles/.
"""

import numpy as np
import pytest
from pathlib import Path
from pycsa import var, utils, interface
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PLOT_DIR = Path(__file__).parent.parent.parent / "plots" / "tests" / "idealised_isosceles"


def _save_fig(name):
    """Save figure to plot directory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {path}")


class TestIdealisedIsosceles:
    """Test suite for the idealised isosceles triangle case."""

    @pytest.fixture
    def baseline_results(self):
        """Baseline numerical results from the JAMES paper."""
        return {
            'num_modes': 22,
            'amplitudes': np.array([
                1243.29667409, 1110972.57606147, 1861.67185697,
                1243.32433928, 1146.82593374, 1110972.57606147
            ]),
            'l2_errors': np.array([
                0., 164291.56804783, 115.71273229,
                85.67668202, 111.37226442, 164291.56804783
            ]),
            'percentage_errors': np.array([
                0., 89256.997, 49.737, 0.002, 7.759, 89256.997
            ])
        }

    @pytest.fixture
    def synthetic_terrain(self):
        """Generate the synthetic terrain with known spectral content."""
        np.random.seed(777)

        sz = 25
        nk = np.random.randint(0, 12, size=sz)
        nl = np.random.randint(-5, 7, size=sz)

        for ii in range(sz):
            if nk[ii] == 0 and nl[ii] < 0:
                nk[ii] += np.random.randint(1, 11)
        pts = [item for item in zip(nk, nl)]
        pts = np.array(list(set(pts)))

        nk = pts[:, 0]
        nl = pts[:, 1]
        sz = len(pts)

        Ak = np.random.random(size=sz) * 100.0
        Al = np.random.random(size=sz) * 100.0
        sck = np.random.randint(0, 2, size=sz)
        scl = np.random.randint(0, 2, size=sz)

        return {
            'nk': nk,
            'nl': nl,
            'Ak': Ak,
            'Al': Al,
            'sck': sck,
            'scl': scl,
            'sz': sz,
            'pts': pts
        }

    @pytest.fixture
    def isosceles_cell(self, synthetic_terrain):
        """Create an isosceles triangle cell with synthetic topography."""
        nhi = 12
        nhj = 12

        grid = var.grid()
        cell = var.topo_cell()
        vid = utils.isosceles(grid, cell)

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        cell.topo = np.zeros_like(cell.lat_grid)

        def sinusoidal_basis(Ak, nk, Al, nl, sc):
            nk_scaled = 2.0 * np.pi * nk / cell.lon.max()
            nl_scaled = 2.0 * np.pi * nl / cell.lat.max()

            if sc == 0:
                bf = Ak * np.cos(nk_scaled * cell.lon_grid + nl_scaled * cell.lat_grid)
            else:
                bf = Al * np.sin(nk_scaled * cell.lon_grid + nl_scaled * cell.lat_grid)

            return bf

        terrain = synthetic_terrain
        for ii in range(terrain['sz']):
            cell.topo += sinusoidal_basis(
                terrain['Ak'][ii], terrain['nk'][ii],
                terrain['Al'][ii], terrain['nl'][ii],
                terrain['sck'][ii]
            )

        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        return cell, triangle, terrain['sz']

    def test_spectral_approximation(self, isosceles_cell, synthetic_terrain, baseline_results):
        """Test that CSA pipeline runs and produces consistent results."""
        cell, triangle, sz = isosceles_cell
        terrain = synthetic_terrain

        nhi = 12
        nhj = 12
        n_modes = 14
        lmbda_reg = 8.0 * 1e-5
        lmbda_fg = 1e-1
        lmbda_sg = 1e-6

        U, V = 1.0, 1.0

        # Build reference spectrum
        freqs_ref = np.zeros((nhi, nhj))
        cnt = 0
        for pt in terrain['pts']:
            kk, ll = pt
            ll += 5
            freqs_ref[ll, kk] = terrain['Ak'][cnt]
            cnt += 1

        # Pure LSFF
        pure_lsff = interface.get_pmf(nhi, nhj, U, V)
        freqs_plsff, _, recon_plsff = pure_lsff.sappx(
            cell, lmbda=0.0, iter_solve=False, save_am=True
        )

        # Regularized LSFF
        reg_lsff = interface.get_pmf(nhi, nhj, U, V)
        freqs_rlsff, _, recon_rlsff = reg_lsff.sappx(
            cell, lmbda=lmbda_reg, iter_solve=False
        )

        # CSA: first approximation on full domain
        first_guess = interface.get_pmf(nhi, nhj, U, V)
        cell_fa = deepcopy(cell)
        cell_fa.get_masked(mask=np.ones_like(cell.topo).astype('bool'))
        cell_fa.wlat = np.diff(cell_fa.lat).mean()
        cell_fa.wlon = np.diff(cell_fa.lon).mean()

        freqs_fg, _, recon_fg = first_guess.sappx(cell_fa, lmbda=lmbda_fg, iter_solve=False)

        # Mode selection
        fq_cpy = np.copy(freqs_fg)
        fq_cpy[np.isnan(fq_cpy)] = 0.0

        indices = []
        for ii in range(n_modes):
            max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            indices.append(max_idx)
            fq_cpy[max_idx] = 0.0

        k_idxs = [pair[1] for pair in indices]
        l_idxs = [pair[0] for pair in indices]

        # Second approximation on triangle
        second_guess = interface.get_pmf(nhi, nhj, U, V)
        second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)

        cell_sa = deepcopy(cell)
        cell_sa.get_masked(triangle=triangle)
        cell_sa.wlat = np.diff(cell_sa.lat).mean()
        cell_sa.wlon = np.diff(cell_sa.lon).mean()

        freqs_csa, _, recon_csa = second_guess.sappx(
            cell_sa, lmbda=lmbda_sg, updt_analysis=True, scale=1.0, iter_solve=False
        )

        # Clean NaN
        freqs_plsff = np.nan_to_num(freqs_plsff)
        freqs_rlsff = np.nan_to_num(freqs_rlsff)
        freqs_csa = np.nan_to_num(freqs_csa)
        freqs_ref = np.nan_to_num(freqs_ref)

        # L2 errors
        err_plsff = np.linalg.norm(freqs_plsff - freqs_ref)
        err_rlsff = np.linalg.norm(freqs_rlsff - freqs_ref)
        err_csa = np.linalg.norm(freqs_csa - freqs_ref)

        assert err_plsff > 1000, "Pure LSFF should have large error (overfits)"
        assert err_rlsff > 0, "Regularized LSFF should have some error"
        assert err_csa > 0, "CSA should have some error"
        assert err_csa < err_plsff, "CSA should perform better than pure LSFF"
        assert 50 < err_csa < 250, f"CSA L2 error {err_csa:.2f} should be ~111 (baseline)"

        sum_plsff = freqs_plsff.sum()
        sum_rlsff = freqs_rlsff.sum()
        sum_csa = freqs_csa.sum()

        assert sum_plsff > 0
        assert sum_rlsff > 0
        assert sum_csa > 0

        # --- Plot: JAMES paper Figure 4 reproduction ---
        masked_topo = np.where(cell.mask, cell.topo, np.nan)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # Row 1: reconstructions
        im00 = axes[0, 0].imshow(masked_topo, cmap='terrain', origin='lower')
        plt.colorbar(im00, ax=axes[0, 0], label='m')
        axes[0, 0].set_title(f'Original ({sz} modes)')

        im01 = axes[0, 1].imshow(recon_plsff, cmap='terrain', origin='lower')
        plt.colorbar(im01, ax=axes[0, 1], label='m')
        rmse_plsff = np.sqrt(np.nanmean((masked_topo - recon_plsff)**2))
        axes[0, 1].set_title(f'Pure LSFF (RMSE={rmse_plsff:.0f} m)')

        im02 = axes[0, 2].imshow(recon_csa, cmap='terrain', origin='lower')
        plt.colorbar(im02, ax=axes[0, 2], label='m')
        rmse_csa = np.sqrt(np.nanmean((masked_topo - recon_csa)**2))
        axes[0, 2].set_title(f'CSA {n_modes} modes (RMSE={rmse_csa:.0f} m)')

        # Row 2: spectra
        vmax = max(np.abs(freqs_ref).max(), np.abs(freqs_plsff).max(),
                   np.abs(freqs_csa).max())

        im10 = axes[1, 0].imshow(np.abs(freqs_ref), cmap='hot_r', aspect='auto',
                                 origin='lower', vmin=0, vmax=vmax)
        plt.colorbar(im10, ax=axes[1, 0], label='|A| (m)')
        axes[1, 0].set_title('Reference Spectrum')

        im11 = axes[1, 1].imshow(np.abs(freqs_plsff), cmap='hot_r', aspect='auto',
                                 origin='lower', vmin=0, vmax=vmax)
        plt.colorbar(im11, ax=axes[1, 1], label='|A| (m)')
        axes[1, 1].set_title(f'Pure LSFF (L2={err_plsff:.0f})')

        im12 = axes[1, 2].imshow(np.abs(freqs_csa), cmap='hot_r', aspect='auto',
                                 origin='lower', vmin=0, vmax=vmax)
        plt.colorbar(im12, ax=axes[1, 2], label='|A| (m)')
        axes[1, 2].set_title(f'CSA (L2={err_csa:.1f})')

        for ax in axes.flat:
            ax.set_xlabel('k')
            ax.set_ylabel('l')

        fig.suptitle('Isosceles Triangle — CSA vs LSFF (cf. JAMES Fig. 4)', fontsize=14)
        plt.tight_layout()
        _save_fig('james_fig4_reproduction.png')

        # --- Plot: L2 error bar chart ---
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ['Pure LSFF', 'Reg. LSFF', f'CSA ({n_modes} modes)',
                   'Baseline (JAMES)']
        errors = [err_plsff, err_rlsff, err_csa, baseline_results['l2_errors'][4]]
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        bars = ax.bar(methods, errors, color=colors, edgecolor='k')

        for bar, err in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{err:.1f}', ha='center', fontsize=10)

        ax.set_ylabel('L2 Error')
        ax.set_title('Spectral Error: Methods Comparison')
        ax.set_yscale('log')
        _save_fig('l2_error_comparison.png')

    def test_mode_count(self, synthetic_terrain, baseline_results):
        """Test that the correct number of unique modes are generated."""
        sz = synthetic_terrain['sz']

        assert sz == baseline_results['num_modes'], \
            f"Expected {baseline_results['num_modes']} unique modes, got {sz}"

    def test_deterministic_terrain_generation(self):
        """Test that terrain generation is deterministic with fixed seed."""
        np.random.seed(777)

        sz1 = 25
        nk1 = np.random.randint(0, 12, size=sz1)
        nl1 = np.random.randint(-5, 7, size=sz1)

        np.random.seed(777)

        sz2 = 25
        nk2 = np.random.randint(0, 12, size=sz2)
        nl2 = np.random.randint(-5, 7, size=sz2)

        np.testing.assert_array_equal(nk1, nk2, err_msg="Terrain generation is not deterministic")
        np.testing.assert_array_equal(nl1, nl2, err_msg="Terrain generation is not deterministic")
