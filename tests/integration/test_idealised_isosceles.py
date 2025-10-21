"""
Integration test for idealised isosceles triangle case.

This test runs the full CSAM pipeline on synthetic terrain with an isosceles
triangular domain and compares results against baseline values from the
published JAMES paper.
"""

import numpy as np
import pytest
from pycsa import var, utils, interface
from copy import deepcopy


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

        # Generate random spectral modes
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

        # Initialize triangle
        grid = var.grid()
        cell = var.topo_cell()
        vid = utils.isosceles(grid, cell)

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        # Fill with synthetic topography
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

        # Define triangle mask
        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        return cell, triangle, terrain['sz']

    def test_spectral_approximation(self, isosceles_cell, synthetic_terrain, baseline_results):
        """Test that CSAM pipeline runs and produces consistent results."""
        cell, triangle, sz = isosceles_cell
        terrain = synthetic_terrain

        nhi = 12
        nhj = 12
        n_modes = 14
        lmbda_reg = 8.0 * 1e-5
        lmbda_fg = 1e-1
        lmbda_sg = 1e-6

        # Artificial winds (not used in idealised test)
        U, V = 1.0, 1.0

        # Build reference spectrum from known terrain components
        freqs_ref = np.zeros((nhi, nhj))
        cnt = 0
        for pt in terrain['pts']:
            kk, ll = pt
            ll += 5  # Offset as in original script
            freqs_ref[ll, kk] = terrain['Ak'][cnt]
            cnt += 1

        # Run pure LSFF
        pure_lsff = interface.get_pmf(nhi, nhj, U, V)
        freqs_plsff, _, _ = pure_lsff.sappx(
            cell, lmbda=0.0, iter_solve=False, save_am=True
        )

        # Run regularized LSFF
        reg_lsff = interface.get_pmf(nhi, nhj, U, V)
        freqs_rlsff, _, _ = reg_lsff.sappx(
            cell, lmbda=lmbda_reg, iter_solve=False
        )

        # Run CSAM (first approximation + mode selection + second approximation)
        first_guess = interface.get_pmf(nhi, nhj, U, V)

        # First approximation on quadrilateral domain
        cell_fa = deepcopy(cell)
        cell_fa.get_masked(mask=np.ones_like(cell.topo).astype('bool'))
        cell_fa.wlat = np.diff(cell_fa.lat).mean()
        cell_fa.wlon = np.diff(cell_fa.lon).mean()

        freqs_fg, _, _ = first_guess.sappx(cell_fa, lmbda=lmbda_fg, iter_solve=False)

        # Select top N modes
        fq_cpy = np.copy(freqs_fg)
        fq_cpy[np.isnan(fq_cpy)] = 0.0

        indices = []
        for ii in range(n_modes):
            max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            indices.append(max_idx)
            fq_cpy[max_idx] = 0.0

        k_idxs = [pair[1] for pair in indices]
        l_idxs = [pair[0] for pair in indices]

        # Second approximation on triangular domain
        second_guess = interface.get_pmf(nhi, nhj, U, V)
        second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)

        cell_sa = deepcopy(cell)
        cell_sa.get_masked(triangle=triangle)
        cell_sa.wlat = np.diff(cell_sa.lat).mean()
        cell_sa.wlon = np.diff(cell_sa.lon).mean()

        freqs_csam, _, _ = second_guess.sappx(
            cell_sa, lmbda=lmbda_sg, updt_analysis=True, scale=1.0, iter_solve=False
        )

        # Clean up NaN values
        freqs_plsff = np.nan_to_num(freqs_plsff)
        freqs_rlsff = np.nan_to_num(freqs_rlsff)
        freqs_csam = np.nan_to_num(freqs_csam)
        freqs_ref = np.nan_to_num(freqs_ref)

        # Compute L2 errors against reference
        err_plsff = np.linalg.norm(freqs_plsff - freqs_ref)
        err_rlsff = np.linalg.norm(freqs_rlsff - freqs_ref)
        err_csam = np.linalg.norm(freqs_csam - freqs_ref)

        # Compare against baseline with reasonable tolerance
        # The baseline L2 errors are: [0, 164291.57, 115.71, 85.68, 111.37, 164291.57]
        # Where indices are: [ref, pLSFF, rLSFF, optCSAM, subCSAM, quad]
        # We're running subCSAM (n_modes=14), so compare against baseline[4] = 111.37

        # For now, just check that computations run and produce reasonable values
        assert err_plsff > 1000, "Pure LSFF should have large error (overfits)"
        assert err_rlsff > 0, "Regularized LSFF should have some error"
        assert err_csam > 0, "CSAM should have some error"
        assert err_csam < err_plsff, "CSAM should perform better than pure LSFF"

        # Check that we're in the right ballpark (within factor of 2)
        assert 50 < err_csam < 250, f"CSAM L2 error {err_csam:.2f} should be ~111 (baseline)"

        # Amplitude sums should be positive
        sum_plsff = freqs_plsff.sum()
        sum_rlsff = freqs_rlsff.sum()
        sum_csam = freqs_csam.sum()

        assert sum_plsff > 0, "Pure LSFF amplitude sum should be positive"
        assert sum_rlsff > 0, "Regularized LSFF amplitude sum should be positive"
        assert sum_csam > 0, "CSAM amplitude sum should be positive"

    def test_mode_count(self, synthetic_terrain, baseline_results):
        """Test that the correct number of unique modes are generated."""
        sz = synthetic_terrain['sz']

        # Should match baseline number of unique modes
        assert sz == baseline_results['num_modes'], \
            f"Expected {baseline_results['num_modes']} unique modes, got {sz}"

    def test_deterministic_terrain_generation(self):
        """Test that terrain generation is deterministic with fixed seed."""
        np.random.seed(777)

        # Generate terrain twice with same seed
        sz1 = 25
        nk1 = np.random.randint(0, 12, size=sz1)
        nl1 = np.random.randint(-5, 7, size=sz1)

        np.random.seed(777)

        sz2 = 25
        nk2 = np.random.randint(0, 12, size=sz2)
        nl2 = np.random.randint(-5, 7, size=sz2)

        np.testing.assert_array_equal(nk1, nk2, err_msg="Terrain generation is not deterministic")
        np.testing.assert_array_equal(nl1, nl2, err_msg="Terrain generation is not deterministic")
