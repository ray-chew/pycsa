"""
Integration test for idealised Delaunay case with Perlin noise terrain.

Tests CSAM on synthetic terrain generated using Perlin noise,
which provides more realistic multi-scale topography than pure sinusoids.
"""

import pytest
import numpy as np
from pycsa import var, utils, interface
try:
    import noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not NOISE_AVAILABLE, reason="noise package not available")
class TestIdealisedDelaunay:
    """Test CSAM on Perlin noise synthetic terrain."""

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

    def test_csam_on_perlin_terrain(self, perlin_terrain):
        """Test CSAM pipeline on Perlin noise terrain."""
        world, res_x, res_y, scale_fac = perlin_terrain

        # CSAM parameters
        U, V = 10.0, 0.0
        nhi, nhj = 24, 48

        # Initialize
        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = world

        # Create isosceles triangle
        vid = utils.isosceles(
            grid, cell,
            ymax=2.0 * np.pi * scale_fac,
            xmax=2.0 * np.pi * scale_fac,
            res=res_x
        )

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        # Create triangle mask
        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        # Run CSAM
        run = interface.get_pmf(nhi, nhj, U, V)
        ampls, uw, recon = run.sappx(cell, lmbda=1e-3, iter_solve=False)

        # Verify results
        assert ampls is not None, "Amplitudes not computed"
        assert ampls.shape == (nhj, nhi), f"Unexpected amplitude shape: {ampls.shape}"
        assert not np.all(np.isnan(ampls)), "All amplitudes are NaN"

        assert uw is not None, "PMF not computed"
        # PMF can be scalar or array depending on configuration
        if isinstance(uw, np.ndarray):
            assert uw.size > 0, "PMF array is empty"
        else:
            assert isinstance(uw, (int, float, np.number)), "PMF should be numeric"

        assert recon is not None, "Reconstruction not computed"
        assert recon.shape == cell.topo.shape, "Reconstruction shape mismatch"

    def test_csam_on_cosine_terrain(self, cosine_terrain):
        """Test CSAM on simple cosine terrain (should recover mode perfectly)."""
        bg, res_x, res_y, scale_fac = cosine_terrain

        # CSAM parameters
        U, V = 10.0, 0.0
        nhi, nhj = 12, 24

        # Initialize
        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = bg

        # Create isosceles triangle
        vid = utils.isosceles(
            grid, cell,
            ymax=2.0 * np.pi * scale_fac,
            xmax=2.0 * np.pi * scale_fac,
            res=res_x
        )

        lat_v = grid.clat_vertices[vid, :]
        lon_v = grid.clon_vertices[vid, :]

        cell.gen_mgrids()

        # Create triangle mask
        triangle = utils.gen_triangle(lon_v, lat_v)
        cell.get_masked(triangle=triangle)

        cell.wlat = np.diff(cell.lat).mean()
        cell.wlon = np.diff(cell.lon).mean()

        # Run CSAM with regularization
        run = interface.get_pmf(nhi, nhj, U, V)
        ampls, uw, recon = run.sappx(cell, lmbda=1e-4, iter_solve=False)

        # For a single cosine mode, we should have:
        # - Most energy concentrated in one or a few modes
        # - Good reconstruction quality

        ampls_clean = np.nan_to_num(ampls)

        # Check that we have non-zero amplitudes
        assert np.any(ampls_clean != 0), "No modes recovered"

        # Check that energy is concentrated (not uniform)
        max_ampl = np.abs(ampls_clean).max()
        mean_ampl = np.abs(ampls_clean).mean()
        assert max_ampl > 3 * mean_ampl, "Energy should be concentrated in few modes"

    def test_mode_selection_on_perlin_terrain(self, perlin_terrain):
        """Test mode selection (top-N modes) on Perlin terrain."""
        world, res_x, res_y, scale_fac = perlin_terrain

        # CSAM parameters
        U, V = 10.0, 0.0
        nhi, nhj = 24, 48
        n_modes = 20

        # Initialize
        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = world

        # Create isosceles triangle
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

        # First approximation (get full spectrum)
        first_appx = interface.get_pmf(nhi, nhj, U, V)
        ampls_fa, uw_fa, recon_fa = first_appx.sappx(cell, lmbda=1e-2, iter_solve=False)

        # Select top N modes
        fq_cpy = np.copy(ampls_fa)
        fq_cpy[np.isnan(fq_cpy)] = 0.0

        indices = []
        for ii in range(n_modes):
            max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            indices.append(max_idx)
            max_val = fq_cpy[max_idx]
            fq_cpy[max_idx] = 0.0

        k_idxs = [pair[1] for pair in indices]
        l_idxs = [pair[0] for pair in indices]

        # Verify mode selection
        assert len(k_idxs) == n_modes, "Incorrect number of k indices"
        assert len(l_idxs) == n_modes, "Incorrect number of l indices"

        # All indices should be within bounds
        assert all(0 <= k < nhi for k in k_idxs), "k index out of bounds"
        assert all(0 <= l < nhj for l in l_idxs), "l index out of bounds"

        # Second approximation with selected modes
        second_appx = interface.get_pmf(nhi, nhj, U, V)
        second_appx.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)

        ampls_sa, uw_sa, recon_sa = second_appx.sappx(
            cell, lmbda=1e-5, updt_analysis=True, scale=1.0, iter_solve=False
        )

        # Verify second approximation
        assert ampls_sa is not None, "Second approx failed"
        assert not np.all(np.isnan(ampls_sa)), "Second approx all NaN"

        # Second approximation should use fewer modes
        ampls_sa_clean = np.nan_to_num(ampls_sa)
        n_nonzero = np.sum(ampls_sa_clean != 0)
        assert n_nonzero <= n_modes + 5, f"Too many modes in second approx: {n_nonzero}"

    def test_deterministic_perlin_generation(self):
        """Test that Perlin noise generation is deterministic with fixed seed."""
        # Generate twice with same parameters
        def generate_perlin():
            res = 50
            scale_fac = 1000.0
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
                        base=42  # Fixed seed
                    )
            return world

        world1 = generate_perlin()
        world2 = generate_perlin()

        # Should be identical
        np.testing.assert_array_equal(
            world1, world2,
            err_msg="Perlin noise generation is not deterministic"
        )

    def test_reconstruction_quality(self, cosine_terrain):
        """Test that reconstruction quality is reasonable for known terrain."""
        bg, res_x, res_y, scale_fac = cosine_terrain

        # CSAM parameters
        U, V = 10.0, 0.0
        nhi, nhj = 24, 48

        # Initialize
        grid = var.grid()
        cell = var.topo_cell()
        cell.topo = bg

        # Create isosceles triangle
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

        # Run CSAM
        run = interface.get_pmf(nhi, nhj, U, V)
        ampls, uw, recon = run.sappx(cell, lmbda=1e-4, iter_solve=False)

        # Compute reconstruction error
        # Only compare where mask is True
        original_masked = cell.topo * cell.mask
        recon_masked = recon * cell.mask

        # Relative L2 error
        l2_error = np.linalg.norm(original_masked - recon_masked) / np.linalg.norm(original_masked)

        # For a simple cosine, reconstruction should be good
        # (not perfect due to triangular domain and regularization)
        assert l2_error < 0.5, f"Reconstruction error too high: {l2_error:.3f}"
