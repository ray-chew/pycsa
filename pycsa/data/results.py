"""Spectral analysis result dataclass.

Moved from ``pycsa.core.var``. The ``get_attrs`` method copies fields
from a Fourier transformer instance + spectrum into the analysis
container; ``grid_kk_ll`` is retained as a deprecated helper for legacy
diagnostic scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class analysis:
    """Container for everything needed to compute idealised
    pseudo-momentum fluxes from a CSA fit.

    ``dk`` and ``dl`` are set at runtime by :meth:`get_attrs` (they're
    computed from the wavenumber meshgrids), so they're not declared
    as fields.
    """

    wlat: Any = None
    wlon: Any = None
    ampls: np.ndarray | None = None

    # Wavenumber meshgrids; populated by get_attrs.
    kks: np.ndarray | None = None
    lls: np.ndarray | None = None

    recon: np.ndarray | None = None

    def get_attrs(self, fobj, freqs) -> None:
        """Copy ``wlat`` / ``wlon`` from a Fourier transformer and
        compute ``kks`` / ``lls`` meshgrids in physical units.

        Sets ``self.dk`` and ``self.dl`` as runtime attributes (mean
        spacing in each wavenumber direction).
        """
        self.wlat = np.copy(fobj.wlat)
        self.wlon = np.copy(fobj.wlon)
        self.ampls = np.copy(freqs)

        self.kks = fobj.m_i / (fobj.Ni)
        self.lls = fobj.m_j / (fobj.Nj)

        wla = self.wlat
        wlo = self.wlon

        kks = self.kks * 2.0 * np.pi
        lls = self.lls * 2.0 * np.pi

        kks = kks / wlo
        lls = lls / wla

        self.dk = np.diff(self.kks).mean()
        self.dl = np.diff(self.lls).mean()

        self.kks, self.lls = np.meshgrid(kks, lls)

    def grid_kk_ll(self, fobj, dat) -> np.ndarray:
        """
        .. deprecated:: 0.90.0
        """
        m_i = fobj.m_i
        m_j = fobj.m_j

        freq_grid = np.zeros((len(m_i), len(m_j)))

        cnt = 0
        for l_idx, ll in enumerate(m_j):
            for k_idx, kk in enumerate(m_i):
                print(kk, ll, k_idx, l_idx, cnt)
                if kk == 0 and ll <= 0:
                    freq_grid[l_idx, k_idx] = 0.0
                else:
                    freq_grid[l_idx, k_idx] = dat[cnt]
                    cnt += 1

        return freq_grid
