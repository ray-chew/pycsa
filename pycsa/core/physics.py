import numpy as np


class ideal_pmf(object):
    """
    Helper class to compute the idealised pseudo-momentum fluxes under one setting.
    """

    def __init__(self, **kwarg):
        """
        Sets up the default values

        Parameters
        ----------
        \*\*kwargs : any
            user-defined values to replace default background wind (``U``, ``V``), Earth's radius (``AE``), and Brunt-Väisälä frequency (``N``)

        """
        self.N = 0.02  # reference brunt-väisälä frequnecy [s^{-1}]
        self.U = -10.0  # reference horizontal wind [m s^{-1}]
        self.V = 2.0  # reference vertical wind [m s^{-1}]
        self.AE = 6371.0008 * 1e3  # Earth's radius in [m]

        # If keyword arguments are specified, we use those values...
        for key, value in kwarg.items():
            setattr(self, key, value)

    def compute_uw_pmf(self, analysis, summed=True):
        """
        Computation method

        Parameters
        ----------
        analysis : :class:`src.var.analysis`
            instance of the `analysis` class.
        summed : bool, optional
            by default True, i.e., returns a sum of the spectrum. Other, return a 2D-like array of the spectrum.

        Returns
        -------
        array-like or float
            depends on the value of ``summed``
        """
        N = self.N
        U = self.U
        V = self.V


        # if ((kks.ndim == 1) and (lls.ndim == 1)):
        #     print(True)
        #     ampls = analysis.ampls[np.nonzero(analysis.ampls)]
        # else:
        #     ampls = analysis.ampls
        ampls = np.copy(analysis.ampls)

        kks = analysis.kks
        lls = analysis.lls

        om = -kks * U - lls * V
        omsq = om**2

        # Compute mms safely: avoid divide-by-zero and sqrt of negatives.
        # We intentionally silence expected divide/invalid warnings and map singularities to 0.
        base = (kks**2 + lls**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.divide(N**2 * base, omsq, out=np.zeros_like(omsq), where=omsq > 0)
            mms = frac - base
            # Clip negatives to zero before sqrt to avoid invalid warnings
            mms = np.sqrt(np.clip(mms, 0.0, None))

        # wave-action density (Ag): safe division with zeros where om == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            Ag = -0.5 * np.divide((ampls**2) * N**2, om, out=np.zeros_like(om), where=om != 0)
        Ag = np.nan_to_num(Ag, nan=0.0, posinf=0.0, neginf=0.0)

        # group velocity in z-direction, computed safely
        denom = (base + mms**2) ** 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            cgz = self.N * np.sqrt(base) * np.divide(mms, denom, out=np.zeros_like(denom), where=denom > 0)
        cgz = np.nan_to_num(cgz, nan=0.0, posinf=0.0, neginf=0.0)

        uw_pmf = Ag * kks * cgz

        if summed:
            return uw_pmf.sum()
        else:
            return uw_pmf
