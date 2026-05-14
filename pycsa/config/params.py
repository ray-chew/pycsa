"""User parameter dataclass.

Defines the required and optional fields for a CSA run. Moved from
``pycsa.core.var`` and refactored to ``@dataclass`` with explicit
defaults. The historical ``cg_spsp`` → ``rect`` coupling
(``self.rect = False if self.cg_spsp else True``) is preserved via
``__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class params:
    """Per-run parameters for the CSA pipeline.

    Fields with ``None`` defaults are compulsory (see :meth:`check_init`
    / :meth:`check_delaunay`). All other fields have sensible defaults
    matching the historical inline ``__init__``.
    """

    # Filenames + paths
    run_case: str = ""
    path_compact_grid: Any = None
    path_compact_topo: Any = None
    path_output: Any = None
    fn_output: Any = None

    # MERIT
    enable_merit: bool = True
    merit_cg: int = 10
    path_merit: Any = None

    # Domain
    lat_extent: Any = None
    lon_extent: Any = None

    run_full_land_model: bool = True

    # Delaunay (compulsory when get_delaunay_triangulation is True)
    delaunay_xnp: Any = None
    delaunay_ynp: Any = None
    rect_set: Any = None
    lxkm: Any = None
    lykm: Any = None

    # Fourier
    nhi: int = 24
    nhj: int = 48
    n_modes: int = 100

    # Artificial wind
    U: float = 10.0
    V: float = 0.0

    # Spec Appx flags
    rect: bool = True
    dfft_first_guess: bool = False
    refine: bool = False
    no_corrections: bool = True
    cg_spsp: bool = False  # coarse-grain the spectral space?

    # Solver flags
    fa_iter_solve: bool = True
    sa_iter_solve: bool = True

    # Penalty terms
    lmbda_fa: float = 1e-2  # first guess
    lmbda_sa: float = 1e-1  # second step

    # Tapering
    taper_ref: bool = False
    taper_fa: bool = False
    taper_sa: bool = False
    taper_art_it: int = 50
    padding: int = 0  # must be less than 60

    # Flags
    get_delaunay_triangulation: bool = False
    recompute_rhs: bool = False
    debug: bool = False
    debug_writer: bool = True
    verbose: bool = False
    plot: bool = False

    def __post_init__(self) -> None:
        # Preserve historical coupling: cg_spsp implies rect=False.
        # Note: this overrides any explicit ``rect=`` passed at
        # construction. Matches the old inline behavior, which set
        # ``rect`` unconditionally based on ``cg_spsp`` at __init__ time.
        self.rect = False if self.cg_spsp else True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def self_test(self) -> bool:
        """Check that compulsory parameters are set and (if requested)
        Delaunay-related params too. Generates ``fn_output`` if missing.
        """
        if self.fn_output is None:
            # Local import to avoid pycsa.core.io ↔ pycsa.config cycle.
            from pycsa.core import io as _io

            self.fn_output = _io.fn_gen(self)

        self.check_init()

        if self.get_delaunay_triangulation:
            self.check_delaunay()

        return True

    def check_init(self) -> None:
        """Raise if compulsory init params are undefined."""
        compulsory = ["lat_extent", "lon_extent"]
        offenders = self.checker(self, compulsory)
        assert len(offenders) == 0, (
            "Compulsory run parameter(s) undefined: %s" % offenders
        )

    def check_delaunay(self) -> None:
        """Raise if compulsory Delaunay params are undefined."""
        compulsory = ["delaunay_xnp", "delaunay_ynp", "rect_set", "lxkm", "lykm"]
        offenders = self.checker(self, compulsory)
        assert len(offenders) == 0, (
            "Compulsory Delaunay run parameter(s) undefined: %s" % offenders
        )

    @staticmethod
    def checker(arg, compulsory_params) -> list[str]:
        """Return names of fields in ``compulsory_params`` that are ``None``."""
        offenders = []
        for key, value in vars(arg).items():
            if key in compulsory_params:
                if value is None:
                    offenders.append(key)
        return offenders

    # ------------------------------------------------------------------
    # Legacy convenience
    # ------------------------------------------------------------------

    def print(self) -> None:
        """Print all attributes to stdout. Preserves the legacy
        ``params.print()`` interface inherited from the old ``obj`` base
        class — used by several run scripts to dump the configuration
        banner at startup.
        """
        for name, value in vars(self).items():
            print(name, value)
