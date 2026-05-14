"""User-facing run-time configuration.

Currently exposes :class:`pycsa.config.params.params`, the per-run
parameter container moved out of ``pycsa.core.var``.
"""

from pycsa.config.params import params

__all__ = ["params"]
