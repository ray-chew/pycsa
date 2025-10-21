"""
pyCSA: Constrained Spectral Approximation Method

A Python package for spectral approximation methods applied to topographic analysis.
"""

__version__ = "0.95.1"

# Core modules - commonly used data structures and utilities
from pycsa.core import var, utils, io, physics, fourier, delaunay, reconstruction, lin_reg

# Wrappers - high-level interfaces
from pycsa.wrappers import interface, diagnostics

# Plotting - visualization tools
from pycsa.plotting import plotter, cart_plot

__all__ = [
    # Core
    "var",
    "utils",
    "io",
    "physics",
    "fourier",
    "delaunay",
    "reconstruction",
    "lin_reg",
    # Wrappers
    "interface",
    "diagnostics",
    # Plotting
    "plotter",
    "cart_plot",
]
