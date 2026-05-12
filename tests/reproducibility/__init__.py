"""Reproducibility test suite (Phase B #2 / GH #31).

Captures golden-reference NetCDF fixtures for three canonical pipelines and
verifies that re-running the pipelines reproduces the captured output within
documented per-variable tolerances.

See ``README.rst`` for how to regenerate fixtures.
"""
