"""Compute-pipeline plumbing.

Currently exposes ``ComputeContext`` — an explicit container for
per-task compute resources (buffer pool + tile-cache accessor) threaded
through the CSA pipeline.
"""

from pycsa.compute.context import ComputeContext

__all__ = ["ComputeContext"]
