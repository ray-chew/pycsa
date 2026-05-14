"""Command-line entry points for pyCSA.

Thin wrappers around the run scripts in ``runs/``. These are declared in
``pyproject.toml`` ``[project.scripts]`` so ``pip install pycsa`` puts
them on ``PATH``.

Currently only ``pycsa-idealised`` ships. Additional entry points for
the regional/global MERIT and ETOPO scripts can be added here once
those scripts gain proper ``main()`` interfaces — the current
jupytext-style scripts in ``runs/icon_*_global.py`` and
``icon_*_regional.py`` aren't suitable for unattended CLI invocation
yet.

The direct ``python -m runs.<script>`` invocation continues to work and
is the supported path for scripts not yet wrapped here.
"""

from __future__ import annotations

from typing import Sequence


def idealised(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``pycsa-idealised`` console script.

    Forwards to :func:`runs.idealised_isosceles.main`; runs the idealised
    isosceles CSA experiment with a fixed deterministic seed and prints a
    numerical summary.
    """
    from runs.idealised_isosceles import main as _main

    return _main(argv)
