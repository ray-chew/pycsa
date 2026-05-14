"""Centralised logging configuration for pyCSA runs.

A thin wrapper around stdlib ``logging`` that:

* attaches a timestamped ``FileHandler`` to the **root** logger (so module
  loggers under ``pycsa.*`` and run scripts all propagate up and land in
  the same log file — the previous per-script setup only captured the
  caller module's own ``__name__`` namespace and silently dropped logs
  from ``pycsa.core.tile_cache`` etc.);
* silences chatty third-party libraries (matplotlib, distributed) by
  default;
* returns the log-file path so the caller can include it in startup
  banners.

Replaces the inline ``setup_logger`` previously living in
``runs/icon_etopo_global.py`` so other run scripts (and tests that want a
file log) can call the same helper without copy-paste.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_SILENCE: tuple[str, ...] = ("matplotlib", "distributed")


def configure_logging(
    log_dir: Path | str = "logs",
    name_prefix: str = "pycsa_run",
    level: int = logging.INFO,
    silence: Sequence[str] = _DEFAULT_SILENCE,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> Path:
    """Attach a timestamped file handler to the root logger.

    Parameters
    ----------
    log_dir
        Directory for the log file (created if missing).
    name_prefix
        Stem of the log filename. The full filename is
        ``{name_prefix}_{YYYYMMDD_HHMMSS}.log``.
    level
        Log level for the root logger and the file handler.
    silence
        Logger names whose level is raised to WARNING so they don't flood
        the file (matplotlib + Dask distributed by default).
    fmt, datefmt
        Standard ``logging.Formatter`` arguments.

    Returns
    -------
    Path
        Absolute path to the newly-created log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name_prefix}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Drop any FileHandler we've added before so re-calling configure_logging
    # doesn't accumulate duplicate writers.
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            h.close()

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    for name in silence:
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file.resolve()
