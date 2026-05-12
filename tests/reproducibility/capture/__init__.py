"""Capture scripts that produce the canonical fixtures under ``../fixtures/``.

Each script is runnable as ``python -m tests.reproducibility.capture.capture_<case>``
and writes ``output.nc``, ``manifest.yml``, and ``figure.png`` into its case
directory. Re-running a capture is a deliberate fixture-update action; see
the README.
"""
