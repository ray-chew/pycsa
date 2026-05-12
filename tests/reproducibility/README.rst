Reproducibility test suite
==========================

Golden-reference fixtures that pin the numerical output of three canonical
pyCSA pipelines. The Phase B refactor PRs (var.py split, ``obj.print()`` →
``logging``, ``BufferPool`` → ``ComputeContext``, tests-out-of-runs) gate on
this suite staying green.

Cases
-----

* **Idealised** — synthetic isosceles-triangle terrain, deterministic seed,
  no external data. Captured from
  ``runs.idealised_isosceles.run()``.

* **Regional MERIT** — small regional run on a single ICON land cell using
  bundled MERIT tile slices.

* **ETOPO single-cell** — one ICON cell loaded + analysed end-to-end using
  bundled ETOPO tile slices.

Layout
------

::

  tests/reproducibility/
  ├── conftest.py          # monkey-patches pycsa.local_paths.paths to fixtures/<case>/input/
  ├── comparator.py        # NetCDFComparator: tolerance gate + hash tripwire
  ├── manifest.py          # manifest.yml load/save + schema
  ├── test_reproducibility.py
  ├── capture/             # capture_<case>.py scripts
  └── fixtures/<case>/
      ├── input/           # bundled, production-shape input slices
      ├── output.nc        # canonical pipeline output
      ├── manifest.yml     # per-variable rtol/atol (gate) + sha256 (tripwire)
      └── figure.png       # eyeballed visualization, embedded in PR description

Running the suite
-----------------

::

  pytest tests/reproducibility/ -v

All three cases run in CI — input data is bundled.

Regenerating a fixture
----------------------

Capture is a deliberate fixture-update action. Run only when refactor PRs
intentionally change numerics:

::

  python -m tests.reproducibility.capture.capture_idealised
  python -m tests.reproducibility.capture.capture_regional_merit
  python -m tests.reproducibility.capture.capture_etopo_single_cell

Each capture writes ``output.nc``, ``manifest.yml``, and ``figure.png`` into
its case directory. Open a fixture-update PR with the new figures embedded in
the description for human sign-off.

Manifest schema
---------------

::

  fixture: idealised
  captured_at: 2026-05-12T13:00:00Z
  captured_from:
    git_sha: <commit>
    python: "3.10.x"
    numpy: "1.x"
    scipy: "1.x"
  variables:
    freqs_csa:
      dtype: float64
      shape: [12, 12]
      sha256: <hex>          # tripwire — warning on drift, not a gate
      rtol: 1.0e-5           # comparator gate
      atol: 1.0e-8
      summary: {min: ..., max: ..., mean: ..., nan_count: 0}
  notes: |
    Free-form. Document any non-default tolerance here.

Defaults match ``tests/conftest.py``'s ``assert_arrays_close`` (rtol=1e-5,
atol=1e-8). Per-variable overrides go in the manifest with a justification
in ``notes``.
