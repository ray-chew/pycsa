.. :module:: changelog

.. towncrier release notes start

1.0.1 (2026-06-17)
------------------

Changed
^^^^^^^

- First release to PyPI, distributed as ``pycsa-specappx`` (the bare ``pycsa``
  name was already taken by an unrelated project). The import name is
  unchanged: ``import pycsa``. The package version is now single-sourced from
  ``pycsa.__version__``.
- The built wheel now ships only the ``pycsa`` package. The ``runs/``
  experiment scripts remain in the repository and the Zenodo archive but are no
  longer installed into ``site-packages``.

Removed
^^^^^^^

- The ``pycsa-idealised`` console script. Run the idealised benchmark directly
  with ``python -m runs.idealised_isosceles``.


1.0.0 (2026-06-09)
------------------

Added
^^^^^

- Reproducibility test infrastructure: ``tests/reproducibility/`` with a
  ``NetCDFComparator`` that gates Phase B refactor PRs against per-variable
  tolerance + SHA256 tripwire. Includes the idealised isosceles fixture
  captured from ``main``; MERIT and ETOPO single-cell fixtures land in
  follow-up commits on the same branch.

  ``runs/idealised_isosceles.py`` is refactored from a jupytext-style script
  to a clean Python module with ``run()`` and ``main()`` entry points
  (``python -m runs.idealised_isosceles``). Numerics are bit-identical to the
  pre-refactor script. (#31)
- New ``pycsa.scheduling`` module hosting ``estimate_cell_memory_gb`` and
  ``group_cells_by_memory`` — previously stuck inside
  ``runs/icon_etopo_global.py`` and consumed by ``tests/test_dynamic_memory.py``
  via a ``sys.path.insert`` hack. Tests now import from ``pycsa.scheduling``
  directly, without pulling Dask in at collection time. (#33)
- ``pyproject.toml`` ``[project]`` table now declares the metadata that
  ``pip show`` and packaging conventions expect: ``description``,
  ``readme``, ``license``, ``requires-python`` (``>=3.10``), ``authors``,
  ``maintainers``, ``keywords``, Trove ``classifiers``, and
  ``[project.urls]`` (homepage, docs, repo, issues). (#41)
- ``pyproject.toml`` declares a ``pycsa-idealised`` console script (backed
  by a new :mod:`pycsa.cli` module) that runs the idealised isosceles CSA
  experiment. Installed automatically by ``pip install pycsa``. The
  equivalent ``python -m runs.idealised_isosceles`` invocation still
  works. (#42)
- New ``examples/icon_regional_minimal.py`` — a real-data CSA pipeline
  demo on ICON cell 2311 (Aleutian arc, ~52°N) using a bundled MERIT
  slice. Runs in ~10 s on a laptop, ships with all data
  (``examples/data/`` is ~260 KB), produces a 2×3 figure showing
  original topography / first-approx reconstruction / final
  reconstruction across the top row and matching spectra + top-mode
  amplitudes below.

  Numerics are computed live (not pinned). The reproducibility suite
  (``tests/reproducibility/regional_merit``) remains the gating
  truth-source; this example is for human inspection. (#44)
- ``CITATION.cff`` (CFF 1.2.0) at the repo root. GitHub auto-renders a
  *"Cite this repository"* button on the project page from this file.

  Minimal software citation: author (Ray Chew), version (tracks
  ``pyproject.toml``), GPL-3.0-or-later SPDX identifier, repo + docs
  URLs, and the project keywords. No ``preferred-citation`` block yet —
  that can be added in a follow-up alongside any paper DOIs. (#47)
- Parameter-tuning extension: a pluggable :class:`Prior` and
  :class:`ModeSelector` (Phase 1) plus automatic per-cell hyperparameter
  selection with second-approximation (SA) stage validation (Phase 2). The
  SA stage now selects its regularisation λ per cell via ``SpatialCV``
  cross-validation instead of a fixed value, and emits baseline-vs-new
  comparison plots as a CI artifact.

  Per-cell elevation is now written to the NetCDF output alongside the
  existing ``cell_area`` metadata. (#51)
- Global ICON+ETOPO runs are now restartable: a run resumes from the last
  completed NetCDF chunk instead of recomputing from the first cell. A
  per-chunk cell-count check catches silently dropped cells on resume. (#52)
- New Andes showcase example and accompanying documentation page,
  demonstrating the CSA pipeline on a steep, anisotropic real-data region.


Changed
^^^^^^^

- ``BufferPool`` and the worker-local tile cache are now reached through an
  explicit ``pycsa.compute.ComputeContext`` dataclass threaded through the
  pipeline (``get_pmf`` / ``first_appx`` / ``second_appx`` / ``f_trans`` /
  ``lin_reg.do`` / ``lin_reg.get_coeffs``). Replaces the previous pattern
  of implicit ``BufferPool()`` construction inside ``get_pmf.__init__`` and
  module-global ``tile_cache.get_worker_cache()`` calls in ``do_cell``.

  The legacy ``buffer_pool=`` keyword on ``f_trans`` / ``lin_reg`` is
  retained as a deprecated alias for one release; passing it emits a
  ``DeprecationWarning``. (#34)
- Library-level diagnostics now go through stdlib ``logging``:

  * New ``pycsa.logging_config.configure_logging`` extracted from
    ``runs/icon_etopo_global.setup_logger``. Attaches the file handler to
    the **root** logger (the old per-script setup silently dropped logs
    from ``pycsa.*`` child loggers that didn't propagate to the script's
    own namespace).
  * ``runs/icon_etopo_global.setup_logger`` is now a thin wrapper around
    ``configure_logging``.
  * ``pycsa.core.delaunay`` and ``pycsa.core.io`` get module-level loggers
    (``logging.getLogger(__name__)``). Their unconditional ``print()``
    calls — Delaunay-triangulation banner, "Data fetched", "Error closing
    …", "Coarse-graining failed (…)" — are converted to ``logger.info`` /
    ``logger.warning`` so they respect log level and land in the same file
    as the run-script's own output.
  * Verbose-gated prints (``if self.verbose: print(...)``) are left alone
    for now — they're already correctly gated.

  The ``obj`` base class and ``obj.print()`` method are retained — deletion
  moves to the ``var.py`` split (next refactor) where the change can be
  coordinated with the dataclass migration. (#35)
- ``pycsa.core.var`` is split into thematic submodules with proper
  ``@dataclass``-based classes:

  * :class:`pycsa.data.cell.grid`, :class:`pycsa.data.cell.topo`,
    :class:`pycsa.data.cell.topo_cell` — grid + topography containers.
  * :class:`pycsa.data.results.analysis` — spectral analysis result.
  * :class:`pycsa.config.params.params` — per-run parameter container.

  ``pycsa.core.var`` is retained as a thin re-export shim so existing
  ``from pycsa.core import var`` and ``var.grid()`` / ``var.params()`` /
  etc. imports keep working through at least one release.

  The generic attribute-bag class :class:`pycsa.core.var.obj` is
  **deprecated**. It still works but emits a ``DeprecationWarning`` on
  construction. Use :class:`types.SimpleNamespace` instead.

  Behavior preserved:

  * :meth:`pycsa.data.cell.grid.apply_f` now skips a ``NON_CONVERTIBLES``
    ``ClassVar`` exclusion set (``links``, ``cell_area``) instead of
    setting a runtime instance attribute — 27 ``apply_f`` call sites
    continue to work unchanged.
  * :attr:`pycsa.config.params.params.rect` is still derived from
    ``cg_spsp`` via ``__post_init__`` (matches the old
    ``self.rect = False if self.cg_spsp else True`` inline logic).
  * :meth:`pycsa.config.params.params.print` is retained on the params
    class directly (previously inherited from ``obj``). (#36)
- CI ``reproducibility`` job now runs as a Python ``3.10`` / ``3.11`` /
  ``3.12`` matrix and produces a ``coverage.xml`` artifact (uploaded from
  the 3.12 matrix slot). Coverage is computed with ``pytest --cov=pycsa
  --cov-report=xml --cov-report=term`` (``pytest-cov`` was already pinned
  in test extras).

  README CI badge is repointed at ``ci.yml`` (the old badge pointed at the
  deleted ``documentation.yml`` workflow and had been silently broken
  since #30). (#43)
- ``docs/source/tutorial.rst`` is now an end-to-end walkthrough of
  :mod:`runs.idealised_isosceles` — generation of the synthetic terrain,
  the four reconstruction methods (pure LSFF, regularised LSFF, optimal
  CSA, sub-optimal CSA), the L2-error comparison, and a "where to go
  next" pointer to the minimal real-data example. Embeds the same
  reproducibility-fixture figure as a reference image
  (``docs/source/_static/idealised_tutorial.png``).

  Previously the page was a seven-line ``.. note:: To be completed``
  stub. (#45)
- New ``docs/source/hpc_reproducibility.rst`` consolidates the global
  ICON+ETOPO recipe — ETOPO download, ``SYSTEM_CONFIG`` presets, cell-
  range / restart story, dual-loop chunking, memory batching, tile-cache
  lifecycle, post-processing (``validate_chunks`` →
  ``merge_netcdf_chunks`` → ``verify_icon_etopo_land_ocean.py``),
  troubleshooting, citation.

  Replaces five untracked ``.md`` files in ``runs/`` and ``scripts/``
  that had drifted out of sync with the code:

  * ``runs/HPC_ETOPO_README.md``
  * ``runs/QUICK_START_128_CORES.md``
  * ``runs/NETCDF_CHUNKING_GUIDE.md``
  * ``runs/README_CHUNKING.md``
  * ``scripts/ETOPO_DOWNLOAD_GUIDE.md``

  Audited and refreshed for the post-Phase-B code state — corrected
  references to a non-existent ``HPC_PERFORMANCE`` config block (it's
  now a ``CONFIGS`` dict + ``SYSTEM_CONFIG`` selector), a non-existent
  ``scripts/download_etopo_15s.sh`` (replaced with direct ``wget``
  recipes), and the actual ``cell_start`` / ``cell_end`` restart knobs.

  Cross-linked from ``index.rst`` and the tutorial's "where to go next". (#46)
- Run scripts now use :class:`types.SimpleNamespace` in place of the
  deprecated ``pycsa.core.var.obj`` attribute-bag. (#52)
- Per-cell diagnostics plotting is extracted into a shared helper reused by
  the idealised, regional, and showcase examples.
- ``SpatialCV`` is now the default λ selector whenever per-cell coordinates
  are available, replacing GCV. GCV under-regularises spatially-correlated
  topography; the spatial cross-validation split matches the out-of-sample
  metric used to evaluate the fit.


Fixed
^^^^^

- Hygiene cleanup as the v1.0.0 release approaches (the actual ``v1.0.0``
  tag is deferred until the Levante HPC validation succeeds):

  * Removed a stale ``TODO`` in ``pycsa/core/tile_cache.py`` that asked
    for "automatic tile discovery based on bounds" — the function it
    flagged immediately delegates to ``_get_merit_tiles_for_bounds()``,
    which does exactly that. The TODO was leftover from an in-progress
    refactor and contradicted the working code below it.
  * Archived 8 stray untracked files into ``local_archive/`` (gitignored):
    ``runs/icon_merit_global_old.py``, ``runs/fn`` (orphan HDF5 binary),
    ``inputs/selected_run_poster.py``, ``scripts/backfill_cell_area.py``,
    ``docs/source/_static/logo_v2.svg``, ``docs/source/_static/logo_v2.png``,
    ``rad-deg-converter/``, ``README_SETUP.md``. Reversible via
    ``mv local_archive/X .`` if any turn out to be needed.

  ``grep -rn 'TODO\|FIXME\|XXX\|HACK' pycsa/`` now returns nothing. (#48)
- Second-approximation (SA) stage semantics corrected: the SA is now fit on
  the **triangular** cell interior with λ chosen by ``SpatialCV``, not on the
  rectangular bounding box. Fitting the bounding box silently regularised the
  fit against the ocean/padding outside the triangle. (#51)
- DKRZ Levante HPC tuning and run robustness:

  * The memory planner no longer overcommits workers; polar memory caps were
    retuned against real Levante node quotas (Dask was OOM-killing
    under-provisioned workers).
  * ``runs/icon_etopo_global`` now honours the ``cell_end`` knob (it was
    silently ignored, so range-limited and restarted runs overran).
  * ``nc_writer`` no longer truncates existing chunk files on
    re-initialisation, so a restart preserves completed chunks. (#52)


0.95.1 (2024-03-18)
-------------------

Changed
^^^^^^^

- refactored code
- regenerated all results with refactored code


0.90.1 (2024-03-15)
-------------------

Added
^^^^^

- added docstrings


0.90.0 (2024-03-11)
-------------------

Added
^^^^^

- added first tracking of code history